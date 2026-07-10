"""Tests de la feature `training-resume`.

Task 1 — **Persistencia del estado de resume (fundación)**: define el estado de resume
(:class:`ResumeState`), su envoltorio (:class:`TrainSnapshot`) y el I/O del *sidecar*
(:func:`save_resume_state` / :func:`load_resume_state`), separado del checkpoint de pesos.

Torch es dependencia dura del módulo (igual que en `test_training.py`), así que se hace
`importorskip` al tope. Se usan redes chicas para correr en CPU en segundos.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from diffusion.data_generation import infinite_bare, make_distribution
from diffusion.models import ScoreMLP
from diffusion.sde import make_sde
from diffusion.training import (
    ResumeState,
    TrainConfig,
    TrainResult,
    TrainSnapshot,
    load_checkpoint,
    load_resume_state,
    save_checkpoint,
    save_resume_state,
    train,
)

SIDECAR_KEYS = {"optimizer_state", "step", "torch_rng_state", "generator_state"}


def _small_net(sde) -> ScoreMLP:
    return ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)


def _data(dist, n=256, batch_size=64, *, shuffle=True):
    """Fuente infinita de tensores crudos que consume ``train`` (loader finito envuelto)."""
    return infinite_bare(dist.dataloader(n, batch_size, shuffle=shuffle))


def _net_with_optimizer_state():
    """``ScoreMLP`` + ``Adam`` que ya dio un paso (el optimizador tiene estado de momentos).

    Un round-trip trivial (optimizador recién creado) tendría ``state == {}`` y no probaría que
    los momentos de Adam sobreviven la serialización. Se fuerza un paso para poblarlo.
    """
    torch.manual_seed(0)
    net = ScoreMLP(data_dim=2, hidden_dim=16, num_blocks=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(8, 2)
    t = torch.rand(8)
    loss = net(x, t).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return net, opt


def _resume_state(opt, *, start_step=5, history=None) -> ResumeState:
    return ResumeState(
        optimizer_state=opt.state_dict(),
        start_step=start_step,
        torch_rng_state=torch.get_rng_state(),
        generator_state=torch.Generator().manual_seed(123).get_state(),
        history=list(history if history is not None else [1.0, 0.5, 0.25]),
    )


# ------------------------------------------------------------ round-trip (1.1)


def test_save_load_resume_state_roundtrip(tmp_path):
    """Ida-y-vuelta del sidecar: preserva optimizador, paso y ambos estados de azar (1.1).

    El ``optimizer_state`` vuelve a cargar en un ``Adam`` fresco (mismos params) y los momentos
    coinciden; el ``step`` coincide; los tensores de RNG son ``torch.equal`` a los originales.
    """
    net, opt = _net_with_optimizer_state()
    torch_rng = torch.get_rng_state()
    gen_state = torch.Generator().manual_seed(123).get_state()

    resume = ResumeState(
        optimizer_state=opt.state_dict(),
        start_step=5,
        torch_rng_state=torch_rng,
        generator_state=gen_state,
        history=[1.0, 0.5, 0.25],
    )

    # Path con directorios intermedios inexistentes: save debe crearlos.
    path = tmp_path / "sub" / "vp_gaussian_step00005.resume.pt"
    out = save_resume_state(path, resume)
    assert out == path
    assert path.exists()

    loaded = load_resume_state(path)

    assert loaded["step"] == 5
    assert torch.equal(loaded["torch_rng_state"], torch_rng)
    assert torch.equal(loaded["generator_state"], gen_state)

    # El optimizer_state carga de vuelta en un Adam fresco y los momentos de Adam coinciden.
    fresh = torch.optim.Adam(net.parameters(), lr=1e-3)
    fresh.load_state_dict(loaded["optimizer_state"])
    orig, new = opt.state_dict()["state"], fresh.state_dict()["state"]
    assert orig.keys() == new.keys()
    assert orig  # el optimizador original tenía estado (no vacío) => la prueba es real
    for k in orig:
        assert torch.equal(orig[k]["exp_avg"], new[k]["exp_avg"])
        assert torch.equal(orig[k]["exp_avg_sq"], new[k]["exp_avg_sq"])


# ------------------------------------------------ history no se duplica (1.3)


def test_sidecar_no_incluye_history(tmp_path):
    """El sidecar persiste solo {optimizer_state, step, RNGs} — NO el ``history`` (1.3).

    El ``history`` ya vive en el ``meta`` del checkpoint de pesos; duplicarlo sería redundante.
    """
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=2, history=[1.0, 2.0])

    path = tmp_path / "ckpt.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert "history" not in loaded
    assert set(loaded) == SIDECAR_KEYS


# ---------------------------------------------------------- fail-fast (1.4)


@pytest.mark.parametrize(
    "field", ["optimizer_state", "start_step", "torch_rng_state", "generator_state"]
)
def test_save_resume_state_falla_si_incompleto(tmp_path, field):
    """Estado incompleto (cualquier campo requerido en ``None``) => ``ValueError`` (1.4).

    No se escribe un sidecar parcial: el archivo NO debe quedar en disco tras el fallo.
    """
    _, opt = _net_with_optimizer_state()
    kwargs = dict(
        optimizer_state=opt.state_dict(),
        start_step=1,
        torch_rng_state=torch.get_rng_state(),
        generator_state=torch.Generator().manual_seed(0).get_state(),
        history=[1.0],
    )
    kwargs[field] = None
    resume = ResumeState(**kwargs)

    path = tmp_path / "parcial.resume.pt"
    with pytest.raises(ValueError, match=field):
        save_resume_state(path, resume)
    assert not path.exists()  # no se persiste un artefacto parcial


# --------------------------------------- checkpoint de pesos intacto (1.2)


def test_sidecar_no_altera_el_checkpoint_de_pesos(tmp_path):
    """Guardar el sidecar no toca el checkpoint de pesos (mismo archivo, misma meta) (1.2)."""
    sde = make_sde("vp")
    net = ScoreMLP(data_dim=2, hidden_dim=16, num_blocks=1)
    result = TrainResult(net=net, history=[1.0, 2.0], sde_name=sde.name, data_dim=sde.data_dim)
    model_spec = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 16, "num_blocks": 1}}

    weights = tmp_path / "vp_gaussian_step00005.pt"
    save_checkpoint(result, weights, model_spec=model_spec)
    weights_bytes = weights.read_bytes()  # foto byte-a-byte antes del sidecar

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    resume = _resume_state(opt, start_step=5, history=[1.0, 2.0])
    sidecar = tmp_path / "vp_gaussian_step00005.resume.pt"
    save_resume_state(sidecar, resume)

    # El checkpoint de pesos quedó intacto byte-a-byte y sigue cargando con la misma meta.
    assert weights.read_bytes() == weights_bytes
    _, meta = load_checkpoint(weights)
    assert set(meta) == {"sde_name", "data_dim", "history", "model"}
    assert meta["sde_name"] == "vp"
    assert meta["history"] == pytest.approx([1.0, 2.0])


# ----------------------------------------- envoltorio TrainSnapshot (1.1)


def test_trainsnapshot_envuelve_result_y_resume():
    """``TrainSnapshot`` agrupa el ``TrainResult`` (pesos+history) y el ``ResumeState`` (sidecar).

    Es el envoltorio con que el estado de resume viaja junto a los pesos en cada checkpoint.
    """
    sde = make_sde("vp")
    net = ScoreMLP(data_dim=2, hidden_dim=8, num_blocks=1)
    result = TrainResult(net=net, history=[1.0], sde_name=sde.name, data_dim=sde.data_dim)
    opt = torch.optim.Adam(net.parameters())
    resume = _resume_state(opt, start_step=1, history=[1.0])

    snap = TrainSnapshot(result=result, resume=resume)

    assert snap.result is result
    assert snap.resume is resume


# =============================================================================
# Task 2.1 — Reanudación del loop de entrenamiento (`train(resume=...)`)
# =============================================================================


# --------------------------------------------- corre solo los restantes (2.2, 2.3)


def test_resume_corre_solo_los_pasos_restantes():
    """Reanudar desde el paso N con ``num_steps`` total corre solo los pasos restantes (2.2)
    y continúa el ``history`` previo hasta cubrir toda la corrida (2.3).

    Se corre una corrida fresca hasta 4 pasos con snapshot en el paso 2 y se **congela** ese
    snapshot (``deepcopy``): así los pesos y el estado del optimizador quedan fijos en el paso 2
    (las corridas siguientes del loop mutarían los tensores in-place). Reanudando desde ese
    snapshot con ``num_steps=4`` el loop itera ``range(2, 4)`` → 2 pasos nuevos, y el ``history``
    final mide 4 arrancando por los 2 previos.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)

    frozen: dict[str, TrainSnapshot] = {}

    def capture(tag, snap):
        # deepcopy: congela pesos + optimizer_state en el paso del snapshot (el loop sigue
        # mutando esos tensores in-place después).
        frozen[tag] = copy.deepcopy(snap)

    net = _small_net(sde)
    train(
        sde,
        net,
        _data(dist),
        TrainConfig(num_steps=4, checkpoint_every=2, seed=0),
        on_checkpoint=capture,
    )

    snap = frozen["step00002"]
    assert snap.resume.start_step == 2
    assert len(snap.resume.history) == 2  # history del paso 2

    result = train(
        sde,
        snap.result.net,  # pesos congelados del paso 2
        _data(dist),
        TrainConfig(num_steps=4, seed=0),  # num_steps = TOTAL a alcanzar
        resume=snap.resume,
    )

    assert len(result.history) == 4  # 2 previos + 2 nuevos == num_steps total (2.3)
    assert len(result.history) - len(snap.resume.history) == 2  # corrió exactamente 2 pasos (2.2)
    assert result.history[:2] == snap.resume.history  # continuó el history previo, no lo reinició


# ----------------------------------------------------- no-op si ya completo (2.4)


@pytest.mark.parametrize("start_step", [4, 5, 10])
def test_resume_no_op_si_ya_completo(start_step):
    """Si el paso inicial ya alcanzó/superó ``num_steps``, no se corre ningún paso (2.4).

    El resultado devuelve el ``history`` previo sin cambios (ni un append más).
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    prior_history = [1.0, 0.5, 0.25, 0.2]

    resume = ResumeState(
        optimizer_state=opt.state_dict(),
        start_step=start_step,
        torch_rng_state=torch.get_rng_state(),
        generator_state=torch.Generator().manual_seed(0).get_state(),
        history=list(prior_history),
    )

    result = train(
        sde, net, _data(dist), TrainConfig(num_steps=4, seed=0), resume=resume
    )

    assert result.history == prior_history  # sin pasos nuevos: history intacto
    assert len(result.history) == len(prior_history)


# --------------------------------------- contrato del callback: TrainSnapshot (1.1)


def test_on_checkpoint_recibe_trainsnapshot_con_resume_state():
    """Con ``checkpoint_every>0`` el callback recibe un :class:`TrainSnapshot` cuyo ``.result`` es
    un :class:`TrainResult` y cuyo ``.resume`` es un :class:`ResumeState` con el optimizador
    poblado, ambos estados de azar y el paso actual (1.1).
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    calls: list[tuple[str, TrainSnapshot]] = []

    net = _small_net(sde)
    train(
        sde,
        net,
        _data(dist),
        TrainConfig(num_steps=9, checkpoint_every=3, seed=0),
        on_checkpoint=lambda tag, snap: calls.append((tag, snap)),
    )

    assert calls  # se emitieron snapshots
    for _tag, snap in calls:
        assert isinstance(snap, TrainSnapshot)
        assert isinstance(snap.result, TrainResult)
        assert isinstance(snap.resume, ResumeState)
        # optimizador poblado (Adam ya dio pasos → momentos presentes).
        assert snap.resume.optimizer_state["state"]
        # ambos estados de azar presentes como tensores.
        assert isinstance(snap.resume.torch_rng_state, torch.Tensor)
        assert isinstance(snap.resume.generator_state, torch.Tensor)

    # Los snapshots periódicos llevan como paso los pasos ya completados (= N del tag).
    periodic = {tag: snap for tag, snap in calls if tag.startswith("step")}
    assert periodic["step00003"].resume.start_step == 3
    assert periodic["step00006"].resume.start_step == 6
    assert len(periodic["step00003"].resume.history) == 3  # history hasta el paso 3


# ------------------------------------------------------------- gate: sin snapshots (1.5)


def test_sin_checkpoint_every_no_emite_snapshots():
    """Con ``checkpoint_every=0`` el callback NUNCA se invoca: no hay puntos de reanudación (1.5)."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    calls: list[str] = []

    train(
        sde,
        _small_net(sde),
        _data(dist),
        TrainConfig(num_steps=6, seed=0),  # checkpoint_every=0 por defecto
        on_checkpoint=lambda tag, snap: calls.append(tag),
    )

    assert calls == []


# ------------------------------------------------------------- regresión resume=None


def test_resume_none_entrena_normal():
    """Sin ``resume`` (default) el loop entrena desde cero como siempre: ``len(history)==num_steps``."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)

    result = train(sde, net, _data(dist), TrainConfig(num_steps=5, seed=0))

    assert len(result.history) == 5
    assert result.net is net
    assert result.sde_name == "vp"
