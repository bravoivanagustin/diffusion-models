"""Tests de la feature `training-resume`.

Task 1 — **Persistencia del estado de resume (fundación)**: define el estado de resume
(:class:`ResumeState`), su envoltorio (:class:`TrainSnapshot`) y el I/O del *sidecar*
(:func:`save_resume_state` / :func:`load_resume_state`), separado del checkpoint de pesos.

Torch es dependencia dura del módulo (igual que en `test_training.py`), así que se hace
`importorskip` al tope. Se usan redes chicas para correr en CPU en segundos.
"""

from __future__ import annotations

import copy
import pathlib

import pytest

torch = pytest.importorskip("torch")

from diffusion.data_generation import infinite_bare, make_distribution
from diffusion.models import ScoreMLP
from diffusion.sde import make_sde
from diffusion.training import (
    ResumePlan,
    ResumeState,
    TrainConfig,
    TrainResult,
    TrainSnapshot,
    discover_snapshots,
    load_checkpoint,
    load_resume,
    load_resume_state,
    resolve_resume,
    resume_sidecar_path,
    save_checkpoint,
    save_resume_state,
    train,
    validate_compatible,
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


# =============================================================================
# Task 2.2 — Equivalencia de la reanudación (gate de fidelidad, 2.6 / 2.1)
# =============================================================================
#
# Se remueve el confundidor del ORDEN DE DATOS con una fuente de **orden fijo**: un iterador
# infinito que yield-ea el MISMO batch en cada paso. Como ``train`` reconstruye ``iter(data)``
# por llamada y una corrida reanudada itera ``range(start_step, num_steps)`` sobre un iterador
# fresco, una fuente constante garantiza dato idéntico en cada paso tanto para la corrida
# ininterrumpida como para la reanudada. (Una fuente barajada/posicional divergiría — esa
# divergencia es la R2.6 relajada y aceptada; el test la controla a propósito para aislar y
# probar exactamente la restauración de optimizador + azar + paso.)

_N = 6  # paso del checkpoint intermedio
_TOTAL = 2 * _N  # total de la corrida (num_steps): el snapshot en _N cae ANTES del último paso


def _const_source(batch):
    """Fuente infinita de ORDEN FIJO: yield-ea el MISMO batch en cada paso (resume-invariante)."""
    while True:
        yield batch


def _fixed_batch(n=64, dim=2, seed=1234):
    """Batch fijo, con un ``Generator`` propio para NO tocar el RNG global de torch.

    El batch se crea antes de ``train``; usar un generador aparte evita perturbar el RNG global
    que la corrida ininterrumpida siembra (``config.seed``) y que la reanudada restaura.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=gen)


def _equiv_net() -> ScoreMLP:
    """Red chica y determinística, misma arquitectura para A / B / contraste."""
    return ScoreMLP(data_dim=2, hidden_dim=32, num_blocks=1)


def _weights_allclose(a, b) -> bool:
    """``True`` si TODO el ``state_dict`` de ``a`` y ``b`` es ``allclose`` (tolerancia tight, default)."""
    sda, sdb = a.state_dict(), b.state_dict()
    assert sda.keys() == sdb.keys()
    return all(torch.allclose(sda[k], sdb[k]) for k in sda)


def _run_uninterrupted():
    """Corrida A: entrena ininterrumpido hasta ``_TOTAL`` sobre la fuente de orden fijo.

    Devuelve ``(sde, batch, result_a, net_a, snap_N)`` donde ``snap_N`` es el
    :class:`TrainSnapshot` **congelado** (``deepcopy``) del paso ``_N``: el loop sigue mutando los
    pesos y el estado del optimizador in-place después del snapshot, así que hay que congelarlo
    para reanudar desde ese punto exacto.
    """
    sde = make_sde("vp")
    batch = _fixed_batch()
    frozen: dict[str, TrainSnapshot] = {}

    def capture(tag, snap):
        frozen[tag] = copy.deepcopy(snap)

    net_a = _equiv_net()
    result_a = train(
        sde,
        net_a,
        _const_source(batch),
        TrainConfig(num_steps=_TOTAL, checkpoint_every=_N, seed=0),
        on_checkpoint=capture,
    )
    return sde, batch, result_a, net_a, frozen[f"step{_N:05d}"]


def test_resume_equivalente_a_corrida_ininterrumpida():
    """Gate de fidelidad (2.6, 2.1): con orden fijo, reanudar equivale a no interrumpir.

    - A: corrida entera hasta ``_TOTAL`` con snapshot en el paso ``_N``.
    - B: red **fresca** con los pesos del paso ``_N`` cargados, reanudada con ``resume=snap.resume``
      hasta ``_TOTAL`` sobre la misma fuente de orden fijo.

    Removido el confundidor del orden de datos, restaurar optimizador + azar + paso hace que B
    reproduzca A: pesos ``allclose`` (tolerancia tight) e ``history`` idéntico paso a paso.
    """
    sde, batch, result_a, net_a, snap = _run_uninterrupted()

    # El snapshot es del paso _N: history de largo _N y start_step == _N; A cubrió _TOTAL.
    assert snap.resume.start_step == _N
    assert len(snap.resume.history) == _N
    assert len(result_a.history) == _TOTAL

    net_b = _equiv_net()
    net_b.load_state_dict(snap.result.net.state_dict())  # pesos congelados del paso _N
    result_b = train(
        sde,
        net_b,
        _const_source(batch),  # misma fuente de orden fijo
        TrainConfig(num_steps=_TOTAL, seed=0),  # num_steps = TOTAL a alcanzar
        resume=snap.resume,
    )

    # Equivalencia de la curva completa (2.3) y de los pesos finales (2.1, 2.6).
    assert result_b.history == result_a.history
    assert _weights_allclose(net_a, net_b)


def test_resume_sin_restaurar_optimizador_difiere():
    """Contraste (2.1): reanudar SIN restaurar el optimizador rompe la equivalencia.

    Idéntico a la reanudación fiel salvo que el ``optimizer_state`` se reemplaza por el de un
    ``Adam`` fresco (estado vacío = warm restart sin momentos). Con todo lo demás igual —pesos,
    azar, paso y datos—, los pesos finales YA NO son ``allclose`` a A: demuestra que la
    restauración del optimizador (los momentos de Adam) es lo que hace fiel a la reanudación.
    """
    sde, batch, result_a, net_a, snap = _run_uninterrupted()

    cfg = TrainConfig(num_steps=_TOTAL, seed=0)
    net_c = _equiv_net()
    net_c.load_state_dict(snap.result.net.state_dict())  # mismos pesos del paso _N que B

    # Optimizador SIN restaurar: estado vacío de un Adam fresco (mismo lr que la corrida, para que
    # la ÚNICA diferencia con la reanudación fiel sea la ausencia de momentos).
    fresh_opt = torch.optim.Adam(net_c.parameters(), lr=cfg.lr)
    no_opt_resume = ResumeState(
        optimizer_state=fresh_opt.state_dict(),  # estado vacío => warm restart, sin momentos
        start_step=snap.resume.start_step,
        torch_rng_state=snap.resume.torch_rng_state,
        generator_state=snap.resume.generator_state,
        history=list(snap.resume.history),
    )

    result_c = train(sde, net_c, _const_source(batch), cfg, resume=no_opt_resume)

    # Sin restaurar el optimizador los pesos finales difieren de A (la restauración importa)...
    assert not _weights_allclose(net_a, net_c)
    # ...y la curva diverge, aunque conserve el prefijo previo (el 1.er loss es pre-update, igual).
    assert result_c.history != result_a.history
    assert result_c.history[:_N] == result_a.history[:_N]


# =============================================================================
# Task 3.1 — Resolver de resume: descubrimiento + decisión skip/fresh/resume
# =============================================================================
#
# Lógica **pura** de rutas/decisión (sin torch, sin entrenar): se simulan los checkpoints con
# archivos ``.pt`` vacíos (``.touch()``) sobre ``tmp_path``. La convención de nombres es la del
# CLI: el checkpoint final es ``X.pt`` y los snapshots hermanos ``X_stepNNNNN.pt`` / ``X_best.pt``
# (más los sidecars ``X_stepNNNNN.resume.pt`` de la feature).

_STEM = "vp_gaussian"


def _final(tmp_path):
    """Ruta del checkpoint final (``X.pt``); el caller decide si existe (``.touch()``) o no."""
    return tmp_path / f"{_STEM}.pt"


def _touch(tmp_path, name):
    p = tmp_path / name
    p.touch()
    return p


# ----------------------------------------------- discover_snapshots (3.3)


def test_discover_snapshots_ordena_y_excluye(tmp_path):
    """Descubre los ``X_stepNNNNN.pt`` hermanos, ordenados ASC por paso (3.3), excluyendo el
    checkpoint final, el ``X_best.pt`` y los sidecars ``.resume.pt``; y sin colar snapshots de
    OTRA corrida (distinto stem) en el mismo directorio.
    """
    final = _final(tmp_path)
    final.touch()  # el final mismo NO es un snapshot
    s2 = _touch(tmp_path, f"{_STEM}_step00002.pt")
    s10 = _touch(tmp_path, f"{_STEM}_step00010.pt")
    _touch(tmp_path, f"{_STEM}_best.pt")  # best: excluido
    _touch(tmp_path, f"{_STEM}_step00010.resume.pt")  # sidecar: excluido
    _touch(tmp_path, "ve_gaussian_step00003.pt")  # otra corrida: excluida (distinto stem)

    snaps = discover_snapshots(final)

    assert [step for step, _ in snaps] == [2, 10]  # ascendente, sin best/sidecar/otra corrida
    assert snaps[0] == (2, s2)
    assert snaps[1] == (10, s10)


def test_discover_snapshots_vacio_sin_snapshots(tmp_path):
    """Sin snapshots hermanos (aunque el directorio o el final existan) → lista vacía."""
    final = _final(tmp_path)
    final.touch()
    assert discover_snapshots(final) == []


def test_discover_snapshots_directorio_inexistente(tmp_path):
    """Si el directorio del final no existe, no hay dónde buscar → lista vacía (sin excepción)."""
    final = tmp_path / "no_existe" / f"{_STEM}.pt"
    assert discover_snapshots(final) == []


# --------------------------------------------------------- skip (3.1)


def test_resolve_skip_si_final_existe(tmp_path):
    """Final presente y sin ``force`` → acción ``skip`` (corrida ya completa) (3.1)."""
    final = _final(tmp_path)
    final.touch()

    plan = resolve_resume(final)

    assert isinstance(plan, ResumePlan)
    assert plan.action == "skip"
    assert plan.weights_path is None
    assert plan.step is None


# --------------------------------------------------------- force (3.2)


def test_resolve_force_reanuda_desde_el_mas_nuevo(tmp_path):
    """Con ``force`` el chequeo del final se saltea: existiendo snapshots, reanuda desde el más
    nuevo en lugar de saltear (3.2)."""
    final = _final(tmp_path)
    final.touch()
    _touch(tmp_path, f"{_STEM}_step00005.pt")
    newest = _touch(tmp_path, f"{_STEM}_step00020.pt")

    plan = resolve_resume(final, force=True)

    assert plan.action == "resume"
    assert plan.step == 20
    assert plan.weights_path == newest


def test_resolve_force_sin_snapshots_es_fresh(tmp_path):
    """Con ``force`` pero sin snapshots (aunque el final exista) → ``fresh`` (reentrena de cero)."""
    final = _final(tmp_path)
    final.touch()

    plan = resolve_resume(final, force=True)

    assert plan.action == "fresh"


# ------------------------------------------------ auto-resume más nuevo (3.3)


def test_resolve_auto_resume_desde_el_mas_nuevo(tmp_path):
    """Final ausente + snapshots presentes → ``resume`` desde el de mayor paso (3.3)."""
    final = _final(tmp_path)  # NO existe
    _touch(tmp_path, f"{_STEM}_step00003.pt")
    _touch(tmp_path, f"{_STEM}_step00007.pt")
    newest = _touch(tmp_path, f"{_STEM}_step00030.pt")

    plan = resolve_resume(final)

    assert plan.action == "resume"
    assert plan.step == 30
    assert plan.weights_path == newest


# --------------------------------------------------------- fresh (3.4)


def test_resolve_fresh_si_no_hay_nada(tmp_path):
    """Final ausente y sin snapshots → ``fresh`` (desde cero) (3.4)."""
    plan = resolve_resume(_final(tmp_path))
    assert plan.action == "fresh"


def test_resolve_fresh_si_final_none():
    """``final_checkpoint=None`` → no hay dónde saltear ni buscar → ``fresh`` (3.4)."""
    plan = resolve_resume(None)
    assert plan.action == "fresh"


# ------------------------------------------------ --resume-from (3.5)


def test_resolve_resume_from_por_paso(tmp_path):
    """``--resume-from`` por número de paso elige ese snapshot puntual, aun con el final presente
    (el pedido explícito manda sobre el skip automático) (3.5)."""
    final = _final(tmp_path)
    final.touch()
    chosen = _touch(tmp_path, f"{_STEM}_step00005.pt")
    _touch(tmp_path, f"{_STEM}_step00010.pt")

    plan = resolve_resume(final, resume_from="5")

    assert plan.action == "resume"
    assert plan.step == 5
    assert plan.weights_path == chosen


def test_resolve_resume_from_por_ruta(tmp_path):
    """``--resume-from`` por ruta usa ese checkpoint y parsea su paso del nombre (3.5)."""
    final = _final(tmp_path)
    chosen = _touch(tmp_path, f"{_STEM}_step00010.pt")

    plan = resolve_resume(final, resume_from=str(chosen))

    assert plan.action == "resume"
    assert plan.weights_path == chosen
    assert plan.step == 10


# ------------------------------------- --resume-from inexistente (3.7)


def test_resolve_resume_from_paso_inexistente_lista_disponibles(tmp_path):
    """``--resume-from`` con un paso inexistente → ``ValueError`` que lista los pasos disponibles
    (3.7)."""
    final = _final(tmp_path)
    _touch(tmp_path, f"{_STEM}_step00002.pt")
    _touch(tmp_path, f"{_STEM}_step00010.pt")

    with pytest.raises(ValueError) as exc:
        resolve_resume(final, resume_from="99")

    msg = str(exc.value)
    assert "2" in msg and "10" in msg  # lista los pasos disponibles


def test_resolve_resume_from_ruta_inexistente_lista_disponibles(tmp_path):
    """``--resume-from`` con una ruta inexistente → ``ValueError`` accionable (lista disponibles).
    """
    final = _final(tmp_path)
    _touch(tmp_path, f"{_STEM}_step00002.pt")

    with pytest.raises(ValueError, match="2"):
        resolve_resume(final, resume_from=str(tmp_path / f"{_STEM}_step99999.pt"))


# ------------------------------------- convención del sidecar


def test_resume_sidecar_path_convencion(tmp_path):
    """``X_stepNNNNN.pt`` → ``X_stepNNNNN.resume.pt`` (sidecar hermano) (Data Models)."""
    weights = tmp_path / f"{_STEM}_step00300.pt"
    assert resume_sidecar_path(weights) == tmp_path / f"{_STEM}_step00300.resume.pt"


# =============================================================================
# Task 3.2 — Carga y validación del punto de reanudación (load_resume /
#            validate_compatible)
# =============================================================================
#
# Carga los pesos + el sidecar del checkpoint elegido y arma el ResumeState listo para reanudar,
# tomando el ``history`` del ``meta`` del checkpoint de PESOS (no del sidecar, que no lo persiste)
# (1.3). Exige el sidecar (falta → error claro, 3.6) y valida compatibilidad EXACTA —SDE,
# ``data_dim`` y receta de red— contra el ``meta`` del checkpoint (2.5).

# Receta de red consistente con ``_net_with_optimizer_state`` (ScoreMLP 2D chico).
_MODEL_SPEC = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 16, "num_blocks": 1}}


def _build_checkpoint_and_sidecar(
    tmp_path,
    *,
    sde_name="vp",
    data_dim=2,
    model_spec=None,
    history=None,
    start_step=5,
    with_sidecar=True,
):
    """Arma un checkpoint de pesos real + (opcional) su sidecar hermano con los helpers commiteados.

    Devuelve ``(weights, sidecar, saved)`` donde ``saved`` es el :class:`ResumeState` que se
    persistió (o ``None`` si ``with_sidecar=False``), para comparar contra lo que ``load_resume``
    reconstruye.
    """
    if history is None:
        history = [1.0, 0.5, 0.25, 0.2, 0.1]
    net, opt = _net_with_optimizer_state()
    result = TrainResult(
        net=net, history=list(history), sde_name=sde_name, data_dim=data_dim
    )
    weights = tmp_path / f"{sde_name}_gaussian_step{start_step:05d}.pt"
    save_checkpoint(result, weights, model_spec=model_spec)
    sidecar = resume_sidecar_path(weights)
    saved = None
    if with_sidecar:
        saved = _resume_state(opt, start_step=start_step, history=history)
        save_resume_state(sidecar, saved)
    return weights, sidecar, saved


def _meta(*, sde_name="vp", data_dim=2, model=_MODEL_SPEC):
    """``meta`` sintético al estilo :func:`save_checkpoint` (la clave ``model`` es opcional)."""
    meta = {"sde_name": sde_name, "data_dim": data_dim, "history": [1.0]}
    if model is not None:
        meta["model"] = model
    return meta


# ------------------------------------------------- load_resume happy path (1.3)


def test_load_resume_happy_path(tmp_path):
    """Cargar un punto válido entrega ``(state_dict, meta, ResumeState)`` listo para reanudar.

    El ``history`` del :class:`ResumeState` viene del ``meta`` del checkpoint de PESOS (1.3) —el
    sidecar no lo persiste—; el paso, el optimizador y el azar vienen del sidecar.
    """
    history = [1.0, 0.5, 0.25, 0.2, 0.1]
    weights, sidecar, saved = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, history=history, start_step=5
    )
    expected = {"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}

    state_dict, meta, resume = load_resume(weights, expected=expected)

    assert isinstance(resume, ResumeState)
    # (1.3) history del meta de los PESOS, no del sidecar.
    assert resume.history == history
    assert resume.history == meta["history"]
    # paso y azar reconstruidos desde el sidecar.
    assert resume.start_step == 5
    assert torch.equal(resume.torch_rng_state, saved.torch_rng_state)
    assert torch.equal(resume.generator_state, saved.generator_state)
    # el state_dict es el de la red guardada (claves de un ScoreMLP; no vacío).
    ref_net = ScoreMLP(data_dim=2, hidden_dim=16, num_blocks=1)
    assert state_dict.keys() == ref_net.state_dict().keys()
    # el meta devuelto es el del checkpoint de pesos.
    assert meta["sde_name"] == "vp"
    assert meta["data_dim"] == 2
    # el optimizer_state (del sidecar) carga en un Adam fresco y trae momentos poblados.
    fresh = torch.optim.Adam(ref_net.parameters(), lr=1e-3)
    fresh.load_state_dict(resume.optimizer_state)
    assert fresh.state_dict()["state"]  # Adam ya tenía estado => round-trip real


def test_load_resume_history_del_meta_no_del_sidecar(tmp_path):
    """El ``history`` reconstruido es exactamente el del ``meta`` del checkpoint de pesos (1.3).

    Se guarda un ``history`` en el checkpoint de pesos y NADA de history en el sidecar (por diseño);
    ``load_resume`` debe rellenar el ``ResumeState`` desde el ``meta``.
    """
    history = [3.0, 2.0, 1.0]
    weights, sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, history=history, start_step=3
    )
    # confirmamos que el sidecar NO persistió history (contrato del sidecar).
    assert "history" not in load_resume_state(sidecar)

    _, meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )
    assert resume.history == history == meta["history"]


# ------------------------------------------------ sidecar faltante (3.6)


def test_load_resume_sidecar_faltante_error_claro(tmp_path):
    """Pesos presentes pero SIN sidecar → error claro que NOMBRA el artefacto faltante (3.6)."""
    weights, sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, with_sidecar=False
    )
    assert not sidecar.exists()
    expected = {"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}

    with pytest.raises((FileNotFoundError, ValueError)) as exc:
        load_resume(weights, expected=expected)

    assert sidecar.name in str(exc.value)  # el mensaje identifica el sidecar faltante


# ------------------------------------------------ validate_compatible (2.5)


def test_validate_compatible_match_no_levanta():
    """``meta`` idéntico a la corrida (SDE + data_dim + receta) → no levanta, devuelve ``None``."""
    assert (
        validate_compatible(_meta(), sde_name="vp", model_spec=_MODEL_SPEC, data_dim=2)
        is None
    )


def test_validate_compatible_sde_difiere():
    """SDE distinta entre ``meta`` y la corrida → ``ValueError`` (2.5)."""
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(sde_name="vp"), sde_name="ve", model_spec=_MODEL_SPEC, data_dim=2
        )


def test_validate_compatible_data_dim_int_difiere():
    """``data_dim`` int distinto (2 vs 4) → ``ValueError`` (2.5)."""
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(data_dim=2), sde_name="vp", model_spec=_MODEL_SPEC, data_dim=4
        )


def test_validate_compatible_data_dim_int_vs_tupla_difiere():
    """``data_dim`` int vs tupla (2 vs (1, 28, 28)) → ``ValueError`` (2.5).

    La comparación es por igualdad, así que un entero y una forma de evento nunca matchean.
    """
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(data_dim=2), sde_name="vp", model_spec=_MODEL_SPEC, data_dim=(1, 28, 28)
        )


def test_validate_compatible_data_dim_tupla_match():
    """``data_dim`` tupla igual (forma de evento de imágenes) → no levanta."""
    assert (
        validate_compatible(
            _meta(data_dim=(1, 28, 28)),
            sde_name="vp",
            model_spec=_MODEL_SPEC,
            data_dim=(1, 28, 28),
        )
        is None
    )


def test_validate_compatible_model_spec_difiere():
    """Receta de red distinta (kwargs distintos) → ``ValueError`` (2.5)."""
    otro = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 128, "num_blocks": 4}}
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(model=_MODEL_SPEC), sde_name="vp", model_spec=otro, data_dim=2
        )


def test_validate_compatible_ambos_model_none_ok():
    """``model_spec=None`` y ``meta`` sin clave ``model`` → match (``None == None``) (2.5)."""
    meta = _meta(model=None)  # sin clave 'model'
    assert "model" not in meta
    assert (
        validate_compatible(meta, sde_name="vp", model_spec=None, data_dim=2) is None
    )


def test_validate_compatible_receta_presente_vs_ausente_difiere():
    """Uno con receta y el otro sin → mismatch en ambos sentidos (2.5)."""
    # meta CON receta, corrida SIN.
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(model=_MODEL_SPEC), sde_name="vp", model_spec=None, data_dim=2
        )
    # meta SIN receta, corrida CON.
    with pytest.raises(ValueError):
        validate_compatible(
            _meta(model=None), sde_name="vp", model_spec=_MODEL_SPEC, data_dim=2
        )


# ------------------------------ load_resume integra la validación (2.5)


def test_load_resume_incompatible_levanta(tmp_path):
    """Un ``expected`` incompatible con el ``meta`` (SDE distinta) → ``ValueError`` (2.5)."""
    weights, _sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, sde_name="vp", model_spec=_MODEL_SPEC, start_step=5
    )
    expected = {"sde_name": "ve", "model_spec": _MODEL_SPEC, "data_dim": 2}

    with pytest.raises(ValueError):
        load_resume(weights, expected=expected)


def test_load_resume_valida_antes_de_exigir_el_sidecar(tmp_path):
    """La compatibilidad se chequea ANTES de exigir el sidecar (2.5 precede a 3.6).

    Con un ``expected`` incompatible y sin sidecar, ``load_resume`` falla por incompatibilidad
    —no por el sidecar faltante— (el mensaje no nombra el sidecar).
    """
    weights, sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, sde_name="vp", model_spec=_MODEL_SPEC, with_sidecar=False
    )
    expected = {"sde_name": "ve", "model_spec": _MODEL_SPEC, "data_dim": 2}

    with pytest.raises(ValueError) as exc:
        load_resume(weights, expected=expected)

    assert sidecar.name not in str(exc.value)  # falló por compat, no por el sidecar


# =============================================================================
# Task 4 — Orquestación del CLI de entrenamiento (integración)
# =============================================================================
#
# Tests de nivel CLI de ``scripts/train.py`` (que no es un módulo del paquete): se carga su
# ``main`` con ``importlib.util.spec_from_file_location`` (el script auto-inyecta ``src`` en
# ``sys.path``) y se lo invoca con ``main(argv=[...])``. Se usan configs YAML mínimos en
# ``tmp_path`` (``num_steps`` chico, ``checkpoint_every`` > 0, ``out.checkpoint`` bajo
# ``tmp_path``) para correr en segundos de CPU y ejercitar el wiring skip/resume/fresh, el
# callback corregido (pesos + sidecar) y el reporte de la acción.

import importlib.util

yaml = pytest.importorskip("yaml")

_TRAIN_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train.py"


def _load_train_main():
    """Carga ``main`` de ``scripts/train.py`` como función invocable (el script no es un módulo)."""
    spec = importlib.util.spec_from_file_location("train_cli_under_test", _TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _write_cli_config(
    tmp_path,
    *,
    num_steps=6,
    checkpoint_every=2,
    with_checkpoint=True,
    with_loss_curve=False,
):
    """Escribe un ``.yaml`` mínimo (ScoreMLP chico) y devuelve ``(config_path, ckpt_path)``.

    Rutas de salida absolutas bajo ``tmp_path`` (posix, que pathlib maneja en Windows) para no
    depender del cwd. ``ckpt_path`` es ``None`` si ``with_checkpoint=False``.
    """
    run_dir = tmp_path / "run"
    ckpt = run_dir / "vp_gaussian.pt"
    out: dict = {}
    if with_checkpoint:
        out["checkpoint"] = ckpt.as_posix()
    if with_loss_curve:
        out["loss_curve"] = (run_dir / "vp_gaussian_loss.png").as_posix()
    cfg = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {
            "num_steps": num_steps,
            "lr": 0.002,
            "seed": 0,
            "checkpoint_every": checkpoint_every,
        },
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1},
        "out": out,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(config_path), (ckpt if with_checkpoint else None)


# ---------------------------------------- corrida fresca: final + snapshot + sidecar (1.1)


def test_cli_fresh_crea_final_snapshot_y_sidecar(tmp_path):
    """Corrida fresca con checkpointing intermedio: crea el final ``.pt``, al menos un snapshot
    ``…_stepNNNNN.pt`` y su sidecar ``…_stepNNNNN.resume.pt`` (prueba la corrección del callback,
    1.1).
    """
    cfg, ckpt = _write_cli_config(tmp_path, num_steps=6, checkpoint_every=2)
    main = _load_train_main()

    rc = main(["--config", cfg])

    assert rc == 0
    assert ckpt.exists()  # checkpoint final (3.9)
    snaps = discover_snapshots(ckpt)  # excluye final/best/sidecars
    assert snaps, "debería haber al menos un snapshot intermedio"
    step, snap_path = snaps[-1]
    sidecar = resume_sidecar_path(snap_path)
    assert sidecar.exists(), "el callback debe persistir el sidecar de resume junto a los pesos"
    # el sidecar es cargable y trae el estado de resume (paso + azar + optimizador).
    sc = load_resume_state(sidecar)
    assert sc["step"] == step
    assert set(sc) == SIDECAR_KEYS


# ---------------------------------------------------- skip si el final existe (3.1) + force (3.2)


def test_cli_skip_si_final_existe_y_force_reentrena(tmp_path, capsys):
    """Con el final presente la corrida se saltea sin sobrescribir (3.1); ``--force`` reentrena
    (reanudando desde el snapshot más nuevo) y sí reescribe el final (3.2).
    """
    cfg, ckpt = _write_cli_config(tmp_path, num_steps=6, checkpoint_every=2)
    main = _load_train_main()
    assert main(["--config", cfg]) == 0
    assert ckpt.exists()

    # Sentinela robusto (independiente de la resolución de mtime): si el skip NO reentrena, estos
    # bytes sobreviven; si el force reentrena, se sobrescriben con un checkpoint real.
    sentinel = b"SENTINEL-NO-TOCAR"
    ckpt.write_bytes(sentinel)
    capsys.readouterr()  # limpiar

    # (3.1) re-correr con el final presente => skip, salida 0, sin tocar el archivo.
    rc = main(["--config", cfg])
    assert rc == 0
    assert ckpt.read_bytes() == sentinel  # NO se sobrescribió
    out_skip = capsys.readouterr().out
    assert "--force" in out_skip and ("complet" in out_skip.lower())

    # (3.2) --force => reentrena (reanuda desde el más nuevo) y reescribe el final.
    rc = main(["--config", cfg, "--force"])
    assert rc == 0
    assert ckpt.read_bytes() != sentinel  # se sobrescribió con un checkpoint real
    out_force = capsys.readouterr().out
    assert "ya completa" not in out_force.lower()  # NO se salteó
    assert "reanud" in out_force.lower()  # force reanuda desde el snapshot más nuevo (3.2/3.3)
    # sigue siendo un checkpoint válido que cubre toda la corrida.
    _, meta = load_checkpoint(ckpt)
    assert len(meta["history"]) == 6


# ---------------------------------------------- auto-resume desde el más nuevo (3.3, 3.8, 3.9)


def test_cli_auto_resume_recrea_final(tmp_path, capsys):
    """Corrida interrumpida simulada (snapshots + sidecars, sin final): re-correr auto-reanuda
    desde el snapshot más nuevo, reporta la acción (3.8) y recrea el final cubriendo toda la
    corrida (3.3, 3.9).
    """
    cfg, ckpt = _write_cli_config(
        tmp_path, num_steps=6, checkpoint_every=2, with_loss_curve=True
    )
    main = _load_train_main()
    assert main(["--config", cfg]) == 0
    snaps = discover_snapshots(ckpt)
    assert snaps
    newest_step = snaps[-1][0]

    # Simular la interrupción: se borra el final pero quedan los snapshots + sidecars.
    ckpt.unlink()
    loss_curve = ckpt.parent / "vp_gaussian_loss.png"
    if loss_curve.exists():
        loss_curve.unlink()
    assert not ckpt.exists()
    capsys.readouterr()  # limpiar

    rc = main(["--config", cfg])

    assert rc == 0
    out = capsys.readouterr().out
    assert "reanud" in out.lower()  # reporta la acción de resume (3.8)
    assert f"paso {newest_step}" in out  # y el origen (paso del snapshot elegido)
    assert ckpt.exists()  # final recreado (3.9)
    _, meta = load_checkpoint(ckpt)
    assert len(meta["history"]) == 6  # la curva/historia cubren toda la corrida (3.9)
    assert loss_curve.exists()  # y la curva de pérdida se reescribe sobre toda la corrida (3.9)


# ------------------------------------------------ advertencia sin puntos de reanudación (1.5)


def test_cli_advierte_sin_puntos_de_reanudacion(tmp_path, capsys):
    """Con ``checkpoint_every=0`` el CLI advierte que la corrida NO dejará puntos de reanudación
    (1.5).
    """
    cfg, _ = _write_cli_config(tmp_path, num_steps=4, checkpoint_every=0)
    main = _load_train_main()

    rc = main(["--config", cfg])

    assert rc == 0
    out = capsys.readouterr().out
    assert "checkpoint_every" in out
    assert "reanud" in out.lower()


# ------------------------------------------------ --resume-from por paso puntual (3.5)


def test_cli_resume_from_step_selecciona_snapshot(tmp_path, capsys):
    """``--resume-from N`` reanuda desde ese snapshot puntual (no el más nuevo) y completa (3.5)."""
    cfg, ckpt = _write_cli_config(tmp_path, num_steps=6, checkpoint_every=2)
    main = _load_train_main()
    assert main(["--config", cfg]) == 0
    ckpt.unlink()  # sin final, para que la acción sea claramente un resume
    snaps = discover_snapshots(ckpt)
    assert len(snaps) >= 2
    chosen_step = snaps[0][0]  # el MÁS VIEJO, distinto del auto (más nuevo)
    capsys.readouterr()

    rc = main(["--config", cfg, "--resume-from", str(chosen_step)])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"paso {chosen_step}" in out  # reanudó desde el snapshot elegido, no el más nuevo
    assert ckpt.exists()
    _, meta = load_checkpoint(ckpt)
    assert len(meta["history"]) == 6


# ------------------------------------------------ --resume-from inexistente => error limpio (3.7)


def test_cli_resume_from_inexistente_error_exit_2(tmp_path, capsys):
    """``--resume-from`` con un paso inexistente => salida 2 (error limpio); lista disponibles."""
    cfg, ckpt = _write_cli_config(tmp_path, num_steps=6, checkpoint_every=2)
    main = _load_train_main()
    assert main(["--config", cfg]) == 0
    ckpt.unlink()
    capsys.readouterr()

    rc = main(["--config", cfg, "--resume-from", "99999"])

    assert rc == 2  # ValueError mapeado a exit code 2 (patrón del script)


# =============================================================================
# ema-weights task 3 — sidecar extendido (crudos + sombra) y carga del punto
# =============================================================================
#
# Con EMA activo el checkpoint de pesos publica la SOMBRA (R2.1/R2.2), así que los pesos con los
# que hay que **continuar entrenando** —los crudos del momento— tienen que viajar en el sidecar y
# ``load_resume`` tiene que preferirlos (R3.1/R3.2). Las claves nuevas son **opcionales**: los
# campos requeridos del sidecar no cambian, así que un sidecar anterior a esta spec (corrida sin
# EMA, cuyo checkpoint ya es crudo) se sigue cargando exactamente como hoy (R3.3).

#: Claves del sidecar de una corrida CON EMA: las de siempre más las dos nuevas.
_SIDECAR_KEYS_EMA = SIDECAR_KEYS | {"raw_model_state", "ema_state"}


def _crudos_y_sombra(net):
    """Dos fotos distinguibles del ``state_dict`` de ``net``: los "crudos" y una "sombra".

    La sombra se desplaza (``+1``) para que ninguna comparación tensor a tensor pueda pasar por
    casualidad (un ``mul`` dejaría iguales los tensores nulos, p. ej. los bias recién inicializados).
    """
    crudos = {k: v.detach().clone() for k, v in net.state_dict().items()}
    sombra = {k: v.detach().clone().add_(1.0) for k, v in net.state_dict().items()}
    return crudos, sombra


# ------------------------------------------------- el sidecar lleva crudos + sombra (3.1)


def test_sidecar_con_ema_persiste_crudos_y_sombra(tmp_path):
    """Con EMA el sidecar incluye además los pesos crudos del momento y la sombra (3.1).

    Round-trip por disco: las dos claves nuevas vuelven tensor a tensor (``torch.equal``) sin
    alterar los campos requeridos (el ``step`` y el azar siguen ahí, el ``history`` sigue afuera).
    """
    net, opt = _net_with_optimizer_state()
    crudos, sombra = _crudos_y_sombra(net)
    resume = _resume_state(opt, start_step=5)
    resume.raw_model_state = crudos
    resume.ema_state = sombra

    path = tmp_path / "vp_gaussian_step00005.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == _SIDECAR_KEYS_EMA
    assert loaded["step"] == 5
    assert "history" not in loaded  # el contrato del sidecar no cambia (1.3)
    for key in crudos:
        assert torch.equal(loaded["raw_model_state"][key], crudos[key]), key
        assert torch.equal(loaded["ema_state"][key], sombra[key]), key


def test_sidecar_sin_ema_no_agrega_claves(tmp_path):
    """Sin EMA el sidecar es **el de hoy**: ni ``raw_model_state`` ni ``ema_state`` (3.3).

    Las claves nuevas se persisten **solo si están**, así que el contenido del sidecar de una
    corrida sin EMA queda idéntico al de antes de esta feature — es lo que hace que la
    retrocompatibilidad sea por construcción y no por tolerancia del lector.
    """
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=2)
    assert resume.raw_model_state is None  # default del dataclass: sin EMA
    assert resume.ema_state is None

    path = tmp_path / "sin_ema.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == SIDECAR_KEYS


# --------------------------------- load_resume prefiere los crudos del sidecar (3.1, 3.2)


def _checkpoint_ema_con_sidecar(tmp_path, *, start_step=4, decay=0.6):
    """Punto de reanudación de una corrida **con EMA**, como lo deja el CLI.

    El checkpoint de pesos publica la **sombra** (política de :func:`save_checkpoint`) y el sidecar
    hermano lleva los **crudos** del momento más la sombra.

    Returns:
        ``(weights, crudos, sombra, history)``.
    """
    net, opt = _net_with_optimizer_state()
    crudos, sombra = _crudos_y_sombra(net)
    history = [1.0, 0.5, 0.25, 0.2]
    result = TrainResult(
        net=net,
        history=list(history),
        config=TrainConfig(ema_decay=decay),
        sde_name="vp",
        data_dim=2,
        ema_state=sombra,
    )
    weights = tmp_path / f"vp_gaussian_step{start_step:05d}.pt"
    save_checkpoint(result, weights, model_spec=_MODEL_SPEC)

    resume = _resume_state(opt, start_step=start_step, history=history)
    resume.raw_model_state = crudos
    resume.ema_state = sombra
    save_resume_state(resume_sidecar_path(weights), resume)
    return weights, crudos, sombra, history


def test_load_resume_con_ema_devuelve_los_crudos_del_sidecar(tmp_path):
    """El checkpoint publica EMA ⇒ ``load_resume`` devuelve los **crudos** del sidecar (3.1, 3.2).

    Es el punto que el enfoque A obliga a extender: devolver el ``state_dict`` del checkpoint de
    pesos cargaría la **sombra** como punto de partida del entrenamiento (pesos promediados
    disfrazados de crudos, corriendo el optimizador sobre un estado que nunca existió). El
    ``meta``/``history`` siguen viniendo del checkpoint de pesos, y el :class:`ResumeState` se arma
    con la sombra para restaurarla en el loop.
    """
    weights, crudos, sombra, history = _checkpoint_ema_con_sidecar(tmp_path, start_step=4)
    expected = {"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}

    state_dict, meta, resume = load_resume(weights, expected=expected)

    # Lo publicado en el checkpoint es la sombra; lo que devuelve load_resume son los crudos.
    publicado, _ = load_checkpoint(weights)
    for key in crudos:
        assert torch.equal(publicado[key], sombra[key]), f"el checkpoint no publicó EMA en {key}"
        assert torch.equal(state_dict[key], crudos[key]), f"no devolvió los crudos en {key}"
    assert any(not torch.equal(state_dict[k], publicado[k]) for k in crudos)  # crudos ≠ EMA

    # El ResumeState lleva la sombra (la restaura ``train``); los crudos NO se duplican en él —
    # ya se devuelven como ``state_dict`` para que el caller los cargue en la red.
    assert resume.ema_state is not None
    for key in sombra:
        assert torch.equal(resume.ema_state[key], sombra[key]), key
    assert resume.raw_model_state is None

    # El resto del contrato de load_resume, intacto: paso y azar del sidecar, history del meta.
    assert resume.start_step == 4
    assert resume.history == history == meta["history"]
    assert meta["ema"] == {"decay": pytest.approx(0.6)}


def test_load_resume_sidecar_viejo_sin_claves_de_ema_como_hoy(tmp_path):
    """Sidecar anterior a esta spec (sin claves de EMA) ⇒ carga **exactamente como hoy** (3.3).

    El sidecar de una corrida sin EMA no tiene ``raw_model_state``, y su checkpoint de pesos ya
    publica los crudos: ``load_resume`` devuelve ese ``state_dict`` como siempre y el
    :class:`ResumeState` queda sin sombra (lo que después habilita el camino sin EMA del loop).
    """
    weights, sidecar, saved = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, start_step=5
    )
    assert set(load_resume_state(sidecar)) == SIDECAR_KEYS  # sidecar "viejo": sin claves nuevas

    publicado, _ = load_checkpoint(weights)
    state_dict, meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )

    assert state_dict.keys() == publicado.keys()
    for key in publicado:
        assert torch.equal(state_dict[key], publicado[key]), key  # los pesos del checkpoint
    assert resume.ema_state is None
    assert resume.raw_model_state is None
    assert "ema" not in meta
    assert resume.start_step == 5
    assert torch.equal(resume.generator_state, saved.generator_state)


# =============================================================================
# gpu-training-efficiency task 3 — persistencia/restauración del escalador AMP
# =============================================================================
#
# Con AMP activo el loop mantiene un ``GradScaler`` cuyo estado (el *scale factor* y su tracker de
# crecimiento) debe viajar en el sidecar para que la reanudación sea fiel (R2.1/R2.2). La clave es
# **opcional** —calca ``ema_state``—: sin AMP el sidecar no gana ninguna clave y los sidecars
# previos siguen válidos (R2.4). Los dos config↔sidecar incoherentes se rechazan (R2.5). En CPU el
# escalador va deshabilitado (``state_dict() == {}``), así que "presencia" se decide por ``is None``,
# no por dict vacío.

#: Claves del sidecar de una corrida CON AMP: las de siempre más la nueva.
_SIDECAR_KEYS_AMP = SIDECAR_KEYS | {"scaler_state"}


def _scaler_state_no_vacio() -> dict:
    """``state_dict()`` de un ``GradScaler`` HABILITADO (no vacío), para un round-trip real.

    En CPU el loop usa un escalador **deshabilitado** (``state_dict() == {}``); acá se fuerza uno
    habilitado para que el round-trip pruebe que las claves del escalador (scale + tracker)
    sobreviven la serialización, igual que ``_net_with_optimizer_state`` fuerza estado en Adam.
    """
    return torch.amp.GradScaler("cpu", enabled=True).state_dict()


def _data_prohibida_amp():
    """Fuente que revienta si se la consume: prueba que los guards fallan ANTES de tocar datos."""
    raise AssertionError("no se debe consumir datos: el guard de AMP falla fail-fast antes")
    yield  # pragma: no cover  (nunca se alcanza; hace de esto un generador)


# ------------------------------------------------- round-trip del sidecar (2.1)


def test_sidecar_con_amp_persiste_scaler_state(tmp_path):
    """Con AMP el sidecar incluye ``scaler_state`` y vuelve por disco intacto (2.1).

    Las claves requeridas no cambian; la clave nueva se suma y round-trippea (el *scale factor* y
    su tracker se preservan). El ``history`` sigue afuera (1.3).
    """
    _, opt = _net_with_optimizer_state()
    scaler_state = _scaler_state_no_vacio()
    resume = _resume_state(opt, start_step=5)
    resume.scaler_state = scaler_state

    path = tmp_path / "vp_gaussian_step00005.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == _SIDECAR_KEYS_AMP
    assert "history" not in loaded  # el contrato del sidecar no cambia (1.3)
    assert loaded["scaler_state"] == scaler_state  # scale + tracker preservados


def test_sidecar_sin_amp_no_agrega_scaler_state(tmp_path):
    """Sin AMP el sidecar es **el de hoy**: ni ``scaler_state`` (2.4).

    La clave nueva se persiste **solo si está**, así que el contenido del sidecar de una corrida
    sin AMP queda idéntico al de antes de esta feature — retrocompatibilidad por construcción.
    """
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=2)
    assert resume.scaler_state is None  # default del dataclass: sin AMP

    path = tmp_path / "sin_amp.resume.pt"
    save_resume_state(path, resume)
    loaded = load_resume_state(path)

    assert set(loaded) == SIDECAR_KEYS
    assert "scaler_state" not in loaded


# ----------------------------------- load_resume expone el scaler_state (2.2)


def test_load_resume_devuelve_scaler_state_del_sidecar(tmp_path):
    """El sidecar trae ``scaler_state`` ⇒ ``load_resume`` lo devuelve en el ``ResumeState`` (2.2)."""
    net, opt = _net_with_optimizer_state()
    history = [1.0, 0.5, 0.25, 0.2, 0.1]
    result = TrainResult(net=net, history=list(history), sde_name="vp", data_dim=2)
    weights = tmp_path / "vp_gaussian_step00005.pt"
    save_checkpoint(result, weights, model_spec=_MODEL_SPEC)

    scaler_state = _scaler_state_no_vacio()
    saved = _resume_state(opt, start_step=5, history=history)
    saved.scaler_state = scaler_state
    save_resume_state(resume_sidecar_path(weights), saved)

    _sd, _meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )

    assert resume.scaler_state == scaler_state


def test_load_resume_sidecar_sin_scaler_state_es_none(tmp_path):
    """Sidecar sin la clave (corrida sin AMP / anterior a la feature) ⇒ ``scaler_state is None`` (2.4)."""
    weights, sidecar, _ = _build_checkpoint_and_sidecar(
        tmp_path, model_spec=_MODEL_SPEC, start_step=5
    )
    assert "scaler_state" not in load_resume_state(sidecar)  # sidecar "viejo"

    _sd, _meta, resume = load_resume(
        weights, expected={"sde_name": "vp", "model_spec": _MODEL_SPEC, "data_dim": 2}
    )
    assert resume.scaler_state is None


# ------------------------------------------------- guards cruzados (2.5)


def test_train_con_amp_rechaza_reanudar_sin_scaler_en_el_sidecar():
    """AMP pedido + sidecar sin escalador ⇒ ``ValueError`` fail-fast antes de entrenar (2.5).

    No se reanuda con un escalador inventado: el punto de reanudación no lo guardó (¿la corrida
    original corrió sin AMP?). El guard va con el resto del fail-fast (la fuente prohibida convierte
    cualquier consumo de datos en un ``AssertionError`` distinto del error esperado).
    """
    sde = make_sde("vp")
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=5)  # scaler_state None por defecto
    cfg = TrainConfig(num_steps=8, seed=0, amp=True)

    with pytest.raises(ValueError) as exc:
        train(sde, _small_net(sde), _data_prohibida_amp(), cfg, resume=resume)

    assert "escalador" in str(exc.value).lower() or "amp" in str(exc.value).lower()


def test_train_sin_amp_rechaza_reanudar_con_scaler_en_el_sidecar():
    """Escalador en el sidecar + AMP no pedido ⇒ ``ValueError`` (2.5, simetría del guard).

    La corrida original usaba AMP; continuar sin AMP descartaría el escalador en silencio. Se falla
    con el mismo criterio fail-fast.
    """
    sde = make_sde("vp")
    _, opt = _net_with_optimizer_state()
    resume = _resume_state(opt, start_step=5)
    resume.scaler_state = _scaler_state_no_vacio()
    cfg = TrainConfig(num_steps=8, seed=0)  # sin AMP

    with pytest.raises(ValueError) as exc:
        train(sde, _small_net(sde), _data_prohibida_amp(), cfg, resume=resume)

    assert "escalador" in str(exc.value).lower() or "amp" in str(exc.value).lower()


# ------------------------------------------------- fidelidad en CPU (2.3)


def test_resume_con_amp_equivalente_a_corrida_ininterrumpida():
    """Gate de fidelidad (2.3): con AMP activo y orden fijo, reanudar equivale a no interrumpir.

    Mismo montaje que ``test_resume_equivalente_a_corrida_ininterrumpida`` pero con ``amp=True``: el
    snapshot del paso ``_N`` lleva el ``scaler_state`` (en CPU va deshabilitado ⇒ ``{}``, pero
    **presente**, no ``None``), y la corrida B —red fresca con los pesos del paso ``_N``, reanudada
    con AMP— reproduce A: ``history`` idéntico y pesos ``allclose``.
    """
    sde = make_sde("vp")
    batch = _fixed_batch()
    frozen: dict[str, TrainSnapshot] = {}

    def capture(tag, snap):
        frozen[tag] = copy.deepcopy(snap)

    net_a = _equiv_net()
    result_a = train(
        sde,
        net_a,
        _const_source(batch),
        TrainConfig(num_steps=_TOTAL, checkpoint_every=_N, seed=0, amp=True),
        on_checkpoint=capture,
    )

    snap = frozen[f"step{_N:05d}"]
    assert snap.resume.start_step == _N
    assert snap.resume.scaler_state is not None  # el escalador viajó en el sidecar (2.1)

    net_b = _equiv_net()
    net_b.load_state_dict(snap.result.net.state_dict())  # pesos congelados del paso _N
    result_b = train(
        sde,
        net_b,
        _const_source(batch),  # misma fuente de orden fijo
        TrainConfig(num_steps=_TOTAL, seed=0, amp=True),
        resume=snap.resume,
    )

    assert result_b.history == result_a.history
    assert _weights_allclose(net_a, net_b)
