"""Tests del módulo de entrenamiento (denoising score matching, `diffusion.training`).

Torch es dependencia dura del módulo, así que se hace `importorskip` al tope. Las corridas de
entrenamiento usan redes y datasets chicos para correr en CPU en segundos.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from diffusion.data_generation import infinite_bare, make_distribution
from diffusion.models import ScoreMLP, ScoreModel, make_model
from diffusion.sde import make_sde
from diffusion.training import (
    RunSpec,
    TrainConfig,
    TrainResult,
    TrainSnapshot,
    build_run,
    dsm_loss,
    load_checkpoint,
    load_config,
    sample_timesteps,
    save_checkpoint,
    train,
)

SDE_NAMES = ["vp", "ve", "sub_vp"]


def _small_net(sde) -> ScoreMLP:
    return ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)


class _DummyScoreNet(torch.nn.Module):
    """Red de score mínima N-D: devuelve el estado escalado por un parámetro entrenable.

    Respeta el contrato ``(x, t) -> score`` con salida de la **misma shape** que ``x`` (funciona
    con cualquier rango, incluido ``(B, C, H, W)``) y expone un parámetro para que la pérdida
    tenga camino de gradiente. Sustituye a la U-Net para ejercitar ``dsm_loss`` sobre un batch
    tipo-imagen sin depender de la arquitectura convolucional.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.scale * x


def _data(dist, n=256, batch_size=64, *, shuffle=True):
    """Fuente infinita de tensores crudos que consume ``train`` (loader finito envuelto)."""
    return infinite_bare(dist.dataloader(n, batch_size, shuffle=shuffle))


def _tiny_config(**overrides) -> TrainConfig:
    base = dict(num_steps=4, seed=0)
    base.update(overrides)
    return TrainConfig(**base)


# ------------------------------------------------------------------ dsm_loss


@pytest.mark.parametrize("name", SDE_NAMES)
def test_dsm_loss_escalar_finito_con_gradiente(name):
    """La pérdida es un escalar finito y diferenciable, con gradientes finitos en la red.

    Parametrizado por las 3 SDEs.
    """
    sde = make_sde(name)
    net = _small_net(sde)
    x0 = torch.randn(32, 2)
    t = torch.rand(32) * sde.T

    loss = dsm_loss(net, sde, x0, t)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)


@pytest.mark.parametrize("name", SDE_NAMES)
def test_dsm_loss_nd_safe_sobre_batch_tipo_imagen(name):
    """La pérdida DSM broadcastea el peso por muestra sobre un batch tipo-imagen (5.3).

    Con una SDE de forma de evento ``(3, 8, 8)`` el peso ``λ(t) = std²`` sale ``(B, 1, 1, 1)``
    (rank-matched vía ``_expand_t``) y broadcastea contra el error ``(B, 3, 8, 8)`` sin error de
    shape: la pérdida es un escalar finito y diferenciable, con gradiente en la red. Confirma
    que ``dsm_loss`` es **N-D-safe sin cambios** en ``losses.py`` (el peso ya queda rank-matched;
    un peso plano ``(B, 1)`` reventaría el broadcasting contra ``(B, 3, 8, 8)``).

    Se usa una red dummy que devuelve la shape del estado (no la U-Net) para aislar el
    broadcasting de la pérdida de la arquitectura. Parametrizado por las 3 SDEs.
    """
    event_shape = (3, 8, 8)
    sde = make_sde(name, data_dim=event_shape)
    net = _DummyScoreNet()
    x0 = torch.randn(4, *event_shape)
    t = torch.rand(4) * sde.T

    loss = dsm_loss(net, sde, x0, t)

    assert loss.ndim == 0  # escalar 0-dim: .mean() colapsó todas las dims de evento
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert grads  # la red dummy expone parámetros (camino de gradiente real)
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)


def test_dsm_loss_reproducible_con_generator():
    sde = make_sde("vp")
    net = _small_net(sde)
    x0 = torch.randn(16, 2)
    t = torch.rand(16)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    assert torch.equal(
        dsm_loss(net, sde, x0, t, generator=g1),
        dsm_loss(net, sde, x0, t, generator=g2),
    )


# ------------------------------------------------------------ sample_timesteps


def test_sample_timesteps_rango_shape_y_reproducibilidad():
    g1 = torch.Generator().manual_seed(0)
    t = sample_timesteps(1000, T=1.0, t_eps=1e-3, generator=g1)
    assert t.shape == (1000,)
    assert float(t.min()) >= 1e-3 - 1e-9
    assert float(t.max()) <= 1.0 + 1e-9

    g2 = torch.Generator().manual_seed(0)
    t2 = sample_timesteps(1000, T=1.0, t_eps=1e-3, generator=g2)
    assert torch.equal(t, t2)


def test_sample_timesteps_respeta_horizonte_distinto():
    g = torch.Generator().manual_seed(1)
    t = sample_timesteps(500, T=3.0, t_eps=0.5, generator=g)
    assert float(t.min()) >= 0.5 - 1e-9
    assert float(t.max()) <= 3.0 + 1e-9


# ------------------------------------------------------------------- train


@pytest.mark.parametrize("name", SDE_NAMES)
def test_train_usa_la_red_recibida_y_registra_data_dim(name):
    """train() usa la red que recibe (no construye ninguna) y registra el data_dim de la SDE."""
    sde = make_sde(name)
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    result = train(sde, net, _data(dist), _tiny_config(num_steps=4))

    assert isinstance(result, TrainResult)
    assert result.net is net  # usa la instancia recibida, no una nueva
    assert result.data_dim == sde.data_dim  # = sde.data_dim (fuente del checkpoint)
    assert result.sde_name == name
    assert len(result.history) == 4  # una entrada por paso (num_steps=4)
    assert all(math.isfinite(v) for v in result.history)


def test_history_es_per_step_e_independiente_de_log_every():
    """history guarda la pérdida de CADA paso: len == num_steps y NO depende de log_every.

    Regresión del bug en que ``log_every`` gobernaba la cadencia de registro (y el CLI la forzaba
    a num_steps//10 → ~10 puntos). Ahora es la serie completa; ``log_every`` solo afecta el print.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    for log_every in (0, 2, 5):
        net = _small_net(sde)
        result = train(
            sde, net, _data(dist), TrainConfig(num_steps=7, log_every=log_every, seed=0)
        )
        assert len(result.history) == 7  # una por paso, sin importar log_every
        assert all(math.isfinite(v) for v in result.history)


def test_train_baja_la_perdida():
    """Smoke de aprendizaje: tras muchos pasos la pérdida final es menor que la inicial.

    Se compara la **tendencia** con medias de bloque (primer vs último quinto), no valores paso a
    paso: la pérdida per-step es ruidosa por el t aleatorio de cada paso.
    """
    sde = make_sde("vp")
    dist = make_distribution("mixture", 2, n_components=8, seed=0)
    torch.manual_seed(0)
    net = ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)
    data = _data(dist, n=512, batch_size=128)
    result = train(sde, net, data, TrainConfig(num_steps=240, seed=0))

    assert len(result.history) == 240  # serie per-step completa
    assert all(math.isfinite(v) for v in result.history)
    # Tendencia sobre medias de bloque (la pérdida per-step es ruidosa por el t aleatorio).
    hist = result.history
    k = max(1, len(hist) // 5)
    assert sum(hist[-k:]) / k < sum(hist[:k]) / k


def test_train_reproducible_con_misma_seed():
    def run():
        torch.manual_seed(0)  # fija los pesos iniciales de la red (idénticos entre corridas)
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=1)
        data = _data(dist, n=256, batch_size=64)
        return train(sde, net, data, TrainConfig(num_steps=20, seed=7)).history

    assert run() == pytest.approx(run())


def test_train_con_grad_clip_corre():
    sde = make_sde("ve")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    result = train(sde, net, _data(dist), _tiny_config(num_steps=4, grad_clip=1.0))
    assert len(result.history) == 4
    assert all(math.isfinite(v) for v in result.history)


def test_trainconfig_acotado_al_loop():
    """TrainConfig lleva num_steps + campos del loop y NO acepta los campos removidos (3.1/3.2)."""
    cfg = TrainConfig(
        num_steps=10, lr=1e-3, t_eps=1e-3, grad_clip=1.0, seed=0, device="cpu", log_every=2
    )
    assert cfg.num_steps == 10
    for removed in (
        "epochs", "batch_size", "n_samples", "embed_dim", "hidden_dim", "num_blocks",
        "activation",
    ):
        assert not hasattr(cfg, removed)
    with pytest.raises(TypeError):
        TrainConfig(epochs=5)  # campo removido: ya no es aceptado


# -------------------------------------------------------------- checkpoints


def test_checkpoint_roundtrip(tmp_path):
    """Round-trip model-agnóstico (R5-c): ``save_checkpoint`` guarda ``state_dict`` + ``meta``
    con receta genérica ``model={name, kwargs}`` (sin campos de arquitectura hardcodeados);
    ``load_checkpoint`` devuelve ``(state_dict, meta)`` **sin** reconstruir; y el caller
    rearma la red con ``make_model`` + ``load_state_dict`` obteniendo la misma salida.

    Usa una arquitectura NO por defecto (``hidden_dim=64``, ``num_blocks=2``) a propósito: la
    reconstrucción debe respetar la receta, no caer en los defaults del constructor.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)  # hidden_dim=64, num_blocks=2 (no son los defaults del ScoreMLP)
    result = train(sde, net, _data(dist), _tiny_config(num_steps=2))

    model_spec = {
        "name": "mlp",
        "kwargs": {"data_dim": sde.data_dim, "hidden_dim": 64, "num_blocks": 2},
    }
    path = tmp_path / "ckpt.pt"
    save_checkpoint(result, path, model_spec=model_spec)

    state_dict, meta = load_checkpoint(path)

    # load_checkpoint devuelve el state_dict crudo (no una red) + la metadata (5.2).
    assert isinstance(state_dict, dict)
    assert "output_proj.weight" in state_dict  # es el state_dict, no un objeto red
    # meta model-agnóstica: sde_name / data_dim / history / receta model (5.1).
    assert meta["sde_name"] == "vp"
    assert meta["data_dim"] == sde.data_dim == 2
    assert meta["history"] == pytest.approx(result.history)
    assert meta["model"] == model_spec
    # Sin hiperparámetros de arquitectura hardcodeados fuera de la receta genérica.
    assert set(meta) == {"sde_name", "data_dim", "history", "model"}

    # El caller reconstruye la red con make_model (receta {name, kwargs}) y le carga los pesos.
    recipe = meta["model"]
    net2 = make_model(recipe["name"], **recipe["kwargs"])
    net2.load_state_dict(state_dict)

    x = torch.randn(8, 2)
    t = torch.rand(8)
    result.net.eval()
    net2.eval()
    with torch.no_grad():
        assert torch.allclose(result.net(x, t), net2(x, t))


def test_save_checkpoint_sin_model_spec_omite_la_receta(tmp_path):
    """Sin ``model_spec`` el checkpoint es válido pero no lleva la clave ``model`` (5.1)."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    result = train(sde, _small_net(sde), _data(dist), _tiny_config(num_steps=2))

    path = tmp_path / "ckpt_sin_receta.pt"
    save_checkpoint(result, path)  # sin model_spec
    state_dict, meta = load_checkpoint(path)

    assert isinstance(state_dict, dict)
    assert "model" not in meta
    assert set(meta) == {"sde_name", "data_dim", "history"}


def test_checkpoint_roundtrip_conserva_forma_de_imagen(tmp_path):
    """La forma de evento de una SDE tipo-imagen viaja por la metadata del checkpoint (4.1) y
    ``make_sde`` la reconstruye desde ahí (4.2).

    El checkpoint transporta ``data_dim`` como el valor **crudo** de la SDE: un entero para el
    dato plano 2D o una **tupla** (forma de evento) para imágenes. Se arma un ``TrainResult`` a
    mano (sin correr entrenamiento): la identidad de la red es irrelevante acá —solo se verifica
    el round-trip de la forma en la meta y la reconstrucción de la SDE—. La generación
    end-to-end a ``(n, *E)`` se cubre en los tests de samplers (task 4.2).
    """
    event_shape = (3, 8, 8)
    sde = make_sde("vp", data_dim=event_shape)
    assert sde.data_dim == event_shape  # la SDE conserva el valor crudo (tupla)

    # Red mínima sin entrenar: solo actúa de portador del state_dict; no se reconstruye acá.
    net = ScoreMLP(data_dim=2, hidden_dim=8, num_blocks=1)
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=sde.data_dim)

    path = tmp_path / "ckpt_imagen.pt"
    save_checkpoint(result, path)
    _, meta = load_checkpoint(path)

    # 4.1: la forma de evento (tupla) sobrevive torch.save/torch.load sin perder el tipo.
    assert meta["data_dim"] == event_shape
    assert isinstance(meta["data_dim"], tuple)

    # 4.2: make_sde reconstruye la SDE con esa forma (data_shape normalizada).
    sde2 = make_sde(meta["sde_name"], data_dim=meta["data_dim"])
    assert sde2.data_shape == event_shape


# ------------------------------------------------ checkpointing intermedio


def test_trainconfig_checkpoint_every_default_cero():
    """El switch de checkpointing intermedio arranca apagado (0 = solo el checkpoint final)."""
    assert TrainConfig().checkpoint_every == 0


def test_train_sin_checkpoint_every_no_llama_callback():
    """Gate por defecto: con ``checkpoint_every=0`` el callback NO se invoca (sin regresión)."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    calls: list[str] = []
    result = train(
        sde,
        _small_net(sde),
        _data(dist),
        _tiny_config(num_steps=6),  # checkpoint_every=0 por defecto
        on_checkpoint=lambda tag, snap: calls.append(tag),
    )
    assert calls == []  # ni snapshots periódicos ni "best"
    assert result.history


def test_train_checkpoint_every_emite_periodicos_y_best():
    """Con ``checkpoint_every=N`` el loop emite snapshots periódicos y al menos un "best".

    Los tags periódicos son ``step{N:05d}`` en los múltiplos de ``checkpoint_every``, **excluido
    el último paso** (lo cubre el checkpoint final del caller). Con ``num_steps=9`` y
    ``checkpoint_every=3`` se esperan los pasos 3 y 6 (el 9 es el último y se omite).
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    calls: list[tuple[str, TrainSnapshot]] = []
    net = _small_net(sde)
    result = train(
        sde,
        net,
        _data(dist),
        _tiny_config(num_steps=9, checkpoint_every=3),
        on_checkpoint=lambda tag, snap: calls.append((tag, snap)),
    )

    tags = [tag for tag, _ in calls]
    periodic = [t for t in tags if t.startswith("step")]
    assert periodic == ["step00003", "step00006"]  # múltiplos de 3, sin el último (9)
    assert "best" in tags  # al menos un mínimo de pérdida de intervalo registrado
    # Cada snapshot es un TrainSnapshot cuyo TrainResult apunta a la MISMA red que se entrena.
    assert all(
        isinstance(snap, TrainSnapshot) and snap.result.net is net for _, snap in calls
    )
    assert result.net is net


def test_train_checkpoints_intermedios_persisten_y_cargan(tmp_path):
    """El callback estilo-CLI escribe snapshots hermanos (…_stepNNNNN.pt / …_best.pt) cargables.

    Reproduce el wiring de ``scripts/train.py``: deriva rutas del checkpoint base con
    ``Path.with_stem`` y persiste con ``save_checkpoint``. Verifica que los archivos existen, que
    el último paso NO genera snapshot periódico (lo cubre el final) y que ``load_checkpoint``
    recupera la metadata esperada.
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    base = tmp_path / "vp_gaussian.pt"
    model_spec = {
        "name": "mlp",
        "kwargs": {"data_dim": sde.data_dim, "hidden_dim": 64, "num_blocks": 2},
    }

    def on_checkpoint(tag, snap):
        # Nuevo contrato: snap es un TrainSnapshot; save_checkpoint consume su .result.
        save_checkpoint(snap.result, base.with_stem(f"{base.stem}_{tag}"), model_spec=model_spec)

    train(
        sde,
        _small_net(sde),
        _data(dist),
        _tiny_config(num_steps=4, checkpoint_every=2),
        on_checkpoint=on_checkpoint,
    )

    step_ckpt = tmp_path / "vp_gaussian_step00002.pt"
    best_ckpt = tmp_path / "vp_gaussian_best.pt"
    assert step_ckpt.exists()  # paso 2 (el 4 es el último: lo omite el periódico)
    assert best_ckpt.exists()
    assert not (tmp_path / "vp_gaussian_step00004.pt").exists()  # último paso excluido

    _, meta = load_checkpoint(step_ckpt)
    assert meta["sde_name"] == "vp"
    assert meta["data_dim"] == 2
    assert meta["model"] == model_spec


# ------------------------------------------------------------------- config


def test_build_run_desde_dict():
    """build_run arma (sde, model, data, config): la red por defecto es un MLP dimensionado
    desde la SDE y la data es un iterador infinito de tensores crudos con el batch_size pedido."""
    raw = {
        "sde": {"name": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "data": {
            "shape": "mixture", "dim": 2, "n_samples": 512, "batch_size": 128,
            "n_components": 8, "seed": 0,
        },
        "train": {"num_steps": 3, "lr": 1e-3, "seed": 0},
        "out": {"checkpoint": "models/x.pt", "loss_curve": "models/x.png"},
    }
    spec = build_run(raw)

    assert isinstance(spec, RunSpec)
    assert spec.sde.name == "vp"
    assert spec.config.num_steps == 3
    # Sin bloque 'model:' -> default MLP dimensionado desde el data_dim de la SDE.
    assert isinstance(spec.model, ScoreModel)
    assert isinstance(spec.model, ScoreMLP)
    assert spec.model.data_dim == spec.sde.data_dim
    # 'data' es un iterador infinito que yield-ea tensores crudos (B, data_dim).
    batch = next(iter(spec.data))
    assert batch.shape == (128, 2)  # batch_size del bloque 'data'
    assert spec.checkpoint.name == "x.pt"
    assert spec.loss_curve.name == "x.png"


def test_build_run_con_bloque_model_sobreescribe_el_default():
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1},
    }
    spec = build_run(raw)
    assert isinstance(spec.model, ScoreMLP)
    assert spec.model.hidden_dim == 32
    assert spec.model.num_blocks == 1
    assert spec.model.data_dim == spec.sde.data_dim  # el data_dim lo sigue aportando la SDE


def test_build_run_inyecta_data_dim_entero_en_el_modelo():
    """Gate de config (4.1): con una SDE de dato plano (``data_dim`` entero) el default MLP
    recibe ``data_dim`` inyectado desde la SDE (path 2D, sin regresión)."""
    raw = {
        "sde": {"name": "vp"},  # data_dim entero (2 por defecto)
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
    }
    spec = build_run(raw)

    assert isinstance(spec.sde.data_dim, int)
    assert spec.model_spec["kwargs"]["data_dim"] == spec.sde.data_dim == 2
    assert spec.model.data_dim == 2


def test_build_run_no_inyecta_forma_tupla_en_el_modelo():
    """Gate de config (4.1): con una SDE de forma de evento multidimensional (tupla) la forma
    NO se inyecta como hiperparámetro del modelo (la U-Net trae su propia config).

    Sin el gate, ``setdefault('data_dim', (3,8,8))`` inyectaría la tupla en la receta del MLP
    y ``ScoreMLP(data_dim=(3,8,8))`` reventaría (``int()`` sobre una tupla). El gate deja la
    forma fuera de la config del modelo; acá se usa el default MLP solo para ejercitar la rama
    del gate (en un run real de imágenes el bloque ``model:`` nombra la U-Net)."""
    raw = {
        "sde": {"name": "vp", "data_dim": (3, 8, 8)},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
    }
    spec = build_run(raw)

    assert spec.sde.data_dim == (3, 8, 8)
    assert not isinstance(spec.sde.data_dim, int)
    assert "data_dim" not in spec.model_spec["kwargs"]


def test_build_run_acepta_checkpoint_every():
    """El validador estricto de ``train:`` acepta el nuevo campo ``checkpoint_every`` y lo pasa
    al ``TrainConfig`` (el camino config-driven activa el checkpointing intermedio por YAML)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "checkpoint_every": 2},
    }
    spec = build_run(raw)
    assert spec.config.checkpoint_every == 2


def test_build_run_falla_sin_claves_obligatorias():
    with pytest.raises(ValueError):
        build_run({"data": {"shape": "gaussian"}})  # falta sde.name
    with pytest.raises(ValueError):
        build_run({"sde": {"name": "vp"}})  # falta data.shape


def test_build_run_rechaza_clave_desconocida():
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "lr_typo": 0.1},  # clave desconocida para TrainConfig
    }
    with pytest.raises(ValueError):
        build_run(raw)


def test_load_config_yaml_y_build_run(tmp_path):
    pytest.importorskip("yaml")
    text = (
        "sde:\n"
        "  name: vp\n"
        "data:\n"
        "  shape: gaussian\n"
        "  dim: 2\n"
        "  n_samples: 256\n"
        "  batch_size: 64\n"
        "train:\n"
        "  num_steps: 2\n"
        "  lr: 0.001\n"
    )
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")

    spec = build_run(load_config(path))
    assert spec.sde.name == "vp"
    assert isinstance(spec.model, ScoreMLP)
    assert spec.config.num_steps == 2
    batch = next(iter(spec.data))
    assert batch.shape == (64, 2)  # batch_size del bloque 'data'


def test_build_run_2d_entrena_end_to_end_tras_el_gate():
    """El camino config-driven 2D corre end-to-end tras el gate de configuración (4, 5.1).

    Compone ``build_run`` (que aplica el gate ``setdefault('data_dim', ...)`` solo para el
    entero 2D) con ``train`` sobre el ``RunSpec`` resultante: la red default (MLP dimensionado
    desde la SDE) entrena sin regresión y produce un ``history`` finito con ``data_dim`` entero.
    Ningún otro test corre ``train`` sobre la salida de ``build_run``; esto cierra el path
    config→entrenamiento del toy 2D."""
    raw = {
        "sde": {"name": "vp"},  # data_dim entero (2) -> el gate SÍ inyecta al MLP
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0},
    }
    spec = build_run(raw)

    result = train(spec.sde, spec.model, spec.data, spec.config)

    assert isinstance(result, TrainResult)
    assert result.net is spec.model  # entrena la red que armó build_run (default MLP)
    assert result.data_dim == spec.sde.data_dim == 2  # dato plano 2D, sin regresión
    assert result.history
    assert all(math.isfinite(v) for v in result.history)


# ---------------------------- time-embedding-scale: round-trip de la receta (task 3.1)


def test_roundtrip_receta_con_time_scale(tmp_path):
    """Round-trip de la escala temporal: config → build_run → checkpoint → make_model (3.1–3.3).

    Una corrida config-driven que declara ``time_scale`` en el bloque ``model:`` construye la
    red con ese valor (3.1), lo persiste dentro de la receta ``model_spec`` del checkpoint
    (3.2) y la reconstrucción desde la receta —sin el config original— produce una red con la
    **misma escala** usada al entrenar, a la que el ``state_dict`` guardado carga sin error y
    con salida idéntica (3.3). La aserción sobre el valor reconstruido fallaría si la receta
    dropeara el kwarg (no es solo un no-exception).
    """
    raw = {
        "sde": {"name": "vp"},
        "data": {
            "shape": "mixture", "dim": 2, "n_samples": 128, "batch_size": 64,
            "n_components": 2, "seed": 0,
        },
        "train": {"num_steps": 2, "seed": 0},
        "model": {
            "name": "mlp", "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1,
            "time_scale": 1000.0,
        },
    }
    spec = build_run(raw)

    # 3.1: la corrida config-driven construye la red con la escala declarada...
    assert spec.model.time_scale == pytest.approx(1000.0)
    # ...y la receta del checkpoint la transporta (reconstruible sin el YAML original).
    assert spec.model_spec["kwargs"]["time_scale"] == pytest.approx(1000.0)

    result = train(spec.sde, spec.model, spec.data, spec.config)
    path = tmp_path / "ckpt_time_scale.pt"
    save_checkpoint(result, path, model_spec=spec.model_spec)

    # 3.2: la escala sobrevive el round-trip por disco dentro de la receta.
    state_dict, meta = load_checkpoint(path)
    recipe = meta["model"]
    assert recipe["kwargs"]["time_scale"] == pytest.approx(1000.0)

    # 3.3: la reconstrucción desde la receta produce una red con la MISMA escala entrenada.
    net2 = make_model(recipe["name"], **recipe["kwargs"])
    assert net2.time_scale == pytest.approx(1000.0)
    net2.load_state_dict(state_dict)  # shapes intactas: la escala no agrega parámetros

    # La red reconstruida es funcionalmente la entrenada (misma salida sobre los mismos insumos).
    x = torch.randn(8, 2)
    t = torch.rand(8)
    result.net.eval()
    net2.eval()
    with torch.no_grad():
        assert torch.allclose(result.net(x, t), net2(x, t))


def test_receta_vieja_sin_time_scale_reconstruye_con_default(tmp_path):
    """Una receta anterior al cambio (sin la clave ``time_scale``) reconstruye con el default (3.4).

    Se arma un checkpoint cuya receta NO trae ``time_scale`` (formato pre-cambio) y se verifica
    que ``make_model`` reconstruye sin error una red con la escala default retrocompatible
    (``time_scale == 1.0``) y que el ``state_dict`` guardado carga sobre ella (shapes idénticas:
    la escala no es un parámetro entrenable).
    """
    kwargs = {"data_dim": 2, "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1}
    net = ScoreMLP(**kwargs)  # red pre-cambio: construida sin el kwarg
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=2)
    path = tmp_path / "ckpt_receta_vieja.pt"
    save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": dict(kwargs)})

    state_dict, meta = load_checkpoint(path)
    assert "time_scale" not in meta["model"]["kwargs"]  # receta vieja, sin la clave

    net2 = make_model(meta["model"]["name"], **meta["model"]["kwargs"])  # sin error (3.4)
    assert net2.time_scale == pytest.approx(1.0)  # default retrocompatible
    net2.load_state_dict(state_dict)  # los pesos viejos cargan sin conflicto de shapes

    x = torch.randn(8, 2)
    t = torch.rand(8)
    net.eval()
    net2.eval()
    with torch.no_grad():
        assert torch.allclose(net(x, t), net2(x, t))


def test_generate_from_checkpoint_reconstruye_con_time_scale(tmp_path):
    """La generación checkpoint-driven reconstruye la red con la escala de la receta (3.3).

    End-to-end en el seam real: un checkpoint cuya receta declara ``time_scale=1000.0`` pasa por
    ``generate_from_checkpoint`` (que reconstruye vía ``make_model(recipe)``) y produce muestras
    finitas con la shape esperada, sin error de reconstrucción ni de carga de pesos.
    """
    from diffusion.samplers import generate_from_checkpoint

    kwargs = {
        "data_dim": 2, "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1,
        "time_scale": 1000.0,
    }
    net = ScoreMLP(**kwargs)
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=2)
    path = tmp_path / "ckpt_scale_gen.pt"
    save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": dict(kwargs)})

    x0 = generate_from_checkpoint(path, "pf_ode", n_samples=8, n_steps=5, seed=0)

    assert x0.shape == (8, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


# ------------------- small-t-training-signal: muestreo de tiempos (task 1.1)

from diffusion.training import (  # noqa: E402
    TimeSampler,
    available_time_samplers,
    make_time_sampler,
)


def test_uniform_time_sampler_identidad_por_seed():
    """El default ``uniform`` reproduce EXACTAMENTE ``sample_timesteps`` con el mismo generator (1.2).

    Es el invariante de retrocompatibilidad: con la misma semilla, la variante uniforme del
    submódulo nuevo produce la misma secuencia de tiempos que la fórmula actual del loop
    (``torch.equal``, no ``allclose``: debe ser la misma llamada a ``torch.rand`` seguida de la
    misma transformación afín). Los pesos de la uniforme son ``None`` (likelihood ratio trivial).
    """
    sampler = make_time_sampler("uniform", T=1.0, t_eps=1e-4)
    assert isinstance(sampler, TimeSampler)

    g1 = torch.Generator().manual_seed(42)
    t, weights = sampler.sample(1000, generator=g1)

    g2 = torch.Generator().manual_seed(42)
    t_ref = sample_timesteps(1000, T=1.0, t_eps=1e-4, generator=g2)

    assert torch.equal(t, t_ref)  # identidad bit a bit, no aproximada
    assert weights is None  # la uniforme no corrige: no hay ratio que aplicar


def test_log_uniform_shape_dtype_y_rango():
    """La log-uniforme cumple el contrato de salida: ``(n,)`` float32 en ``[t_eps, T]``."""
    sampler = make_time_sampler("log_uniform", T=1.0, t_eps=1e-4)
    g = torch.Generator().manual_seed(0)
    t, weights = sampler.sample(4096, generator=g)

    assert t.shape == (4096,)
    assert t.dtype == torch.float32
    assert float(t.min()) >= 1e-4 - 1e-9
    assert float(t.max()) <= 1.0 + 1e-9
    assert weights is not None
    assert weights.shape == (4096,)
    assert weights.dtype == torch.float32


def test_log_uniform_concentra_masa_en_t_chico():
    """La distribución recomendada pone ≥ 30% de las muestras en ``[t_eps, 0.01]`` (1.5).

    Con ``t = t_eps·(T/t_eps)^u`` y ``u ~ U(0,1)``, la masa teórica de ``[t_eps, 0.01]`` es
    ``ln(0.01/t_eps)/ln(T/t_eps) = ln(100)/ln(10000) = 0.50`` para ``t_eps=1e-4, T=1``.

    Tolerancia estadística: la fracción empírica con ``n = 200_000`` tiene
    ``SE = sqrt(0.5·0.5/200_000) ≈ 0.0011``, así que el umbral 0.30 queda a ~180·SE del valor
    teórico 0.50 — el test no puede fallar por azar con la seed fija (ni con casi ninguna otra).
    """
    sampler = make_time_sampler("log_uniform", T=1.0, t_eps=1e-4)
    g = torch.Generator().manual_seed(0)
    t, _ = sampler.sample(200_000, generator=g)

    masa = float((t <= 0.01).float().mean())
    assert masa >= 0.30
    # Chequeo de coherencia con la fórmula (0.50 teórico), no parte del gate del requisito.
    assert masa == pytest.approx(0.50, abs=0.01)


def test_log_uniform_pesos_formula_y_media_uno():
    """Los pesos son el likelihood ratio contra la uniforme: fórmula puntual y ``E_q[w] = 1``.

    Fórmula: ``w(t) = p_unif(t)/q(t) = t·ln(T/t_eps)/(T − t_eps)`` — positiva en todo el rango.

    Tolerancia estadística de la media: con ``t`` log-uniforme en ``[1e-4, 1]``,
    ``Var[w] = ln(r)·(T²−t_eps²)/(2(T−t_eps)²) − 1 ≈ ln(10⁴)/2 − 1 ≈ 3.6`` (``sd ≈ 1.9``), así que
    con ``n = 200_000`` la media empírica tiene ``SE ≈ 1.9/√200_000 ≈ 0.0043``; la tolerancia
    ``abs=0.02`` es ≈ 4.7·SE — holgada para la seed fija sin ocultar un ratio mal calculado.
    """
    T, t_eps = 1.0, 1e-4
    sampler = make_time_sampler("log_uniform", T=T, t_eps=t_eps)
    g = torch.Generator().manual_seed(0)
    t, weights = sampler.sample(200_000, generator=g)

    assert torch.all(weights > 0)
    esperado = t * math.log(T / t_eps) / (T - t_eps)
    assert torch.allclose(weights, esperado)
    assert float(weights.mean()) == pytest.approx(1.0, abs=0.02)


def test_time_sampler_factory_y_validacion():
    """Factory fail-fast (1.1, 1.6): registry ordenado, nombre desconocido con opciones listadas,
    y ``t_eps`` fuera de ``(0, T)`` rechazado en construcción."""
    assert available_time_samplers() == ["log_uniform", "uniform"]

    with pytest.raises(ValueError, match="log_uniform.*uniform"):
        make_time_sampler("desconocido", T=1.0, t_eps=1e-4)

    for bad_eps in (0.0, -1e-4, 1.0, 2.0):  # fuera de (0, T) con T=1.0
        with pytest.raises(ValueError):
            make_time_sampler("uniform", T=1.0, t_eps=bad_eps)
        with pytest.raises(ValueError):
            make_time_sampler("log_uniform", T=1.0, t_eps=bad_eps)


@pytest.mark.parametrize("name", ["uniform", "log_uniform"])
def test_time_sampler_reproducible_con_generator(name):
    """Mismo generator → secuencia idéntica; seeds distintas → secuencias distintas."""
    sampler = make_time_sampler(name, T=1.0, t_eps=1e-4)

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    t1, w1 = sampler.sample(256, generator=g1)
    t2, w2 = sampler.sample(256, generator=g2)
    assert torch.equal(t1, t2)
    if w1 is not None:
        assert torch.equal(w1, w2)

    g3 = torch.Generator().manual_seed(8)
    t3, _ = sampler.sample(256, generator=g3)
    assert not torch.equal(t1, t3)


# ------------- small-t-training-signal: pesos en dsm_loss (task 1.2)


def _per_sample_dsm(net, sde, x0, t, *, generator):
    """Réplica por-muestra de la fórmula de ``dsm_loss`` con los mismos componentes públicos.

    Devuelve ``(B,)`` con ``mean_evento( λ(t) · (s_θ − s_real)² )`` por dato: su media de batch
    coincide con el escalar de ``dsm_loss`` (misma seed de ``generator``), lo que se verifica
    explícitamente en el test de equivalencia antes de usarla para estadística.
    """
    x_t, eps = sde.perturb(x0, t, generator=generator)
    score_real, weight = sde.score_target(x0, t, eps)
    with torch.no_grad():
        score_pred = net(x_t, t)
    err = weight * (score_pred - score_real).pow(2)
    return err.reshape(err.shape[0], -1).mean(dim=1)


def test_dsm_loss_retrocompatible_sin_pesos():
    """Sin ``sample_weights`` (omitido o ``None``) el valor es idéntico al actual (1.3, 5.1).

    Se compara bit a bit (``torch.equal``) contra la réplica manual de la fórmula vigente
    (``perturb`` → ``score_target`` → media pesada por λ) con la misma seed, y entre la llamada
    sin kwarg y la llamada con ``sample_weights=None`` explícito: ambas deben recorrer el mismo
    camino de código.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    x0 = torch.randn(32, 2, generator=torch.Generator().manual_seed(1))
    t = sample_timesteps(32, T=sde.T, t_eps=1e-3, generator=torch.Generator().manual_seed(2))

    # Réplica manual de la fórmula actual (el "valor previo") con la misma seed del kernel.
    g_ref = torch.Generator().manual_seed(3)
    x_t, eps = sde.perturb(x0, t, generator=g_ref)
    score_real, weight = sde.score_target(x0, t, eps)
    esperado = (weight * (net(x_t, t) - score_real).pow(2)).mean()

    g1 = torch.Generator().manual_seed(3)
    sin_kwarg = dsm_loss(net, sde, x0, t, generator=g1)
    g2 = torch.Generator().manual_seed(3)
    con_none = dsm_loss(net, sde, x0, t, generator=g2, sample_weights=None)

    assert torch.equal(sin_kwarg, esperado)
    assert torch.equal(con_none, sin_kwarg)


@pytest.mark.parametrize(
    ("data_shape", "make_net"),
    [
        ((32, 2), None),  # dato 2D: red MLP real
        ((8, 1, 4, 4), _DummyScoreNet),  # dato tipo imagen: dummy N-D-safe
    ],
    ids=["2d", "imagen"],
)
def test_dsm_loss_aplica_pesos_por_muestra(data_shape, make_net):
    """Los pesos por muestra escalan la pérdida y broadcastean sobre las dims de evento (1.3).

    - ``w = 1`` → idéntico a sin pesos; ``w = 2`` → exactamente 2×.
    - Shapes ``(B,)`` y ``(B, 1)`` producen el mismo resultado.
    - Funciona igual con dato 2D ``(B, 2)`` y con dato imagen ``(B, 1, 4, 4)`` (el peso se
      expande a ``(B, 1, …, 1)`` como hace ``λ(t)``).
    """
    B = data_shape[0]
    event_shape = data_shape[1:]
    sde = make_sde("vp", data_dim=event_shape[0] if len(event_shape) == 1 else event_shape)
    net = _small_net(sde) if make_net is None else make_net()
    x0 = torch.randn(*data_shape, generator=torch.Generator().manual_seed(5))
    t = sample_timesteps(B, T=sde.T, t_eps=1e-3, generator=torch.Generator().manual_seed(6))

    def loss_con(w):
        g = torch.Generator().manual_seed(7)
        return dsm_loss(net, sde, x0, t, generator=g, sample_weights=w)

    base = loss_con(None)
    assert torch.allclose(loss_con(torch.ones(B)), base)
    assert torch.allclose(loss_con(2.0 * torch.ones(B)), 2.0 * base)
    # (B,) y (B, 1) son shapes equivalentes para el mismo vector de pesos.
    w = torch.rand(B, generator=torch.Generator().manual_seed(8)) + 0.5
    assert torch.allclose(loss_con(w), loss_con(w.reshape(B, 1)))


def test_dsm_loss_equivalencia_en_esperanza_log_uniforme():
    """La log-uniforme corregida estima la MISMA pérdida esperada que la uniforme (1.4, 5.1).

    Es el test clave del importance sampling: con un modelo fijo chico (``ScoreMLP`` sin
    entrenar, seed fija) y la SDE VP, la pérdida esperada ``E[λ(t)·‖s_θ − s_real‖²]`` se estima
    por Monte Carlo de dos formas — (a) ``t`` uniforme sin pesos y (b) ``t`` log-uniforme con su
    likelihood ratio ``w(t)`` — y ambas deben coincidir dentro del error estadístico.

    Tolerancia (derivación): cada estimador es una media de ``n`` términos i.i.d. por-muestra
    (``ℓ_i`` para (a); ``w_i·ℓ_i`` para (b)), así que su error estándar es
    ``SE = sd_empírico/√n``. La diferencia de las dos medias tiene, si fueran independientes,
    ``SE_comb = sqrt(SE_a² + SE_b²)``; acá comparten el mismo ``x0`` (números aleatorios comunes,
    correlación ≥ 0), lo que solo puede REDUCIR la varianza de la diferencia, así que
    ``SE_comb`` es un techo conservador. Se exige ``|media_a − media_b| ≤ 3·SE_comb``: bajo la
    hipótesis nula (mismo objetivo en esperanza, R1.3) un desvío de 3·SE tiene probabilidad
    < 0.3%, y con seeds fijas el test es determinístico — no puede volverse flaky. Con
    ``n = 200_000`` la tolerancia resultante es ~1% de la pérdida, así que un ratio mal
    calculado (p. ej. sin el factor ``ln(T/t_eps)``, que sesga la media en ~7× — 1/ln(1000);
    incluso la falta total de corrección, sesgo ~4% de la media, cae fuera de la tolerancia).

    Las estadísticas por-muestra se calculan con ``_per_sample_dsm`` (réplica de la fórmula con
    los mismos componentes públicos ``perturb``/``score_target``), cuya coherencia con
    ``dsm_loss`` — sin y con pesos — se asserta antes de usarla.
    """
    torch.manual_seed(0)  # inicialización determinística de la red (modelo fijo)
    net = ScoreMLP(data_dim=2, embed_dim=16, hidden_dim=16, num_blocks=1)
    net.eval()
    sde = make_sde("vp")
    T, t_eps, n = sde.T, 1e-3, 200_000

    x0 = torch.randn(n, 2, generator=torch.Generator().manual_seed(10))

    # (a) t uniforme, sin pesos.
    t_unif, w_unif = make_time_sampler("uniform", T=T, t_eps=t_eps).sample(
        n, generator=torch.Generator().manual_seed(11)
    )
    assert w_unif is None
    por_muestra_a = _per_sample_dsm(net, sde, x0, t_unif, generator=torch.Generator().manual_seed(12))

    # (b) t log-uniforme, corregido por el likelihood ratio.
    t_log, w = make_time_sampler("log_uniform", T=T, t_eps=t_eps).sample(
        n, generator=torch.Generator().manual_seed(13)
    )
    por_muestra_b = w * _per_sample_dsm(net, sde, x0, t_log, generator=torch.Generator().manual_seed(14))

    # Coherencia de la réplica: su media de batch ES el escalar de dsm_loss (misma seed),
    # sin pesos y con pesos — así el test mide de verdad la pérdida de producción.
    assert torch.allclose(
        dsm_loss(net, sde, x0, t_unif, generator=torch.Generator().manual_seed(12)),
        por_muestra_a.mean(),
    )
    assert torch.allclose(
        dsm_loss(net, sde, x0, t_log, generator=torch.Generator().manual_seed(14), sample_weights=w),
        por_muestra_b.mean(),
    )

    # Estadística en float64 para que la comparación no dependa de la suma en float32.
    a = por_muestra_a.double()
    b = por_muestra_b.double()
    media_a, media_b = a.mean().item(), b.mean().item()
    se_a = (a.std() / math.sqrt(n)).item()
    se_b = (b.std() / math.sqrt(n)).item()
    se_comb = math.sqrt(se_a**2 + se_b**2)

    assert abs(media_a - media_b) <= 3.0 * se_comb, (
        f"media_unif={media_a:.6f}, media_log_unif_corregida={media_b:.6f}, "
        f"|dif|={abs(media_a - media_b):.6f} > 3·SE_comb={3.0 * se_comb:.6f}"
    )
    # La tolerancia debe ser chica frente a la pérdida (el test no pasa por manga ancha).
    assert 3.0 * se_comb < 0.05 * media_a
