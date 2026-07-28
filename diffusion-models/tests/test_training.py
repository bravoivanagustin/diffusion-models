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


def test_train_checkpoint_every_emite_solo_periodicos():
    """Con ``checkpoint_every=N`` el loop emite **solo** snapshots periódicos (ningún "best").

    Los tags periódicos son ``step{N:05d}`` en los múltiplos de ``checkpoint_every``, **excluido
    el último paso** (lo cubre el checkpoint final del caller). Con ``num_steps=9`` y
    ``checkpoint_every=3`` se esperan los pasos 3 y 6 (el 9 es el último y se omite).

    NB (excepción documentada, R6.2 / tarea 2.3 de la spec ``ema-weights``): este test **existía**
    y exigía el tag ``"best"``; se reescribió para verificar su **ausencia** cuando el mecanismo se
    retiró (R2.6, decisión del autor 27/07/2026: la selección por pérdida cruda per-step es ruidosa
    —``t`` aleatorio— y correlaciona mal con calidad de muestras, que es justo la razón de ser del
    EMA). Es la única excepción a la convención de no modificar tests existentes.
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
    assert tags == ["step00003", "step00006"]  # múltiplos de 3, sin el último (9)
    assert "best" not in tags  # el mecanismo "best" ya no existe (R2.6)
    # Cada snapshot es un TrainSnapshot cuyo TrainResult apunta a la MISMA red que se entrena.
    assert all(
        isinstance(snap, TrainSnapshot) and snap.result.net is net for _, snap in calls
    )
    assert result.net is net


def test_train_checkpoints_intermedios_persisten_y_cargan(tmp_path):
    """El callback estilo-CLI escribe snapshots hermanos (…_stepNNNNN.pt) cargables.

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
    assert not best_ckpt.exists()  # el tag "best" se retiró (R2.6): no hay artefacto que escribir
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


# ------------- small-t-training-signal: config y loop (task 1.3)


def test_trainconfig_time_sampling_default_uniforme():
    """``TrainConfig()`` sin el campo nuevo declara ``time_sampling="uniform"`` (1.2, 5.1).

    El default retrocompatible: quien no configura nada obtiene el muestreo uniforme actual.
    """
    cfg = TrainConfig()
    assert cfg.time_sampling == "uniform"
    # El campo es configurable como cualquier otro del dataclass.
    assert TrainConfig(time_sampling="log_uniform").time_sampling == "log_uniform"


def test_train_default_bit_identico_al_loop_previo():
    """Con el default (``time_sampling="uniform"``) ``train()`` es BIT-IDÉNTICO al loop previo (1.2).

    Prueba de equivalencia determinística elegida: una **réplica manual del código previo del
    loop** (``sample_timesteps`` + ``dsm_loss`` sin pesos, mismo orden de operaciones y mismo
    consumo del RNG) contra ``train()`` con la config default. Ambas corridas parten de los
    mismos pesos iniciales (``torch.manual_seed`` antes de construir la red) y la misma seed del
    loop, así que si el cambio alterara el stream del generator, la secuencia de ``t`` o la
    fórmula de la pérdida, la igualdad EXACTA (no aproximada) de historia y pesos finales
    fallaría. Nota: este test también pasa sobre el código anterior al campo — es el guard de
    regresión del invariante R1.2, no un test de feature.
    """

    def _fresh():
        torch.manual_seed(123)  # pesos iniciales idénticos entre ambas corridas
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=0)
        data = _data(dist, n=128, batch_size=32, shuffle=False)
        return sde, net, data, TrainConfig(num_steps=3, seed=0)

    # (a) train() con la config default (sin declarar time_sampling).
    sde, net_a, data, cfg = _fresh()
    hist_a = train(sde, net_a, data, cfg).history

    # (b) réplica manual del loop previo al cambio: misma inicialización del azar, un batch por
    # paso, t por sample_timesteps y dsm_loss SIN pesos (el código de trainer.py pre-1.3).
    sde, net_b, data, cfg = _fresh()
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    net_b = net_b.to(device)
    net_b.train()
    data_iter = iter(data)
    optimizer = torch.optim.Adam(net_b.parameters(), lr=cfg.lr)
    hist_b = []
    for _ in range(cfg.num_steps):
        x0 = next(data_iter).to(device)
        t = sample_timesteps(x0.shape[0], sde.T, cfg.t_eps, generator=generator, device=device)
        loss = dsm_loss(net_b, sde, x0, t, generator=generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        hist_b.append(loss.item())

    assert hist_a == hist_b  # igualdad exacta de floats por paso, no aproximada
    sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"peso final distinto en {key}"


def test_train_rechaza_time_sampling_desconocido_antes_de_entrenar():
    """Una distribución desconocida revienta con ``ValueError`` ANTES de cualquier paso (1.6).

    El loop construye el sampler una única vez, fail-fast: si el nombre no está registrado, el
    error sale antes de consumir data — la fuente centinela convierte cualquier consumo prematuro
    en un ``AssertionError`` distinto del ``ValueError`` esperado.
    """

    class _DataProhibida:
        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("el loop consumió data antes de validar time_sampling")

    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(num_steps=2, seed=0, time_sampling="desconocida")
    with pytest.raises(ValueError, match="desconocida"):
        train(sde, net, _DataProhibida(), cfg)


def test_train_con_log_uniform_entrena_y_usa_la_distribucion():
    """Con ``time_sampling="log_uniform"`` el loop entrena con pérdidas finitas y la distribución
    declarada realmente gobierna el muestreo: misma seed, historia distinta a la del uniforme."""

    def _run(time_sampling):
        torch.manual_seed(123)
        sde = make_sde("vp")
        net = _small_net(sde)
        dist = make_distribution("gaussian", 2, seed=0)
        cfg = TrainConfig(num_steps=3, seed=0, time_sampling=time_sampling)
        return train(sde, net, _data(dist, n=128, batch_size=32, shuffle=False), cfg).history

    hist_log = _run("log_uniform")
    assert len(hist_log) == 3
    assert all(math.isfinite(v) for v in hist_log)
    assert hist_log != _run("uniform")  # el campo cambia de verdad el muestreo de t


def test_build_run_acepta_time_sampling_y_sigue_estricto(tmp_path):
    """El bloque ``train:`` del YAML acepta la clave nueva y la validación estricta sigue
    rechazando claves inválidas (1.7); la corrida config-driven con la distribución recomendada
    construye el ``TrainConfig`` con ella y entrena end-to-end (observable de la tarea)."""
    pytest.importorskip("yaml")
    text = (
        "sde:\n"
        "  name: vp\n"
        "data:\n"
        "  shape: gaussian\n"
        "  dim: 2\n"
        "  n_samples: 128\n"
        "  batch_size: 64\n"
        "  seed: 0\n"
        "train:\n"
        "  num_steps: 2\n"
        "  seed: 0\n"
        "  time_sampling: log_uniform\n"
    )
    path = tmp_path / "run_log_uniform.yaml"
    path.write_text(text, encoding="utf-8")

    spec = build_run(load_config(path))
    assert spec.config.time_sampling == "log_uniform"

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert len(result.history) == 2
    assert all(math.isfinite(v) for v in result.history)

    # La validación estricta de claves del bloque train: sigue funcionando con el campo nuevo
    # registrado (una clave inválida se rechaza nombrándola).
    bad = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "time_sampling_typo": "log_uniform"},
    }
    with pytest.raises(ValueError, match="time_sampling_typo"):
        build_run(bad)


# ------------- small-t-training-signal: activación config-driven de la parametrización (task 2.2)

from diffusion.models import EpsilonScoreWrapper  # noqa: E402  (sección append-only, task 2.2)


def test_build_run_con_score_parametrization_envuelve_y_persiste_la_clave():
    """Con ``score_parametrization: epsilon`` en el bloque ``model:``, ``build_run`` envuelve la
    red con :class:`EpsilonScoreWrapper` y registra la clave en la receta del checkpoint (2.4, 2.5).

    La receta transporta la clave como hermana de ``name``/``kwargs`` y los ``kwargs`` quedan
    PELADOS (sin la clave): el camino de reconstrucción es ``make_model(name, **kwargs)`` y
    después el wrap — si la clave se filtrara a los kwargs, la factory la rechazaría o la
    ignoraría en silencio.
    """
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 1, "seed": 0},
        "model": {
            "name": "mlp", "hidden_dim": 32, "num_blocks": 1,
            "score_parametrization": "epsilon",
        },
    }
    spec = build_run(raw)

    # La red del RunSpec ES el wrapper, con la red interna construida desde el bloque model:.
    assert isinstance(spec.model, EpsilonScoreWrapper)
    assert isinstance(spec.model.inner, ScoreMLP)
    assert spec.model.inner.hidden_dim == 32
    assert spec.model.inner.num_blocks == 1
    assert spec.model.inner.data_dim == spec.sde.data_dim  # el data_dim lo sigue aportando la SDE

    # La receta persiste la clave al lado de name/kwargs; los kwargs quedan pelados.
    assert spec.model_spec["score_parametrization"] == "epsilon"
    assert spec.model_spec["name"] == "mlp"
    assert "score_parametrization" not in spec.model_spec["kwargs"]


def test_build_run_sin_la_clave_pipeline_identico_al_actual():
    """Sin ``score_parametrization`` el pipeline es idéntico al actual: red pelada (sin wrapper)
    y receta con la forma vieja, sin la clave nueva (retrocompatibilidad, 2.4)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1},
    }
    spec = build_run(raw)

    assert not isinstance(spec.model, EpsilonScoreWrapper)
    assert isinstance(spec.model, ScoreMLP)
    assert "score_parametrization" not in spec.model_spec
    assert set(spec.model_spec) == {"name", "kwargs"}  # forma vieja de la receta, intacta


def test_build_run_rechaza_score_parametrization_desconocida():
    """Un valor desconocido de ``score_parametrization`` revienta con ``ValueError`` explícito
    que menciona el valor recibido (fail-fast antes de entrenar)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1},
        "model": {"name": "mlp", "score_parametrization": "otra"},
    }
    with pytest.raises(ValueError, match="otra"):
        build_run(raw)


def test_build_run_envuelve_con_la_sigma_de_la_sde_de_la_corrida():
    """El wrap usa la σ de la SDE de la corrida: el score del RunSpec es exactamente
    ``-inner(x, t) / clamp(σ_t, 1e-5)`` con la σ de ``spec.sde.marginal_prob`` (2.1 vía 2.4)."""
    raw = {
        "sde": {"name": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "data": {"shape": "gaussian", "dim": 2, "seed": 0},
        "train": {"num_steps": 1, "seed": 0},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1,
                  "score_parametrization": "epsilon"},
    }
    spec = build_run(raw)

    torch.manual_seed(0)
    x = torch.randn(16, 2)
    t = torch.rand(16).clamp_min(1e-3)
    spec.model.eval()
    with torch.no_grad():
        score = spec.model(x, t)
        std = spec.sde.marginal_prob(x, t)[1]
        esperado = -spec.model.inner(x, t) / std.clamp_min(1e-5)
    assert torch.allclose(score, esperado, rtol=0.0, atol=0.0)


def test_roundtrip_checkpoint_con_parametrizacion_epsilon(tmp_path):
    """Smoke del seam config→train→checkpoint con la parametrización activa (2.5, 5.1).

    ``train()`` consume el RunSpec envuelto (2 pasos, historia finita); el checkpoint guardado
    con ``model_spec`` transporta la clave en la meta, y la reconstrucción por la receta —
    ``make_model(name, **kwargs)`` + ``load_state_dict`` — funciona sobre la RED PELADA (el
    state_dict persistido no tiene prefijo de wrapper)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1,
                  "score_parametrization": "epsilon"},
    }
    spec = build_run(raw)

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert result.net is spec.model  # entrena la red envuelta que armó build_run
    assert len(result.history) == 2
    assert all(math.isfinite(v) for v in result.history)

    path = tmp_path / "ckpt_epsilon.pt"
    save_checkpoint(result, path, model_spec=spec.model_spec)

    state_dict, meta = load_checkpoint(path)
    recipe = meta["model"]
    assert recipe["score_parametrization"] == "epsilon"
    assert "score_parametrization" not in recipe["kwargs"]

    # Reconstrucción por la receta: red pelada + state_dict sin prefijo de wrapper.
    net2 = make_model(recipe["name"], **recipe["kwargs"])
    net2.load_state_dict(state_dict)
    x = torch.randn(4, 2)
    t = torch.rand(4)
    spec.model.eval()
    net2.eval()
    with torch.no_grad():
        # La red reconstruida ES la interna entrenada (misma salida en unidades de ε).
        assert torch.allclose(net2(x, t), spec.model.inner(x, t))


# ------------- small-t-training-signal: reconstrucción al generar (task 2.3)


def test_generate_from_checkpoint_seam_e2e_con_ambos_ejes(tmp_path):
    """Seam e2e config→train→checkpoint→generate con ambos ejes activos (2.3, 2.5).

    Una corrida corta con ``time_sampling: log_uniform`` (eje 1) y ``score_parametrization:
    epsilon`` (eje 2) produce un checkpoint cuya generación checkpoint-driven devuelve muestras
    finitas con la shape esperada. Además se verifica que ``generate_from_checkpoint``
    reconstruyó la red ENVUELTA con la MISMA σ: como la función no expone la red, se compara
    por equivalencia — la reconstrucción a mano (``make_model`` + ``load_state_dict`` +
    :class:`EpsilonScoreWrapper` con la σ de la SDE default) integrada con el mismo sampler y
    la misma semilla produce ``torch.equal`` (mismo prior y mismo ruido por seed). Si generate
    dejara la red pelada (sin dividir por σ), el score consumido sería otro y las trayectorias
    divergirían. El contrato de los samplers no cambia: el wrapper entra como ``score_fn``
    igual que una red pelada (2.3).
    """
    from diffusion.samplers import generate_from_checkpoint, make_sampler

    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0, "time_sampling": "log_uniform"},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1,
                  "score_parametrization": "epsilon"},
    }
    spec = build_run(raw)
    result = train(spec.sde, spec.model, spec.data, spec.config)
    path = tmp_path / "ckpt_seam_ambos_ejes.pt"
    save_checkpoint(result, path, model_spec=spec.model_spec)

    x0 = generate_from_checkpoint(path, "euler", n_samples=8, n_steps=5, seed=0)

    assert x0.shape == (8, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))

    # Reconstrucción a mano del MISMO pipeline: red pelada por la receta + wrap con la σ de la
    # SDE default reconstruida desde la meta (mismo criterio que el lado de entrenamiento).
    state_dict, meta = load_checkpoint(path)
    recipe = meta["model"]
    assert recipe["score_parametrization"] == "epsilon"  # la clave viajó en el checkpoint
    net2 = make_model(recipe["name"], **recipe["kwargs"])
    net2.load_state_dict(state_dict)
    sde = make_sde(meta["sde_name"], data_dim=meta["data_dim"])
    wrapper = EpsilonScoreWrapper(net2, lambda x, t: sde.marginal_prob(x, t)[1])
    wrapper.eval()
    sampler = make_sampler("euler", sde, wrapper, n_steps=5)
    generator = torch.Generator()
    generator.manual_seed(0)
    esperado = sampler.sample(8, generator=generator)

    # Igualdad exacta: mismo prior, mismo ruido, mismo score ⇒ misma trayectoria bit a bit.
    assert torch.equal(x0, esperado)


def test_generate_from_checkpoint_receta_vieja_reconstruye_pelada(tmp_path):
    """Una receta anterior al cambio (sin ``score_parametrization``) genera SIN wrap (2.6).

    Checkpoint con la forma vieja de la receta (``{name, kwargs}``): la generación corre sin
    error y equivale bit a bit a integrar con la red PELADA reconstruida a mano — el
    comportamiento previo queda intacto (retrocompatibilidad sin warning ni wrap).
    """
    from diffusion.samplers import generate_from_checkpoint, make_sampler

    kwargs = {"data_dim": 2, "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1}
    net = ScoreMLP(**kwargs)
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=2)
    path = tmp_path / "ckpt_receta_vieja_gen.pt"
    save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": dict(kwargs)})

    x0 = generate_from_checkpoint(path, "euler", n_samples=8, n_steps=5, seed=0)

    assert x0.shape == (8, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))

    # Equivalencia contra la reconstrucción pelada: sin la clave no hay wrap.
    state_dict, meta = load_checkpoint(path)
    assert "score_parametrization" not in meta["model"]  # receta vieja, sin la clave
    net2 = make_model(meta["model"]["name"], **meta["model"]["kwargs"])
    net2.load_state_dict(state_dict)
    net2.eval()
    sde = make_sde(meta["sde_name"], data_dim=meta["data_dim"])
    sampler = make_sampler("euler", sde, net2, n_steps=5)
    generator = torch.Generator()
    generator.manual_seed(0)
    esperado = sampler.sample(8, generator=generator)

    assert torch.equal(x0, esperado)


def test_generate_from_checkpoint_rechaza_parametrizacion_desconocida(tmp_path):
    """Un valor desconocido de ``score_parametrization`` en la receta revienta con
    ``ValueError`` que menciona el valor recibido — mismo criterio de validación que
    ``build_run`` en el lado de entrenamiento (los dos call sites validan igual)."""
    from diffusion.samplers import generate_from_checkpoint

    kwargs = {"data_dim": 2, "embed_dim": 16, "hidden_dim": 16, "num_blocks": 1}
    net = ScoreMLP(**kwargs)
    result = TrainResult(net=net, history=[1.0], sde_name="vp", data_dim=2)
    path = tmp_path / "ckpt_parametrizacion_invalida.pt"
    save_checkpoint(
        result, path,
        model_spec={"name": "mlp", "kwargs": dict(kwargs), "score_parametrization": "otra"},
    )

    with pytest.raises(ValueError, match="otra"):
        generate_from_checkpoint(path, "euler", n_samples=4, n_steps=3, seed=0)


# ------------- ema-weights: la sombra EMA como pieza aislada (task 1)

from diffusion.training import EmaShadow  # noqa: E402  (sección append-only, task 1)


class _EmaToyModule(torch.nn.Module):
    """Módulo mínimo para ejercitar la matemática de la sombra sin optimizador.

    Un único parámetro entrenable (``w``, que el test pisa a mano paso a paso para simular la
    trayectoria de Adam) y un buffer constante (``denom``, que imita el del embedding
    sinusoidal): así se puede verificar de una vez la fórmula del EMA sobre los parámetros y la
    política de buffers (copiados del módulo vivo al publicar).
    """

    def __init__(self, w0: torch.Tensor) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(w0.clone())
        self.register_buffer("denom", torch.tensor([2.0, 4.0]))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.w.sum() * x


def _replica_ema(theta0: torch.Tensor, thetas: list[torch.Tensor], decay: float) -> torch.Tensor:
    """Réplica cerrada e independiente de la sombra (misma convención de contador).

    ``ema_0 = θ_0`` (los pesos con los que se construyó la sombra) y, para cada paso completado
    ``s = 1, 2, …`` (1-indexado), ``ema_s = d_s·ema_{s−1} + (1−d_s)·θ_s`` con la rampa de warmup
    ``d_s = min(d, (1+s)/(10+s))`` — la convención fijada en `research.md` ("Ponderación y warmup").
    """
    ema = theta0.clone()
    for s, theta in enumerate(thetas, start=1):
        d = min(decay, (1.0 + s) / (10.0 + s))
        ema = d * ema + (1.0 - d) * theta
    return ema


def _corrida_sintetica(net: _EmaToyModule, shadow: EmaShadow, thetas: list[torch.Tensor]) -> None:
    """Simula la corrida: pisa el parámetro con cada ``θ_s`` y actualiza la sombra (1-indexado)."""
    for s, theta in enumerate(thetas, start=1):
        with torch.no_grad():
            net.w.copy_(theta)
        shadow.update(s)


def _thetas(n: int, seed: int, dim: int = 3) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(dim, generator=g) for _ in range(n)]


def test_ema_shadow_matematica_replica_cerrada_en_warmup():
    """La sombra iguala la réplica cerrada mientras la rampa de warmup manda (1.3).

    Con ``d = 0.999`` y 6 pasos, ``(1+s)/(10+s) ≤ 7/16 = 0.4375 < d``: **todos** los pasos usan
    el peso de la rampa, nunca el decay configurado. La comparación es contra una réplica
    aritmética independiente (no contra la propia implementación) con tolerancia ajustada de
    forma cerrada en float32.
    """
    theta0 = torch.tensor([0.5, -1.0, 2.0])
    net = _EmaToyModule(theta0)
    shadow = EmaShadow(net, decay=0.999)
    thetas = _thetas(6, seed=11)

    _corrida_sintetica(net, shadow, thetas)

    esperado = _replica_ema(theta0, thetas, 0.999)
    assert torch.allclose(shadow.state_dict()["w"], esperado, rtol=0, atol=1e-6)
    # Sentinela: la sombra promedia de verdad (ni la última foto ni la inicial).
    assert not torch.allclose(esperado, thetas[-1], atol=1e-3)
    assert not torch.allclose(esperado, theta0, atol=1e-3)


def test_ema_shadow_matematica_replica_cerrada_cruza_el_warmup():
    """La sombra iguala la réplica cerrada cruzando warmup → régimen estacionario (1.3).

    Con ``d = 0.5`` la rampa alcanza el decay configurado en ``(1+s)/(10+s) = 0.5 ⇒ s = 8``: en
    12 pasos los primeros 7 usan la rampa (peso < 0.5) y del 8º en adelante manda ``d`` fijo. El
    decay chico es deliberado (distingue los dos regímenes en pocos pasos). Además se verifica
    que la rampa **está aplicada**: una réplica con ``d`` constante (sin warmup) da otro valor.
    """
    theta0 = torch.tensor([1.0, 0.0, -0.5])
    net = _EmaToyModule(theta0)
    shadow = EmaShadow(net, decay=0.5)
    thetas = _thetas(12, seed=5)

    # Los dos regímenes quedan efectivamente ejercitados por esta secuencia.
    assert (1.0 + 7) / (10.0 + 7) < 0.5  # paso 7: manda la rampa
    assert (1.0 + 8) / (10.0 + 8) == 0.5  # paso 8: la rampa toca el decay configurado

    _corrida_sintetica(net, shadow, thetas)

    esperado = _replica_ema(theta0, thetas, 0.5)
    assert torch.allclose(shadow.state_dict()["w"], esperado, rtol=0, atol=1e-6)

    sin_warmup = theta0.clone()
    for theta in thetas:
        sin_warmup = 0.5 * sin_warmup + 0.5 * theta
    assert not torch.allclose(esperado, sin_warmup, atol=1e-4)


@pytest.mark.parametrize(
    "bad", [0.0, 1.0, -0.5, 1.5, float("inf"), float("-inf"), float("nan")]
)
def test_ema_shadow_rechaza_decay_invalido(bad):
    """Decay no finito o fuera del intervalo abierto (0, 1) ⇒ ``ValueError`` con el valor (1.6).

    Fail-fast en construcción, antes de tocar la trayectoria de entrenamiento; el mensaje debe
    incluir el valor recibido (mismo criterio que la validación de ``t_eps`` del time sampler).
    """
    net = _EmaToyModule(torch.zeros(3))
    with pytest.raises(ValueError) as exc:
        EmaShadow(net, decay=bad)
    assert str(bad) in str(exc.value)


@pytest.mark.parametrize("ok", [1e-8, 0.5, 0.999, 1 - 1e-8])
def test_ema_shadow_acepta_decay_valido(ok):
    """Cualquier decay finito en el intervalo abierto (0, 1) se acepta (1.6, borde permitido)."""
    net = _EmaToyModule(torch.zeros(3))
    assert EmaShadow(net, decay=ok).decay == pytest.approx(ok)


@pytest.mark.parametrize("bad_step", [0, -1, -10])
def test_ema_shadow_rechaza_step_invalido(bad_step):
    """``update`` exige el contador 1-indexado de pasos completados (precondición ``step >= 1``)."""
    net = _EmaToyModule(torch.zeros(3))
    shadow = EmaShadow(net, decay=0.9)
    with pytest.raises(ValueError) as exc:
        shadow.update(bad_step)
    assert str(bad_step) in str(exc.value)


def _modulos_para_agnosticismo() -> dict[str, torch.nn.Module]:
    """Tres módulos de forma distinta para probar la agnosticismo de la sombra (1.5)."""
    sde = make_sde("vp")
    return {
        "mlp": ScoreMLP(data_dim=2, embed_dim=16, hidden_dim=32, num_blocks=1),
        "dummy": _DummyScoreNet(),
        "wrapper": EpsilonScoreWrapper(
            ScoreMLP(data_dim=2, embed_dim=16, hidden_dim=32, num_blocks=1),
            lambda x, t: sde.marginal_prob(x, t)[1],
        ),
    }


@pytest.mark.parametrize("clave", ["mlp", "dummy", "wrapper"])
def test_ema_shadow_claves_iguales_a_las_del_modulo(clave):
    """Las claves de ``state_dict()`` == las del ``state_dict`` del módulo, sea cual sea la red (1.5).

    Invariante de compatibilidad: lo que publica la sombra se carga en la red reconstruida con
    ``load_state_dict(strict=True)``. Se cubren tres formas de módulo sin ramificar por tipo:
    el ``ScoreMLP`` (parámetros + buffer del embedding), una red mínima de un solo parámetro y
    el :class:`EpsilonScoreWrapper` — cuyo ``state_dict`` delega al interno, así que las claves
    de la sombra son **de red pelada** (sin prefijo ``_net.``), lo que hace componer EMA con la
    parametrización ε sin traducción de claves.
    """
    net = _modulos_para_agnosticismo()[clave]
    shadow = EmaShadow(net, decay=0.9)
    shadow.update(1)

    foto = shadow.state_dict()
    assert set(foto) == set(net.state_dict())
    if clave == "wrapper":
        assert not any(k.startswith("_net.") for k in foto)  # claves de red pelada
    net.load_state_dict(foto)  # strict=True: encaja sin faltantes ni sobrantes


def test_ema_shadow_publica_parametros_ema_y_buffers_del_modulo_vivo():
    """``state_dict()`` = parámetros EMA + buffers copiados del módulo vivo (decisión de research)."""
    theta0 = torch.tensor([0.5, -1.0, 2.0])
    net = _EmaToyModule(theta0)
    shadow = EmaShadow(net, decay=0.9)
    _corrida_sintetica(net, shadow, _thetas(4, seed=3))

    with torch.no_grad():  # el buffer del vivo cambia ⇒ la foto lo refleja tal cual
        net.denom.copy_(torch.tensor([7.0, 9.0]))

    foto = shadow.state_dict()
    assert torch.equal(foto["denom"], net.denom)  # buffer: copia del vivo, no promediado
    assert not torch.equal(foto["w"], net.w.detach())  # parámetro: EMA, ≠ pesos crudos


def test_ema_shadow_state_dict_devuelve_clones():
    """La foto es un clon estable: ni la sombra ni el llamador se pisan entre sí.

    Postcondición del diseño (fotos estables para snapshots/checkpoints): actualizar la sombra
    después de tomar la foto no cambia la foto, y mutar la foto no cambia la sombra.
    """
    net = _EmaToyModule(torch.zeros(3))
    shadow = EmaShadow(net, decay=0.9)
    _corrida_sintetica(net, shadow, _thetas(3, seed=1))

    foto = shadow.state_dict()
    congelada = {k: v.clone() for k, v in foto.items()}

    _corrida_sintetica(net, shadow, _thetas(3, seed=2))  # la sombra sigue moviéndose
    assert all(torch.equal(foto[k], congelada[k]) for k in foto)
    assert not torch.equal(shadow.state_dict()["w"], foto["w"])

    antes = shadow.state_dict()["w"].clone()
    foto["w"].add_(100.0)  # mutar la foto no toca la sombra
    assert torch.equal(shadow.state_dict()["w"], antes)


def test_ema_shadow_es_pasiva_y_no_consume_rng():
    """La sombra observa: no escribe en el módulo ni consume RNG (1.4, 1.5).

    Invariante de pasividad a nivel de la pieza: tras construir, actualizar y publicar, los
    parámetros y buffers del módulo quedan **exactamente** como los dejó el "optimizador", y el
    estado del RNG global de torch no se movió (la actualización es aritmética determinística).
    """
    net = _EmaToyModule(torch.tensor([0.5, -1.0, 2.0]))
    thetas = _thetas(4, seed=8)

    torch.manual_seed(123)
    rng_antes = torch.get_rng_state()
    shadow = EmaShadow(net, decay=0.9)
    _corrida_sintetica(net, shadow, thetas)
    shadow.state_dict()

    assert torch.equal(net.w.detach(), thetas[-1])  # el módulo quedó donde lo dejó el paso
    assert torch.equal(net.denom, torch.tensor([2.0, 4.0]))
    assert torch.equal(torch.get_rng_state(), rng_antes)  # sin RNG


def test_ema_shadow_determinista_ante_la_misma_secuencia():
    """Misma secuencia de pesos ⇒ misma sombra, bit a bit (1.5: determinismo por seed)."""
    theta0 = torch.tensor([0.25, 0.5, -0.75])
    thetas = _thetas(7, seed=4)

    fotos = []
    for _ in range(2):
        net = _EmaToyModule(theta0)
        shadow = EmaShadow(net, decay=0.99)
        _corrida_sintetica(net, shadow, thetas)
        fotos.append(shadow.state_dict()["w"])
    assert torch.equal(fotos[0], fotos[1])


def test_ema_shadow_load_state_reanuda_la_sombra():
    """``load_state`` restaura la sombra: interrumpir y reanudar == corrida seguida (1.3, R3).

    Sombra A corre 9 pasos. Sombra B arranca de cero, corre 4, se le carga la foto de A al paso
    4 (como haría el resume desde el sidecar) y sigue los 5 pasos restantes con el mismo
    contador global: el resultado es idéntico bit a bit al de A.
    """
    theta0 = torch.tensor([1.5, -0.5, 0.25])
    thetas = _thetas(9, seed=2)

    net_a = _EmaToyModule(theta0)
    shadow_a = EmaShadow(net_a, decay=0.95)
    fotos_a = []
    for s, theta in enumerate(thetas, start=1):
        with torch.no_grad():
            net_a.w.copy_(theta)
        shadow_a.update(s)
        fotos_a.append(shadow_a.state_dict())

    net_b = _EmaToyModule(theta0)
    shadow_b = EmaShadow(net_b, decay=0.95)
    for s, theta in enumerate(thetas[:4], start=1):
        with torch.no_grad():
            net_b.w.copy_(theta)
        shadow_b.update(s)
    shadow_b.load_state(fotos_a[3])  # restauración desde la foto del paso 4
    for s, theta in enumerate(thetas[4:], start=5):
        with torch.no_grad():
            net_b.w.copy_(theta)
        shadow_b.update(s)

    assert torch.equal(shadow_b.state_dict()["w"], shadow_a.state_dict()["w"])


def test_ema_shadow_load_state_rechaza_foto_incompleta():
    """Una foto sin las claves de la sombra se rechaza con ``ValueError`` que las nombra.

    Nunca continuar con una sombra a medias en silencio (mismo criterio fail-fast del resto del
    módulo).
    """
    net = _EmaToyModule(torch.zeros(3))
    shadow = EmaShadow(net, decay=0.9)
    with pytest.raises(ValueError) as exc:
        shadow.load_state({"denom": torch.tensor([2.0, 4.0])})
    assert "w" in str(exc.value)


# ------------- ema-weights: orquestación de la sombra en el loop (task 2.1)


class _EmaModuloDeStateDictOpaco(torch.nn.Module):
    """Módulo cuyo ``state_dict`` **no reenvía** ``keep_vars`` al de ``nn.Module``.

    Caso patológico realista (un wrapper que reescribe ``state_dict`` a mano): los tensores que
    ve la sombra son copias detachadas, así que la identificación por identidad no encuentra
    ningún parámetro entrenable y la sombra quedaría **vacía** — promediando nada en silencio.
    Sirve para verificar el guard fail-fast de :class:`EmaShadow`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(3))

    def state_dict(self, *args, **kwargs):  # noqa: D102
        kwargs.pop("keep_vars", None)  # el bug que el guard tiene que detectar
        return super().state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.w.sum() * x


class _DataProhibidaEma:
    """Fuente centinela: cualquier consumo de datos es un fallo del orden fail-fast del loop."""

    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("el loop consumió data antes de validar ema_decay")


def _fresh_ema(**overrides):
    """Corrida chica reproducible: mismos pesos iniciales y misma fuente en cada invocación."""
    torch.manual_seed(321)  # pesos iniciales idénticos entre corridas comparadas
    sde = make_sde("vp")
    net = _small_net(sde)
    dist = make_distribution("gaussian", 2, seed=0)
    data = _data(dist, n=128, batch_size=32, shuffle=False)
    base = dict(num_steps=4, seed=0)
    base.update(overrides)
    return sde, net, data, TrainConfig(**base)


def _generador(cfg: TrainConfig) -> torch.Generator:
    """El ``generator`` del loop, explícito, para poder comparar el azar consumido."""
    g = torch.Generator(device=torch.device(cfg.device))
    g.manual_seed(cfg.seed)
    return g


def _replica_loop_crudo(sde, net, data, cfg, generator):
    """Réplica manual del loop **sin sombra**, bit a bit equivalente al camino default.

    Mismo orden de operaciones y mismo consumo del RNG que :func:`train` con
    ``time_sampling="uniform"`` (el invariante que ya sostiene el test del default de
    ``time_sampling``). El caller siembra el azar global antes de llamar.

    Returns:
        ``(history, thetas)``: la pérdida por paso y, en ``thetas[s-1]``, el clon del
        ``state_dict`` tras el paso completado ``s`` (1-indexado) — la trayectoria cruda de Adam
        que la sombra tiene que promediar.
    """
    device = torch.device(cfg.device)
    net = net.to(device)
    net.train()
    data_iter = iter(data)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    history: list[float] = []
    thetas: list[dict] = []
    for _ in range(cfg.num_steps):
        x0 = next(data_iter).to(device)
        t = sample_timesteps(x0.shape[0], sde.T, cfg.t_eps, generator=generator, device=device)
        loss = dsm_loss(net, sde, x0, t, generator=generator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        thetas.append({k: v.detach().clone() for k, v in net.state_dict().items()})
    return history, thetas


def test_trainconfig_ema_decay_default_sin_ema():
    """``TrainConfig()`` sin el campo nuevo declara ``ema_decay=None`` — sin EMA (1.1).

    El default retrocompatible: quien no configura nada entrena exactamente como hasta hoy.
    """
    cfg = TrainConfig()
    assert cfg.ema_decay is None
    # El campo es configurable como cualquier otro del dataclass.
    assert TrainConfig(ema_decay=0.999).ema_decay == pytest.approx(0.999)


def test_train_default_sin_ema_bit_identico_al_loop_previo():
    """Con el default (``ema_decay=None``) ``train()`` es BIT-IDÉNTICO al loop previo (1.2).

    Mismo patrón de prueba que el default de ``time_sampling``: una réplica manual del loop (sin
    sombra, mismo orden de operaciones) contra ``train()`` con la config default. Se comparan las
    tres cosas que el criterio 1.2 nombra —historia (igualdad EXACTA de floats), pesos finales
    (``torch.equal``) y el azar consumido (RNG global de torch + estado del ``generator``)—: si la
    rama nueva tocara el stream o el orden, alguna fallaría. Además ``ema_state`` queda en
    ``None``: sin configurar nada, el resultado no gana contenido.
    """
    sde, net_a, data, cfg = _fresh_ema()
    g_a = _generador(cfg)
    res_a = train(sde, net_a, data, cfg, generator=g_a)
    rng_a = torch.get_rng_state()

    sde, net_b, data, cfg = _fresh_ema()
    g_b = _generador(cfg)
    torch.manual_seed(cfg.seed)  # lo que hace train() al arrancar desde cero
    hist_b, _ = _replica_loop_crudo(sde, net_b, data, cfg, g_b)
    rng_b = torch.get_rng_state()

    assert res_a.history == hist_b  # igualdad exacta de floats por paso, no aproximada
    assert res_a.ema_state is None
    sd_a, sd_b = net_a.state_dict(), net_b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"peso final distinto en {key}"
    assert torch.equal(rng_a, rng_b)  # misma secuencia del RNG global
    assert torch.equal(g_a.get_state(), g_b.get_state())  # mismo stream del generator


def test_train_con_ema_promedia_la_trayectoria_cruda_de_adam():
    """Con EMA activo, ``result.ema_state`` == réplica cerrada sobre la trayectoria cruda (1.3).

    Verifica la **orquestación** (no la aritmética de la pieza, ya cubierta): el loop actualiza
    la sombra una vez por paso, después del ``optimizer.step()``, con el contador 1-indexado de
    pasos completados. La réplica manual del loop registra los ``θ_s`` de cada paso y
    :func:`_replica_ema` los promedia desde ``θ_0`` con ``d_s = min(d, (1+s)/(10+s))``. Un
    ``update`` de más/de menos, o un contador corrido, rompe la igualdad. Los buffers no se
    promedian (se copian del módulo vivo), así que se contrastan aparte.
    """
    decay = 0.6  # chico a propósito: la rampa y el techo se distinguen en pocos pasos
    sde, net_a, data, cfg = _fresh_ema(ema_decay=decay)
    res_a = train(sde, net_a, data, cfg, generator=_generador(cfg))

    sde, net_b, data, cfg = _fresh_ema(ema_decay=decay)
    theta0 = {k: v.detach().clone() for k, v in net_b.state_dict().items()}
    entrenables = set(dict(net_b.named_parameters()))
    g_b = _generador(cfg)
    torch.manual_seed(cfg.seed)
    _, thetas = _replica_loop_crudo(sde, net_b, data, cfg, g_b)

    assert res_a.ema_state is not None
    assert set(res_a.ema_state) == set(net_a.state_dict())  # foto publicable tal cual
    for key in entrenables:
        esperado = _replica_ema(theta0[key], [th[key] for th in thetas], decay)
        assert torch.allclose(res_a.ema_state[key], esperado, rtol=0, atol=1e-6), key
    for key in set(theta0) - entrenables:  # buffers: copia del vivo, no promedio
        assert torch.equal(res_a.ema_state[key], net_a.state_dict()[key]), key

    # Sentinela del contador: un ``update`` de más (o el contador corrido) daría otro valor, así
    # que la igualdad de arriba no es trivialmente satisfacible.
    clave = next(iter(entrenables))
    de_mas = _replica_ema(
        theta0[clave], [th[clave] for th in thetas] + [thetas[-1][clave]], decay
    )
    assert not torch.allclose(res_a.ema_state[clave], de_mas, rtol=0, atol=1e-6)


def test_train_con_ema_no_altera_la_trayectoria_de_optimizacion():
    """Pasividad: EMA on/off con la misma semilla ⇒ historia y pesos CRUDOS idénticos (1.4).

    La sombra es un observador: no escribe en la red, no toca el optimizador y no consume RNG.
    Dos corridas con la misma semilla —una con EMA y otra sin— deben terminar con la misma
    historia, los mismos pesos crudos (``torch.equal``) y el ``generator`` en el mismo estado. El
    cierre del test verifica que el EMA igual hizo algo: la foto difiere de los pesos crudos.
    """

    def _run(ema_decay):
        sde, net, data, cfg = _fresh_ema(ema_decay=ema_decay)
        g = _generador(cfg)
        res = train(sde, net, data, cfg, generator=g)
        crudos = {k: v.detach().clone() for k, v in net.state_dict().items()}
        return res, crudos, g.get_state()

    res_off, crudos_off, g_off = _run(None)
    res_on, crudos_on, g_on = _run(0.9)

    assert res_on.history == res_off.history  # igualdad exacta, no aproximada
    assert crudos_on.keys() == crudos_off.keys()
    for key in crudos_off:
        assert torch.equal(crudos_on[key], crudos_off[key]), f"peso crudo distinto en {key}"
    assert torch.equal(g_on, g_off)  # la sombra no consume RNG

    assert res_off.ema_state is None
    assert res_on.ema_state is not None
    assert any(not torch.equal(res_on.ema_state[k], crudos_on[k]) for k in crudos_on)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5, float("inf"), float("nan")])
def test_train_rechaza_ema_decay_invalido_antes_de_consumir_data(bad):
    """Un ``ema_decay`` inválido revienta con ``ValueError`` ANTES de tocar los datos (1.6).

    El loop construye la sombra una única vez, fail-fast, con el mismo criterio que el time
    sampler: la fuente centinela convierte cualquier consumo prematuro en un ``AssertionError``
    distinto del ``ValueError`` esperado, y el mensaje debe nombrar el valor recibido.
    """
    sde = make_sde("vp")
    net = _small_net(sde)
    cfg = TrainConfig(num_steps=2, seed=0, ema_decay=bad)
    with pytest.raises(ValueError) as exc:
        train(sde, net, _DataProhibidaEma(), cfg)
    assert str(bad) in str(exc.value)


def test_train_con_ema_snapshots_llevan_fotos_clonadas_e_independientes():
    """La foto clonada viaja en el resultado y en cada snapshot intermedio (observable de 2.1).

    ``_snapshot`` fotografía vía ``_result``, así que los checkpoints intermedios llevan la
    sombra **de su momento**: dos snapshots sucesivos difieren y son objetos independientes
    (mutar uno no toca al otro ni al resultado final). Es lo que hace segura la publicación.
    """
    sde, net, data, cfg = _fresh_ema(num_steps=6, checkpoint_every=2, ema_decay=0.9)
    fotos: list[tuple[str, dict]] = []

    def _cb(tag, snapshot):
        if tag.startswith("step"):
            fotos.append((tag, snapshot.result.ema_state))

    res = train(sde, net, data, cfg, on_checkpoint=_cb, generator=_generador(cfg))

    assert [tag for tag, _ in fotos] == ["step00002", "step00004"]
    (_, f2), (_, f4) = fotos
    assert all(estado is not None for _, estado in fotos)
    # La sombra avanzó entre snapshots y sigue avanzando hasta el final de la corrida.
    assert any(not torch.equal(f2[k], f4[k]) for k in f2)
    assert any(not torch.equal(f4[k], res.ema_state[k]) for k in f4)

    # Independencia: mutar la foto del paso 2 no altera la del 4 ni la final (clones profundos).
    congelada_f4 = {k: v.clone() for k, v in f4.items()}
    congelada_fin = {k: v.clone() for k, v in res.ema_state.items()}
    for v in f2.values():
        v.add_(100.0)
    assert all(torch.equal(f4[k], congelada_f4[k]) for k in f4)
    assert all(torch.equal(res.ema_state[k], congelada_fin[k]) for k in res.ema_state)


def test_build_run_acepta_ema_decay_y_sigue_estricto():
    """El bloque ``train:`` acepta ``ema_decay`` y la validación estricta sigue viva (4.1).

    ``TrainConfig`` se valida por **introspección de sus fields**, así que el campo nuevo queda
    disponible desde la config sin tocar ``build_run``; una corrida config-driven corta entrena
    con él (observable de la tarea) y una clave inválida se sigue rechazando nombrándola. Se
    ejercita con un ``dict`` —el mismo que produce ``load_config``— para no depender de PyYAML
    (el camino de archivo va en el test hermano).
    """
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 2, "seed": 0, "ema_decay": 0.999},
    }
    spec = build_run(raw)
    assert spec.config.ema_decay == pytest.approx(0.999)

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert len(result.history) == 2
    assert all(math.isfinite(v) for v in result.history)
    assert result.ema_state is not None
    assert set(result.ema_state) == set(spec.model.state_dict())

    bad = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"num_steps": 1, "ema_decay_typo": 0.999},
    }
    with pytest.raises(ValueError, match="ema_decay_typo"):
        build_run(bad)


def test_build_run_desde_yaml_con_ema_decay(tmp_path):
    """El mismo camino pero desde un archivo YAML real: ``train.ema_decay`` llega al loop (4.1).

    Complementa al test hermano cubriendo el front-end de archivo (``load_config``); se skipea si
    PyYAML no está instalado, como el resto de los tests del camino YAML de la suite.
    """
    pytest.importorskip("yaml")
    text = (
        "sde:\n"
        "  name: vp\n"
        "data:\n"
        "  shape: gaussian\n"
        "  dim: 2\n"
        "  n_samples: 128\n"
        "  batch_size: 64\n"
        "  seed: 0\n"
        "train:\n"
        "  num_steps: 2\n"
        "  seed: 0\n"
        "  ema_decay: 0.999\n"
    )
    path = tmp_path / "run_ema.yaml"
    path.write_text(text, encoding="utf-8")

    spec = build_run(load_config(path))
    assert spec.config.ema_decay == pytest.approx(0.999)

    result = train(spec.sde, spec.model, spec.data, spec.config)
    assert len(result.history) == 2
    assert result.ema_state is not None


def test_ema_shadow_rechaza_modulo_sin_tensores_rastreables():
    """Un módulo cuyo ``state_dict`` no reenvía ``keep_vars`` se rechaza en construcción.

    Guard fail-fast: sin él la sombra quedaría vacía y **promediaría nada en silencio**,
    publicando pesos crudos disfrazados de EMA. El mensaje nombra el problema (``keep_vars``) en
    lugar de dejar pasar una corrida con EMA inerte.
    """
    with pytest.raises(ValueError) as exc:
        EmaShadow(_EmaModuloDeStateDictOpaco(), decay=0.9)
    assert "keep_vars" in str(exc.value)


# ------------- ema-weights: publicación en checkpoints + hermano de crudos (task 2.2)

from diffusion.training import discover_snapshots  # noqa: E402  (sección append-only, task 2.2)

#: Receta de la red chica de la suite (``_small_net``): lo que el caller pasa como ``model_spec``.
_RECETA_CHICA = {"name": "mlp", "kwargs": {"data_dim": 2, "hidden_dim": 64, "num_blocks": 2}}


def _corrida_publicable(**overrides):
    """Corrida corta reproducible + su receta, lista para publicar en un checkpoint.

    Reusa los helpers de la sección anterior (``_fresh_ema``/``_generador``: misma red chica,
    mismos pesos iniciales y mismo azar en cada invocación).

    Returns:
        ``(result, model_spec, crudos)``: el :class:`TrainResult`, la receta genérica de la red y
        un clon del ``state_dict`` **crudo** final (para contrastarlo contra lo publicado).
    """
    sde, net, data, cfg = _fresh_ema(**overrides)
    result = train(sde, net, data, cfg, generator=_generador(cfg))
    crudos = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return result, {"name": "mlp", "kwargs": dict(_RECETA_CHICA["kwargs"])}, crudos


def test_save_checkpoint_con_ema_publica_la_sombra_y_marca_el_decay(tmp_path):
    """Con EMA activo el checkpoint publica la sombra como su ``state_dict`` (2.1) + marca (2.3).

    Un solo punto de publicación: ``save_checkpoint`` toma ``result.ema_state`` en lugar de los
    pesos vivos. Se verifica lo tres cosas que pide el criterio: (a) el ``model_state`` es la foto
    EMA **tensor a tensor** (``torch.equal``) y difiere de los crudos —si publicara los crudos, la
    corrida con EMA no aportaría nada—; (b) el **formato no cambia** (mismas claves de blob y de
    meta que hoy, más la marca) y la receta sigue reconstruyendo la red; (c) la meta registra la
    trazabilidad ``ema.decay``.
    """
    decay = 0.6
    result, model_spec, crudos = _corrida_publicable(num_steps=6, ema_decay=decay)

    path = tmp_path / "vp_pub.pt"
    save_checkpoint(result, path, model_spec=model_spec)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert set(blob) == {"model_state", "meta"}  # formato intacto (2.1)

    state_dict, meta = load_checkpoint(path)
    assert state_dict.keys() == crudos.keys()
    for key in result.ema_state:
        assert torch.equal(state_dict[key], result.ema_state[key]), f"no publicó la sombra en {key}"
    assert any(not torch.equal(state_dict[k], crudos[k]) for k in crudos)  # ≠ pesos crudos

    # Marca de trazabilidad (2.3): quién publicó y con qué decay.
    assert meta["ema"] == {"decay": pytest.approx(decay)}
    # El resto de la meta es la de siempre (2.1: receta incluida, sin campos nuevos de más).
    assert set(meta) == {"sde_name", "data_dim", "history", "model", "ema"}
    assert meta["sde_name"] == "vp"
    assert meta["data_dim"] == 2
    assert meta["history"] == pytest.approx(result.history)
    assert meta["model"] == model_spec

    # La receta sigue reconstruyendo la red y los pesos EMA cargan con strict=True.
    net2 = make_model(meta["model"]["name"], **meta["model"]["kwargs"])
    net2.load_state_dict(state_dict)


def test_save_checkpoint_sin_ema_contenido_identico_al_actual(tmp_path):
    """Sin EMA el checkpoint es **idéntico al actual** en contenido, sin la marca nueva (2.5).

    Retrocompatibilidad: ``ema_decay=None`` ⇒ ``model_state`` == los pesos vivos ``torch.equal``
    tensor a tensor y meta con exactamente las claves de antes (la ausencia de ``ema`` es lo que
    un consumidor lee como "pesos crudos", 2.3).
    """
    result, model_spec, crudos = _corrida_publicable(num_steps=6)  # ema_decay=None (default)
    assert result.ema_state is None

    path = tmp_path / "vp_sin_ema.pt"
    save_checkpoint(result, path, model_spec=model_spec)

    state_dict, meta = load_checkpoint(path)
    assert state_dict.keys() == crudos.keys()
    for key in crudos:
        assert torch.equal(state_dict[key], crudos[key]), f"peso publicado distinto en {key}"
    assert "ema" not in meta
    assert set(meta) == {"sde_name", "data_dim", "history", "model"}


def test_checkpoints_intermedios_publican_la_sombra_de_su_momento(tmp_path):
    """Los checkpoints intermedios periódicos publican también la sombra de su momento (2.2).

    Reproduce el wiring del CLI (``on_checkpoint`` → ``save_checkpoint`` sobre rutas hermanas):
    como los intermedios pasan por el **mismo** punto de publicación, ganan el EMA sin cambio
    propio. Dos snapshots sucesivos **difieren** (la foto es un clonado profundo del momento, no
    una referencia viva que terminaría igual al final de la corrida) y ambos llevan la marca.
    """
    decay = 0.6
    sde, net, data, cfg = _fresh_ema(num_steps=6, checkpoint_every=2, ema_decay=decay)
    base = tmp_path / "vp_inter.pt"
    fotos: dict[str, dict] = {}

    def on_checkpoint(tag, snapshot):
        save_checkpoint(
            snapshot.result, base.with_stem(f"{base.stem}_{tag}"), model_spec=dict(_RECETA_CHICA)
        )
        if tag.startswith("step"):
            fotos[tag] = {k: v.clone() for k, v in snapshot.result.ema_state.items()}

    train(sde, net, data, cfg, on_checkpoint=on_checkpoint, generator=_generador(cfg))

    assert sorted(fotos) == ["step00002", "step00004"]
    publicados = {}
    for tag in ("step00002", "step00004"):
        state_dict, meta = load_checkpoint(tmp_path / f"vp_inter_{tag}.pt")
        assert meta["ema"] == {"decay": pytest.approx(decay)}  # marca en los intermedios (2.3)
        for key in fotos[tag]:
            assert torch.equal(state_dict[key], fotos[tag][key]), f"{tag}: no publicó su sombra"
        publicados[tag] = state_dict

    # La sombra avanzó entre snapshots: los dos artefactos no son el mismo estado.
    assert any(
        not torch.equal(publicados["step00002"][k], publicados["step00004"][k])
        for k in publicados["step00002"]
    )


def test_save_checkpoint_hermano_de_crudos_publica_los_pesos_crudos(tmp_path):
    """``raw_sibling=True`` escribe además ``{stem}_raw.pt`` con los pesos CRUDOS finales (2.7).

    El hermano es un checkpoint **estándar** (mismo formato, receta incluida) para habilitar la
    comparativa crudo-vs-EMA de la misma corrida: su ``state_dict`` son los crudos
    (``torch.equal``) y difiere del EMA del principal; su meta **no** lleva la marca ``ema`` (la
    ausencia = pesos crudos, 2.3) y sí un puntero al checkpoint del que es contraparte. Se verifica
    que todo el tooling existente lo consume sin cambios (``load_checkpoint`` + ``make_model`` +
    ``generate_from_checkpoint``) y que el descubrimiento de snapshots **no** lo levanta.
    """
    from diffusion.samplers import generate_from_checkpoint

    result, model_spec, crudos = _corrida_publicable(num_steps=6, ema_decay=0.6)

    path = tmp_path / "vp_final.pt"
    save_checkpoint(result, path, model_spec=model_spec, raw_sibling=True)

    raw_path = tmp_path / "vp_final_raw.pt"
    assert raw_path.exists()

    raw_state, raw_meta = load_checkpoint(raw_path)
    ema_state, ema_meta = load_checkpoint(path)
    for key in crudos:
        assert torch.equal(raw_state[key], crudos[key]), f"el hermano no lleva los crudos en {key}"
    assert any(not torch.equal(raw_state[k], ema_state[k]) for k in crudos)  # crudos ≠ EMA

    # Marcado como contraparte cruda y sin la marca de EMA (no se puede confundir con el oficial).
    assert "ema" not in raw_meta
    assert raw_meta["raw_of"] == path.name
    assert ema_meta["ema"] == {"decay": pytest.approx(0.6)}
    # Formato estándar con receta: el resto de la meta es la del principal.
    assert set(raw_meta) == {"sde_name", "data_dim", "history", "model", "raw_of"}
    assert raw_meta["model"] == model_spec
    assert raw_meta["history"] == pytest.approx(result.history)

    # Reconstruible por la receta y consumible por la generación checkpoint-driven.
    net2 = make_model(raw_meta["model"]["name"], **raw_meta["model"]["kwargs"])
    net2.load_state_dict(raw_state)
    x_raw = generate_from_checkpoint(raw_path, "euler", n_samples=8, n_steps=5, seed=0)
    x_ema = generate_from_checkpoint(path, "euler", n_samples=8, n_steps=5, seed=0)
    for x in (x_raw, x_ema):
        assert x.shape == (8, 2)
        assert x.dtype == torch.float32
        assert torch.all(torch.isfinite(x))
    # Pesos distintos ⇒ muestras distintas con la misma semilla (la comparativa tiene sentido).
    assert not torch.equal(x_raw, x_ema)

    # El hermano no es un snapshot periódico: el descubrimiento no lo levanta (2.7).
    assert discover_snapshots(path) == []


def test_hermano_de_crudos_no_se_escribe_sin_ema(tmp_path):
    """Sin EMA activo, ``raw_sibling=True`` **no** escribe el hermano (sería un duplicado).

    El hermano existe para separar crudos de EMA; sin EMA el checkpoint principal YA publica los
    crudos, así que emitirlo sería escribir dos veces lo mismo. El parámetro queda por eso seguro
    de activar incondicionalmente en el guardado final del CLI (2.5: sin EMA, nada cambia).
    """
    result, model_spec, crudos = _corrida_publicable(num_steps=4)  # sin EMA
    path = tmp_path / "vp_sin_ema_final.pt"
    save_checkpoint(result, path, model_spec=model_spec, raw_sibling=True)

    assert not (tmp_path / "vp_sin_ema_final_raw.pt").exists()
    state_dict, meta = load_checkpoint(path)
    assert "ema" not in meta and "raw_of" not in meta
    for key in crudos:
        assert torch.equal(state_dict[key], crudos[key])


def test_discover_snapshots_ignora_el_hermano_de_crudos(tmp_path):
    """Mecánico: ``{stem}_raw.pt`` no matchea el patrón ``_stepNNNNN`` del descubrimiento (2.7).

    Se tocan los tres artefactos posibles de una corrida (final, hermano de crudos y un snapshot
    periódico con su sidecar) sin escribir contenido: la política de descubrimiento mira solo
    nombres. Solo el periódico debe salir en la lista — el hermano nunca puede elegirse como punto
    de reanudación.
    """
    base = tmp_path / "vp_mixture.pt"
    for nombre in (
        "vp_mixture.pt",
        "vp_mixture_raw.pt",
        "vp_mixture_step00002.pt",
        "vp_mixture_step00002.resume.pt",
    ):
        (tmp_path / nombre).touch()

    assert discover_snapshots(base) == [(2, tmp_path / "vp_mixture_step00002.pt")]


def test_publicacion_ema_compone_con_la_parametrizacion_epsilon(tmp_path):
    """El EMA publicado compone con la parametrización ε de la receta (2.4).

    Corrida corta config-driven con la red **envuelta** (``EpsilonScoreWrapper``): su
    ``state_dict`` delega al interno, así que la sombra —y por lo tanto el checkpoint publicado—
    queda en claves de **red pelada**. Consecuencia verificada acá: el checkpoint EMA se
    reconstruye por ``make_model`` + wrap por receta y ``generate_from_checkpoint`` produce
    muestras finitas sin ningún cambio en su contrato; el hermano de crudos también.
    """
    from diffusion.samplers import generate_from_checkpoint

    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 128, "batch_size": 64, "seed": 0},
        "train": {"num_steps": 4, "seed": 0, "ema_decay": 0.6},
        "model": {"name": "mlp", "hidden_dim": 32, "num_blocks": 1,
                  "score_parametrization": "epsilon"},
    }
    spec = build_run(raw)
    assert isinstance(spec.model, EpsilonScoreWrapper)
    result = train(spec.sde, spec.model, spec.data, spec.config)

    # Claves de red pelada (el wrapper delega): sin prefijo ``_net.``, como las de make_model.
    assert result.ema_state is not None
    assert all(not k.startswith("_net.") for k in result.ema_state)
    assert set(result.ema_state) == set(make_model("mlp", data_dim=2, hidden_dim=32,
                                                  num_blocks=1).state_dict())

    path = tmp_path / "ckpt_epsilon_ema.pt"
    save_checkpoint(result, path, model_spec=spec.model_spec, raw_sibling=True)
    raw_path = tmp_path / "ckpt_epsilon_ema_raw.pt"

    state_dict, meta = load_checkpoint(path)
    assert meta["model"]["score_parametrization"] == "epsilon"  # la receta viaja igual que antes
    assert meta["ema"] == {"decay": pytest.approx(0.6)}
    for key in result.ema_state:
        assert torch.equal(state_dict[key], result.ema_state[key])

    # Reconstrucción por la receta: red pelada + wrap con la σ de la SDE (lo que hace generate).
    sde = make_sde(meta["sde_name"], data_dim=meta["data_dim"])
    net2 = make_model(meta["model"]["name"], **meta["model"]["kwargs"])
    net2.load_state_dict(state_dict)
    wrapper = EpsilonScoreWrapper(net2, lambda x, t: sde.marginal_prob(x, t)[1])
    wrapper.eval()

    x_ema = generate_from_checkpoint(path, "euler", n_samples=8, n_steps=5, seed=0)
    x_raw = generate_from_checkpoint(raw_path, "euler", n_samples=8, n_steps=5, seed=0)
    for x in (x_ema, x_raw):
        assert x.shape == (8, 2)
        assert torch.all(torch.isfinite(x))
    assert not torch.equal(x_ema, x_raw)  # el hermano lleva otros pesos (crudos)
