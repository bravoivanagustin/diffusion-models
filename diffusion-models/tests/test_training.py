"""Tests del módulo de entrenamiento (denoising score matching, `diffusion.training`).

Torch es dependencia dura del módulo, así que se hace `importorskip` al tope. Las corridas de
entrenamiento usan redes y datasets chicos para correr en CPU en segundos.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from diffusion.data_generation import infinite_bare, make_distribution
from diffusion.models import ScoreMLP, ScoreModel
from diffusion.sde import make_sde
from diffusion.training import (
    RunSpec,
    TrainConfig,
    TrainResult,
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
    assert len(result.history) >= 1
    assert all(math.isfinite(v) for v in result.history)


def test_train_history_no_vacio_con_log_every_cero():
    """history se registra a cadencia fija (desacoplada de log_every): nunca queda vacío."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    result = train(sde, net, _data(dist), TrainConfig(num_steps=5, log_every=0, seed=0))

    assert result.history  # no vacío ni con el default log_every=0
    assert all(math.isfinite(v) for v in result.history)


def test_train_baja_la_perdida():
    """Smoke de aprendizaje: tras muchos pasos la pérdida final es menor que la inicial.

    Se compara la **tendencia** (history[-1] < history[0]), no valores paso a paso: al pasar de
    épocas a pasos cambia el orden de consumo de ruido.
    """
    sde = make_sde("vp")
    dist = make_distribution("mixture", 2, n_components=8, seed=0)
    torch.manual_seed(0)
    net = ScoreMLP(data_dim=sde.data_dim, hidden_dim=64, num_blocks=2)
    data = _data(dist, n=512, batch_size=128)
    result = train(sde, net, data, TrainConfig(num_steps=240, seed=0))

    assert all(math.isfinite(v) for v in result.history)
    assert result.history[-1] < result.history[0]


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
    assert len(result.history) >= 1
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
    """Guardar y recargar reconstruye la red con los mismos pesos y la misma meta.

    El contrato de checkpoint es transitorio en esta task: ``save_checkpoint`` toma la receta de
    red de la propia red entrenada (no de ``TrainConfig``, que ya no la lleva) y ``load_checkpoint``
    la reconstruye como ``ScoreMLP`` (se vuelve model-agnóstico en la task 3.1).
    """
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    net = _small_net(sde)
    result = train(sde, net, _data(dist), _tiny_config(num_steps=2))

    path = tmp_path / "ckpt.pt"
    save_checkpoint(result, path)
    net2, meta = load_checkpoint(path)

    assert net2.data_dim == sde.data_dim == 2
    assert meta["sde_name"] == "vp"
    assert meta["history"] == pytest.approx(result.history)

    x = torch.randn(8, 2)
    t = torch.rand(8)
    result.net.eval()
    with torch.no_grad():
        assert torch.allclose(result.net(x, t), net2(x, t))


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
