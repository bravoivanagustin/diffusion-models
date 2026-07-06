"""Tests del módulo de entrenamiento (denoising score matching, `diffusion.training`).

Torch es dependencia dura del módulo, así que se hace `importorskip` al tope. Las corridas de
entrenamiento usan redes y datasets chicos para correr en CPU en segundos.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from diffusion.data_generation import make_distribution
from diffusion.models import ScoreMLP
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


def _tiny_config(**overrides) -> TrainConfig:
    base = dict(epochs=2, n_samples=128, batch_size=64, hidden_dim=32, num_blocks=1, seed=0)
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
def test_train_devuelve_red_con_data_dim_correcto(name):
    """train() instancia la red con el data_dim de la SDE y traza finita."""
    sde = make_sde(name)
    dist = make_distribution("gaussian", 2, seed=0)
    result = train(sde, dist, _tiny_config(epochs=1))

    assert isinstance(result, TrainResult)
    assert result.net.data_dim == sde.data_dim
    assert result.sde_name == name
    assert len(result.history) == 1
    assert all(math.isfinite(v) for v in result.history)


def test_train_baja_la_perdida():
    """Smoke de aprendizaje: tras varias épocas la pérdida final es menor que la inicial."""
    sde = make_sde("vp")
    dist = make_distribution("mixture", 2, n_components=8, seed=0)
    config = _tiny_config(epochs=60, n_samples=512, batch_size=128, hidden_dim=64, num_blocks=2)
    result = train(sde, dist, config)

    assert all(math.isfinite(v) for v in result.history)
    assert result.history[-1] < result.history[0]


def test_train_reproducible_con_misma_seed():
    def run():
        sde = make_sde("vp")
        dist = make_distribution("gaussian", 2, seed=1)
        return train(sde, dist, _tiny_config(epochs=5, n_samples=256, seed=7)).history

    assert run() == pytest.approx(run())


def test_train_con_grad_clip_corre():
    sde = make_sde("ve")
    dist = make_distribution("gaussian", 2, seed=0)
    result = train(sde, dist, _tiny_config(epochs=2, grad_clip=1.0))
    assert len(result.history) == 2
    assert all(math.isfinite(v) for v in result.history)


# -------------------------------------------------------------- checkpoints


def test_checkpoint_roundtrip(tmp_path):
    """Guardar y recargar reconstruye la red con los mismos pesos y la misma meta."""
    sde = make_sde("vp")
    dist = make_distribution("gaussian", 2, seed=0)
    result = train(sde, dist, _tiny_config(epochs=1))

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
    raw = {
        "sde": {"name": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "data": {"shape": "mixture", "dim": 2, "n_samples": 512, "n_components": 8, "seed": 0},
        "train": {"epochs": 3, "batch_size": 128, "lr": 1e-3, "seed": 0},
        "out": {"checkpoint": "models/x.pt", "loss_curve": "models/x.png"},
    }
    spec = build_run(raw)

    assert isinstance(spec, RunSpec)
    assert spec.sde.name == "vp"
    assert spec.distribution.name == "mixture"
    assert spec.config.epochs == 3
    assert spec.config.n_samples == 512  # n_samples viaja de 'data' a TrainConfig
    assert spec.checkpoint.name == "x.pt"
    assert spec.loss_curve.name == "x.png"


def test_build_run_falla_sin_claves_obligatorias():
    with pytest.raises(ValueError):
        build_run({"data": {"shape": "gaussian"}})  # falta sde.name
    with pytest.raises(ValueError):
        build_run({"sde": {"name": "vp"}})  # falta data.shape


def test_build_run_rechaza_clave_desconocida():
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2},
        "train": {"epochs": 1, "lr_typo": 0.1},
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
        "train:\n"
        "  epochs: 2\n"
        "  lr: 0.001\n"
    )
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")

    spec = build_run(load_config(path))
    assert spec.sde.name == "vp"
    assert spec.distribution.name == "gaussian"
    assert spec.config.epochs == 2
    assert spec.config.n_samples == 256
