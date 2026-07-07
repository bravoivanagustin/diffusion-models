"""Tests del módulo de generación de datos de juguete (`diffusion.data_generation`)."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest

import itertools

from diffusion.data_generation import (
    Gaussian,
    TwoMoons,
    available_shapes,
    infinite_bare,
    make_distribution,
)

ALL_SHAPES = ["gaussian", "mixture", "two_moons", "spiral", "swiss_roll"]
DEFAULT_DIM = {
    "gaussian": 2,
    "mixture": 2,
    "two_moons": 2,
    "spiral": 2,
    "swiss_roll": 3,
}

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "data_generation.py"


@pytest.mark.parametrize("name", ALL_SHAPES)
def test_sample_shape_dtype_finite(name):
    dim = DEFAULT_DIM[name]
    x = make_distribution(name, dim, seed=0).sample(256)
    assert x.shape == (256, dim)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))


def test_available_shapes():
    assert set(available_shapes()) == set(ALL_SHAPES)


def test_factory_returns_right_type():
    assert isinstance(make_distribution("two_moons", 2), TwoMoons)
    assert isinstance(make_distribution("gaussian", 5), Gaussian)


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        make_distribution("does_not_exist", 2)


@pytest.mark.parametrize("name,bad_dim", [("two_moons", 3), ("spiral", 3), ("swiss_roll", 2)])
def test_unsupported_dim_raises(name, bad_dim):
    with pytest.raises(ValueError):
        make_distribution(name, bad_dim)


@pytest.mark.parametrize("name,dim", [("gaussian", 5), ("mixture", 4)])
def test_generalizable_shapes_accept_any_dim(name, dim):
    x = make_distribution(name, dim, seed=1).sample(64)
    assert x.shape == (64, dim)


@pytest.mark.parametrize("name", ALL_SHAPES)
def test_reproducibility(name):
    dim = DEFAULT_DIM[name]
    a = make_distribution(name, dim, seed=7).sample(128)
    b = make_distribution(name, dim, seed=7).sample(128)
    c = make_distribution(name, dim, seed=8).sample(128)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_standardize_gives_zero_mean_unit_std():
    x = make_distribution("gaussian", 3, scale=5.0, standardize=True, seed=0).sample(5000)
    assert np.allclose(x.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(x.std(axis=0), 1.0, atol=0.1)


def test_non_positive_n_raises():
    with pytest.raises(ValueError):
        make_distribution("gaussian", 2, seed=0).sample(0)


def test_torch_helpers():
    torch = pytest.importorskip("torch")
    dist = make_distribution("two_moons", 2, seed=0)
    t = dist.sample_torch(100)
    assert tuple(t.shape) == (100, 2)
    assert t.dtype == torch.float32
    loader = dist.dataloader(100, batch_size=32)
    batch = next(iter(loader))[0]
    assert batch.shape[1] == 2
    assert batch.shape[0] <= 32


def test_infinite_bare_does_not_exhaust():
    # Loader finito de 128 puntos en batches de 64 => 2 batches. Consumir 5
    # veces no debe agotarse (reinicia el recorrido). (4.1, 4.3)
    pytest.importorskip("torch")
    dist = make_distribution("two_moons", 2, seed=0)
    it = infinite_bare(dist.dataloader(128, batch_size=64))
    batches = list(itertools.islice(it, 5))
    assert len(batches) == 5


def test_infinite_bare_yields_bare_tensor():
    # Cada elemento es un tensor crudo (B, 2), no una tupla (x0,). (4.2)
    torch = pytest.importorskip("torch")
    dist = make_distribution("two_moons", 2, seed=0)
    it = infinite_bare(dist.dataloader(128, batch_size=64))
    for batch in itertools.islice(it, 5):
        assert isinstance(batch, torch.Tensor)
        assert not isinstance(batch, tuple)
        assert batch.ndim == 2
        assert batch.shape[1] == 2
        assert batch.shape[0] <= 64


def test_cli_smoke(tmp_path):
    out = tmp_path / "d.npz"
    png = tmp_path / "d.png"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--shape", "two_moons", "--dim", "2",
         "--n-samples", "200", "--seed", "0", "--out", str(out), "--preview", str(png)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists() and png.exists()
    data = np.load(out)
    assert data["X"].shape == (200, 2)
    assert data["X"].dtype == np.float32
