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
from diffusion.data_generation.shapes import ExactGaussianMixture

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


# --------------------------------------------------------------------------
# ExactGaussianMixture: parámetros exactos, validación y accesores (1.1, 1.2,
# 1.3, 1.6, 5.1). El contrato de muestreo (composición, reproducibilidad,
# dtype/shape) lo fija la tarea 2.2; acá solo se prueba la construcción.
# La clase todavía no se exporta desde la API pública del módulo (tarea 2.4),
# así que se importa del submódulo donde vive.
# --------------------------------------------------------------------------

EXACT_WEIGHTS = [0.6, 0.4]
EXACT_MEANS = [[-2.0, 0.0], [2.0, 1.0]]
EXACT_COVS = [
    [[1.0, 0.0], [0.0, 1.0]],
    [[2.0, 0.5], [0.5, 0.75]],
]


def _exact_mixture(**overrides):
    """Mixtura exacta de 2 componentes, con los parámetros que se quieran pisar."""
    kwargs = {
        "weights": EXACT_WEIGHTS,
        "means": EXACT_MEANS,
        "covariances": EXACT_COVS,
    }
    kwargs.update(overrides)
    return ExactGaussianMixture(**kwargs)


def test_exact_mixture_exposes_parameters_before_sampling():
    # 1.1, 1.2: los tres accesores están disponibles sin haber muestreado.
    mix = _exact_mixture(seed=0)
    assert mix.color_ is None  # nunca se muestreó
    assert mix.weights_.shape == (2,)
    assert mix.means_.shape == (2, 2)
    assert mix.covariances_.shape == (2, 2, 2)
    assert mix.weights_.dtype == np.float64
    assert mix.means_.dtype == np.float64
    assert mix.covariances_.dtype == np.float64
    assert np.array_equal(mix.weights_, np.asarray(EXACT_WEIGHTS))
    assert np.array_equal(mix.means_, np.asarray(EXACT_MEANS))
    assert np.array_equal(mix.covariances_, np.asarray(EXACT_COVS))


def test_exact_mixture_accessors_return_copies():
    # 1.2: los parámetros verdaderos son inmutables desde afuera.
    mix = _exact_mixture()
    mix.weights_[0] = 99.0
    mix.means_[0, 0] = 99.0
    mix.covariances_[0, 0, 0] = 99.0
    assert np.array_equal(mix.weights_, np.asarray(EXACT_WEIGHTS))
    assert np.array_equal(mix.means_, np.asarray(EXACT_MEANS))
    assert np.array_equal(mix.covariances_, np.asarray(EXACT_COVS))


def test_exact_mixture_copies_constructor_arrays():
    # 1.2, 1.6: la mixtura es dueña de sus parámetros. Si aliaseara los arrays
    # del llamador, mutarlos después de construir rompería en silencio los
    # invariantes ya validados (pesos que suman uno, covarianzas SPD) y —peor—
    # desincronizaría covariances_ de la factorización de Cholesky persistida,
    # que es la que gobierna el muestreo.
    weights = np.array(EXACT_WEIGHTS, dtype=np.float64)
    means = np.array(EXACT_MEANS, dtype=np.float64)
    covariances = np.array(EXACT_COVS, dtype=np.float64)
    mix = ExactGaussianMixture(
        weights=weights, means=means, covariances=covariances, seed=0
    )

    # Mutaciones que la validación habría rechazado de haberlas visto.
    weights[0] = 99.0  # los pesos ya no suman uno
    means[0, 0] = 99.0
    covariances[0, 0, 0] = 100.0  # varianza inflada
    covariances[1] = [[-1.0, 0.0], [0.0, 3.0]]  # indefinida

    assert np.array_equal(mix.weights_, np.asarray(EXACT_WEIGHTS, dtype=np.float64))
    assert np.array_equal(mix.means_, np.asarray(EXACT_MEANS, dtype=np.float64))
    assert np.array_equal(mix.covariances_, np.asarray(EXACT_COVS, dtype=np.float64))
    # Los invariantes de la construcción siguen valiendo.
    assert float(mix.weights_.sum()) == pytest.approx(1.0)
    for cov in mix.covariances_:
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)
    # Y lo que reportan los accesores sigue siendo lo que usa el muestreo: la
    # Cholesky guardada reconstruye exactamente las covarianzas publicadas.
    chols = mix._chols
    assert np.allclose(chols @ np.swapaxes(chols, -1, -2), mix.covariances_)


def test_exact_mixture_accepts_full_spd_covariance():
    # 1.3: covarianza rotada (no diagonal) y anisotrópica: se acepta tal cual.
    ang = np.pi / 6.0
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    cov = rot @ np.diag([4.0, 0.25]) @ rot.T
    mix = ExactGaussianMixture(
        weights=[1.0], means=[[0.0, 0.0]], covariances=[cov], seed=0
    )
    assert np.allclose(mix.covariances_[0], cov)
    # Sigue siendo SPD y no diagonal: los autovalores son los pedidos.
    eigvals = np.linalg.eigvalsh(mix.covariances_[0])
    assert np.allclose(np.sort(eigvals), [0.25, 4.0])
    assert not np.isclose(mix.covariances_[0, 0, 1], 0.0)


@pytest.mark.parametrize(
    "overrides,culprit",
    [
        # Peso negativo (compensado para que sume uno: solo falla el signo).
        ({"weights": [-0.2, 1.2]}, "weights"),
        # Pesos que no suman uno.
        ({"weights": [0.5, 0.4]}, "weights"),
        # Sin componentes.
        ({"weights": []}, "weights"),
        # K inconsistente entre pesos y medias: con K=3 pesos, el que queda mal
        # dimensionado es means, y es a means a quien nombra el error.
        ({"weights": [0.5, 0.3, 0.2]}, "means"),
        ({"means": [[0.0, 0.0]]}, "means"),
        # K inconsistente entre pesos y covarianzas.
        ({"covariances": [[[1.0, 0.0], [0.0, 1.0]]]}, "covariances"),
        # Medias que no son 2D.
        ({"means": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}, "means"),
        # Covarianza no simétrica.
        ({"covariances": [EXACT_COVS[0], [[1.0, 0.3], [0.9, 1.0]]]}, "covariances"),
        # Covarianza simétrica pero indefinida.
        ({"covariances": [EXACT_COVS[0], [[1.0, 2.0], [2.0, 1.0]]]}, "covariances"),
        # Covarianza semidefinida (autovalor nulo): no es definida positiva.
        ({"covariances": [EXACT_COVS[0], [[1.0, 0.0], [0.0, 0.0]]]}, "covariances"),
    ],
)
def test_exact_mixture_rejects_invalid_parameters(overrides, culprit):
    # 1.6: cada entrada inválida nombra el parámetro culpable.
    with pytest.raises(ValueError, match=culprit):
        _exact_mixture(**overrides)


def test_exact_mixture_rejects_non_2d_dim():
    # 1.6: la exactitud barata de esta línea de trabajo vive en dim=2.
    with pytest.raises(ValueError, match="dim"):
        _exact_mixture(dim=3)


def test_exact_mixture_rejects_empirical_standardization():
    # 5.1: la estandarización empírica estima la transformación a partir del
    # sorteo, así que rompe la exactitud. Se rechaza fuerte en lugar de dejar
    # que make_distribution la descarte en silencio.
    with pytest.raises(ValueError, match="standardize"):
        _exact_mixture(standardize=True)


def test_exact_mixture_accepts_standardize_false_explicitly():
    # 5.1: el parámetro existe solo para poder rechazar el caso peligroso.
    mix = _exact_mixture(standardize=False)
    assert mix.standardize is False


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
