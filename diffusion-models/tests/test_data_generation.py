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


def _empirical_moments(x, labels, k):
    """Media y covarianza empíricas de los puntos etiquetados como componente ``k``."""
    pts = np.asarray(x, dtype=np.float64)[np.asarray(labels) == k]
    return pts.mean(axis=0), np.cov(pts, rowvar=False)


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
    # Y lo que reportan los accesores sigue siendo lo que usa el muestreo: los
    # momentos empíricos de cada componente reproducen los parámetros
    # publicados. Es el chequeo por la puerta pública del mismo invariante (si
    # los arrays se aliasaran, `covariances_` reportaría la varianza inflada de
    # 100 mientras el muestreo seguiría usando la factorización original).
    x = mix.sample(8000)
    for i in range(2):
        mean_emp, cov_emp = _empirical_moments(x, mix.color_, i)
        assert np.allclose(mean_emp, mix.means_[i], atol=0.2)
        assert np.allclose(cov_emp, mix.covariances_[i], atol=0.2)


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
    # 1.6: cada entrada inválida nombra el parámetro culpable. El patrón va
    # anclado al principio del mensaje: los errores de K inconsistente
    # mencionan a `weights` de pasada (es quien declara el K de referencia), así
    # que sin el ancla el test no discriminaría a quién se está culpando.
    with pytest.raises(ValueError, match=f"^{culprit}"):
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


# --------------------------------------------------------------------------
# ExactGaussianMixture: contrato de muestreo (1.4, 1.5).
#
# El muestreo es determinístico en dos sentidos independientes que se prueban
# por separado: (a) la *composición* —cuántos puntos toca a cada componente— no
# se sortea, se reparte por mayor residuo, así que es exacta y no aproximada; y
# (b) la *geometría* de cada punto sí se sortea, pero de un rng sembrado, así
# que es reproducible.
# --------------------------------------------------------------------------


def _ring_mixture(weights, *, scale=0.25, seed=0):
    """Mixtura exacta con los pesos dados, medias en un anillo y covarianzas iguales.

    Sirve para los tests de composición, donde lo que importa son los pesos y no
    la geometría: las medias van bien separadas solo para que las etiquetas sean
    verificables sin ambigüedad.
    """
    weights = np.asarray(weights, dtype=np.float64)
    k = weights.size
    ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    means = 5.0 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    covs = np.broadcast_to(np.eye(2) * scale**2, (k, 2, 2))
    return ExactGaussianMixture(
        weights=weights, means=means, covariances=covs, seed=seed
    )


def test_exact_mixture_sample_returns_two_columns_float32():
    # 1.4: shape (n, 2) y precisión simple, como el resto de las formas.
    x = _exact_mixture(seed=0).sample(257)
    assert x.shape == (257, 2)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))


def test_exact_mixture_composition_matches_weights_exactly():
    # 1.4: con w_k * n entero la composición es *exacta*, no aproximada: 600/400
    # y no "600 ± ruido binomial". Sortear la componente de cada punto no
    # pasaría este test.
    mix = _exact_mixture(seed=0)  # weights = [0.6, 0.4]
    x = mix.sample(1000)
    counts = np.bincount(mix.color_, minlength=2)
    assert x.shape == (1000, 2)
    assert counts.tolist() == [600, 400]


def test_exact_mixture_composition_uses_largest_remainder():
    # 1.4: con w_k * n NO entero hay un resto que repartir. Pesos [0.7, 0.2,
    # 0.1] y n=13 dan w*n = [9.1, 2.6, 1.3] => parte entera [9, 2, 1] (suma 12)
    # y el punto que falta va al **mayor residuo** (0.6, la componente 1).
    # Discrimina las alternativas: un reparto parejo daría [5, 4, 4] y darle el
    # sobrante al peso más grande daría [10, 2, 1].
    mix = _ring_mixture([0.7, 0.2, 0.1], seed=3)
    mix.sample(13)
    assert np.bincount(mix.color_, minlength=3).tolist() == [9, 3, 1]


def test_exact_mixture_composition_ties_break_by_component_index():
    # 1.4: con residuos empatados el desempate es estable por índice, así que el
    # reparto es reproducible y no depende del orden interno del sort. Pesos
    # iguales de a tercios con n=10 dan parte entera [3, 3, 3] y el punto
    # restante va a la componente 0.
    mix = _ring_mixture([1 / 3, 1 / 3, 1 / 3], seed=4)
    mix.sample(10)
    assert np.bincount(mix.color_, minlength=3).tolist() == [4, 3, 3]


def test_exact_mixture_composition_with_fewer_points_than_components():
    # 1.4: con n < K casi toda la parte entera es cero y el reparto lo decide el
    # residuo: los 3 puntos van a las 3 componentes de mayor peso, sin perder
    # ninguno. Los pesos van en orden creciente para que el resultado esperado
    # ([0, 1, 1, 1]) no coincida con el de un reparto parejo ([1, 1, 1, 0]).
    mix = _ring_mixture([0.1, 0.2, 0.3, 0.4], seed=5)
    x = mix.sample(3)
    counts = np.bincount(mix.color_, minlength=4)
    assert x.shape == (3, 2)
    assert int(counts.sum()) == 3
    assert counts.tolist() == [0, 1, 1, 1]


def test_exact_mixture_zero_weight_component_gets_no_points():
    # 1.4: un peso nulo es un peso válido (no negativo) y su componente no
    # aporta ni un punto; la etiqueta correspondiente no aparece.
    mix = _ring_mixture([0.0, 0.5, 0.5], seed=6)
    mix.sample(100)
    assert np.bincount(mix.color_, minlength=3).tolist() == [0, 50, 50]
    assert 0 not in set(mix.color_.tolist())


def test_exact_mixture_same_seed_gives_identical_samples():
    # 1.5: dos instancias con la misma semilla dan exactamente las mismas
    # muestras (igualdad bit a bit, no "cerca").
    a = _exact_mixture(seed=7).sample(256)
    b = _exact_mixture(seed=7).sample(256)
    assert np.array_equal(a, b)


def test_exact_mixture_repeated_calls_with_same_seed_are_identical():
    # 1.5: "las mismas muestras que en la llamada anterior" también dentro de la
    # misma instancia: sample() resiembra el rng en cada llamada, así que no
    # arrastra estado del sorteo previo.
    mix = _exact_mixture(seed=7)
    a = mix.sample(256)
    labels_a = mix.color_.copy()
    b = mix.sample(256)
    assert np.array_equal(a, b)
    assert np.array_equal(labels_a, mix.color_)


def test_exact_mixture_different_seed_gives_different_samples():
    # 1.5: la reproducibilidad no es un muestreo degenerado; con otra semilla la
    # geometría cambia (la composición, en cambio, es la misma a propósito).
    a = _exact_mixture(seed=7).sample(256)
    c = _exact_mixture(seed=8).sample(256)
    assert not np.array_equal(a, c)


def test_exact_mixture_publishes_label_per_point():
    # 1.4: la etiqueta de componente se publica por punto, como la legacy, y
    # cada etiqueta es un índice de componente válido.
    mix = _exact_mixture(seed=0)
    x = mix.sample(64)
    assert mix.color_.shape == (len(x),)
    assert mix.color_.dtype.kind in "iu"
    assert set(mix.color_.tolist()) <= {0, 1}


def test_exact_mixture_color_reflects_only_the_current_sample():
    # 1.4: convención heredada de PointDistribution.sample(), que resetea
    # `color_` al inicio de cada muestreo: la etiqueta describe la muestra que
    # se acaba de devolver y nunca acumula ni arrastra la anterior.
    mix = _exact_mixture(seed=0)
    assert mix.color_ is None
    mix.sample(120)
    assert mix.color_.shape == (120,)
    mix.sample(30)
    assert mix.color_.shape == (30,)


def test_exact_mixture_label_matches_the_nearest_declared_mean():
    # 1.4: la etiqueta no es decorativa; identifica de qué componente salió cada
    # punto. Con modos bien separados (radio 5, escala 0.25) la componente más
    # cercana es la verdadera con probabilidad abrumadora.
    mix = _ring_mixture([0.5, 0.5], seed=9)
    x = mix.sample(400).astype(np.float64)
    dist = np.linalg.norm(x[:, None, :] - mix.means_[None, :, :], axis=-1)
    assert np.array_equal(dist.argmin(axis=1), mix.color_)


def test_exact_mixture_samples_are_grouped_by_component():
    # 1.4 — decisión de contrato: las muestras salen **agrupadas por
    # componente** (todos los puntos de la 0, después los de la 1, …), sin
    # permutación final, a diferencia de la legacy `mixture` (sklearn baraja) y
    # de `Spiral` (permuta explícitamente).
    #
    # Es aceptable acá y se fija a propósito: (a) la clase no entra al registry,
    # así que no puede llegar por YAML a una corrida con `shuffle: false`, que
    # es el único caso donde el orden se filtraría a los batches; (b) el consumo
    # natural es por `dataloader(..., shuffle=True)`, que reordena de todos
    # modos; (c) el oráculo analítico lee los parámetros, no el orden; y (d)
    # agrupado permite indexar los puntos de una componente sin filtrar,
    # justamente lo que necesitan las verificaciones por componente. Permutar
    # costaría un sorteo extra del rng —cambiando el mapa semilla → muestra— sin
    # agregar información.
    mix = _ring_mixture([0.5, 0.3, 0.2], seed=10)
    mix.sample(200)
    labels = mix.color_
    assert np.array_equal(labels, np.sort(labels))
    assert labels.tolist() == [0] * 100 + [1] * 60 + [2] * 40


def test_exact_mixture_per_component_moments_match_parameters():
    # 1.4: cada componente sale de aplicar la factorización de su covarianza a
    # ruido normal estándar, así que sus momentos empíricos reproducen la media
    # y la covarianza declaradas —incluida la correlación de la componente
    # rotada, que una escala solo diagonal perdería—. Tolerancia de Monte Carlo:
    # el reparto deja 7200 y 4800 puntos por componente, y el error estándar de
    # una entrada de la covarianza es ~sigma^2 * sqrt(2/n) <= 0.04 para las
    # varianzas de orden 1-2 de este caso, así que atol=0.2 son varios errores
    # estándar y además la semilla está fija.
    mix = _exact_mixture(seed=11)
    x = mix.sample(12_000)
    for i in range(2):
        mean_emp, cov_emp = _empirical_moments(x, mix.color_, i)
        assert np.allclose(mean_emp, mix.means_[i], atol=0.2), i
        assert np.allclose(cov_emp, mix.covariances_[i], atol=0.2), i
    # La segunda componente es la correlacionada: el test no pasaría si el
    # muestreo ignorara los términos fuera de la diagonal.
    assert mix.covariances_[1][0, 1] == pytest.approx(0.5)


def test_exact_mixture_sampling_does_not_touch_the_true_parameters():
    # 1.2, 1.4: el estado es inmutable tras la construcción — muestrear no
    # re-ajusta los parámetros (a diferencia de la estandarización empírica de
    # la clase base, que recalcula media y desvío en cada sample()).
    mix = _exact_mixture(seed=0)
    before = (mix.weights_, mix.means_, mix.covariances_)
    mix.sample(500)
    mix.sample(37)
    assert np.array_equal(mix.weights_, before[0])
    assert np.array_equal(mix.means_, before[1])
    assert np.array_equal(mix.covariances_, before[2])
    assert mix.mean_ is None and mix.std_ is None


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
