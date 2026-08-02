"""Tests de la verdad analítica del laboratorio 2D (`diffusion.analytic`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import diffusion.analytic
from diffusion.analytic.quadrature import QuadratureGrid, auto_grid, integrate


# --------------------------------------------------------------- paquete y fronteras


def test_paquete_importable_desde_la_ruta_publica():
    """El módulo se importa como `diffusion.analytic` y publica un `__all__`."""
    assert diffusion.analytic.__name__ == "diffusion.analytic"
    assert isinstance(diffusion.analytic.__all__, list)
    assert all(
        hasattr(diffusion.analytic, nombre) for nombre in diffusion.analytic.__all__
    )


def test_no_importa_samplers_ni_training():
    """Dirección de dependencias `data_generation → sde → analytic` (criterio 2.4).

    Se chequea en un intérprete nuevo porque `sys.modules` del proceso de pytest ya
    tiene cargados los módulos que importan las otras suites.
    """
    src = Path(diffusion.analytic.__file__).resolve().parents[2]
    codigo = (
        "import sys, diffusion.analytic; "
        "print([m for m in ('diffusion.samplers', 'diffusion.training') "
        "if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


# ------------------------------------------------------------------------ cuadratura 2D


def _identidad(escala: float) -> "torch.Tensor":
    """Covarianza isotrópica ``escala · I`` de forma ``(2, 2)``."""
    return escala * torch.eye(2, dtype=torch.float64)


def _diagonal(a: float, b: float) -> "torch.Tensor":
    """Covarianza diagonal ``diag(a, b)`` de forma ``(2, 2)``."""
    return torch.diag(torch.tensor([a, b], dtype=torch.float64))


def _rotada(a: float, b: float, angulo: float) -> "torch.Tensor":
    """Covarianza ``diag(a, b)`` rotada un ángulo dado (en radianes)."""
    c, s = torch.cos(torch.tensor(angulo)), torch.sin(torch.tensor(angulo))
    rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    return rot @ _diagonal(a, b) @ rot.T


def _gaussiana(media: "torch.Tensor", cov: "torch.Tensor"):
    """Devuelve la densidad gaussiana 2D de parámetros dados como callable de ``(N, 2)``."""
    import math

    def densidad(puntos: "torch.Tensor") -> "torch.Tensor":
        mu = media.to(puntos.dtype)
        precision = torch.linalg.inv(cov.to(puntos.dtype))
        dif = puntos - mu
        cuadratica = ((dif @ precision) * dif).sum(dim=-1)
        det = torch.linalg.det(cov.to(puntos.dtype))
        norma = 2.0 * math.pi * torch.sqrt(det)
        return torch.exp(-0.5 * cuadratica) / norma

    return densidad


def _mixtura(pesos, medias, covarianzas):
    """Devuelve la densidad de una mixtura de gaussianas como callable de ``(N, 2)``."""
    componentes = [_gaussiana(mu, cov) for mu, cov in zip(medias, covarianzas)]

    def densidad(puntos: "torch.Tensor") -> "torch.Tensor":
        total = torch.zeros(puntos.shape[0], dtype=puntos.dtype)
        for peso, componente in zip(pesos, componentes):
            total = total + peso * componente(puntos)
        return total

    return densidad


def test_la_cuadratura_no_importa_sde_ni_data_generation():
    """La cuadratura es una utilidad numérica autónoma: solo depende de ``torch``."""
    src = Path(diffusion.analytic.__file__).resolve().parents[2]
    codigo = (
        "import sys, diffusion.analytic.quadrature; "
        "print([m for m in ('diffusion.sde', 'diffusion.data_generation', "
        "'diffusion.samplers', 'diffusion.training') if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_integrar_gaussiana_isotropica_da_masa_uno():
    """El observable de la task: una gaussiana de covarianza conocida integra a uno."""
    media = torch.zeros(2, dtype=torch.float64)
    cov = _identidad(0.5)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...])
    assert integrate(_gaussiana(media, cov), grid) == pytest.approx(1.0, abs=1e-8)


def test_integrar_gaussiana_anisotropica_da_masa_uno():
    """La malla se adapta a una componente mucho más angosta en un eje."""
    media = torch.tensor([0.5, -0.5], dtype=torch.float64)
    cov = _diagonal(1.0, 0.02)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...])
    assert not grid.truncated
    assert integrate(_gaussiana(media, cov), grid) == pytest.approx(1.0, abs=1e-8)


def test_integrar_gaussiana_rotada_da_masa_uno():
    """Con covarianza rotada el paso lo fija el autovalor mínimo, no la diagonal."""
    import math

    media = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    cov = _rotada(1.0, 0.01, math.pi / 4)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...])
    assert not grid.truncated
    assert integrate(_gaussiana(media, cov), grid) == pytest.approx(1.0, abs=1e-8)


def test_integrar_mixtura_de_dos_componentes_da_masa_uno():
    """La malla cubre todas las medias a la vez, no solo una."""
    medias = torch.tensor([[-2.0, 0.0], [2.0, 1.0]], dtype=torch.float64)
    covarianzas = torch.stack([_identidad(0.3), _diagonal(0.5, 0.1)])
    grid = auto_grid(means=medias, covariances=covarianzas)
    densidad = _mixtura([0.3, 0.7], medias, covarianzas)
    assert integrate(densidad, grid) == pytest.approx(1.0, abs=1e-8)


def test_el_paso_lo_fija_la_componente_mas_angosta():
    """El paso sale de ``min_k sqrt(lambda_min)``, no de la componente ancha ni del promedio."""
    medias = torch.zeros(2, 2, dtype=torch.float64)
    covarianzas = torch.stack([_identidad(4.0), _identidad(0.01)])
    grid = auto_grid(means=medias, covariances=covarianzas, points_per_sigma=6.0)

    paso_angosta = 0.1 / 6.0  # sqrt(0.01) / points_per_sigma
    paso_ancha = 2.0 / 6.0
    paso_promedio = ((2.0 + 0.1) / 2.0) / 6.0

    assert grid.spacing <= paso_angosta
    assert grid.spacing == pytest.approx(paso_angosta, rel=1e-2)
    assert grid.spacing < paso_promedio / 2.0
    assert grid.spacing < paso_ancha / 2.0


def test_la_malla_no_colapsa_cuando_el_ruido_del_kernel_tiende_a_cero():
    """Piso de concentración: sumar un ruido diminuto no cambia la malla.

    Es la contracara de dimensionar por ``sigma_t``: si el paso saliera del desvío del
    kernel, con ``sigma_t = 1e-5`` la resolución pedida explotaría.
    """
    media = torch.zeros(1, 2, dtype=torch.float64)
    base = _identidad(0.25)[None, ...]
    sigma_t = 1e-5
    ruideada = base + (sigma_t**2) * torch.eye(2, dtype=torch.float64)[None, ...]

    grid_base = auto_grid(means=media, covariances=base)
    grid_ruideada = auto_grid(means=media, covariances=ruideada)

    assert grid_ruideada.n_points == grid_base.n_points
    assert not grid_ruideada.truncated


def test_prior_std_ensancha_el_dominio():
    """Integrar contra una distribución de referencia amplia extiende el dominio."""
    media = torch.zeros(1, 2, dtype=torch.float64)
    cov = _identidad(0.25)[None, ...]
    sin_prior = auto_grid(means=media, covariances=cov)
    con_prior = auto_grid(means=media, covariances=cov, prior_std=5.0, n_sigma=8.0)

    assert con_prior.half_width > sin_prior.half_width
    assert con_prior.half_width == pytest.approx(40.0)


def test_prior_std_chico_no_encoge_el_dominio():
    """El prior amplía el dominio, nunca lo recorta por debajo de las componentes."""
    media = torch.zeros(1, 2, dtype=torch.float64)
    cov = _identidad(1.0)[None, ...]
    sin_prior = auto_grid(means=media, covariances=cov)
    con_prior = auto_grid(means=media, covariances=cov, prior_std=0.01)
    assert con_prior.half_width == pytest.approx(sin_prior.half_width)


def test_el_dominio_cubre_las_medias_mas_varios_desvios():
    """``half_width`` = mayor ``|media| + n_sigma * sqrt(lambda_max)`` entre componentes."""
    medias = torch.tensor([[3.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    covarianzas = torch.stack([_identidad(1.0), _identidad(4.0)])
    grid = auto_grid(means=medias, covariances=covarianzas, n_sigma=4.0)
    # componente 0: 3 + 4*1 = 7 ; componente 1: 1 + 4*2 = 9
    assert grid.half_width == pytest.approx(9.0)


def test_resolucion_por_encima_del_tope_devuelve_malla_truncada():
    """Segundo observable: el tope de puntos queda marcado en la malla devuelta."""
    media = torch.zeros(1, 2, dtype=torch.float64)
    cov = _diagonal(1.0, 1e-4)[None, ...]
    grid = auto_grid(means=media, covariances=cov, max_points=64)
    assert grid.truncated
    assert grid.n_points == 64


def test_resolucion_dentro_del_tope_no_marca_la_malla():
    """Sin recorte, ``truncated`` queda en falso (su valor por defecto)."""
    media = torch.zeros(1, 2, dtype=torch.float64)
    cov = _identidad(1.0)[None, ...]
    grid = auto_grid(means=media, covariances=cov, max_points=4096)
    assert not grid.truncated
    assert grid.n_points < 4096


def test_una_malla_severamente_truncada_delata_la_perdida_de_masa():
    """La masa integrada es el detector de una malla insuficiente."""
    media = torch.zeros(2, dtype=torch.float64)
    cov = _diagonal(1.0, 1e-4)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...], max_points=8)
    assert grid.truncated
    assert integrate(_gaussiana(media, cov), grid) < 0.5


def test_spacing_consistente_con_half_width_y_n_points():
    """El paso es el ancho total dividido por la cantidad de intervalos."""
    grid = QuadratureGrid(half_width=4.0, n_points=9)
    assert grid.spacing == pytest.approx(8.0 / 8.0)
    assert not grid.truncated


def test_la_malla_es_inmutable():
    """``QuadratureGrid`` es un valor congelado: describe una malla, no la muta."""
    grid = QuadratureGrid(half_width=1.0, n_points=4)
    with pytest.raises((AttributeError, TypeError)):
        grid.n_points = 8  # type: ignore[misc]


def test_integrate_da_el_mismo_valor_sea_cual_sea_el_dtype_del_caller():
    """La suma corre en doble precisión aunque ``fn`` devuelva ``float32``."""
    media = torch.zeros(2, dtype=torch.float64)
    cov = _identidad(0.5)
    grid = auto_grid(means=media[None, :], covariances=cov[None, ...])
    densidad = _gaussiana(media, cov)

    def densidad_f32(puntos: "torch.Tensor") -> "torch.Tensor":
        return densidad(puntos).to(torch.float32)

    exacta = integrate(densidad, grid)
    degradada = integrate(densidad_f32, grid)
    assert exacta == pytest.approx(1.0, abs=1e-8)
    assert degradada == pytest.approx(exacta, abs=1e-5)
    assert isinstance(degradada, float)


def test_integrate_acumula_en_la_precision_pedida_y_no_en_la_del_caller():
    """La suma se acumula en doble precisión aunque los valores lleguen en ``float32``.

    Con decenas de miles de nodos, acumular en la precisión de ``fn`` deja un error
    relativo del orden de ``1e-7``; acumular en doble lo deja en el ruido de ``1e-15``.
    """
    grid = QuadratureGrid(half_width=1.0, n_points=201)
    valor_f32 = torch.tensor(1.0 / 3.0, dtype=torch.float32)

    def constante_f32(puntos: "torch.Tensor") -> "torch.Tensor":
        return valor_f32.expand(puntos.shape[0])

    esperado = grid.n_points**2 * float(valor_f32) * grid.spacing**2
    assert integrate(constante_f32, grid) == pytest.approx(esperado, rel=1e-12)


def test_integrate_pasa_los_puntos_en_el_dtype_pedido():
    """El dtype de la cuadratura lo fija ``integrate``, no el caller."""
    vistos: list["torch.dtype"] = []
    formas: list[tuple[int, ...]] = []

    def espia(puntos: "torch.Tensor") -> "torch.Tensor":
        vistos.append(puntos.dtype)
        formas.append(tuple(puntos.shape))
        return torch.zeros(puntos.shape[0], dtype=puntos.dtype)

    grid = QuadratureGrid(half_width=1.0, n_points=5)
    integrate(espia, grid)
    assert set(vistos) == {torch.float64}
    assert all(forma[1] == 2 for forma in formas)
    assert sum(forma[0] for forma in formas) == 25

    vistos.clear()
    integrate(espia, grid, dtype=torch.float32)
    assert set(vistos) == {torch.float32}


def test_integrate_de_una_constante_es_la_suma_de_riemann():
    """Suma de Riemann pura: ``sum(valores) * spacing**2``, sin pesos de trapecio."""
    grid = QuadratureGrid(half_width=1.0, n_points=101)
    area = integrate(
        lambda puntos: torch.ones(puntos.shape[0], dtype=puntos.dtype), grid
    )
    assert area == pytest.approx((101 * grid.spacing) ** 2, rel=1e-12)


# ------------------------------------------------- cuadratura 2D: entradas inválidas


def test_grid_rechaza_pocos_puntos():
    """Una malla necesita al menos dos puntos por eje para tener paso."""
    with pytest.raises(ValueError, match=r"^n_points"):
        QuadratureGrid(half_width=1.0, n_points=1)


@pytest.mark.parametrize("ancho", [0.0, -1.0, float("inf"), float("nan")])
def test_grid_rechaza_medio_ancho_invalido(ancho: float):
    """El medio ancho tiene que ser finito y positivo."""
    with pytest.raises(ValueError, match=r"^half_width"):
        QuadratureGrid(half_width=ancho, n_points=4)


def test_auto_grid_rechaza_medias_con_forma_equivocada():
    """Las medias son ``(K, 2)``: cualquier otra forma es un error del caller."""
    with pytest.raises(ValueError, match=r"^means"):
        auto_grid(
            means=torch.zeros(3, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
        )


def test_auto_grid_rechaza_covarianzas_con_forma_equivocada():
    """Las covarianzas son ``(K, 2, 2)``."""
    with pytest.raises(ValueError, match=r"^covariances"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0),
        )


def test_auto_grid_rechaza_cantidades_desparejas():
    """Tiene que haber una covarianza por media."""
    with pytest.raises(ValueError, match=r"^covariances"):
        auto_grid(
            means=torch.zeros(2, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
        )


def test_auto_grid_rechaza_covarianza_no_simetrica():
    """La precondición es covarianzas SPD."""
    cov = torch.tensor([[1.0, 0.5], [-0.5, 1.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"^covariances"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=cov[None, ...],
        )


def test_auto_grid_rechaza_covarianza_singular():
    """Sin autovalor mínimo positivo no hay paso que resolver la componente."""
    cov = _diagonal(1.0, 0.0)
    with pytest.raises(ValueError, match=r"^covariances"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=cov[None, ...],
        )


def test_auto_grid_rechaza_medias_no_finitas():
    """Una media infinita haría un dominio infinito."""
    medias = torch.tensor([[0.0, float("nan")]], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"^means"):
        auto_grid(means=medias, covariances=_identidad(1.0)[None, ...])


@pytest.mark.parametrize("valor", [0.0, -1.0])
def test_auto_grid_rechaza_n_sigma_invalido(valor: float):
    """El dominio se mide en desvíos: el factor tiene que ser positivo."""
    with pytest.raises(ValueError, match=r"^n_sigma"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
            n_sigma=valor,
        )


@pytest.mark.parametrize("valor", [0.0, -3.0])
def test_auto_grid_rechaza_points_per_sigma_invalido(valor: float):
    """La resolución pedida se mide en puntos por desvío: tiene que ser positiva."""
    with pytest.raises(ValueError, match=r"^points_per_sigma"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
            points_per_sigma=valor,
        )


@pytest.mark.parametrize("valor", [1, 0, -4])
def test_auto_grid_rechaza_tope_de_puntos_invalido(valor: int):
    """El tope tiene que dejar espacio para una malla con paso."""
    with pytest.raises(ValueError, match=r"^max_points"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
            max_points=valor,
        )


@pytest.mark.parametrize("valor", [0.0, -2.0, float("inf")])
def test_auto_grid_rechaza_prior_std_invalido(valor: float):
    """La escala de la distribución de referencia tiene que ser finita y positiva."""
    with pytest.raises(ValueError, match=r"^prior_std"):
        auto_grid(
            means=torch.zeros(1, 2, dtype=torch.float64),
            covariances=_identidad(1.0)[None, ...],
            prior_std=valor,
        )


def test_integrate_rechaza_dtype_no_flotante():
    """La cuadratura es en punto flotante; un dtype entero es un error del caller."""
    grid = QuadratureGrid(half_width=1.0, n_points=4)
    with pytest.raises(ValueError, match=r"^dtype"):
        integrate(
            lambda puntos: torch.zeros(puntos.shape[0]), grid, dtype=torch.int64
        )


def test_integrate_rechaza_fn_con_forma_de_salida_equivocada():
    """``fn`` recibe ``(N, 2)`` y devuelve ``(N,)``: cualquier otra forma se avisa."""
    grid = QuadratureGrid(half_width=1.0, n_points=4)
    with pytest.raises(ValueError, match=r"^fn"):
        integrate(lambda puntos: puntos, grid)
