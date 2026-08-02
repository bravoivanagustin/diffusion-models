"""Tests de la verdad analítica del laboratorio 2D (`diffusion.analytic`)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import diffusion.analytic
from diffusion.analytic.mixture_oracle import MixtureOracle
from diffusion.analytic.quadrature import QuadratureGrid, auto_grid, integrate
from diffusion.data_generation import ExactGaussianMixture, GaussianMixture
from diffusion.sde import ForwardSDE, make_sde


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


# ------------------------------- oráculo: admisibilidad y parámetros marginales


#: Las tres variantes escalar-gaussianas del Eje 1.
_SDES_ESCALARES = ["vp", "ve", "sub_vp"]

#: Defaults de las variantes registradas, replicados acá a propósito: la forma cerrada
#: esperada se escribe con ``math`` y sin importar ``diffusion.sde.schedules``, así que es
#: una **segunda implementación** y no una tautología contra el paquete.
_BETA_MIN, _BETA_MAX = 0.1, 20.0
_SIGMA_MIN, _SIGMA_MAX = 0.01, 5.0

#: Tiempos de referencia: el régimen casi singular, dos intermedios y el horizonte.
_TIEMPOS = [1e-4, 0.05, 0.5, 1.0]


def _beta_integral(t: float) -> float:
    """``∫_0^t beta(s) ds`` del schedule lineal de VP y sub-VP, escrito a mano."""
    return _BETA_MIN * t + 0.5 * (_BETA_MAX - _BETA_MIN) * t**2


def _alpha_sigma_cerrados(nombre: str, t: float) -> tuple[float, float]:
    """Forma cerrada de ``(alpha_t, sigma_t)`` por variante, calculada con ``math``.

    Convenciones que este helper fija de forma independiente:

    - VP: ``alpha = e^{-½∫beta}``, ``sigma = sqrt(1 - e^{-∫beta})``.
    - VE: ``alpha = 1`` (no contrae la media), ``sigma`` geométrico.
    - sub-VP: mismo ``alpha`` que VP y ``sigma = 1 - e^{-∫beta}`` **sin raíz**, que es la
      convención que publica el paquete y que el oráculo debe consumir tal cual.
    """
    if nombre == "vp":
        b = _beta_integral(t)
        return math.exp(-0.5 * b), math.sqrt(1.0 - math.exp(-b))
    if nombre == "ve":
        return 1.0, _SIGMA_MIN * (_SIGMA_MAX / _SIGMA_MIN) ** t
    if nombre == "sub_vp":
        b = _beta_integral(t)
        return math.exp(-0.5 * b), 1.0 - math.exp(-b)
    raise AssertionError(f"variante sin forma cerrada en el test: {nombre!r}")


def _mixtura_exacta() -> ExactGaussianMixture:
    """Mixtura exacta chica: pesos desbalanceados y una componente rotada y estirada."""
    return ExactGaussianMixture(
        2,
        weights=[0.3, 0.7],
        means=[[-1.5, 0.5], [2.0, -1.0]],
        covariances=[
            _rotada(1.0, 0.09, math.pi / 3).tolist(),
            _diagonal(0.25, 0.04).tolist(),
        ],
        seed=0,
    )


class _SDEEscalarInventada(ForwardSDE):
    """SDE escalar-gaussiana que no está en el registry.

    Existe para probar que el oráculo cubre la **familia** por un único camino de código y
    no las tres variantes conocidas: su kernel es ``N(alpha_t x_0, sigma_t² I)`` con
    ``alpha_t = 1 - 0.4 t`` y ``sigma_t = 0.2 + t``, que no coinciden con ningún schedule
    del paquete.
    """

    name = "escalar_inventada"

    def sde(self, x, t):  # pragma: no cover - el oráculo no usa los coeficientes
        return torch.zeros_like(x), self._expand_t(t, x)

    def marginal_prob(self, x0, t):
        tt = self._expand_t(t, x0)
        return (1.0 - 0.4 * tt) * x0, 0.2 + tt

    def prior_sampling(self, shape, **kwargs):  # pragma: no cover - fuera de alcance
        return torch.zeros(shape)


class _SDEContadora(_SDEEscalarInventada):
    """Como :class:`_SDEEscalarInventada` pero cuenta las consultas al contrato marginal."""

    def __init__(self) -> None:
        super().__init__()
        self.consultas = 0

    def marginal_prob(self, x0, t):
        self.consultas += 1
        return super().marginal_prob(x0, t)


class _SDEDiagonal(_SDEEscalarInventada):
    """SDE fuera de la familia: escala **cada coordenada** por un factor distinto.

    Su desvío sí colapsa a un escalar por muestra, así que lo único que la delata es la
    verificación de proporcionalidad de la media.
    """

    name = "diagonal_falsa"

    def marginal_prob(self, x0, t):
        tt = self._expand_t(t, x0)
        escala = torch.tensor([1.0, 0.5], dtype=x0.dtype, device=x0.device)
        return (1.0 - 0.4 * tt) * x0 * escala, 0.2 + tt


class _SDEMezcladora(_SDEEscalarInventada):
    """SDE fuera de la familia: su media **promedia** las dos coordenadas del dato.

    Es el caso que justifica sondear con coordenadas distintas entre sí: con un vector de
    unos, ``mean = alpha_t · (x + y)/2`` da exactamente ``alpha_t · x_0`` y el rechazo se
    perdería. Con ``(1, 2)`` la media sale ``(1.5, 1.5)``, que no es proporcional a ``(1, 2)``.
    """

    name = "mezcladora_falsa"

    def marginal_prob(self, x0, t):
        tt = self._expand_t(t, x0)
        promedio = x0.mean(dim=-1, keepdim=True).expand_as(x0)
        return (1.0 - 0.4 * tt) * promedio, 0.2 + tt


class _SDEPorDimension(_SDEEscalarInventada):
    """SDE fuera de la familia: su desvío es **por coordenada**, no por muestra."""

    name = "por_dimension_falsa"

    def marginal_prob(self, x0, t):
        tt = self._expand_t(t, x0)
        escala = torch.tensor([1.0, 3.0], dtype=x0.dtype, device=x0.device)
        return x0, (0.2 + tt) * escala


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_marginal_params_coincide_con_la_forma_cerrada_de_cada_variante(nombre: str):
    """Un solo camino de código da ``alpha`` y ``sigma`` en las tres variantes escalares.

    Se contrasta contra la forma cerrada escrita a mano en el test, no contra el paquete.

    Tolerancia: ``1e-9`` es holgadísima frente al ruido de la doble precisión, pero contempla
    la cancelación catastrófica de ``1 - e^{-∫beta}`` cuando ``t`` es diminuto (en
    ``t = 1e-4`` el desvío de sub-VP vale ``1e-5`` y la resta pierde unos cinco dígitos, así
    que ``math.exp`` y ``torch.exp`` pueden separarse ``~1e-11`` con solo diferir en un ulp).
    Cualquier convención equivocada —tomar una raíz, re-derivar el schedule— se aparta
    órdenes de magnitud, así que el test no pierde poder discriminante.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    t = torch.tensor(_TIEMPOS, dtype=torch.float64)

    alpha, sigma = oraculo.marginal_params(t)

    assert alpha.shape == (len(_TIEMPOS), 1)
    assert sigma.shape == (len(_TIEMPOS), 1)
    for i, valor in enumerate(_TIEMPOS):
        a, s = _alpha_sigma_cerrados(nombre, valor)
        assert float(alpha[i, 0]) == pytest.approx(a, rel=1e-9)
        assert float(sigma[i, 0]) == pytest.approx(s, rel=1e-9)


def test_ve_no_contrae_la_media():
    """En VE el kernel deja el dato en su lugar: ``alpha_t == 1`` para todo ``t``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))
    t = torch.tensor(_TIEMPOS, dtype=torch.float64)
    alpha, _ = oraculo.marginal_params(t)
    assert torch.equal(alpha, torch.ones_like(alpha))


def test_el_desvio_de_sub_vp_se_consume_sin_raiz():
    """sub-VP publica ``std = 1 - e^{-∫beta}`` (sin raíz) y el oráculo lo toma tal cual.

    Es el test que discrimina la "corrección" plausible: con ``t = 0.05`` el desvío vale
    ``0.0294`` y su raíz ``0.1716``, casi seis veces más, así que tomar la raíz no puede
    pasar inadvertido.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("sub_vp"))
    _, sigma = oraculo.marginal_params(torch.tensor([0.05], dtype=torch.float64))

    esperado = 1.0 - math.exp(-_beta_integral(0.05))
    assert float(sigma[0, 0]) == pytest.approx(esperado, rel=1e-12)
    assert float(sigma[0, 0]) < math.sqrt(esperado) / 4.0


def test_el_desvio_de_sub_vp_queda_por_debajo_del_de_vp():
    """Invariante de la variante: su varianza es estrictamente menor que la de VP."""
    t = torch.tensor([0.05, 0.5, 1.0], dtype=torch.float64)
    mixtura = _mixtura_exacta()
    _, sigma_vp = MixtureOracle(mixtura, make_sde("vp")).marginal_params(t)
    _, sigma_sub = MixtureOracle(mixtura, make_sde("sub_vp")).marginal_params(t)
    assert bool((sigma_sub < sigma_vp).all())


def test_una_sde_escalar_gaussiana_nueva_pasa_por_el_mismo_camino():
    """Sin ramificación por variante: anda una SDE de la familia ajena al registry."""
    oraculo = MixtureOracle(_mixtura_exacta(), _SDEEscalarInventada())
    t = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64)

    alpha, sigma = oraculo.marginal_params(t)

    esperado_alpha = torch.tensor([[1.0], [0.88], [0.6]], dtype=torch.float64)
    esperado_sigma = torch.tensor([[0.2], [0.5], [1.2]], dtype=torch.float64)
    assert torch.allclose(alpha, esperado_alpha, rtol=1e-12, atol=1e-15)
    assert torch.allclose(sigma, esperado_sigma, rtol=1e-12, atol=1e-15)


def test_los_parametros_marginales_no_heredan_la_precision_del_caller():
    """Los pasos intermedios corren en doble aunque ``t`` llegue en simple precisión.

    Los tiempos elegidos son exactos en ``float32``, así que la conversión no aporta error:
    la única fuente posible de un error relativo ``~1e-7`` sería evaluar el schedule en
    simple precisión.
    """
    tiempos = [0.25, 0.5, 1.0]
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))

    alpha, sigma = oraculo.marginal_params(torch.tensor(tiempos, dtype=torch.float32))

    assert alpha.dtype is torch.float64
    assert sigma.dtype is torch.float64
    for i, valor in enumerate(tiempos):
        a, s = _alpha_sigma_cerrados("vp", valor)
        assert float(alpha[i, 0]) == pytest.approx(a, rel=1e-12)
        assert float(sigma[i, 0]) == pytest.approx(s, rel=1e-12)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_el_tiempo_plano_y_en_columna_dan_el_mismo_resultado(nombre: str):
    """Contrato de shapes del proyecto: ``t`` se acepta como ``(B,)`` y como ``(B, 1)``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    plano = torch.tensor([0.05, 0.7], dtype=torch.float64)
    columna = plano.reshape(-1, 1)

    alpha_plano, sigma_plano = oraculo.marginal_params(plano)
    alpha_columna, sigma_columna = oraculo.marginal_params(columna)

    assert torch.equal(alpha_plano, alpha_columna)
    assert torch.equal(sigma_plano, sigma_columna)
    assert torch.equal(
        oraculo.component_covariances(plano),
        oraculo.component_covariances(columna),
    )


def test_component_covariances_es_alpha2_por_sigma_por_componente():
    """``Sigma_k(t) = alpha_t² Sigma_k + sigma_t² I``, con la componente rotada incluida."""
    mixtura = _mixtura_exacta()
    oraculo = MixtureOracle(mixtura, make_sde("vp"))
    tiempos = [0.05, 0.9]
    t = torch.tensor(tiempos, dtype=torch.float64)

    covs = oraculo.component_covariances(t)

    assert covs.shape == (2, 2, 2, 2)
    assert covs.dtype is torch.float64
    bases = torch.as_tensor(mixtura.covariances_, dtype=torch.float64)
    identidad = torch.eye(2, dtype=torch.float64)
    for i, valor in enumerate(tiempos):
        a, s = _alpha_sigma_cerrados("vp", valor)
        esperado = a**2 * bases + s**2 * identidad
        assert torch.allclose(covs[i], esperado, rtol=1e-10, atol=1e-14)


def test_component_covariances_conserva_la_rotacion_de_la_componente():
    """Sumar ``sigma_t² I`` no diagonaliza: la componente rotada sigue correlacionada.

    Discrimina la versión que se queda con la diagonal de ``Sigma_k`` en lugar de la matriz
    completa, que daría un término fuera de la diagonal nulo.
    """
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("ve"))
    covs = oraculo.component_covariances(torch.tensor([0.3], dtype=torch.float64))

    fuera_de_diagonal = float(covs[0, 0, 0, 1])
    assert abs(fuera_de_diagonal) > 0.1
    assert float(covs[0, 0, 1, 0]) == pytest.approx(fuera_de_diagonal, rel=1e-12)


def test_component_covariances_devuelve_un_tensor_nuevo_en_cada_llamada():
    """Invariante de pureza: escribir en la salida no contamina las llamadas siguientes."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    t = torch.tensor([0.4], dtype=torch.float64)

    primera = oraculo.component_covariances(t).clone()
    contaminada = oraculo.component_covariances(t)
    contaminada[...] = 0.0

    assert torch.equal(oraculo.component_covariances(t), primera)


@pytest.mark.parametrize("nombre", _SDES_ESCALARES)
def test_component_covariances_es_simetrica_y_definida_positiva(nombre: str):
    """``Sigma_k(t)`` hereda la definición positiva de la mixtura para todo ``t``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde(nombre))
    covs = oraculo.component_covariances(torch.tensor(_TIEMPOS, dtype=torch.float64))

    assert torch.allclose(covs, covs.transpose(-1, -2), rtol=0.0, atol=1e-15)
    autovalores = torch.linalg.eigvalsh(covs)
    assert bool((autovalores > 0.0).all())


# ---------------------------- oráculo: inversa y determinante de forma cerrada


def test_la_forma_cerrada_2x2_coincide_con_el_algebra_lineal_generica():
    """La inversa y el determinante 2×2 se calculan a mano; acá se contrastan con el solver.

    El solver genérico es la fuente independiente: la producción **no** lo usa, justamente
    porque en 2×2 la adjunta sobre el determinante es exacta y barata.
    """
    from diffusion.analytic.mixture_oracle import _inverse_and_det_2x2

    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("sub_vp"))
    covs = oraculo.component_covariances(torch.tensor([0.05, 0.8], dtype=torch.float64))

    inv, det = _inverse_and_det_2x2(covs)

    assert inv.shape == covs.shape
    assert det.shape == covs.shape[:-2]
    assert torch.allclose(inv, torch.linalg.inv(covs), rtol=1e-10, atol=1e-14)
    assert torch.allclose(det, torch.linalg.det(covs), rtol=1e-10, atol=1e-14)
    producto = inv @ covs
    esperado = torch.eye(2, dtype=torch.float64).expand_as(producto)
    assert torch.allclose(producto, esperado, rtol=0.0, atol=1e-10)


def test_la_forma_cerrada_2x2_no_transpone_la_matriz():
    """Con una matriz **no simétrica** un error de signo o de transposición se ve.

    Las covarianzas son simétricas, así que confundir la adjunta con su transpuesta pasaría
    inadvertido si solo se probara con ellas.
    """
    from diffusion.analytic.mixture_oracle import _inverse_and_det_2x2

    m = torch.tensor([[[1.0, 2.0], [0.5, 3.0]]], dtype=torch.float64)

    inv, det = _inverse_and_det_2x2(m)

    assert float(det[0]) == pytest.approx(1.0 * 3.0 - 2.0 * 0.5, rel=1e-15)
    assert torch.allclose(inv, torch.linalg.inv(m), rtol=1e-12, atol=1e-15)


def test_la_forma_cerrada_2x2_rechaza_una_matriz_con_determinante_no_positivo():
    """Sin determinante positivo no hay covarianza que invertir: se avisa, no se da NaN."""
    from diffusion.analytic.mixture_oracle import _inverse_and_det_2x2

    singular = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    with pytest.raises(ValueError, match=r"^cov"):
        _inverse_and_det_2x2(singular)


def test_la_forma_cerrada_2x2_rechaza_una_forma_que_no_es_2x2():
    """La forma cerrada es específica de 2×2: cualquier otra forma es un error del caller."""
    from diffusion.analytic.mixture_oracle import _inverse_and_det_2x2

    with pytest.raises(ValueError, match=r"^cov"):
        _inverse_and_det_2x2(torch.eye(3, dtype=torch.float64))


# ------------------------------------ oráculo: rechazos en construcción (5.1, 5.2)


def test_rechaza_una_mixtura_legacy_sin_parametros_exactos():
    """La mixtura legacy no publica pesos, medias ni covarianzas verdaderos (criterio 5.1)."""
    with pytest.raises(ValueError, match=r"^mixture debe ser una ExactGaussianMixture"):
        MixtureOracle(GaussianMixture(2, seed=1), make_sde("vp"))


def test_rechaza_una_distribucion_que_no_es_una_mixtura():
    """Cualquier otra forma de datos tampoco tiene parámetros en forma cerrada."""
    from diffusion.data_generation import make_distribution

    with pytest.raises(ValueError, match=r"^mixture debe ser una ExactGaussianMixture"):
        MixtureOracle(make_distribution("two_moons", 2, seed=0), make_sde("vp"))


def test_rechaza_la_estandarizacion_empirica_de_la_mixtura_legacy():
    """El caso real (criterio 5.2): la config legacy del laboratorio usa ``standardize=True``.

    El rechazo por estandarización se evalúa **antes** que el de exactitud, así que el
    mensaje que ve el autor explica lo que de verdad rompe la exactitud: la transformación
    se estimó del sorteo.
    """
    legacy = GaussianMixture(2, standardize=True, seed=1)
    with pytest.raises(ValueError, match=r"^mixture no puede usar la estandarización") as info:
        MixtureOracle(legacy, make_sde("vp"))
    assert "sorteo" in str(info.value)


def test_rechaza_una_mixtura_exacta_a_la_que_le_prendieron_la_estandarizacion():
    """El flag se consulta en el objeto recibido, no se asume por su tipo."""
    mixtura = _mixtura_exacta()
    mixtura.standardize = True
    with pytest.raises(ValueError, match=r"^mixture no puede usar la estandarización"):
        MixtureOracle(mixtura, make_sde("vp"))


def test_rechaza_una_sde_cuya_media_no_es_proporcional_al_dato():
    """Familia escalar-gaussiana: ``mean == alpha_t · x_0`` con un escalar por muestra."""
    with pytest.raises(NotImplementedError, match=r"^sde no pertenece") as info:
        MixtureOracle(_mixtura_exacta(), _SDEDiagonal())
    assert "proporcional" in str(info.value)


def test_rechaza_una_sde_cuya_media_mezcla_las_coordenadas():
    """El sondeo usa coordenadas distintas entre sí, así que un kernel que las promedia cae.

    Con un vector de unos este caso pasaría el chequeo de proporcionalidad sin ser de la
    familia: el test le pone dientes a la elección de los puntos de sondeo.
    """
    with pytest.raises(NotImplementedError, match=r"^sde no pertenece") as info:
        MixtureOracle(_mixtura_exacta(), _SDEMezcladora())
    assert "proporcional" in str(info.value)


def test_rechaza_una_sde_cuyo_desvio_no_colapsa_a_un_escalar_por_muestra():
    """El desvío por coordenada delata un kernel que no es ``N(alpha x_0, sigma² I)``."""
    with pytest.raises(NotImplementedError, match=r"^sde no pertenece") as info:
        MixtureOracle(_mixtura_exacta(), _SDEPorDimension())
    assert "escalar por muestra" in str(info.value)


def test_la_pertenencia_a_la_familia_no_se_consulta_por_un_atributo():
    """El chequeo es **estructural**: ``is_augmented`` ya no existe en el paquete.

    Guard de compatibilidad hacia adelante: ninguna de las tres SDEs registradas cae en el
    caso, así que la única forma de verificarlo es con una SDE inventada, y la única forma de
    detectarla es mirando lo que devuelve el contrato marginal.
    """
    for sde in (_SDEDiagonal(), _SDEPorDimension(), make_sde("vp")):
        assert not hasattr(sde, "is_augmented")


@pytest.mark.parametrize(
    "caso",
    [
        "mixtura_legacy",
        "mixtura_estandarizada",
        "sde_diagonal",
        "sde_mezcladora",
        "sde_por_dimension",
    ],
)
def test_las_entradas_inadmisibles_fallan_en_construccion_y_no_al_integrar(caso: str):
    """Fail-fast: ninguna corrida "exacta" arranca sobre premisas que no lo son.

    La lista ``construidos`` es lo que le da dientes al test: con validación diferida el
    constructor devolvería un oráculo, no habría excepción y ``pytest.raises`` fallaría.
    """
    mixtura: object = _mixtura_exacta()
    sde: ForwardSDE = make_sde("vp")
    if caso == "mixtura_legacy":
        mixtura = GaussianMixture(2, seed=1)
    elif caso == "mixtura_estandarizada":
        mixtura = GaussianMixture(2, standardize=True, seed=1)
    elif caso == "sde_diagonal":
        sde = _SDEDiagonal()
    elif caso == "sde_mezcladora":
        sde = _SDEMezcladora()
    else:
        sde = _SDEPorDimension()

    construidos: list[MixtureOracle] = []
    with pytest.raises((ValueError, NotImplementedError)):
        construidos.append(MixtureOracle(mixtura, sde))  # type: ignore[arg-type]

    assert construidos == []


def test_la_validacion_de_familia_consulta_el_contrato_marginal_en_construccion():
    """Contracara del fail-fast: el sondeo estructural ya corrió al construir el oráculo."""
    sde = _SDEContadora()
    MixtureOracle(_mixtura_exacta(), sde)
    assert sde.consultas >= 1


# --------------------------------------- oráculo: validación de la entrada de tiempo


@pytest.mark.parametrize("metodo", ["marginal_params", "component_covariances"])
@pytest.mark.parametrize(
    "malo",
    [
        torch.zeros((), dtype=torch.float64),
        torch.zeros(3, 2, dtype=torch.float64),
        torch.zeros(2, 1, 1, dtype=torch.float64),
        torch.zeros(0, dtype=torch.float64),
    ],
)
def test_rechaza_un_tiempo_con_forma_invalida(metodo: str, malo: "torch.Tensor"):
    """``t`` es un valor por punto: ``(B,)`` o ``(B, 1)`` con ``B >= 1``."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        getattr(oraculo, metodo)(malo)


def test_rechaza_un_tiempo_que_no_es_tensor():
    """El contrato del proyecto es tensorial; un float suelto es un error del caller."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.marginal_params(0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("valor", [-1e-6, float("nan"), float("inf")])
def test_rechaza_un_tiempo_negativo_o_no_finito(valor: float):
    """El proceso corre en ``[0, T]``: fuera de ahí el schedule no significa nada."""
    oraculo = MixtureOracle(_mixtura_exacta(), make_sde("vp"))
    with pytest.raises(ValueError, match=r"^t debe"):
        oraculo.marginal_params(torch.tensor([valor], dtype=torch.float64))
