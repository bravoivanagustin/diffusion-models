"""Oráculo exacto de una mixtura gaussiana 2D ruideada por una SDE escalar-gaussiana.

Un kernel forward gaussiano **preserva la familia**: si los datos son
``p_0 = Σ_k w_k N(μ_k, Σ_k)`` y el kernel de perturbación es ``N(α_t x_0, σ_t² I)``, la
marginal ruideada sigue siendo una mixtura de gaussianas,

``p_t = Σ_k w_k N(α_t μ_k, Σ_k(t))``  con  ``Σ_k(t) = α_t² Σ_k + σ_t² I``,

y de ahí salen en forma cerrada la densidad, la log-densidad y el score. Este módulo
implementa esa propagación: es el **patrón de referencia** contra el que se mide el resto
del laboratorio 2D.

De dónde salen ``α_t`` y ``σ_t`` (la decisión de diseño central)
---------------------------------------------------------------

**Del contrato marginal que la SDE ya publica**, nunca re-derivando schedules ni
ramificando por variante. Las tres SDEs escalares del Eje 1 cumplen ``mean == α_t · x_0``
elementwise, así que evaluar ``marginal_prob`` sobre un vector de unos devuelve ``α_t`` en
la media y ``σ_t`` en el desvío — el mismo truco que ya usa la suite de samplers. Un único
camino de código cubre VP, VE y sub-VP (y cualquier SDE escalar-gaussiana futura), y queda
automáticamente consistente con lo que el entrenamiento realmente usó.

Dos convenciones del paquete que se consumen **tal cual**:

- **sub-VP publica su desvío sin raíz**: ``marginal_prob`` devuelve ``std = 1 - e^{-∫β}``,
  y la varianza del kernel es ese valor al cuadrado. No hay que "corregirlo" tomándole una
  raíz: con ``t = 0.05`` el desvío vale ``0.0294`` y su raíz ``0.1716``, casi seis veces
  más, así que el error sería grosero y silencioso.
- **VE no contrae la media**: su ``α_t`` es exactamente ``1``, porque su kernel deja el
  dato en su lugar y solo le apila ruido.

Familia admitida y cómo se verifica
-----------------------------------

Solo la **familia escalar-gaussiana** (kernel ``N(α_t x_0, σ_t² I)`` con ``α_t``/``σ_t``
escalares por muestra). La pertenencia se verifica **estructuralmente** sobre lo que
devuelve el contrato marginal —la media es proporcional a ``x_0`` y el desvío colapsa a un
escalar por muestra—, no consultando ningún atributo de familia: ``is_augmented`` **ya no
existe** en el paquete (era de CLD, eliminado del proyecto el 05/07/2026) y agregar uno
estaría fuera de la frontera de este módulo. Hoy ninguna de las tres SDEs registradas cae
en el rechazo: es un guard de compatibilidad hacia adelante, no una rama con usuario
actual.

Precisión: los parámetros de la mixtura se guardan en **doble** precisión y los pasos
intermedios (el armado de ``Σ_k(t)``, su inversa, las responsabilidades) se calculan ahí,
con independencia de que el caller trabaje en ``float32``. La degradación al dtype del
caller ocurre recién en la salida de las cantidades que consumen los samplers.
"""

from __future__ import annotations

import torch

from ..data_generation import ExactGaussianMixture
from ..sde import ForwardSDE

#: Puntos de sondeo del chequeo estructural de familia. Sus coordenadas son **no nulas**
#: (para poder despejar ``α_t`` dividiendo) y **distintas entre sí**: con un vector de unos
#: cualquier kernel que **mezclara** coordenadas —por ejemplo uno que las promediara—
#: devolvería una media que igual parece proporcional a ``x_0``, y el rechazo se perdería.
_PROBE_X0: tuple[tuple[float, float], ...] = ((1.0, 2.0), (3.0, -1.5))

#: Fracciones del horizonte en las que se sondea el contrato marginal.
_PROBE_FRACTIONS: tuple[float, ...] = (0.25, 0.75)

#: Tolerancia relativa con la que se exige ``mean == α_t · x_0`` en el sondeo. Holgada
#: respecto del ruido de la doble precisión, pero órdenes de magnitud por debajo de
#: cualquier escala por coordenada que valga la pena rechazar.
_PROPORTIONALITY_TOL: float = 1e-10


class MixtureOracle:
    """Verdad exacta de una mixtura 2D ruideada por una SDE escalar-gaussiana.

    Determinístico, sin estado mutable y sin necesidad de gradientes: no guarda cantidades
    derivadas entre llamadas (así no hay cachés que se desincronicen si el caller cambia el
    horizonte) y no muta ni la mixtura ni la SDE que recibe.

    La admisibilidad se valida **una vez, en construcción** (ver :meth:`__init__`), de modo
    que una corrida no arranque sobre premisas falsas y falle recién al integrar.

    Note:
        El oráculo describe la mixtura **exacta**. Los modelos entrenados sobre datos con
        estandarización empírica no quedan descritos por él: en ese caso la transformación
        se estima a partir de la muestra sorteada, así que los parámetros efectivos
        dependen de ``n`` y de la semilla. Las corridas del laboratorio 2D deben usar
        mixturas de parámetros exactos.
    """

    def __init__(self, mixture: ExactGaussianMixture, sde: ForwardSDE) -> None:
        """Construye el oráculo validando que la mixtura y la SDE sean admisibles.

        Los parámetros verdaderos de la mixtura se copian a tensores de doble precisión:
        el oráculo es dueño de su copia, así que nada de lo que devuelva puede
        desincronizarse de lo que se validó acá.

        Args:
            mixture: Mixtura de parámetros exactos (pesos, medias y covarianzas conocidos
                en forma cerrada).
            sde: Proceso forward de la familia escalar-gaussiana. Solo se le consulta el
                contrato marginal (``marginal_prob``) y el horizonte (``T``).

        Raises:
            ValueError: Si ``mixture`` pide la estandarización empírica, o si no es una
                mixtura de parámetros exactos. En los dos casos el resultado no sería
                exacto, que es la premisa del módulo.
            NotImplementedError: Si ``sde`` no pertenece a la familia escalar-gaussiana,
                verificado de forma estructural sobre lo que devuelve su contrato marginal.
        """
        if getattr(mixture, "standardize", False):
            raise ValueError(
                "mixture no puede usar la estandarización empírica (standardize=True): "
                "esa transformación se estima a partir de la muestra sorteada, así que los "
                "parámetros efectivos dependen del sorteo (de n y de la semilla) y el "
                "resultado no sería exacto. Construí una ExactGaussianMixture con los "
                f"parámetros que querés y standardize=False; recibí "
                f"{type(mixture).__name__} con standardize=True."
            )
        if not isinstance(mixture, ExactGaussianMixture):
            raise ValueError(
                "mixture debe ser una ExactGaussianMixture, la única fuente de datos que "
                "publica sus pesos, medias y covarianzas verdaderos en forma cerrada; "
                f"recibí {type(mixture).__name__}, cuyos parámetros no son consultables, "
                "así que ninguna cantidad derivada de ellos sería exacta."
            )

        self._mixture = mixture
        self._sde = sde

        # Parámetros verdaderos en doble precisión: los accesores de la mixtura ya
        # devuelven copias, así que estos tensores no aliasan nada del caller.
        self._weights = torch.as_tensor(mixture.weights_, dtype=torch.float64)
        self._means = torch.as_tensor(mixture.means_, dtype=torch.float64)
        self._covariances = torch.as_tensor(mixture.covariances_, dtype=torch.float64)

        self._validate_scalar_gaussian_family()

    # ------------------------------------------------- parámetros del kernel en t

    def marginal_params(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Factor de contracción ``α_t`` y desvío del kernel ``σ_t`` en cada tiempo.

        Los lee del contrato marginal de la SDE evaluándolo sobre un vector de unos, que
        para la familia escalar-gaussiana devuelve ``mean == α_t`` y ``std == σ_t``. No
        re-deriva ningún schedule y no ramifica por variante: el mismo camino de código
        sirve para VP, VE y sub-VP.

        El sondeo se arma en doble precisión, así que el schedule se evalúa en doble
        aunque ``t`` llegue en ``float32``.

        Args:
            t: Tiempos, de forma ``(B,)`` o ``(B, 1)``, finitos y no negativos.

        Returns:
            El par ``(alpha, sigma)``, cada uno de forma ``(B, 1)`` en ``float64`` y en el
            device de ``t``. ``sigma`` es el desvío **tal como lo publica la SDE**: para
            sub-VP eso es ``1 - e^{-∫β}``, sin raíz.

        Raises:
            ValueError: Si ``t`` no es un tensor de forma ``(B,)`` o ``(B, 1)`` con
                ``B >= 1``, o si tiene valores no finitos o negativos.
        """
        tt = _normalize_time(t)
        batch = tt.shape[0]
        ones = torch.ones(batch, 2, dtype=torch.float64, device=tt.device)
        mean, std = self._sde.marginal_prob(ones, tt)
        return _as_column(mean, batch), _as_column(std, batch)

    def component_covariances(self, t: torch.Tensor) -> torch.Tensor:
        """Covarianzas de las componentes ya propagadas al tiempo ``t``.

        Es ``Σ_k(t) = α_t² Σ_k + σ_t² I``: el kernel escala la covarianza de los datos por
        ``α_t²`` y le suma ruido isotrópico. Cada llamada arma un tensor nuevo, así que
        escribir en la salida no contamina las llamadas siguientes.

        Args:
            t: Tiempos, de forma ``(B,)`` o ``(B, 1)``, finitos y no negativos.

        Returns:
            Tensor ``(B, K, 2, 2)`` en ``float64`` y en el device de ``t``, simétrico y
            definido positivo por construcción.

        Raises:
            ValueError: Si ``t`` no cumple el contrato de :meth:`marginal_params`.
        """
        alpha, sigma = self.marginal_params(t)
        base = self._covariances.to(device=alpha.device)  # (K, 2, 2)
        identidad = torch.eye(2, dtype=torch.float64, device=alpha.device)
        alpha2 = (alpha**2).reshape(-1, 1, 1, 1)  # (B, 1, 1, 1)
        sigma2 = (sigma**2).reshape(-1, 1, 1, 1)
        return alpha2 * base + sigma2 * identidad

    # ------------------------------------------------------------------ internos

    def _validate_scalar_gaussian_family(self) -> None:
        """Verifica estructuralmente que la SDE sea de la familia escalar-gaussiana.

        Sondea el contrato marginal con puntos de coordenadas distintas y en dos tiempos, y
        exige las dos propiedades que definen al kernel ``N(α_t x_0, σ_t² I)``: que la media
        sea proporcional a ``x_0`` con **un solo escalar por muestra**, y que el desvío
        colapse a un escalar por muestra en lugar de traer un valor por coordenada o una
        matriz.

        No consulta ningún atributo de familia: ``is_augmented`` ya no existe en el paquete
        y agregarlo estaría fuera de la frontera de este módulo.

        Raises:
            NotImplementedError: Si el contrato marginal no cumple alguna de las dos
                propiedades.
        """
        x0 = torch.tensor(_PROBE_X0, dtype=torch.float64)
        horizonte = float(self._sde.T)
        t = torch.tensor(
            [f * horizonte for f in _PROBE_FRACTIONS], dtype=torch.float64
        ).reshape(-1, 1)
        mean, std = self._sde.marginal_prob(x0, t)

        if not _collapses_to_scalar_per_sample(std, x0.shape[0]):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal "
                f"devolvió un desvío de forma {tuple(std.shape)}, que no colapsa a un "
                f"escalar por muestra para un lote de {x0.shape[0]}. Este oráculo solo "
                "describe kernels N(alpha_t x_0, sigma_t^2 I); una SDE con desvío por "
                "coordenada o con estado aumentado necesita su propia forma cerrada."
            )
        if tuple(mean.shape) != tuple(x0.shape):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal "
                f"devolvió una media de forma {tuple(mean.shape)} para un dato de forma "
                f"{tuple(x0.shape)}."
            )
        alpha = mean[:, :1] / x0[:, :1]
        if not torch.allclose(mean, alpha * x0, rtol=_PROPORTIONALITY_TOL, atol=1e-12):
            raise NotImplementedError(
                "sde no pertenece a la familia escalar-gaussiana: su contrato marginal no "
                "devolvió una media proporcional a x_0 (se espera mean == alpha_t * x_0 "
                "con alpha_t escalar por muestra, que es lo que hace recuperable alpha_t "
                "sondeando con un vector de unos)."
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"{type(self).__name__}(K={self._weights.shape[0]}, "
            f"sde={type(self._sde).__name__})"
        )


# ---------------------------------------------------------------- helpers numéricos


def _normalize_time(t: torch.Tensor) -> torch.Tensor:
    """Normaliza ``t`` a ``(B, 1)`` en doble precisión, validando el contrato de shapes.

    Sigue la convención del proyecto (``t`` aceptado como ``(B,)`` o ``(B, 1)``) y lleva el
    tiempo a doble precisión **antes** de que la SDE evalúe su schedule, para que los pasos
    intermedios no hereden la precisión del caller.

    Args:
        t: Tiempos candidatos.

    Returns:
        ``t`` como tensor ``(B, 1)`` ``float64``, desconectado del grafo, en su device.

    Raises:
        ValueError: Si ``t`` no es un tensor de torch, si su forma no es ``(B,)`` ni
            ``(B, 1)`` con ``B >= 1``, o si tiene valores no finitos o negativos.
    """
    if not isinstance(t, torch.Tensor):
        raise ValueError(
            "t debe ser un tensor de torch de forma (B,) o (B, 1) con un tiempo por "
            f"punto; recibí un objeto de tipo {type(t).__name__}."
        )
    forma_valida = t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1)
    if not forma_valida or t.shape[0] < 1:
        raise ValueError(
            "t debe tener forma (B,) o (B, 1) con B >= 1 (un tiempo por punto); recibí "
            f"forma {tuple(t.shape)}."
        )
    tt = t.detach().to(torch.float64).reshape(t.shape[0], 1)
    if not bool(torch.isfinite(tt).all()) or bool((tt < 0.0).any()):
        raise ValueError(
            "t debe ser finito y no negativo: el proceso corre en [0, T] y fuera de ese "
            "rango el schedule de la SDE no está definido."
        )
    return tt


def _as_column(value: torch.Tensor, batch: int) -> torch.Tensor:
    """Colapsa un escalar por muestra a la forma canónica ``(batch, 1)``.

    El contrato marginal devuelve escalares por muestra con las dimensiones de evento en
    uno (``(B, 1)`` en 2D), y una SDE puede devolverlos ya broadcasteados desde una sola
    fila. Esta función unifica esos casos sin asumir la forma exacta.

    Args:
        value: Tensor con un escalar por muestra: ``(B, 1, …, 1)``, ``(1, 1, …, 1)`` o de
            rango cero.
        batch: Cantidad de muestras esperada.

    Returns:
        Tensor de forma ``(batch, 1)``.
    """
    if value.ndim == 0:
        return value.reshape(1, 1).expand(batch, 1)
    columna = value.reshape(value.shape[0], -1)[:, :1]
    if columna.shape[0] != batch:
        columna = columna.expand(batch, 1)
    return columna


def _collapses_to_scalar_per_sample(value: torch.Tensor, batch: int) -> bool:
    """Indica si ``value`` es un escalar por muestra (y no un valor por coordenada).

    Es el criterio estructural con el que se reconoce el desvío de un kernel isotrópico:
    todas las dimensiones más allá de la del lote tienen que ser uno, y la del lote tiene
    que ser la del dato o uno (broadcast).

    Args:
        value: Tensor a inspeccionar.
        batch: Cantidad de muestras del sondeo.

    Returns:
        ``True`` si describe un escalar por muestra.
    """
    if value.ndim == 0:
        return True
    if value.shape[0] not in (1, batch):
        return False
    return all(d == 1 for d in value.shape[1:])


def _inverse_and_det_2x2(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inversa y determinante de matrices ``2×2`` en forma cerrada.

    En dos dimensiones no hace falta un solver genérico: el determinante es ``ad - bc`` y la
    inversa es la adjunta ``[[d, -b], [-c, a]]`` dividida por él. Es exacto, barato, se
    vectoriza sobre cualquier cantidad de dimensiones de lote y —a diferencia de una
    factorización— no introduce ramas dependientes de los datos.

    Está pensada para las covarianzas ``Σ_k(t)``, que son definidas positivas, así que
    exige determinante positivo en lugar de solamente no nulo.

    Args:
        cov: Tensor de forma ``(…, 2, 2)``.

    Returns:
        El par ``(inversa, determinante)`` con formas ``(…, 2, 2)`` y ``(…)``.

    Raises:
        ValueError: Si la forma no termina en ``(2, 2)``, o si algún determinante no es
            finito y positivo (una matriz singular o no definida positiva no es una
            covarianza válida y devolver infinitos sería peor que avisar).
    """
    if cov.ndim < 2 or cov.shape[-2:] != (2, 2):
        raise ValueError(
            f"cov debe tener forma (..., 2, 2); recibida {tuple(cov.shape)}."
        )
    a = cov[..., 0, 0]
    b = cov[..., 0, 1]
    c = cov[..., 1, 0]
    d = cov[..., 1, 1]
    det = a * d - b * c
    if not bool(torch.isfinite(det).all()) or bool((det <= 0.0).any()):
        raise ValueError(
            "cov debe tener determinante finito y positivo para ser invertible como "
            f"covarianza; el mínimo encontrado fue {float(det.min())!r}."
        )
    fila0 = torch.stack((d, -b), dim=-1)
    fila1 = torch.stack((-c, a), dim=-1)
    adjunta = torch.stack((fila0, fila1), dim=-2)
    return adjunta / det[..., None, None], det
