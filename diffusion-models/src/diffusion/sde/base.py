"""Procesos forward (SDEs) que destruyen los datos hacia ruido.

Cada SDE define ``dx = f(x, t) dt + g(t) dW`` —el proceso que ruidea un ``x_0`` de
``data_generation`` para fabricar el par de entrenamiento ``x_t``— y, sobre todo, el
**target del score** :math:`\\nabla_x \\log p_t(x_t \\mid x_0)` que la red
(:class:`diffusion.models.ScoreMLP`) debe aprender. Es el **Eje 1** del estudio de ablación
(ver ``docs/project/ejes.md``): cambiar la SDE cambia ``p_t`` y por lo tanto exige
reentrenar; cambiar el sampler (Eje 2) reusa el mismo score.

La SDE es una de las piezas donde **sí vive la estocasticidad** del pipeline (junto con el
dato y el sampler reverso), por contraste con la red, que es determinística.

A diferencia de ``data_generation`` —cuyo core es numpy con torch diferido—, este módulo
importa **torch directamente**: opera sobre tensores, produce los pares de entrenamiento y
alimenta los samplers, así que torch es dependencia dura (igual que ``models``).

Clase base abstracta acá; las variantes escalar-gaussianas (VP/VE/sub-VP) viven en
:mod:`variants`; el registry/factory en :mod:`__init__`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ForwardSDE(ABC):
    """Base de todos los procesos forward.

    Una SDE concreta fija :attr:`name` (clave del registry), recibe :attr:`data_dim` en el
    constructor (anda en cualquier dimensión; 2 por defecto) e implementa los tres métodos
    abstractos :meth:`sde`, :meth:`marginal_prob` y :meth:`prior_sampling`.

    Para la **familia escalar-gaussiana** (VP/VE/sub-VP) el kernel de perturbación es
    ``p_t(x_t | x_0) = N(mean, std^2 I)`` con ``std`` escalar por muestra; por eso
    :meth:`perturb` y :meth:`score_target` son **concretos** acá y se derivan enteramente
    de :meth:`marginal_prob`.
    """

    #: Clave usada en el registry y la factory. Sobreescribir en cada subclase.
    name: str = ""

    #: Piso para ``std`` antes de dividir (evita división por cero en ``t -> 0``).
    _std_eps: float = 1e-5

    def __init__(self, data_dim: int = 2, T: float = 1.0) -> None:
        """Inicializa la SDE.

        Args:
            data_dim: Dimensión del estado (= la que aprende la red). Configurable y anda
                en cualquier dimensión: ``2`` por defecto para datos 2D.
            T: Horizonte temporal. El proceso corre en ``t in [0, T]``.

        Raises:
            ValueError: Si ``data_dim`` es menor que 1.
        """
        if data_dim < 1:
            raise ValueError(f"data_dim debe ser >= 1; recibí data_dim={data_dim}")
        self.data_dim = int(data_dim)
        self.T = float(T)

    # ------------------------------------------------------------- a implementar

    @abstractmethod
    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Coeficientes ``(drift, diffusion)`` de ``dx = f(x,t) dt + g(t) dW``.

        Args:
            x: Estado de shape ``(B, data_dim)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            ``(drift, diffusion)`` con ``drift`` de shape ``(B, data_dim)`` y
            ``diffusion`` de shape ``(B, 1)``.
        """
        raise NotImplementedError

    @abstractmethod
    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Media y desvío del kernel de perturbación ``p_t(x_t | x_0)``.

        Para la familia escalar-gaussiana el kernel es ``N(mean, std^2 I)``.

        Args:
            x0: Dato limpio de shape ``(B, data_dim)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            ``(mean, std)`` con ``mean`` de shape ``(B, data_dim)`` y ``std`` de shape
            ``(B, 1)`` (se broadcastea sobre las dimensiones).
        """
        raise NotImplementedError

    @abstractmethod
    def prior_sampling(
        self,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Muestrea de la distribución terminal ``p_T`` (el prior del sampler).

        Args:
            shape: Shape de la salida, p. ej. ``(B, data_dim)``.
            generator: Generador de torch opcional para reproducibilidad.
            device: Dispositivo de la salida.
            dtype: Tipo de la salida (default ``float32``).

        Returns:
            Tensor de shape ``shape`` muestreado de ``p_T``.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------- API

    def perturb(
        self, x0: torch.Tensor, t: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Muestrea ``x_t`` del kernel de perturbación y devuelve el ruido usado.

        Implementación para la familia escalar-gaussiana:
        ``x_t = mean + std * eps`` con ``eps ~ N(0, I)``.

        Args:
            x0: Dato limpio de shape ``(B, data_dim)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.
            generator: Generador opcional para reproducibilidad.

        Returns:
            ``(x_t, eps)``, ambos de shape ``(B, data_dim)``.
        """
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn(
            x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
        )
        x_t = mean + std * eps
        return x_t, eps

    def score_target(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score real del kernel y peso de la pérdida para denoising score matching.

        Para la familia escalar-gaussiana, con ``x_t = mean + std * eps``::

            ∇_{x_t} log p_t(x_t | x_0) = -(x_t - mean) / std^2 = -eps / std

        y el peso recomendado es ``lambda(t) = std^2`` (pesado tipo verosimilitud, que
        vuelve la pérdida equivalente a ``|| std * s_theta + eps ||^2``).

        Args:
            x0: Dato limpio de shape ``(B, data_dim)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.
            eps: Ruido usado en :meth:`perturb`, shape ``(B, data_dim)``.

        Returns:
            ``(score_real, weight)`` con ``score_real`` de shape ``(B, data_dim)`` y
            ``weight`` de shape ``(B, 1)``.
        """
        _, std = self.marginal_prob(x0, t)
        std = std.clamp_min(self._std_eps)
        score_real = -eps / std
        weight = std ** 2
        return score_real, weight

    # ----------------------------------------------------------------- internos

    @staticmethod
    def _expand_t(t: torch.Tensor) -> torch.Tensor:
        """Normaliza ``t`` de shape ``(B,)`` o ``(B, 1)`` a ``(B, 1)`` para broadcast."""
        return t.reshape(-1, 1)

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return f"{type(self).__name__}(T={self.T})"
