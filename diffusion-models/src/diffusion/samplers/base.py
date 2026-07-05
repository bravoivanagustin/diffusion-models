"""Base abstracta de los samplers del proceso reverso (Eje 2).

Un :class:`ReverseSampler` integra numéricamente la ecuación reversa de Anderson (1982)

    ``dx = [f(x,t) - g(t)^2 ∇_x log p_t(x)] dt + g(t) dW̄``

—o su flujo de probabilidad determinístico (PF-ODE)— para generar muestras ``x_0`` a
partir del prior de ruido ``p_T``, **reusando** el score aprendido ``s_θ(x,t)`` sin
reentrenar la red. Es el **Eje 2** del estudio de ablación (ver ``docs/project/ejes.md``):
cambiar el sampler reusa el mismo score; cambiar la SDE (Eje 1) sí obliga a reentrenar.

Patrón Template Method: este ABC fija el algoritmo de integración compartido (grilla
temporal uniforme, drifts reversos derivados de ``sde.sde`` y del score), y cada sampler
concreto define solo su :meth:`step`. La red se consume como función pura ``ScoreFn`` y la
estocasticidad vive en el sampler (EM/PC) o se anula (PF-ODE/Heun), nunca en la red.

Igual que :mod:`diffusion.sde`, este módulo importa **torch directamente** (opera sobre
tensores; torch es dependencia dura).
"""

from __future__ import annotations

import abc
from typing import Callable

import torch

from diffusion.sde import ForwardSDE

#: Contrato de inyección del score: ``(x: (B, data_dim), t: (B,) | (B,1)) -> (B, data_dim)``.
#: Tanto una :class:`diffusion.mlp.ScoreMLP` entrenada como un score analítico en forma
#: cerrada encajan sin cambios en el sampler.
ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ReverseSampler(abc.ABC):
    """Base de todos los samplers del proceso reverso.

    Un sampler concreto fija :attr:`name` (clave del futuro registry) e implementa
    :meth:`step`. El ABC aporta la grilla temporal uniforme de ``T`` a ``t_eps``, los dos
    drifts reversos compartidos (:meth:`_reverse_drift` para la SDE y :meth:`_pfode_drift`
    para el flujo de probabilidad) y la normalización temporal.

    La red se consume como función pura :attr:`score_fn` y **nunca** se muta.
    """

    #: Clave del registry/factory, p. ej. ``"euler"``. Sobreescribir en cada subclase.
    name: str = ""

    def __init__(
        self,
        sde: ForwardSDE,
        score_fn: ScoreFn,
        *,
        n_steps: int = 500,
        t_eps: float = 1e-3,
    ) -> None:
        """Inicializa el sampler.

        Args:
            sde: Proceso forward (Eje 1) del que se derivan los coeficientes ``(f, g)`` y
                el prior ``p_T``.
            score_fn: Función pura ``(x, t) -> score`` que aproxima ``∇_x log p_t(x)``.
            n_steps: Número de pasos (intervalos) de integración; ``>= 1``.
            t_eps: Tiempo terminal de la integración, un piso ``> 0`` que evita integrar
                hasta ``t = 0`` exacto; debe cumplir ``0 < t_eps < sde.T``.

        Raises:
            ValueError: Si ``n_steps < 1`` o ``t_eps`` cae fuera de ``(0, sde.T)``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps debe ser >= 1; recibí n_steps={n_steps}")
        if not (0.0 < t_eps < sde.T):
            raise ValueError(
                f"t_eps debe cumplir 0 < t_eps < sde.T={sde.T}; recibí t_eps={t_eps}"
            )
        self.sde = sde
        self.score_fn = score_fn
        self.n_steps = int(n_steps)
        self.t_eps = float(t_eps)

    # --------------------------------------------------------------- a implementar

    @abc.abstractmethod
    def step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Avanza un paso de integración de ``t`` a ``t + dt`` (con ``dt < 0``).

        Cada sampler concreto define su discretización; las subclases **no** recalculan la
        grilla (la maneja el driver).

        Args:
            x: Estado actual de shape ``(B, data_dim)``.
            t: Tiempo actual de shape ``(B,)`` o ``(B, 1)``.
            dt: Tamaño de paso (negativo: se integra en tiempo decreciente).
            generator: Generador de torch para los samplers estocásticos; los
                determinísticos (PF-ODE/Heun) lo ignoran.

        Returns:
            El nuevo estado de shape ``(B, data_dim)``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------- driver

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        *,
        init: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Integra el proceso reverso de ``T`` a ``t_eps`` y devuelve las muestras ``x_0``.

        Driver compartido (Template Method): arranca del prior ``p_T`` —o del ``init``
        provisto— y recorre la grilla temporal en tiempo **decreciente** (``dt < 0``),
        delegando cada paso en :meth:`step`. Corre bajo ``torch.no_grad()`` y en
        ``float32``, sin tocar los parámetros de la red (el score se consume como función
        pura), de modo que cambiar de sampler nunca reentrena (Eje 2).

        Args:
            n_samples: Número de muestras a generar (``N``).
            init: Estado inicial ``x_T`` de shape ``(n_samples, sde.data_dim)``. Si es
                ``None`` se sortea de ``sde.prior_sampling``; pasarlo aísla el determinismo
                del integrador del muestreo del prior.
            generator: Generador de torch para reproducibilidad; alimenta tanto el muestreo
                del prior como los pasos estocásticos (EM/PC). Los samplers determinísticos
                (PF-ODE/Heun) lo ignoran en :meth:`step`.
            return_trajectory: Si es ``True``, devuelve además la trayectoria completa.

        Returns:
            El estado final ``x_0`` de shape ``(n_samples, sde.data_dim)`` en ``float32``.
            Si ``return_trajectory`` es ``True``, una tupla ``(x_0, trayectoria)`` donde la
            trayectoria tiene shape ``(n_steps + 1, n_samples, sde.data_dim)`` e incluye el
            estado inicial ``x_T`` (capa ``0``) y cada estado intermedio.
        """
        if init is None:
            x = self.sde.prior_sampling(
                (n_samples, self.sde.data_dim), generator=generator, dtype=torch.float32
            )
        else:
            x = init.to(dtype=torch.float32)

        grid = self._time_grid()
        trajectory: list[torch.Tensor] = [x.clone()] if return_trajectory else []

        for i in range(self.n_steps):
            t_cur = grid[i]
            t_next = grid[i + 1]
            dt = (t_next - t_cur).item()  # negativo: tiempo decreciente
            t_batch = t_cur.expand(n_samples, 1)
            x = self.step(x, t_batch, dt, generator=generator)
            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory, dim=0)
        return x

    # ----------------------------------------------------------- helpers compartidos

    def _time_grid(self) -> torch.Tensor:
        """Grilla temporal uniforme de ``T`` a ``t_eps``.

        Returns:
            Tensor ``float32`` de shape ``(n_steps + 1,)``, decreciente, con extremos
            ``T`` (primero) y ``t_eps`` (último).
        """
        return torch.linspace(self.sde.T, self.t_eps, self.n_steps + 1, dtype=torch.float32)

    def _reverse_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift de la SDE reversa ``f - g^2 s``.

        Args:
            x: Estado de shape ``(B, data_dim)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, data_dim)``.
        """
        t = self._expand_t(t)
        f, g = self.sde.sde(x, t)
        s = self.score_fn(x, t)
        return f - (g ** 2) * s

    def _pfode_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift del flujo de probabilidad (PF-ODE) ``f - ½ g^2 s``.

        Comparte las mismas marginales que la SDE reversa pero sin término de ruido.

        Args:
            x: Estado de shape ``(B, data_dim)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, data_dim)``.
        """
        t = self._expand_t(t)
        f, g = self.sde.sde(x, t)
        s = self.score_fn(x, t)
        return f - 0.5 * (g ** 2) * s

    # ----------------------------------------------------------------- internos

    @staticmethod
    def _expand_t(t: torch.Tensor) -> torch.Tensor:
        """Normaliza ``t`` de shape ``(B,)`` o ``(B, 1)`` a ``(B, 1)`` para broadcast."""
        return t.reshape(-1, 1)

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        return (
            f"{type(self).__name__}(sde={type(self.sde).__name__}, "
            f"n_steps={self.n_steps}, t_eps={self.t_eps})"
        )
