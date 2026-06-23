r"""Critically-Damped Langevin Diffusion (CLD) — Dockhorn et al. (ICLR 2022).

A diferencia de la familia escalar (VP/VE/sub-VP), CLD **aumenta** el estado con un momento
:math:`v`: el estado es :math:`u = (x, v)` con ``data_dim = 2 * spatial_dim`` (``4`` para
datos 2D: ``(x_1, x_2, v_1, v_2)``). El **ruido entra solo por** :math:`v`, y la red aprende el score
sobre el momento :math:`\nabla_v \log p_t(v \mid x)` (que, para un gaussiano conjunto,
coincide con la componente-``v`` del score conjunto :math:`\nabla_u \log p_t(u)`).

Por dimensión espacial el proceso es un OU lineal 2D::

    dx = M^{-1} v dt
    dv = -(Γ M^{-1} v + β x) dt + sqrt(2Γ) dW

con amortiguamiento **crítico** ``Γ = 2 sqrt(β M)`` (autovalor doble ``λ = -sqrt(β/M)``).
Para un sistema lineal con coeficientes constantes el kernel ``p_t(u_t | x_0)`` es gaussiano
y se resuelve en forma cerrada vía la matriz de transición ``Φ(t) = exp(A t)`` (acá con
autovalor repetido, ``Φ(t) = e^{λt}[(1-λt) I + t A]``) más la integral de covarianza
``W(t) = ∫_0^t Φ(τ) G Gᵀ Φ(τ)ᵀ dτ``, todas con primitivas exactas. Los tests validan estas
fórmulas contra una simulación Monte Carlo (Euler–Maruyama) del forward.

Convención de la interfaz: por ser aumentada, ``x0`` es la **posición** (shape
``(B, spatial_dim)`` = ``(B, data_dim // 2)``, lo que entrega ``data_generation``); el
momento inicial se marginaliza (``v_0 ~ N(0, M I)``, equilibrio). :meth:`perturb` devuelve
el estado completo ``u_t`` (shape ``(B, data_dim)``) y :meth:`marginal_prob` devuelve
``(mean, L)`` con ``L`` el factor de Cholesky 2×2 por dimensión (no un ``std`` escalar), por
lo que CLD **sobreescribe** ``perturb``/``score_target``/``marginal_prob``/``sde``/``prior_sampling``.
"""

from __future__ import annotations

import math

import torch

from .base import ForwardSDE


class CLDSDE(ForwardSDE):
    """SDE de Langevin críticamente amortiguada con estado aumentado posición-momento.

    Defaults estilo Dockhorn et al.: ``beta=4.0``, ``mass=0.25`` → ``Γ=2``, ``λ=-4``
    (buena mezcla en ``t ∈ [0, 1]``). Distribución estacionaria (prior ``p_T``):
    ``x ~ N(0, 1/β)``, ``v ~ N(0, M)``, sin correlación.
    """

    name = "cld"
    is_augmented = True

    def __init__(
        self, beta: float = 4.0, mass: float = 0.25, data_dim: int = 4, T: float = 1.0
    ) -> None:
        """Inicializa la CLD-SDE.

        Args:
            beta: Coeficiente ``β`` (constante) del acople posición-momento.
            mass: Masa ``M`` del momento. El amortiguamiento crítico fija
                ``Γ = 2 sqrt(β M)``.
            data_dim: Dimensión del estado aumentado, debe ser **par**
                ``= 2 * spatial_dim`` (``4`` para datos 2D). La posición ocupa las primeras
                ``spatial_dim`` componentes y el momento las últimas. Anda en cualquier
                dimensión espacial.
            T: Horizonte temporal.

        Raises:
            ValueError: Si ``data_dim`` no es par.
        """
        if data_dim % 2 != 0:
            raise ValueError(
                f"data_dim de CLD debe ser par (= 2 * spatial_dim); recibí "
                f"data_dim={data_dim}"
            )
        super().__init__(data_dim=data_dim, T=T)
        self.spatial_dim = data_dim // 2
        self.beta = float(beta)
        self.mass = float(mass)
        self.m_inv = 1.0 / self.mass
        self.gamma = 2.0 * math.sqrt(self.beta * self.mass)   # Γ = 2 sqrt(βM)
        self.lam = -math.sqrt(self.beta / self.mass)          # autovalor doble λ = -sqrt(β/M)

    # --------------------------------------------------------- matriz de transición

    def _phi(self, tt: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Entradas de ``Φ(t) = exp(A t)`` (autovalor repetido), cada una ``(B, 1)``."""
        lam = self.lam
        e = torch.exp(lam * tt)
        phi_xx = e * (1.0 - lam * tt)
        phi_xv = e * tt * self.m_inv
        phi_vx = -e * self.beta * tt
        phi_vv = e * (1.0 + lam * tt)
        return phi_xx, phi_xv, phi_vx, phi_vv

    def _cov(self, tt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Covarianza 2×2 por dimensión ``(Σ_xx, Σ_xv, Σ_vv)``, cada una ``(B, 1)``.

        ``Σ(t) = Φ Σ_0 Φᵀ + ∫_0^t Φ(τ) GGᵀ Φ(τ)ᵀ dτ`` con ``Σ_0 = diag(0, M)``
        (``v_0 ~ N(0, M)``) y ``GGᵀ = diag(0, 2Γ)``. Las integrales son exactas con
        ``a = 2λ``: ``I0 = (e^{at}-1)/a``, ``I1 = (e^{at}(at-1)+1)/a²``,
        ``I2 = (e^{at}(a²t²-2at+2)-2)/a³``.
        """
        lam, m_inv, gamma, mass = self.lam, self.m_inv, self.gamma, self.mass
        _, phi_xv, _, phi_vv = self._phi(tt)

        a = 2.0 * lam
        e = torch.exp(a * tt)
        i0 = (e - 1.0) / a
        i1 = (e * (a * tt - 1.0) + 1.0) / a ** 2
        i2 = (e * (a ** 2 * tt ** 2 - 2.0 * a * tt + 2.0) - 2.0) / a ** 3

        # Parte 1 (Φ Σ_0 Φᵀ, con Σ_0 = diag(0, M)) + Parte 2 (W(t) = 2Γ ∫ φ φᵀ).
        sxx = mass * phi_xv ** 2 + 2.0 * gamma * m_inv ** 2 * i2
        sxv = mass * phi_xv * phi_vv + 2.0 * gamma * m_inv * (i1 + lam * i2)
        svv = mass * phi_vv ** 2 + 2.0 * gamma * (i0 + 2.0 * lam * i1 + lam ** 2 * i2)
        return sxx, sxv, svv

    def _cholesky(self, tt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Factor de Cholesky inferior ``(L11, L21, L22)`` de la covarianza por dimensión."""
        sxx, sxv, svv = self._cov(tt)
        l11 = torch.sqrt(sxx).clamp_min(self._std_eps)
        l21 = sxv / l11
        l22 = torch.sqrt((svv - l21 ** 2).clamp_min(self._std_eps ** 2))
        return l11, l21, l22

    def _mean(self, x0: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        """Media del kernel ``(B, data_dim)`` desde la posición ``x0`` ``(B, spatial_dim)``."""
        phi_xx, _, phi_vx, _ = self._phi(tt)
        mean_x = phi_xx * x0          # E[v_0] = 0 -> la media depende solo de x0
        mean_v = phi_vx * x0
        return torch.cat([mean_x, mean_v], dim=-1)

    # --------------------------------------------------------------------- API

    def sde(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Coeficientes del forward sobre el estado aumentado ``u = (x, v)``.

        Args:
            x: Estado aumentado de shape ``(B, data_dim)`` (p. ej. ``(x_1, x_2, v_1, v_2)``
                en 2D).
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            ``(drift, diffusion)`` ambos de shape ``(B, data_dim)``. La difusión es
            **estructurada**: ``0`` en el bloque de posición y ``sqrt(2Γ)`` en el de
            momento (el ruido entra solo por ``v``). Diverge a propósito del contrato
            escalar ``(B, 1)`` de la familia VP/VE/sub-VP.
        """
        pos, mom = x[:, : self.spatial_dim], x[:, self.spatial_dim :]
        drift_x = self.m_inv * mom
        drift_v = -(self.gamma * self.m_inv * mom + self.beta * pos)
        drift = torch.cat([drift_x, drift_v], dim=-1)
        g = math.sqrt(2.0 * self.gamma)
        diffusion = torch.cat([torch.zeros_like(pos), torch.full_like(mom, g)], dim=-1)
        return drift, diffusion

    def marginal_prob(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Media y factor de Cholesky del kernel conjunto ``p_t(u_t | x_0)``.

        Args:
            x0: Posición limpia de shape ``(B, spatial_dim)`` (lo que entrega
                ``data_generation``).
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            ``(mean, L)`` con ``mean`` de shape ``(B, data_dim)`` y ``L`` de shape
            ``(B, 2, 2)`` (factor de Cholesky inferior de la covarianza 2×2 por dimensión
            espacial, igual para todas las dimensiones). **No** devuelve un ``std`` escalar
            como la familia VP/VE/sub-VP.
        """
        tt = self._expand_t(t)
        mean = self._mean(x0, tt)
        l11, l21, l22 = self._cholesky(tt)
        b = l11.shape[0]
        L = torch.zeros(b, 2, 2, dtype=l11.dtype, device=l11.device)
        L[:, 0, 0] = l11[:, 0]
        L[:, 1, 0] = l21[:, 0]
        L[:, 1, 1] = l22[:, 0]
        return mean, L

    def perturb(
        self, x0: torch.Tensor, t: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Muestrea ``u_t`` del kernel conjunto y devuelve el ruido estándar usado.

        Args:
            x0: Posición limpia de shape ``(B, spatial_dim)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.
            generator: Generador opcional para reproducibilidad.

        Returns:
            ``(u_t, noise)`` ambos de shape ``(B, data_dim)``; ``noise`` es ``N(0, I)``
            estándar ordenado como posición seguida de momento.
        """
        tt = self._expand_t(t)
        mean = self._mean(x0, tt)
        l11, l21, l22 = self._cholesky(tt)
        n = torch.randn(
            (x0.shape[0], self.data_dim),
            generator=generator,
            device=x0.device,
            dtype=x0.dtype,
        )
        d = self.spatial_dim
        n_x, n_v = n[:, :d], n[:, d:]
        x_t = mean[:, :d] + l11 * n_x
        v_t = mean[:, d:] + l21 * n_x + l22 * n_v
        return torch.cat([x_t, v_t], dim=-1), n

    def score_target(
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Score conjunto del kernel a partir del ruido estándar de :meth:`perturb`.

        Como ``u_t - mean = L n`` (por dimensión), el score conjunto es
        ``∇_u log p_t(u_t|x_0) = -(Lᵀ)^{-1} n``. La componente de momento
        (``score_v = -n_v / L22``) es el target de HSM ``∇_v log p_t(v|x)`` que aprende la
        red; se devuelve el vector completo de shape ``(B, data_dim)``.

        Args:
            x0: Posición limpia de shape ``(B, spatial_dim)``.
            t: Tiempo de shape ``(B,)`` o ``(B, 1)``.
            eps: Ruido estándar devuelto por :meth:`perturb`, shape ``(B, data_dim)``.

        Returns:
            ``(score_real, weight)`` con ``score_real`` de shape ``(B, data_dim)`` y ``weight`` de
            shape ``(B, 1)`` (pesado de HSM diferido al loop de entrenamiento → ``1``).
        """
        tt = self._expand_t(t)
        l11, l21, l22 = self._cholesky(tt)
        d = self.spatial_dim
        n_x, n_v = eps[:, :d], eps[:, d:]
        score_x = -(n_x / l11 - l21 * n_v / (l11 * l22))
        score_v = -(n_v / l22)
        score_real = torch.cat([score_x, score_v], dim=-1)
        weight = torch.ones(eps.shape[0], 1, dtype=eps.dtype, device=eps.device)
        return score_real, weight

    def prior_sampling(
        self,
        shape: tuple[int, ...],
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Muestrea del prior estacionario: ``x ~ N(0, 1/β)``, ``v ~ N(0, M)``.

        Args:
            shape: Shape de la salida; la última dimensión debe ser ``data_dim`` (par).
            generator: Generador opcional para reproducibilidad.
            device: Dispositivo de la salida.
            dtype: Tipo de la salida (default ``float32``).

        Returns:
            Tensor de shape ``shape`` con el bloque de posición escalado por
            ``sqrt(1/β)`` y el de momento por ``sqrt(M)``.
        """
        z = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        std_x = math.sqrt(1.0 / self.beta)
        std_v = math.sqrt(self.mass)
        scale = torch.ones(shape[-1], dtype=z.dtype, device=z.device)
        scale[: shape[-1] // 2] = std_x
        scale[shape[-1] // 2 :] = std_v
        return z * scale
