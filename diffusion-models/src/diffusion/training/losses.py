"""Núcleo del entrenamiento: la pérdida de denoising score matching (DSM).

El corazón del módulo es :func:`dsm_loss`, que combina las tres piezas ya entregadas para
un único batch:

1. ``data_generation`` aporta el dato limpio ``x_0`` (para la familia escalar es el estado
   completo; para CLD es la **posición**, que la SDE aumenta internamente con el momento).
2. ``sde.perturb`` lo ruidea hasta ``x_t`` y devuelve el ruido estándar ``eps`` usado.
3. ``sde.score_target`` da el score real del kernel ``∇_{x_t} log p_t(x_t | x_0)`` y el peso
   ``λ(t)`` de la pérdida.

La red (:class:`diffusion.mlp.ScoreMLP`) predice ``s_θ(x_t, t)`` y se minimiza el error
pesado ``λ(t) · ||s_θ - score_real||²``. Esta función es **agnóstica a la SDE**: el mismo
código corre VP/VE/sub-VP (estado escalar) y CLD (estado aumentado) porque ``perturb`` y
``score_target`` ya devuelven las shapes correctas y el peso adecuado (``σ_t²`` en la familia
escalar, ``1`` en CLD —el pesado de HSM queda diferido acá—).

Es la única pieza donde no hay I/O ni estado: dados ``(net, sde, x_0, t)`` devuelve un escalar
diferenciable, así que se testea directamente sin loop ni archivos.
"""

from __future__ import annotations

import torch

from ..sde import ForwardSDE


def dsm_loss(
    net: torch.nn.Module,
    sde: ForwardSDE,
    x0: torch.Tensor,
    t: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pérdida de denoising score matching para un batch.

    Calcula ``mean( λ(t) · || s_θ(x_t, t) - ∇_{x_t} log p_t(x_t | x_0) ||² )`` muestreando un
    único ``x_t`` por dato (estimador de un punto del DSM, suficiente para batches grandes).

    Args:
        net: Red de score ``s_θ`` (típicamente :class:`diffusion.mlp.ScoreMLP`); recibe
            ``(x_t, t)`` y devuelve un tensor de la misma shape que ``x_t``.
        sde: Proceso forward que define el kernel de perturbación y el target del score.
        x0: Dato limpio de shape ``(B, D)`` —``D = sde.data_dim`` para la familia escalar;
            ``D = sde.spatial_dim`` (la posición) para CLD—.
        t: Tiempo de shape ``(B,)`` o ``(B, 1)``, normalmente en ``[t_eps, T]``.
        generator: Generador opcional de torch para el ruido del kernel (reproducibilidad).

    Returns:
        Escalar (tensor 0-dim) diferenciable con la pérdida media del batch.
    """
    x_t, eps = sde.perturb(x0, t, generator=generator)
    score_real, weight = sde.score_target(x0, t, eps)
    score_pred = net(x_t, t)
    # weight es (B, 1) y broadcastea sobre las D componentes del score.
    return (weight * (score_pred - score_real).pow(2)).mean()


def sample_timesteps(
    n: int,
    T: float,
    t_eps: float,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Muestrea ``n`` tiempos uniformes en ``[t_eps, T]``.

    El piso ``t_eps > 0`` evita ``t = 0``, donde el desvío del kernel ``σ_t → 0`` y el target
    del score ``-eps/σ_t`` se vuelve numéricamente inestable.

    Args:
        n: Cantidad de tiempos (típicamente el tamaño del batch).
        T: Horizonte temporal (``sde.T``).
        t_eps: Piso del muestreo (p. ej. ``1e-3``).
        generator: Generador opcional de torch para reproducibilidad.
        device: Dispositivo de la salida.
        dtype: Tipo de la salida (default ``float32``).

    Returns:
        Tensor de shape ``(n,)`` con tiempos en ``[t_eps, T]``.
    """
    u = torch.rand(n, generator=generator, device=device, dtype=dtype)
    return t_eps + (T - t_eps) * u
