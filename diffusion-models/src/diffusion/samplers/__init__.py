"""Proceso reverso (Eje 2): samplers que integran la SDE/ODE inversa.

Por ahora solo expone el ABC :class:`~diffusion.samplers.base.ReverseSampler`. El
registry/factory (``make_sampler``/``available_samplers``) y los samplers concretos
(Euler–Maruyama, PF-ODE, Heun, predictor–corrector) llegan en tasks posteriores.
"""

from __future__ import annotations

from .base import ReverseSampler, ScoreFn

__all__ = ["ReverseSampler", "ScoreFn"]
