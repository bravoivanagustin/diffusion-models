"""Redes de score: la variable de control del estudio de ablación.

Agrupa las redes que aproximan el score :math:`\\nabla_x \\log p_t(x)` y sus piezas, con
misma arquitectura en todas las celdas SDE × sampler:

- :mod:`diffusion.models.layers` — piezas compartidas entre redes (embedding sinusoidal de
  tiempo, activaciones).
- :mod:`diffusion.models.mlp` — :class:`ScoreMLP`, la red para datos de juguete 2D (Fase 1).
- :mod:`diffusion.models.base` — el Protocol :class:`ScoreModel`, el contrato
  ``(x, t) -> score`` que toda red satisface estructuralmente.

La U-Net de imágenes (Fase 2) se sumará como ``diffusion.models.unet``.

Uso típico::

    from diffusion.models import ScoreMLP

    net = ScoreMLP(data_dim=2)
    score = net(x, t)
"""

from __future__ import annotations

from .base import ScoreModel
from .layers import SinusoidalEmbedding
from .mlp import ResidualBlock, ScoreMLP

__all__ = ["ScoreModel", "SinusoidalEmbedding", "ResidualBlock", "ScoreMLP"]
