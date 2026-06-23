"""Red de score (MLP) para datos de juguete.

Expone la red que aproxima el score :math:`\\nabla_x \\log p_t(x)` y sus piezas.
Es la variable de control del estudio de ablación: misma arquitectura en todas
las celdas SDE × sampler. Ver :mod:`diffusion.mlp.score_mlp`.

Uso típico::

    from diffusion.mlp import ScoreMLP

    net = ScoreMLP(data_dim=2)
    score = net(x, t)
"""

from __future__ import annotations

from .score_mlp import ResidualBlock, ScoreMLP, SinusoidalEmbedding

__all__ = ["SinusoidalEmbedding", "ResidualBlock", "ScoreMLP"]
