"""Red de score (U-Net) para imágenes — Fase 2.

Aproxima el score :math:`s_\\theta(x, t) \\approx \\nabla_x \\log p_t(x)` sobre tensores
imagen ``(B, C, H, W)`` con una U-Net condicionada en el tiempo. Igual que la
:class:`~diffusion.models.mlp.ScoreMLP` de Fase 1, es la **variable de control** del estudio
de ablación: su arquitectura e hiperparámetros quedan fijos en las 12 celdas SDE × sampler,
y la red es **enteramente determinística** — GroupNorm, sin dropout, batchnorm ni ninguna
capa estocástica. Toda la estocasticidad vive *fuera* de esta clase, en el marco de SDEs
(proceso forward, muestreo de pares de entrenamiento, sampler reverso).

Este archivo reúne los bloques privados de la U-Net (no se re-exportan del paquete: la
proyección temporal, el bloque residual convolucional, la atención y el cambio de
resolución) y la clase pública ``ScoreUNet``. El embedding de tiempo
:class:`~diffusion.models.layers.SinusoidalEmbedding` y el registry de activaciones son
compartidos entre redes y viven en :mod:`diffusion.models.layers`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import SinusoidalEmbedding, _make_activation


class TimeMLP(nn.Module):
    """Proyección del tiempo al vector de condicionamiento de la U-Net.

    Embebe ``t`` con el :class:`~diffusion.models.layers.SinusoidalEmbedding`
    compartido y lo proyecta con un MLP de dos capas:
    ``SinusoidalEmbedding -> Linear -> activación -> Linear``. La salida
    ``(B, time_embed_dim)`` se computa **una sola vez** por forward y la
    comparten todos los bloques de la red (cada bloque la re-proyecta a sus
    canales).
    """

    def __init__(
        self,
        embed_dim: int,
        time_embed_dim: int,
        activation: str = "silu",
    ) -> None:
        """Inicializa la proyección temporal.

        Args:
            embed_dim: Dimensión del embedding sinusoidal de entrada (debe ser
                par; lo valida el embedding reusado).
            time_embed_dim: Dimensión del vector de condicionamiento de salida.
            activation: Nombre de la activación entre las dos lineales.
        """
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.time_embed_dim = int(time_embed_dim)

        self.embed = SinusoidalEmbedding(embed_dim)
        self.lin1 = nn.Linear(embed_dim, time_embed_dim)
        self.act = _make_activation(activation)
        self.lin2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Proyecta el tiempo al vector de condicionamiento.

        Args:
            t: Tensor de tiempos de shape ``(B,)`` o ``(B, 1)`` (lo normaliza el
                embedding reusado), en cualquier escala usada por las SDEs del
                repo (``[0, 1]``, ``[0, T]`` o pasos enteros).

        Returns:
            Tensor de shape ``(B, time_embed_dim)``.
        """
        return self.lin2(self.act(self.lin1(self.embed(t))))
