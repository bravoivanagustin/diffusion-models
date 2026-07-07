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


class ConvResBlock(nn.Module):
    """Bloque residual convolucional con inyección aditiva del tiempo.

    Análogo convolucional del :class:`~diffusion.models.mlp.ResidualBlock` lineal
    del MLP (comparten la idea, no el código — la pieza compartida vive en
    :mod:`~diffusion.models.layers`, no acá). El flujo es::

        GroupNorm -> activación -> Conv 3x3           # primera etapa
        + proyección temporal  (broadcast espacial)   # condicionamiento
        GroupNorm -> activación -> Conv 3x3           # segunda etapa
        + skip                                        # conexión residual

    El vector temporal ``t_emb`` (el ``(B, time_embed_dim)`` que produce
    :class:`TimeMLP`, compartido por todos los bloques del forward) se re-proyecta
    a los ``out_channels`` de este bloque con una lineal y se suma **tras la
    primera convolución**, expandido a ``(B, out_channels, 1, 1)`` para
    broadcastear sobre las dimensiones espaciales ``H`` y ``W``. Ese
    condicionamiento es lo que hace que dos tiempos distintos den salidas
    distintas.

    El skip es identidad cuando ``in_channels == out_channels``; si difieren, una
    convolución ``1x1`` proyecta la entrada a ``out_channels`` para poder sumarla.
    La normalización es :class:`~torch.nn.GroupNorm` (determinística e
    independiente del batch): **no** hay dropout ni batchnorm, en línea con la red
    como variable de control.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int = 8,
        activation: str = "silu",
    ) -> None:
        """Inicializa el bloque.

        Args:
            in_channels: Canales del tensor de entrada.
            out_channels: Canales del tensor de salida (y de la proyección
                temporal).
            time_embed_dim: Dimensión del vector temporal ``t_emb`` de entrada
                (la salida de :class:`TimeMLP`).
            groups: Grupos de las capas :class:`~torch.nn.GroupNorm`; debe dividir
                tanto a ``in_channels`` como a ``out_channels``.
            activation: Nombre de la activación (compartida con el resto del
                módulo vía :func:`~diffusion.models.layers._make_activation`).
        """
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.time_embed_dim = int(time_embed_dim)

        # Primera etapa: normaliza la entrada, activa y convoluciona a out_channels.
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = _make_activation(activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Proyección del tiempo a los canales de salida (se suma tras conv1).
        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        # Segunda etapa: normaliza, activa y convoluciona manteniendo out_channels.
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = _make_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip: identidad si los canales coinciden, conv 1x1 si hay que reajustar.
        if in_channels == out_channels:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Aplica el bloque con inyección de tiempo y conexión residual.

        Args:
            x: Tensor de shape ``(B, in_channels, H, W)``.
            t_emb: Vector temporal de shape ``(B, time_embed_dim)`` (salida de
                :class:`TimeMLP`), compartido por todos los bloques del forward.

        Returns:
            Tensor de shape ``(B, out_channels, H, W)`` (misma resolución
            espacial que la entrada).
        """
        h = self.conv1(self.act1(self.norm1(x)))       # (B, out_channels, H, W)
        # Broadcast del tiempo sobre H y W: (B, out_channels) -> (B, out_channels, 1, 1).
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))        # (B, out_channels, H, W)
        return h + self.skip(x)
