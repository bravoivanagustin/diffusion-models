"""Red de score (MLP) para datos de juguete 2D.

Aproxima el score :math:`s_\\theta(x, t) \\approx \\nabla_x \\log p_t(x)` con un MLP
condicionado en el tiempo. Es la **variable de control** del estudio de ablación:
su arquitectura e hiperparámetros se mantienen idénticos en todas las celdas SDE ×
sampler, y la red es **enteramente determinística** — no hay dropout, batchnorm ni
ninguna capa estocástica. Toda la estocasticidad vive *fuera* de esta clase, en el
marco de SDEs (proceso forward, muestreo de pares de entrenamiento, sampler reverso).

Tres módulos, en orden:

- :class:`SinusoidalEmbedding` — embebe el escalar de tiempo ``t`` en un vector.
- :class:`ResidualBlock` — bloque MLP con conexión residual (skip identidad).
- :class:`ScoreMLP` — la red completa: embedding de tiempo + bloques residuales.

Uso típico::

    from diffusion.mlp import ScoreMLP

    net = ScoreMLP(data_dim=2)          # VP / VE / sub-VP
    score = net(x, t)                    # x: (B, 2), t: (B,) -> (B, 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Funciones de activación soportadas, por nombre.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _make_activation(name: str) -> nn.Module:
    """Devuelve una instancia de la activación ``name``.

    Args:
        name: Nombre de la activación (p. ej. ``"silu"`` o ``"relu"``).

    Returns:
        Una nueva instancia del ``nn.Module`` de activación.

    Raises:
        ValueError: Si ``name`` no está entre las activaciones soportadas.
    """
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError:
        opts = ", ".join(sorted(_ACTIVATIONS))
        raise ValueError(
            f"Activación desconocida '{name}'. Opciones: {opts}"
        ) from None


class SinusoidalEmbedding(nn.Module):
    """Embebe un escalar de tiempo ``t`` en un vector con senos y cosenos.

    Sigue la codificación posicional de Transformers: para cada frecuencia se
    aporta un seno y un coseno, intercalados en el vector de salida::

        embed(t)_{2i}   = sin(t / 10000^{2i/d})
        embed(t)_{2i+1} = cos(t / 10000^{2i/d})

    con ``i = 0, …, d/2 - 1`` y ``d = embed_dim``. Las frecuencias
    (denominadores) se precomputan en :meth:`__init__` y se guardan como buffer
    (no son parámetros: no se aprenden). Funciona para cualquier ``t`` flotante
    no negativo, sin supuestos sobre su escala (el rango depende de la SDE:
    ``[0, 1]``, ``[0, T]`` o pasos enteros).
    """

    def __init__(self, embed_dim: int = 128) -> None:
        """Inicializa el embedding.

        Args:
            embed_dim: Dimensión del vector de salida. Debe ser par (cada
                frecuencia aporta un seno y un coseno).

        Raises:
            ValueError: Si ``embed_dim`` no es par.
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim debe ser par (un seno y un coseno por frecuencia); "
                f"recibí embed_dim={embed_dim}"
            )
        self.embed_dim = int(embed_dim)
        # Denominadores 10000^{2i/d} para i = 0 .. d/2 - 1, shape (d/2,).
        i = torch.arange(embed_dim // 2, dtype=torch.float32)
        denom = torch.pow(10000.0, (2.0 * i) / embed_dim)
        #: Buffer (no aprendible): se mueve con .to(device) junto al módulo.
        self.register_buffer("denom", denom)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embebe el tiempo.

        Args:
            t: Tensor de tiempos de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Tensor de shape ``(B, embed_dim)`` con senos y cosenos intercalados.
        """
        t = t.reshape(-1)  # (B, 1) o (B,) -> (B,)
        # args[b, i] = t_b / denom_i  ->  (B, d/2)
        args = t[:, None] / self.denom[None, :]
        # Intercalar sin/cos: stack -> (B, d/2, 2) -> reshape (B, d).
        emb = torch.stack((torch.sin(args), torch.cos(args)), dim=-1)
        return emb.reshape(t.shape[0], self.embed_dim)


class ResidualBlock(nn.Module):
    """Bloque MLP residual: dos lineales con activación y skip identidad.

    El flujo es ``Linear -> activación -> Linear`` y luego se suma la entrada
    (skip): ``salida = bloque(x) + x``. No hay proyección aprendida en el skip;
    entrada y salida tienen la misma dimensión.
    """

    def __init__(self, hidden_dim: int, activation: str = "silu") -> None:
        """Inicializa el bloque.

        Args:
            hidden_dim: Ancho de las capas lineales (entrada = salida).
            activation: Nombre de la activación entre las dos lineales.
        """
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = _make_activation(activation)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica el bloque con conexión residual.

        Args:
            x: Tensor de shape ``(B, hidden_dim)``.

        Returns:
            Tensor de shape ``(B, hidden_dim)`` (misma shape que la entrada).
        """
        return self.lin2(self.act(self.lin1(x))) + x


class ScoreMLP(nn.Module):
    """Red de score completa: embedding de tiempo + bloques residuales.

    Concatena ``x`` con el embedding de ``t``, proyecta a ``hidden_dim``, pasa
    por ``num_blocks`` :class:`ResidualBlock` y proyecta de vuelta a ``data_dim``
    (sin activación final: el score es no acotado y puede ser positivo o
    negativo). La salida tiene la misma dimensión que ``x``, porque el score
    :math:`\\nabla_x \\log p_t(x)` vive en el mismo espacio que ``x``.

    ``data_dim=2`` para VP-SDE, VE-SDE y sub-VP (el punto es ``(x, y)``).
    """

    def __init__(
        self,
        data_dim: int = 2,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        activation: str = "silu",
    ) -> None:
        """Inicializa la red.

        Args:
            data_dim: Dimensión del dato de entrada y de la salida (el score).
            embed_dim: Dimensión del embedding de tiempo.
            hidden_dim: Ancho de las capas ocultas en todos los bloques.
            num_blocks: Cantidad de bloques residuales.
            activation: Nombre de la activación (se pasa a cada bloque y a la
                proyección de entrada).
        """
        super().__init__()
        self.data_dim = int(data_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)

        self.time_embed = SinusoidalEmbedding(embed_dim)
        self.input_proj = nn.Linear(data_dim + embed_dim, hidden_dim)
        self.input_act = _make_activation(activation)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predice el score en ``(x, t)``.

        Args:
            x: Dato ruidoso de shape ``(B, data_dim)``.
            t: Tiempo / nivel de ruido de shape ``(B,)`` o ``(B, 1)``.

        Returns:
            Score predicho de shape ``(B, data_dim)``.
        """
        t_emb = self.time_embed(t)                  # (B, embed_dim)
        h = torch.cat([x, t_emb], dim=-1)           # (B, data_dim + embed_dim)
        h = self.input_act(self.input_proj(h))      # (B, hidden_dim)
        h = self.blocks(h)                          # (B, hidden_dim)
        return self.output_proj(h)                  # (B, data_dim)


if __name__ == "__main__":
    # Smoke test manual: instancia la red, corre un forward y reporta tamaños.
    net = ScoreMLP()
    x = torch.randn(16, 2)
    t = torch.rand(16)
    out = net(x, t)
    print(f"ScoreMLP(data_dim=2): salida {tuple(out.shape)}")

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {n_params:,}")
