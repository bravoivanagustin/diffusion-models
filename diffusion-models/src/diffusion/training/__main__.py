"""Smoke test manual del módulo: entrena pocas épocas y reporta la pérdida.

Correr (desde ``diffusion-models/src/``)::

    python -m diffusion.training

Para cada SDE entrena una red chica sobre un gaussiano de juguete unas pocas épocas en CPU e
imprime la pérdida inicial y final (debería bajar). Se usa ``-m`` porque el módulo usa imports
relativos y no es ejecutable como script suelto.
"""

from __future__ import annotations

from diffusion.data_generation import make_distribution
from diffusion.sde import available_sdes, make_sde
from diffusion.training import TrainConfig, train


def main() -> None:
    dist = make_distribution("mixture", dim=2, n_components=8, seed=0)
    config = TrainConfig(
        epochs=40, n_samples=512, batch_size=128, lr=2e-3,
        hidden_dim=64, num_blocks=2, seed=0,
    )
    for name in available_sdes():
        sde = make_sde(name)
        result = train(sde, dist, config)
        print(
            f"{name:7s} data_dim={sde.data_dim}  "
            f"pérdida inicial={result.history[0]:.4f} -> final={result.history[-1]:.4f}"
        )


if __name__ == "__main__":
    main()
