"""Loop de entrenamiento por denoising score matching (DSM).

El cuarto módulo del TP: el eslabón que une ``data_generation`` (los ``x_0``), ``models`` (la
red de score) y ``sde`` (el proceso forward). Enseña a :class:`diffusion.models.ScoreMLP` a
aproximar ``s_θ(x, t) ≈ ∇_x log p_t(x)`` para una SDE dada, minimizando la pérdida de DSM.

Uso típico (a mano)::

    from diffusion.sde import make_sde
    from diffusion.data_generation import make_distribution
    from diffusion.training import TrainConfig, train

    sde = make_sde("vp")
    dist = make_distribution("mixture", dim=2, n_components=8, seed=0)
    result = train(sde, dist, TrainConfig(epochs=300, n_samples=4000))

Uso típico (config-driven, una celda del estudio por archivo)::

    from diffusion.training import load_config, build_run, train, save_checkpoint
    spec = build_run(load_config("config/vp_mixture.yaml"))
    result = train(spec.sde, spec.distribution, spec.config)
    save_checkpoint(result, spec.checkpoint)
"""

from __future__ import annotations

from .config import RunSpec, build_run, load_config
from .losses import dsm_loss, sample_timesteps
from .trainer import TrainConfig, TrainResult, load_checkpoint, save_checkpoint, train

__all__ = [
    "dsm_loss",
    "sample_timesteps",
    "TrainConfig",
    "TrainResult",
    "train",
    "save_checkpoint",
    "load_checkpoint",
    "RunSpec",
    "load_config",
    "build_run",
]
