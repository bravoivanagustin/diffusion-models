"""Loop de entrenamiento por denoising score matching y persistencia de la red.

Reúne el muestreo de pares de entrenamiento (:mod:`losses`), la red de score
(:class:`diffusion.models.ScoreMLP`) y un proceso forward (:class:`diffusion.sde.ForwardSDE`)
en un :func:`train` que devuelve la red entrenada y la historia de pérdida.

La red es la **variable de control** del estudio de ablación: sus hiperparámetros viven en
:class:`TrainConfig` con los defaults de ``ScoreMLP`` y normalmente no se tocan entre
variantes. La **regla del Eje 1** vive acá implícitamente: cada llamada a :func:`train`
instancia una red nueva, así que cambiar de SDE = un entrenamiento desde cero (los samplers
del Eje 2 reusan la misma red sin reentrenar).

:func:`save_checkpoint` / :func:`load_checkpoint` guardan los pesos junto con la metadata
mínima (nombre de la SDE, ``data_dim``, hiperparámetros de la red) para que los samplers
puedan reconstruir la red sin el config original.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import torch

from ..models import ScoreMLP
from ..sde import ForwardSDE
from .losses import dsm_loss, sample_timesteps


@dataclass
class TrainConfig:
    """Hiperparámetros de una corrida de entrenamiento.

    Describe por completo la corrida: optimización + tamaño del dataset + arquitectura de la
    red. Pensado para construirse desde un archivo de config (ver :mod:`diffusion.training.config`),
    pero es un dataclass plano y se arma igual de fácil a mano en los tests.
    """

    # --- optimización ---
    epochs: int = 200
    batch_size: int = 256
    n_samples: int = 4000
    lr: float = 2e-3
    t_eps: float = 1e-3
    grad_clip: float | None = None
    seed: int | None = 0
    device: str = "cpu"
    log_every: int = 0  # 0 = silencioso; N = imprime cada N épocas
    # --- red (variable de control: defaults de ScoreMLP, normalmente fijos) ---
    embed_dim: int = 128
    hidden_dim: int = 256
    num_blocks: int = 4
    activation: str = "silu"


@dataclass
class TrainResult:
    """Resultado de :func:`train`: la red entrenada y la traza de la corrida."""

    net: ScoreMLP
    history: list[float] = field(default_factory=list)  # pérdida media por época
    config: TrainConfig = field(default_factory=TrainConfig)
    sde_name: str = ""


def train(
    sde: ForwardSDE,
    distribution,
    config: TrainConfig,
    *,
    generator: torch.Generator | None = None,
) -> TrainResult:
    """Entrena una :class:`ScoreMLP` para aproximar el score de ``sde`` por DSM.

    Args:
        sde: Proceso forward (define el kernel y el target del score). Su ``data_dim`` fija la
            dimensión de la red (2 para VP/VE/sub-VP).
        distribution: Fuente de datos ``p_data(x_0)`` (una ``PointDistribution`` de
            :mod:`diffusion.data_generation`); se le pide un ``DataLoader``.
        config: Hiperparámetros de la corrida.
        generator: Generador opcional para el ruido del kernel / muestreo de ``t``. Si es
            ``None`` se crea uno sembrado con ``config.seed``.

    Returns:
        :class:`TrainResult` con la red entrenada, la historia de pérdida (una entrada por
        época), el ``config`` usado y el nombre de la SDE.
    """
    device = torch.device(config.device)
    if config.seed is not None:
        torch.manual_seed(config.seed)
    if generator is None:
        generator = torch.Generator(device=device)
        if config.seed is not None:
            generator.manual_seed(config.seed)

    net = ScoreMLP(
        data_dim=sde.data_dim,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        activation=config.activation,
    ).to(device)
    net.train()

    loader = distribution.dataloader(config.n_samples, config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    history: list[float] = []
    for epoch in range(config.epochs):
        running, n_batches = 0.0, 0
        for (x0,) in loader:
            x0 = x0.to(device)
            t = sample_timesteps(
                x0.shape[0], sde.T, config.t_eps, generator=generator, device=device
            )
            loss = dsm_loss(net, sde, x0, t, generator=generator)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimizer.step()

            running += loss.item()
            n_batches += 1

        avg = running / max(n_batches, 1)
        history.append(avg)
        if config.log_every and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(f"[{sde.name}] época {epoch + 1}/{config.epochs}  pérdida={avg:.6f}")

    return TrainResult(net=net, history=history, config=config, sde_name=sde.name)


# ----------------------------------------------------------------- persistencia


def save_checkpoint(result: TrainResult, path: str | pathlib.Path) -> pathlib.Path:
    """Guarda la red entrenada y su metadata en ``path`` (``.pt`` de torch).

    Args:
        result: Resultado de :func:`train`.
        path: Ruta de salida (se crean los directorios intermedios).

    Returns:
        La ruta donde se guardó (como :class:`pathlib.Path`).
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    blob = {
        "model_state": result.net.state_dict(),
        "meta": {
            "sde_name": result.sde_name,
            "data_dim": result.net.data_dim,
            "model": {
                "embed_dim": cfg.embed_dim,
                "hidden_dim": cfg.hidden_dim,
                "num_blocks": cfg.num_blocks,
                "activation": cfg.activation,
            },
            "history": list(result.history),
        },
    }
    torch.save(blob, out)
    return out


def load_checkpoint(
    path: str | pathlib.Path, *, map_location: torch.device | str = "cpu"
) -> tuple[ScoreMLP, dict]:
    """Reconstruye la :class:`ScoreMLP` entrenada desde un checkpoint de :func:`save_checkpoint`.

    Args:
        path: Ruta del ``.pt`` guardado.
        map_location: Dispositivo donde cargar los pesos (default ``"cpu"``).

    Returns:
        ``(net, meta)`` con la red en modo ``eval`` y la metadata guardada.
    """
    # weights_only=False: es nuestro propio checkpoint (incluye un dict de metadata).
    blob = torch.load(path, map_location=map_location, weights_only=False)
    meta = blob["meta"]
    m = meta["model"]
    net = ScoreMLP(
        data_dim=meta["data_dim"],
        embed_dim=m["embed_dim"],
        hidden_dim=m["hidden_dim"],
        num_blocks=m["num_blocks"],
        activation=m["activation"],
    )
    net.load_state_dict(blob["model_state"])
    net.eval()
    return net, meta
