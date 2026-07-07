"""Loop de entrenamiento por denoising score matching y persistencia de la red.

Reúne el muestreo de pares de entrenamiento (:mod:`losses`), una red de score cualquiera
(cualquier :class:`diffusion.models.ScoreModel`: el ``ScoreMLP`` de Fase 1, la ``ScoreUNet``
de Fase 2, …) y un proceso forward (:class:`diffusion.sde.ForwardSDE`) en un :func:`train`
que devuelve la red entrenada y la historia de pérdida.

:func:`train` es **agnóstico a la red y al origen de datos**: recibe la ``model`` ya
construida y un iterador **infinito** de tensores crudos, y corre un loop **por pasos**
(``config.num_steps``). No construye la red ni ramifica por su tipo — esa responsabilidad vive
en el caller (``make_model`` / el config-driven). La **regla del Eje 1** sigue vigente: cambiar
de SDE = una red nueva y un entrenamiento desde cero (los samplers del Eje 2 reusan la misma
red sin reentrenar).

:func:`save_checkpoint` / :func:`load_checkpoint` guardan los pesos junto con la metadata
mínima (nombre de la SDE, ``data_dim``, hiperparámetros de la red) para que los samplers
puedan reconstruir la red sin el config original.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from ..models import ScoreMLP, ScoreModel
from ..sde import ForwardSDE
from .losses import dsm_loss, sample_timesteps


@dataclass
class TrainConfig:
    """Hiperparámetros del **loop** de entrenamiento (solo optimización y corrida).

    Ya no carga hiperparámetros de red (viven en el constructor de la red / ``make_model``) ni
    de tamaño de dataset (viven en la fuente de datos). Pensado para construirse desde un
    archivo de config (ver :mod:`diffusion.training.config`), pero es un dataclass plano y se
    arma igual de fácil a mano en los tests.
    """

    num_steps: int = 1000  # pasos de optimización (≈ epochs × n_samples/batch_size viejos)
    lr: float = 2e-3
    t_eps: float = 1e-3
    grad_clip: float | None = None
    seed: int | None = 0
    device: str = "cpu"
    log_every: int = 0  # 0 = silencioso; N = imprime cada N pasos (no afecta el history)


@dataclass
class TrainResult:
    """Resultado de :func:`train`: la red entrenada y la traza de la corrida."""

    net: ScoreModel  # cualquier red de score (ScoreMLP, ScoreUNet, …), no atada a una clase
    history: list[float] = field(default_factory=list)  # pérdida media por intervalo de registro
    config: TrainConfig = field(default_factory=TrainConfig)
    sde_name: str = ""
    data_dim: int = 0  # = sde.data_dim; lo copia save_checkpoint a meta (lo usa generate.py)


def train(
    sde: ForwardSDE,
    model: ScoreModel,
    data: Iterator[torch.Tensor],
    config: TrainConfig,
    *,
    generator: torch.Generator | None = None,
) -> TrainResult:
    """Entrena la red ``model`` para aproximar el score de ``sde`` por DSM.

    Loop **por pasos** y agnóstico a la red: usa la ``model`` recibida (no construye ninguna ni
    ramifica por su tipo) y consume ``data`` con ``next()`` — un batch por paso.

    Args:
        sde: Proceso forward (define el kernel y el target del score). Su ``data_dim`` queda
            registrado en el resultado (lo usa el checkpoint).
        model: Red de score ya construida (cualquier :class:`ScoreModel`). Se mueve al
            dispositivo de forma idempotente y se pone en modo entrenamiento.
        data: Iterador/iterable **infinito** que yield-ea tensores crudos ``(B, ...)`` (p. ej.
            ``infinite_bare(distribution.dataloader(...))``). Se le pide un batch por paso.
        config: Hiperparámetros del loop de entrenamiento.
        generator: Generador opcional para el ruido del kernel / muestreo de ``t``. Si es
            ``None`` se crea uno sembrado con ``config.seed``.

    Returns:
        :class:`TrainResult` con la red entrenada, la historia de pérdida (pérdida media por
        intervalo de registro; nunca vacía: siempre incluye el último paso), el ``config``
        usado, el nombre de la SDE y su ``data_dim``.
    """
    device = torch.device(config.device)
    if config.seed is not None:
        torch.manual_seed(config.seed)
    if generator is None:
        generator = torch.Generator(device=device)
        if config.seed is not None:
            generator.manual_seed(config.seed)

    net = model.to(device)  # idempotente: no falla si el caller ya la movió
    net.train()

    data_iter = iter(data)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    # El history se registra a una cadencia fija, desacoplada de ``log_every`` (que gobierna
    # solo el print): si ``log_every>0`` coincide con él; si no, una cadencia interna que
    # deja ~100 puntos. Siempre se registra además el último paso, así history nunca queda
    # vacío (ni con el ``log_every=0`` por defecto).
    cadence = config.log_every if config.log_every > 0 else max(1, config.num_steps // 100)

    history: list[float] = []
    running, n_batches = 0.0, 0
    for step in range(config.num_steps):
        x0 = next(data_iter).to(device)
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

        is_last = step == config.num_steps - 1
        if (step + 1) % cadence == 0 or is_last:
            avg = running / max(n_batches, 1)
            history.append(avg)
            running, n_batches = 0.0, 0
            if config.log_every > 0:
                print(f"[{sde.name}] paso {step + 1}/{config.num_steps}  pérdida={avg:.6f}")

    return TrainResult(
        net=net, history=history, config=config, sde_name=sde.name, data_dim=sde.data_dim
    )


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
    # Transitorio: la receta de red ya no vive en ``TrainConfig`` (que se adelgazó), así que se
    # lee de la propia red entrenada (un ``ScoreMLP`` expone estos atributos). ``activation`` no
    # se guarda como atributo → default ``"silu"`` (el único valor usado). Esto conserva el
    # formato de checkpoint byte-compatible con el ``load_checkpoint`` actual y con los samplers
    # (se reemplaza por la receta model-agnóstica en la task 3.1).
    net = result.net
    blob = {
        "model_state": net.state_dict(),
        "meta": {
            "sde_name": result.sde_name,
            "data_dim": net.data_dim,
            "model": {
                "embed_dim": net.embed_dim,
                "hidden_dim": net.hidden_dim,
                "num_blocks": net.num_blocks,
                "activation": "silu",
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
