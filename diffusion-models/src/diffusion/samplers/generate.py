"""Orquestación de generación checkpoint-driven (Eje 2).

Cierra el pipeline forward→score→**sampleo** desde un checkpoint entrenado: carga la red y
su metadata (:func:`diffusion.training.load_checkpoint`), reconstruye la SDE del Eje 1
(:func:`diffusion.sde.make_sde`) y arma el sampler del Eje 2 (:func:`make_sampler`), genera
las muestras ``x_0`` y opcionalmente las persiste en un ``.npz``.

Es un *seam* de integración deliberado: junta ``samplers`` + ``training`` + ``sde`` sin
modificar sus contratos. La red llega ya en modo ``eval`` (responsabilidad que
:func:`generate_from_checkpoint` garantiza para el driver de :mod:`~diffusion.samplers.base`).

Uso típico (correr desde ``diffusion-models/``)::

    from diffusion.samplers import generate_from_checkpoint

    x0 = generate_from_checkpoint(
        "checkpoints/vp_mixture.pt", "pf_ode",
        n_samples=2000, n_steps=500, seed=0, out="data/vp_mixture_pf_ode.npz",
    )
"""

from __future__ import annotations

import pathlib

import torch

from ..sde import make_sde
from ..training import load_checkpoint


def generate_from_checkpoint(
    checkpoint_path: str | pathlib.Path,
    sampler_name: str,
    *,
    n_samples: int,
    n_steps: int = 500,
    seed: int | None = None,
    out: str | pathlib.Path | None = None,
    save_trajectory: bool = False,
    map_location: str = "cpu",
    **sampler_kwargs,
) -> torch.Tensor:
    """Genera muestras ``x_0`` a partir de un checkpoint entrenado y opcionalmente las guarda.

    Reconstruye la SDE y la red desde la metadata del checkpoint (``sde_name``, ``data_dim``,
    hiperparámetros de red), arma el sampler ``sampler_name`` con la factory e integra el
    proceso reverso. No reentrena ni muta la red (Eje 2): la pone en ``eval`` y la consume
    como función pura ``(x, t) -> score``.

    Args:
        checkpoint_path: Ruta del ``.pt`` producido por
            :func:`diffusion.training.save_checkpoint`. Debe existir.
        sampler_name: Clave del sampler en el registry (``"euler"``, ``"pf_ode"``,
            ``"heun"``, ``"pc"``).
        n_samples: Número de muestras a generar (``N``).
        n_steps: Número de pasos de integración del sampler.
        seed: Si no es ``None``, siembra un :class:`torch.Generator` para que la generación
            sea reproducible (incluye el muestreo del prior y los pasos estocásticos).
        out: Si se provee, ruta del ``.npz`` donde guardar las muestras (clave ``samples``,
            y ``trajectory`` cuando ``save_trajectory`` es ``True``). Se crean los
            directorios intermedios.
        save_trajectory: Si es ``True``, captura la trayectoria de integración y la incluye
            en la salida (y en el ``.npz`` si ``out`` se provee).
        map_location: Dispositivo donde cargar los pesos del checkpoint (default ``"cpu"``).
        **sampler_kwargs: Parámetros extra del sampler elegido (p. ej. ``t_eps``; para
            ``"pc"`` también ``n_corrector``/``snr``); los no aplicables se descartan.

    Returns:
        El tensor de muestras ``x_0`` de shape ``(n_samples, data_dim)`` en ``float32``.

    Raises:
        FileNotFoundError: Si ``checkpoint_path`` no existe.
        KeyError: Si el checkpoint carece de las claves esperadas en su metadata
            (``sde_name``/``data_dim``); el contrato lo provee
            :func:`diffusion.training.save_checkpoint`.
        ValueError: Si ``sampler_name`` no está en el registry (lista las opciones válidas).
    """
    # Import diferido para evitar cualquier ciclo de import durante la inicialización del
    # paquete (``__init__`` importa este módulo; ``make_sampler`` vive en ``__init__``).
    from . import make_sampler

    path = pathlib.Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint inexistente: {path}")

    net, meta = load_checkpoint(path, map_location=map_location)
    try:
        sde_name = meta["sde_name"]
        data_dim = meta["data_dim"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"Checkpoint inválido en {path}: la metadata no tiene las claves esperadas "
            "('sde_name', 'data_dim'). ¿Se guardó con diffusion.training.save_checkpoint?"
        ) from exc

    net.eval()
    sde = make_sde(sde_name, data_dim=data_dim)

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    sampler = make_sampler(sampler_name, sde, net, n_steps=n_steps, **sampler_kwargs)
    result = sampler.sample(
        n_samples, generator=generator, return_trajectory=save_trajectory
    )
    if save_trajectory:
        x0, trajectory = result
    else:
        x0, trajectory = result, None

    if out is not None:
        import numpy as np

        out_path = pathlib.Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"samples": x0.cpu().numpy()}
        if save_trajectory:
            arrays["trajectory"] = trajectory.cpu().numpy()
        np.savez(out_path, **arrays)

    return x0
