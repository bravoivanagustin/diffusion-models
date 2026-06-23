"""Capa de configuración: de un archivo YAML a una corrida lista para :func:`train`.

Cada **celda del estudio de ablación** (una combinación SDE × dataset × hiperparámetros) se
describe en un ``.yaml`` versionable. Esta capa es un front-end fino y aislado del núcleo: el
loop y la pérdida no saben nada de archivos. :func:`load_config` lee el YAML a un ``dict`` y
:func:`build_run` lo ensambla en un :class:`RunSpec` —reusando los factories
``make_sde``/``make_distribution``, que ya filtran kwargs por firma—.

Estructura esperada del YAML::

    sde:                  # -> make_sde(name, **resto)
      name: vp
      beta_min: 0.1
    data:                 # -> make_distribution(shape, dim, **resto); n_samples va al config
      shape: mixture
      dim: 2
      n_samples: 4000
      n_components: 8
    train:                # -> campos de TrainConfig
      epochs: 300
      lr: 0.002
    model:                # opcional: hiperparámetros de la red (variable de control)
      hidden_dim: 256
    out:                  # rutas de salida (relativas al cwd)
      checkpoint: models/vp_mixture.pt
      loss_curve: models/vp_mixture_loss.png
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, fields

from ..data_generation import PointDistribution, make_distribution
from ..sde import ForwardSDE, make_sde
from .trainer import TrainConfig


@dataclass
class RunSpec:
    """Una corrida lista: SDE + fuente de datos + hiperparámetros + rutas de salida."""

    sde: ForwardSDE
    distribution: PointDistribution
    config: TrainConfig
    checkpoint: pathlib.Path | None = None
    loss_curve: pathlib.Path | None = None


def load_config(path: str | pathlib.Path) -> dict:
    """Lee un archivo YAML de configuración a un ``dict``.

    Args:
        path: Ruta del ``.yaml``.

    Returns:
        El contenido parseado como ``dict``.

    Raises:
        ModuleNotFoundError: Si ``pyyaml`` no está instalado.
    """
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
        raise ModuleNotFoundError(
            "Leer configs YAML necesita PyYAML y no está instalado. "
            "Instalalo con `uv add pyyaml` (o `pip install pyyaml`)."
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_run(raw: dict) -> RunSpec:
    """Ensambla un :class:`RunSpec` desde un ``dict`` de configuración.

    Args:
        raw: Config parseado (de :func:`load_config` o construido a mano).

    Returns:
        Un :class:`RunSpec` con la SDE, la distribución, el :class:`TrainConfig` y las rutas.

    Raises:
        ValueError: Si faltan claves obligatorias (``sde.name``, ``data.shape``).
    """
    raw = dict(raw or {})

    # --- SDE ---
    sde_raw = dict(raw.get("sde") or {})
    if "name" not in sde_raw:
        raise ValueError("config: falta 'sde.name' (p. ej. vp / ve / sub_vp / cld).")
    sde = make_sde(**sde_raw)

    # --- datos ---
    data_raw = dict(raw.get("data") or {})
    shape = data_raw.pop("shape", None) or data_raw.pop("name", None)
    if shape is None:
        raise ValueError("config: falta 'data.shape' (p. ej. gaussian / mixture / two_moons).")
    dim = data_raw.pop("dim", 2)
    n_samples = data_raw.pop("n_samples", None)
    distribution = make_distribution(shape, dim, **data_raw)

    # --- hiperparámetros (train + model) -> TrainConfig ---
    train_raw = dict(raw.get("train") or {})
    train_raw.update(raw.get("model") or {})  # 'model' es azúcar para los campos de red
    if n_samples is not None:
        train_raw.setdefault("n_samples", n_samples)
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(train_raw) - valid
    if unknown:
        raise ValueError(
            f"config: claves desconocidas en train/model: {sorted(unknown)}. "
            f"Válidas: {sorted(valid)}."
        )
    config = TrainConfig(**train_raw)

    # --- salidas ---
    out_raw = dict(raw.get("out") or {})
    checkpoint = out_raw.get("checkpoint")
    loss_curve = out_raw.get("loss_curve")
    return RunSpec(
        sde=sde,
        distribution=distribution,
        config=config,
        checkpoint=pathlib.Path(checkpoint) if checkpoint else None,
        loss_curve=pathlib.Path(loss_curve) if loss_curve else None,
    )
