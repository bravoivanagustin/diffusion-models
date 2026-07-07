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
    data:                 # -> make_distribution(shape, dim, **resto); n_samples/batch_size
      shape: mixture      #    describen la fuente de datos (no van al TrainConfig)
      dim: 2
      n_samples: 4000
      batch_size: 256
      n_components: 8
    train:                # -> campos de TrainConfig (solo el loop de optimización)
      num_steps: 300
      lr: 0.002
    model:                # opcional: receta de la red -> make_model(name, **resto)
      name: mlp           #   si falta, se usa {name: mlp} dimensionado desde el dato/SDE
      hidden_dim: 256
    out:                  # rutas de salida (relativas al cwd)
      checkpoint: models/vp_mixture.pt
      loss_curve: models/vp_mixture_loss.png
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, fields

from ..data_generation import infinite_bare, make_distribution
from ..models import ScoreModel, make_model
from ..sde import ForwardSDE, make_sde
from .trainer import TrainConfig

# Defaults de la fuente de datos cuando el bloque ``data:`` no los especifica (valores de la
# corrida por épocas previa, para no cambiar el comportamiento de los configs existentes).
_DEFAULT_N_SAMPLES = 4000
_DEFAULT_BATCH_SIZE = 256


@dataclass
class RunSpec:
    """Una corrida lista: SDE + red + fuente de datos + hiperparámetros + rutas de salida.

    Lleva la red (``model``) y el iterador infinito de datos (``data``) ya construidos, listos
    para pasarle a :func:`~diffusion.training.train` (en vez de una ``distribution`` finita).
    Además transporta ``model_spec`` —la receta ``{name, kwargs}`` con la que se construyó la
    red— para que ``scripts/train.py`` la pase a :func:`~diffusion.training.save_checkpoint` y
    el checkpoint quede reconstruible sin el config original.
    """

    sde: ForwardSDE
    model: ScoreModel
    data: Iterator
    config: TrainConfig
    model_spec: dict | None = None  # receta {name, kwargs} para el checkpoint model-agnóstico
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
        Un :class:`RunSpec` con la SDE, la red, la fuente de datos infinita, el
        :class:`TrainConfig` y las rutas.

    Raises:
        ValueError: Si faltan claves obligatorias (``sde.name``, ``data.shape``), si el bloque
            ``train:`` trae claves desconocidas para :class:`TrainConfig`, o si el ``name`` del
            bloque ``model:`` no está registrado en ``make_model``.
    """
    raw = dict(raw or {})

    # --- SDE ---
    sde_raw = dict(raw.get("sde") or {})
    if "name" not in sde_raw:
        raise ValueError("config: falta 'sde.name' (p. ej. vp / ve / sub_vp).")
    sde = make_sde(**sde_raw)

    # --- datos: n_samples/batch_size son parámetros de la fuente (ya no del TrainConfig) ---
    data_raw = dict(raw.get("data") or {})
    shape = data_raw.pop("shape", None) or data_raw.pop("name", None)
    if shape is None:
        raise ValueError("config: falta 'data.shape' (p. ej. gaussian / mixture / two_moons).")
    dim = data_raw.pop("dim", 2)
    n_samples = data_raw.pop("n_samples", None) or _DEFAULT_N_SAMPLES
    batch_size = data_raw.pop("batch_size", None) or _DEFAULT_BATCH_SIZE
    shuffle = data_raw.pop("shuffle", True)
    distribution = make_distribution(shape, dim, **data_raw)
    data = infinite_bare(distribution.dataloader(n_samples, batch_size, shuffle=shuffle))

    # --- hiperparámetros del loop -> TrainConfig (validación estricta contra sus campos) ---
    train_raw = dict(raw.get("train") or {})
    valid = {f.name for f in fields(TrainConfig)}
    unknown = set(train_raw) - valid
    if unknown:
        raise ValueError(
            f"config: claves desconocidas en train: {sorted(unknown)}. "
            f"Válidas: {sorted(valid)}."
        )
    config = TrainConfig(**train_raw)

    # --- red: bloque 'model:' opcional (default {name: mlp} dimensionado desde el dato/SDE);
    # las claves del bloque van a make_model, que filtra por firma (no se validan acá) ---
    model_raw = dict(raw.get("model") or {})
    model_name = model_raw.pop("name", "mlp")
    model_raw.setdefault("data_dim", sde.data_dim)  # dimensiona el default MLP desde la SDE
    model = make_model(model_name, **model_raw)
    # Receta genérica {name, kwargs} para el checkpoint model-agnóstico: la misma con la que se
    # construyó la red, así generate.py la reconstruye con make_model sin el config original.
    model_spec = {"name": model_name, "kwargs": dict(model_raw)}

    # --- salidas ---
    out_raw = dict(raw.get("out") or {})
    checkpoint = out_raw.get("checkpoint")
    loss_curve = out_raw.get("loss_curve")
    return RunSpec(
        sde=sde,
        model=model,
        data=data,
        config=config,
        model_spec=model_spec,
        checkpoint=pathlib.Path(checkpoint) if checkpoint else None,
        loss_curve=pathlib.Path(loss_curve) if loss_curve else None,
    )
