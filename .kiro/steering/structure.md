# Estructura del Proyecto

## Filosofía de organización

**Layered por concepto del pipeline de difusión.** El código bajo `src/diffusion/` se divide en un
módulo por etapa (`data_generation` → `mlp` → `sde` → `training` → samplers). Las dependencias fluyen
hacia adelante: `training` combina `data_generation` (los `x_0`), `mlp` (la red) y `sde` (el forward),
sin ciclos. La **documentación en `docs/` es la fuente de verdad** del alcance y la teoría; el código
la sigue.

## Patrones de directorios

### Paquete de código
**Ubicación**: `diffusion-models/src/diffusion/<módulo>/`
**Patrón**: cada módulo es un paquete con `base.py` (clase base abstracta), implementaciones
concretas y un `__init__.py` que expone el **registry + factory** y el `__all__` público.
**Ejemplo**: `sde/` → `base.ForwardSDE`, `variants.py` (VP/VE/sub-VP), y
`__init__` con `make_sde` / `available_sdes` / `REGISTRY`.

### Documentación
**Ubicación**: `docs/project/` (overview, ejes experimentales, crónica fechada, doc por módulo,
referencias) y `docs/knowledge/` (notas teóricas propias). `docs/papers/` guarda los PDFs ancla.
**Regla**: **cada módulo de código nuevo suma su doc** `docs/project/<módulo>.md`.

### Tests
**Ubicación**: `diffusion-models/tests/`
**Patrón**: una suite por módulo — `tests/test_<módulo>.py`. Debe quedar **en verde** antes de
avanzar al siguiente módulo.

### Scripts / CLIs
**Ubicación**: `diffusion-models/scripts/` (p. ej. `data_generation.py`, `train.py`). Algunos módulos
también ofrecen `python -m diffusion.<módulo>` vía `__main__.py`.

### Datos generados
**Ubicación**: `diffusion-models/data/` — gitignored y reproducible desde `--seed`.

## Convenciones de nombres

- **Módulos / archivos**: `snake_case` (`score_mlp.py`, `data_generation/`).
- **Clases**: `PascalCase` (`ScoreMLP`, `ForwardSDE`, `VPSDE`, `TrainConfig`).
- **Factories**: `make_<cosa>(name, **kwargs)`; introspección por nombre con `available_<cosas>()`.
- **Idioma**: docstrings y docs en **español** (voz del autor); términos técnicos en su forma
  convencional (drift, score, sampler, NFE, DSM, …).

## Organización de imports

```python
# Import público SIN prefijo `src.` (lo resuelve el pythonpath del pyproject):
from diffusion.mlp import ScoreMLP
from diffusion.sde import make_sde, available_sdes
from diffusion.training import TrainConfig, train

# Dentro del código, imports relativos entre submódulos del mismo paquete:
from .base import ForwardSDE
from .variants import VPSDE, VESDE, SubVPSDE
```

## Principios de organización del código

- **Base abstracta + variantes + registry**: agregar una variante = nueva clase concreta registrada
  en el `REGISTRY` del `__init__`; ningún caller debería cambiar.
- **Regla de reentrenamiento**: tocar el **forward SDE (Eje 1)** obliga a reentrenar; cambiar el
  **sampler (Eje 2)** no.
- **`data_dim` por variante**: la `ScoreMLP` se instancia con `data_dim=sde.data_dim` (2 en la
  Fase 1 toy 2D).

---
_Documentar patrones, no árboles de archivos. Un archivo nuevo que sigue el patrón no debería exigir
actualizar este documento._
