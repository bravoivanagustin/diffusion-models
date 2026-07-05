# Convenciones de Testing

`pytest` es parte del entregable, no un extra. Cada módulo se entrega con su suite **en verde** antes
de avanzar al siguiente (desarrollo incremental). Suites actuales: `data_generation`, `mlp`, `sde`
(47 tests), `training` (17 tests), `samplers` (134 tests).

## Organización

- **Una suite por módulo**: `diffusion-models/tests/test_<módulo>.py`. Sin clases; funciones
  `test_*` agrupadas por comentarios de sección (registry/factory, shapes/dtype, límites, seams…).
- **Correr**: `python -m pytest -q` desde `tp-final/` o desde `diffusion-models/` (pytest resuelve el
  rootdir vía `pyproject.toml`).

## Import diferido de torch

`torch` es opcional para los tests de código liviano. Al tope del archivo que lo necesita:

```python
import pytest
torch = pytest.importorskip("torch")   # skipea la suite entera si no hay torch
from diffusion.sde import make_sde      # imports del paquete DESPUÉS del importorskip
```

Para un test puntual que cruza submódulos, importar adentro de la función (`from diffusion.mlp import
ScoreMLP`) para no acoplar la colección del archivo.

## Qué se testea (patrones recurrentes)

- **Contrato de shape/dtype**: salidas con la shape esperada y `dtype == torch.float32`;
  `torch.isfinite` sobre todo lo que se devuelve. `t` aceptado como `(B,)` y `(B,1)` (mismo resultado).
- **Registry/factory**: `available_*()` == conjunto esperado; la factory devuelve el tipo correcto;
  nombre desconocido → `ValueError`; kwargs que no aplican se descartan sin romper.
- **Determinismo / reproducibilidad**: mismo `generator`/`seed` → tensores idénticos
  (`torch.equal`); seeds distintos → distintos. La red es determinística (sin dropout/batchnorm).
- **Validación matemática independiente** (no solo "no crashea):
  - **Forma cerrada vs diferencias finitas**: p. ej. la ODE de la varianza
    `dΣ/dt == 2·f·Σ + g²` chequeada con diferencias finitas (`rtol≈1e-2`).
  - **Score analítico**: comparar `score_target` contra la fórmula cerrada (`-eps/std`), no contra
    sí mismo.
  - **Monte Carlo**: validar un kernel de perturbación nuevo simulando el forward por
    Euler–Maruyama (`n≈40000`) y comparando media/varianza/covarianza empíricas contra la forma
    cerrada (tolerancia relativa ~5%).
- **Seams entre módulos**: un test mínimo que conecta las piezas reales (p. ej. `sde.perturb` →
  `ScoreMLP(data_dim)` → `sde.score_target`) y verifica que las shapes encajan.
- **Límites y casos borde**: `t→0` y `t→T` (medias/desvíos esperados), `data_dim` arbitrario
  parametrizado, invariantes (VE sin drift, sub-VP con varianza estrictamente bajo VP).

## Estilo

- `@pytest.mark.parametrize` sobre nombres de variante (`SCALAR = ["vp","ve","sub_vp"]`) y sobre
  dimensiones (`[1,3,7]`).
- Helpers chicos para fixtures de datos (`_x0_t(n, dim, seed)`), no fixtures globales pesadas.
- Tolerancias **explícitas y justificadas** por el método (`atol`/`rtol`); más laxas para Monte
  Carlo, ajustadas para forma cerrada.
- Tests en CPU, rápidos; el `n` grande de Monte Carlo es la excepción consciente.

> Nota OneDrive: pytest puede emitir un `PytestCacheWarning` (`WinError 5`) inofensivo. Silenciable
> con `-p no:cacheprovider`.

---
_Patrones de testeo, no la lista de tests. Tests nuevos que siguen estos patrones no exigen actualizar este doc._
