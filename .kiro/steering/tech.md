# Stack Tecnológico

## Arquitectura

Paquete Python `diffusion` con layout `src/`. Un **módulo por concepto** del pipeline de difusión
(`data_generation`, `models`, `sde`, `training`, …). La estocasticidad vive **alrededor** de la red —en
el dato, en el forward SDE y en el sampler—, nunca dentro de ella. Cada familia de variantes
(formas de puntos, SDEs) se expone por un par **registry + factory** (`make_distribution`,
`make_sde`) sobre una clase base abstracta (`PointDistribution`, `ForwardSDE`).

## Tecnologías centrales

- **Lenguaje**: Python ≥3.10 (entorno real: 3.14 en Windows).
- **Cómputo**: `torch` (probado en 2.12.0+cpu; corre en CPU, sin GPU en Fase 1), `numpy`.
- **Datos / viz**: `scikit-learn` (formas toy), `matplotlib` (previews y figuras).
- **Config / tests**: `pyyaml` (corridas config-driven), `pytest`.
- **Entorno**: gestionado con **`uv`** (`uv.lock` en la raíz).

## Estándares de desarrollo

### Desarrollo incremental con tests
Construir **un módulo a la vez** y entregarlo con su suite de pytest **en verde** antes de avanzar.
No acumular módulos sin tests. Diseñar para testeabilidad.

### Imports diferidos de dependencias pesadas
`torch` se importa **dentro de las funciones** que lo necesitan, no a nivel de módulo, para que el
código liviano (formas, configs) se pueda testear sin él. Los tests que requieren torch usan
`pytest.importorskip("torch")`.

### La red es determinística
Nada de dropout, batchnorm ni capas estocásticas dentro de la red. Mantenerla **fija** entre celdas
del estudio: variar la arquitectura rompería la ablación.

### Tipos de datos
Salidas en `float32`. Helpers torch con import diferido conviven con la ruta numpy.

## Entorno de desarrollo

```bash
# Tests — desde tp-final/ o desde diffusion-models/ (pytest resuelve el rootdir):
python -m pytest -q

# CLIs (desde diffusion-models/):
python scripts/data_generation.py --shape two_moons --dim 2 --n-samples 2000 --seed 0 \
    --out data/two_moons.npz --preview data/two_moons.png
python scripts/train.py config/vp_mixture.yaml      # corrida config-driven
# Algunos módulos también exponen `python -m diffusion.sde` / `-m diffusion.training`.
```

## Decisiones técnicas clave

- **Sin `pip install -e .`**: el import público sin prefijo `src.` lo resuelve el `pythonpath` del
  `pyproject.toml` (`from diffusion.sde import make_sde`).
- **`pyproject.toml` y `uv.lock` en la raíz** del repo (`tp-final/`), apuntando al código bajo
  `diffusion-models/src`.
- **Factory tolerante a kwargs** (`make_sde`): descarta los kwargs que no aplican a la variante, así
  un caller genérico pasa siempre el mismo conjunto de parámetros.
- **Corridas config-driven**: una celda del estudio = un archivo YAML (`build_run` → `train` →
  `save_checkpoint`), para reproducir la matriz de experimentos.
- **Nota OneDrive**: el repo vive bajo OneDrive; pytest puede emitir un `PytestCacheWarning`
  (`WinError 5`) inofensivo. Silenciable con `-p no:cacheprovider`.

---
_Documentar estándares y patrones, no cada dependencia._
