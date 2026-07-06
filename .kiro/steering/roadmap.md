# Roadmap

## Overview

Reestructurar la red de score en un subpaquete `diffusion/models/` que separe las piezas compartidas (`layers.py`) de cada red concreta (`mlp.py`, `unet.py`), como preparación para la Fase 2: una **ScoreUNet convolucional escrita desde cero** para imágenes. El trabajo se divide en dos pasos que no se pisan: primero el refactor puro (mover código sin cambiar comportamiento, suite de pytest en verde), y recién después la U-Net sobre esa base limpia.

**Decisión de alcance (05/07/2026):** la U-Net de Fase 2 **deja de ser "de librería"** (diffusers / denoising-diffusion-pytorch, como decían `ejes.md` y `CLAUDE.md`) y pasa a **construirse a mano** en `models/unet.py`. El estudio de ablación sigue válido: la red sigue siendo la variable de control, fija en todas las celdas. `docs/project/ejes.md` y `CLAUDE.md` deben actualizarse para reflejar esto (entra en el alcance de la spec `score-unet`).

## Approach Decision

- **Chosen**: refactor directo (sin spec) + spec nueva `score-unet` para la U-Net a mano.
- **Why**: el refactor es movimiento mecánico de código, protegido por la suite existente — no necesita gate de spec. La U-Net es trabajo nuevo con decisiones de diseño reales (canales, atención, inyección de tiempo, determinismo) y sí lo necesita.
- **Rejected alternatives**: (a) una sola spec que cubra refactor + U-Net — proceso pesado para la parte mecánica y mezcla dos pasos que conviene mantener separados; (b) solo el refactor sin brief — se perdería el contexto de diseño de la U-Net acordado en discovery.

## Scope

- **In**: subpaquete `diffusion/models/` (`layers.py`, `mlp.py`, `unet.py`, `base.py`), ScoreUNet escrita a mano, actualización de imports internos, docs y steering afectados.
- **Out**: dataset final de imágenes (sigue a definir), evaluación FID / IS, entrenamiento de imágenes en GPU, y la evaluación / visualización de Fase 1 (módulo aparte, no empezado).

## Constraints

- **Red determinística**: la U-Net usa GroupNorm (determinístico) y **no lleva dropout**; la mitigación de memorización se apoya en flip horizontal + EMA. Igual que el MLP: nada de capas estocásticas dentro de la red.
- **Regla de layers.py**: solo entra lo que ambas redes usan **sin modificar** (`_ACTIVATIONS`, `_make_activation`, `SinusoidalEmbedding`). Cada red mantiene su propio bloque residual: el `ResidualBlock` lineal es del MLP; el bloque convolucional con inyección de tiempo es de la U-Net. Comparten la idea, no el código.
- **Stack**: Python 3.14 + torch 2.12 CPU; los smoke tests (`if __name__ == "__main__"`) de cada archivo deben correr en CPU. pytest en verde en cada paso.

## Boundary Strategy

- **Why this split**: el refactor no cambia ningún comportamiento observable (mismos parámetros, misma salida) y la suite existente lo verifica; separarlo de la U-Net hace que, si algo se rompe, se sepa cuál de los dos pasos fue.
- **Shared seams to watch**: la frontera `layers.py` ↔ redes (qué es compartido de verdad), y el acople actual de `training.trainer` con `ScoreMLP` (instanciación hardcodeada en `trainer.py`), que la spec `score-unet` va a tener que abordar.

## Existing Spec Updates

(ninguna — la spec `samplers` está completa y los samplers ya son agnósticos del score, que se inyecta como callable)

## Direct Implementation Candidates

- [x] models-restructure — mover `diffusion/mlp/score_mlp.py` → `diffusion/models/mlp.py`, extraer `layers.py` (`_ACTIVATIONS`, `_make_activation`, `SinusoidalEmbedding`), agregar `base.py` (Protocol `ScoreModel`) y `__init__.py` con re-exports; eliminar el paquete `diffusion/mlp/`; actualizar imports en `training/trainer.py`, `samplers/__main__.py` y los 4 archivos de test (incl. ~10 `importorskip("diffusion.mlp")` en `test_samplers.py`); actualizar `docs/project/mlp.md`, `docs/project/sde.md`, `.claude/CLAUDE.md`, `.kiro/steering/structure.md` y `.kiro/steering/testing.md`. Por qué directo: movimiento puro de código, comportamiento idéntico, suite existente lo protege. Los notebooks no se tocan (importan vía `diffusion.training`).

## Specs (dependency order)

- [ ] score-unet — ScoreUNet convolucional escrita a mano en `models/unet.py` (bloque residual conv con inyección de tiempo, self-attention en 16×16, down/upsampling, encoder + bottleneck + decoder con skips). Dependencies: models-restructure (item directo, debe completarse antes)
