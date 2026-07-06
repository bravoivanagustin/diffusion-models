# Brief: score-unet

## Problem

La Fase 2 del TP (imágenes) necesita una red de score que opere sobre tensores `(B, C, H, W)`; la `ScoreMLP` actual solo maneja datos 2D `(B, 2)`. Sin esta red no hay forma de correr la matriz SDE × sampler sobre imágenes ni de medir FID / IS.

## Current State

- `diffusion.models` (post-refactor `models-restructure`, ver roadmap): `layers.py` con las piezas compartidas (`_ACTIVATIONS`, `_make_activation`, `SinusoidalEmbedding`), `mlp.py` con `ScoreMLP` + su `ResidualBlock` lineal, `base.py` con el Protocol `ScoreModel` (`(x, t) → misma shape que x`). La U-Net no existe.
- **Decisión de alcance (05/07/2026):** la U-Net se escribe **a mano** — revierte lo que decían `ejes.md` y `CLAUDE.md` ("se reutiliza una U-Net de librería"). Esos dos documentos deben actualizarse como parte de esta spec.
- **Acople conocido**: `training/trainer.py` instancia `ScoreMLP` hardcodeado (construcción en `train` y reconstrucción desde checkpoint en `load_checkpoint`); el Protocol solo documenta el contrato, no desacopla el trainer. Esta spec debe decidir cómo el entrenamiento construye la red (¿extensión del trainer acá, o spec aparte de training-Fase-2?).
- Los samplers ya son agnósticos: reciben el score como callable inyectable. Verificar en gap analysis si sus supuestos de shape aceptan `(B, C, H, W)` o asumen `(B, D)`.

## Desired Outcome

`ScoreUNet` en `diffusion/models/unet.py`: forward `(x: (B, C, H, W), t: (B,)) → (B, C, H, W)`, enteramente determinística, con smoke test `if __name__ == "__main__"` que instancia, corre un forward con `x` de shape `(2, 3, 64, 64)`, verifica que la salida tenga esa misma shape y reporta el conteo de parámetros. Suite de pytest propia en verde y doc del módulo en `docs/project/`.

## Approach

Escrita desde cero, importando `SinusoidalEmbedding` y `_make_activation` desde `layers.py`. Componentes a definir (no existen aún):

- **Bloque residual convolucional**: GroupNorm → SiLU → Conv, con el vector de tiempo sumado adentro del bloque (inyección por broadcast sobre canales). Es una clase distinta del `ResidualBlock` lineal del MLP — comparten la idea (residual con skip), no el código.
- **Self-attention** en la resolución 16×16.
- **Down/upsampling** y el ensamblado final: encoder + bottleneck + decoder con skip connections.
- Footguns conocidos a cuidar en diseño: contabilidad de canales, concatenación de skips, broadcasting del tiempo.

## Scope

- **In**: `models/unet.py` (bloque conv residual, atención, down/up, `ScoreUNet`), su suite de tests, su doc en `docs/project/`, y la actualización de `ejes.md` / `CLAUDE.md` por el cambio "U-Net de librería → a mano".
- **Out**: dataset final de imágenes (sigue a definir; el smoke test usa `(2, 3, 64, 64)` como target de shape), evaluación FID / IS, corridas de entrenamiento reales en GPU, evaluación / visualización de Fase 1.

## Boundary Candidates

- La red en sí (`unet.py`) — el corazón de la spec.
- La adaptación del `training` para construir/checkpointear una red que no sea `ScoreMLP` — decidir en requirements si entra en esta spec o se separa.

## Out of Boundary

- Cambios a `ScoreMLP` o a `layers.py` más allá de imports (la arquitectura del MLP es variable de control, queda fija).
- Los samplers (Eje 2) — ya implementados y score-agnósticos; solo se verifica compatibilidad de shapes.
- Elección del dataset de imágenes y pipeline de datos de Fase 2.

## Upstream / Downstream

- **Upstream**: `models-restructure` (item directo del roadmap — debe completarse antes); `layers.py` (SinusoidalEmbedding, activaciones); `diffusion.sde` (los forward VP/VE/sub-VP con `data_dim` configurable — verificar que `perturb`/`score_target` operen sobre tensores imagen).
- **Downstream**: entrenamiento de Fase 2 (DSM sobre imágenes), evaluación FID / IS, la matriz 3×4 sobre imágenes.

## Existing Spec Touchpoints

- **Extends**: ninguna (la spec `samplers` está completa; `training`, `mlp`, `sde`, `data_generation` no tienen spec propia).
- **Adjacent**: `samplers` (contrato de score callable y supuestos de shape), `training` (acople con `ScoreMLP` en trainer y checkpoints).

## Constraints

- **Determinismo**: GroupNorm (determinístico), **sin dropout** ni capas estocásticas; la mitigación de memorización se corre a flip horizontal + EMA (fuera de la red).
- **Variable de control**: una vez definida, la arquitectura queda fija (mismos hiperparámetros) en las 12 celdas del estudio — no se ajusta por celda.
- **Stack**: Python 3.14, torch 2.12 CPU para smoke tests y suite; el entrenamiento real en imágenes requerirá GPU (fuera de esta spec).
- **Convenciones del repo**: import diferido de torch en tests (`pytest.importorskip`), doc del módulo en `docs/project/`, español para docs, smoke test `__main__` en cada archivo.
