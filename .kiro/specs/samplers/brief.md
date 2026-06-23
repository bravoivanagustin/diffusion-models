# Brief: samplers

## Problem
El proyecto puede destruir datos hacia ruido (forward SDE) y aprender el score `s_θ(x,t)`, pero
todavía no puede **generar muestras**: falta el proceso reverso (Eje 2). Sin samplers no hay
resultados que comparar, que es el objetivo del estudio de ablación.

## Current State
`data_generation`, `mlp`, `sde` y `training` están terminados y testeados. No existe código de
sampleo (grep confirma: solo `sample_timesteps` de training). El score entrenado queda en checkpoints
con metadata (`sde_name`, `data_dim`, hiperparámetros de red).

## Desired Outcome
Un módulo `diffusion.samplers` que, dado un `ForwardSDE` y un score (`ScoreMLP` o callable analítico),
integre la SDE/ODE reversa y genere `x_0`. Cuatro samplers documentados, intercambiables sin
reentrenar, validados sobre VP/VE/sub-VP, con suite de pytest en verde.

## Approach
Espejar el patrón de `sde/`: `base.py` con un `ReverseSampler` (ABC) que aporta la grilla temporal,
los drifts reversos compartidos (`f - g²s` y `f - ½g²s`) y el driver `sample()`, y un archivo por
sampler que solo implementa `step()`. Registry/factory en `__init__.py` (`make_sampler` /
`available_samplers`, kwargs filtrados por firma). El score se inyecta como `Callable`, lo que conecta
con `ScoreMLP` y habilita el score analítico para validar. Generación config/checkpoint-driven
reusando `training.load_checkpoint`.

## Scope
- **In**: los 4 samplers (Euler–Maruyama, PF-ODE, Heun, predictor–corrector) sobre SDEs escalares
  (VP/VE/sub-VP); base ABC + registry/factory; captura opcional de trayectoria; CLI/generación
  config-driven desde checkpoint; tests (contrato + determinismo + validación con score analítico);
  doc del módulo.
- **Out**: dinámica reversa validada de CLD; visualización/ploteo y métricas (FID/IS); U-Net/Fase 2.

## Boundary Candidates
- Driver de integración compartido (grilla temporal + drifts reversos + loop) vs. el `step()` propio
  de cada sampler.
- Inyección del score como `Callable` (red entrenada o score analítico) — seam de modularidad.
- Capa de generación config/checkpoint-driven (CLI) vs. el núcleo de sampleo.

## Out of Boundary
- Entrenamiento / pesado HSM de CLD (vive en `training`/`sde`).
- Evaluación, visualización y métricas.

## Upstream / Downstream
- **Upstream**: `sde` (drift/diffusion, prior), `mlp` (ScoreMLP como score_fn), `training`
  (load_checkpoint + metadata).
- **Downstream**: un futuro módulo de **evaluación/visualización** de Fase 1 (consume trayectorias y
  muestras); la matriz de experimentos 4×4; Fase 2 (imágenes/U-Net).

## Existing Spec Touchpoints
- **Extends**: ninguno (primer spec del proyecto).
- **Adjacent**: `sde`, `mlp`, `training` (módulos, no specs) — no duplicar su responsabilidad.

## Constraints
- La red es variable de control: el sampler no la modifica. Cambiar de sampler no reentrena.
- Determinístico donde corresponde (PF-ODE, Heun); reproducible vía `torch.Generator` donde es
  estocástico (EM, PC).
- `float32`, `t` aceptado como `(B,)` y `(B,1)`, estabilidad en `t→0` (piso `t_eps`).
- Python 3.14 / torch CPU; convención: doc en `docs/project/` + suite de pytest en verde.
