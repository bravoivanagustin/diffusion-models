# Brief: nd-shapes

## Problem
El pipeline de difusión ya está listo para imágenes en casi todos sus eslabones —`ScoreUNet` (red),
`train(sde, model, data, config)` model+data-agnóstico, y la fuente `infinite_batches` que emite
`(B, 3, 64, 64)` en `[-1, 1]`— pero `sde` y `samplers` **siguen asumiendo datos planos `(B, data_dim)`**.
Entrenar o samplear imágenes hoy rompe por broadcasting: los coeficientes de la SDE salen `(B, 1)` y no
encajan contra `(B, C, H, W)`.

## Current State
- Entregados: `ScoreUNet`, `train` model+data-agnóstico (train-decoupling), `image-data-source`
  (`infinite_batches` → `(B, 3, 64, 64)` en `[-1,1]`). `models/base.py` ya documenta `x` como `(B,C,H,W)`.
- `sde` (familia escalar VP/VE/sub-VP) y `samplers` operan solo en `(B, data_dim)`. Bloqueos exactos
  (verificados en discovery):
  - `_expand_t` fuerza `t → (B, 1)` en `sde/base.py:173` y `samplers/base.py:213`.
  - `sample()` arma el prior como `(n_samples, self.sde.data_dim)` (samplers/base.py:147).
  - Los productos `beta*x` / `alpha*x0` / `std*eps` / `eps/std` (variants.py L52/62/172/182; base
    `perturb`/`score_target`) rompen con coeficiente `(B,1)` contra N-D.
- Los cuerpos de los samplers (`randn(x.shape)`, `reshape(B,-1).norm(...)`, `g²·s`, drifts) **ya
  generalizan**; el bloqueo real está solo en el prior shape y en el broadcasting de coeficientes.
- CLD **no existe en el código** (eliminado del proyecto): no aplica a esta generalización.

## Desired Outcome
`sde` (familia escalar) y `samplers` operan sobre **cualquier event shape sin hardcodear dimensiones**:
el toy 2D `(B, 2)` se comporta idéntico y las imágenes `(B, 3, 64, 64)` funcionan
(perturb/score_target/sde/marginal_prob y `sample()` dan shapes correctas y broadcast correcto). Suite
de pytest en verde, parametrizada sobre 2D y una forma tipo-imagen chica.

## Approach
Generalización mínima por broadcasting, **backward-compatible**:
1. **`_expand_t` ndim-aware** (en `sde/base.py` y `samplers/base.py`): reshape `t` a `(B, 1, 1, …)` con
   tantos `1` como dims sobrantes de `x`, para que todo coeficiente derivado de `t` broadcastee contra
   `x` de cualquier rango. El toy 2D queda idéntico.
2. **`data_dim` → forma de evento**: la SDE acepta un `int` (→ `(d,)`, compat 2D) o una tupla `(C,H,W)`
   y expone la event shape para que `sample()` arme el prior como `(n_samples, *event_shape)`.
   `prior_sampling` ya es shape-agnóstico (recibe la tupla del caller).
3. **Ripple mínimo de plomería**: la forma viaja en la metadata de checkpoint (`training/trainer.py`),
   en `config.py` y en `generate_from_checkpoint`, para que la generación reconstruya la SDE de
   imágenes (`make_sde(name, data_dim=forma)`).
4. **Tests**: parametrizar `sde` + `samplers` sobre `(2,)` y una forma tipo-imagen chica (p. ej.
   `(3, 8, 8)`, rápida en CPU); chequeo end-to-end de que `sample()` devuelve `(B, C, H, W)`.

Sin dependencias nuevas (broadcasting puro de torch; torchvision ya está para el dato).

## Scope
- **In**: broadcasting N-D de la familia escalar (VP/VE/sub-VP) en `sde` (base + variants); event-shape
  en `sde` y en el prior de `samplers`; `_expand_t` ndim-aware (ambos módulos); ripple mínimo de la
  forma en checkpoint meta / config / `generate.py`; tests parametrizados 2D + imagen-chica.
- **Out**: CLD (no existe); la U-Net y la fuente de imágenes (ya entregadas); corridas reales de
  entrenamiento de imágenes / GPU / FID-IS; visualización y des-normalización; evaluación de Fase 1.

## Boundary Candidates
- El broadcasting ndim-aware (`_expand_t` + coeficientes) — el corazón del cambio, compartido
  conceptualmente entre `sde` y `samplers`.
- La event-shape como geometría del dato: dónde vive (atributo de la SDE) y cómo fluye al prior y al
  checkpoint.
- La plomería de reconstrucción config/checkpoint-driven (transportar la forma int|tupla).

## Out of Boundary
- El loop de entrenamiento en sí (train ya es data-agnóstico). Solo se **verifica** que la pérdida DSM
  aplique el peso `(B,1)` de forma N-D-safe; si no lo es, es un ajuste chico dentro de esta spec o un
  follow-on (a decidir en diseño).
- La red, el dato, la evaluación/visualización.

## Upstream / Downstream
- **Upstream**: `sde` (familia escalar), `samplers`, y la plomería de `training` (checkpoint
  meta/config) de train-decoupling; la fuente `image-data-source` (contrato `(B,3,64,64)`), `ScoreUNet`.
- **Downstream**: entrenamiento y sampleo reales sobre imágenes (Fase 2); el futuro módulo de
  evaluación / visualización (consumirá muestras `(B, C, H, W)`).

## Existing Spec Touchpoints
- **Extends**: `samplers` (completa; su diseño acotaba datos 2D — esta spec levanta esa restricción,
  disparando su Revalidation Trigger implícito). Toca `sde` (no es spec; precede a Kiro).
- **Adjacent**: `train-decoupling` (formato de checkpoint model-agnóstico — la forma pasa a viajar en
  la meta); `image-data-source` (provee el `(B,C,H,W)`). No duplicar su responsabilidad.

## Constraints
- **Backward-compatible**: el toy 2D `(B, 2)` no cambia de comportamiento (misma salida, tests en verde).
- **No hardcodear `(3, 64, 64)`**: la generalización es por rango arbitrario.
- `float32`; `t` aceptado como `(B,)` y `(B,1)`; estabilidad en `t → 0` (pisos `t_eps` / `_std_eps`).
- Python 3.14 / torch 2.12 CPU; convención: doc en `docs/project/` + suite de pytest en verde en cada
  paso. Sin dependencias nuevas.
