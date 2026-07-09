# Requirements Document

## Project Description (Input)
Generalizar `sde` (familia escalar VP/VE/sub-VP) y `samplers` para operar sobre **cualquier event
shape** —no solo el toy 2D `(B, data_dim)` sino también tensores de imagen `(B, C, H, W)`— por
broadcasting, sin hardcodear la forma concreta `(3, 64, 64)`.

**Quién tiene el problema:** el autor del TP, con el pipeline de difusión ya listo para imágenes en
casi todos los eslabones (`ScoreUNet`, `train(sde, model, data, config)` model+data-agnóstico, y la
fuente `infinite_batches` que emite `(B, 3, 64, 64)` en `[-1,1]`), pero que **no puede entrenar ni
samplear imágenes** porque `sde` y `samplers` asumen datos planos.

**Situación actual:** `sde` (VP/VE/sub-VP) y `samplers` operan solo sobre `(B, data_dim)`. Los
coeficientes dependientes de `t` salen `(B, 1)` y no broadcastean contra `(B, C, H, W)`; el prior del
sampler se arma con una forma plana; la geometría del dato se expresa como un entero (`data_dim`) que
viaja por la metadata de checkpoint. Los cuerpos de los samplers ya son agnósticos al rango. CLD no
existe en el código (eliminado del proyecto): no aplica.

**Qué debe cambiar:** que la familia escalar de `sde` y los `samplers` funcionen sobre event shapes de
rango arbitrario, **backward-compatible** con el toy 2D, de modo que las imágenes `(B, 3, 64, 64)`
fluyan por perturbación, target del score, muestreo del prior, integración reversa y generación
desde checkpoint, sin fijar dimensiones en el código.

## Introduction

Este documento define los requisitos de `nd-shapes`: la generalización de la familia escalar de SDEs
(VP/VE/sub-VP) y de los samplers para operar sobre datos de **cualquier forma de evento** —el toy 2D
`(B, 2)` y, sobre todo, imágenes `(B, C, H, W)`— por broadcasting sobre las dimensiones de evento. Los
requisitos describen **comportamiento observable en shapes y valores** (qué formas entran y salen, qué
se mantiene invariante para 2D, qué debe funcionar para imágenes), no la implementación interna, que
se decide en diseño. Es el último bloqueo del camino de imágenes de Fase 2: red, entrenamiento
desacoplado y fuente de datos ya están entregados.

## Boundary Context

- **In scope**: broadcasting N-D de la familia escalar (VP/VE/sub-VP) en `sde`; muestreo del prior y
  generación de muestras N-D en `samplers`; la forma del dato como parámetro (entero para "plano" o
  forma multidimensional para imágenes) expuesta a los consumidores; el transporte de esa forma por la
  metadata de checkpoint / configuración para la generación; compatibilidad idéntica con el toy 2D;
  tests parametrizados 2D + forma tipo-imagen; y la verificación (con ajuste si hiciera falta) de que
  el pesado de la pérdida DSM sea N-D-safe.
- **Out of scope**: CLD (no existe en el código); la red `ScoreUNet` y la fuente `image-data-source`
  (ya entregadas); corridas reales de entrenamiento en GPU, métricas FID / IS; visualización y
  des-normalización para ver imágenes; la evaluación / visualización de Fase 1.
- **Adjacent expectations**: depende del contrato `(B, 3, 64, 64)` en `[-1,1]` de `image-data-source`
  y del formato de checkpoint model-agnóstico de `train-decoupling` (la forma del dato pasa a viajar en
  su metadata — coordinación en ese seam). **No posee** la red, el dato, el loop de entrenamiento (solo
  toca el pesado DSM en la medida necesaria) ni la evaluación.

## Requirements

### Requirement 1: Familia escalar de SDEs sobre event shapes arbitrarios
**Objective:** Como autor, quiero que VP/VE/sub-VP operen sobre datos de cualquier forma de evento,
para ruidear y fabricar el target del score tanto de puntos 2D como de imágenes con el mismo código.

#### Acceptance Criteria
1. When se invoca `perturb`, `score_target`, `marginal_prob` o `sde` con estado `x`/`x0` de shape
   `(B, *E)` para cualquier forma de evento `E` (p. ej. `(2,)` o `(3, 64, 64)`), the proceso forward
   `sde` shall producir salidas cuyas shapes coincidan/broadcasteen correctamente con `(B, *E)`, sin
   errores de shape.
2. When `perturb` recibe `x0` de shape `(B, *E)`, the proceso forward `sde` shall devolver `(x_t, eps)`
   de shape `(B, *E)` en `float32`.
3. When `score_target` recibe estado de shape `(B, *E)`, the proceso forward `sde` shall devolver un
   `score` de shape `(B, *E)` y un peso por muestra.
4. When se pasa `t` como `(B,)` o `(B, 1)` junto a un estado de shape `(B, *E)`, the proceso forward
   `sde` shall producir el mismo resultado en ambos casos y aplicar los coeficientes dependientes de
   `t` broadcasteando sobre todas las dimensiones de evento.
5. The proceso forward `sde` shall producir salidas finitas (sin `NaN` ni `Inf`) para formas de imagen
   como `(3, 64, 64)`.

### Requirement 2: Muestreo y generación N-D en los samplers
**Objective:** Como autor, quiero que los cuatro samplers generen muestras de cualquier forma de
evento, para poder samplear imágenes reusando el mismo score.

#### Acceptance Criteria
1. When se solicita `sample(n_samples)` para una SDE cuya forma de evento es `E`, the sampler shall
   muestrear el prior y devolver `x_0`, ambos de shape `(n_samples, *E)` en `float32`.
2. The sampler shall producir muestras finitas para formas de imagen (p. ej. `(3, 64, 64)`) en los
   cuatro samplers (Euler–Maruyama, PF-ODE, Heun, predictor–corrector).
3. Where se solicita la trayectoria, the sampler shall devolverla con shape `(n_steps+1, n_samples, *E)`.
4. When un sampler integra sobre un estado de shape `(B, *E)`, the sampler shall expandir y pasar `t`
   de modo que los coeficientes de la SDE broadcasteen contra `(B, *E)`.

### Requirement 3: La forma del dato como parámetro (event shape)
**Objective:** Como autor, quiero declarar la geometría del dato sin fijar dimensiones en el código,
para que el mismo marco sirva a 2D y a imágenes de cualquier tamaño.

#### Acceptance Criteria
1. When se construye una SDE indicando la forma del dato como un entero (dato plano, forma `(d,)`) o
   como una forma multidimensional (p. ej. `(C, H, W)`), the proceso forward `sde` shall aceptarla y
   exponerla para que los consumidores (samplers) armen el prior.
2. If se indica una forma inválida (alguna dimensión menor que 1), then the proceso forward `sde` shall
   rechazarla con un error claro.
3. The sistema shall soportar formas de evento de rango arbitrario sin fijar en el código la forma
   `(3, 64, 64)` ni ninguna otra específica.

### Requirement 4: Generación config/checkpoint-driven para imágenes
**Objective:** Como autor, quiero reconstruir y samplear una SDE de imágenes desde un checkpoint,
para generar muestras de imágenes sin reescribir la configuración a mano.

#### Acceptance Criteria
1. When se guarda el checkpoint de un modelo entrenado sobre datos de forma `E`, the plomería de
   checkpoint/configuración shall transportar la forma de evento en la metadata.
2. When se ejecuta la generación desde un checkpoint cuya forma de evento es `E`, the generador shall
   reconstruir la SDE con esa forma y producir muestras de shape `(n_samples, *E)`.
3. If la metadata del checkpoint no permite reconstruir la forma del dato, then the generador shall
   fallar con un error claro.

### Requirement 5: Compatibilidad con el toy 2D y verificación
**Objective:** Como autor, quiero que la generalización no rompa la Fase 1, para seguir corriendo el
estudio 2D sin cambios mientras habilito imágenes.

#### Acceptance Criteria
1. While se opera sobre datos 2D de shape `(B, 2)`, the sistema shall comportarse de forma idéntica al
   comportamiento previo (misma salida numérica dada la misma entrada/seed), sin regresiones en la
   suite de pruebas existente.
2. The suite de pruebas shall cubrir `sde` y `samplers` parametrizados sobre una forma 2D `(2,)` y una
   forma tipo-imagen chica, con un chequeo end-to-end de que `sample()` devuelve `(n_samples, *E)`.
3. When la pérdida DSM combina el peso por muestra con el error sobre un estado de shape `(B, *E)`, the
   loop de entrenamiento shall aplicar el peso sin errores de broadcasting (objetivo del camino de
   imágenes).
