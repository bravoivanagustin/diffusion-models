# Requirements Document

## Project Description (Input)
Módulo `diffusion.samplers`: el proceso reverso (Eje 2) del estudio de difusión.

**Quién tiene el problema:** el autor del TP de Cálculo Estocástico, que ya cuenta con los módulos
`data_generation`, `mlp`, `sde` y `training` (forward SDE + aprendizaje del score `s_θ(x,t)`)
terminados y testeados, pero todavía no puede **generar muestras**.

**Situación actual:** no existe ningún código de sampleo (solo `sample_timesteps` dentro de
`training`). El score entrenado vive en checkpoints con metadata (`sde_name`, `data_dim`,
hiperparámetros de red), listo para reusarse, pero falta el integrador del proceso reverso.

**Qué debe cambiar:** construir un módulo `diffusion.samplers` que, dado un `ForwardSDE` y un score
(la `ScoreMLP` entrenada o un callable analítico), integre numéricamente la SDE/ODE reversa y genere
`x_0` a partir de ruido. Debe implementar los **cuatro samplers** documentados en
`docs/project/ejes.md` —Euler–Maruyama, Probability-Flow ODE (Euler), Heun y predictor–corrector—,
intercambiables **sin reentrenar** (todos comparten el mismo score). La estructura espeja a `sde/`
(base ABC + un proceso por archivo + registry/factory `make_sampler`/`available_samplers`),
completamente modular y conectada con los demás módulos vía el score y el checkpoint.

## Introduction

Este documento define los requisitos del módulo `diffusion.samplers`, que cierra el **Eje 2** del
estudio de ablación: el proceso reverso que genera muestras `x_0` integrando numéricamente la SDE/ODE
inversa a partir de ruido, reusando el score aprendido. Los requisitos describen el **comportamiento
observable** del módulo (qué genera, cómo se selecciona y configura, qué garantías de
reproducibilidad y correctitud ofrece), no su implementación interna, que se decide en la fase de
diseño. El módulo es la variable estocástica del estudio: cambiar de sampler no reentrena la red.

## Boundary Context

- **In scope**: los cuatro samplers (Euler–Maruyama, PF-ODE, Heun, predictor–corrector) operando y
  **validados** sobre las SDEs escalares (VP, VE, sub-VP); selección de sampler por nombre con
  factory/registry; inyección del score como función `(x, t) → score` (red entrenada o score
  analítico); captura opcional de la trayectoria de integración; generación config/checkpoint-driven
  por CLI; validación matemática vía score analítico; documentación del módulo.
- **Out of scope**: la dinámica reversa **validada** de CLD (el estado aumentado queda como seam, su
  convergencia depende del pesado HSM pendiente en `training`/`sde`); cualquier visualización,
  ploteo o métrica (FID/IS); la U-Net y la Fase 2 de imágenes.
- **Adjacent expectations**: el módulo **depende** de `sde` (coeficientes de la SDE y muestreo del
  prior), de `mlp` (la red de score como función inyectable) y de `training` (carga de checkpoints y
  su metadata). **No posee** el entrenamiento, el pesado de la pérdida (incluido el HSM de CLD), ni
  la evaluación/visualización de resultados.

## Requirements

### Requirement 1: Generación de muestras por integración reversa
**Objective:** Como autor del TP, quiero generar muestras `x_0` integrando el proceso reverso desde el
prior de ruido, para obtener los resultados que el estudio de ablación necesita comparar.

#### Acceptance Criteria
1. When se solicita la generación de `N` muestras, the sampler shall devolver un tensor `x_0` de
   shape `(N, data_dim)` en `float32`.
2. The sampler shall iniciar la integración muestreando del prior `p_T` de la SDE y avanzar en tiempo
   decreciente hasta un tiempo terminal cercano a cero.
3. When se solicita la generación con el score provisto, the sampler shall integrar el proceso reverso
   sin modificar ni reentrenar la red.
4. The sampler shall producir salidas finitas (sin `NaN` ni `Inf`) para las SDEs escalares
   soportadas (VP, VE, sub-VP).
5. Where se solicita la captura de trayectoria, the sampler shall devolver además la secuencia de
   estados intermedios recorridos durante la integración.

### Requirement 2: Catálogo de los cuatro samplers documentados
**Objective:** Como autor, quiero los cuatro samplers del Eje 2, para cubrir la dimensión completa de
samplers de la matriz de experimentos 4×4.

#### Acceptance Criteria
1. The módulo de samplers shall ofrecer los samplers Euler–Maruyama, Probability-Flow ODE (Euler),
   Heun y predictor–corrector.
2. The sampler Euler–Maruyama shall integrar la SDE reversa inyectando ruido en cada paso
   (comportamiento estocástico).
3. The sampler Probability-Flow ODE shall integrar la ODE de flujo de probabilidad sin inyección de
   ruido (comportamiento determinístico).
4. The sampler Heun shall integrar la ODE de flujo de probabilidad con un esquema de segundo orden,
   con un costo observable de dos evaluaciones de score por paso.
5. The sampler predictor–corrector shall combinar un paso de SDE reversa con un número configurable de
   correcciones de Langevin por nivel de ruido.

### Requirement 3: Reuso del score sin reentrenar (intercambiabilidad del Eje 2)
**Objective:** Como autor, quiero variar el sampler manteniendo fijos la red y la SDE, para atribuir
toda diferencia medible a la elección del sampler y no a la ingeniería.

#### Acceptance Criteria
1. Where se cambia el sampler manteniendo el mismo score y la misma SDE, the módulo de samplers shall
   generar muestras sin requerir ningún reentrenamiento.
2. The sampler shall obtener el score mediante una función inyectada que recibe `(x, t)` y devuelve el
   score, de modo que tanto una red entrenada como un score analítico en forma cerrada sean
   utilizables sin cambios en el sampler.
3. The sampler shall not alterar los parámetros de la red durante la generación.

### Requirement 4: Selección de sampler por nombre
**Objective:** Como autor, quiero seleccionar e instanciar cualquier sampler por nombre de forma
uniforme, para construir las celdas del estudio con una interfaz común.

#### Acceptance Criteria
1. When se solicita un sampler por nombre, the factory de samplers shall instanciar el sampler
   correspondiente configurado con la SDE y el score provistos.
2. The factory de samplers shall exponer la lista de nombres de samplers disponibles.
3. If se solicita un nombre de sampler desconocido, then the factory de samplers shall rechazar la
   solicitud con un error que enumere las opciones válidas.
4. When se pasan parámetros que no aplican al sampler elegido, the factory de samplers shall
   descartarlos sin fallar, de modo que un llamador genérico pueda pasar siempre el mismo conjunto de
   parámetros.

### Requirement 5: Determinismo y reproducibilidad
**Objective:** Como autor, quiero resultados reproducibles y determinismo donde corresponde, para que
las celdas del estudio sean comparables entre sí.

#### Acceptance Criteria
1. When se ejecuta un sampler determinístico (PF-ODE o Heun) dos veces con la misma entrada inicial,
   the sampler shall producir resultados idénticos.
2. When se ejecuta un sampler estocástico (Euler–Maruyama o predictor–corrector) dos veces con el
   mismo generador de aleatoriedad sembrado, the sampler shall producir resultados idénticos.
3. When se ejecuta un sampler estocástico con semillas distintas, the sampler shall producir
   resultados distintos.

### Requirement 6: Generación config/checkpoint-driven por CLI
**Objective:** Como autor, quiero generar muestras a partir de un checkpoint entrenado vía
configuración/CLI, para reproducir celdas del estudio sin escribir código.

#### Acceptance Criteria
1. When se ejecuta la generación a partir de un checkpoint, the generador shall reconstruir la SDE y
   la red desde la metadata del checkpoint (`sde_name`, `data_dim`, hiperparámetros de red).
2. When la generación por CLI completa correctamente, the generador shall guardar las muestras
   generadas en disco.
3. Where se solicita en la configuración, the generador shall guardar también la trayectoria de la
   integración.
4. If el checkpoint o la configuración referenciada no existe o es inválida, then the generador shall
   rechazar la ejecución con un mensaje de error claro.

### Requirement 7: Correctitud matemática verificable
**Objective:** Como autor, quiero validar la correctitud de los samplers de forma independiente del
entrenamiento de la red, para asegurar que la matemática del proceso reverso está bien implementada.

#### Acceptance Criteria
1. When se ejecuta un sampler con el score analítico en forma cerrada de un objetivo conocido, the
   sampler shall generar muestras cuya media y covarianza coincidan con las del objetivo dentro de una
   tolerancia de Monte Carlo.
2. The suite de pruebas shall cubrir cada uno de los cuatro samplers sobre las tres SDEs escalares
   (VP, VE, sub-VP).
3. When `data_generation`, `mlp`, `sde` y un sampler se conectan en una corrida de extremo a extremo,
   the suite de pruebas shall verificar que las shapes encajan a lo largo de toda la cadena.

### Requirement 8: Robustez numérica e interfaz temporal
**Objective:** Como autor, quiero que el sampler respete las convenciones numéricas del proyecto, para
que sea intercambiable con los demás módulos sin sorpresas.

#### Acceptance Criteria
1. The sampler shall aceptar el tiempo `t` tanto en shape `(B,)` como `(B, 1)` produciendo el mismo
   resultado.
2. While la integración se aproxima a `t = 0`, the sampler shall mantener la estabilidad numérica
   evitando divisiones por cero.
3. The sampler shall operar en `float32` de forma consistente con `sde`, `mlp` y `training`.
4. The sampler shall permitir configurar el número de pasos de integración.
