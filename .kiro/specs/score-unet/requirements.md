# Requirements Document

## Project Description (Input)
La Fase 2 del TP (imágenes) necesita una red de score convolucional: la `ScoreMLP` actual solo opera sobre datos 2D `(B, 2)`, y sin una U-Net no se puede correr la matriz SDE × sampler sobre imágenes ni medir FID / IS. Por decisión del autor (05/07/2026) la U-Net se escribe **a mano** — revierte el plan de "U-Net de librería" de `ejes.md` / `CLAUDE.md`, que deben actualizarse como parte de esta spec.

El entregable es una `ScoreUNet` en `diffusion/models/unet.py`: bloque residual convolucional (GroupNorm → SiLU → Conv, con inyección del embedding de tiempo dentro del bloque), self-attention en la resolución 16×16, down/upsampling, y encoder + bottleneck + decoder con skip connections, reusando `SinusoidalEmbedding` y `_make_activation` de `models/layers.py`. Contrato: `(x: (B, C, H, W), t: (B,)) -> (B, C, H, W)` (satisface el Protocol `ScoreModel` estructuralmente). La red es **enteramente determinística** (GroupNorm, sin dropout; la mitigación de memorización se apoya en flip horizontal + EMA, fuera de la red) y su arquitectura queda **fija** como variable de control en las 12 celdas del estudio de ablación.

Verificación esperada: smoke test `__main__` con `x` de shape `(2, 3, 64, 64)` (salida de la misma shape + conteo de parámetros), suite de pytest propia en verde en CPU, y doc del módulo en `docs/project/`. Contexto completo y touchpoints en `brief.md` (el trainer instancia `ScoreMLP` hardcodeado — decidir si su desacople entra en esta spec — y hay que verificar los supuestos de shape de `sde`/`samplers` para tensores imagen).

## Introduction

Este documento define los requisitos de la **`ScoreUNet`**: la red de score convolucional para
imágenes que habilita la Fase 2 del estudio de ablación, segunda red del módulo `diffusion.models`
junto a la `ScoreMLP` de Fase 1. Los requisitos describen el **comportamiento observable** de la red
(su contrato de entrada/salida, las resoluciones y canales que acepta, sus garantías de determinismo
y su verificación en CPU), no su arquitectura interna, que se decide en la fase de diseño. Igual que
el MLP, la red es la **variable de control**: una vez definida queda fija en las 12 celdas de la
matriz 3×4, y toda la estocasticidad del pipeline vive fuera de ella.

**Decisión de alcance registrada (requirements, 05/07/2026):** esta spec entrega **solo la red**.
La adecuación del resto del pipeline a tensores imagen queda explícitamente fuera (ver Boundary
Context): los módulos `sde` y `samplers` hoy asumen estado `(B, data_dim)` (contrato documentado en
`.kiro/steering/numerics.md`) y el `training` construye y recarga una `ScoreMLP` concreta; su
generalización será una o más specs posteriores.

## Boundary Context

- **In scope**: la red `ScoreUNet` como nueva red de score del módulo `models` (contrato
  `(x, t) -> score` sobre tensores imagen); su suite de pytest en CPU; su smoke test ejecutable como
  módulo; la actualización de la doc del módulo (`docs/project/models.md`) y de los documentos de
  alcance (`docs/project/ejes.md`, `CLAUDE.md`, steering `product.md`) que todavía dicen "U-Net de
  librería".
- **Out of scope**: la generalización de `sde` y `samplers` a tensores imagen (sus coeficientes por
  muestra se broadcastean como `(B, 1)` y el prior del driver de sampleo se muestrea con shape
  `(N, data_dim)`: hoy solo operan sobre estado 2D); el desacople del `training` (construcción y
  recarga de checkpoints atadas a `ScoreMLP`); el dataset de Fase 2 y su pipeline de datos (incluido
  el flip horizontal); el EMA de pesos; FID / IS y toda evaluación; las corridas reales de
  entrenamiento en imágenes (GPU).
- **Adjacent expectations**: la red **reusa** las piezas compartidas del módulo `models` (embedding
  de tiempo y registry de activaciones) y **satisface** el contrato `ScoreModel` estructuralmente.
  Las specs futuras que generalicen `sde`/`samplers`/`training` deberán consumir la red vía el
  contrato `(x, t) -> score` **sin exigirle cambios**; la pérdida de DSM ya es agnóstica a la shape,
  así que el bloqueo para entrenar en imágenes vive en esos módulos, no en esta red.

## Requirements

### Requirement 1: Contrato de la red de score para imágenes
**Objective:** Como autor del TP, quiero una red de score convolucional que mapee `(x_t, t)` al score
sobre tensores imagen, para habilitar la Fase 2 del estudio con el mismo marco de SDEs de la Fase 1.

#### Acceptance Criteria
1. When se invoca el forward con `x` de shape `(B, C, H, W)` en `float32` y `t` de shape `(B,)`,
   the ScoreUNet shall devolver un tensor de shape `(B, C, H, W)` en `float32`.
2. When `t` se provee como `(B,)` o como `(B, 1)`, the ScoreUNet shall producir el mismo resultado
   en ambos casos.
3. The ScoreUNet shall producir salidas finitas (sin `NaN` ni `Inf`) para entradas finitas, con
   tiempos en cualquiera de las escalas usadas por las SDEs del repo (`[0, 1]`, `[0, T]`).
4. When se evalúa el mismo `x` con dos tiempos `t` distintos, the ScoreUNet shall producir salidas
   distintas (el condicionamiento temporal es efectivo).
5. The ScoreUNet shall producir salida no acotada, capaz de tomar valores positivos y negativos
   (ninguna activación final la restringe).
6. The ScoreUNet shall satisfacer el contrato `ScoreModel` del módulo `models` (verificable con el
   Protocol en runtime).

### Requirement 2: Resoluciones y canales soportados
**Objective:** Como autor, quiero que la red acepte los tamaños de imagen candidatos de la Fase 2,
para no atar la arquitectura a un dataset que sigue a definir.

#### Acceptance Criteria
1. The ScoreUNet shall aceptar el número de canales `C` como configuración, con al menos `C=1`
   (escala de grises) y `C=3` (RGB) verificados por tests.
2. The ScoreUNet shall procesar al menos las resoluciones de referencia `32×32` y `64×64`
   (cubriendo los datasets candidatos: FashionMNIST/CIFAR-10 y gatos 64×64).
3. If la resolución de entrada no es compatible con los niveles de reducción espacial de la
   arquitectura, the ScoreUNet shall levantar `ValueError` con un mensaje que indique la
   restricción incumplida.

### Requirement 3: Determinismo — la red como variable de control
**Objective:** Como autor, quiero una red enteramente determinística, para que toda la
estocasticidad del estudio viva fuera de la red y la ablación siga siendo atribuible a las SDEs y
los samplers.

#### Acceptance Criteria
1. When se evalúa dos veces el mismo `(x, t)` en modo evaluación, the ScoreUNet shall producir
   salidas idénticas.
2. The ScoreUNet shall no contener capas estocásticas ni de estadística de batch (dropout,
   batchnorm), verificable recorriendo sus submódulos.
3. When una misma muestra se evalúa sola o dentro de un batch junto a otras muestras (modo
   evaluación), the ScoreUNet shall producir para ella salidas numéricamente equivalentes (la
   normalización interna no depende del resto del batch).
4. When se realiza un backward sobre una salida de la red, the ScoreUNet shall exponer gradientes
   finitos en todos sus parámetros entrenables.

### Requirement 4: Configuración de la arquitectura de referencia
**Objective:** Como autor, quiero los hiperparámetros de la red expuestos con defaults que definan
la arquitectura de referencia, para fijarla una sola vez y mantenerla idéntica en las 12 celdas.

#### Acceptance Criteria
1. The ScoreUNet shall exponer sus hiperparámetros como argumentos del constructor con valores por
   defecto (sin números mágicos enterrados en el código), de modo que la instanciación sin
   argumentos defina la arquitectura de referencia del estudio.
2. When se instancia dos veces con los mismos argumentos, the ScoreUNet shall tener exactamente la
   misma cantidad de parámetros entrenables.
3. If se solicita una activación con nombre desconocido, the ScoreUNet shall levantar `ValueError`
   (las mismas activaciones soportadas que el resto del módulo `models`).

### Requirement 5: Verificación en CPU — smoke y suite
**Objective:** Como autor, quiero verificar la red en CPU sin GPU, para sostener el flujo
incremental del repo (un módulo a la vez, suite en verde).

#### Acceptance Criteria
1. When el módulo se ejecuta como `python -m diffusion.models.unet`, the smoke test shall
   instanciar la red de referencia, correr un forward con `x` de shape `(2, 3, 64, 64)` y reportar
   la shape de salida (igual a la de entrada) y el conteo de parámetros entrenables.
2. The suite de pytest de la red shall correr en verde en CPU, usando configuraciones reducidas
   para mantener el tiempo de la suite en el orden del resto del repo.
3. While torch no está disponible en el entorno, the suite shall omitirse (skip) en lugar de
   fallar, siguiendo la convención del repo.
4. When se integra la red al módulo `models`, the suite completa del repo shall permanecer en verde
   (ningún test existente se rompe).

### Requirement 6: Documentación y actualización del alcance
**Objective:** Como autor, quiero que la documentación refleje la decisión de escribir la U-Net a
mano, para que `docs/` siga siendo la fuente de verdad del alcance.

#### Acceptance Criteria
1. When la red se entrega, the documentación del módulo (`docs/project/models.md`) shall describir
   la `ScoreUNet`: su contrato, sus piezas, la arquitectura de referencia y cómo correr su smoke.
2. The documentos de alcance (`docs/project/ejes.md`, `CLAUDE.md`, steering `product.md`) shall
   actualizarse para registrar que la U-Net de Fase 2 se construye a mano (decisión del
   05/07/2026) en lugar de reusarse de librería.
3. The documentación shall registrar explícitamente que la mitigación de memorización (flip
   horizontal + EMA) queda fuera de la red, en el entrenamiento futuro de Fase 2.
