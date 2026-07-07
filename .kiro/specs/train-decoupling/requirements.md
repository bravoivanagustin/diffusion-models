# Requirements Document

## Project Description (Input)
El `train()` actual del módulo `diffusion.training` construye la red por dentro (`ScoreMLP(...)` hardcodeado, con hiperparámetros tomados de `TrainConfig`) y consume un `distribution` finito vía `dataloader`, con un loop por épocas. Eso impide entrenar la `ScoreUNet` de Fase 2 (u otra red) con el mismo loop, y es el acople que la spec `score-unet` dejó anotado como pendiente.

Esta feature desacopla `train()` de la construcción de la red y del origen de datos, y pasa a un loop por pasos:

```python
def train(
    sde: ForwardSDE,
    model: ScoreModel,     # ya construida; train no sabe si es MLP o U-Net
    data,                  # iterador infinito que yield-ea tensores crudos (B, ...)
    config: TrainConfig,
    *,
    generator: torch.Generator | None = None,
) -> TrainResult:
```

Cambios: `net = model.to(device)` (idempotente) + `net.train()` en vez de construir la red; loop único `for step in range(config.num_steps)` consumiendo `x0 = next(data_iter).to(device)` (tensor crudo, sin unpack `(x0,)`); `history` por intervalo de logging (no por época). `TrainConfig` adelgaza: quita `epochs`, `n_samples`, `batch_size`, `embed_dim`, `hidden_dim`, `num_blocks`, `activation`; agrega `num_steps`; conserva `lr`, `grad_clip`, `t_eps`, `device`, `seed`, `log_every`. Se agrega un adaptador `infinite_bare(loader)` en `data_generation` que hace infinito un `DataLoader` finito y yield-ea el tensor crudo. `sample_timesteps`, `dsm_loss`, el paso de optimización, el grad-clip y la lógica de generator/seed quedan idénticos.

Dos decisiones de frontera resueltas en discovery (07/07/2026), ambas del lado **sin regresión** (la convención del repo es suite en verde en cada paso): (1) **checkpoints model-agnósticos** — `save_checkpoint` guarda `state_dict` + `sde_name` + `history` (sin hiperparámetros de red); `load_checkpoint` devuelve `(state_dict, meta)` y el caller reconstruye la red y carga el state; se actualiza `samplers/generate.py` (consumidor vivo de `load_checkpoint`) y sus tests para que la generación checkpoint-driven siga funcionando. (2) **config-driven actualizado** — `build_run` construye también el modelo (bloque `model:`) y el iterador de datos (`dataloader` envuelto en `infinite_bare`); `RunSpec` lleva `model` + `data`; `scripts/train.py` y los tests de config se actualizan; el CLI YAML sigue andando de punta a punta.

Fuera de alcance: el pipeline de datos de imágenes / `infinite_batches` / dataset final (sigue "a definir", spec futura), el EMA de pesos (paso posterior), y cualquier cambio a `ScoreMLP`/`ScoreUNet`/`layers`/`sde` más allá de imports. Verificación: correr el MLP con la firma nueva sobre swiss roll y confirmar que la pérdida baja (comparar tendencia, no valores paso a paso, porque cambia el orden de consumo de ruido al desaparecer el borde de época); `num_steps` se elige ≈ `epochs × (n_samples / batch_size)` de la corrida vieja para una comparación justa. Contexto completo, touchpoints y ripple en `brief.md`.

## Introduction

Este documento define los requisitos del refactor del módulo `diffusion.training` (**train-decoupling**):
convertir `train()` en un loop **agnóstico a la red y al origen de datos**, de forma que la misma
función entrene el `ScoreMLP` de Fase 1 y la `ScoreUNet` de Fase 2 sin ramificar por tipo. Los
requisitos describen el **comportamiento observable** del módulo (el contrato de sus funciones
públicas, qué configura, qué garantías de reproducibilidad ofrece y qué sigue funcionando sin
regresión), no su implementación interna, que se decide en la fase de diseño. Como es un refactor de
una API de biblioteca, la forma de las funciones públicas (`train`, `TrainConfig`, `save_checkpoint`,
`load_checkpoint`, `build_run`, `infinite_bare`) **es** el comportamiento observable, y por eso los
criterios las nombran explícitamente.

## Boundary Context

- **In scope**: la firma y el loop de `train()`; el adelgazamiento de `TrainConfig`; el adaptador
  `infinite_bare` en `data_generation`; el contrato model-agnóstico de `save_checkpoint`/
  `load_checkpoint` y la adaptación de su consumidor `samplers/generate.py`; la actualización del
  front-end config-driven (`build_run`/`RunSpec`/`scripts/train.py` + esquema YAML); las suites de
  test afectadas y la doc del módulo `training.md`.
- **Out of scope**: el pipeline de datos de **imágenes** (`infinite_batches`, dataset de gatos /
  CIFAR-10 / FashionMNIST — sigue a definir, spec futura); el **EMA** de pesos; cualquier cambio a
  `ScoreMLP`, `ScoreUNet`, `layers`, `sde`, `dsm_loss` o `sample_timesteps` más allá de imports/tipos;
  la evaluación / visualización de Fase 1.
- **Adjacent expectations**: `train()` **depende** de que el caller construya la red (`ScoreModel`) y
  arme la fuente de datos; **depende** de `dsm_loss`/`sample_timesteps` (invariantes) y de
  `ForwardSDE`. El módulo `samplers` **espera** poder seguir generando desde un checkpoint: esta spec
  cambia el contrato de checkpoint y **debe** adaptar `samplers/generate.py` sin romper esa capacidad.
  Esta spec **no posee** la construcción de las redes ni la definición de los datasets.

## Requirements

### Requirement 1: Firma de `train()` agnóstica a la red y a los datos
**Objective:** Como autor del TP, quiero que `train()` reciba la red ya construida y una fuente de
datos genérica, para entrenar tanto el MLP como la U-Net con el mismo loop sin ramificar por tipo.

#### Acceptance Criteria
1. The training module shall exponer `train(sde, model, data, config, *, generator=None)`, donde
   `model` es una red ya construida (que satisface el contrato `ScoreModel`) y `data` es un iterador.
2. When se invoca `train`, the training module shall usar la `model` recibida sin construir ninguna
   red por dentro ni depender del tipo concreto (MLP o U-Net).
3. When `train` recibe la `model`, the training module shall moverla al dispositivo de la corrida de
   forma idempotente (no falla si el caller ya la movió) y ponerla en modo entrenamiento.
4. The training module shall no importar ni referenciar `ScoreMLP`/`ScoreUNet` en la ruta de `train`
   (queda agnóstico a la red concreta).
5. The `TrainResult` shall describir su red como una `ScoreModel` genérica (no atada a `ScoreMLP`).

### Requirement 2: Loop de entrenamiento por pasos
**Objective:** Como autor, quiero un loop por pasos sobre un iterador infinito, para desacoplar la
duración del entrenamiento del tamaño del dataset y consumir cualquier fuente de datos por igual.

#### Acceptance Criteria
1. The training module shall correr exactamente `config.num_steps` pasos de optimización, tomando un
   batch por paso con `next()` sobre el iterador de datos.
2. The training module shall tratar cada batch como un **tensor crudo** `(B, ...)` (sin desempaquetar
   una tupla `(x0,)`).
3. While corre el loop, the training module shall registrar en `history` la pérdida promedio por
   **intervalo de logging** (no por época).
4. Where `config.log_every` es mayor que cero, the training module shall emitir una línea de progreso
   en cada intervalo de logging y en el último paso.
5. The training module shall devolver un `TrainResult` con la red entrenada, el `history` de pérdida,
   el `config` usado y el nombre de la SDE.

### Requirement 3: `TrainConfig` acotado al loop de entrenamiento
**Objective:** Como autor, quiero que la configuración de entrenamiento deje de cargar
hiperparámetros de red y de dataset, para que cada responsabilidad viva donde corresponde (la red en
su constructor, el dataset en su fuente).

#### Acceptance Criteria
1. The `TrainConfig` shall incluir `num_steps` y conservar `lr`, `grad_clip`, `t_eps`, `device`,
   `seed` y `log_every`.
2. The `TrainConfig` shall no incluir campos de arquitectura de red (`embed_dim`, `hidden_dim`,
   `num_blocks`, `activation`) ni de tamaño de dataset (`epochs`, `n_samples`, `batch_size`).

### Requirement 4: Adaptador de datos infinito
**Objective:** Como autor, quiero un adaptador que convierta un `DataLoader` finito de puntos en la
fuente infinita de tensores crudos que espera `train()`, para que el swiss roll siga usando su
`dataloader` habitual solo envuelto.

#### Acceptance Criteria
1. The `data_generation` module shall exponer un adaptador `infinite_bare(loader)` que produce un
   iterador que nunca se agota (recorre el loader repetidamente).
2. When el loader subyacente yield-ea `(x0,)`, the adaptador shall yield-ear el tensor `x0` crudo (sin
   la tupla).
3. When el iterador de `infinite_bare` se consume más veces que el largo del loader finito, the
   adaptador shall seguir entregando batches (reinicia el recorrido).

### Requirement 5: Checkpoints model-agnósticos
**Objective:** Como autor, quiero que guardar y cargar checkpoints no dependa de la clase de la red,
para que sirvan igual al MLP y a la U-Net y para que la generación con samplers siga funcionando.

#### Acceptance Criteria
1. When se guarda un checkpoint, the `save_checkpoint` shall persistir el `state_dict` de la red, el
   nombre de la SDE y el `history`, sin hiperparámetros de arquitectura de red.
2. When se carga un checkpoint, the `load_checkpoint` shall devolver el `state_dict` y la metadata
   (`meta`) sin reconstruir por sí mismo una red concreta.
3. The generación checkpoint-driven de `samplers` shall seguir produciendo muestras a partir de un
   checkpoint: el caller reconstruye la red, carga el `state_dict` y corre el sampler.
4. If un checkpoint no tiene la metadata mínima esperada, the generación checkpoint-driven shall
   fallar con un error claro (no un fallo silencioso ni una muestra inválida).

### Requirement 6: Front-end config-driven actualizado (sin romper el CLI)
**Objective:** Como autor, quiero seguir describiendo una celda del estudio en un YAML y correrla por
CLI, ahora que la red y los datos se construyen afuera de `train()`.

#### Acceptance Criteria
1. When se arma una corrida desde un config, the `build_run` shall construir la red (a partir del
   bloque `model:` del config) y la fuente de datos infinita, además de la SDE y el `TrainConfig`.
2. The `RunSpec` shall exponer la red y la fuente de datos listas para pasarle a `train()`
   (en lugar de una `distribution` finita).
3. When se corre `scripts/train.py` sobre un config YAML válido, the CLI shall entrenar de punta a
   punta y, si el config lo pide, guardar el checkpoint y la curva de pérdida.
4. If el config trae claves desconocidas o faltan las obligatorias, the `build_run` shall fallar con
   un `ValueError` que las nombre (se conserva la validación estricta actual).

### Requirement 7: Preservación de invariantes y ausencia de regresiones
**Objective:** Como autor, quiero que el refactor no cambie la matemática ni la reproducibilidad ni
rompa nada existente, para poder confiar en que solo cambió la forma de invocar el entrenamiento.

#### Acceptance Criteria
1. The refactor shall dejar `dsm_loss`, `sample_timesteps`, el paso de optimización y el grad-clip con
   el mismo comportamiento observable que hoy.
2. When se entrena con el mismo `seed`/`generator`, the training module shall producir corridas
   reproducibles (misma semilla → misma traza).
3. When se entrena el `ScoreMLP` sobre el swiss roll con la firma nueva y un `num_steps` comparable a
   la corrida por épocas previa, the training module shall mostrar una pérdida que **baja** (se compara
   la tendencia, no los valores paso a paso).
4. When se corre la suite completa del repo tras el refactor, the suite shall quedar en verde sin
   regresiones (los tests que hoy cubren `train`, checkpoints, config-driven y la generación con
   samplers se adaptan al contrato nuevo, no se rompen).
