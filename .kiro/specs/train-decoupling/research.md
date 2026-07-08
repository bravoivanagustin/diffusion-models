# Gap Analysis — train-decoupling

Fecha: 2026-07-07. Alcance analizado: `requirements.md` de `train-decoupling` (R1–R7) contra el
estado actual del módulo `diffusion.training` y sus consumidores. Mapa a nivel de línea obtenido por
scout; rutas bajo `diffusion-models/`.

## 1. Estado actual (assets y acople)

### training/trainer.py
- **`TrainConfig`** (`trainer.py:30-53`): mezcla optimización (`epochs, batch_size, n_samples, lr,
  t_eps, grad_clip, seed, device, log_every`) con arquitectura de red (`embed_dim, hidden_dim,
  num_blocks, activation`). El refactor quita 7 campos y agrega `num_steps`.
- **`TrainResult`** (`:56-63`): `net: ScoreMLP` (`:60`) — tipo concreto, hay que generalizar a
  `ScoreModel`.
- **`train(sde, distribution, config, *, generator)`** (`:66-132`): construye la red con los campos
  de `TrainConfig` (`:96-102`); arma `loader = distribution.dataloader(config.n_samples,
  config.batch_size, shuffle=True)` (`:105`); doble loop épocas × `for (x0,) in loader` (`:109-111`,
  desempaqueta la 1-tupla); `history` por época (`:127-128`). La lógica seed/generator (`:88-94`), el
  paso de optimización, el grad-clip (`:118-122`) y las llamadas a `sample_timesteps`/`dsm_loss`
  (`:113-116`) se preservan **idénticas**.
- **Checkpoints**: `save_checkpoint` (`:138-166`) escribe `model_state`, y `meta` con `sde_name`,
  `data_dim = result.net.data_dim` (`:155`), un bloque `model` con los 4 hiperparámetros de red
  tomados de `cfg` (`:156-161`) e `history`. `load_checkpoint` (`:169-194`) **reconstruye una
  `ScoreMLP`** desde `meta["model"]`+`meta["data_dim"]` (`:185-191`) y devuelve `(net, meta)`.

### training/config.py — front-end config-driven
- `RunSpec` (`config.py:39-47`): `sde, distribution, config, checkpoint, loss_curve`. → pasa a llevar
  `model` + `data` en vez de `distribution`.
- `build_run` (`:73-126`): `make_sde` (`:91`), `make_distribution` (`:100`), funnelea el bloque
  `model:` del YAML dentro de `TrainConfig` (`:104`), inyecta `n_samples` data→config (`:105-106`), y
  valida claves desconocidas contra `fields(TrainConfig)` con `ValueError` (`:107-113`). Ese funnel y
  esa validación se rompen al mover los campos de red fuera de `TrainConfig`.

### scripts/train.py + training/__main__.py — consumidores del API viejo
- `scripts/train.py`: `build_run→train(spec.sde, spec.distribution, spec.config)` (`:96`), overrides
  `spec.config.epochs`/`.device` (`:82-89`), imprime `.n_samples`/`.batch_size`/`.epochs` (`:91-95`),
  `save_checkpoint` (`:102-104`), curva de pérdida (`:43-61`).
- `training/__main__.py:21-27`: construye `TrainConfig(epochs=…, n_samples=…, batch_size=…,
  hidden_dim=…, num_blocks=…)` y llama `train(sde, dist, config)` — **también rompe** (el smoke del
  módulo `python -m diffusion.training`).

### data_generation
- `PointDistribution.dataloader(n, batch_size, *, shuffle=True)` (`base.py:79-85`) envuelve un
  `TensorDataset` de un solo tensor → yield-ea **1-tuplas `(x0,)`**. **Firma real: `n` posicional**
  (no `n_samples=`). `sample_torch` (`:74-77`). `__all__` en `data_generation/__init__.py:47-57` — es
  donde iría `infinite_bare`. **No existe ningún helper `infinite_*`** en el repo (grep 0 hits).

### samplers/generate.py — consumidor vivo del contrato de checkpoint
- `generate_from_checkpoint` (`generate.py:32`): `net, meta = load_checkpoint(path)` (`:87`); usa
  **solo** `meta["sde_name"]` y `meta["data_dim"]` (`:89-90`) + el `net` reconstruido; **no lee**
  `meta["model"]` ni `meta["history"]`; reconstruye la SDE con `make_sde(sde_name, data_dim=data_dim)`
  (`:98`) y corre el sampler (`:105-108`). `scripts/sample.py` es 100% checkpoint-driven (path +
  flags de sampler, **sin** argumentos de modelo).
- **`samplers/design.md:57` declara un revalidation trigger explícito**: "cambia el formato del
  checkpoint o las claves de `meta` (`sde_name`, `data_dim`, `model`)". Este refactor **dispara** ese
  trigger → `samplers/generate.py` + sus tests deben re-verificarse (es la decisión 1 del brief).

### Invariantes que NO cambian
`sample_timesteps`, `dsm_loss` (`losses.py`), el paso de optimización, el grad-clip y la lógica
generator/seed. `ScoreMLP`/`ScoreUNet`/`layers`/`sde` no se tocan (solo imports/tipos).

## 2. Mapa requisito → asset

| Req | Asset actual | Gap |
|---|---|---|
| R1 firma agnóstica | `train` construye `ScoreMLP` (`trainer.py:96-102`); `TrainResult.net: ScoreMLP` | **Constraint**: quitar construcción + import; generalizar tipo a `ScoreModel` |
| R2 loop por pasos | doble loop épocas×batches, `(x0,)`, history/época (`:109-130`) | **Constraint**: reescribir a `for step in range(num_steps)` + `next()` + tensor crudo + history/intervalo |
| R3 TrainConfig acotado | `TrainConfig` con 7 campos de más (`:40-53`) | **Constraint**: quitar 7, agregar `num_steps` |
| R4 `infinite_bare` | no existe; `dataloader` yield-ea `(x0,)` (`base.py:85`) | **Missing**: adaptador nuevo en `data_generation` |
| R5 checkpoints model-agnósticos | save/load atados a `ScoreMLP`+`meta["model"]` (`:152-194`); `generate.py` espera net reconstruido | **Missing + Unknown**: contrato nuevo; **cómo reconstruye `generate.py` la red** (ver §5, decisión clave) |
| R6 config-driven | `build_run` funnelea `model:`→`TrainConfig`, `RunSpec.distribution` (`config.py:104,44`) | **Constraint**: `build_run` construye modelo + data iter; `RunSpec` lleva `model`+`data`; validación se recalibra |
| R7 invariantes + sin regresión | 3 suites + `__main__` + 2 CLIs consumen el API viejo | **Constraint**: adaptar tests/consumidores sin romperlos; verificar tendencia de pérdida |

## 3. Opciones de implementación

### Opción A — Refactor in-place del módulo `training` (recomendada para el núcleo)
Editar `trainer.py` (`train`, `TrainConfig`, `TrainResult`, save/load), agregar `infinite_bare` a
`data_generation`, actualizar `config.py`/`scripts`/`__main__` y las suites. Es intrínsecamente un
refactor de un módulo existente con contrato público — no hay componente "nuevo" que aislar salvo el
adaptador.
- ✅ Mantiene la estructura del repo (un módulo por etapa); cambios localizados.
- ✅ La suite existente actúa de red de seguridad (adaptada, no reescrita).
- ❌ Toca varios archivos a la vez; el orden importa (contrato de checkpoint antes de `generate.py`).

### Opción B — Módulo/loop nuevo en paralelo, migrar y borrar el viejo
Escribir un `train` nuevo al lado, migrar consumidores, luego borrar el viejo.
- ✅ Permite A/B de la pérdida contra el viejo antes de cortar.
- ❌ Duplicación temporal, dos `train` conviviendo, más churn; innecesario para un refactor de esta
  escala. Rechazada.

### Opción C — Híbrida por fases dentro de la misma spec
Fase i: núcleo (`train`+`TrainConfig`+`infinite_bare`) con un shim que mantenga el checkpoint viejo
para no romper `samplers` todavía; Fase ii: contrato de checkpoint model-agnóstico + `generate.py` +
config-driven. Fase iii: limpieza.
- ✅ Cada fase deja la suite verde; aísla el cambio de checkpoint (el más delicado) de la firma.
- ✅ Encaja con el orden natural de tareas y con la regla "verde en cada paso".
- ❌ El shim intermedio es trabajo desechable.
- Nota: es A **ordenada por dependencia**, no un módulo nuevo. En la práctica el diseño/tareas de A
  ya deben respetar este orden; C solo lo hace explícito.

## 4. Esfuerzo y riesgo

- **Esfuerzo: M (3–7 días).** El núcleo (`train`/`TrainConfig`/`infinite_bare`) es chico y mecánico;
  el peso está en el contrato de checkpoint + `generate.py` + config-driven + adaptar 3 suites y 2
  CLIs + `__main__`, todo bajo "sin regresión".
- **Riesgo: Medio.** No hay tecnología nueva ni dependencias; el riesgo es de **integración**: el
  cambio de checkpoint cruza a `samplers` (revalidation trigger disparado) y la reconstrucción de la
  red desde checkpoint no tiene una respuesta obvia (§5). La verificación de la pérdida es por
  tendencia (el orden de consumo de ruido cambia al quitar el borde de época), no bit-exacta.

## 5. Recomendaciones para diseño + Research Needed

**Enfoque preferido: Opción A ejecutada en el orden de C** (dependencia primero): (1) `TrainConfig`
+ `train` + `infinite_bare` con verificación de pérdida en swiss roll; (2) contrato de checkpoint
model-agnóstico + `generate.py` + su suite; (3) config-driven (`build_run`/`RunSpec`/`scripts`) +
`__main__`; (4) doc `training.md`. Cada paso deja la suite verde.

**DECISIÓN CLAVE a resolver en diseño (o rebote corto al autor) — reconstrucción de la red desde
checkpoint.** R5.1 pide guardar "sin hiperparámetros de red" y R5.2 que `load_checkpoint` devuelva
`(state_dict, meta)` sin reconstruir. Pero `generate.py`/`scripts/sample.py` son 100%
checkpoint-driven y hoy dependen de que `load_checkpoint` **devuelva la red ya armada**; un
`state_dict` pelado no basta para instanciar la arquitectura correcta (MLP vs U-Net, con sus kwargs).
Alguien tiene que saber **qué** red construir. Tres resoluciones viables:
- **(R5-a) El caller pasa la red.** `generate_from_checkpoint(model, checkpoint, ...)` recibe la red
  construida (o un `model_builder`); `scripts/sample.py` la arma. Fiel a "el caller construye la
  red"; cambia la firma del CLI de sampleo y le agrega args de modelo. Puro state_dict (cumple R5.1
  literal).
- **(R5-b) Receta genérica en el checkpoint + `make_model`.** El checkpoint guarda un descriptor
  genérico `{model_name, model_kwargs}` (no campos ScoreMLP hardcodeados) y un registry `make_model`
  reconstruye cualquier red; `generate.py` sigue sin args de modelo. Mantiene el CLI intacto pero
  **matiza R5.1** (no es "sin info de red", es "receta genérica, no atada a ScoreMLP") y agrega un
  factory (que el brief prefería evitar salvo necesidad — acá el config-driven y el sample CLI la
  necesitan).
- **(R5-c) Híbrida.** `load_checkpoint` devuelve `(state_dict, meta)` (cumple R5.2); `meta` incluye
  opcionalmente la receta genérica; `generate_from_checkpoint` usa la receta si está, o acepta un
  `model` explícito. Cubre CLI y API a la vez.
- Recomendación: **(R5-c)** concilia R5.2 con el CLI existente y con el config-driven (que ya
  necesita construir el modelo desde el bloque `model:` → un `make_model` sirve para ambos). Requiere
  refinar la redacción de R5.1 ("sin hiperparámetros ScoreMLP-específicos" en vez de "sin info de
  red"). **Confirmar con el autor si acepta una receta/`make_model` genérica en el checkpoint**, o si
  prefiere R5-a (puro state_dict + red por el caller, con el sample CLI ganando args de modelo).

**Otras decisiones de diseño:**
- `num_steps` de referencia para swiss roll ≈ `epochs × (n_samples/batch_size)` de la corrida vieja
  (p. ej. la de `__main__`: 40 × 512/128 = 160) para comparar parejo. Valor de tarea, no de frontera.
- Esquema YAML nuevo: dónde viven `batch_size`/`n_samples` (ahora del data source) y los kwargs de
  red (bloque `model:` → `make_model`), y cómo `build_run` arma `model`+`data`. Recalibrar la
  validación estricta de claves (hoy contra `fields(TrainConfig)`).
- `generate.py` necesita `data_dim` para `make_sde`: es parámetro de la **SDE/dato**, no de la
  arquitectura de red → puede seguir en `meta` sin violar R5.1 en ninguna variante.
- `test_data_generation.py:83-92` afirma que `dataloader` yield-ea 1-tuplas: **no cambia** (el
  adaptador envuelve esa salida; el `dataloader` queda igual). No es un gap, es un invariante a
  respetar.

**Research Needed (llevar a diseño, no bloquea):**
- Confirmar la variante de R5 (a/b/c) con el autor antes de fijar el contrato de checkpoint (es la
  frontera con `samplers`).
- Forma exacta del bloque `model:` del YAML y del registry `make_model` si se elige R5-b/c (nombres
  de red + kwargs; qué redes registra: `mlp`, `unet`).

---

# Síntesis de diseño — train-decoupling (fase de diseño, 2026-07-07)

## Decisión del autor (07/07/2026)
- **Requirements R1–R7 aprobados.**
- **R5 = variante híbrida (R5-c).** El checkpoint se vuelve model-agnóstico **con receta genérica**:
  `load_checkpoint(path) -> (state_dict, meta)` (cumple R5.2), y `meta` incluye una **receta
  genérica** `model = {name, kwargs}` (no campos ScoreMLP-hardcodeados → refina R5.1 a "sin campos
  específicos de ScoreMLP"). Un registry `make_model(name, **kwargs)` reconstruye cualquier red
  registrada (`mlp`→ScoreMLP, `unet`→ScoreUNet). `generate_from_checkpoint` usa la receta (o acepta
  un `model=` explícito), así el sample CLI sigue funcionando sin argumentos de modelo.

## Design Decisions (síntesis)

### Decisión: `make_model` — registry de redes en `diffusion.models`
- **Contexto**: R5-c y el config-driven (bloque `model:` del YAML) necesitan construir una red por
  nombre. Dos consumidores reales (`build_run`, `generate_from_checkpoint`) → la indirección se
  justifica (no es especulativa).
- **Elegido**: `make_model(name, **kwargs)` + `available_models()`/`REGISTRY` en
  `diffusion/models/__init__.py`, espejando `make_sde`/`make_distribution` (patrón del repo, ver
  `structure.md`). Registra `"mlp"→ScoreMLP`, `"unet"→ScoreUNet`. **Additivo**: no cambia
  `ScoreMLP`/`ScoreUNet`/`layers`.
- **Refinamiento de frontera**: el brief decía "models fuera de alcance salvo imports". R5-c (elegida
  por el autor) exige este factory additivo → esta spec toca `models` **solo** para agregar
  `make_model` (sin tocar las redes). Se registra explícitamente en Boundary Commitments.

### Decisión: de dónde sale la receta del checkpoint
- `train` es agnóstico y no conoce los kwargs de la red → **no** inventa la receta. La receta la
  aporta **quien construye la red**: el config-driven (`build_run` conoce el bloque `model:`) y el
  caller directo. Mecanismo: `save_checkpoint(result, path, *, model_spec=None)` acepta la receta;
  `RunSpec` la transporta en el camino config-driven. Sin receta, el checkpoint es válido pero
  `generate_from_checkpoint` exigirá un `model=` explícito. **No** se agrega un método `get_config()`
  al Protocol `ScoreModel` (evita tocar las redes; simplificación).

### Decisión: `data_dim` sigue en `meta`
- Es parámetro de la SDE/dato (lo usa `make_sde(sde_name, data_dim=...)` en `generate.py:98`), no de
  la arquitectura de red → permanece en `meta` sin violar R5.1 en ninguna variante.

### Enmiendas de validate-design (07/07/2026)
- **Issue 1 — `history` vs `log_every`**: el snippet original solo registraba `history` dentro del
  gate de print (`if config.log_every and ...`), así que con el default `log_every=0` quedaba vacío
  (rompería el chequeo de caída de pérdida 7.3). **Resuelto**: `history` se registra a cadencia fija
  e incluye siempre el último paso; `log_every` gobierna **solo** el print.
- **Issue 2 — `data_dim` del checkpoint**: `train` model-agnóstico no tiene `net.data_dim` (ScoreUNet
  no lo tiene). **Resuelto**: `TrainResult` lleva `data_dim = sde.data_dim` (train tiene el sde);
  `save_checkpoint` lo copia a `meta["data_dim"]` (lo necesita `generate.py` para `make_sde`).
- **Issue 3 — bloque `model:` del YAML**: `config/vp_mixture.yaml` no tiene bloque `model:` (era
  azúcar opcional). **Resuelto**: el bloque `model:` sigue **opcional**; `build_run` usa por defecto
  `{name:"mlp"}` dimensionado desde el dato/SDE cuando falta → los configs existentes no se rompen.

### Generalización / Build-vs-Adopt / Simplificación
- **Generalización**: `train(model: ScoreModel, data)` generaliza sobre MLP/U-Net; `make_model`
  generaliza la construcción; `infinite_bare` generaliza el loader finito al contrato de iterador
  infinito; la receta `{name, kwargs}` generaliza sobre clases de red.
- **Adopt**: sin dependencias nuevas — `infinite_bare` es un generador Python puro; el registry
  reusa el patrón interno del repo. Nada externo que adoptar.
- **Simplificación**: NO se agrega EMA, NI pipeline de imágenes, NI `get_config()` en el Protocol.
  El único abstracto nuevo (`make_model`) tiene dos consumidores reales. `RunSpec` pasa a llevar
  `model`+`data` (no una `distribution` + campos de red).
