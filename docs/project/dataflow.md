# Flujo de datos cross-módulo (`dataflow`)

Este documento describe **cómo interactúan los módulos en orden**, de punta a punta: qué produce cada uno, qué recibe el siguiente, con qué **shape / dtype / rango** cruza cada frontera (*seam*), qué artefactos (YAML, `.pt`, `.npz`, `.png`) los conectan, y —sobre todo— **dónde algo puede salir mal en silencio** entre módulos. Es el mapa que falta cuando uno mira un módulo aislado: los docs por módulo (`data_generation.md`, `models.md`, `sde.md`, `training.md`, `samplers.md`) cuentan el *adentro*; este cuenta el *entre*.

La fuente de verdad es el código (paquete `diffusion` bajo `diffusion-models/src/diffusion/`), no los docs previos. Todo lo de acá se derivó leyendo el código en el estado actual del repo (incluye el subsistema de *resume* y el checkpointing intermedio ya commiteados).

## El pipeline, en orden

Cinco módulos, en secuencia. La **red** (`models`) es la variable de control; toda la estocasticidad vive en los otros cuatro (dato, forward SDE, sampler).

```
 data_generation        models            sde                training              samplers
   p_data(x0)      ─►  s_θ(x,t)     ─►  x_t=α_t·x0+σ_t·ε ─►  minimiza DSM     ─►  integra la reversa
   (fuente x0)         (la red)          (Eje 1, forward)    (aprende s_θ)         (Eje 2) ─► x0 nuevo
        │                 │                   │                   │                     │
        └── iterador ─────┴── (x_t,t)→score ──┴── target -ε/σ ────┴─ checkpoint .pt ────┘
            infinito                          + peso σ²             (state_dict+meta)
```

Importa entender que **el pipeline corre en dos invocaciones de CLI separadas**, en dos procesos distintos que **no comparten memoria**: primero `scripts/train.py` (entrena y persiste un checkpoint), después `scripts/sample.py` (carga el checkpoint y genera). El **único artefacto que cruza entre ambos es el checkpoint `.pt`** (pesos + metadata). No hay estado vivo compartido: si algo no está en el `.pt`, el sampler no lo ve (ver *Problemas conocidos* → hiperparámetros de la SDE).

## A. Entrenamiento — de un YAML a un checkpoint (`scripts/train.py`)

Secuencia real, con la función que hace cada paso y el artefacto que entrega al siguiente:

1. **`load_config(path)`** (`training/config.py`) parsea el `.yaml` a un `dict`. Necesita PyYAML (si falta → `ModuleNotFoundError`).
2. **`build_run(dict)`** (`config.py`) ensambla un `RunSpec` — el punto donde se **cablean los cuatro módulos upstream** de una: llama `make_sde`, `make_distribution` + `infinite_bare`, `make_model` y `TrainConfig`. (Detalle del seam abajo.)
3. **Overrides de CLI** mutan `spec.config`: `--num-steps`, `--device`, `--checkpoint-every`, `--quiet`.
4. **`resolve_resume(spec.checkpoint, force, resume_from)`** (`training/resume.py`) decide, **mirando solo el filesystem** (no entrena ni escribe), entre `skip` / `resume` / `fresh`. Si es `skip` (el checkpoint final ya existe y no hay `--force`), el script imprime y devuelve `0` sin entrenar.
5. Si `checkpoint_every > 0` **y** hay `out.checkpoint`, se arma el callback **`on_checkpoint(tag, snapshot)`** que persistirá **dos** artefactos por snapshot: los pesos (`save_checkpoint`) y el *sidecar* de resume (`save_resume_state`).
6. Si la acción es `resume`, **`load_resume(weights_path, expected=…)`** (`resume.py`) devuelve `(state_dict, meta, ResumeState)` y el CLI hace `spec.model.load_state_dict(state_dict)` (la red carga los pesos **antes** de entrenar; `train` no carga pesos).
7. **`train(spec.sde, spec.model, spec.data, spec.config, on_checkpoint=…, resume=…)`** (`training/trainer.py`) corre el loop **por pasos** sobre `range(start_step, num_steps)`. Cada paso:
   - `x0 = next(data_iter).to(device)` — un batch crudo del iterador infinito.
   - `t = sample_timesteps(B, sde.T, config.t_eps, generator=…)` → `(B,)` uniforme en `[t_eps, T]`.
   - `loss = dsm_loss(net, sde, x0, t, generator=…)`, que internamente encadena: `x_t, eps = sde.perturb(x0, t)` → `score_real, weight = sde.score_target(x0, t, eps)` → `score_pred = net(x_t, t)` → `mean(weight · (score_pred − score_real)²)`.
   - `zero_grad → backward → (grad_clip opcional) → optimizer.step()` (Adam).
   - `history.append(loss.item())` (una entrada **por paso**); disparo de los callbacks periódico/`best`; print de consola.
   - Devuelve `TrainResult{net, history, config, sde_name, data_dim}`.
8. **`save_checkpoint(result, spec.checkpoint, model_spec=spec.model_spec)`** escribe el blob `.pt` (contrato en el seam 4). **`save_loss_curve`** escribe un `.png` de la curva de pérdida per-step.

## B. Generación — de un checkpoint a muestras (`scripts/sample.py` → `samplers.generate_from_checkpoint`)

9. **`generate_from_checkpoint(checkpoint_path, sampler_name, *, n_samples, n_steps=500, seed=None, out=None, save_trajectory=False, map_location="cpu", model=None, **sampler_kwargs)`** (`samplers/generate.py`):
   - `load_checkpoint(path)` → `(state_dict, meta)`. **No reconstruye** nada.
   - Lee `meta["sde_name"]`, `meta["data_dim"]` (si faltan → `KeyError`).
   - **Reconstruye la red**: si hay receta `meta["model"]` → `make_model(recipe["name"], **recipe["kwargs"])` + `load_state_dict`; si no y se pasó `model=` → usa esa instancia; si no hay ninguna de las dos → `ValueError`. Después `net.eval()`.
   - **Reconstruye la SDE**: `sde = make_sde(sde_name, data_dim=data_dim)` — **solo** con nombre + `data_dim` (ver el footgun de hiperparámetros).
   - `generator = torch.Generator(); generator.manual_seed(seed)` si se dio `seed` (CPU).
   - `sampler = make_sampler(sampler_name, sde, net, n_steps=…, **sampler_kwargs)`.
   - `sampler.sample(n_samples, generator=…, return_trajectory=…)` arranca de `sde.prior_sampling((n_samples, *sde.data_shape))` e integra `T → t_eps`. Devuelve `x_0` en **espacio crudo**.
   - Si hay `out`, escribe un `.npz` con `samples` (y `trajectory` opcional). Devuelve **solo** `x_0` (la trayectoria va únicamente al `.npz`).

Cambiar el **sampler** (Eje 2) reusa el mismo score sin reentrenar; cambiar la **SDE** (Eje 1) obliga a una corrida de entrenamiento nueva.

## Los seams, en detalle

### Seam 0 — YAML → `RunSpec` (`build_run`)

`build_run` mapea el `dict` del YAML a `RunSpec{sde, model, data, config, model_spec, checkpoint, loss_curve}`:

- **`sde:`** → `make_sde(**sde_raw)`; requiere `sde.name` (si no → `ValueError`). El resto (`beta_min`, …) pasa a la SDE; la factory filtra por firma del constructor.
- **`data:`** → `shape` (o el alias legacy `name`) requerido; `dim` default 2. `n_samples` (default 4000), `batch_size` (default 256) y `shuffle` (default True) se **sacan como parámetros de la fuente** (no son campos de `TrainConfig`). El resto va a `make_distribution(shape, dim, **rest)`. Luego `data = infinite_bare(distribution.dataloader(n_samples, batch_size, shuffle))`. **`build_run` solo cablea la pista toy-2D**: las imágenes (`infinite_batches`) **no** las invoca `build_run` (ver *Estado del pipeline*).
- **`train:`** → validación **estricta**: cualquier clave desconocida para `TrainConfig` → `ValueError`. Luego `TrainConfig(**train_raw)`.
- **`model:`** (opcional) → `name` default `"mlp"`. El **gate `data_dim` int-vs-tupla**: si `sde.data_dim` es `int` (toy 2D) se inyecta `data_dim` en la receta del MLP (`setdefault`), así el MLP default queda dimensionado desde la SDE; si es una **tupla** (imágenes) **no** se inyecta, porque la U-Net toma `in_channels`/`image_size` y `ScoreMLP` haría `int(tupla)` y reventaría. Por eso una corrida de imágenes debe especificar los kwargs de la U-Net en el YAML.
- **`model_spec`** que se guarda en el checkpoint = `{"name": model_name, "kwargs": dict(model_raw)}`, capturado **después** de la inyección de `data_dim` (para el MLP la receta incluye `data_dim=2`). Es lo que vuelve al checkpoint **auto-reconstruible**.
- **`out:`** → `checkpoint` / `loss_curve` a `pathlib.Path` o `None`.

### Seam 1 — `data_generation` → `training` (el stream de datos)

`train` consume `data` con `next()`, un batch por paso, y **nunca desempaqueta tuplas** (eso ya lo hizo el adaptador). Dos fuentes, mismo contrato de "iterador infinito de tensores crudos", distinto empaquetado:

| Fuente | Qué yield-ea | Shape | Rango |
|---|---|---|---|
| Toy 2D: `infinite_bare(dist.dataloader(...))` | tensor **pelado** (desempaqueta la 1-tupla `(x0,)` del DataLoader) | `(B, dim)` usualmente `(B, 2)`, `float32` | coords crudas (o estandarizadas si `standardize=True`) |
| Imágenes: `infinite_batches(root, batch_size, …)` | tensor **pelado** (no tupla), `drop_last=True` | `(B, 3, image_size, image_size)`, `float32` | **`[-1, 1]`** (`ToTensor`→`[0,1]` y `Normalize(0.5, 0.5)`) |

El consumidor **no debe** asumir: batches exactos en la pista toy (el DataLoader de puntos usa `drop_last=False`, así que el último batch por época puede ser más corto); orden de batches reproducible sin sembrar (el shuffle tiene su propia semilla); ni rango `[0,1]` en imágenes (es `[-1,1]`, alineado con el prior de la SDE).

### Seam 2 — `sde` ↔ `training` (el target del score, Eje 1)

`dsm_loss` le pide a la SDE exactamente dos cosas por paso:

- `x_t, eps = sde.perturb(x0, t)` — el par de entrenamiento (`x_t = mean + std·ε`, con `ε ~ N(0,I)`; `ε` se sortea con el `generator` inyectado).
- `score_real, weight = sde.score_target(x0, t, eps)` — el target `∇ log p_t(x_t|x0) = −ε/σ_t` y el peso `λ(t) = σ_t²`. Sutileza: `score_target` **clampea** `std` a un piso `1e-5` antes de dividir y de elevar al cuadrado, así que en `t` muy chico el peso es `max(σ_t, 1e-5)²`, no literalmente `σ_t²`. El piso de `t` (`t_eps`) vive **upstream** (en `sample_timesteps`), no en la SDE: `marginal_prob` devuelve `std` cruda (VP/sub-VP dan exactamente 0 en `t=0`).

Internamente la SDE reescala `t` con `_expand_t(t, x)` a `(B, 1, …, 1)` según el **rango** de `x` (para 2D → `(B,1)`, byte-idéntico al código pre-generalización; para imágenes → `(B,1,1,1)`), de modo que `std`/`weight` broadcastean solos. La pérdida es **N-D-safe sin ramificar**.

### Seam 3 — `models` ↔ (`training` + `samplers`) (el contrato de la red)

La red satisface el Protocol `ScoreModel`: callable `(x, t) → score` con `score.shape == x.shape`, salida no acotada (sin activación final). Es tipado **estructural** (ninguna red hereda del Protocol). Un detalle **load-bearing y no obvio**: la red recibe `t` con **shape distinto según la etapa** — en entrenamiento `sample_timesteps` da `t` de shape `(B,)`; en sampleo el driver hace `t_cur.expand(n_samples, 1)` → `(B, 1)`. Ambas redes (MLP y U-Net) aceptan las dos formas porque `SinusoidalEmbedding` hace `t.reshape(-1)`; si se escribiera una red nueva, **debe** tolerar `(B,)` y `(B,1)`.

### Seam 4 — `training` → `samplers` (el checkpoint, la única frontera entre procesos)

El blob `.pt` que escribe `save_checkpoint` es **model-agnóstico** (nunca importa una clase de red):

```
{
  "model_state": net.state_dict(),
  "meta": {
    "sde_name": <str>,           # = sde.name
    "data_dim": <int | tuple>,   # crudo: int (2D) o tupla (C,H,W) (imágenes)
    "history":  [floats],        # pérdida per-step
    "model":    {"name","kwargs"}  # OPCIONAL: solo si se pasó model_spec
  }
}
```

`load_checkpoint(path)` devuelve `(state_dict, meta)` **crudos** (no reconstruye; `KeyError` si faltan `model_state`/`meta`). La reconstrucción canónica la hace `generate_from_checkpoint`: SDE vía `make_sde(sde_name, data_dim)`, red vía `make_model(meta["model"]["name"], **kwargs)` + `load_state_dict`. **Si no hay receta `model` y no se pasa `model=` → `ValueError`.** El camino config-driven siempre guarda la receta (checkpoints de CLI son auto-reconstruibles); un `save_checkpoint(result, path)` a mano (sin `model_spec`) obliga a pasar `model=` al generar.

### Seam de resume — sidecar + pesos (`resume.py`)

Cuando `checkpoint_every > 0`, cada snapshot periódico deja **dos** archivos hermanos:

- `X_stepNNNNN.pt` — el checkpoint de **pesos** (`save_checkpoint`): `state_dict` + `meta{sde_name, data_dim, history, model?}`.
- `X_stepNNNNN.resume.pt` — el **sidecar** de resume (`save_resume_state`): `{optimizer_state, step, torch_rng_state, generator_state}`, **sin** `history` (ya vive en el `meta` de los pesos).

El sidecar guarda exactamente lo que los pesos no pueden: el estado de **Adam**, el paso alcanzado y **ambos RNG** (el global de torch + el `generator` del ruido/muestreo de `t`). Por eso el resume es **fidedigno** (no es weights-only): al reanudar, el caller carga los pesos y `train(resume=…)` restaura optimizador + azar + paso + history. Decisión de resume (`resolve_resume`, pura):

1. `--resume-from PATH|STEP` → siempre `resume` desde ese snapshot puntual (manda sobre el skip); si no resuelve → `ValueError` que lista los snapshots.
2. si no, y el final existe y no hay `--force` → `skip` (corrida completa).
3. si no (final ausente **o** `--force`): descubre snapshots; si hay → `resume` desde el más nuevo; si no → `fresh`.

`validate_compatible` bloquea reanudar sobre un estado que no corresponde: exige igualdad **exacta** de `sde_name`, `data_dim` y receta `model` (no chequea los kwargs de la SDE — ver footgun). Ojo: `num_steps` es el **total** a alcanzar, no pasos adicionales; reanudar con `num_steps ≤ start_step` es un no-op silencioso.

## Shapes y espacio de valores, de punta a punta

| Frontera | Shape | Espacio de valores |
|---|---|---|
| `data_generation` toy | `(B, dim)` `float32` (tras `infinite_bare`) | coords crudas (o estandarizadas) |
| `data_generation` imágenes | `(B, 3, image_size, image_size)` `float32` (`infinite_batches`, bare) | **`[-1, 1]`** |
| `t` en entrenamiento | `(B,)` (`sample_timesteps`) | `[t_eps, T]` |
| `sde.perturb`/`score_target` | operan sobre `(B, *E)`; `_expand_t` → `(B,1,…,1)`; `std`/`weight` `(B,1,…,1)` | — |
| `net(x_t, t)` | `x`: `(B,*E)`; `t`: **`(B,)`** (train) y **`(B,1)`** (sample) | score no acotado |
| `t` en sampleo | `(B,1)` (grilla decreciente `T → t_eps`) | `[t_eps, T]` |
| salida del sampler | `(n_samples, *sde.data_shape)` `float32`; trayectoria `(n_steps+1, N, *shape)` | **espacio crudo** (imágenes: ~`[-1,1]`, puede excederse) |

**La des-normalización `[-1,1] → [0,1]` para ver imágenes NO está en la librería.** Existe solo como helper `denorm` en el notebook `06`. `generate_from_checkpoint` devuelve `x_0` crudo y el `.npz` guarda crudo: cualquier visualización debe des-normalizar por su cuenta.

## Estado del pipeline: Fase 1 vs Fase 2

**Fase 1 (toy 2D) — probada de punta a punta por CLI.** El camino config-driven (`build_run` → `train` → `save_checkpoint` → `generate_from_checkpoint`) **solo** cablea la pista de puntos. El único `.yaml` que se versiona, `config/vp_mixture.yaml`, es una celda 2D (VP × mezcla de 8 gaussianas, MLP). Los notebooks `01` (dato + forward), `02` (train + sample, VP + Euler–Maruyama) y `05` (reconstrucción desde checkpoint) la ejercitan.

**Fase 2 (imágenes) — la maquinaria está completa y corre de punta a punta, pero SOLO vía notebook, no por CLI.** `infinite_batches` + `make_sde("vp", data_dim=(3,H,W))` + `make_model("unet", …)` + `train` + `save_checkpoint(model_spec=…)` + `generate_from_checkpoint` funcionan juntos (notebook `06`). Pero **`build_run` no puede construir una corrida de imágenes**: ningún `data.shape` mapea a imágenes (`make_distribution` solo conoce las 5 formas toy). No hay `.yaml`/CLI para Fase 2; las imágenes son **notebook-only**. Lo único probado es el test de memorización sobre las 2 fotos de `cats-prueba` a 32×32 con una U-Net ~1M de parámetros (sobreajuste deliberado). El dataset final de imágenes sigue **a definir** (decisión del autor).

**Módulo de evaluación / visualización: NO existe.** No hay subpaquete `eval`/`viz`/`metrics` bajo `src/diffusion/` (los cinco son `data_generation`, `models`, `sde`, `training`, `samplers`), ni código de FID/IS. Toda la visualización vive ad-hoc en notebooks. Es el próximo módulo pendiente de la Fase 1, y lo decide el autor.

## Problemas conocidos / footguns cross-módulo

Estos son los puntos donde la salida de un módulo puede romper —o degradar en silencio— al siguiente. Van ordenados por gravedad.

1. **Los hiperparámetros de la SDE se pierden al samplear (el más grave).** El `meta` del checkpoint guarda **solo** `sde_name` + `data_dim`, nunca `beta_min/beta_max/sigma_min/sigma_max/T`. `generate_from_checkpoint` reconstruye con `make_sde(sde_name, data_dim=data_dim)` → **defaults del constructor**. Si una corrida entrenó VP con betas no-default, o VE con `sigma_max` custom, el sampleo usa **silenciosamente** los defaults → dinámica reversa equivocada. `config/vp_mixture.yaml` funciona solo porque su `beta_min=0.1/beta_max=20.0` coinciden con los defaults de `VPSDE`. El resume sí es seguro (rearma la SDE desde el mismo YAML), pero `validate_compatible` tampoco chequea los kwargs de la SDE.
2. **Acople escala-de-σ de VE vs escala del dato.** `VESDE` default `sigma_max=5.0` está afinado a datos 2D estandarizados, **no** al `50` de imágenes de Song et al. El prior es `N(0, σ_max²·I)` y debe matchear la escala del dato. Para imágenes en `[-1,1]` el `sigma_max` correcto es otro; combinado con el footgun #1, habría que fijarlo al entrenar **y** se descarta al samplear.
3. **No hay des-normalización en la librería.** Las muestras de imagen salen crudas (`[-1,1]`); solo el notebook `06` des-normaliza. Cualquier consumidor externo del `.npz` debe saber que tiene que hacerlo.
4. **El contrato de shape de `t` está partido**: la red ve `(B,)` al entrenar y `(B,1)` al samplear. Funciona solo porque `SinusoidalEmbedding` acepta ambos; una red nueva debe respetarlo.
5. **La generación es CPU-only.** `generate_from_checkpoint` no toma dispositivo para generar: `prior_sampling` recibe `device=None` (→ CPU) y el `torch.Generator()` es de CPU. Un checkpoint entrenado en CUDA igual samplea en CPU. El entrenamiento sí honra `config.device`; el sampleo no — asimetría.
6. **La equidad de NFE entre samplers es responsabilidad del caller.** `n_steps` es la cantidad de pasos, pero Heun = 2 NFE/paso y PC = `1+K` NFE/paso. `--n-steps` solo no iguala NFE; nada en el código lo fuerza. La matriz de `ejes.md` asume NFE igualado → hay que arreglarlo a mano.
7. **La portabilidad del checkpoint (MLP↔U-Net) depende enteramente de la receta `model`.** Un checkpoint sin receta (guardado a mano) es inusable sin pasar una instancia `model=` compatible; `load_state_dict` revienta ante un mismatch de shapes (sin guard amistoso).
8. **La semilla está fragmentada.** Entrenar: `config.seed` siembra `torch.manual_seed` + un `Generator` (ruido del kernel, muestreo de `t`); el shuffle del DataLoader tiene **su propia** semilla. Samplear: una semilla aparte siembra un `Generator` nuevo (prior + pasos estocásticos). No hay una única semilla de punta a punta: la reproducibilidad total exige setear varias.
9. **`build_run` no puede construir una corrida de imágenes** (ver arriba): la Fase 2 es notebook-only por diseño actual.

Además, tres hallazgos de los **notebooks de auditoría** (limitaciones reales del pipeline, no bugs de un módulo aislado):

- **La red es ciega a `t` chico** (`audit_04`): `SinusoidalEmbedding` usa frecuencias tipo-Transformer (`sin(t/10000^{2i/d})`), pensadas para posiciones **enteras**; con `t ∈ [1e-4, 1e-2]` los embeddings quedan casi idénticos, así que la red no distingue niveles de ruido bajos. Impacta la calidad cerca de `t=0`. **No hay EMA en el repo.**
- **La red aprende el score, no `ε`** (`audit_02`): confirmado que aproxima `−ε/σ`, con una meseta de error ~35% en `t ≲ 0.03` (consistente con la ceguera a `t` chico).
- **Los samplers estocásticos inyectan ruido en el último paso** (`audit_03`): EM y el predictor de PC agregan `g·√|dt|·Z` en **todos** los pasos, incluido el que aterriza en `t_eps` (no hay rama "ruido apagado al final", a diferencia de varias implementaciones de referencia). El `x_0` devuelto es técnicamente `x_{t_eps}`, nunca exactamente `t=0`.

## Superficie de import público (por subpaquete)

- `from diffusion import …` → solo `__version__` (la raíz no re-exporta nada).
- `from diffusion.data_generation import …` → `PointDistribution`, `REGISTRY`, `available_shapes`, `make_distribution`, `infinite_bare`, `infinite_batches`, `report_small_images`, `Gaussian`, `GaussianMixture`, `TwoMoons`, `Spiral`, `SwissRoll`. (`CatImages` **no** está en `__all__`; se accede vía `diffusion.data_generation.images.CatImages`.)
- `from diffusion.models import …` → `ScoreModel`, `SinusoidalEmbedding`, `ResidualBlock`, `ScoreMLP`, `ScoreUNet`, `REGISTRY`, `available_models`, `make_model`.
- `from diffusion.sde import …` → `ForwardSDE`, `REGISTRY`, `available_sdes`, `make_sde`, `VPSDE`, `VESDE`, `SubVPSDE`.
- `from diffusion.training import …` → `dsm_loss`, `sample_timesteps`, `TrainConfig`, `TrainResult`, `train`, `save_checkpoint`, `load_checkpoint`, `ResumeState`, `TrainSnapshot`, `save_resume_state`, `load_resume_state`, `ResumePlan`, `resolve_resume`, `discover_snapshots`, `resume_sidecar_path`, `validate_compatible`, `load_resume`, `RunSpec`, `load_config`, `build_run`.
- `from diffusion.samplers import …` → `ReverseSampler`, `ScoreFn`, `REGISTRY`, `available_samplers`, `make_sampler`, `generate_from_checkpoint`, `EulerMaruyama`, `ProbabilityFlowODE`, `HeunODE`, `PredictorCorrector`.

## Los notebooks (qué ejercita qué)

Viven en `diffusion-models/notebooks/`. Ninguno es parte de la librería; son demostraciones y auditorías ejecutadas a mano.

| Notebook | Qué cubre |
|---|---|
| `01_data_and_forward` | dato (`data_generation`) + kernel forward (`sde`); sin entrenar ni samplear (mezcla 2D). |
| `02_train_and_sample` | primer ciclo completo: VP + Euler–Maruyama (2D). |
| `03_image_data_source` | `infinite_batches` en aislamiento (fuente de Fase 2, `(B,3,64,64)`, `[-1,1]`). |
| `04_image_forward` | forward SDE sobre imágenes (análogo de `01`; sin entrenar ni samplear). |
| `05_reconstruct_from_checkpoint` | recarga model-agnóstica + integración reversa desde checkpoint (2D). |
| `06_train_unet_cats` | Fase 2 de punta a punta: test de memorización (2 gatos, 32×32, U-Net ~1M, VP; usa PC porque los samplers ODE divergen sobre el modelo sobreajustado). Contiene el helper `denorm`. |
| `audit_01_marginales` | verifica que la marginal `p_t` de entrenamiento == la implícita del sampler (ambas de `marginal_prob`). |
| `audit_02_score_vs_epsilon` | verifica que la red aprende el **score** (`−ε/σ`), no `ε`; meseta de error ~35% en `t ≲ 0.03`. |
| `audit_03_ruido_ultimo_paso` | confirma la inyección de ruido incondicional en el último paso de EM/PC. |
| `audit_04_error_por_t_gatos` | error del score por `t` sobre los 2 gatos; hallazgo raíz: ceguera a `t` chico por el embedding sinusoidal. Sin EMA en el repo. |
