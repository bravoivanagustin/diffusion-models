# Crónica — TP Final Cálculo Estocástico

Este documento contiene el historial del projecto. Aca se suben creaciones, modificaciones, descubrimientos, experimentos, etc. 

### 29/05/2026

**Categoría:** Desarrollo

**Resumen:** Primer módulo de código del TP: generación de datasets de puntos de juguete (`diffusion.data_generation`) con tests, CLI y preview, más la reorganización del proyecto en `diffusion-models/`.

**Contexto:** El repositorio era hasta ahora solo documentación. Se decidió empezar a construir el código de a poco y con tests en cada paso. El primer módulo elegido fue la generación de datasets de puntos de juguete (la pista 2D rápida de `ejes.md`), reusando una red fija (MLP o U-Net) más adelante.

**Acciones realizadas:**
- Creado el paquete `diffusion` bajo `diffusion-models/src/`, con el módulo `data_generation`: clase base `PointDistribution` (ABC), 5 formas (Gaussian, GaussianMixture, TwoMoons, Spiral, SwissRoll) y un registry/factory (`make_distribution`).
- Generación con scikit-learn (`make_moons`/`make_blobs`/`make_swiss_roll`) + numpy (espiral); salida `float32` y helpers torch (`sample_torch`/`dataloader`, import diferido). Dims híbridas: cada forma declara las que soporta.
- CLI `scripts/data_generation.py`: genera, guarda `.npz` (X + meta + color/mean/std) y un PNG de preview (PCA→2D si dim>2); error limpio y stdio UTF-8.
- Suite de pytest (22 tests, todos en verde): shape/dtype/finitud, validación de dim, reproducibilidad por seed, registry, estandarización, helpers torch y smoke del CLI.
- Instalado `torch 2.12.0+cpu` (anda en Python 3.14).
- Reorganización: todo movido a `diffusion-models/`; `pyproject.toml` llevado ahí para que imports/tests funcionen. Cache de pytest desactivado (OneDrive). Agregado `.gitignore` en la raíz.

**Follow-ups:**
- Próximo módulo: el forward SDE (`sde/` — VP/VE/sub-VP/CLD) o la red (MLP o U-Net), según `ejes.md`.
- Definir el dataset final (gatos / CIFAR-10 / toy) — sigue pendiente en `proyecto.md`.
- Al iniciar git (`git init` en `tp-final/`), aplicar el `.gitignore`.

### 01/06/2026

**Categoría:** Desarrollo

**Resumen:** Segundo módulo de código: la red de score (`diffusion.mlp`) — un MLP determinístico condicionado en el tiempo (`ScoreMLP`) para datos 2D, con su suite de tests.

**Contexto:** Con `data_generation` ya entregando los `x_0`, el paso siguiente de la Fase 1 (toy 2D) de `ejes.md` es la red que aprende el score `s_θ(x,t) ≈ ∇_x log p_t(x)`. Para datos de puntos la red es un MLP chico (la U-Net queda para la Fase 2, imágenes). Como es la **variable de control** del estudio de ablación, se construyó enteramente determinística.

**Acciones realizadas:**
- Creado el módulo `diffusion.mlp` (`score_mlp.py`) con tres clases: `SinusoidalEmbedding` (embedding de tiempo con senos/cosenos; denominadores como buffer no aprendible, `embed_dim` par, acepta `t` como `(B,)` o `(B,1)`), `ResidualBlock` (Linear→activación→Linear + skip identidad) y `ScoreMLP` (embedding de `t` concatenado con `x`, `num_blocks` bloques residuales, proyección final sin activación). El paquete (`mlp/__init__.py`) re-exporta las tres clases (`__all__`); import público sin prefijo `src.`: `from diffusion.mlp import ScoreMLP`.
- Hiperparámetros por constructor, sin números mágicos: `data_dim` (2 para VP/VE/sub-VP, 4 para CLD), `embed_dim=128`, `hidden_dim=256`, `num_blocks=4`, `activation` (silu/relu/gelu/tanh). Con los defaults, ~560k parámetros entrenables.
- Red **enteramente determinística**: sin dropout ni batchnorm (un test lo verifica recorriendo `.modules()`); la estocasticidad vive afuera, en el dato y el forward/sampler. Torch es dependencia dura del módulo (a diferencia de `data_generation`, que lo importa diferido).
- Suite de pytest (22 tests, en verde; la suite completa del repo —44 tests— sigue sin regresiones): `embed_dim` par, shapes/escalas, aceptación de `t` como `(B,)` y `(B,1)`, intercalado sin/cos y valores acotados en `[-1,1]` del embedding, `denom` como buffer; `ResidualBlock` (preserva shape + rechaza activación inválida); `ScoreMLP` (salida `(B,2)`/`(B,4)`, `num_blocks` configurable, determinismo, ausencia de capas estocásticas, params > 0, gradientes finitos).
- Smoke `__main__` en `score_mlp.py` (forward dummy + conteo de params + caso CLD `data_dim=4`) y doc del módulo en `docs/project/mlp.md`.

**Follow-ups:**
- Próximo módulo (de `ejes.md`): el forward SDE (`sde/`: VP/VE/sub-VP/CLD + el target del score) y el loop de entrenamiento (denoising score matching); después, los samplers.
- Para CLD, instanciar `ScoreMLP(data_dim=4)` (estado aumentado posición-momento).
- Sigue pendiente el dataset final de imágenes (gatos / CIFAR-10 / FashionMNIST), de `proyecto.md`.

### 04/06/2026

**Categoría:** Desarrollo

**Resumen:** Tercer módulo de código: los procesos forward (`diffusion.sde`) — VP, VE, sub-VP y CLD, con el target del score, su registry/factory y suite de tests (incl. validación Monte Carlo del kernel de CLD).

**Contexto:** Con `data_generation` (los `x_0`) y `mlp` (la red de score) ya entregados, el siguiente eslabón de la Fase 1 de `ejes.md` es el **Eje 1**: el proceso estocástico que ruidea `x_0` para fabricar el par de entrenamiento y define el target `∇_x log p_t(x_t|x_0)`. Se decidió implementar las **cuatro** variantes en una sola entrega (incl. CLD, la más compleja) y dejar el helper de pérdida DSM y el loop de entrenamiento para un módulo `training/` futuro.

**Acciones realizadas:**
- Creado el módulo `diffusion.sde` (`base.py` + `schedules.py` + `variants.py` + `cld.py` + `__init__.py` + `__main__.py`), con torch como dependencia dura (como `mlp`).
- Clase base `ForwardSDE` (ABC): atributos `name`/`data_dim`/`is_augmented`; abstractos `sde`/`marginal_prob`/`prior_sampling`; `perturb`/`score_target` concretos para la familia escalar-gaussiana (`x_t = mean + std·ε`, target `-ε/σ_t`, peso `σ_t²`).
- VP, VE y sub-VP con sus fórmulas cerradas (α_t, σ_t, g(t)) del marco de Song et al. (2021); `sigma_max=5.0` por defecto en VE (escala del toy 2D).
- CLD (Dockhorn et al., 2022): estado aumentado posición-momento (`data_dim=4`), kernel conjunto en forma cerrada vía `Φ(t)=exp(At)` (autovalor doble) + integral de covarianza exacta; score sobre el momento; prior estacionario `x~N(0,1/β)`, `v~N(0,M)`; defaults `β=4`, `M=0.25` (`Γ=2√(βM)`).
- Registry/factory (`make_sde`, `available_sdes`) calcado de `data_generation`; import público `from diffusion.sde import make_sde`.
- `data_dim` como parámetro del constructor: el módulo anda en **cualquier dimensión** (la familia escalar es agnóstica al dim; CLD usa `spatial_dim = data_dim // 2`, con `data_dim` par), así escala a la Fase 2 (imágenes) sin tocar el código.
- Suite de pytest (56 tests, en verde; suite completa —100 tests— sin regresiones): registry, shapes/dtype, `t` como `(B,)`/`(B,1)`, determinismo, límites del kernel, chequeo de cálculo `dΣ/dt` por diferencias finitas, `score_target`, varianza del prior, seam `sde × mlp`, dimensión arbitraria (escalar 1/3/7; CLD `spatial_dim` 1/3/5), y **validación Monte Carlo** del kernel de CLD contra Euler–Maruyama.
- Smoke `__main__` (`python -m diffusion.sde`) y doc en `docs/project/sde.md`.

**Follow-ups:**
- Próximo módulo (de `ejes.md`): el **loop de entrenamiento** (denoising score matching, un entrenamiento por variante del Eje 1).
- Después, los **`samplers/`** (Euler–Maruyama, PF-ODE, Heun, predictor–corrector) — el Eje 2.
- El pesado de HSM para CLD y la pérdida DSM en sí quedan para `training/`.
- Sigue pendiente el dataset final de imágenes (gatos / CIFAR-10 / FashionMNIST), de `proyecto.md`.

**Categoría:** Desarrollo

**Resumen:** Cuarto módulo de código: el loop de entrenamiento (`diffusion.training`) por denoising score matching, con corridas por config YAML, CLI y suite de tests. VP/VE/sub-VP convergen; CLD queda a la espera del pesado de HSM.

**Contexto:** Con `data_generation` (los `x_0`), `mlp` (la red de score) y `sde` (los procesos forward con su target) ya entregados, faltaba el eslabón que los une: el loop que entrena `ScoreMLP` para aproximar el score por DSM. Es el paso previo a los samplers del Eje 2. Se decidió hacerlo **config-driven** (YAML, una corrida = un archivo) porque el estudio es una matriz de ablación 4×4 y conviene versionar cada celda; el núcleo (pérdida + loop) quedó agnóstico al archivo de config y a la SDE.

**Acciones realizadas:**
- Creado el paquete `diffusion.training` (`losses.py` + `trainer.py` + `config.py` + `__init__.py` + `__main__.py`), con torch como dependencia dura (como `mlp`/`sde`).
- `dsm_loss(net, sde, x0, t)`: el corazón testeable y puro —`perturb` → `score_target` → `net` → MSE pesado `mean(λ(t)·‖s_θ−target‖²)`—, **agnóstico a la SDE**: el batch 2D se pasa crudo y `perturb`/`score_target` dan las shapes correctas (incl. CLD `data_dim=4`, donde `x0` es la posición y la SDE aumenta el estado). `sample_timesteps`: `t ~ U[t_eps, T]` (el piso `t_eps` evita `t=0`).
- `train(sde, distribution, config)`: loop con Adam, una **red nueva por llamada** (un entrenamiento por variante del Eje 1), `grad_clip` opcional, historia de pérdida por época, reproducible por seed; `TrainConfig`/`TrainResult` dataclasses. `save_checkpoint`/`load_checkpoint` guardan pesos + metadata (SDE, `data_dim`, hiperparámetros) para que los samplers reconstruyan la red.
- Corridas por config YAML: `load_config`/`build_run` arman `(sde, distribution, TrainConfig, rutas)` reusando `make_sde`/`make_distribution`; CLI `scripts/train.py --config ...` que entrena y guarda checkpoint `.pt` + curva de pérdida PNG. Ejemplos en `config/vp_mixture.yaml` y `config/cld_mixture.yaml`. Agregada la dependencia `pyyaml`.
- Suite de pytest (20 tests, en verde; suite completa —120 tests— sin regresiones): `dsm_loss` por las 4 SDEs (escalar/finito/gradientes, seam CLD), `sample_timesteps`, `train` (data_dim 2/4, pérdida que baja en VP, reproducibilidad, `grad_clip`), checkpoint ida y vuelta (incl. CLD), `build_run`/`load_config`. Doc del módulo en `docs/project/training.md`.
- **Verificación**: VP/VE/sub-VP entrenan y convergen (VP en la mezcla: 0.76 → 0.28 en 30 épocas). CLD corre mecánicamente pero **no converge**: su `score_target` devuelve `weight=1` (HSM diferido) y el target del momento explota con `t→0`.

**Follow-ups:**
- **Pesado de HSM para CLD**: decidir la fórmula del peso y dónde vive (training vs `sde`); sin él CLD no converge. Recién después, ejercitar las celdas de CLD.
- Próximo módulo: los **samplers** del reverso (Euler–Maruyama, PF-ODE, Heun, predictor–corrector) — el Eje 2 —, que reusan los checkpoints entrenados.
- Sigue pendiente el dataset final de imágenes (gatos / CIFAR-10 / FashionMNIST), de `proyecto.md`.
- (Menor) El `.claude/CLAUDE.md` quedó desactualizado (lista `sde` y `training` como no implementados); conviene refrescarlo.

### 23/06/2026

**Categoría:** Desarrollo

**Resumen:** Quinto módulo de código: los samplers del proceso reverso (`diffusion.samplers`, **Eje 2**) — Euler–Maruyama, Probability-Flow ODE, Heun y predictor–corrector —, construidos vía el flujo Kiro spec-driven. Validados sobre VP/VE/sub-VP; CLD queda con guarda explícita (fuera de alcance, atado al pesado HSM).

**Contexto:** Con `sde` (forward) y `training` (la red entrenada) ya entregados, faltaba cerrar el ciclo: integrar la ecuación reversa para generar muestras a partir de ruido. Es el **Eje 2** de `ejes.md`, y a diferencia del Eje 1 **no reentrena** (los cuatro samplers comparten el mismo score). Se construyó usando el pipeline Kiro spec-driven, y en discovery se acotó el alcance: SDEs escalares primero (CLD con guarda, pendiente del HSM), samplers "puros" con captura opcional de trayectoria (la visualización va en un módulo aparte).

**Acciones realizadas:**
- Creado el paquete `diffusion.samplers` (`base.py` + un archivo por sampler + `generate.py` + `__init__.py` + `__main__.py`) con torch como dependencia dura; patrón **Template Method**, espejo de `sde/`.
- `ReverseSampler` (ABC): el **score como función inyectable** (`ScoreFn`; `ScoreMLP` la cumple tal cual, y admite el score analítico para validar); grilla temporal `T→t_eps`; drifts reversos compartidos (`f−g²s` y `f−½g²s`); driver `sample(...)` (arranca del prior o de `init`, integra hacia atrás bajo `no_grad`/`float32` sin mutar la red, `return_trajectory` opcional); guarda que rechaza SDEs aumentadas (CLD).
- Los cuatro samplers, uno por archivo: `euler` (SDE estocástico, baseline), `pf_ode` (ODE determinístico), `heun` (ODE 2º orden, 2 evals/paso), `pc` (Euler–Maruyama + `n_corrector` correcciones de Langevin con `ε` por target de SNR; `snr=0.16`, `n_corrector=1`). Integran la SDE reversa de Anderson y la PF-ODE de Song et al.
- Registry/factory (`make_sampler`, `available_samplers`) calcado de `sde`, con filtrado de kwargs por firma. Generación config/checkpoint-driven: `generate_from_checkpoint` reusa `training.load_checkpoint`, reconstruye la SDE desde la metadata y guarda `.npz`; CLI `scripts/sample.py`.
- Suite de pytest parametrizada sobre **4 samplers × 3 SDEs escalares** (suite completa en verde, 254 pasan; el único skip es preexistente, por `pyyaml` ausente en el entorno): contrato/factory, determinismo/reproducibilidad, **correctitud con score analítico** (cada sampler recupera una gaussiana conocida `N(μ,Σ₀)` dentro de tolerancia Monte Carlo) y generación desde checkpoint. Doc del módulo en `docs/project/samplers.md`.
- **Proceso (Kiro spec-driven):** discovery → requirements → design (con `/kiro-validate-design`) → tasks → implementación autónoma (un subagente por tarea + review adversarial independiente + validación final GO). Artefactos en `.kiro/specs/samplers/` (brief, requirements, design, research, tasks); rama `feat/samplers`, 14 sub-tareas, un commit por tarea.
- Refrescado el `.claude/CLAUDE.md` (resuelve el follow-up menor de la entrada anterior): marca `sde`/`training` como hechos y suma `samplers` como el módulo en curso.

**Follow-ups:**
- **Pesado de HSM para CLD** sigue pendiente; recién después, la dinámica reversa de CLD (hoy la guarda la rechaza). Nota validada: con score exacto, VE + samplers determinísticos dejan un offset de media residual — es correcto (prior `N(0,σ_max²)` vs marginal `N(μ,σ₀²+σ_max²)`), no un bug.
- Módulo de **evaluación / visualización** de Fase 1 (campos de score, trayectorias, densidad, comparación con el score analítico de la mezcla; FID/IS en Fase 2). Los samplers ya exponen `return_trajectory` para alimentarlo.
- La **matriz 4×4 escalar** ya es ejecutable (VP/VE/sub-VP × los 4 samplers, reusando checkpoints).
- Sigue pendiente el dataset final de imágenes (gatos / CIFAR-10 / FashionMNIST), de `proyecto.md`.

### 05/07/2026

**Categoría:** Desarrollo

**Resumen:** CLD se eliminó del alcance del TP: el costo (pesado de HSM + dinámica reversa aumentada) no justificaba la cuarta SDE. El Eje 1 queda con VP/VE/sub-VP y la matriz pasa a 3×4 = 12 celdas; suite completa en verde (241 tests).

**Contexto:** CLD arrastraba dos pendientes acoplados desde las entradas del 04/06 y 23/06: sin el pesado de HSM el entrenamiento no convergía (el target del momento explota con `t→0`), y los samplers lo rechazaban con una guarda a la espera de la dinámica reversa aumentada. Era la única variante que rompía el contrato escalar-gaussiano (kernel conjunto, Cholesky, estado aumentado) y ramificaba la base (`is_augmented`) y sus consumidores. Se decidió recortar el alcance: el estudio de ablación se sostiene igual con tres SDEs, y CLD queda como opción de literatura (Dockhorn et al., 2022, en `referencias.md`), no de implementación.

**Acciones realizadas:**
- Borrados `sde/cld.py` y `config/cld_mixture.yaml`; `CLDSDE` fuera del registry y del `__all__` de `diffusion.sde`.
- Eliminados el hook `is_augmented` de `ForwardSDE` y la guarda contra SDEs aumentadas en `ReverseSampler`/`PredictorCorrector`: sin CLD eran código muerto. El contrato queda escalar-gaussiano puro.
- Limpieza de docstrings, comentarios y mensajes de error en `sde` (base, variants, `__main__`), `training` (losses, trainer, config), `mlp` (incl. el caso `data_dim=4` del smoke) y `scripts/train.py`.
- Tests: `test_sde.py` 56→47 (fuera el bloque CLD completo, incl. la validación Monte Carlo), `test_training.py` 20→17 (el round-trip de checkpoints se convirtió a VP para conservar esa cobertura), `test_samplers.py` sin el test de la guarda. Suite completa: **241 passed, 1 skipped** (el skip preexistente de `pyyaml`).
- Docs sincronizados: `sde.md`, `training.md` (fuera la sección «Estado de CLD»), `samplers.md`, `ejes.md` (matriz 3×4 = 12 celdas), `mlp.md`, `data_generation.md` y `to-do.md` (los dos pendientes de CLD → ⚪ Descartado).
- `.claude/CLAUDE.md` y `.kiro/steering/` (product, experiment-matrix, numerics, testing, structure) sincronizados con el alcance nuevo, con nota fechada de la eliminación para no reintroducir CLD por accidente. De paso, `CLAUDE.md` ahora lista `samplers` como quinto módulo entregado (estaba desactualizado).

**Follow-ups:**
- Re-ejecutar los notebooks 01 y 02: los outputs guardados siguen mostrando la lista de SDEs con `cld` y un traceback de la era HSM (las celdas de código están limpias).
- Módulo de **evaluación / visualización** de Fase 1 (campos de score, trayectorias, densidad, comparación con el score analítico de la mezcla; FID/IS en Fase 2). Los samplers ya exponen `return_trajectory`.

### 06/07/2026

**Categoría:** Desarrollo

**Resumen:** Reestructura de la red de score: `diffusion.mlp` pasa al subpaquete `diffusion.models` (`layers.py` compartido + `mlp.py` + `base.py` con el Protocol `ScoreModel`), como base limpia para la U-Net de Fase 2. Refactor puro, sin cambio de comportamiento.

**Contexto:** Antes de escribir la U-Net convolucional de Fase 2 hacía falta separar las piezas compartidas entre redes de las específicas de cada una (el módulo `mlp` mezclaba el embedding de tiempo reusable con el MLP concreto). Se decidió partir el trabajo en dos pasos que no se pisan: primero el movimiento mecánico de código (protegido por la suite existente, sin gate de spec), y recién después la U-Net. La decisión de construir la U-Net **a mano** (no de librería) se fijó acá.

**Acciones realizadas:**
- Creado el subpaquete `diffusion.models`: `layers.py` (piezas compartidas: `SinusoidalEmbedding` + activaciones), `mlp.py` (`ScoreMLP`, sin cambios), `base.py` (Protocol `ScoreModel`, contrato `(x, t) → score`) y `__init__.py` con re-exports. Import público `from diffusion.models import ScoreMLP`.
- Eliminado el paquete `diffusion.mlp`; actualizados los imports internos en `training/`, `samplers/` y los tests.
- Docs: `docs/project/mlp.md` → `docs/project/models.md`; `.claude/CLAUDE.md` y steering sincronizados.
- Refactor puro: mismos parámetros y misma salida; la suite existente (241) protege el movimiento y queda en verde sin regresiones.

**Follow-ups:**
- Construir la `ScoreUNet` a mano sobre esta base (spec `score-unet`).

**Categoría:** Desarrollo

**Resumen:** Nueva red de Fase 2: `ScoreUNet`, una U-Net convolucional para imágenes **construida a mano** en `models/unet.py`, vía el flujo Kiro spec-driven. ~17.2 M params, determinística; suite 241 → 263.

**Contexto:** La Fase 2 necesita una red de score `(B,C,H,W) → (B,C,H,W)`. Se descartó reusar una U-Net de librería (diffusers / denoising-diffusion-pytorch): igual que el MLP, la red es la variable de control del estudio de ablación, así que conviene tenerla propia y fija. Se construyó sobre el subpaquete `models/` reestructurado el mismo día.

**Acciones realizadas:**
- Nueva `ScoreUNet` en `models/unet.py`, ensamblada a mano: `TimeMLP` (proyección temporal desde el embedding sinusoidal), `ConvResBlock` (bloque residual convolucional con inyección de tiempo), self-attention espacial (a 16×16), down/upsampling, y encoder + bottleneck + decoder con skips.
- Determinística: GroupNorm, sin dropout (misma regla que el MLP). Fail-fast `ValueError` (`image_size`/`groups` en `__init__`; H/W y canales en `forward`). Con los defaults (`base_channels=64`, `channel_mults=(1,2,2,4)`, `attn_resolutions=(16,)`), ~17.2 M params.
- Suite de `ScoreUNet` (contrato de shape, determinismo/gradientes, config/errores/arquitectura de referencia): 241 → **263 passed** sin regresiones. Fix de una flakiness (semilla + `atol`) en `test_scoreunet_batch_independence`.
- Doc en `docs/project/models.md`; `ejes.md`/`CLAUDE.md` actualizados a "U-Net a mano". **Proceso Kiro:** discovery → requirements → gap → design (con validación) → tasks → impl autónoma (subagente por tarea + review adversarial + validación GO). Artefactos en `.kiro/specs/score-unet/`.

**Follow-ups:**
- Desacoplar `train()` de la construcción de la red para poder alimentar `ScoreMLP` **o** `ScoreUNet` (spec `train-decoupling`).

### 07/07/2026

**Categoría:** Desarrollo

**Resumen:** `train-decoupling`: `train(sde, model, data, config)` se vuelve agnóstico a la red y al dato —recibe la red ya construida y un iterador infinito de tensores crudos, con loop por pasos— y los checkpoints pasan a ser model-agnósticos. Suite 263 → 275. Kiro spec-driven.

**Contexto:** El `train()` clavaba `ScoreMLP` por dentro (instanciación hardcodeada) y consumía una `PointDistribution`. Con la `ScoreUNet` ya entregada, eso bloqueaba la Fase 2. Se decidió invertir la dependencia: el caller construye la red y arma la fuente de datos; `train` solo corre el loop de DSM. En discovery se acordó que en esta spec también se vuelven model-agnósticos los checkpoints y se actualiza el front-end config-driven.

**Acciones realizadas:**
- Registry `make_model(name, **kwargs)` en `diffusion.models` (`{mlp, unet}`), espejo de `make_sde`/`make_distribution`.
- Adaptador `infinite_bare(loader)` en `data_generation`: generador infinito que desempaqueta la 1-tupla `(x0,)` y yield-ea el tensor crudo — el contrato de `data`.
- Flip del API: `train(sde, model, data, config)` recibe la red ya construida (`ScoreModel`) + un iterador infinito; loop **por pasos** (`num_steps`, `next(data)` por step). `TrainConfig` adelgazado (solo el loop: sin hiperparámetros de red ni de dataset).
- Checkpoints model-agnósticos (R5-c): `save` guarda `state_dict` + meta (`sde_name`, `data_dim`, `history`, receta `model:{name,kwargs}`); `load` devuelve `(state_dict, meta)` y el caller reconstruye vía `make_model`. Actualizado `generate_from_checkpoint`.
- Front-end config-driven actualizado (`build_run`/`RunSpec`/`scripts/train.py`/YAML): `n_samples`/`batch_size` al bloque `data:`, `num_steps` en `train:`, bloque `model:` opcional. Doc en `docs/project/training.md` + notebook al API nuevo. Suite: 263 → **275 passed**.

**Follow-ups:**
- Falta la **fuente de datos de imágenes** (el `infinite_batches` "dataset a definir") para alimentar la Fase 2.
- `sde`/`samplers` todavía asumen `(B, data_dim)`: generalizarlos a `(B,C,H,W)` es un bloqueo aparte.

### 08/07/2026

**Categoría:** Desarrollo

**Resumen:** `image-data-source`: la fuente de datos de imágenes de Fase 2 — `infinite_batches(root, batch_size, …)` en `data_generation`, que entrega `(B,3,64,64)` en `[-1,1]` con el mismo contrato que el toy 2D. Agrega `torchvision==0.27.0`. Suite 275 → 291. Kiro spec-driven.

**Contexto:** Con `ScoreUNet` y el `train` model-agnóstico listos, faltaba con qué alimentarlos en imágenes: el `infinite_batches` que el roadmap dejó anotado como "dataset a definir". Debía cumplir el mismo contrato que la fuente toy 2D (iterador infinito de tensores crudos), pero de imágenes. En discovery se fijó: transforms con torchvision (no a mano), higiene report-only "too-small" (el dedup sigue en `scripts/limpiar_imagenes.py`) y framing explícito (center-crop vs deform).

**Acciones realizadas:**
- Nuevo `data_generation/images.py` con imports pesados (torch/torchvision/PIL) diferidos: `CatImages` (Dataset sin labels; descubrimiento por `rglob` ordenado; `.convert("RGB")` obligatorio; devuelve tensor pelado), `_build_transform` (flip horizontal opcional → encuadre `Resize`+`CenterCrop` o `Resize` deformante → `ToTensor` → `Normalize` a `[-1,1]`), `infinite_batches` (DataLoader `drop_last` + wrapper infinito; **fail-fast** `ValueError` si la carpeta no existe/está vacía/tiene menos imágenes que `batch_size`) y `report_small_images` (higiene report-only, no borra).
- Export `infinite_batches`/`report_small_images`; el import diferido mantiene `import diffusion.data_generation` liviano (no arrastra torchvision).
- Agregada la dependencia `torchvision==0.27.0` (+ `pillow`) — wheel cp314-win CPU, fija `torch==2.12.0`; steering `tech.md` actualizado.
- 16 tests autocontenidos (imágenes sintéticas en `tmp_path`, sin depender de `data/cats-prueba`): 275 → **291 passed**. Doc en `docs/project/data_generation.md`. **Proceso Kiro** completo; artefactos en `.kiro/specs/image-data-source/`.

**Follow-ups:**
- El camino de entrenamiento de imágenes de punta a punta sigue bloqueado: `sde`/`samplers` asumen `(B, data_dim)` — generalizarlos a event shapes N-D es la próxima spec.

### 09/07/2026

**Categoría:** Desarrollo

**Resumen:** Generalización N-D de `sde`/`samplers` a event shapes arbitrarios (spec `nd-shapes`, mergeada a master), que habilita imágenes `(B,C,H,W)` sin romper el toy 2D; más el notebook `04_image_forward.ipynb` que visualiza el proceso forward sobre las fotos de gatos de `cats-prueba`.

**Contexto:** Con la `ScoreUNet` (score-unet), el `train` model+data-agnóstico (train-decoupling) y la fuente `infinite_batches` (image-data-source) ya entregados, el único eslabón que faltaba para el camino de imágenes de Fase 2 era que `sde` y `samplers` operaran sobre `(B,C,H,W)`: asumían datos planos `(B, data_dim)` y los coeficientes de la SDE salían `(B,1)`, que no broadcastean contra imágenes. El roadmap ya lo marcaba como "bloqueo separado". Se hizo vía el flujo Kiro spec-driven; el cambio resultó contenido (por broadcasting, sin hardcodear la forma).

**Acciones realizadas:**
- **Spec `nd-shapes`** (Kiro: discovery → requirements → validate-gap → design → validate-design → tasks → impl autónomo), mergeada a master (rama `feat/nd-shapes`, 7 commits: 1 de spec + 6 de tareas, con review adversarial independiente por tarea y validación final GO).
- `sde/base.py`: `_expand_t(t, ref)` **rank-aware** (reshape de `t` a `(B,1,…,1)` según el rango de `x`; para rango 2 devuelve `(B,1)`, byte-idéntico a antes); `data_dim` acepta `int | tuple` y expone `data_shape` normalizada, conservando el valor crudo. `variants.py`: threading del tensor de referencia en las 6 llamadas a `_expand_t` de VP/VE/sub-VP (sin cambios de fórmula).
- `samplers/base.py`: `sample()` arma el prior desde `data_shape` (`(n, *E)`); el resto del driver, los `step()` y el `_expand_t` del sampler no cambian (la SDE re-expande `t` contra el estado).
- Plomería: la forma de evento viaja en la metadata de checkpoint (`training/trainer.py`), `generate_from_checkpoint` reconstruye la SDE con ella, y `config.py` gatea la inyección de `data_dim` al modelo solo cuando es entero (path MLP 2D; una tupla no se mete como hiperparámetro de la U-Net). `dsm_loss` quedó N-D-safe **sin cambios** (el peso `std²` es `(B,1,1,1)` y broadcastea solo).
- Tests parametrizados 2D + imagen-chica `(3,8,8)`: familia escalar N-D, invariancia 2D byte-idéntica, prior/muestras N-D en los 4 samplers, round-trip end-to-end de generación de imágenes vía checkpoint (con una `ScoreUNet` real chica) y DSM N-D. Suite completa en verde: **322 passed, 2 skipped**.
- **Notebook `04_image_forward.ipynb`** (análogo de `01` para imágenes): carga los 2 gatos de `cats-prueba` (`infinite_batches`, sin augmentation, determinista), aplica el forward y visualiza —reusando los helpers `denorm`/`show_grid` de `03`— una grilla 3 SDEs × (tiempos + prior) para un gato y una "tira" VP por imagen. Verificado corriendo las celdas headless en el venv (`uv`/torchvision): carga `(2,3,64,64)` en `[-1,1]`, forward y figuras sin error (VP/sub-VP disuelven rápido a `N(0,I)`; VE conserva señal y explota tarde).

**Follow-ups:**
- Refrescar el markdown de cierre de `03_image_data_source.ipynb`, que todavía dice que `sde`/`samplers` no operan sobre `(B,C,H,W)` — obsoleto con `nd-shapes`.
- El notebook `04` requiere el `.venv` con `torchvision` (el `python` del PATH no lo tiene).
- (Hueco de la crónica) Sumar entradas para score-unet, train-decoupling e image-data-source, no cronicados entre el 05/07 y hoy.
- Próximo (Fase 2): entrenar la `ScoreUNet` sobre gatos y correr el reverso para generar — el análogo de imágenes de `02`.

**Categoría:** Desarrollo

**Resumen:** Notebook `03_image_data_source.ipynb`: demostración aislada de `infinite_batches` sobre `data/cats-prueba/`, con outputs ejecutados (contrato, higiene, des-normalización, augment, framing, reproducibilidad, fail-fast).

**Contexto:** Para *ver* funcionar la fuente de imágenes sin montar una corrida completa, se armó un notebook de demostración —al lado de `01`/`02`, y previo al `04`— que ejercita el módulo con la API pública sobre los 2 gatos de prueba.

**Acciones realizadas:**
- Nuevo `diffusion-models/notebooks/03_image_data_source.ipynb` (mismo bootstrap y kernel que `01`/`02`), ejecutado con outputs embebidos: contrato `(2,3,64,64)` en `[-1,1]` e infinitud; higiene report-only; des-normalización `[-1,1]→[0,1]`; augment (flip horizontal) on/off; framing `crop=True` vs `crop=False` (ilustrado sobre una imagen no cuadrada sintética, porque los gatos ya son 64×64 cuadrados); reproducibilidad por `seed`; fail-fast; y una grilla de 16 de `cats-v1`.
- Hallazgo registrado: todo el dataset de gatos ya viene pre-escalado a 64×64 RGB, así que sobre ese dato `crop` y la higiene "<64" son no-ops (importan para carpetas de fotos de tamaño arbitrario). El notebook aporta los helpers `denorm`/`show_grid` que después reusa el `04`.

**Follow-ups:**
- El markdown de cierre del notebook todavía dice que `sde`/`samplers` no operan sobre `(B,C,H,W)`: quedó obsoleto con `nd-shapes` (mismo follow-up que la entrada de nd-shapes).

**Categoría:** Desarrollo

**Resumen:** Checkpointing intermedio opt-in en `training`: `TrainConfig.checkpoint_every` (default `0`) + un callback `on_checkpoint` en `train`, que habilita snapshots periódicos (`…_stepNNNNN.pt`) y un `…_best.pt` junto al checkpoint final. Sin regresión (default apagado). Suite 322 → 327 passed. Cambio simple, sin proceso Kiro (acordado con el autor).

**Contexto:** Hasta ahora el modelo se guardaba **una sola vez, al final** del entrenamiento (`save_checkpoint` lo llama el caller tras `train`; el loop no persistía nada). Se pidió poder guardar también estados intermedios. Es una modificación acotada a un módulo, así que se saltó el flujo Kiro (spec/requirements/design) por overkill y se construyó de a poco con la suite en verde, respetando el diseño del módulo: `train` es **agnóstico y sin I/O**.

**Acciones realizadas:**
- `TrainConfig.checkpoint_every: int = 0` — switch único: `0` = solo el checkpoint final (comportamiento histórico, sin regresión); `N>0` = además snapshots periódicos + best.
- `train(..., on_checkpoint=None)`: el loop decide **cuándo** (cadencia periódica propia, chequeada cada paso para no atarse a la de `history`, con el último paso excluido porque lo cubre el final; y best-so-far sobre la pérdida **media de intervalo**) e invoca el callback con `(tag, snapshot)` —un `TrainResult` foto del estado—; el callback decide **cómo/dónde** persistir. Así `train` sigue sin tocar el filesystem.
- `scripts/train.py`: arma el callback cuando hay `out.checkpoint`, derivando rutas hermanas con `Path.with_stem` (`vp_mixture.pt` → `vp_mixture_step00050.pt` / `vp_mixture_best.pt`) y reusando `save_checkpoint` con el mismo `model_spec` (cada snapshot es tan reconstruible como el final). Nuevo override `--checkpoint-every`; aviso si `checkpoint_every>0` sin `out.checkpoint`.
- Tests (5 nuevos): gate apagado no llama al callback (sin regresión), emisión de periódicos correctos (múltiplos de `N`, sin el último) + al menos un `best`, persistencia/carga de los `…_stepNNNNN.pt`/`…_best.pt` estilo-CLI, y `build_run` pasa `train.checkpoint_every` al `TrainConfig`. Suite completa: **327 passed, 2 skipped**. Smoke CLI-equivalente verificado (pasos 5/10/15 + best + final, cargables). Docs: `docs/project/training.md`, docstrings de `config.py`/`train.py` y el YAML de ejemplo.

**Follow-ups:**
- No se implementó *resume* (retomar el entrenamiento desde un snapshot): el checkpoint guarda solo el `state_dict` de la red, no el estado del optimizador ni el paso. Si se quiere, es una extensión aparte.
- El best usa la pérdida DSM (ruidosa), así que es orientativo; para comparar estados suele servir más un snapshot periódico.
