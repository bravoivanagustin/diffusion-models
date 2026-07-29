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

**Categoría:** Desarrollo

**Resumen:** `training-resume`: el entrenamiento se vuelve reanudable — cada snapshot periódico gana un sidecar (`…_resume.pt` con optimizer + paso + RNG), `train()` continúa corridas interrumpidas con fidelidad exacta (entera ≡ interrumpida+resumida) y el CLI decide solo entre skip/resume/fresh. Además: primer entrenamiento largo de la celda VP×mixture (19200 pasos → `models/phase_1/`), notebook `05_reconstruct_from_checkpoint.ipynb` y reorganización de dependencias (grupo `analysis`). Suite 327 → **394 passed**. Kiro spec-driven.

**Contexto:** Resuelve el follow-up de la entrada anterior: el checkpointing intermedio dejaba snapshots con solo el `state_dict` — sin estado del optimizador, ni paso, ni RNG —, así que una corrida larga interrumpida (el caso real: la `ScoreUNet` de Fase 2 en CPU/GPU) perdía todo el cómputo previo. En discovery (09/07) se eligió el **sidecar por checkpoint** en vez de embeber el estado en el `.pt`: el estado de Adam pesa ~2× los parámetros e inflaría cada snapshot ~2-3×; el sidecar mantiene el `.pt` de pesos liviano (los samplers lo ignoran) y permite reanudar desde cualquier snapshot.

**Acciones realizadas:**
- Persistencia: `save_resume_state`/`load_resume_state` en `training` — sidecar `…_resume.pt` con `{optimizer_state, step, torch_rng_state, generator_state}`; el `history` no se duplica (ya vive en el meta del `.pt` de pesos).
- Loop reanudable: `train()` acepta un payload de resume — carga el estado de Adam, restaura el RNG (global de torch + `generator`), itera `range(start_step, num_steps)` (`num_steps` es el **total** a alcanzar, no pasos extra) y continúa el `history`. El contrato `TrainSnapshot` transporta el payload en `on_checkpoint` para que el caller escriba el sidecar junto al `.pt`.
- **Gate de fidelidad:** test que exige que interrumpir-y-reanudar dé el mismo modelo y el mismo `history` que la corrida entera — el gate de correctitud de toda la spec.
- Resolver (`training/resume.py`): descubrimiento del snapshot más nuevo (parseo de `…_stepNNNNN.pt`), decisión skip/resume/fresh, y carga + validación del punto de reanudación (compatibilidad contra la config; falla claro, no migra entre configs distintas).
- Orquestación del CLI (`scripts/train.py`): si el checkpoint final ya existe → saltea (corrida completa; `--force` reentrena); si no → auto-resume desde el último snapshot con sidecar; sin snapshots → arranca de cero (con aviso). `--resume-from PATH|STEP` elige el punto.
- Primer entrenamiento largo de Fase 1: `config/vp_mixture.yaml` pasa de la corrida-humo (240 pasos) a una real (**19200 pasos**, `t_eps` 1e-4), con salidas versionadas bajo `models/phase_1/`.
- Notebook `05_reconstruct_from_checkpoint.ipynb`: como el `02` pero **sin entrenar acá** — carga el checkpoint model-agnóstico entrenado afuera (`load_checkpoint` → `make_model` → `make_sde`), integra el reverso (Euler–Maruyama) desde `N(0,I)` y reconstruye la mezcla; la curva de pérdida sale de `meta["history"]`.
- `pyproject.toml`: `matplotlib`/`pytest` salen de las dependencias duras al nuevo grupo `analysis` (junto a `jupyter`/`ipykernel`) — el runtime del paquete queda mínimo; lock de `uv` regenerado.
- Suite completa: **394 passed, 0 skipped** (los 2 skips preexistentes ya corren en el entorno nuevo); `tests/test_resume.py` nuevo. **Proceso Kiro** (discovery → requirements → design → tasks aprobadas → impl por tareas, un commit por tarea); artefactos en `.kiro/specs/training-resume/` — ojo: `.kiro/` quedó gitignored, la spec vive solo local.

**Follow-ups:**
- **EMA de pesos** quedó fuera de alcance (anotado en el roadmap como paso posterior); también quedaron afuera el resume distribuido/multi-GPU y la migración entre configs distintas.
- El resume solo tiene puntos de reanudación si la corrida usó `checkpoint_every > 0`; sin snapshots ni checkpoint final, el CLI arranca de cero.
- `docs/project/training.md` (y `to-do.md`) quedaron sin la sección de resume — estaba en el scope de la spec y no se escribió.

### 10/07/2026

**Categoría:** Desarrollo

**Resumen:** Últimos retoques para dejar el repo listo para seguir el trabajo en Linux: afinada la config de la celda VP×mixture (dataset 5000, `lr` 0.001, vuelve `grad_clip`, bloque `model:` explícito con `hidden_dim: 512`, fuera `standardize`), re-entrenada la celda y re-ejecutado el notebook `05` contra el checkpoint nuevo.

**Contexto:** La corrida larga de Fase 1 se va a continuar en Linux; antes de migrar se dejó la config afinada y los artefactos regenerados y consistentes con ella.

**Acciones realizadas:**
- `config/vp_mixture.yaml`: `n_samples` 4000 → 5000, fuera `standardize`, `lr` 0.002 → 0.001, vuelve `grad_clip: 1.0`, y el bloque `model:` queda explícito y versionado (MLP, `embed_dim` 128, `hidden_dim` **512**, 4 bloques, silu).
- Re-entrenada la celda con esa config: `models/phase_1/vp_mixture_loss.png` regenerada (el checkpoint `.pt` queda local — `*.pt` está gitignored).
- Re-ejecutado `05_reconstruct_from_checkpoint.ipynb` contra el checkpoint nuevo (outputs embebidos actualizados).

### 24/07/2026

**Categoría:** Desarrollo

**Resumen:** Escala temporal del embedding (`time-embedding-scale`): integrado `time_scale=1000` en el embedding de tiempo (spec Kiro completa, 7 commits, suite 430 en verde); los dos gates de aceptación fallaron y el diagnóstico medido redirige el fix a una spec nueva (muestreo de t + parametrización 1/σ).

**Contexto:** Una sesión de auditoría (notebooks `audit_01`–`audit_04`) midió que las redes de score no aprenden nada del score para t ≤ 0.01 (error ≈ baseline nulo incluso sobre el batch memorizado de los 2 gatos) y rastreó la causa al `SinusoidalEmbedding`: frecuencias de Transformer para posiciones enteras 0..10³ con t continuo en [0,1] — el condicionamiento no distingue t chicos. Se armó la spec Kiro `time-embedding-scale` (requirements → design → tasks, con gates de aceptación numéricos sobre los notebooks de auditoría) para el fix de referencia: escalar t antes del embedding (`t*999` de Song / pasos enteros de DDPM).

**Acciones realizadas:**
- Implementación por tareas con revisor independiente (commits `02f98eb`, `d409ab8`, `20d9784`, `9954136`, `68249ed`, `7777640`): kwarg `scale` en `SinusoidalEmbedding` (default 1.0 retrocompatible bit a bit, validación fail-fast), passthrough `time_scale` en `ScoreMLP`/`TimeMLP`/`ScoreUNet`, tests de factory/paridad entre redes/round-trip de la receta del checkpoint; suite completa 430 en verde sin modificar tests existentes.
- `time_scale: 1000` declarado en `config/vp_mixture.yaml` y en los `UNET_KWARGS` del notebook 06; la receta del checkpoint lo persiste y `generate_from_checkpoint` reconstruye con él.
- Reentrenadas las dos celdas (gatos vía notebook 06; 2D vía CLI, 19200 pasos) — checkpoints reemplazados en su ubicación estándar; los previos quedan obsoletos y no deben mezclarse en comparaciones.
- Gates medidos re-ejecutando `audit_04` y `audit_02` sin editar su lógica: **FAIL ambos**. Gatos: `eps_total = 0.953` en t=1e-3 (exigía ≤ 0.05), con mejoras del 30–60% en t ≥ 0.05 y el condicionamiento ahora sí resolviendo t chico (distancia del vector temporal 0.089 → 1.12). 2D: ratio de normas 0.708 en t=1e-3 (exigía [0.9, 1.1]), **peor** que la red vieja a t chico (0.84 → 0.71), leve mejora a t medio/alto.
- Diagnóstico con esa evidencia: el embedding ya no es el cuello de botella; a t chico mandan el ruido/señal del estimador de DSM (~30–94×) y la escasez de muestras con t uniforme, y en gatos además la magnitud ~1/σ del score (dato delta: 2 imágenes memorizadas).
- Decisión del autor (opción A): **no** escalar al fallback GFF que fijaba R4.4 (es otro embedding y la evidencia lo descarta); `time_scale` queda como cambio necesario-pero-no-suficiente; se abre una spec nueva para atacar la causa medida: muestreo no uniforme de t + parametrización 1/σ de la salida (EMA como complemento opcional).

**Follow-ups:**
- Inicializar la spec nueva (muestreo de t + parametrización 1/σ) y decidir ahí si EMA entra en alcance.
- `docs/project/models.md` actualizado con el parámetro nuevo (default, valor recomendado, referencia DDPM/Song) — misma sesión.
- Los notebooks `audit_01`–`audit_04` quedan como evidencia; todavía sin commitear (decisión pendiente del autor).

### 25/07/2026

**Categoría:** Desarrollo

**Resumen:** Señal de entrenamiento a t chico (`small-t-training-signal`): integrados el muestreo de t con importance sampling (log-uniforme + corrección por likelihood ratio, objetivo idéntico en esperanza) y la parametrización ε del score (`EpsilonScoreWrapper`, estilo `get_score_fn` de Song); gates formales FAIL pero con mejoras grandes y atribución por eje medida — la decisión del paso siguiente quedó registrada abajo.

**Contexto:** Sucesora de `time-embedding-scale` (24/07): con la resolución del condicionamiento resuelta, el error del score a t chico persistía por dos causas medidas — ruido/señal del estimador DSM con t uniforme, y magnitud ~1/σ de la salida con dato delta. Spec Kiro completa (requirements → design con corridas de atribución e hipótesis pre-escritas → tasks → impl con revisor por tarea); decisiones de alcance del autor: EMA afuera, reweighting sí. Las tareas 1.1–1.2 las implementó el autor directamente y entraron al loop con revisión adversarial (el test de equivalencia MC fue validado por sabotaje: detecta corrección faltante/invertida/factor errado).

**Acciones realizadas:**
- Eje 1 (commits `6bc942f`, `026e7c3`, `4e95b8f`): `training/time_sampling.py` (registry/factory, uniforme bit a bit idéntica al histórico por seed, log-uniforme con ~50% de masa en [1e-4, 1e-2]), `sample_weights` en `dsm_loss` y `TrainConfig.time_sampling` con uso en el loop (default probado bit a bit contra el loop previo).
- Eje 2 (commits `20e5509`, `3b5df38`, `2690ef6`): `EpsilonScoreWrapper` (σ como callable opaco, `state_dict` transparente, sin import de sde), activación por `score_parametrization: epsilon` en `build_run` con persistencia en la receta, y reconstrucción simétrica en `generate_from_checkpoint` (seam probado bit a bit; recetas viejas → red pelada).
- Configs (commit `0983993`) + celdas de carga de `audit_02`/`audit_04` actualizadas (única edición permitida; medición intacta). Suite completa: 464 en verde sin tocar tests preexistentes.
- **Corridas de atribución 2D** (ratio en t=1e-3 / banda [0.01, 0.075]): base `time_scale` = 0.708 / 0.69–0.89; (i) solo log-uniforme = 0.713 / **0.80–0.94**; (ii) solo ε = **5.02** (amplificación ruido/σ; mejor t grande: err 0.036 en t=1); (iii) ambos = 3.05 / 0.85–1.03. **Gate 2D: FAIL** (pedía [0.9, 1.1] hasta t=1e-3).
- **Gatos (ambos ejes, 2000 pasos):** eps_total 0.953 → **0.363** en t=1e-3 (2.6×) y 0.648 → **0.082** en t=0.01 (8×, umbral 0.05); cumple desde t ≥ 0.05. **Gate: FAIL por margen chico.** Denoising t=0.5 → MSE 0.028; generación finita.
- **Diagnóstico (hipótesis (c) refinada):** las causas eran las correctas — cada eje movió exactamente su palanca. El residuo se concentra en t ≤ 1e-2: en gatos es presupuesto de entrenamiento (2000 pasos, batch 2); en 2D es ruido irreducible del estimador que la parametrización ε amplifica (señal ε óptima ~0.04 vs ruido ~1 → el modo de falla pasa de encogimiento a sobre-disparo). Sin fallback prefijado ni umbrales relajados (regla R3.4).

**Follow-ups:**
- **Decisión del autor (25/07):** subir el presupuesto de la celda de gatos (num_steps 2000 → 10000) y re-medir su gate; el gate 2D en t ≤ 5e-3 se documenta como límite del estimador (impacto en muestras despreciable: σ(5e-3) ≈ 0.02 vs σ₀ = 0.3). Resultado de la re-medición: se registra al pie de esta entrada al completarse.
- Checkpoints previos (los de `time-embedding-scale`) obsoletos; no mezclar artefactos pre/post cambio.
- `docs/project/training.md` y `models.md` actualizados con ambos ejes (misma sesión).

> **Resultado de la re-medición (26/07/2026, cierre de la spec).** El autor corrió la celda de gatos con **5000 pasos** (en vez de los 10000 planeados): `eps_total` = **0.252** en t=1e-3 y **0.069** en t=0.01 (umbral 0.05 — el gate formal sigue FAIL en esos dos t), y **0.015–0.003** en t ≥ 0.05 (5–13× mejor que el punto de partida de la spec). La tendencia 2000→5000 muestra rendimientos decrecientes (error ∝ pasos^(-0.2..-0.4) en la franja chica): cruzar en t=0.01 pediría ~20–40k pasos y en t=1e-3 probablemente no se alcance por pasos solos (2 imágenes, batch 2). El impacto del residuo en las muestras es despreciable (sub-corregir 25% de σ(1e-3)≈0.0105 ≈ 0.003 del rango dinámico). **Decisión del autor: registrar esta medición como final y cerrar la spec**; el residuo a t ≤ 1e-2 queda documentado como límite de presupuesto/estimador, no como defecto de los ejes implementados (la atribución confirmó ambas causas).

### 28/07/2026

**Categoría:** Desarrollo

**Resumen:** EMA de los pesos (`ema-weights`): sombra exponencial opt-in en el loop, publicada en los checkpoints, con hermano de crudos para poder comparar; retiro del checkpoint "best"; reanudación que preserva crudos y sombra. Suite 464 → **519 en verde**. Medición informativa (sin gate): en la celda 2D el EMA mejora el error del score **1.5–7× a todo t** contra los crudos de la misma corrida, y queda **igual que la referencia del 26/07 en t=1e-3** — confirma que el residuo a t chico no es lo que el EMA arregla.

**Contexto:** Sucesora de `small-t-training-signal` (25–26/07). El problema: se sampleaba con los pesos crudos del último paso de Adam —una foto arbitraria de una trayectoria ruidosa (batch 2 en gatos, t aleatorio)— cuando las implementaciones de referencia (Song `score_sde_pytorch`, DDPM) samplean siempre con una media móvil exponencial de los pesos. Spec Kiro completa (discovery → requirements → design → validate-design → tasks → impl con implementador + revisor adversarial independiente por tarea, con mutation testing sobre los tests nuevos y worktree restaurado en cada review). Decisiones de alcance del autor (27/07): **enfoque A** (el checkpoint oficial publica EMA, los crudos viajan en el sidecar), **decay 0.999 con rampa de warmup** estilo Karras, **registro informativo sin gate numérico**, **retiro del "best"**, y **hermano de crudos** `X_raw.pt` para la comparativa crudo-vs-EMA de la misma corrida.

**Acciones realizadas:**
- **La sombra** (commit `326dad3`): `training/ema.py` con `EmaShadow` — `ema = d_s·ema + (1−d_s)·θ` y `d_s = min(d, (1+s)/(10+s))` (s = pasos completados, 1-indexado); `state_dict()` completo (parámetros EMA clonados + buffers del módulo vivo), `load_state()` para reanudar, validación fail-fast del decay. Indexa por las claves del `state_dict()` del módulo y selecciona los entrenables por identidad de tensor: por eso compone con `EpsilonScoreWrapper` sin saber nada de él (la sombra queda en claves de red pelada). Tests con réplica cerrada independiente, con y sin régimen de warmup — el review confirmó por mutación que la tolerancia estricta (`atol=1e-6`) es lo que las hace load-bearing.
- **El loop** (commit `430f05b`): `TrainConfig.ema_decay` (default `None` = sin EMA, **bit a bit** idéntico: misma secuencia de RNG, mismos pesos, misma historia), sombra construida fail-fast tras mover la red al device y antes de consumir datos, `update(step+1)` tras cada `optimizer.step()`, foto clonada en `TrainResult.ema_state`. Observador pasivo: no escribe en la red ni consume RNG. `build_run` aceptó `ema_decay` **sin tocar `config.py`** (valida el bloque `train:` por introspección de `fields(TrainConfig)`).
- **Publicación** (commit `7a2d949`): `save_checkpoint` es el punto único — publica la sombra como `model_state` y marca `meta["ema"] = {"decay": d}`, lo que cubre el final y los intermedios periódicos con un solo cambio; el formato `{model_state, meta}` no cambia, así que generación, wrapper ε y celdas de carga de los audits consumen EMA **sin modificarse**. `raw_sibling=True` escribe además `X_raw.pt` con los crudos finales, marcado `meta["raw_of"]` y **sin** clave `ema` (la ausencia de marca ya significa "pesos crudos": ponerle un dict `ema` a un artefacto crudo lo haría leer como EMA a cualquier consumidor que haga `"ema" in meta`). `discover_snapshots` no lo levanta.
- **Retiro del "best"** (commit `90189b8`): el loop emite solo la cadencia periódica. Motivo: elegir un checkpoint por la **pérdida cruda per-step** es ruidoso (t aleatorio) y correlaciona mal con la calidad de las muestras — que es justo la razón de ser del EMA; el mecanismo nunca se usó para ninguna decisión del estudio. Los `X_best.pt` que quedaron en disco se siguen tolerando (`discover_snapshots` los excluye). Excepción documentada de R6.2: el retiro obligó a tocar **dos** tests existentes (el que exigía el tag, reescrito para verificar su ausencia, y uno cuyo `assert best_ckpt.exists()` era insatisfacible, con la aserción invertida).
- **Reanudación** (commit `d51062b`): el hueco que abría el enfoque A —con EMA los intermedios publican la sombra, y `load_resume` devolvía ese `state_dict` como si fueran crudos— se cierra con `raw_model_state` y `ema_state` como claves **opcionales** del sidecar (los requeridos no cambian, así que los sidecars viejos cargan igual) y `load_resume` prefiriendo los crudos cuando están. Al reanudar la sombra se **restaura**, no se reconstruye. Guards fail-fast en ambos sentidos (EMA pedido sin sombra en el sidecar; sombra presente sin EMA pedido), antes de consumir datos y antes de restaurar el optimizador. Test de round-trip por el camino real de persistencia: interrumpida ≡ ininterrumpida en crudos, sombra y checkpoint publicado (`torch.equal`).
- **Celdas** (commit `9124d74`): `config/vp_mixture.yaml` declara `ema_decay: 0.999`; el notebook 06 lo declara en su `TrainConfig`, activa `raw_sibling=True` en el guardado final y su celda de **denoising ahora reconstruye la red desde el checkpoint recién guardado** (`load_checkpoint` + `make_model(receta)` + wrap por `score_parametrization` + `eval`, el mismo camino que la generación) en vez de usar la red viva de pesos crudos — así las dos demos del notebook consumen los pesos EMA publicados.
- **Reentreno 2D + medición informativa (celda `vp_mixture`, 19200 pasos):** la métrica es la de la celda `check3` de `audit_02` **sin editar su lógica** (se ejecuta su propio source; lo único que cambia para medir los crudos es el nombre del `.pt`).

| t | ratio EMA | ratio crudo | err rel. EMA | err rel. crudo | crudo/EMA |
|---|---|---|---|---|---|
| 1e-3 | 3.093 | 3.752 | 2.155 | 3.287 | 1.5× |
| 0.01 | 0.968 | 1.159 | 0.212 | 0.753 | 3.6× |
| 0.075 | 0.901 | 0.923 | 0.149 | 0.287 | 1.9× |
| 1.0 | 0.981 | 0.988 | 0.028 | 0.119 | 4.3× |

- **Lectura de la tabla (dos conclusiones distintas):** (a) **crudo vs EMA de la misma corrida** — el EMA gana a **todo** t, entre 1.5× y 7× en error relativo (mediana ≈ 2.6×), y la banda [0.01, 0.075] queda en 0.90–0.97 contra 0.92–1.16 de los crudos; (b) **EMA vs la referencia del 26/07** (ratio 3.05 en t=1e-3, banda 0.85–1.03) — **sin cambio**: 3.093 y 0.90–0.97. El EMA reduce la varianza del estimador de los *pesos*, no el residuo del score a t chico, que sigue siendo límite de presupuesto/estimador como quedó documentado el 26/07.

**Follow-ups:**
- **Celda de gatos: reentrenada por el autor el mismo día** (5000 pasos, pérdida 0.2426 → 0.0081, sin picos; 148/149 tensores difieren entre EMA y crudos). Resultados al pie de esta entrada.
- **Hallazgo nuevo, sin diagnosticar — picos aislados de pérdida.** La corrida 2D tuvo **8 pasos de 19200 con pérdida > 10** (máximo 8.2e6 en el paso 18045; el último pico, en el 18147) y se recuperó sola cada vez: el p95 por ventana de 960 pasos se mantuvo en ≈0.45 y la ventana final promedió 0.329, igual que antes de los picos. **No es divergencia**, pero el print de media móvil del CLI la hace ver así (mostró 157 y 4317 en las últimas dos ventanas). Con `grad_clip=1.0`, λ(t)=σ_t² y el peso de importance sampling —que **sub**-pondera t chico— una pérdida de 8e6 pide que la red haya emitido ~10³ en ese paso: no está explicado. Candidato a auditoría propia; además impide atribuir limpiamente la comparativa de esta corrida entre "el EMA promedia el jitter normal de Adam" y "el EMA atenuó la secuela de los picos".
- **No-mezcla:** `models/phase_1/vp_mixture.pt` fue reemplazado (ahora publica EMA) y quedó su hermano `vp_mixture_raw.pt`. Los checkpoints previos a esta entrada son **obsoletos**: no mezclar artefactos pre/post EMA en comparaciones ni figuras.
- `docs/project/training.md` actualizado con el EMA (contrato, convención de lectura de la marca en la meta, hermano de crudos, qué viaja en el sidecar) y con el retiro del "best"; de paso quedaron corregidas las menciones obsoletas a `X_best.pt` en el YAML de la celda y en el docstring de `config.py`, y el "hueco de cobertura" del subsistema de resume, que ya era falso (`tests/test_resume.py` existe, con 54 tests).
- **Reproducibilidad, límite conocido:** `build_run` construye la red **antes** de que `train()` aplique `config.seed`, así que dos procesos del CLI con el mismo YAML arrancan de pesos iniciales distintos (el azar inicial sale del RNG global ambiente). Candidato a backlog: sembrar antes de construir la red.

> **Resultado de la celda de gatos (28/07/2026, cierre de la spec).** El autor reentrenó `cats-prueba` con EMA (notebook 06 completo, 5000 pasos — el mismo presupuesto de la referencia del 26/07). Métrica: la de las celdas `load`/`eval` de `audit_04` **sin editar su lógica** (se ejecuta su propio source; lo único que cambia para los crudos es el nombre del `.pt`).
>
> | t | `eps_total` EMA | `eps_total` crudo | crudo/EMA | referencia 26/07 |
> |---|---|---|---|---|
> | 1e-4 | 0.4277 | 0.6739 | 1.58× | — |
> | **1e-3** | **0.1366** | 0.2517 | 1.84× | 0.252 |
> | **0.01** | **0.0274** | 0.0686 | 2.50× | 0.069 |
> | 0.05 | 0.0050 | 0.0145 | 2.88× | 0.015 |
> | 1.0 | 0.0014 | 0.0029 | 2.09× | 0.003 |
>
> **Lo importante son dos cosas.** (1) **El hermano de crudos reproduce la referencia del 26/07 casi exactamente** (0.2517 vs 0.252 en t=1e-3; 0.0686 vs 0.069 en t=0.01): es el control limpio que faltaba — la única diferencia entre las dos corridas es el EMA, y la mejora se le puede atribuir. (2) **El EMA cruzó el umbral que el presupuesto no había podido cruzar**: el gate de `small-t-training-signal` pedía `eps_total` ≤ 0.05 y el 26/07 quedó en 0.069 en t=0.01, con la proyección de que cruzar pediría ~20–40k pasos; con EMA y los **mismos 5000 pasos** el valor es **0.0274**, holgadamente por debajo. En t=1e-3 sigue arriba (0.1366), como se esperaba: ahí el límite es el estimador/presupuesto, no la varianza de los pesos.
>
> Contraste con la celda 2D, donde el EMA **no** movió el ratio en t=1e-3 (3.05 → 3.093): las dos celdas fallan a t chico por razones distintas —en gatos es presupuesto (y promediar pesos equivale a comprar pasos efectivos), en 2D es ruido irreducible del estimador que la parametrización ε amplifica— y el EMA solo ayuda en la primera. La atribución del 25/07 queda confirmada por un camino independiente.

### 29/07/2026

**Categoría:** Desarrollo

**Resumen:** Eficiencia de entrenamiento en GPU (`gpu-training-efficiency`): tres palancas **opt-in y sin regresión** para que la U-Net de Fase 2 rinda en GPU sin tocar la arquitectura —precisión mixta (AMP), carga de datos eficiente (workers + memoria fijada + transferencia no bloqueante) y autotune de kernels convolucionales—. Con los defaults la corrida es **bit a bit** la histórica en CPU. Multi-GPU / DDP **explícitamente fuera de alcance**.

**Contexto:** El loop ya movía red y batches al `device`, pero le faltaban las palancas que hacen rendir una U-Net convolucional en GPU: cuando el dataset de imágenes de Fase 2 crezca, la GPU queda ociosa esperando datos y se desperdicia throughput y VRAM. La spec suma esas palancas ordenadas por impacto (AMP la grande), todas activables desde la config y verificables **sin GPU** (la suite corre en CPU). Decisiones de alcance del autor: **DDP/DataParallel afuera** (y también `torch.compile`, kernels custom, y exponer estos knobs en el camino de **puntos**, que corre en CPU por diseño); **AMP como campo booleano** de `TrainConfig` calcando el patrón EMA (opt-in → estado opcional → clave opcional en el sidecar → guards cruzados config↔sidecar); `cudnn.benchmark` **auto-on en CUDA** sin flag; `non_blocking=True` **incondicional**; y el no-determinismo menor de `cudnn.benchmark` en GPU **aceptado** (consistente con la fidelidad "equivalente, no bit-idéntica" del resume, review 29/07). El speedup real solo se observa en GPU: en CPU se verifica **no-regresión**, no velocidad (no hay benchmark cuantitativo en CI).

**Acciones realizadas:**
- **Smoke de la API primero.** Antes de cablear nada, un test mínimo confirmó en CPU el contrato del que dependen las dos piezas de AMP en torch 2.12: `torch.autocast(device_type="cpu", enabled=False)` no-op por semilla, `torch.amp.GradScaler("cpu", enabled=False)` con `scale`/`unscale_`/`step`/`update` passthrough, y round-trip de `state_dict()`/`load_state_dict()`. La API unificada `torch.amp.*` (no la deprecada `torch.cuda.amp.*`) responde como el diseño asumía; no hizo falta el plan de fallback.
- **Precisión mixta (AMP) — `TrainConfig.amp: bool = False`.** Forward + pérdida bajo `torch.autocast`; escalador de gradiente **condicional** (se construye solo con `amp=True`, `enabled=(device.type=="cuda")`, passthrough en CPU pero existe para uniformar el camino); núcleo de optimización que **des-escala antes del recorte** (`scaler.unscale_` → `clip_grad_norm_` → `scaler.step` → `scaler.update`) para que el `grad_clip` opere sobre la norma real; EMA/pesos en float32 (el autocast no cambia el dtype de los parámetros) → el checkpoint publica lo mismo y su formato no cambia. Fail-fast: `amp` es **`bool` estricto** (un `int` 0/1 o un float revientan antes del primer batch). Con `amp=False` no se construye escalador y el núcleo queda byte-idéntico al previo.
- **Reanudación fiel con AMP.** `ResumeState.scaler_state` (campo **opcional**, calca `ema_state`): el snapshot puebla `scaler.state_dict()` cuando el escalador existe, `save_resume_state` escribe la clave **solo si está** (sin AMP el sidecar no gana nada; los sidecars previos siguen válidos), `load_resume` la lee con `get`, y el loop la restaura antes de continuar. **Guards cruzados** fail-fast en ambos sentidos (AMP pedido sin `scaler_state` guardado; `scaler_state` presente sin AMP pedido). En CPU el escalador va deshabilitado y su `state_dict()` es `{}` (vacío, no `None`): la presencia se decide por `is None`, así que la ruta de persistencia se ejercita igual sin GPU (la fidelidad del *scale factor* dinámico solo es afirmable en CUDA).
- **Carga de datos eficiente.** La rama `images` de `build_data_source` expone `num_workers` (default 0) y `pin_memory` (default False), mapeados tal cual a `infinite_batches` (que ya los aceptaba) y sumados a las **claves conocidas** antes del rechazo de unknowns (una clave de carga con typo sigue fallando enumerándola). Complementaria, la transferencia de batch es **no bloqueante incondicional** (`.to(device, non_blocking=True)`): inofensiva en CPU/sin memoria fijada, útil con `pin_memory=True` + CUDA.
- **Autotune de kernels.** `_enable_cudnn_autotune` prende `torch.backends.cudnn.benchmark` **solo** en CUDA (aprovecha los shapes fijos de la corrida); en CPU no toca nada (sin efecto observable en el camino de la suite).
- **Verificación integral + docs (esta tarea).** Smoke end-to-end en `test_config_image.py`: `build_run` de una corrida de imágenes con U-Net mínima **y las palancas activas por config** (`train.amp: true`, `data.num_workers`/`pin_memory`) → `train` → **resume**, unos pasos en CPU, con el `scaler_state` viajando en el snapshot. `docs/project/training.md` documenta las tres palancas (AMP, knobs de dataloader + `non_blocking`, `cudnn.benchmark`) con la nota CPU-vs-GPU. **Suite completa en verde: 567 tests** (sin regresión de los configs/sidecars existentes de puntos e imágenes).

**Follow-ups:**
- **El speedup no está medido** (fuera de alcance por decisión: no hay GPU en CI). Cuando haya una corrida real de Fase 2 en GPU, medir throughput/VRAM con y sin AMP a presupuesto de pasos igualado.
- **DDP / multi-GPU** quedó **explícitamente diferido**; reintroducir solo con pedido del autor (igual que CLD).
- Las celdas de imágenes que quieran las palancas deben declararlas en su `.yaml` (`train.amp: true`, `data.num_workers`/`pin_memory`); sin las claves, la corrida es la de siempre.
