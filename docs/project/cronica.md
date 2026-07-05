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

### 05/07/2026

**Categoría:** Desarrollo

**Resumen:** CLD ya converge. Se pasó el objetivo de entrenamiento a **Hybrid Score Matching (HSM)** —la red aprende solo el score del momento— y con eso la pérdida dejó de explotar (de miles a unidades). Se cableó el loop al nuevo contrato y se actualizaron tests y docs.

**Contexto:** El loop de entrenamiento (entrada del 04/06) dejaba CLD sin converger: `sde.score_target` devolvía el score **conjunto** `(B, 4)` con `weight=1`, y el término de posición `-n_x/L₁₁` explotaba cuando `t→0`. El target de HSM figuraba como pendiente. Se cambió una línea en `sde/cld.py` para que `score_target` devuelva **solo** la componente de momento `∇_v log p_t(v|x) = -n_v/L₂₂`, shape `(B, spatial_dim)` —el target de HSM—. Eso rompía el contrato con los consumidores (9 tests en rojo por el cambio de shape), así que se completó el cableado.

**Acciones realizadas:**
- **Diagnóstico:** el cambio de contrato `(B, 4) → (B, 2)` dejaba 9 tests en rojo — `dsm_loss` fallaba con mismatch de shape (la red predice 4, el target pasó a 2) y 6 tests de `sde` asumían el score conjunto. Los samplers **no** se ven afectados (no usan `score_target`; además ya rechazan CLD con guarda).
- **Loop (`training/losses.py`):** `dsm_loss`, para SDEs aumentadas (`sde.is_augmented`), ahora compara solo la mitad de momento de la red (`net(x_t, t)[:, spatial_dim:]`) contra el target `(B, spatial_dim)`. Es la única ramificación del loop.
- **Por qué converge:** aprender solo el momento —lo que HSM pide— descarta el término de posición, cuya entrada de Cholesky `L₁₁ → 0` con `t→0` era la fuente de la explosión. La del momento `L₂₂` se mantiene `O(1)` (el momento conserva su varianza de equilibrio `M`), así que el target queda **acotado**. Verificado: la pérdida de CLD pasó de ~8k–33k (oscilando) a ~7 → 4.7 (acotada y decreciente).
- **Tests:** actualizados los 6 tests de CLD de `test_sde.py` al nuevo contrato `(B, spatial_dim)`; agregado un test en `test_training.py` que fija el comportamiento HSM (la pérdida usa solo el momento). Suite completa en verde (256 con `pyyaml` presente).
- **Docs:** `training.md` (nueva sección «CLD: objetivo de HSM» + el seam), `sde.md` (contrato de `score_target`), `to-do.md` (HSM de CLD → hecho). Corregido el cuerpo del docstring de `score_target` en `cld.py`.

**Follow-ups:**
- (Opcional) Pesado `λ(t)` tipo-verosimilitud para CLD (hoy `weight=1`): afinaría la convergencia, no es bloqueante.
- **Muestreo de CLD**: los samplers siguen con la guarda que rechaza SDEs aumentadas; falta la dinámica reversa aumentada del Eje 2 para cerrar la fila CLD de la matriz.