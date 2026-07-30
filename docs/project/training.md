# Módulo `training` — el loop de entrenamiento (denoising score matching)

Cuarto módulo de código del TP. Es el **eslabón que une** las tres piezas ya entregadas: `data_generation` (los datos limpios $x_0$), `models` (la red de score $s_\theta$) y `sde` (el proceso forward que define el target). Su trabajo es **entrenar** una red de score $s_\theta(x,t) \approx \nabla_x \log p_t(x)$ —cualquier `ScoreModel`, que **le pasa el caller** ya construida— para que aproxime el score de una SDE dada, minimizando la pérdida de **denoising score matching (DSM)**. El loop es **agnóstico a la red y al origen de datos**: no construye la red ni ramifica por su tipo, y consume un iterador **infinito** de tensores crudos (un batch por paso).

Vive en `diffusion-models/src/diffusion/training/`. Import público sin prefijo `src.`:

```python
from diffusion.training import TrainConfig, train, dsm_loss, save_checkpoint
```

## Qué hace, en un paso

Para un batch de datos limpios $x_0$, un paso de entrenamiento es:

1. Muestrear un tiempo por dato $t \sim \mathcal{U}[t_\text{eps},\, T]$.
2. Ruidear: `x_t, eps = sde.perturb(x0, t)` — muestrea $x_t$ del kernel de perturbación $p_t(x_t \mid x_0)$ y devuelve el ruido estándar usado.
3. Target: `target, weight = sde.score_target(x0, t, eps)` — el score real del kernel $\nabla_{x_t} \log p_t(x_t \mid x_0)$ y el peso $\lambda(t)$ de la pérdida.
4. Predecir: `pred = net(x_t, t)`.
5. Pérdida y paso: $\mathcal{L} = \text{mean}\big(\lambda(t)\,\lVert \text{pred} - \text{target}\rVert^2\big)$, `backward()`, paso de **Adam**.

La pérdida es el estimador de un punto del DSM: un único $(t, x_t)$ por dato, suficiente para batches grandes (ver `docs/knowledge/score-based.md` y `ddpm.md`).

## La pérdida DSM y el pesado $\lambda(t)$

$$\mathcal{L}(\theta) = \mathbb{E}_{x_0,\,t,\,\varepsilon}\Big[\lambda(t)\,\big\lVert s_\theta(x_t, t) - \nabla_{x_t}\log p_t(x_t\mid x_0)\big\rVert^2\Big]$$

Para la **familia escalar-gaussiana** (VP/VE/sub-VP) el kernel es $\mathcal{N}(\text{mean}, \sigma_t^2 I)$, con $x_t = \text{mean} + \sigma_t\varepsilon$, de modo que el target es $\nabla_{x_t}\log p_t = -\varepsilon/\sigma_t$ y el peso recomendado por el módulo `sde` es $\lambda(t) = \sigma_t^2$. Ese pesado tipo-verosimilitud **cancela** el $1/\sigma_t$ del target y vuelve la pérdida equivalente a una de **predicción de ruido**, numéricamente estable:

$$\mathcal{L} = \mathbb{E}\big[\lVert \sigma_t\, s_\theta(x_t,t) + \varepsilon \rVert^2\big].$$

El loop **no inventa** este peso: lo toma tal cual lo devuelve `sde.score_target`. Esa es la clave que mantiene la matemática en el módulo `sde` y la optimización acá.

> **El muestreo de $t$ (configurable desde 25/07/2026, spec `small-t-training-signal`).** El default sortea uniforme en $[t_\text{eps}, T]$ con $t_\text{eps} > 0$; el piso evita $t = 0$, donde $\sigma_t \to 0$ y el target $-\varepsilon/\sigma_t$ se vuelve inestable. Además, `TrainConfig.time_sampling` permite elegir la distribución por nombre (`"uniform"`, el default bit a bit idéntico al histórico, o `"log_uniform"`): la log-uniforme $q(t) \propto 1/t$ concentra las muestras en $t$ chico (≈50% de la masa en $[10^{-4}, 10^{-2}]$ con `t_eps=1e-4`, contra ~1% de la uniforme) y la pérdida se corrige por el **likelihood ratio** $w(t) = p_\text{unif}(t)/q(t)$ (importance sampling puro: la pérdida esperada queda **matemáticamente idéntica** a la del muestreo uniforme — solo baja la varianza del estimador en la franja de $t$ chico; la equivalencia está verificada por Monte Carlo en la suite, con tolerancia $3\cdot SE$ validada por sabotaje). La corrección la aplica `dsm_loss` vía `sample_weights`; el sampler vive en `training/time_sampling.py` (registry/factory `make_time_sampler`).

## El seam agnóstico a la SDE

El mismo `train()` corre las tres SDEs **sin ramificar** por tipo. El batch de puntos 2D que entrega `data_generation` se pasa **crudo** a `sde.perturb`/`sde.score_target`, que ya devuelven las shapes correctas: estado escalar, red `data_dim=2`, $x_0$ es el punto $(x, y)$.

La red la construye el **caller** —`make_model("mlp", data_dim=sde.data_dim)` o una instancia explícita— y se la pasa a `train`, que no la instancia ni ramifica por su tipo (es agnóstico a la red y al origen de datos). Todo lo demás es idéntico. Es la materialización en código de "la red es la variable de control": misma arquitectura, mismos hiperparámetros, solo cambia la SDE.

## Regla del Eje 1: un entrenamiento por variante

Cambiar el forward SDE cambia $p_t(x)$ y por lo tanto el score a aprender → **hay que reentrenar**. Cada corrida construye una red nueva desde cero (el caller la instancia y se la pasa a `train`), así que entrenar VP, VE y sub-VP son tres corridas independientes. Los samplers del **Eje 2** reusan la red entrenada **sin** reentrenar. Por eso conviene una corrida = un archivo de config versionable (ver abajo), una por celda del estudio.

## API

Núcleo (en `losses.py`, sin estado ni I/O — se testea directo):

| Función | Qué hace |
|---|---|
| `dsm_loss(net, sde, x0, t, *, generator=None, sample_weights=None)` | Pérdida DSM de un batch; escalar diferenciable. Con `sample_weights` (el likelihood ratio del muestreo de $t$) la media es ponderada — con `None` el camino es bit a bit el histórico. |
| `sample_timesteps(n, T, t_eps, *, generator=None, device=None)` | $n$ tiempos $\sim\mathcal{U}[t_\text{eps},T]$, shape `(n,)`. |
| `make_time_sampler(name, T, t_eps)` / `available_time_samplers()` | Factory del muestreo de $t$ (`time_sampling.py`): `"uniform"` (delega en `sample_timesteps`, pesos `None`) o `"log_uniform"` (pesos $w(t) = t\,\ln(T/t_\text{eps})/(T-t_\text{eps})$, $E_q[w]=1$). Contrato `sample(n, *, generator, device) -> (t, pesos)`. |

Loop y persistencia (en `trainer.py`):

| Símbolo | Qué es |
|---|---|
| `TrainConfig` | Dataclass **acotado al loop**: `num_steps, lr, t_eps, grad_clip, seed, device, log_every, checkpoint_every, keep_last_checkpoints, time_sampling, ema_decay, amp`. `time_sampling` (default `"uniform"`, retrocompatible) elige la distribución del muestreo de $t$; el loop construye el sampler una vez (fail-fast antes de consumir datos) y pasa los pesos a `dsm_loss`. Ya no lleva hiperparámetros de red (van al constructor / `make_model`) ni de dataset (`n_samples`/`batch_size` van a la fuente de datos). `log_every` es **solo para el print** de consola (media móvil), desacoplado del `history`. `checkpoint_every` (default `0`) activa el checkpointing intermedio (ver abajo); `keep_last_checkpoints` (default `None` = conservar **todos**) fija la **retención rolling** de esos snapshots (ver abajo). `ema_decay` (default `None` = **sin EMA**) activa la sombra EMA de los pesos (ver abajo). `amp` (default `False` = **sin precisión mixta**) activa AMP en el núcleo de optimización (ver abajo). |
| `TrainResult` | `net` entrenada (cualquier `ScoreModel`), `history` (**serie per-step completa**: una entrada por paso, `len == num_steps`), `config`, `sde_name`, `data_dim` (`= sde.data_dim`, lo copia el checkpoint) y `ema_state` (la foto **clonada** de la sombra EMA — mismas claves que el `state_dict` de la red, con los parámetros promediados—, o `None` sin EMA). |
| `train(sde, model, data, config, *, generator=None, on_checkpoint=None, resume=None)` | Corre el loop **por pasos** (`num_steps`) y devuelve `TrainResult`. Recibe la red ya construida y un iterador infinito de datos; no instancia la red ni ramifica por su tipo. `on_checkpoint(tag, snapshot)` es un callback **opcional** de checkpointing intermedio (el `snapshot` es un `TrainSnapshot`; ver abajo): el loop decide *cuándo*, el callback decide *cómo/dónde* — `train` no toca el filesystem. `resume` (un `ResumeState` opcional) hace el loop **reanudable** (ver "Reanudación"). |
| `save_checkpoint(result, path, *, model_spec=None, raw_sibling=False)` / `load_checkpoint(path)` | Persistencia **model-agnóstica** (R5-c): guarda `state_dict` + `meta{sde_name, data_dim, history, model?, ema?}`; es el **punto único de publicación** del EMA (con `result.ema_state` presente guarda la sombra y marca `meta["ema"]`; con `raw_sibling=True` escribe además el hermano `{stem}_raw.pt` de crudos — ver "EMA de los pesos"). `load_checkpoint` devuelve `(state_dict, meta)` sin reconstruir la red (ver más abajo). |
| `TrainSnapshot` / `ResumeState` | Lo que viaja en cada snapshot intermedio. `TrainSnapshot{result: TrainResult, resume: ResumeState}`: los pesos+history (para `save_checkpoint`) **y** el estado para reanudar. `ResumeState{optimizer_state, start_step, torch_rng_state, generator_state, history}`: lo que los pesos no capturan (Adam + paso + azar), más `ema_state` y `raw_model_state` **opcionales** (`None` sin EMA). |
| `save_resume_state(path, resume)` / `load_resume_state(path)` | Persistencia del **sidecar** de resume (`…_resume.pt`): guarda `{optimizer_state, step, torch_rng_state, generator_state}` (sin `history`, que vive en el `meta` de los pesos), más `raw_model_state`/`ema_state` **solo si están** (corrida con EMA). `save_resume_state` levanta `ValueError` si falta alguno de los cuatro campos requeridos. |
| `EmaShadow(module, decay)` | La sombra EMA (en `ema.py`): `update(step)` la avanza un paso, `state_dict()` da la foto publicable (clonada) y `load_state(shadow)` la restaura al reanudar. Es un observador pasivo del módulo (ver abajo). |

> **La curva de pérdida (`history`) es la serie per-step completa** — una entrada por paso, no un promedio por ventana. Es lo más fiel: la pérdida DSM de un paso es de cola pesada (depende mucho del `t` aleatorio que se sortea cada paso), así que promediar la distorsiona; guardando todo, siempre se puede suavizar después para graficar. El print de consola (`log_every`) muestra una media móvil, pero es **solo display** y está desacoplado de lo que se guarda.

Compañeros del flujo (viven en otros módulos, pero el loop los necesita):

| Símbolo | Módulo | Qué hace |
|---|---|---|
| `make_model(name, **kwargs)` | `diffusion.models` | Construye la red desde una receta `(name, kwargs)` (registry `mlp` / `unet`); descarta los kwargs que no aplican a la firma. El caller la usa para armar la red que le pasa a `train`. |
| `infinite_bare(loader)` | `diffusion.data_generation` | Envuelve un `DataLoader` finito (el de `PointDistribution.dataloader`) en un iterador **infinito** de tensores crudos: lo recorre en bucle y desempaqueta la 1-tupla `(x0,)`. Es lo que `train` consume con `next()`. |

Config-driven (en `config.py`):

| Símbolo | Qué es |
|---|---|
| `load_config(path)` | Lee un YAML a `dict` (necesita `pyyaml`). |
| `build_run(raw)` | Ensambla un `RunSpec` reusando `make_sde`/`make_distribution`/`make_model` y envolviendo el dataloader con `infinite_bare`. Valida `train:` contra los campos de `TrainConfig` (rechaza claves desconocidas). |
| `RunSpec` | Una corrida lista: `sde` + red (`model`) + fuente de datos infinita (`data`) + `TrainConfig` + `model_spec` (la receta `{name, kwargs}` para el checkpoint) + rutas de salida (`checkpoint`, `loss_curve`). |

Reanudación (en `resume.py`, **puro y sin torch** salvo `load_resume`):

| Símbolo | Qué es |
|---|---|
| `resolve_resume(final_checkpoint, *, force=False, resume_from=None) -> ResumePlan` | Decide `skip` / `resume` / `fresh` mirando **solo** el filesystem (no entrena ni escribe). |
| `ResumePlan{action, weights_path, step}` | DTO con la decisión. |
| `discover_snapshots(final) -> list[(step, path)]` | Lista los `…_stepNNNNN.pt` hermanos del final, ordenados por paso. Solo matchea ese patrón sobre el mismo stem, así que quedan afuera el final, los sidecars, el hermano `_raw`, las otras corridas del directorio y los `…_best.pt` **legados** (el loop ya no los emite, pero los que quedaron en disco se toleran). |
| `resume_sidecar_path(weights_path) -> Path` | `X_stepNNNNN.pt` → `X_stepNNNNN.resume.pt`. |
| `validate_compatible(meta, *, sde_name, model_spec, data_dim)` | Igualdad **exacta** de SDE / `data_dim` / receta de red, antes de reanudar (`ValueError` si difieren). |
| `load_resume(weights_path, *, expected, map_location="cpu") -> (state_dict, meta, ResumeState)` | Reúne pesos + sidecar en un `ResumeState` listo para `train(resume=…)`; `FileNotFoundError` si falta el sidecar. El `state_dict` que devuelve son los **crudos del sidecar** cuando están (corrida con EMA: el checkpoint publica la sombra) y los del checkpoint cuando no (sidecar viejo = como siempre). |

## Corridas por config (YAML)

Cada celda del estudio se describe en un `.yaml`. El core no sabe de archivos: `config.py` es un front-end fino que arma un `RunSpec` (SDE + red + fuente de datos infinita + `TrainConfig` + receta de red). Estructura:

```yaml
sde:                 # -> make_sde(name, **resto)
  name: vp           # vp | ve | sub_vp
  beta_min: 0.1
  beta_max: 20.0
data:                # -> make_distribution(shape, dim, **resto)
  shape: mixture
  dim: 2
  n_samples: 4000    # tamaño del dataset (parámetro de la fuente, NO del TrainConfig)
  batch_size: 256    # tamaño de batch (parámetro de la fuente, NO del TrainConfig)
  n_components: 8
  standardize: true
  seed: 0
train:               # -> campos de TrainConfig (solo el loop de optimización)
  num_steps: 240     # pasos de optimización (reemplaza epochs)
  lr: 0.002
  t_eps: 1.0e-3
  grad_clip: 1.0     # opcional
  seed: 0
  device: cpu
  checkpoint_every: 0  # opcional; 0 = solo el checkpoint final. N>0 = snapshots intermedios
  keep_last_checkpoints: null  # opcional; null = conservar todos. N>=1 = solo los N más nuevos (borra los viejos)
  ema_decay: 0.999     # opcional; sin la clave = sin EMA (pesos crudos, como antes)
  amp: false           # opcional; false = sin precisión mixta (como antes). true = AMP (ver abajo)
# model:             # opcional: la red es la variable de control (normalmente fija)
#   name: mlp        #   si falta, se usa {name: mlp} dimensionado desde el dato/SDE
#   hidden_dim: 256
out:                 # rutas relativas al cwd
  checkpoint: models/vp_mixture.pt
  loss_curve: models/vp_mixture_loss.png
```

Ejemplo listo en `diffusion-models/config/`: `vp_mixture.yaml`.

## Checkpoint model-agnóstico (R5-c)

`save_checkpoint`/`load_checkpoint` no dependen de ninguna clase de red concreta. El `.pt` guarda el `state_dict` de la red y una `meta` mínima: `{sde_name, data_dim, history}` más —opcionalmente— una **receta genérica** `model = {"name": str, "kwargs": dict}` y la marca `ema` de la sección siguiente. Esa receta la aporta el caller vía `save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": {...}})`; en el camino config-driven la transporta el `RunSpec` (`spec.model_spec`) y `scripts/train.py` la pasa sola.

`load_checkpoint(path)` devuelve `(state_dict, meta)` **sin reconstruir** la red: es el caller quien arma la red y le carga los pesos. La reconstrucción canónica es `make_model(recipe["name"], **recipe["kwargs"])` seguida de `net.load_state_dict(state_dict)`, y la hace `diffusion.samplers.generate_from_checkpoint`, que cierra el pipeline forward→score→sampleo desde el checkpoint. Si el checkpoint se guardó **sin** receta (`model_spec=None`) sigue siendo válido, pero al generar hay que pasarle una red explícita (`model=`). Así el mismo checkpoint sirve al `ScoreMLP` (Fase 1) y a la `ScoreUNet` (Fase 2) sin que `training` importe ninguna red concreta.

## EMA de los pesos (opt-in, desde 27/07/2026 — spec `ema-weights`)

La pérdida DSM per-step es muy ruidosa (el $t$ se sortea de nuevo en cada paso, y en la celda de gatos el batch es chico), así que **los pesos del último paso de Adam son una foto arbitraria** de una trayectoria que oscila. Las implementaciones de referencia no samplean con esa foto: mantienen durante el entrenamiento una **media móvil exponencial (EMA)** de los parámetros y generan con ella. Song et al. lo hacen en `score_sde_pytorch` (`models/ema.py`, `ExponentialMovingAverage` — es la convención que sigue esta implementación), DDPM (Ho et al., 2020) también, y de EDM (Karras et al., 2022) viene la idea de la **rampa de warmup** para que el promedio no quede dominado por la inicialización en corridas cortas. El módulo suma esa pieza —y solo esa— en `training/ema.py` (`EmaShadow`):

$$\text{ema}_s = d_s\,\text{ema}_{s-1} + (1 - d_s)\,\theta_s, \qquad d_s = \min\!\Big(d,\ \frac{1+s}{10+s}\Big)$$

donde $s$ es la cantidad de **pasos completados** (1-indexado: se llama `update(s)` justo después del `optimizer.step()` número $s$, y la sombra arranca en $\text{ema}_0 = \theta_0$) y $d$ es el decay configurado. El factor $(1+s)/(10+s)$ es la rampa: en $s=1$ vale $2/11 \approx 0.18$ (el arranque es casi un promedio simple) y crece hacia $1$, tocando $d = 0.999$ recién en $s \approx 8990$ — para corridas de 2k–19k pasos la ventana efectiva sigue creciendo durante buena parte de la corrida, que es justo el comportamiento buscado.

- **Opt-in y retrocompatible.** `TrainConfig.ema_decay = None` (**default**) = sin EMA: no se construye ninguna sombra, no hay ramas nuevas activas y la corrida es **bit a bit** la histórica (mismos pesos finales, mismo `history`, misma secuencia de RNG con la misma semilla). El decay recomendado del estudio es **0.999**, y es lo que declara `config/vp_mixture.yaml`.
- **Fail-fast.** La sombra se construye **después** de mover la red al device (clona sus tensores, tienen que estar en el device final) y **antes** de consumir el primer batch: un `ema_decay` no finito o fuera del intervalo **abierto** $(0,1)$ revienta con `ValueError` —con el valor recibido en el mensaje— antes de entrenar, igual que un `time_sampling` desconocido. También falla si no logró rastrear ningún tensor entrenable del módulo (una sombra vacía publicaría pesos crudos disfrazados de EMA).
- **Observador pasivo.** La sombra solo *lee* los tensores de la red después del paso del optimizador: no escribe en la red, no toca el optimizador y **no consume RNG**. Con la misma semilla, la trayectoria de optimización (pesos crudos + curva de pérdida) es idéntica a la de una corrida sin EMA — el EMA no cambia lo que se optimiza, solo agrega qué se publica.
- **Agnóstica a la red.** Opera sobre el `state_dict` del módulo recibido sin ramificar por su tipo (`ScoreMLP`, `ScoreUNet`), así que compone con el `EpsilonScoreWrapper`: su `state_dict` delega al interno, de modo que las claves de la sombra quedan de **red pelada** y el checkpoint publicado sigue siendo reconstruible con `make_model` + la receta.
- **Parámetros vs buffers.** El promedio se aplica a los tensores **entrenables**; los buffers no entrenables se copian del módulo vivo al publicar. Es la convención de referencia y acá es *exacta*, no aproximada: los buffers del repo son constantes (el `denom` del embedding sinusoidal; GroupNorm no lleva running stats).

**Qué publica el checkpoint.** `save_checkpoint` es el **punto único de publicación**: si `result.ema_state` está presente, lo que guarda como `model_state` es la sombra —no los pesos vivos del último paso— y la `meta` gana la marca de trazabilidad `ema = {"decay": d}`. El formato del blob no cambia (`{model_state, meta}`), así que la generación, el wrapper ε y los notebooks de auditoría consumen los pesos promediados **sin modificarse**. Como *todos* los checkpoints pasan por esa función, la política vale igual para el final y para los snapshots periódicos del callback (cada intermedio publica la sombra de *su* momento). Sin EMA el contenido es idéntico al de antes de la feature (mismos pesos, misma meta, sin la clave `ema`). La convención de lectura de la marca, la que interpretan los consumidores:

| `meta` | Qué son los pesos publicados |
|---|---|
| **sin** `ema` | **crudos** — es lo que son todos los checkpoints anteriores a esta feature (retrocompatibilidad). |
| con `ema = {"decay": d}` | la **sombra EMA** con ese decay. |
| con `raw_of = "<archivo>.pt"` (y sin `ema`) | **crudos**: el *hermano de crudos* del checkpoint EMA que nombra. |

**El hermano de crudos.** `save_checkpoint(..., raw_sibling=True)` escribe además `{stem}_raw.pt` con los **pesos crudos finales**: un checkpoint estándar (mismo formato, misma receta) cuya meta lleva `raw_of` en lugar de `ema`. Sirve para la comparativa crudo-vs-EMA de la **misma** corrida sin tocar la lógica de medición de los audits. Lo activa el guardado **final** (`scripts/train.py` lo pasa siempre; los snapshots intermedios no lo necesitan, sus crudos ya viajan en el sidecar de resume). Sin EMA activo no escribe nada —el principal ya publica los crudos y el hermano sería un duplicado exacto—, y el sufijo `_raw` no matchea el patrón `_stepNNNNN`, así que nunca se elige como punto de reanudación.

> **Retiro del checkpoint "best" (27/07/2026).** La misma spec **retiró** el tag `best` que emitía el loop: hoy la única cadencia que emite es la periódica (`stepNNNNN`). El motivo es exactamente el que justifica el EMA: elegir un checkpoint por la **pérdida cruda per-step** es ruidoso —el $t$ de cada paso es aleatorio— y correlaciona mal con la calidad de las muestras; además el mecanismo nunca se usó para ninguna decisión del estudio. Los `X_best.pt` que quedaron en disco de corridas previas se siguen **tolerando** (`discover_snapshots` los excluye, como siempre). La decisión está registrada en `.kiro/specs/ema-weights/research.md`.

## Eficiencia en GPU (opt-in, desde 29/07/2026 — spec `gpu-training-efficiency`)

El loop ya movía la red y cada batch al `device` configurado, pero no tenía las palancas que hacen rendir una U-Net convolucional en GPU (Fase 2, dataset de imágenes grande). Esta spec suma **tres palancas ortogonales**, todas **opt-in y sin regresión**: con los defaults la corrida es **bit a bit** la histórica en CPU, y el estudio de ablación no se toca (la red sigue determinística y fija; todo el no-determinismo nuevo vive **afuera** de la red). **Multi-GPU / DDP quedó explícitamente fuera de alcance** (decisión del autor), igual que `torch.compile`, los kernels custom y exponer estas palancas en el camino de datos de **puntos** (corre en CPU por diseño).

> **Nota CPU-vs-GPU.** El **speedup** real de estas tres palancas solo se observa en **GPU**. En CPU —el camino de la suite de tests— el autocast usa bfloat16, el escalador va deshabilitado (passthrough), `cudnn.benchmark` no se toca y `non_blocking` es transparente: se verifica que el camino completo **no rompe ni cambia el resultado** (no-regresión), no la velocidad. Por eso la suite ejercita las ramas activadas sin necesitar GPU, pero no mide performance (no hay benchmark cuantitativo en CI).

### 1 · Precisión mixta (AMP) — `TrainConfig.amp`

`amp` (default `False` = **desactivada**, bit a bit idéntica al loop previo: cero ramas nuevas activas, núcleo de optimización byte-idéntico — R1.4) activa la precisión mixta en el núcleo de `train()`:

- **`autocast` sobre forward + pérdida.** El forward de la red y `dsm_loss` se computan bajo `torch.autocast(device_type=device.type, enabled=config.amp)` (la API unificada `torch.amp.*`, no la deprecada `torch.cuda.amp.*`). En GPU corre en la precisión reducida del device; en CPU usa bfloat16. El `backward` queda **fuera** del contexto (contrato de `torch.amp`). Con `amp=False` el contexto es `enabled=False`, un no-op documentado.
- **Escalador de gradiente condicional.** Solo con `amp=True` se construye `scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))` — se calca el patrón EMA (se crea *solo* si la palanca está activa). En **CUDA** escala el gradiente para el paso del optimizador; en **CPU** el escalado no aplica (autocast bf16), así que queda `enabled=False` (passthrough) pero **existe** para uniformar el camino y ejercitar la persistencia en los tests. Con `amp=False` el escalador es `None` y el núcleo queda idéntico al histórico.
- **Des-escalar antes del recorte (R1.2).** Con escalador y `grad_clip` configurado, el orden es `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → `clip_grad_norm_` → `scaler.step(optimizer)` → `scaler.update()`: el recorte opera sobre la norma **real** del gradiente, no la escalada.
- **EMA y pesos en float32 (R1.3).** El autocast no cambia el dtype de los **parámetros** (siguen float32); la sombra EMA se actualiza **después** del paso real (igual que sin AMP) y vive en float32. Así el checkpoint publica exactamente los mismos pesos y su **formato / contrato de consumo no cambia** — la generación, el wrapper ε y los audits consumen igual.
- **Fail-fast (R1.5).** `amp` es **`bool` estricto**: un `int` (incluidos `0`/`1`) o un `float` **no** son booleanos, así que revientan con `ValueError` —con el valor recibido en el mensaje— **antes de consumir el primer batch**, con el mismo criterio que un `time_sampling` desconocido o un `ema_decay` fuera de rango.

**Reanudación fiel con AMP (R2).** El sidecar de resume gana una clave **opcional** más, `scaler_state` (el `GradScaler.state_dict()`), que viaja **solo si** el escalador existe (AMP activo). Al reanudar, el loop lo **restaura** en su escalador antes de continuar (así el factor de escala sigue del valor ya ajustado, no del inicial). Es idéntico al patrón de EMA y **retrocompatible**: sin AMP el sidecar no gana ninguna clave, así que los sidecars anteriores a la feature siguen cargando igual (R2.4). Las dos combinaciones incoherentes config↔sidecar se **rechazan** con `ValueError` antes de entrenar (R2.5): AMP pedido sin `scaler_state` guardado, y `scaler_state` presente con AMP no pedido — nunca se reanuda desde un estado de precisión inconsistente. En CPU el escalador va deshabilitado y su `state_dict()` es `{}` (un dict **vacío**, no `None`): la "presencia" se decide por `is None`, así que un `{}` cuenta como presente y la ruta de persistencia se ejercita igual sin GPU (la fidelidad del *scale factor* dinámico solo es afirmable en CUDA; en CPU se valida la ruta, no el valor).

### 2 · Carga de datos eficiente — knobs del bloque `data:` de imágenes

La rama `images` de `build_data_source` expone dos knobs más, mapeados tal cual a `infinite_batches` (que ya los aceptaba): `num_workers` (default `0`: carga en el proceso principal) y `pin_memory` (default `False`: sin memoria fijada). Sus defaults reproducen el comportamiento actual **sin cambio observable** (R3.2); declararlos alimenta la carga paralela y la memoria fijada que evitan que la GPU quede ociosa esperando imágenes en datasets grandes (R3.1). Se suman a las **claves conocidas** *antes* del rechazo de unknowns, así que una clave de carga desconocida (p. ej. un typo `num_workerss`) sigue fallando enumerándola (R3.4). Solo aplica al camino de **imágenes**; el de **puntos** no gana knobs (corre en CPU por diseño).

Complementaria a los knobs, la **transferencia de batch es no bloqueante** de forma **incondicional**: `next(data_iter).to(device, non_blocking=True)` (R3.3). Es inofensiva sin memoria fijada y en CPU (torch la ignora si no aplica); su beneficio real aparece con `pin_memory=True` + CUDA. `train()` no conoce el `pin_memory` del loader, así que no se gatea — de ahí que sea incondicional.

### 3 · Autotune de kernels convolucionales — `cudnn.benchmark`

En el setup de `train()`, si el device es **CUDA** se prende `torch.backends.cudnn.benchmark = True` (`_enable_cudnn_autotune`): cuDNN elige los kernels convolucionales más rápidos para los **shapes fijos** de la corrida (la arquitectura es la variable de control, los shapes no cambian entre pasos). Es **auto-on en CUDA, sin flag de config**. En **CPU no toca nada** (R4.2): el estado global queda como estaba, así que el camino de la suite no tiene efecto observable. Solo cambia la **elección de kernels**, no la arquitectura ni el objetivo (R4.3) — puede introducir un no-determinismo menor en GPU, consistente con la fidelidad "equivalente, no bit-idéntica" de la reanudación (aceptado por el autor).

## Checkpointing intermedio (opt-in)

Por defecto **solo se guarda el checkpoint final** (el `state_dict` tal como quedó en el último paso — o la sombra EMA, si la corrida la mantiene): el loop no persiste nada, y `save_checkpoint` lo escribe una vez terminado `train`. Para guardar también estados intermedios está el switch `checkpoint_every` de `TrainConfig`:

- `checkpoint_every = 0` (**default**) — nada intermedio; comportamiento idéntico al histórico (sin regresión).
- `checkpoint_every = N > 0` — además del final, el loop pide guardar un **snapshot periódico** cada `N` pasos (tag `step{N:05d}` → `…_stepNNNNN.pt`), con cadencia propia (se chequea cada paso, así `N` no tiene que ser múltiplo de nada). El **último** paso se omite porque ya lo cubre el checkpoint final. Es la **única** cadencia que emite el loop (el tag `best` se retiró; ver arriba).

El diseño mantiene `train` **sin I/O**: el loop decide **cuándo** (la política de cadencia) e invoca un callback `on_checkpoint(tag, snapshot)`, donde `snapshot` es un **`TrainSnapshot`** (`snapshot.result` = pesos+history, `snapshot.resume` = estado para reanudar); el **caller** decide **cómo/dónde** persistir. En el camino config-driven eso lo arma `scripts/train.py`, que deriva la ruta hermana del `out.checkpoint` con `Path.with_stem` (`vp_mixture.pt` → `vp_mixture_step00050.pt`) y persiste **dos** artefactos por snapshot: los pesos (`save_checkpoint(snapshot.result, …)`) y el **sidecar** de resume (`save_resume_state(resume_sidecar_path(tagged), snapshot.resume)`), así una interrupción deja un punto reanudable. A mano:

```python
base = pathlib.Path("models/vp_mixture.pt")
def on_checkpoint(tag, snap):                              # snap: TrainSnapshot
    tagged = base.with_stem(f"{base.stem}_{tag}")
    save_checkpoint(snap.result, tagged, model_spec=my_model_spec)      # pesos + meta
    save_resume_state(resume_sidecar_path(tagged), snap.resume)          # sidecar (opcional)
train(sde, model, data, TrainConfig(num_steps=1000, checkpoint_every=200), on_checkpoint=on_checkpoint)
```

> **Ojo:** el callback recibe un `TrainSnapshot`, no un `TrainResult` — hay que pasar `snap.result` a `save_checkpoint` (pasar `snap` directo falla). El sidecar es opcional: sin él tenés snapshots de pesos, pero no podés **reanudar** desde ellos.

Se activa por YAML (`train.checkpoint_every`) o con el override `--checkpoint-every` del CLI. Si `checkpoint_every > 0` pero no hay `out.checkpoint`, el CLI avisa y no escribe snapshots; y si `checkpoint_every = 0`, avisa que la corrida **no deja puntos de reanudación**.

## Reanudación (resume): skip / resume / fresh

Los snapshots intermedios no son solo para inspeccionar: habilitan **reanudar** una corrida larga interrumpida sin perder el estado del optimizador ni el azar. La pieza clave es que un snapshot periódico deja **dos** archivos hermanos:

- `X_stepNNNNN.pt` — los **pesos** + `meta{sde_name, data_dim, history, model?}` (`save_checkpoint`).
- `X_stepNNNNN.resume.pt` — el **sidecar** (`save_resume_state`): `{optimizer_state, step, torch_rng_state, generator_state}` (+ `raw_model_state`/`ema_state` con EMA activo, + `scaler_state` con AMP activo; ver abajo y la sección de eficiencia en GPU) — exactamente lo que los pesos no capturan (los momentos de Adam, el paso alcanzado y **ambos** RNG: el global de torch y el `generator` del ruido/muestreo de `t`). El `history` **no** se duplica en el sidecar: vive en el `meta` de los pesos.

Por eso el resume es **fidedigno** (no es weights-only): al reanudar, `train(resume=…)` restaura optimizador + azar + paso + `history` y sigue el mismo stream. `num_steps` se interpreta como el **total** a alcanzar (corre `range(start_step, num_steps)`); reanudar con `num_steps ≤ start_step` es un **no-op** silencioso.

**Con EMA activo** el estado de la corrida queda *partido* entre los dos artefactos: el checkpoint de pesos publica la **sombra**, así que el sidecar gana dos claves más —`raw_model_state` (los pesos **crudos** del momento, clonados: es con lo que hay que seguir optimizando) y `ema_state` (la sombra, para restaurarla en vez de reconstruirla desde los pesos, que valen $\theta_N$ y no el promedio acumulado)—. Son **opcionales**: sin EMA el sidecar no gana ninguna clave, así que los sidecars anteriores a la feature siguen cargando igual que siempre. `load_resume` devuelve entonces los crudos del sidecar como el `state_dict` a cargar en la red (el checkpoint queda solo como fuente de `meta`/`history`), y si esa clave no está —sidecar viejo, corrida sin EMA— se comporta exactamente como antes. Las dos combinaciones incoherentes entre config y sidecar se **rechazan** con `ValueError` antes de entrenar: EMA pedido sin sombra guardada (se perdería el promedio de todos los pasos ya corridos) y sombra guardada sin EMA pedido (se descartaría en silencio, y el final publicaría crudos donde los intermedios publicaron el promedio).

`scripts/train.py` orquesta la decisión con `resolve_resume` (puro, solo mira el filesystem):

1. **`--resume-from PATH|STEP`** → siempre `resume` desde ese snapshot puntual (manda sobre el skip). Si no resuelve (ni ruta existente ni paso descubierto) → error que lista los snapshots disponibles (exit 2).
2. si no, y el checkpoint final ya existe y **no** hay `--force` → **`skip`** (corrida completa; no se sobrescribe nada, exit 0).
3. si no (final ausente **o** `--force`): descubre snapshots; si hay → **`resume`** desde el **más nuevo** (mayor paso); si no → **`fresh`** (desde cero).

Antes de reanudar, `load_resume` valida compatibilidad **exacta** (`validate_compatible`): misma SDE, mismo `data_dim` y misma receta de red — reanudar sobre un estado que no corresponde falla con `ValueError`. Si falta el sidecar del snapshot elegido → `FileNotFoundError` que lo nombra (probablemente la corrida original no usó `checkpoint_every>0`).

> **Limitación:** `validate_compatible` compara `sde_name`/`data_dim`/receta, **no** los hiperparámetros de la SDE (betas, `sigma_max`). Como el resume rearma la SDE desde el **mismo** YAML, en la práctica es consistente; pero es el mismo hueco que el footgun del checkpoint (ver `dataflow.md`).

## Cómo correr

```bash
# Uso a mano (Python): el caller arma la red y la fuente de datos, train solo entrena.
from diffusion.sde import make_sde
from diffusion.data_generation import make_distribution, infinite_bare
from diffusion.models import make_model
from diffusion.training import TrainConfig, train, save_checkpoint

sde = make_sde("vp")
dist = make_distribution("mixture", dim=2, n_components=8, seed=0)
model = make_model("mlp", data_dim=sde.data_dim)                 # red = variable de control
data = infinite_bare(dist.dataloader(4000, 256, shuffle=True))   # iterador infinito de x0
result = train(sde, model, data, TrainConfig(num_steps=240))
save_checkpoint(result, "models/vp_mixture.pt",
                model_spec={"name": "mlp", "kwargs": {"data_dim": sde.data_dim}})

# CLI por config (desde diffusion-models/):
python scripts/train.py --config config/vp_mixture.yaml
python scripts/train.py --config config/vp_mixture.yaml --num-steps 50 --device cpu
python scripts/train.py --config config/vp_mixture.yaml --checkpoint-every 50   # deja snapshots + sidecars
python scripts/train.py --config config/vp_mixture.yaml --force                 # reentrena/reanuda aunque exista el final
python scripts/train.py --config config/vp_mixture.yaml --resume-from 50         # reanuda desde el snapshot del paso 50

# Smoke del módulo (desde diffusion-models/src/):
python -m diffusion.training
```

Flags del CLI (`scripts/train.py`): `--config` (req.), `--num-steps`, `--device`, `--checkpoint-every`, `--keep-last N` (overrides del `.yaml`; `--keep-last` = **retención rolling**: tras cada snapshot, conserva solo los N `…_stepNNNNN` más nuevos —con su sidecar— y borra los viejos vía `prune_snapshots`; el checkpoint final y su `_raw` nunca se tocan, y siempre queda al menos el más nuevo para reanudar), `--force` (saltea el `skip` si el final existe), `--resume-from PATH|STEP` (reanuda desde un snapshot puntual) y `--quiet` (apaga la barra de progreso y el print por paso). Por defecto el CLI muestra una **barra de progreso** (`tqdm`, a stderr) con porcentaje, ETA e it/s y la pérdida (media móvil de `log_every`) como postfix; al reanudar la barra va de `start_step` a `num_steps`, así el % y el ETA son correctos. Es **display-only** (a nivel librería, `train(..., progress=True)`): no cambia el resultado ni el `history`. El CLI guarda el checkpoint final (`.pt`) y una curva de pérdida (`.png`) en las rutas de `out`; con `checkpoint_every>0` deja además los `…_stepNNNNN.pt` + `…_stepNNNNN.resume.pt`, y con `train.ema_decay` configurado el hermano de crudos `…_raw.pt` junto al final (el guardado final pasa `raw_sibling=True`). El `…_best.pt` que dejaba antes **ya no se emite**.

## Stack y dependencias

Torch es **dependencia dura** del módulo (como `mlp` y `sde`): importa `torch` directo (no diferido). El front-end de config agrega **`pyyaml`** (>= 6) a las dependencias del proyecto.

## Tests

`diffusion-models/tests/test_training.py` (111 tests, en verde; suite completa sin regresiones):

- `dsm_loss` para las **3 SDEs**: escalar finito, diferenciable, gradientes finitos en la red; reproducible con `generator`.
- `sample_timesteps`: shape, rango $[t_\text{eps}, T]$, reproducibilidad, horizonte $T$ distinto.
- `train`: **usa la red recibida** y registra el `data_dim` correcto por SDE (las 3 variantes); `history` es la **serie per-step completa** (`len == num_steps`), independiente de `log_every`; la **pérdida baja** (medias de bloque) en VP sobre la mezcla de gaussianas; reproducibilidad con misma `seed`; camino `grad_clip`.
- `TrainConfig`: acotado al loop (no expone campos de red ni de dataset); `checkpoint_every` arranca en `0`.
- `save_checkpoint`/`load_checkpoint`: ida y vuelta reconstruye la red con los mismos pesos; sin `model_spec` el checkpoint omite la receta `model`.
- **Checkpointing intermedio**: con `checkpoint_every=0` el callback no se invoca (sin regresión); con `N>0` emite **solo** los snapshots periódicos correctos (múltiplos de `N`, excluido el último paso) y **ningún** `best` (el test del mecanismo retirado se reescribió para verificar su ausencia, incluido que no queda el artefacto `…_best.pt` en disco); el wiring estilo-CLI escribe los `…_stepNNNNN.pt` cargables con la metadata esperada.
- **Muestreo de $t$ y parametrización ε**: los samplers de `time_sampling.py` (rango, concentración de masa, fórmula de los pesos y $E_q[w]=1$, reproducibilidad), la equivalencia en esperanza de la log-uniforme por Monte Carlo, el camino bit a bit del default y el round-trip de la receta con `score_parametrization: epsilon` hasta `generate_from_checkpoint`.
- **EMA** (`ema-weights`): la matemática de `EmaShadow` contra una **réplica cerrada** dentro y cruzando el warmup; validación del decay y del contador; claves y clones del `state_dict` (parámetros EMA + buffers del módulo vivo); pasividad (no escribe en la red, no consume RNG) y determinismo; `load_state` y su rechazo de fotos incompletas; el default `ema_decay=None` **bit a bit** idéntico al loop previo; el loop promediando la trayectoria cruda de Adam sin alterarla; publicación de la sombra + marca `ema.decay` en el checkpoint final y en los intermedios, contenido idéntico al actual sin EMA, hermano de crudos (y su no-escritura sin EMA, y que `discover_snapshots` no lo levanta), composición con el wrapper ε; `build_run` aceptando `ema_decay` desde `dict` y desde YAML sin romper la validación estricta; y los dos guards de coherencia al reanudar.
- **Eficiencia en GPU** (`gpu-training-efficiency`): el default `amp=False` **bit a bit** idéntico al loop previo (misma trayectoria por semilla, sin ramas nuevas); `amp=True` corriendo en CPU (autocast bf16 + escalador passthrough) y aprendiendo; el `unscale_` **antes** del recorte con `grad_clip`; el orden EMA tras el paso con AMP; fail-fast ante un `amp` no booleano; y que `cudnn.benchmark`/`non_blocking` no rompen en CPU. Los knobs de carga (`num_workers`/`pin_memory`) se testean en `test_config_image.py` (pase de valores a `infinite_batches`, defaults, y clave de carga desconocida → error).
- `build_run`/`load_config`: ensamblado desde `dict` y desde YAML; el bloque `model:` sobreescribe el default; `train.checkpoint_every` se pasa al `TrainConfig`; falla ante claves faltantes (`sde.name`, `data.shape`) o desconocidas en `train:`. Un **smoke end-to-end de imágenes** en `test_config_image.py` arma una corrida con `kind: images` + U-Net mínima **y las palancas de eficiencia activas por config** (`train.amp: true`, `data.num_workers`/`pin_memory`) y corre `build_run → train → resume` unos pasos en CPU: prueba que el camino completo no rompe y que el `scaler_state` viaja en el snapshot de reanudación.

`diffusion-models/tests/test_resume.py` (61 tests, en verde) cubre el subsistema de **reanudación**: el round-trip del sidecar y su fail-fast (sin duplicar el `history`, sin tocar el checkpoint de pesos), `train(resume=…)` corriendo solo los pasos restantes y el no-op si ya está completo, el contrato del callback (`TrainSnapshot`), `discover_snapshots` (orden y exclusiones), `resolve_resume` en sus cuatro caminos (`skip`/`force`/auto-resume/`fresh`) y `--resume-from` por paso y por ruta con sus errores accionables, `validate_compatible` (SDE / `data_dim` / receta), `load_resume` (happy path, `history` del `meta`, sidecar faltante, incompatibilidad), el wiring del CLI de punta a punta (`scripts/train.py` cargado como módulo), y las claves de EMA del sidecar (crudos + sombra van y vuelven; sin EMA no aparece ninguna clave nueva; `load_resume` devuelve los crudos del sidecar cuando están y se comporta como siempre cuando no). Con **AMP** (`gpu-training-efficiency`) suma: round-trip del `scaler_state` en el sidecar (presente aun en CPU: `{}` ≠ `None`), retrocompat de sidecars sin la clave, los dos guards cruzados config↔sidecar (AMP pedido sin escalador guardado, y escalador presente sin AMP pedido), y la fidelidad de una corrida interrumpida+reanudada con `amp=True` en CPU.

> La convergencia solo se asserta para VP (el smoke de aprendizaje); de las demás variantes se testea la mecánica (finitud, shapes, reproducibilidad).

> **La fidelidad del resume está testeada**, con y sin EMA: hay un round-trip que compara una corrida **interrumpida y reanudada** contra la **ininterrumpida** por igualdad exacta (`torch.equal` tensor a tensor + `history` idéntico), y un test de sabotaje que confirma que sin restaurar el optimizador el resultado **difiere** (o sea que la igualdad no es trivial). La versión con EMA recorre el camino real del CLI (`save_checkpoint` + `save_resume_state` → `load_resume`) y exige coincidencia de las **tres** cosas: pesos crudos, sombra y checkpoint publicado.
