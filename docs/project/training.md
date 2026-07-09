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

> **El muestreo de $t$.** Se sortea uniforme en $[t_\text{eps}, T]$ con $t_\text{eps} > 0$ (default `1e-3`). El piso evita $t = 0$, donde $\sigma_t \to 0$ y el target $-\varepsilon/\sigma_t$ se vuelve inestable. $T = \texttt{sde.T}$.

## El seam agnóstico a la SDE

El mismo `train()` corre las tres SDEs **sin ramificar** por tipo. El batch de puntos 2D que entrega `data_generation` se pasa **crudo** a `sde.perturb`/`sde.score_target`, que ya devuelven las shapes correctas: estado escalar, red `data_dim=2`, $x_0$ es el punto $(x, y)$.

La red la construye el **caller** —`make_model("mlp", data_dim=sde.data_dim)` o una instancia explícita— y se la pasa a `train`, que no la instancia ni ramifica por su tipo (es agnóstico a la red y al origen de datos). Todo lo demás es idéntico. Es la materialización en código de "la red es la variable de control": misma arquitectura, mismos hiperparámetros, solo cambia la SDE.

## Regla del Eje 1: un entrenamiento por variante

Cambiar el forward SDE cambia $p_t(x)$ y por lo tanto el score a aprender → **hay que reentrenar**. Cada corrida construye una red nueva desde cero (el caller la instancia y se la pasa a `train`), así que entrenar VP, VE y sub-VP son tres corridas independientes. Los samplers del **Eje 2** reusan la red entrenada **sin** reentrenar. Por eso conviene una corrida = un archivo de config versionable (ver abajo), una por celda del estudio.

## API

Núcleo (en `losses.py`, sin estado ni I/O — se testea directo):

| Función | Qué hace |
|---|---|
| `dsm_loss(net, sde, x0, t, *, generator=None)` | Pérdida DSM de un batch; escalar diferenciable. |
| `sample_timesteps(n, T, t_eps, *, generator=None, device=None)` | $n$ tiempos $\sim\mathcal{U}[t_\text{eps},T]$, shape `(n,)`. |

Loop y persistencia (en `trainer.py`):

| Símbolo | Qué es |
|---|---|
| `TrainConfig` | Dataclass **acotado al loop**: `num_steps, lr, t_eps, grad_clip, seed, device, log_every, checkpoint_every`. Ya no lleva hiperparámetros de red (van al constructor / `make_model`) ni de dataset (`n_samples`/`batch_size` van a la fuente de datos). `log_every` es **solo para el print** de consola (media móvil), desacoplado del `history`. `checkpoint_every` (default `0`) activa el checkpointing intermedio (ver abajo). |
| `TrainResult` | `net` entrenada (cualquier `ScoreModel`), `history` (**serie per-step completa**: una entrada por paso, `len == num_steps`), `config`, `sde_name` y `data_dim` (`= sde.data_dim`, lo copia el checkpoint). |
| `train(sde, model, data, config, *, generator=None, on_checkpoint=None)` | Corre el loop **por pasos** (`num_steps`) y devuelve `TrainResult`. Recibe la red ya construida y un iterador infinito de datos; no instancia la red ni ramifica por su tipo (agnóstico a la red y al origen de datos). `on_checkpoint(tag, snapshot)` es un callback **opcional** de checkpointing intermedio: el loop decide *cuándo* llamarlo, el callback decide *cómo/dónde* persistir — `train` no toca el filesystem. |
| `save_checkpoint(result, path, *, model_spec=None)` / `load_checkpoint(path)` | Persistencia **model-agnóstica** (R5-c): guarda `state_dict` + `meta{sde_name, data_dim, history, model?}`; `load_checkpoint` devuelve `(state_dict, meta)` sin reconstruir la red (ver más abajo). |

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
# model:             # opcional: la red es la variable de control (normalmente fija)
#   name: mlp        #   si falta, se usa {name: mlp} dimensionado desde el dato/SDE
#   hidden_dim: 256
out:                 # rutas relativas al cwd
  checkpoint: models/vp_mixture.pt
  loss_curve: models/vp_mixture_loss.png
```

Ejemplo listo en `diffusion-models/config/`: `vp_mixture.yaml`.

## Checkpoint model-agnóstico (R5-c)

`save_checkpoint`/`load_checkpoint` no dependen de ninguna clase de red concreta. El `.pt` guarda el `state_dict` de la red y una `meta` mínima: `{sde_name, data_dim, history}` más —opcionalmente— una **receta genérica** `model = {"name": str, "kwargs": dict}`. Esa receta la aporta el caller vía `save_checkpoint(result, path, model_spec={"name": "mlp", "kwargs": {...}})`; en el camino config-driven la transporta el `RunSpec` (`spec.model_spec`) y `scripts/train.py` la pasa sola.

`load_checkpoint(path)` devuelve `(state_dict, meta)` **sin reconstruir** la red: es el caller quien arma la red y le carga los pesos. La reconstrucción canónica es `make_model(recipe["name"], **recipe["kwargs"])` seguida de `net.load_state_dict(state_dict)`, y la hace `diffusion.samplers.generate_from_checkpoint`, que cierra el pipeline forward→score→sampleo desde el checkpoint. Si el checkpoint se guardó **sin** receta (`model_spec=None`) sigue siendo válido, pero al generar hay que pasarle una red explícita (`model=`). Así el mismo checkpoint sirve al `ScoreMLP` (Fase 1) y a la `ScoreUNet` (Fase 2) sin que `training` importe ninguna red concreta.

## Checkpointing intermedio (opt-in)

Por defecto **solo se guarda el checkpoint final** (el `state_dict` tal como quedó en el último paso): el loop no persiste nada, y `save_checkpoint` lo escribe una vez terminado `train`. Para guardar también estados intermedios está el switch `checkpoint_every` de `TrainConfig`:

- `checkpoint_every = 0` (**default**) — nada intermedio; comportamiento idéntico al histórico (sin regresión).
- `checkpoint_every = N > 0` — además del final, el loop pide guardar:
  - un **snapshot periódico** cada `N` pasos (tag `step{N:05d}` → `…_stepNNNNN.pt`), con cadencia propia (se chequea cada paso, así `N` no tiene que ser múltiplo de nada). El **último** paso se omite porque ya lo cubre el checkpoint final.
  - un **best-so-far** (tag `best` → `…_best.pt`), que se reescribe cada vez que baja una **media de ventana** de la pérdida (cadencia interna `eval_every = num_steps//100`, suavizada — la de un paso suelto es muy ruidosa por el `t` aleatorio). *Nota:* aún así la pérdida DSM es ruidosa, así que "best" es orientativo; para comparar suele ser más útil un snapshot periódico.

El diseño mantiene `train` **sin I/O**: el loop decide **cuándo** (la política de cadencia y best) e invoca un callback `on_checkpoint(tag, snapshot)`; el **caller** decide **cómo/dónde** persistir. En el camino config-driven eso lo arma `scripts/train.py`, que deriva rutas hermanas del `out.checkpoint` con `Path.with_stem` (`vp_mixture.pt` → `vp_mixture_step00050.pt`, `vp_mixture_best.pt`) y reusa `save_checkpoint` (con el mismo `model_spec`, así cada snapshot es reconstruible igual que el final). A mano:

```python
base = pathlib.Path("models/vp_mixture.pt")
def on_checkpoint(tag, snap):
    save_checkpoint(snap, base.with_stem(f"{base.stem}_{tag}"), model_spec=spec.model_spec)
train(sde, model, data, TrainConfig(num_steps=1000, checkpoint_every=200), on_checkpoint=on_checkpoint)
```

Se activa por YAML (`train.checkpoint_every`) o con el override `--checkpoint-every` del CLI. Si `checkpoint_every > 0` pero no hay `out.checkpoint`, el CLI avisa y no escribe snapshots (no hay ruta base de dónde colgarlos).

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

# Smoke del módulo (desde diffusion-models/src/):
python -m diffusion.training
```

El CLI guarda el checkpoint (`.pt`) y una curva de pérdida (`.png`) en las rutas de `out`.

## Stack y dependencias

Torch es **dependencia dura** del módulo (como `mlp` y `sde`): importa `torch` directo (no diferido). El front-end de config agrega **`pyyaml`** (>= 6) a las dependencias del proyecto.

## Tests

`diffusion-models/tests/test_training.py` (33 tests, en verde; suite completa sin regresiones):

- `dsm_loss` para las **3 SDEs**: escalar finito, diferenciable, gradientes finitos en la red; reproducible con `generator`.
- `sample_timesteps`: shape, rango $[t_\text{eps}, T]$, reproducibilidad, horizonte $T$ distinto.
- `train`: **usa la red recibida** y registra el `data_dim` correcto por SDE (las 3 variantes); `history` es la **serie per-step completa** (`len == num_steps`), independiente de `log_every`; la **pérdida baja** (medias de bloque) en VP sobre la mezcla de gaussianas; reproducibilidad con misma `seed`; camino `grad_clip`.
- `TrainConfig`: acotado al loop (no expone campos de red ni de dataset); `checkpoint_every` arranca en `0`.
- `save_checkpoint`/`load_checkpoint`: ida y vuelta reconstruye la red con los mismos pesos; sin `model_spec` el checkpoint omite la receta `model`.
- **Checkpointing intermedio**: con `checkpoint_every=0` el callback no se invoca (sin regresión); con `N>0` emite los snapshots periódicos correctos (múltiplos de `N`, excluido el último paso) más al menos un `best`; el wiring estilo-CLI escribe los `…_stepNNNNN.pt`/`…_best.pt` cargables con la metadata esperada.
- `build_run`/`load_config`: ensamblado desde `dict` y desde YAML; el bloque `model:` sobreescribe el default; `train.checkpoint_every` se pasa al `TrainConfig`; falla ante claves faltantes (`sde.name`, `data.shape`) o desconocidas en `train:`.

> La convergencia solo se asserta para VP (el smoke de aprendizaje); de las demás variantes se testea la mecánica (finitud, shapes, reproducibilidad).
