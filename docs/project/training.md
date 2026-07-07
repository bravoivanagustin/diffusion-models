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
| `TrainConfig` | Dataclass **acotado al loop**: `num_steps, lr, t_eps, grad_clip, seed, device, log_every`. Ya no lleva hiperparámetros de red (van al constructor / `make_model`) ni de dataset (`n_samples`/`batch_size` van a la fuente de datos). |
| `TrainResult` | `net` entrenada (cualquier `ScoreModel`), `history` (pérdida media por intervalo de registro), `config`, `sde_name` y `data_dim` (`= sde.data_dim`, lo copia el checkpoint). |
| `train(sde, model, data, config, *, generator=None)` | Corre el loop **por pasos** (`num_steps`) y devuelve `TrainResult`. Recibe la red ya construida y un iterador infinito de datos; no instancia la red ni ramifica por su tipo (agnóstico a la red y al origen de datos). |
| `save_checkpoint(result, path, *, model_spec=None)` / `load_checkpoint(path)` | Persistencia **model-agnóstica** (R5-c): guarda `state_dict` + `meta{sde_name, data_dim, history, model?}`; `load_checkpoint` devuelve `(state_dict, meta)` sin reconstruir la red (ver más abajo). |

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

`diffusion-models/tests/test_training.py` (21 tests, en verde; suite completa sin regresiones):

- `dsm_loss` para las **3 SDEs**: escalar finito, diferenciable, gradientes finitos en la red; reproducible con `generator`.
- `sample_timesteps`: shape, rango $[t_\text{eps}, T]$, reproducibilidad, horizonte $T$ distinto.
- `train`: **usa la red recibida** y registra el `data_dim` correcto por SDE (las 3 variantes); `history` no vacío con `log_every=0`; la **pérdida baja** en VP sobre la mezcla de gaussianas; reproducibilidad con misma `seed`; camino `grad_clip`.
- `TrainConfig`: acotado al loop (no expone campos de red ni de dataset).
- `save_checkpoint`/`load_checkpoint`: ida y vuelta reconstruye la red con los mismos pesos; sin `model_spec` el checkpoint omite la receta `model`.
- `build_run`/`load_config`: ensamblado desde `dict` y desde YAML; el bloque `model:` sobreescribe el default; falla ante claves faltantes (`sde.name`, `data.shape`) o desconocidas en `train:`.

> La convergencia solo se asserta para VP (el smoke de aprendizaje); de las demás variantes se testea la mecánica (finitud, shapes, reproducibilidad).
