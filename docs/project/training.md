# Módulo `training` — el loop de entrenamiento (denoising score matching)

Cuarto módulo de código del TP. Es el **eslabón que une** las tres piezas ya entregadas: `data_generation` (los datos limpios $x_0$), `mlp` (la red de score $s_\theta$) y `sde` (el proceso forward que define el target). Su trabajo es **entrenar** a `ScoreMLP` para que aproxime el score $s_\theta(x,t) \approx \nabla_x \log p_t(x)$ de una SDE dada, minimizando la pérdida de **denoising score matching (DSM)**.

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

El mismo `train()` corre las cuatro SDEs **sin ramificar** por tipo. El batch de puntos 2D que entrega `data_generation` se pasa **crudo** a `sde.perturb`/`sde.score_target`, que ya devuelven las shapes correctas:

- **VP/VE/sub-VP:** estado escalar, red `data_dim=2`. $x_0$ es el punto $(x, y)$.
- **CLD:** estado aumentado posición–momento, red `data_dim=4`. $x_0$ es la **posición** 2D; la SDE agrega el momento internamente y `perturb` devuelve el estado completo $u_t=(x_t, v_t)$.

La red se instancia con `ScoreMLP(data_dim=sde.data_dim)` — 2 o 4 según la variante — y todo lo demás es idéntico. Es la materialización en código de "la red es la variable de control": misma arquitectura, mismos hiperparámetros, solo cambia la SDE.

## Regla del Eje 1: un entrenamiento por variante

Cambiar el forward SDE cambia $p_t(x)$ y por lo tanto el score a aprender → **hay que reentrenar**. Cada llamada a `train()` instancia una red nueva desde cero, así que entrenar VP, VE, sub-VP y CLD son cuatro corridas independientes. Los samplers del **Eje 2** (a futuro) reusan la red entrenada **sin** reentrenar. Por eso conviene una corrida = un archivo de config versionable (ver abajo), una por celda del estudio.

## Estado de CLD ⚠️

El loop soporta CLD **mecánicamente** (el seam funciona: `data_dim=4`, pérdida finita, checkpoint ida y vuelta — todo testeado), **pero CLD todavía no converge.** Su `score_target` devuelve `weight = 1` a propósito: el módulo `sde` **difiere** el pesado de HSM (Hybrid Score Matching) al loop de entrenamiento. Sin ese peso, el target del score del momento $-n_v/L_{22}$ explota cuando $t \to 0$ (porque $L_{22} \to 0$) y la pérdida queda dominada por los tiempos chicos, sin converger.

Esto es exactamente lo que `ejes.md`, `sde.md` y `to-do.md` marcan como pendiente ("el pesado de HSM para CLD"). **No se implementó acá** porque la fórmula del peso es una elección de modelado que decide el autor (ver `to-do.md`). Hasta entonces, las celdas de CLD del estudio quedan a la espera; VP/VE/sub-VP —el núcleo de la Fase 1— entrenan y convergen.

## API

Núcleo (en `losses.py`, sin estado ni I/O — se testea directo):

| Función | Qué hace |
|---|---|
| `dsm_loss(net, sde, x0, t, *, generator=None)` | Pérdida DSM de un batch; escalar diferenciable. |
| `sample_timesteps(n, T, t_eps, *, generator=None, device=None)` | $n$ tiempos $\sim\mathcal{U}[t_\text{eps},T]$, shape `(n,)`. |

Loop y persistencia (en `trainer.py`):

| Símbolo | Qué es |
|---|---|
| `TrainConfig` | Dataclass: `epochs, batch_size, n_samples, lr, t_eps, grad_clip, seed, device, log_every` + hiperparámetros de red (`embed_dim, hidden_dim, num_blocks, activation`). |
| `TrainResult` | `net` entrenada, `history` (pérdida media por época), `config`, `sde_name`. |
| `train(sde, distribution, config, *, generator=None)` | Corre el loop; devuelve `TrainResult`. Instancia una red nueva (un entrenamiento por variante). |
| `save_checkpoint(result, path)` / `load_checkpoint(path)` | Guarda/recarga la red + metadata (SDE, `data_dim`, hiperparámetros) en un `.pt`. |

Config-driven (en `config.py`):

| Símbolo | Qué es |
|---|---|
| `load_config(path)` | Lee un YAML a `dict` (necesita `pyyaml`). |
| `build_run(raw)` | Ensambla un `RunSpec(sde, distribution, config, checkpoint, loss_curve)` reusando `make_sde`/`make_distribution`. |
| `RunSpec` | Una corrida lista: SDE + datos + `TrainConfig` + rutas de salida. |

## Corridas por config (YAML)

Cada celda del estudio se describe en un `.yaml`. El core no sabe de archivos: `config.py` es un front-end fino que arma `(sde, distribution, TrainConfig)`. Estructura:

```yaml
sde:                 # -> make_sde(name, **resto)
  name: vp           # vp | ve | sub_vp | cld
  beta_min: 0.1
  beta_max: 20.0
data:                # -> make_distribution(shape, dim, **resto); n_samples va al TrainConfig
  shape: mixture
  dim: 2
  n_samples: 4000
  n_components: 8
  standardize: true
  seed: 0
train:               # -> campos de TrainConfig
  epochs: 300
  batch_size: 256
  lr: 0.002
  t_eps: 1.0e-3
  grad_clip: 1.0     # opcional
  seed: 0
  device: cpu
# model:             # opcional: la red es la variable de control (normalmente fija)
#   hidden_dim: 256
out:                 # rutas relativas al cwd
  checkpoint: models/vp_mixture.pt
  loss_curve: models/vp_mixture_loss.png
```

Ejemplos listos en `diffusion-models/config/`: `vp_mixture.yaml` y `cld_mixture.yaml`.

## Cómo correr

```bash
# Uso a mano (Python):
from diffusion.sde import make_sde
from diffusion.data_generation import make_distribution
from diffusion.training import TrainConfig, train, save_checkpoint

sde = make_sde("vp")
dist = make_distribution("mixture", dim=2, n_components=8, seed=0)
result = train(sde, dist, TrainConfig(epochs=300, n_samples=4000))
save_checkpoint(result, "models/vp_mixture.pt")

# CLI por config (desde diffusion-models/):
python scripts/train.py --config config/vp_mixture.yaml
python scripts/train.py --config config/vp_mixture.yaml --epochs 50 --device cpu

# Smoke del módulo (desde diffusion-models/src/):
python -m diffusion.training
```

El CLI guarda el checkpoint (`.pt`) y una curva de pérdida (`.png`) en las rutas de `out`.

## Stack y dependencias

Torch es **dependencia dura** del módulo (como `mlp` y `sde`): importa `torch` directo (no diferido). El front-end de config agrega **`pyyaml`** (>= 6) a las dependencias del proyecto.

## Tests

`diffusion-models/tests/test_training.py` (20 tests, en verde; suite completa sin regresiones):

- `dsm_loss` para las **4 SDEs**: escalar finito, diferenciable, gradientes finitos en la red (cubre el seam de CLD `data_dim=4`); reproducible con `generator`.
- `sample_timesteps`: shape, rango $[t_\text{eps}, T]$, reproducibilidad, horizonte $T$ distinto.
- `train`: `data_dim` correcto por SDE (2/4) y traza finita (las 4 variantes); la **pérdida baja** en VP sobre la mezcla de gaussianas; reproducibilidad con misma `seed`; camino `grad_clip`.
- `save_checkpoint`/`load_checkpoint`: ida y vuelta reconstruye la red (incl. CLD `data_dim=4`) con los mismos pesos.
- `build_run`/`load_config`: ensamblado desde `dict` y desde YAML; CLD arma `data_dim=4`; falla ante claves faltantes (`sde.name`, `data.shape`) o desconocidas.

> La convergencia solo se asserta para la familia escalar (VP); de CLD se testea la **mecánica** (finitud, shapes, checkpoint), no la convergencia — coherente con el estado ⚠️ de arriba.
