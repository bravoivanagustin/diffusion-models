# Módulo `sde` — los procesos forward (Eje 1)

Ubicación: `diffusion-models/src/diffusion/sde/`. Implementa los **procesos forward** `dx = f(x,t) dt + g(t) dW` que destruyen los datos hacia ruido y, sobre todo, el **target del score** `∇_x log p_t(x_t|x_0)` que la red (`ScoreMLP`) debe aprender. Es el **Eje 1** del estudio de ablación de `ejes.md`: tres variantes —**VP**, **VE** y **sub-VP**—, cada una con su `p_t` y por lo tanto su propio entrenamiento.

---

## 1. Objetivo del módulo y cómo se acopla al pipeline

`data_generation` produce los `x_0`; la **red** (`mlp`) aprende a mapear `(x_t, t) -> score`; pero falta lo que conecta ambos: el proceso que ruidea `x_0` para fabricar el par `(x_t, target)` y que define **qué** score debe aprender la red. **Este módulo es ese proceso.** Dónde se ubica en el flujo (la caja del medio es este módulo):

```
 data_generation          ►  FORWARD SDE  ◄          red                 sampler reverso
     x_0       ──►  x_t = α_t·x_0 + σ_t·ε   ──►   s_θ(x_t, t)    ──►   integra la SDE/ODE inversa
                    (añade ruido; fija el       (módulo mlp)         (genera muestras nuevas)
                     target del score)
```

- **Entrada / salida**: `perturb(x0, t)` toma `x0` de shape `(B, *E)` (con `E` la forma de evento: `(2,)` en 2D, `(C,H,W)` en imágenes) y `t` de shape `(B,)` o `(B, 1)`, samplea el kernel de perturbación y devuelve `(x_t, ε)`; `score_target(x0, t, ε)` devuelve el score real del kernel (el target de la red) y un peso. `sde(x, t)` da el `(drift, diffusion)` y `prior_sampling(shape)` muestrea el ruido inicial `p_T` (shape `(n, *E)`) que los samplers integran hacia atrás.
- **Rol en la ablación**: la SDE es el Eje 1, **variable independiente**. La red queda fija; variar la SDE cambia `p_t` y el score que se aprende.
- **Regla de reentrenamiento**: cambiar el **forward SDE (Eje 1)** define un `p_t` distinto → **un entrenamiento por variante**. Cambiar el **sampler (Eje 2)** reusa el mismo score, sin reentrenar. (Ver `ejes.md`.)

A diferencia de `data_generation` —cuyo core es numpy con torch diferido—, este módulo **importa torch directamente** (como `mlp`): opera sobre tensores, fabrica los pares de entrenamiento y alimenta los samplers, así que torch es dependencia dura.

---

## 2. Funcionamiento y arquitectura

### Archivos

```
src/diffusion/sde/
  base.py        # ForwardSDE (ABC) + perturb/score_target concretos (familia escalar)
  schedules.py   # funciones puras: β(t) lineal + su integral, σ(t) geométrico
  variants.py    # VPSDE, VESDE, SubVPSDE (familia escalar-gaussiana)
  __init__.py    # REGISTRY + make_sde + available_sdes
  __main__.py    # smoke (python -m diffusion.sde)
tests/test_sde.py   # suite de pytest (69 tests)
```

`schedules.py` aísla la matemática pura (la integral de `β` es idéntica en VP y sub-VP) y es la superficie de mayor valor para tests numéricos.

### La clase base `ForwardSDE`

ABC con:

- **Atributo de clase** `name` (clave del registry).
- **Constructor** `__init__(data_dim: int | tuple[int, ...] = 2, T: float = 1.0)`. Guarda `self.T`, el **valor crudo** `self.data_dim` (int o tupla, tal cual se pasó — lo consume el checkpoint) y la **forma de evento normalizada** `self.data_shape` (siempre una tupla, vía `_normalize_shape`). Constante de clase `_std_eps = 1e-5` (piso del `std` en `score_target`).
- **Tres métodos abstractos** por variante: `sde(x,t) -> (drift, diffusion)`, `marginal_prob(x0,t) -> (mean, std)` y `prior_sampling(shape)`.

**Generalización a N-D (event shapes).** El módulo opera sobre `x` de shape `(B, *E)` para **cualquier** forma de evento `E`: `E=(2,)` para el toy 2D, `E=(C,H,W)` para imágenes. `data_dim` acepta un **entero** (dim plana) o una **tupla** (forma de evento). El truco que lo hace posible sin ramificar es el staticmethod **`_expand_t(t, ref)`**, que reescala `t` a `(B, 1, …, 1)` con tantos unos como el **rango** de `x` (rango 2 → `(B,1)`, byte-idéntico al código pre-generalización; rango 4 → `(B,1,1,1)`), de modo que `std`/`weight`/coeficientes broadcastean solos contra `x`. `_normalize_shape` valida la forma: rechaza tupla vacía y dimensiones `< 1` (`ValueError`).

Para la **familia escalar-gaussiana** el kernel es `p_t(x_t|x_0) = N(mean, std² I)` con `std` escalar por muestra, así que `perturb` y `score_target` son **concretos en la base** y se derivan enteramente de `marginal_prob`:

- `perturb(x0, t, *, generator=None)`: `x_t = mean + std·ε`, con `ε ~ N(0, I)` (mismo device/dtype que `x0`); devuelve `(x_t, ε)`, ambos `(B, *E)`.
- `score_target(x0, t, ε)`: `∇_{x_t} log p_t(x_t|x_0) = -ε/σ_t`, con peso `λ(t) = σ_t²` (pesado tipo verosimilitud, que vuelve la pérdida equivalente a `‖σ_t·s_θ + ε‖²`). Sutileza: `std` se **clampea** a `_std_eps=1e-5` **antes** de dividir y de elevar al cuadrado, así que en `t→0` el target y el peso usan `max(σ_t, 1e-5)`, no `σ_t` literal. `marginal_prob`, en cambio, devuelve el `std` **crudo** (VP/sub-VP dan exactamente 0 en `t=0`): quien divida por ese `std` fuera de `score_target` debe poner su propio piso — por eso `t_eps` vive upstream (en el muestreo de `t`), no acá.

### Las tres SDEs (`t ∈ [0, T]`, `T=1`)

Familia escalar (con `β(t) = β_min + t(β_max−β_min)` e `∫₀ᵗβ = β_min·t + ½(β_max−β_min)t²`):

| SDE | drift `f(x,t)` | `diffusion g(t)` | mean del kernel | std del kernel | prior `p_T` |
|---|---|---|---|---|---|
| **VP** | `-½β(t)x` | `√β(t)` | `α_t·x_0`, `α_t=e^{-½∫β}` | `√(1−e^{-∫β})` | `N(0, I)` |
| **VE** | `0` | `σ(t)√(2 ln(σ_max/σ_min))` | `x_0` | `σ(t)=σ_min(σ_max/σ_min)^t` | `N(0, σ_max² I)` |
| **sub-VP** | `-½β(t)x` | `√(β(t)(1−e^{-2∫β}))` | `α_t·x_0` (igual que VP) | `1−e^{-∫β}` (sin raíz) | `N(0, I)` |

VP/sub-VP comparten media (mismo `α_t`); el desvío de sub-VP es estrictamente menor que el de VP (la varianza queda *por debajo*, de ahí "sub-VP").

Hiperparámetros por constructor (sin números mágicos):

| SDE | Constructor (defaults) |
|---|---|
| `VPSDE` | `beta_min=0.1, beta_max=20.0, data_dim=2, T=1.0` |
| `VESDE` | `sigma_min=0.01, sigma_max=5.0, data_dim=2, T=1.0` (`sigma_max` ≈ escala del toy 2D, **no** el 50 de imágenes) |
| `SubVPSDE` | `beta_min=0.1, beta_max=20.0, data_dim=2, T=1.0` |

`data_dim` es configurable en todas y acepta **entero o tupla**: la familia escalar broadcastea el `std` escalar contra cualquier forma de evento (vía `_expand_t` rank-aware). Así el módulo escala a la Fase 2 (imágenes, `data_dim=(3,H,W)`) sin tocar el código y sin regresión en 2D.

> **Footgun cross-módulo — los hiperparámetros de la SDE no viajan en el checkpoint.** `save_checkpoint` guarda solo `sde_name` y `data_dim`, **no** `beta_min/beta_max/sigma_min/sigma_max/T`. Al generar, `generate_from_checkpoint` reconstruye con `make_sde(sde_name, data_dim=...)` → **defaults del constructor**. Si entrenaste con betas o `sigma_max` no-default, el sampleo usa silenciosamente otros valores. Ver `dataflow.md` (footgun #1). Además el `sigma_max=5.0` de VE está afinado al toy 2D estandarizado, no al `50` de imágenes: hay que ajustarlo por dato (y hoy se perdería al samplear).

### Dónde SÍ vive la estocasticidad (lo central para el TP)

Acá vive una de las tres fuentes de aleatoriedad del pipeline: `perturb` saca un `t` y un ruido `ε ~ N(0, I)` para fabricar `x_t`, y la SDE define la densidad ruidosa `p_t`. Es el reflejo de la red, que es **determinística**: toda la estocasticidad vive *afuera* de ella —en el dato (`data_generation`), **acá** en el forward, y en el sampler reverso— y **nunca** dentro de la red (ver `mlp.md`). Por eso `perturb`/`prior_sampling` aceptan un `torch.Generator` opcional: la aleatoriedad es explícita y reproducible.

### El target del score (DSM) — y la línea con el módulo de entrenamiento

`score_target` *es* "el target del score" que pide el diseño: el score real del kernel hacia el que se empuja `s_θ`. La pérdida de **denoising score matching** —que combina este target con la salida de la red— y todo el **loop de entrenamiento** (optimizer, muestreo de `t`, checkpoints, resume) viven en el módulo `diffusion.training` (ya implementado; ver `training.md`), no acá: este módulo se queda en las primitivas matemáticas de la SDE (`sde`, `marginal_prob`, `prior_sampling`, `perturb`, `score_target`).

### Tests (`tests/test_sde.py`)

69 tests (pytest, con `importorskip("torch")`): registry/factory (round-trip de tipos, nombre desconocido → `ValueError`, filtrado de kwargs por firma); shapes/dtype `float32` de `marginal_prob`/`perturb`/`sde`; `t` aceptado como `(B,)` y `(B,1)`; determinismo de `perturb` con `Generator` seedeado; **límites del kernel** (VP: `t→0` ⇒ `mean≈x0, std≈0`; `t=T` ⇒ `mean→0, std→1`; VE sin drift y `std(T)≈σ_max`; sub-VP `std < VP` con misma media); **chequeo de cálculo** (`dΣ/dt` por diferencias finitas consistente con `2 f_coef Σ + g²`); `score_target` (`-ε/σ_t`, peso `σ_t²`, signo opuesto a `ε`); varianza del prior; **seam `sde × mlp`** (la salida de `ScoreMLP` calza con `score_target`). Y el bloque **N-D (event shapes)**: la familia escalar en dims 1/3/7, `_expand_t` rank-aware (`(B,1)` en 2D vs `(B,1,1,1)` en `(3,8,8)`), invariancia 2D byte-idéntica tras la generalización, `data_shape` normalizada, y validación de formas inválidas (tupla vacía, dims `< 1`). Correr:

```
python -m pytest -q                                  # toda la suite
python -m pytest -q diffusion-models/tests/test_sde.py   # solo este módulo
```

### Ejemplo de uso (API)

```python
from diffusion.sde import make_sde
from diffusion.models import ScoreMLP

sde = make_sde("vp")                       # "ve" / "sub_vp"
net = ScoreMLP(data_dim=sde.data_dim)      # cualquier dim; 2 por defecto

x_t, eps = sde.perturb(x0, t)              # par de entrenamiento (ruido explícito)
target, weight = sde.score_target(x0, t, eps)   # hacia esto se empuja net(x_t, t)
```

El módulo trae un smoke (`__main__.py`): corre `perturb` sobre las 3 SDEs e imprime media/escala del kernel en `t≈0` y `t=T`. Correr (desde `diffusion-models/src/`): `python -m diffusion.sde`.

---

## 3. Aguas abajo: entrenamiento y samplers (ya implementados)

Con la SDE entregando `(x_t, target)`, el **loop de entrenamiento** (`diffusion.training`) minimiza el denoising score matching `‖σ_t·s_θ(x_t,t) + ε‖²` —un entrenamiento por variante del Eje 1— y los **samplers** (`diffusion.samplers`: Euler–Maruyama, PF-ODE, Heun, predictor–corrector) consumen `sde.sde(x,t)` (drift/diffusion), `sde.prior_sampling` y el `s_θ` ya entrenado para integrar la ecuación reversa (Eje 2). Ambos módulos ya están entregados — ver `training.md`, `samplers.md` y el flujo completo en `dataflow.md`. El diseño experimental está en `ejes.md`.
