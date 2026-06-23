# Módulo `data_generation` — generación de datasets de puntos

Ubicación: `diffusion-models/src/diffusion/data_generation/`. Genera datasets de **puntos** en `R^dim` con distintas **formas** (toy data), pensados para iterar rápido en CPU mientras se estudian las partes estocásticas del pipeline de difusión.

---

## 1. Objetivo del módulo y cómo se acopla al pipeline

En un modelo de difusión la red nunca ve "la imagen" directamente: aprende un **score** `s_θ(x,t) ≈ ∇_x log p_t(x)`, donde `p_t` es la distribución de datos después de un tiempo `t` de ruido. Para eso, lo primero que hace falta es **la distribución de datos misma**, `p_data(x)` — los `x_0` que el proceso *forward* va a destruir hacia ruido. **Este módulo es esa fuente de `x_0`.**

Trabajamos con **puntos** (todavía no imágenes) porque son la pista barata de `ejes.md`: permiten correr la matriz de experimentos (4 SDEs × 4 samplers) en CPU, en segundos, antes de escalar a imágenes. Una "forma" (two moons, espiral, …) es una `p_data` 2D conocida y visualizable, lo que hace fácil ver a ojo si el sampler reverso reconstruye bien la distribución.

**Cómo fluye el dato:**

```
 data_generation         forward SDE                  red                 sampler reverso
     x_0       ──►  x_t = α_t·x_0 + σ_t·ε   ──►   s_θ(x_t, t)    ──►   integra la SDE/ODE inversa
 (este módulo)      (añade ruido; define target)   (aprende score)      (genera muestras nuevas)
```

- **Salida**: `np.ndarray` `float32` de shape `(n, dim)`, o un `torch.Tensor`, o un `DataLoader` listo para el loop de entrenamiento.
- **Rol en la ablación**: el dataset es **variable de control** — se mantiene fijo (igual que la red) mientras se varían el forward SDE y el sampler, para que toda diferencia medible se atribuya a la matemática y no al dato. (Ver `ejes.md`.)

---

## 2. Funcionamiento y arquitectura

### Archivos

```
src/diffusion/data_generation/
  base.py        # PointDistribution (clase base abstracta)
  shapes.py      # las 5 formas concretas
  __init__.py    # REGISTRY + make_distribution() + available_shapes()
scripts/data_generation.py     # CLI: genera, guarda .npz y un PNG de preview
tests/test_data_generation.py  # suite de pytest
```

### `PointDistribution` — el contrato común (`base.py`)

Toda forma hereda de `PointDistribution` e implementa un único método, `_sample_raw`. La clase base se encarga del resto:

- **`sample(n) -> np.ndarray (n, dim) float32`** — el método público. Internamente:
  1. crea `rng = np.random.default_rng(seed)` (reproducible),
  2. llama `_sample_raw(n, rng)` (geometría cruda, float64),
  3. valida la shape,
  4. si `standardize=True`, centra y escala a media 0 / std 1 por columna (y guarda `mean_`/`std_`),
  5. castea a `float32`.
- **`sample_torch(n)`** y **`dataloader(n, batch_size, shuffle=True)`** — helpers para entrenamiento. **Importan torch de forma diferida (lazy)**: el módulo se usa y se testea sin torch instalado; solo lo pide cuando llamás estos helpers.
- **Atributos** tras `sample`: `dim`, `standardize`, `noise`, `seed`, `mean_`/`std_` (para des-estandarizar después), `color_` (etiqueta por punto, para colorear el preview).
- **Validación de dimensión (esquema híbrido)**: cada forma declara `supported_dims` (`None` = cualquier `dim ≥ 1`, o un `frozenset` como `{2}`). Pedir una dim no soportada produce un `ValueError` claro. Así conviven formas N-dim con formas solo-2D/3D.

### Las 5 formas (`shapes.py`)

| Forma | `name` | dims | backend | params propios |
|---|---|---|---|---|
| Gaussiana isotrópica | `gaussian` | cualquiera (≥1) | numpy | `scale` |
| Mezcla de gaussianas | `mixture` | cualquiera (≥1) | sklearn `make_blobs` | `n_components`, `cluster_std`, `radius` |
| Two moons | `two_moons` | solo 2 | sklearn `make_moons` | `noise` |
| Espiral(es) | `spiral` | solo 2 | numpy | `noise`, `n_arms`, `turns` |
| Swiss roll | `swiss_roll` | solo 3 | sklearn `make_swiss_roll` | `noise` |

`mixture` en 2D ubica los centros en un anillo (las clásicas "8 gaussianas"); en N-dim, en direcciones aleatorias sobre una hiperesfera de radio `radius`.

### Registry y factory (`__init__.py`)

- `REGISTRY: dict[name -> clase]` con las 5 formas.
- `available_shapes() -> list[str]` — los nombres válidos (ordenados); el CLI los usa como `choices`.
- `make_distribution(name, dim, **kwargs) -> PointDistribution` — crea la forma por nombre y **filtra los `kwargs`** según la firma del constructor de cada forma. Así un caller genérico (el CLI) puede pasar siempre el mismo set de flags y cada forma toma solo los que entiende.

### Dónde vive la estocasticidad (lo central para el TP)

Toda la aleatoriedad está **localizada y es reproducible**:

- Un `rng` por llamada a `sample`, sembrado con `seed` → misma `seed` ⇒ mismos puntos.
- Las formas que usan sklearn derivan su `random_state` del mismo `rng` (`_seed_from`), así la reproducibilidad es consistente con o sin sklearn de por medio.
- La red (próximo módulo) será **determinística**: la estocasticidad vive acá (en el dato) y en el forward/sampler, **nunca dentro de la red**.

### CLI (`scripts/data_generation.py`)

```
python scripts/data_generation.py --shape two_moons --dim 2 --n-samples 2000 \
    --seed 0 --out data/two_moons.npz --preview data/two_moons.png
```

Flags: `--shape` (req.), `--dim`, `--n-samples`, `--seed`, `--noise`, `--n-components` (mixture), `--standardize`, `--out` (`.npz`), `--preview` (`.png`). Los errores de dim salen limpios (exit 2) y la salida fuerza UTF-8 para no romper acentos en Windows.

- **`.npz`**: clave `X` (`float32`), `meta` (JSON con shape/dim/n/seed/standardize/noise) y `color`/`mean`/`std` cuando aplican. Se usa `.npz` para meter varios arrays + metadata en un solo archivo, sin pickle (seguro y portable). Cargar:
  ```python
  import numpy as np, json
  d = np.load("data/two_moons.npz"); X = d["X"]; meta = json.loads(str(d["meta"]))
  ```
- **Preview PNG**: scatter directo si `dim ≤ 2`; **PCA→2D** si `dim > 2`; coloreado por `color_`.
- **Convención de nombres**: archivos como `data/gaussian/gaussian_2_1000_1.npz` (`shape_dim_n_seed[_comp]`) son una convención **manual** al pasar `--out` — el script no la genera por su cuenta.

> Los datasets toy son **reproducibles desde la `seed`** en milisegundos, así que persistirlos es comodidad (compartir un dataset exacto, desacoplar generación de entrenamiento), no una necesidad: en entrenamiento se puede llamar la librería directo.

### Tests (`tests/test_data_generation.py`)

22 tests (pytest): shape/dtype/finitud por forma, validación de dims (errores esperados), reproducibilidad por seed, registry/factory, estandarización (media≈0 / std≈1), helpers torch (`importorskip`) y un smoke end-to-end del CLI. Correr:

```
cd diffusion-models
python -m pytest -q
```

### Ejemplo de uso (API)

```python
from diffusion.data_generation import make_distribution

dist = make_distribution("mixture", dim=2, n_components=8, seed=0)
X = dist.sample(2000)                              # np.ndarray (2000, 2) float32
loader = dist.dataloader(2000, batch_size=128)     # torch DataLoader (import lazy)
```

---

## 3. Siguiente módulo: redes neuronales

El próximo paso es la **red que aprende el score** `s_θ(x,t) ≈ ∇_x log p_t(x)` (equivalente, según la parametrización, a predecir el ruido `ε` o el `x_0`). Para **datos de puntos la red es un MLP** —chico, con un embedding del tiempo `t`—; la **U-Net** entra recién cuando pasemos a imágenes.

**Cómo se acopla con este módulo:**

- `data_generation` entrega los `x_0` (vía `sample` / `sample_torch` / `dataloader`).
- El **forward SDE** samplea un `t` y ruido `ε` para construir el par de entrenamiento `(x_t, target)`.
- La red mapea `(x_t, t) -> score/ε`. Es **determinística y fija** (variable de control): toda la estocasticidad queda afuera, en el dato y en el forward/sampler.

**Interfaz esperada para que enganche sin fricción:**

- Mantener la salida en `float32` y, para entrenar, usar `standardize=True` (escala sana ≈ N(0,1)), guardando `mean_`/`std_` para des-estandarizar las muestras generadas.
- El `DataLoader` de este módulo alimenta directamente el loop de entrenamiento.

**Próximos archivos** (a definir por el autor, sin implementar todavía): por ejemplo `models/` (el MLP con embedding de `t`), y después `sde/` (VP/VE/sub-VP/CLD) + el loop de entrenamiento. El diseño completo está en `ejes.md`.
