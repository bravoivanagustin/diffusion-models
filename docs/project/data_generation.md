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

## 3. Fuente de datos de imágenes (Fase 2)

Además de las formas de puntos, este módulo aloja la **fuente de datos de imágenes** de la Fase 2: `diffusion.data_generation.images`, que convierte una carpeta de fotos de gatos (sin labels) en el mismo contrato que consume el `train` genérico — un iterador infinito de tensores crudos, pero de imágenes `(B, 3, image_size, image_size)` en `[-1, 1]` en lugar de puntos 2D. Su responsabilidad es única: llevar archivos en disco a un stream de batches; toda la lógica de SDE, pérdida, modelo y entrenamiento queda afuera.

### `infinite_batches` — el stream de batches (drop-in para `train`)

`infinite_batches(root, batch_size, *, image_size=64, augment=True, crop=True, num_workers=0, shuffle=True, seed=None, pin_memory=False)` devuelve un **iterador infinito**: cada `next()` entrega un **tensor pelado** (no una tupla) de shape `(batch_size, 3, image_size, image_size)` en `float32`, con valores en `[-1, 1]`. Es exactamente el contrato de `data` que definió `train-decoupling`, así que se enchufa directo al loop de entrenamiento sin que este ramifique ni sepa que detrás hay imágenes en vez de puntos.

- **Infinitud**: internamente arma un `torch.utils.data.DataLoader` finito y lo envuelve en un wrapper que reinicia el recorrido al agotarse (`while True: yield from loader`), de modo que nunca levanta `StopIteration`.
- **Batches exactos**: el loader usa `drop_last=True`, así **todos** los batches salen con exactamente `batch_size` elementos (se descarta un último batch incompleto).
- **Barajado reproducible**: con `shuffle=True` baraja en cada recorrido interno; con `seed` fijo (más `augment=False` y `num_workers=0`) la secuencia de batches es determinística, vía un `torch.Generator` sembrado.
- **Fail-fast**: los errores cortan **antes** de devolver el iterador (la función no es un generador). Si `root` no existe o no contiene imágenes → `ValueError`; si hay imágenes pero son menos que `batch_size` → `ValueError` explícito (con `drop_last=True` el loader quedaría vacío y el `while True` giraría sin yield-ear nunca: un cuelgue silencioso que se evita nombrando `n` y `batch_size`).

### La cadena de transforms (declarada, no escrita a mano)

El procesamiento de píxeles lo hace `torchvision.transforms` — el módulo solo **declara** la cadena PIL→tensor, no reimplementa resize/crop/IO/normalización. Cada imagen pasa por: convertir a **RGB de 3 canales** (`.convert("RGB")`, obligatorio: garantiza 3 canales aunque el original sea escala de grises, RGBA o CMYK) → **encuadre** → `ToTensor` (float32 en `[0, 1]`, canales primero) → `Normalize([0.5]*3, [0.5]*3)` (recentra a `[-1, 1]`).

- **Encuadre configurable (`crop`)**: con `crop=True` (por defecto) se preserva el aspect ratio — `Resize(image_size)` escala el lado corto y `CenterCrop(image_size)` recorta el centro al cuadrado; con `crop=False` se usa `Resize((image_size, image_size))`, que deforma la imagen al cuadrado sin recortar.
- **Augmentation (`augment`)**: con `augment=True` se antepone `RandomHorizontalFlip(p=0.5)` — volteo **solo horizontal**. La cadena **nunca** incluye volteos verticales ni rotaciones: un gato al revés no es una muestra válida de la distribución. Con `augment=False` no se aplica ningún volteo (salida sin augmentation, apta para validación/inspección).

> **Salida en `[-1, 1]` vs des-normalización**: este módulo entrega `[-1, 1]` (alineado con el prior del forward SDE). La des-normalización inversa `[-1, 1] → [0, 1]` para **visualizar** las muestras generadas **no** es responsabilidad de esta fuente: es una tarea de sampling/eval.

### `report_small_images` — higiene report-only

`report_small_images(root, *, min_size=64, verbose=False) -> list[Path]` recorre las imágenes descubiertas y devuelve las que tienen `min(width, height) < min_size` — el **lado corto**, que es el que el `Resize` del pipeline escalaría *hacia arriba* (upscale), metiendo imágenes borrosas al modelo. Es un chequeo de higiene **separado del flujo de carga** (no corre en cada batch ni lo dispara `infinite_batches`) y **report-only**: no borra ni modifica ningún archivo (solo lee las dimensiones con PIL, sin decodificar los píxeles) y devuelve la lista para que el autor decida qué descartar. **No** implementa detección de duplicados: ese dedup pixel-a-pixel es responsabilidad de `scripts/limpiar_imagenes.py`.

### Descubrimiento de archivos

`_discover_image_paths` recorre `root` recursivamente (`rglob`) y se queda con los archivos cuya extensión (en minúsculas) esté en `IMAGE_EXTENSIONS` (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`), en **orden determinístico** (`sorted`, reproducible entre corridas). Si `root` no existe o no contiene ninguna imagen, levanta `ValueError` (en lugar de un dataset vacío silencioso).

### Imports diferidos: el import del paquete sigue liviano

`infinite_batches` y `report_small_images` se re-exportan en `data_generation/__init__.py` (`__all__`), pero **`import diffusion.data_generation` no arrastra torchvision ni torch**: `images.py` mantiene el tope de módulo liviano (solo stdlib) y difiere `torch` / `torchvision` / `PIL` dentro de las funciones. La clase `CatImages` (el `Dataset` sin labels; **interna**, no se exporta) se resuelve de forma perezosa vía `__getattr__` (PEP 562), importando torch recién en el primer acceso. Es el mismo criterio que el torch diferido del core de puntos: el uso liviano (formas 2D, `make_distribution`) sigue funcionando aunque torchvision no esté instalado.

### Dependencia nueva y cómo correr el smoke

- **Dependencia**: `torchvision==0.27.0` (+ `pillow`), registrada en `tech.md`. El wheel `cp314-win_amd64` CPU fija `torch==2.12.0` y soporta Python 3.14.
- **Smoke (`__main__`)**: carga las imágenes reales de `data/cats-prueba/`, arma un batch y verifica con `assert` el contrato de salida (shape `(2, 3, 64, 64)`, `float32`, rango ~`[-1, 1]`), y después corre `report_small_images` sobre la misma carpeta. Se corre **como módulo** (por los imports diferidos):
  ```
  cd diffusion-models
  python -m diffusion.data_generation.images
  ```
- **Tests**: la suite `tests/test_image_data.py` verifica el contrato (shape, dtype, rango, infinitud, 3 canales) con **imágenes sintéticas autocontenidas** en un directorio temporal, sin depender de `data/cats-prueba/` (gitignored y local). Se omite (skip) si torchvision no está disponible, siguiendo la convención del repo.

### Ejemplo de uso (API)

```python
from diffusion.data_generation import infinite_batches, report_small_images

data = infinite_batches("data/cats-v1", batch_size=32, image_size=64, augment=True)
batch = next(data)                 # torch.Tensor (32, 3, 64, 64) float32 en [-1, 1]
# `data` es drop-in para el parámetro `data` de diffusion.training.train

small = report_small_images("data/cats-v1", min_size=64)   # rutas a revisar (no borra nada)
```

---

## 4. Siguiente módulo: redes neuronales

El próximo paso es la **red que aprende el score** `s_θ(x,t) ≈ ∇_x log p_t(x)` (equivalente, según la parametrización, a predecir el ruido `ε` o el `x_0`). Para **datos de puntos la red es un MLP** —chico, con un embedding del tiempo `t`—; la **U-Net** entra recién cuando pasemos a imágenes.

**Cómo se acopla con este módulo:**

- `data_generation` entrega los `x_0` (vía `sample` / `sample_torch` / `dataloader`).
- El **forward SDE** samplea un `t` y ruido `ε` para construir el par de entrenamiento `(x_t, target)`.
- La red mapea `(x_t, t) -> score/ε`. Es **determinística y fija** (variable de control): toda la estocasticidad queda afuera, en el dato y en el forward/sampler.

**Interfaz esperada para que enganche sin fricción:**

- Mantener la salida en `float32` y, para entrenar, usar `standardize=True` (escala sana ≈ N(0,1)), guardando `mean_`/`std_` para des-estandarizar las muestras generadas.
- El `DataLoader` de este módulo alimenta directamente el loop de entrenamiento.

**Próximos archivos** (a definir por el autor, sin implementar todavía): por ejemplo `models/` (el MLP con embedding de `t`), y después `sde/` (VP/VE/sub-VP) + el loop de entrenamiento. El diseño completo está en `ejes.md`.
