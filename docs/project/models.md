# Módulo `models` — las redes de score

Ubicación: `diffusion-models/src/diffusion/models/`. Implementa las **redes que aprenden el score** `s_θ(x,t) ≈ ∇_x log p_t(x)`: hoy el **MLP** para datos de **puntos** (toy data 2D); la **U-Net** para imágenes (Fase 2) se sumará como `unet.py`. La red es la **variable de control** del estudio de ablación.

> **Nota (05/07/2026):** este módulo nació como `diffusion/mlp/` (un solo archivo `score_mlp.py`) y se reestructuró al subpaquete `diffusion/models/` para hacerle lugar a la U-Net de Fase 2: las piezas compartidas entre redes se movieron a `layers.py`, el MLP quedó en `mlp.py` y el contrato `(x, t) -> score` se explicitó en `base.py`. El comportamiento es idéntico (misma arquitectura, mismos ~560 k parámetros); solo cambió el import: `from diffusion.models import ScoreMLP`.

---

## 1. Objetivo del módulo y cómo se acopla al pipeline

En un modelo de difusión la red nunca genera la muestra de un saque: aprende el **score** `∇_x log p_t(x)` —el campo que apunta hacia zonas más probables de `p_t`, la distribución de datos tras un tiempo `t` de ruido—. Con ese score, el sampler integra la **SDE reversa** (Anderson, 1982) y convierte ruido en una muestra nueva. **Este módulo es esa red**: mapea `(x_t, t) -> score`.

`data_generation` produce los `x_0`; el **forward SDE** los ruidea para fabricar el par de entrenamiento; **acá** vive la función que estima el score. Dónde se ubica en el flujo:

```
 data_generation         forward SDE                  red                 sampler reverso
     x_0       ──►  x_t = α_t·x_0 + σ_t·ε   ──►   s_θ(x_t, t)    ──►   integra la SDE/ODE inversa
                    (añade ruido; define target)  (este módulo)        (genera muestras nuevas)
```

- **Entrada / salida**: `forward(x, t)` toma `x` de shape `(B, data_dim)` y `t` de shape `(B,)` o `(B, 1)`, y devuelve el score de shape `(B, data_dim)` —misma dimensión que `x`, porque el score `∇_x log p_t(x)` vive en el mismo espacio que `x`—. Ese contrato común a todas las redes del subpaquete está explicitado en el Protocol `ScoreModel` (`base.py`).
- **Rol en la ablación**: la red es **variable de control**. Se mantiene **fija** (misma arquitectura, mismos hiperparámetros) en las 12 celdas de la matriz (3 SDEs × 4 samplers), para que toda diferencia medible se atribuya a la matemática (la SDE y el sampler) y **no** a la ingeniería de la red. (Ver `ejes.md`.)
- **Regla de reentrenamiento**: la red es agnóstica a la SDE, pero cada **forward SDE** (Eje 1) define un `p_t` distinto y por lo tanto un score distinto → **un entrenamiento por variante de SDE**. Cambiar el **sampler** (Eje 2) reusa el mismo score, sin reentrenar.

---

## 2. Funcionamiento y arquitectura

### Archivos

```
src/diffusion/models/
  layers.py      # piezas COMPARTIDAS entre redes: activaciones (_make_activation) + SinusoidalEmbedding
  mlp.py         # ResidualBlock (lineal) + ScoreMLP (+ smoke block __main__)
  base.py        # ScoreModel: el Protocol (x, t) -> score (tipado estructural)
  unet.py        # (Fase 2, pendiente) ScoreUNet convolucional para imágenes
  __init__.py    # re-exporta ScoreModel, SinusoidalEmbedding, ResidualBlock, ScoreMLP
tests/test_models.py   # suite de pytest
```

**La regla de admisión de `layers.py`**: solo entra lo que todas las redes usan **sin modificar**. Por eso el embedding de tiempo vive ahí (es literalmente el mismo en MLP y U-Net) pero el bloque residual **no**: el del MLP es lineal (`nn.Linear`) y el de la U-Net será convolucional con inyección de tiempo — comparten la idea (residual con skip), no el código. Cada red mantiene su propio bloque en su archivo.

Toda la red es **PyTorch puro** (`nn.Module`). A diferencia de `data_generation` —cuyo core es numpy y sólo importa torch de forma diferida—, este módulo **importa torch directamente**: es una red neuronal, torch es dependencia dura (`torch>=2.0`).

### Las piezas compartidas (`layers.py`)

**`SinusoidalEmbedding(embed_dim=128)`** — convierte el escalar de tiempo `t` en un vector, con la codificación de Transformers (senos y cosenos a distintas frecuencias):

```
embed(t)_{2i}   = sin(t / 10000^{2i/d})
embed(t)_{2i+1} = cos(t / 10000^{2i/d})       i = 0 … d/2-1,  d = embed_dim
```

Los denominadores `10000^{2i/d}` se **precomputan en `__init__`** y se guardan como **buffer** (`register_buffer`) —no son parámetros: no se aprenden, pero acompañan al módulo en `.to(device)`—. `embed_dim` debe ser **par** (cada frecuencia aporta un seno y un coseno); si no, `ValueError`. Funciona para **cualquier `t` flotante no negativo, sin supuestos sobre su escala**: el rango del tiempo depende de la SDE (`[0, 1]`, `[0, T]`, o pasos enteros), así que el embedding no asume ninguno.

`layers.py` también aloja el registry de activaciones (`_ACTIVATIONS` / `_make_activation`): `silu`, `relu`, `gelu`, `tanh`; nombre desconocido → `ValueError`.

### El MLP (`mlp.py`)

**`ResidualBlock(hidden_dim, activation="silu")`** — bloque MLP con conexión residual: `Linear → activación → Linear`, y luego se suma la entrada (skip **identidad**, sin proyección aprendida): `salida = bloque(x) + x`. Entrada y salida tienen la misma shape `(B, hidden_dim)`.

**`ScoreMLP(data_dim=2, embed_dim=128, hidden_dim=256, num_blocks=4, activation="silu")`** — la red completa. Embebe `t`, lo **concatena** con `x`, proyecta a `hidden_dim`, pasa por `num_blocks` bloques residuales y proyecta de vuelta a `data_dim`. La proyección final **no** lleva activación: el score es no acotado (puede ser positivo o negativo).

```
   x ∈ R^data_dim   t ∈ R
        │              │
        │   [SinusoidalEmbedding] → t_emb ∈ R^embed_dim
        └──► concat ◄──┘
               │
   [Linear(data_dim + embed_dim → hidden_dim)] + activación   (proyección de entrada)
               │
   [ResidualBlock] × num_blocks
               │
   [Linear(hidden_dim → data_dim)]                            (proyección de salida, sin activación)
               │
            score ∈ R^data_dim
```

Hiperparámetros (todos argumentos del constructor, sin números mágicos en el código):

| Param | Default | Qué controla |
|---|---|---|
| `data_dim` | `2` | dim del dato = dim de la salida. `2` para VP/VE/sub-VP `(x, y)`. |
| `embed_dim` | `128` | dim del embedding de tiempo (debe ser par). |
| `hidden_dim` | `256` | ancho de las capas ocultas en todos los bloques. |
| `num_blocks` | `4` | cantidad de bloques residuales. |
| `activation` | `"silu"` | activación; soporta `silu`, `relu`, `gelu`, `tanh` (nombre desconocido → `ValueError`). |

Con los defaults la red tiene **~560 k parámetros entrenables**.

### El contrato (`base.py`)

**`ScoreModel`** — un `typing.Protocol` (`runtime_checkable`) que documenta la firma común: callable `(x, t) -> score` con `score.shape == x.shape`. Es tipado **estructural**: ninguna red lo importa ni hereda de él —lo satisfacen por tener la firma correcta— y sirve para anotar código que recibe "una red de score cualquiera" (p. ej. un futuro `train(model: ScoreModel, ...)` agnóstico a la red).

### Dónde NO vive la estocasticidad (lo central para el TP)

La red es **enteramente determinística**: dado el mismo `(x, t)` produce **exactamente** la misma salida. **No hay dropout, ni batchnorm, ni ninguna capa estocástica** (hay un test que lo verifica recorriendo `.modules()`). Esto es deliberado y es lo que sostiene la ablación: toda la aleatoriedad del pipeline vive **afuera** de la red —en el dato (`data_generation`), en el muestreo de pares del forward SDE, y en el sampler reverso—, **nunca dentro de la red**. (El reflejo exacto del módulo anterior, donde la estocasticidad sí vivía en el dato; ver `data_generation.md`.)

### Por qué un MLP (y no una U-Net)

Para **puntos** la `p_data` es 2D y de baja dimensión: un MLP chico con embedding de `t` basta y corre en CPU en segundos, ideal para barrer la matriz 3×4 de `ejes.md`. La **U-Net** —con su estructura convolucional multi-escala— recién aporta cuando la entrada es una imagen. La interfaz `(x, t) -> score` se mantiene (es el Protocol `ScoreModel`); sólo cambia el cuerpo de la red.

### Tests (`tests/test_models.py`)

22 tests (pytest, con `importorskip("torch")`): `embed_dim` impar levanta `ValueError`; shapes del embedding y aceptación de `t` como `(B,)` y `(B, 1)` (mismo resultado); intercalado correcto sin/cos y valores acotados en `[-1, 1]`; `denom` es buffer y no parámetro; el `ResidualBlock` preserva shape y rechaza activaciones inválidas; `ScoreMLP` da salida `(B, 2)` y `(B, 4)`, **determinismo** (dos forwards iguales → salida idéntica), ausencia de dropout/batchnorm, conteo de parámetros > 0, y flujo de gradientes finito. Correr:

```
python -m pytest -q                                    # toda la suite
python -m pytest -q diffusion-models/tests/test_models.py   # solo este módulo
```

### Ejemplo de uso (API)

```python
from diffusion.models import ScoreMLP

net = ScoreMLP(data_dim=2)        # VP / VE / sub-VP
score = net(x, t)                 # x: (B, 2), t: (B,) o (B, 1)  ->  score: (B, 2)
```

El propio `mlp.py` trae un bloque `__main__` de smoke test: instancia la red por defecto, corre un forward sobre un batch dummy, imprime la shape de salida y el conteo de parámetros. Correr (con `diffusion-models/src` en `PYTHONPATH`): `python -m diffusion.models.mlp` — ya no funciona como script suelto (`python .../mlp.py`) porque el archivo usa imports relativos a `layers.py`.

---

## 3. Siguiente módulo: el forward SDE y el loop de entrenamiento

La red ya sabe mapear `(x_t, t) -> score`, pero **todavía no aprende nada**: falta el **target** que la entrene. Eso lo aporta el **forward SDE** (Eje 1):

- Dado un `x_0` de `data_generation`, samplea un tiempo `t` y ruido `ε` y construye `x_t = α_t·x_0 + σ_t·ε`. Los `α_t`, `σ_t` (y el target del score) los fija cada SDE: **VP**, **VE**, **sub-VP**.
- El loop de entrenamiento minimiza un **denoising score matching**: empuja `s_θ(x_t, t)` hacia el score real de `p_t` (equivalente, según la parametrización, a predecir `ε` o `x_0`).

**Cómo engancha con este módulo sin fricción:**

- La red queda **fija** entre variantes (variable de control); sólo se **reentrena** cuando cambia el forward SDE (Eje 1), nunca cuando cambia el sampler (Eje 2).
- `data_generation` alimenta los `x_0` (vía `sample_torch` / `dataloader`); se recomienda `standardize=True` (escala ≈ N(0,1)) y guardar `mean_`/`std_` para des-estandarizar las muestras generadas.

**Actualización (05/07/2026):** aquellos módulos ya existen — ver `sde.md`, `training.md` y `samplers.md`. Lo pendiente en **este** paquete es la U-Net de Fase 2 (`unet.py`), con su spec en `.kiro/specs/score-unet/`.
