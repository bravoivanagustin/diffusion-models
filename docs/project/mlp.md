# Módulo `mlp` — la red de score

Ubicación: `diffusion-models/src/diffusion/mlp/`. Implementa la **red que aprende el score** `s_θ(x,t) ≈ ∇_x log p_t(x)` para datos de **puntos** (toy data 2D). Para puntos la red es un **MLP** chico con un embedding del tiempo `t`; la **U-Net** entra recién cuando se pase a imágenes. Es la **variable de control** del estudio de ablación.

---

## 1. Objetivo del módulo y cómo se acopla al pipeline

En un modelo de difusión la red nunca genera la muestra de un saque: aprende el **score** `∇_x log p_t(x)` —el campo que apunta hacia zonas más probables de `p_t`, la distribución de datos tras un tiempo `t` de ruido—. Con ese score, el sampler integra la **SDE reversa** (Anderson, 1982) y convierte ruido en una muestra nueva. **Este módulo es esa red**: mapea `(x_t, t) -> score`.

`data_generation` produce los `x_0`; el **forward SDE** los ruidea para fabricar el par de entrenamiento; **acá** vive la función que estima el score. Dónde se ubica en el flujo:

```
 data_generation         forward SDE                  red                 sampler reverso
     x_0       ──►  x_t = α_t·x_0 + σ_t·ε   ──►   s_θ(x_t, t)    ──►   integra la SDE/ODE inversa
                    (añade ruido; define target)  (este módulo)        (genera muestras nuevas)
```

- **Entrada / salida**: `forward(x, t)` toma `x` de shape `(B, data_dim)` y `t` de shape `(B,)` o `(B, 1)`, y devuelve el score de shape `(B, data_dim)` —misma dimensión que `x`, porque el score `∇_x log p_t(x)` vive en el mismo espacio que `x`—.
- **Rol en la ablación**: la red es **variable de control**. Se mantiene **fija** (misma arquitectura, mismos hiperparámetros) en las 16 celdas de la matriz (4 SDEs × 4 samplers), para que toda diferencia medible se atribuya a la matemática (la SDE y el sampler) y **no** a la ingeniería de la red. (Ver `ejes.md`.)
- **Regla de reentrenamiento**: la red es agnóstica a la SDE, pero cada **forward SDE** (Eje 1) define un `p_t` distinto y por lo tanto un score distinto → **un entrenamiento por variante de SDE**. Cambiar el **sampler** (Eje 2) reusa el mismo score, sin reentrenar.

---

## 2. Funcionamiento y arquitectura

### Archivos

```
src/diffusion/mlp/
  score_mlp.py   # SinusoidalEmbedding, ResidualBlock, ScoreMLP (+ smoke block __main__)
  __init__.py    # re-exporta las tres clases
tests/test_score_mlp.py   # suite de pytest
```

Toda la red es **PyTorch puro** (`nn.Module`). A diferencia de `data_generation` —cuyo core es numpy y sólo importa torch de forma diferida—, este módulo **importa torch directamente**: es una red neuronal, torch es dependencia dura (`torch>=2.0`).

### Las tres clases (`score_mlp.py`)

**`SinusoidalEmbedding(embed_dim=128)`** — convierte el escalar de tiempo `t` en un vector, con la codificación de Transformers (senos y cosenos a distintas frecuencias):

```
embed(t)_{2i}   = sin(t / 10000^{2i/d})
embed(t)_{2i+1} = cos(t / 10000^{2i/d})       i = 0 … d/2-1,  d = embed_dim
```

Los denominadores `10000^{2i/d}` se **precomputan en `__init__`** y se guardan como **buffer** (`register_buffer`) —no son parámetros: no se aprenden, pero acompañan al módulo en `.to(device)`—. `embed_dim` debe ser **par** (cada frecuencia aporta un seno y un coseno); si no, `ValueError`. Funciona para **cualquier `t` flotante no negativo, sin supuestos sobre su escala**: el rango del tiempo depende de la SDE (`[0, 1]`, `[0, T]`, o pasos enteros), así que el embedding no asume ninguno.

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
| `data_dim` | `2` | dim del dato = dim de la salida. `2` para VP/VE/sub-VP `(x, y)`; `4` para **CLD** (estado aumentado posición-momento `(x, y, v_x, v_y)`). |
| `embed_dim` | `128` | dim del embedding de tiempo (debe ser par). |
| `hidden_dim` | `256` | ancho de las capas ocultas en todos los bloques. |
| `num_blocks` | `4` | cantidad de bloques residuales. |
| `activation` | `"silu"` | activación; soporta `silu`, `relu`, `gelu`, `tanh` (nombre desconocido → `ValueError`). |

Con los defaults la red tiene **~560 k parámetros entrenables**.

### Dónde NO vive la estocasticidad (lo central para el TP)

La red es **enteramente determinística**: dado el mismo `(x, t)` produce **exactamente** la misma salida. **No hay dropout, ni batchnorm, ni ninguna capa estocástica** (hay un test que lo verifica recorriendo `.modules()`). Esto es deliberado y es lo que sostiene la ablación: toda la aleatoriedad del pipeline vive **afuera** de la red —en el dato (`data_generation`), en el muestreo de pares del forward SDE, y en el sampler reverso—, **nunca dentro de la red**. (El reflejo exacto del módulo anterior, donde la estocasticidad sí vivía en el dato; ver `data_generation.md`.)

### Por qué un MLP (y no una U-Net)

Para **puntos** la `p_data` es 2D y de baja dimensión: un MLP chico con embedding de `t` basta y corre en CPU en segundos, ideal para barrer la matriz 4×4 de `ejes.md`. La **U-Net** —con su estructura convolucional multi-escala— recién aporta cuando la entrada es una imagen. La interfaz `(x, t) -> score` se mantiene; sólo cambia el cuerpo de la red.

### Tests (`tests/test_score_mlp.py`)

22 tests (pytest, con `importorskip("torch")`): `embed_dim` impar levanta `ValueError`; shapes del embedding y aceptación de `t` como `(B,)` y `(B, 1)` (mismo resultado); intercalado correcto sin/cos y valores acotados en `[-1, 1]`; `denom` es buffer y no parámetro; el `ResidualBlock` preserva shape y rechaza activaciones inválidas; `ScoreMLP` da salida `(B, 2)` y `(B, 4)`, **determinismo** (dos forwards iguales → salida idéntica), ausencia de dropout/batchnorm, conteo de parámetros > 0, y flujo de gradientes finito. Correr:

```
python -m pytest -q                                       # toda la suite
python -m pytest -q diffusion-models/tests/test_score_mlp.py   # solo este módulo
```

### Ejemplo de uso (API)

```python
from diffusion.mlp import ScoreMLP

net = ScoreMLP(data_dim=2)        # VP / VE / sub-VP   (data_dim=4 para CLD)
score = net(x, t)                 # x: (B, 2), t: (B,) o (B, 1)  ->  score: (B, 2)
```

El propio `score_mlp.py` trae un bloque `__main__` de smoke test: instancia la red por defecto, corre un forward sobre un batch dummy, imprime la shape de salida y el conteo de parámetros, y verifica el caso CLD (`data_dim=4`, entrada y salida 4D). Correr: `python diffusion-models/src/diffusion/mlp/score_mlp.py`.

---

## 3. Siguiente módulo: el forward SDE y el loop de entrenamiento

La red ya sabe mapear `(x_t, t) -> score`, pero **todavía no aprende nada**: falta el **target** que la entrene. Eso lo aporta el **forward SDE** (Eje 1):

- Dado un `x_0` de `data_generation`, samplea un tiempo `t` y ruido `ε` y construye `x_t = α_t·x_0 + σ_t·ε`. Los `α_t`, `σ_t` (y el target del score) los fija cada SDE: **VP**, **VE**, **sub-VP**, **CLD**.
- El loop de entrenamiento minimiza un **denoising score matching**: empuja `s_θ(x_t, t)` hacia el score real de `p_t` (equivalente, según la parametrización, a predecir `ε` o `x_0`).

**Cómo engancha con este módulo sin fricción:**

- La red queda **fija** entre variantes (variable de control); sólo se **reentrena** cuando cambia el forward SDE (Eje 1), nunca cuando cambia el sampler (Eje 2).
- `data_generation` alimenta los `x_0` (vía `sample_torch` / `dataloader`); se recomienda `standardize=True` (escala ≈ N(0,1)) y guardar `mean_`/`std_` para des-estandarizar las muestras generadas.
- Para **CLD** habrá que instanciar `ScoreMLP(data_dim=4)` (estado aumentado con momento).

**Próximos archivos** (a definir por el autor, sin implementar todavía): `sde/` (VP/VE/sub-VP/CLD + el target del score), el loop de entrenamiento, y luego `samplers/` (Euler–Maruyama, PF-ODE, Heun, predictor–corrector). El diseño completo está en `ejes.md`.
