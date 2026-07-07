# Módulo `models` — las redes de score

Ubicación: `diffusion-models/src/diffusion/models/`. Implementa las **redes que aprenden el score** `s_θ(x,t) ≈ ∇_x log p_t(x)`: el **MLP** (`ScoreMLP`) para datos de **puntos** (toy data 2D, Fase 1) y la **U-Net** (`ScoreUNet`) para **imágenes** (Fase 2, en `unet.py`). La red es la **variable de control** del estudio de ablación.

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
  unet.py        # TimeMLP + ConvResBlock + AttentionBlock + Down/Upsample + ScoreUNet (imágenes, Fase 2) (+ smoke __main__)
  __init__.py    # re-exporta ScoreModel, SinusoidalEmbedding, ResidualBlock, ScoreMLP, ScoreUNet
tests/test_models.py   # suite de pytest
```

**La regla de admisión de `layers.py`**: solo entra lo que todas las redes usan **sin modificar**. Por eso el embedding de tiempo vive ahí (es literalmente el mismo en MLP y U-Net) pero el bloque residual **no**: el del MLP es lineal (`nn.Linear`) y el de la U-Net es convolucional con inyección de tiempo (`ConvResBlock` en `unet.py`) — comparten la idea (residual con skip), no el código. Cada red mantiene su propio bloque en su archivo.

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

### La U-Net (`unet.py`)

Para **imágenes** (Fase 2) la `p_data` es de alta dimensión y con estructura espacial: ahí el MLP no alcanza y entra la **U-Net**, la segunda red de score del módulo. `ScoreUNet` mapea `(x_t, t) -> score` sobre tensores imagen exactamente con el mismo rol y contrato que el MLP; sólo cambia el cuerpo (convolucional multi-escala en vez de lineal). Es una **U-Net estilo DDPM** (encoder → bottleneck → decoder con *skip connections* concatenadas por canales y condicionamiento temporal aditivo), en la línea de Ho et al. (2020) y Song et al. (2021). Por decisión del autor (05/07/2026) se escribe **a mano**, no se reusa de librería.

**Contrato**: `forward(x, t)` toma `x` de shape `(B, C, H, W)` y `t` de shape `(B,)` o `(B, 1)`, y devuelve el score de shape `(B, C, H, W)` en `float32` (misma shape y dtype que `x`). Satisface el Protocol `ScoreModel` de `base.py` de forma **estructural** (no hereda de él ni lo importa). Es **enteramente determinística** —normalización por `GroupNorm`, sin dropout ni batchnorm ni ninguna capa estocástica— y la salida es **no acotada**: la cabeza final no lleva activación, porque el score puede ser positivo o negativo.

**Las piezas** (bloques privados de `unet.py`, no re-exportados; reusan `SinusoidalEmbedding` y `_make_activation` de `layers.py`):

- **`TimeMLP(embed_dim, time_embed_dim, activation)`** — la proyección del tiempo: `SinusoidalEmbedding → Linear → activación → Linear`. Produce el vector de condicionamiento `(B, time_embed_dim)` **una sola vez** por forward, y lo comparten todos los bloques (cada uno lo re-proyecta a sus canales). Acepta `t` en `(B,)` o `(B, 1)` (lo normaliza el embedding reusado).
- **`ConvResBlock(in_channels, out_channels, time_embed_dim, groups, activation)`** — el bloque residual convolucional con **inyección aditiva del tiempo**: `GroupNorm → act → Conv 3×3`, luego suma la proyección temporal (`Linear(time_embed_dim, out_channels)` expandida a `(B, out_channels, 1, 1)` para broadcastear sobre `H` y `W`), después `GroupNorm → act → Conv 3×3` y se suma el skip (identidad si `in_channels == out_channels`, si no una conv `1×1`). Ese condicionamiento es lo que hace que dos tiempos distintos den salidas distintas. Es el **análogo convolucional** del `ResidualBlock` lineal del MLP: comparten la idea (residual con skip), **no** el código — por la misma regla de admisión de `layers.py`, cada red mantiene su propio bloque.
- **`AttentionBlock(channels, groups)`** — auto-atención espacial *single-head* con residual: `GroupNorm → proyección QKV (conv 1×1) → scaled_dot_product_attention → proyección de salida (conv 1×1) + skip`. Aplana cada mapa `(B, C, H, W)` a `(B, H·W, C)` para tratar las `H·W` posiciones como tokens; `torch.nn.functional.scaled_dot_product_attention` calcula la atención (escala `1/√C` interna) y es determinística en CPU. **Preserva la shape**.
- **`Downsample(channels)`** — reducción espacial ×2: conv `3×3` con `stride 2` (aprende el submuestreo). **`Upsample(channels)`** — ampliación ×2: interpolación *nearest* ×2 + conv `3×3` (separar reescalado y conv evita el *checkerboard* de la conv transpuesta). Ambos conservan los canales.

**Arquitectura de referencia (defaults del constructor)**: `in_channels=3`, `image_size=64`, `base_channels=64`, `channel_mults=(1, 2, 2, 4)`, `num_res_blocks=2`, `embed_dim=128`, `time_embed_dim=256`, `attn_resolutions=(16,)`, `groups=8`, `activation="silu"` — **≈ 17.2 M parámetros entrenables** (`17,234,307`). La atención se coloca **en construcción**, por resolución **absoluta**: en los niveles cuya resolución `image_size / 2**i` pertenece a `attn_resolutions` (16×16 con los defaults) y **siempre** en el bottleneck; con `image_size` 64 o 32 la misma config `(16,)` cumple "atención en 16×16". Los hiperparámetros van todos como argumentos del constructor (sin números mágicos), con validaciones **fail-fast** por `ValueError`: `image_size` divisible por `2**(len(channel_mults)-1)` (un `Downsample` por nivel salvo el último), `groups` divide a **todos** los anchos de canal `base_channels·m`, y en el `forward` que el alto/ancho de `x` sea `image_size` y sus canales `in_channels`. Como en el MLP, la arquitectura de los defaults es la **de referencia** del estudio: es la **variable de control** y queda idéntica en las 12 celdas de la matriz 3×4.

| Param | Default | Qué controla |
|---|---|---|
| `in_channels` | `3` | canales de la imagen de entrada = de la salida (`1` grises, `3` RGB). |
| `image_size` | `64` | resolución de trabajo (`H == W`); fija las resoluciones por nivel y la colocación de la atención. |
| `base_channels` | `64` | canales del primer nivel (los demás son `base_channels · channel_mults[i]`). |
| `channel_mults` | `(1, 2, 2, 4)` | multiplicador de canales por nivel; su longitud = cantidad de niveles. |
| `num_res_blocks` | `2` | `ConvResBlock` por nivel del encoder. |
| `embed_dim` | `128` | dim del embedding sinusoidal de tiempo (debe ser par). |
| `time_embed_dim` | `256` | dim del vector de condicionamiento de `TimeMLP` (4× base, convención DDPM). |
| `attn_resolutions` | `(16,)` | resoluciones **absolutas** donde va `AttentionBlock`; el bottleneck la lleva siempre. |
| `groups` | `8` | grupos de `GroupNorm`; debe dividir a todos los anchos de canal. |
| `activation` | `"silu"` | activación; soporta `silu`, `relu`, `gelu`, `tanh` (desconocida → `ValueError`). |

**El smoke** (`__main__` del propio `unet.py`): instancia `ScoreUNet()` con los defaults, corre un forward sobre un batch dummy `(2, 3, 64, 64)`, imprime la shape de salida y el conteo de parámetros. Correr (con `diffusion-models/src` en `PYTHONPATH`), **solo vía `-m`** como el MLP (usa imports relativos a `layers.py`): `python -m diffusion.models.unet`.

> **Nota — la mitigación de memorización vive fuera de la red.** `ScoreUNet` **no** lleva dropout (rompería el determinismo, ver más abajo). La mitigación de la **memorización** del dataset de Fase 2 —**flip horizontal** de las imágenes y **EMA** (media móvil exponencial) de los pesos— **no** es parte de la red: vive en el **entrenamiento futuro de Fase 2** (data augmentation y post-proceso de los pesos), no dentro de esta clase. Coherente con la regla del TP: toda la estocasticidad y el tuning de entrenamiento viven *alrededor* de la red, nunca dentro.

### El contrato (`base.py`)

**`ScoreModel`** — un `typing.Protocol` (`runtime_checkable`) que documenta la firma común: callable `(x, t) -> score` con `score.shape == x.shape`. Es tipado **estructural**: ninguna red lo importa ni hereda de él —lo satisfacen por tener la firma correcta— y sirve para anotar código que recibe "una red de score cualquiera" (p. ej. un futuro `train(model: ScoreModel, ...)` agnóstico a la red).

### Dónde NO vive la estocasticidad (lo central para el TP)

La red es **enteramente determinística**: dado el mismo `(x, t)` produce **exactamente** la misma salida. **No hay dropout, ni batchnorm, ni ninguna capa estocástica** (hay un test que lo verifica recorriendo `.modules()`). Esto es deliberado y es lo que sostiene la ablación: toda la aleatoriedad del pipeline vive **afuera** de la red —en el dato (`data_generation`), en el muestreo de pares del forward SDE, y en el sampler reverso—, **nunca dentro de la red**. (El reflejo exacto del módulo anterior, donde la estocasticidad sí vivía en el dato; ver `data_generation.md`.)

### Por qué un MLP (y no una U-Net)

Para **puntos** la `p_data` es 2D y de baja dimensión: un MLP chico con embedding de `t` basta y corre en CPU en segundos, ideal para barrer la matriz 3×4 de `ejes.md`. La **U-Net** —con su estructura convolucional multi-escala— recién aporta cuando la entrada es una imagen. La interfaz `(x, t) -> score` se mantiene (es el Protocol `ScoreModel`); sólo cambia el cuerpo de la red.

### Tests (`tests/test_models.py`)

43 tests (pytest, con `importorskip("torch")`). Del **embedding y el MLP**: `embed_dim` impar levanta `ValueError`; shapes del embedding y aceptación de `t` como `(B,)` y `(B, 1)` (mismo resultado); intercalado correcto sin/cos y valores acotados en `[-1, 1]`; `denom` es buffer y no parámetro; el `ResidualBlock` preserva shape y rechaza activaciones inválidas; `ScoreMLP` da salida `(B, 2)` y `(B, 4)`, **determinismo** (dos forwards iguales → salida idéntica), ausencia de dropout/batchnorm, conteo de parámetros > 0, y flujo de gradientes finito. De la **`ScoreUNet`** (sobre una config *tiny* para correr rápido en CPU, más un único test con los defaults de referencia): contrato de shape `(B, C, H, W) → (B, C, H, W)` parametrizado en `C ∈ {1, 3}` × `image_size ∈ {32, 64}`; `t` como `(B,)` y `(B, 1)` (mismo resultado) y salidas finitas en varias escalas de `t`; condicionamiento efectivo (dos `t` distintos → salidas distintas) y salida de ambos signos; `isinstance(net, ScoreModel)`; determinismo, ausencia de dropout/batchnorm, independencia del batch (muestra sola vs en batch), gradientes finitos, conteo de parámetros reproducible, y los `ValueError` fail-fast (activación desconocida, `groups` incompatible, `image_size` no divisible, tamaño de `x` distinto de `image_size`). Correr:

```
python -m pytest -q                                    # toda la suite
python -m pytest -q diffusion-models/tests/test_models.py   # solo este módulo
```

### Ejemplo de uso (API)

```python
from diffusion.models import ScoreMLP, ScoreUNet

net = ScoreMLP(data_dim=2)        # Fase 1 — puntos: VP / VE / sub-VP
score = net(x, t)                 # x: (B, 2), t: (B,) o (B, 1)  ->  score: (B, 2)

unet = ScoreUNet()                # Fase 2 — imágenes: arquitectura de referencia (defaults)
score = unet(x, t)                # x: (B, 3, 64, 64), t: (B,) o (B, 1)  ->  score: (B, 3, 64, 64)
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

**Actualización (07/07/2026):** aquellos módulos ya existen — ver `sde.md`, `training.md` y `samplers.md` —, y la **U-Net de Fase 2 (`ScoreUNet`, en `unet.py`) ya está entregada** (spec en `.kiro/specs/score-unet/`): con ella el módulo `models` tiene sus **dos redes de score**. Lo que resta en la Fase 1 es la **evaluación / visualización** (campos de score, trayectorias de partículas, reconstrucción de densidad); el entrenamiento en imágenes de Fase 2 (dataset, flip horizontal, EMA, FID/IS, corridas GPU) queda fuera de esta red y de este módulo.
