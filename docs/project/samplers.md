# Módulo `samplers` — el proceso reverso (Eje 2)

Ubicación: `diffusion-models/src/diffusion/samplers/`. Implementa el **proceso reverso**: integra numéricamente la SDE/ODE inversa para **generar muestras** `x_0` a partir de ruido, reusando el score aprendido `s_θ(x,t)`. Es el **Eje 2** del estudio de ablación de `ejes.md`: cuatro **samplers** —**Euler–Maruyama**, **Probability-Flow ODE**, **Heun** y **predictor–corrector**— que comparten el mismo score y se diferencian solo en *cómo* discretizan la reversa. Cambiar de sampler **no requiere reentrenar**.

---

## 1. Objetivo del módulo y cómo se acopla al pipeline

`sde` define el forward que ruidea `x_0`; `mlp` aprende `(x_t, t) -> score`; `training` entrena esa red. Lo que faltaba es cerrar el ciclo: **integrar la ecuación reversa para volver de ruido a datos.** Este módulo es esa última caja:

```
 data_generation     ►   forward SDE    ►      red        ►   training   ►   SAMPLER REVERSO
     x_0                  x_t = α_t·x_0          s_θ(x_t,t)      s_θ ya       integra la SDE/ODE
                          + σ_t·ε                (módulo mlp)   entrenado    inversa  ──►  x_0 nuevo
```

- **Entrada / salida**: un sampler se construye con una `ForwardSDE` y un **score** (una función `(x, t) -> score`); `sample(n_samples)` arranca del prior `p_T` (`sde.prior_sampling`) e integra hacia atrás hasta `t≈0`, devolviendo `x_0` de shape `(n_samples, data_dim)`. Opcionalmente devuelve la **trayectoria** completa.
- **Rol en la ablación**: el sampler es el Eje 2, **variable independiente**. La red y la SDE quedan fijas; variar el sampler cambia solo la integración numérica de la reversa.
- **Regla de reentrenamiento**: cambiar el **sampler (Eje 2)** reusa el mismo `s_θ` —**no** se reentrena—. (Reentrenar es del Eje 1; ver `sde.md` y `ejes.md`.)

Como `sde`, `mlp` y `training`, este módulo **importa torch directamente**: opera sobre tensores. Solo `numpy` aparece (diferido) para guardar el `.npz` de salida. **Sin** dependencias nuevas.

### Las ecuaciones que integra

- **SDE reversa** (Anderson, 1982): `dx = [f(x,t) − g(t)²·∇ₓ log p_t(x)] dt + g(t) dW̄`.
- **Probability-Flow ODE** (Song et al., 2021): `dx = [f(x,t) − ½ g(t)²·∇ₓ log p_t(x)] dt` — comparte las **mismas marginales** `p_t` que la SDE reversa, pero es determinística.

En ambas, `∇ₓ log p_t(x)` se reemplaza por el score aprendido `s_θ(x,t)`; `f`, `g` salen de `sde.sde(x,t)`.

---

## 2. Funcionamiento y arquitectura

### Archivos

```
src/diffusion/samplers/
  base.py                  # ScoreFn + ReverseSampler (ABC): grilla, drifts reversos, sample(), step() abstracto
  euler_maruyama.py        # EulerMaruyama       (name "euler")   — paso SDE estocástico (baseline)
  pf_ode.py                # ProbabilityFlowODE  (name "pf_ode")  — paso ODE determinístico
  heun.py                  # HeunODE             (name "heun")    — ODE 2º orden (2 evals/paso)
  predictor_corrector.py   # PredictorCorrector  (name "pc")      — EM + K correcciones de Langevin
  generate.py              # generate_from_checkpoint: checkpoint -> sampleo -> .npz
  __init__.py              # REGISTRY + make_sampler + available_samplers
  __main__.py              # smoke (python -m diffusion.samplers)
scripts/sample.py          # CLI de generación (argparse)
tests/test_samplers.py     # suite de pytest
```

Patrón **Template Method**, espejo de `sde/`: la base fija el algoritmo de integración (un loop hacia atrás sobre una grilla temporal) y **cada sampler define solo su paso** (`step`). Un proceso por archivo.

### La clase base `ReverseSampler`

ABC con un atributo de clase `name` (clave del registry) y un constructor `__init__(sde, score_fn, *, n_steps=500, t_eps=1e-3)`. Lo que aporta a todas las variantes:

- **El score como función inyectable** — `ScoreFn = Callable[[Tensor, Tensor], Tensor]`, con `(x (B,d), t (B,)|(B,1)) -> score (B,d)`. Una `ScoreMLP` **ya cumple** esa firma (`net(x,t)`), así que entra tal cual; pero también acepta un **score analítico en forma cerrada**, lo que habilita la validación de correctitud (ver Tests) desacoplada del entrenamiento.
- **La grilla temporal** uniforme de `T` a `t_eps` (`n_steps+1` puntos), integrada en tiempo **decreciente** (`dt<0`). El piso `t_eps` (default `1e-3`) evita `t=0`, donde el score diverge.
- **Los drifts reversos compartidos**, derivados de `sde.sde(x,t)` y `score_fn(x,t)`: `_reverse_drift = f − g²·s` (SDE) y `_pfode_drift = f − ½g²·s` (ODE).
- **El driver `sample(n_samples, *, init=None, generator=None, return_trajectory=False)`**: arranca de `sde.prior_sampling` (o del `init` provisto), recorre la grilla llamando al `step()` abstracto, corre bajo `torch.no_grad()` en `float32` y **no toca los parámetros de la red**. Devuelve `x_0`; con `return_trajectory=True`, también la trayectoria `(n_steps+1, n_samples, data_dim)`.

`step(self, x, t, dt, *, generator)` es **abstracto**: lo único que cambia entre samplers.

### Los cuatro samplers

Con `d(x,t)` el drift correspondiente y `Z ~ N(0, I)`:

| Sampler (`name`) | Paso (`step`) | NFE/paso | ¿Estocástico? | Carácter |
|---|---|---|---|---|
| **Euler–Maruyama** (`euler`) | `x + (f − g²s)·dt + g·√\|dt\|·Z` | 1 | sí | discretización de la SDE reversa; baseline "puro estocástico" |
| **Probability-Flow ODE** (`pf_ode`) | `x + (f − ½g²s)·dt` | 1 | no | ODE determinística, mismas marginales que la SDE |
| **Heun** (`heun`) | predictor `x̂ = x + d(x,t)·dt`; corrector `x + ½[d(x,t) + d(x̂, t+dt)]·dt`, con `d = f − ½g²s` | 2 | no | ODE de 2º orden (trapezoidal); mejor precisión por NFE |
| **Predictor–corrector** (`pc`) | un paso de Euler–Maruyama + `n_corrector` correcciones de Langevin al nivel `t+dt`: `x ← x + ε·s + √(2ε)·Z` | 1 + K | sí | mayor techo de calidad; refina hacia `p_t` en cada nivel |

El paso de Langevin de **PC** fija `ε` por un *target de signal-to-noise ratio*: `ε = 2·(snr·‖Z‖ / ‖s‖)²` (normas L2 medias por batch, con piso en el denominador para estabilidad). Parámetros propios del constructor: `n_corrector=1`, `snr=0.16` (estilo Song et al., 2021); son **tunables** y los demás samplers los descartan por firma (ver factory).

PF-ODE y Heun son **determinísticos** (ignoran `generator`); EM y PC son **estocásticos** y reproducibles vía `torch.Generator`.

### El registry / factory

`__init__.py` arma un `REGISTRY = {cls.name: cls}` con los cuatro samplers, idéntico al patrón de `sde`:

- `available_samplers() -> list[str]` — nombres ordenados (`["euler", "heun", "pc", "pf_ode"]`).
- `make_sampler(name, sde, score_fn, **kwargs) -> ReverseSampler` — instancia la variante, **filtra los kwargs por la firma** del constructor (así un caller genérico pasa siempre el mismo conjunto: `snr`/`n_corrector` se aplican a `pc` y se descartan para el resto) y lanza `ValueError` enumerando las opciones si el nombre es desconocido.

### Dónde SÍ vive la estocasticidad (lo central para el TP)

Acá vive la **tercera** fuente de aleatoriedad del pipeline (junto con el dato y el forward): el ruido que inyectan EM y PC durante la integración. Es, de nuevo, el reflejo de la red determinística — toda la estocasticidad vive *afuera* de ella. PF-ODE y Heun, en cambio, son el extremo determinístico del mismo eje. Por eso el ruido se sortea siempre con un `torch.Generator` opcional: explícito y reproducible.

### Generación desde checkpoint (config-driven)

`generate_from_checkpoint(checkpoint_path, sampler_name, *, n_samples, n_steps=500, seed=None, out=None, save_trajectory=False, ...)` cierra el camino de extremo a extremo, reusando `training.load_checkpoint`: carga la red entrenada y su metadata, reconstruye la SDE desde `meta` (`make_sde(meta["sde_name"], data_dim=meta["data_dim"])`), arma el sampler con la factory, genera y —si se da `out`— guarda un `.npz` (clave `samples`, más `trajectory` opcional). El checkpoint ya transporta todo lo necesario, así que el sampleo corre **sin** el config de entrenamiento original. La CLI `scripts/sample.py` (argparse) es un wrapper fino sobre esta función.

### Tests (`tests/test_samplers.py`)

Suite de pytest (en verde, `importorskip("torch")`), parametrizada sobre **los 4 samplers × las 3 SDEs escalares** (VP/VE/sub-VP). Cubre:

- **Contrato y factory**: shapes `(N, data_dim)`, `float32`, finitud; grilla `T→t_eps`; trayectoria con shape coherente; `t` como `(B,)` y `(B,1)`; `n_steps` configurable; `make_sampler`/`available_samplers`, nombre desconocido → `ValueError`, filtrado de kwargs en ambos sentidos; los parámetros de la red **no cambian** tras `sample()`.
- **Determinismo / reproducibilidad**: PF-ODE y Heun idénticos con el mismo `init` (e idénticos aunque cambie el seed → ignoran el ruido); EM y PC reproducibles con el mismo `generator` sembrado y distintos con otra semilla.
- **Correctitud matemática (el test clave)**: con el **score analítico** en forma cerrada de un target gaussiano `N(μ, σ₀²I)` —derivado de `sde.marginal_prob` como `s(x,t) = −(x − α_t·μ)/(α_t²σ₀² + σ_t²)`—, cada sampler **recupera** la distribución de datos (media y covarianza dentro de tolerancia Monte Carlo). Desacopla la corrección del sampler del entrenamiento de la red, y cubre las 12 celdas escalares.
- **Generación checkpoint-driven**: checkpoint (sin entrenar, vía `save_checkpoint`) → `generate_from_checkpoint` → `.npz`; reproducible con `seed`; ruta/metadata inválida → error claro.

> **Nota numérica.** Con score exacto, VE + samplers determinísticos (PF-ODE/Heun) recuperan la media con un error residual (~0.16) que el resto no tiene: es **correcto**, no un bug — el prior estándar de VE es `N(0, σ_max²)`, mientras la marginal terminal real es `N(μ, σ₀²+σ_max²)`, y el flujo determinístico no borra ese offset de media (los estocásticos sí lo mezclan). Relevante para el futuro módulo de evaluación.

Correr:

```
python -m pytest -q                                       # toda la suite
python -m pytest -q diffusion-models/tests/test_samplers.py   # solo este módulo
```

### Ejemplo de uso (API)

```python
from diffusion.sde import make_sde
from diffusion.training import load_checkpoint
from diffusion.samplers import make_sampler, available_samplers

net, meta = load_checkpoint("models/vp_mixture.pt")     # red entrenada + metadata
sde = make_sde(meta["sde_name"], data_dim=meta["data_dim"])

sampler = make_sampler("heun", sde, net, n_steps=200)   # "euler" / "pf_ode" / "heun" / "pc"
x0 = sampler.sample(2000)                                # muestras (2000, data_dim)
x0, traj = sampler.sample(2000, return_trajectory=True)  # + trayectoria para visualizar
```

El módulo trae un smoke (`__main__.py`): corre los cuatro samplers sobre una red **sin entrenar** y reporta shape/finitud. Correr (desde `diffusion-models/src/`): `python -m diffusion.samplers`. La CLI de generación (desde `diffusion-models/`): `python scripts/sample.py <checkpoint> --sampler heun --n-samples 2000 --out muestras.npz`.

---

## 3. Estado y qué sigue

Con los samplers entregados, la **matriz 3×4 del estudio** ya es ejecutable: 3 SDEs (VP/VE/sub-VP, las tres convergen) × 4 samplers, todas reusando el score sin reentrenar. Pendientes, en orden:

- **Evaluación / visualización** — campos de score, trayectorias de partículas, reconstrucción de densidad y la comparación contra el score analítico de la mezcla. Los samplers ya exponen `return_trajectory` para alimentarlo; el ploteo y las métricas (FID/IS en Fase 2) van en un módulo aparte.
- **Fase 2 (imágenes + U-Net)** — el mismo marco escala reusando una U-Net de librería; los samplers son agnósticos a `data_dim`, así que no requieren cambios estructurales. El diseño completo está en `ejes.md`.
