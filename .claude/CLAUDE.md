# CLAUDE.md

Orientación para Claude Code (y cualquier sesión futura) sobre este proyecto. Es el
**TP Final de Cálculo Estocástico — 1er cuatrimestre 2026**.

> **Importante:** el proyecto ya **dejó de ser solo documentación**: hay un paquete Python
> (`diffusion-models/`) con cinco módulos terminados y testeados (`data_generation`, `models`, `sde`,
> `training` y `samplers`). Pero **todavía falta parte de la arquitectura y la decide el
> autor**: no inventes ni armes por tu cuenta el módulo que falta (la **evaluación /
> visualización** de Fase 1). Si se pide implementar algo,
> **primero acordá el alcance y el lugar**, y construí **de a poco, con la suite de pytest en verde en
> cada paso** (ver Convenciones).

## Qué es el proyecto

Trabajo de investigación para la materia Cálculo Estocástico, con un enfoque **orientado a la
implementación**. El tema son los **modelos de difusión** (*diffusion models*): modelos generativos
que aprovechan dinámicas estocásticas para generar imágenes a partir de ruido.

La meta concreta es construir un **sampleador** y estudiar cómo distintas elecciones de los
componentes *estocásticos* (no de la red neuronal) cambian los resultados generados.

El trabajo procede en **dos fases** (detalle en `docs/project/ejes.md`):

- **Fase 1 — toy 2D + MLP (en curso).** Distribuciones 2D de juguete (Swiss roll, mezcla de
  gaussianas, círculos) con una red de score **MLP construida desde cero** (chica, corre en CPU en
  minutos). Es el núcleo teórico: permite visualizar el campo de score, las trayectorias de
  partículas y la evolución de la densidad, y —para la mezcla de gaussianas— comparar contra el
  **score analítico**.
- **Fase 2 — imágenes + U-Net (red implementada; dataset a definir).** Escalar el mismo marco de SDEs
  a FashionMNIST / CIFAR-10 con una **U-Net convolucional propia** (`ScoreUNet` en `diffusion.models.unet`,
  construida a mano — decisión 05/07/2026), para medir FID / IS.

- **Dataset:** el toy 2D ya está implementado y es la pista activa para iterar rápido en CPU. El
  **dataset final de imágenes** (gatos / CIFAR-10 / FashionMNIST) **sigue a definir**.
- **La red es la variable de control:** sea MLP (Fase 1) o U-Net (Fase 2), es **determinística** y se
  mantiene **fija** (misma arquitectura, mismos hiperparámetros) en todas las celdas del estudio. Ambas
  se construyen desde cero: el MLP (≈ unas pocas capas con embedding de tiempo) y la U-Net convolucional
  (`ScoreUNet`, GroupNorm, sin dropout).

## La idea central: un estudio de ablación controlado

Mantener fija la arquitectura (misma red, mismos hiperparámetros, mismo dataset) y variar **solo lo
estocástico**. Así toda diferencia medible se atribuye a la matemática (las SDEs y los samplers) y no
a la ingeniería — que es justo lo que importa en un trabajo de cálculo estocástico. Hay dos ejes
independientes que se combinan en una matriz de experimentos.

### Eje 1 — Proceso *forward* (SDE) · **requiere reentrenar**

Cada SDE define cómo se destruye la distribución de datos hacia ruido, y cambia el *score* que la red
debe aprender (la densidad $p_t(x)$ es distinta para cada SDE) → **un entrenamiento por variante**.

- **VP-SDE** (*Variance Preserving*) — límite continuo de DDPM.
- **VE-SDE** (*Variance Exploding*) — límite continuo de NCSN.
- **sub-VP SDE** — mismo drift que VP, pero con varianza acotada por debajo de la de VP.

### Eje 2 — *Sampler* del proceso reverso · **no requiere reentrenar**

Con un mismo score $s_\theta(x,t)$ aprendido, se integra numéricamente la ecuación reversa de
distintas formas (todos los samplers comparten el mismo score).

- **Euler–Maruyama** — discretización SDE, estocástica (baseline "puro estocástico").
- **Probability-Flow ODE (Euler)** — determinística, con las **mismas marginales** que la SDE.
- **Heun** — ODE de 2º orden; sampler por defecto de EDM (mejor precisión por NFE).
- **Predictor–Corrector** — paso de SDE + correcciones de Langevin (mayor techo de calidad).

La matriz combinada es **3 × 4 = 12 celdas**. Cada celda se evalúa en ambas fases: en **Fase 1 (2D)**
con campos de score, trayectorias de partículas y reconstrucción de densidad (más el score analítico
para la mezcla de gaussianas); en **Fase 2 (imágenes)** con FID / IS a presupuestos de NFE igualados
y una grilla cualitativa de muestras. El detalle completo (ecuaciones, tabla de reentrenamiento,
costos de GPU) está en `docs/project/ejes.md`.

## Conceptos clave del dominio

- **SDE forward:** $dx = f(x,t)\,dt + g(t)\,dW$ — $f$ = *drift* (dirección), $g$ = coeficiente de
  difusión (ruido).
- **SDE reversa (Anderson, 1982):** $dx = [\,f - g^2\,\nabla_x\log p_t(x)\,]\,dt + g\,d\bar W$.
- **Score:** $\nabla_x\log p_t(x)$, el campo que apunta a zonas más probables; es lo que la red
  aproxima con $s_\theta(x,t)$.
- **La red (MLP en Fase 1, U-Net en Fase 2):** función **determinística**. Toda la estocasticidad
  vive *alrededor* de la red — en el proceso forward, en el muestreo de pares de entrenamiento, y en
  el sampler reverso — **no** dentro de ella.
- DDPM es el caso discreto/particular del marco *Score-Based* por SDEs.

## El código

El código vive en `diffusion-models/` (paquete `diffusion`, layout `src/`). El `pyproject.toml` y el
`uv.lock` están en la **raíz** del repo (`tp-final/`, junto a `docs/` y este `CLAUDE.md`) y apuntan al
código bajo `diffusion-models/`.

```
tp-final/
├── pyproject.toml        # raíz; pythonpath/testpaths/packages.find → diffusion-models/...
├── uv.lock               # entorno gestionado con uv
├── docs/                 # fuente de verdad del alcance y la teoría (ver mapa abajo)
└── diffusion-models/
    ├── scripts/              # CLIs: data_generation.py (datasets + preview), train.py (corrida YAML)
    ├── src/diffusion/
    │   ├── data_generation/   # PointDistribution (ABC) + 5 formas + registry/factory
    │   ├── models/            # redes de score: layers.py (compartido) + mlp.py (ScoreMLP) + unet.py (ScoreUNet, Fase 2) + base.py (ScoreModel)
    │   ├── sde/               # ForwardSDE (ABC) + VP/VE/sub-VP + score_target + make_sde
    │   ├── training/          # loop de DSM: train/TrainConfig, dsm_loss, checkpoints, configs YAML
    │   └── samplers/          # ReverseSampler (ABC) + EM/PF-ODE/Heun/PC + make_sampler
    ├── tests/                 # pytest (una suite por módulo)
    └── data/                  # datasets generados (gitignored; reproducibles desde --seed)
```

**Implementado hasta ahora** (cada módulo con su doc en `docs/project/` y su suite de pytest en verde):

- `diffusion.data_generation` — produce la fuente de datos $p_\text{data}(x_0)$: 5 formas de puntos de
  juguete (`gaussian`, `mixture`, `two_moons`, `spiral`, `swiss_roll`), salida `float32` + helpers
  torch (import diferido), y un CLI que guarda `.npz` + preview. Ver `docs/project/data_generation.md`.
- `diffusion.models` — las **redes de score** $s_\theta(x,t)\approx\nabla_x\log p_t(x)$ (antes
  `diffusion.mlp`, reestructurado 05/07/2026): `layers.py` con las piezas compartidas entre redes
  (`SinusoidalEmbedding`, activaciones), `mlp.py` con `ScoreMLP` para datos 2D (**enteramente
  determinística**, sin dropout ni batchnorm, `data_dim=2` para VP/VE/sub-VP), y `base.py` con el
  Protocol `ScoreModel` (contrato `(x, t) -> score`). La U-Net de Fase 2 vive en `unet.py`: `ScoreUNet`,
  una red convolucional construida a mano (decisión 05/07/2026 — no se reusa una de librería), también
  **determinística** (GroupNorm, sin dropout). Ver `docs/project/models.md`.
- `diffusion.sde` — el **Eje 1**: procesos *forward* `dx=f\,dt+g\,dW` (`VPSDE`, `VESDE`, `SubVPSDE`)
  sobre la base abstracta `ForwardSDE`, con `make_sde`/`available_sdes`. Producen el par de
  entrenamiento (`perturb`) y el **target del score** (`score_target`); `data_dim` configurable en
  cualquier dimensión. Ver `docs/project/sde.md`.
- `diffusion.training` — el **loop de entrenamiento** por *denoising score matching* que une los tres
  anteriores: `train`/`TrainConfig`, helper `dsm_loss`, checkpoints (`save/load_checkpoint`) y
  corridas config-driven por YAML (`load_config`→`build_run`) + CLI `scripts/train.py`. VP/VE/sub-VP
  convergen. Ver `docs/project/training.md`.
- `diffusion.samplers` — el **Eje 2**: los 4 samplers del reverso (Euler–Maruyama, PF-ODE, Heun,
  predictor–corrector) sobre la base abstracta `ReverseSampler` (score inyectable, driver `sample`,
  `return_trajectory`), con registry/factory y CLI `scripts/sample.py`. Validados con score analítico
  sobre VP/VE/sub-VP. Ver `docs/project/samplers.md`.

**Todavía no implementado** (lo decide el autor, módulo a módulo): la **evaluación / visualización**
de Fase 1 (campos de score, trayectorias, densidad).

> **Nota (05/07/2026):** **CLD se eliminó del alcance del proyecto** (existió como cuarta SDE, con
> HSM pendiente, y se descartó). El Eje 1 queda con VP/VE/sub-VP; no lo reintroduzcas sin pedido
> explícito del autor.

**Stack:** Python 3.14 (Windows); `torch 2.12.0+cpu` (anda en 3.14), numpy, scikit-learn, matplotlib,
pytest. Entorno gestionado con `uv`.

**Cómo correr:**

```
# Tests — desde tp-final/ o desde diffusion-models/ (pytest encuentra el rootdir solo):
python -m pytest -q

# CLIs — desde diffusion-models/:
python scripts/data_generation.py --shape two_moons --dim 2 --n-samples 2000 --seed 0 \
    --out data/two_moons.npz --preview data/two_moons.png
python scripts/train.py config/vp_mixture.yaml      # una celda del estudio = un YAML
```

Import público sin prefijo `src.`: `from diffusion.models import ScoreMLP`,
`from diffusion.data_generation import make_distribution`, `from diffusion.sde import make_sde`,
`from diffusion.training import TrainConfig, train`. No hace falta `pip install -e .` (lo resuelve el
`pythonpath` del pyproject).

> Nota OneDrive: el repo vive bajo OneDrive, así que pytest puede emitir un `PytestCacheWarning`
> (`WinError 5`, acceso denegado) inofensivo al escribir su cache. Se puede silenciar con
> `addopts = "-p no:cacheprovider"` en el pyproject si molesta.

## Mapa de la documentación

Toda la información de alcance y teoría vive en `docs/`. Cada módulo de código nuevo lleva su doc
correspondiente en `docs/project/`:

- `docs/project/proyecto.md` — overview y objetivo, en español, con la voz del autor.
- `docs/project/ejes.md` — diseño experimental completo: las dos fases, los dos ejes, la matriz 3×4,
  la red como variable de control y los requisitos de reentrenamiento.
- `docs/project/cronica.md` — bitácora fechada de avances (decisiones y entregas, módulo a módulo).
- `docs/project/to-do.md` — tareas pendientes, derivadas de los "Próximos pasos" de `cronica.md`.
- `docs/project/data_generation.md` — doc del módulo `data_generation`.
- `docs/project/models.md` — doc del módulo `models` (las redes de score: layers compartidas + MLP + U-Net `ScoreUNet`).
- `docs/project/sde.md` — doc del módulo `sde` (los procesos forward, Eje 1).
- `docs/project/training.md` — doc del módulo `training` (el loop de DSM).
- `docs/project/samplers.md` — doc del módulo `samplers` (el reverso, Eje 2).
- `docs/project/referencias.md` — mapa de literatura del área (capa teórica SDE/ODE, capa de
  samplers, y procesos estocásticos exóticos) con el set mínimo de citas esperable.
- `docs/knowledge/ddpm.md` — notas propias sobre DDPM (Ho et al., 2020).
- `docs/knowledge/score-based.md` — notas propias sobre Score-Based SDEs (Song et al., 2021).

## Referencias ancla

- **Song et al., ICLR 2021** — marco unificado por SDEs (VP/VE/sub-VP, SDE reversa, PF-ODE,
  predictor–corrector). El paper central del trabajo.
- **Ho et al., NeurIPS 2020** — DDPM.
- **Anderson, 1982** — teorema de reversión temporal de SDEs.
- **Song, Meng & Ermon, ICLR 2021** — DDIM / PF-ODE.
- **Karras et al., NeurIPS 2022** — EDM (sampler de Heun, *design space*).

Lista completa y comentada en `docs/project/referencias.md`.

## Convenciones

- **Desarrollo incremental con tests:** construir **un módulo a la vez** y entregarlo junto a su suite
  de pytest **en verde** antes de avanzar; no acumular módulos sin tests. Diseñar para testeabilidad
  (dependencias pesadas como torch con import diferido + `pytest.importorskip`).
- **La red es la variable de control:** mantenela **fija** (MLP o U-Net) — variar la arquitectura
  rompería el estudio de ablación. La red es **determinística**: nada de dropout, batchnorm ni capas
  estocásticas dentro de ella; toda la aleatoriedad vive afuera (dato, forward SDE, sampler).
- **Regla de reentrenamiento:** cambiar el **forward SDE (Eje 1)** obliga a reentrenar; cambiar el
  **sampler (Eje 2)** no.
- **Idioma:** las notas propias del autor están en **español**; mantené ese idioma para contenido
  nuevo salvo que se pida lo contrario. Los términos técnicos y matemáticos van en su forma
  convencional (drift, score, sampler, NFE, …).
- **Fuente de verdad:** los documentos en `docs/` mandan sobre el alcance y la teoría; cada módulo de
  código nuevo suma su doc en `docs/project/`.
