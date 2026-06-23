# Gap Analysis — `samplers`

_Análisis de la brecha entre los requisitos de `samplers` y el código existente, para informar la
fase de diseño. Es un proyecto brownfield (4 módulos terminados), pero el feature es un **módulo nuevo
greenfield** que reusa patrones ya establecidos. No existe nada de código de sampleo todavía
(verificado: `src/diffusion/samplers/`, `scripts/sample.py` y `docs/project/samplers.md` no existen)._

## 1. Investigación del estado actual

### Activos reutilizables (ya existen, verificados)

- **`sde/base.py :: ForwardSDE`** — toda la matemática del forward que el sampler consume:
  - `sde(x, t) -> (drift (B,d), diffusion (B,1))` — `f` y `g` para el drift reverso `f - g²·s` y el
    de PF-ODE `f - ½g²·s`.
  - `marginal_prob(x0, t) -> (mean (B,d), std (B,1))` — clave para el score analítico de validación.
  - `prior_sampling(shape, *, generator, device, dtype=float32)` — el punto de partida `x_T`.
  - Atributos `T`, `data_dim`, `is_augmented`, piso `_std_eps = 1e-5`, helper `_expand_t` (normaliza
    `t` de `(B,)`/`(B,1)` a `(B,1)`).
- **`sde/__init__.py`** — el patrón registry/factory exacto a espejar: `REGISTRY = {cls.name: cls …}`,
  `available_sdes()`, `make_sde(name, **kwargs)` con **filtrado de kwargs por `inspect.signature`**.
- **`mlp/score_mlp.py :: ScoreMLP.forward(x, t) -> (B, data_dim)`** — **ya cumple** la firma de un
  `score_fn(x, t) -> score`. No requiere adaptador.
- **`training/trainer.py :: load_checkpoint(path, *, map_location="cpu") -> (ScoreMLP, dict)`** — el
  `meta` trae `sde_name`, `data_dim`, `model{embed_dim, hidden_dim, num_blocks, activation}`,
  `history`. Suficiente para reconstruir SDE (`make_sde(meta["sde_name"], data_dim=…)`) y red sin el
  config original.
- **`training/config.py`** — patrón config-driven a copiar: `load_config` (import diferido de PyYAML),
  `build_run` → `RunSpec`, filtrado de kwargs por firma. **`scripts/train.py`** — patrón de CLI.
- **Convenciones de test** (`tests/test_sde.py`): `pytest.importorskip("torch")`, `parametrize` sobre
  variantes, chequeo shape/dtype/finitud, determinismo con `torch.Generator`, validación por **Monte
  Carlo** (`n≈40000`) y por **forma cerrada**, tests de **seam** entre módulos, tolerancias
  explícitas por método.

### Convenciones que el módulo debe respetar

Layout `sde/`-style (base ABC + un proceso por archivo + `__init__` con factory + `__main__`); nombres
`snake_case`/`PascalCase`; docstrings y doc en **español**; `float32`; `t` aceptado como `(B,)` y
`(B,1)`; estabilidad en `t→0`; la **red es variable de control y determinística** (el sampler no la
muta); doc en `docs/project/` + suite de pytest en verde.

## 2. Mapa Requisito → Activo (con brechas etiquetadas)

| Req | Activo reutilizable | Brecha (Missing / Unknown / Constraint) |
|-----|---------------------|------------------------------------------|
| R1 Generación reversa | `sde.prior_sampling`, `sde.sde`, `score_fn` | **Missing**: driver de integración (grilla temporal, loop hacia atrás, captura de trayectoria). |
| R2 Cuatro samplers | Discretizaciones en `docs/project/ejes.md` | **Missing**: `step()` de EM, PF-ODE, Heun, PC. |
| R3 Reuso sin reentrenar | `ScoreMLP.forward` ya es un `score_fn` | **Missing** (trivial): tipar el score como `Callable`. **Constraint**: no mutar la red (steering). |
| R4 Factory por nombre | Patrón de `sde/__init__.py` (kwargs por firma) | **Missing**: `REGISTRY`, `make_sampler`, `available_samplers`. |
| R5 Determinismo/reproducibilidad | Convención `torch.Generator` (`sde.perturb`) | **Missing**: pasar el `generator` por todo el loop; PF-ODE/Heun sin ruido. |
| R6 CLI config/checkpoint-driven | `load_checkpoint` + meta, `training/config.py`, `scripts/train.py` | **Missing**: `scripts/sample.py` + capa de config de generación. **Constraint**: import diferido de PyYAML. |
| R7 Correctitud por score analítico | `sde.marginal_prob` (da `p_t` en forma cerrada para target gaussiano); convención Monte Carlo | **Unknown/Missing**: **no existe** ningún score analítico en el repo (ni en `data_generation`). Ver §6. |
| R8 Robustez numérica e interfaz `t` | `_expand_t`, `_std_eps`, `float32` (en `sde.base`) | **Missing**: aplicar esos patrones en el sampler. |
| Seam CLD (fuera de alcance validado) | Flag `is_augmented` | **Constraint/Unknown**: la difusión de CLD es `(B, data_dim)` estructurada, no `(B,1)`; el manejo genérico de shape debe no romperse. Ver §6. |

## 3. Opciones de implementación

### Opción A — Extender componentes existentes
Meter los samplers dentro de `sde/` o `training/`. **Rechazada**: viola la separación por etapa del
pipeline (steering `structure.md`), mezcla forward con reverso, y contradice la decisión explícita del
autor de espejar `sde/` como módulo propio. Bloatearía `training`.

- ✅ Cero archivos nuevos de paquete. ❌ Rompe boundaries, ensucia módulos terminados y testeados.

### Opción B — Crear módulo nuevo `diffusion.samplers` (recomendada)
Paquete nuevo que espeja `sde/`: `base.py` (`ReverseSampler` ABC con grilla temporal + drifts reversos
compartidos + driver `sample()` + `step()` abstracto), un archivo por sampler
(`euler_maruyama.py`, `pf_ode.py`, `heun.py`, `predictor_corrector.py`), `__init__.py`
(registry/factory), `__main__.py` (smoke). Conecta vía `score_fn` inyectable y `load_checkpoint`.

- ✅ Separación de responsabilidades limpia; testeable en aislamiento; no toca módulos existentes;
  alinea con steering y con la decisión del autor. ✅ Reusa patrones probados (factory, generator,
  `_expand_t`/`_std_eps`).
- ❌ Más archivos; requiere diseñar bien la interfaz `score_fn` y el seam de shape de difusión.

### Opción C — Híbrido
Módulo nuevo para el núcleo de sampleo, pero la **capa de generación config/CLI** (R6) podría vivir
como extensión de `training/config.py` (que ya parsea YAML y reconstruye corridas). Decisión a tomar
en diseño: ¿la config de sampleo reusa `build_run`/`RunSpec` o tiene su propia capa fina en
`samplers`/`scripts`? Es la única parte con caso real de extensión.

- ✅ Evita duplicar el parsing de YAML y la reconstrucción desde checkpoint. ❌ Acopla `samplers` a la
  forma interna de `training.config`; riesgo de mezclar responsabilidades de entrenamiento y sampleo.

## 4. Esfuerzo y Riesgo

- **Núcleo (base ABC + driver) + 4 samplers (R1–R3, R5, R8)** — Esfuerzo **M**, Riesgo **Bajo**: la
  matemática está documentada y los patrones existen; lo único delicado es el manejo de shape de
  difusión y el determinismo.
- **Factory/registry (R4)** — Esfuerzo **S**, Riesgo **Bajo**: copia directa de `sde/__init__.py`.
- **CLI config/checkpoint-driven (R6)** — Esfuerzo **S–M**, Riesgo **Bajo–Medio**: depende de la
  decisión Opción B vs C; los activos (`load_checkpoint`, `config.py`) ya existen.
- **Predictor–corrector (R2.5)** — Esfuerzo **S**, Riesgo **Medio**: requiere fijar la fórmula del
  paso de Langevin `ε` (target de SNR) y `K`; sin defaults razonables el corrector diverge.
- **Validación por score analítico (R7)** — Esfuerzo **S–M**, Riesgo **Medio**: hay que construir el
  target analítico (no existe en el repo); la tolerancia Monte Carlo debe calibrarse.

**Total estimado**: **M (3–7 días)**, riesgo global **Bajo–Medio**.

## 5. Recomendaciones para la fase de diseño

**Enfoque preferido: Opción B** (módulo nuevo espejo de `sde/`), con la decisión Opción-B-vs-C de R6
resuelta explícitamente en diseño.

**Decisiones clave a tomar en `/kiro-spec-design`:**
1. **Interfaz del `step()`**: firma única `step(x, t, dt, *, generator) -> x_next` para los cuatro, o
   variantes (Heun necesita 2 evals, PC un loop interno de `K`). Definir cómo la base comparte
   `_reverse_drift` / `_pfode_drift`.
2. **Inyección del score**: `score_fn: Callable[[Tensor, Tensor], Tensor]` (la `ScoreMLP` entra tal
   cual); confirmar que también acepta el score analítico de validación.
3. **Capa de generación (R6)**: ¿reusar `training.build_run`/`RunSpec` (Opción C) o capa propia fina
   en `samplers` + `scripts/sample.py` (Opción B)? Recomendación: capa propia mínima que llame a
   `load_checkpoint` y `make_sde`, para no acoplar a la config de entrenamiento.
4. **Grilla temporal y `t_eps`**: uniforme vs. específica por variante (VE suele usar espaciado
   geométrico); valor del piso `t_eps` (orden `1e-3`) coherente con `_std_eps`.
5. **PC**: fórmula de `ε` por SNR (Song et al. §4) y `K` por defecto.
6. **Seam de shape de difusión**: cómo trata el driver genérico la difusión `(B,1)` (escalar) vs.
   `(B, data_dim)` (CLD). Para esta iteración basta con soportar escalar y dejar el seam de CLD
   documentado, sin validar.

**Items "Research Needed" a arrastrar:**
- **Target analítico para R7 (recomendado)**: no hace falta código de producción nuevo ni un score de
  `data_generation`. Para validar el sampler alcanza con un **target gaussiano conocido**: si
  `x_0 ~ N(μ, Σ_0)`, entonces bajo VP/VE/sub-VP `x_t ~ N(mean_coef·μ, mean_coef²·Σ_0 + std²·I)` —todo
  derivable en el test desde `sde.marginal_prob`—, y el score analítico es
  `s(x,t) = -Σ_t⁻¹ (x - μ_t)`. El sampler con ese score debe recuperar `N(μ, Σ_0)` (comparar
  media/covarianza con tolerancia Monte Carlo). El **score analítico de la mezcla** (log-sum-exp) es
  más complejo y pertenece al **futuro módulo de evaluación/visualización**, no a la validación del
  sampler.
- Tolerancias Monte Carlo y `n_steps`/`N` mínimos para que el test de R7 sea estable y rápido en CPU.
- Defaults de `n_steps` por sampler para un costo razonable en Fase 1.

**No requiere investigación externa de dependencias**: sin librerías nuevas; `torch` CPU ya es el
stack establecido y toda la matemática (EM, PF-ODE, Heun, PC) está documentada en `docs/project/ejes.md`.

---

## Síntesis de diseño (lentes aplicadas antes de escribir `design.md`)

**1. Generalización.** Los cuatro samplers son variaciones de un mismo problema: integrar el proceso
reverso hacia atrás sobre una grilla temporal compartida, con un update por paso. Se generaliza la
**interfaz** (no la implementación): un ABC `ReverseSampler` (patrón **Template Method**) posee
`sample()` + los drifts reversos compartidos (`f - g²s`, `f - ½g²s`); cada sampler concreto define solo
`step()`. Segunda generalización: el score se abstrae a un único `ScoreFn = Callable[[x,t], score]`
que cubre tanto la `ScoreMLP` entrenada como el score analítico de validación — sin adaptador, porque
`ScoreMLP.forward(x,t)` ya cumple esa firma.

**2. Build vs. Adopt.** Todo se **adopta** de patrones internos existentes: registry/factory con
filtrado de kwargs por `inspect.signature` (de `sde/__init__.py`), `training.load_checkpoint` +
metadata para la generación, `torch.Generator` para reproducibilidad, y las convenciones
`_expand_t`/`_std_eps`/`float32` de `sde/base.py`. **Cero** dependencias externas nuevas; la matemática
está en `ejes.md`.

**3. Simplificación.** Decisiones para mantener el diseño mínimo:
- **Sin sampler de CLD**: el ABC rechaza SDEs aumentadas (`is_augmented`) con error claro; el seam
  queda documentado pero no se construye (fuera de alcance, atado al HSM pendiente).
- **Sin capa de config YAML** (se descartó la Opción C de extender `training.config`): la generación
  es `argparse` sobre un helper testeable `generate_from_checkpoint`; el checkpoint ya aporta la
  config del modelo. Evita acoplar `samplers` a la config de entrenamiento.
- **Grilla temporal uniforme** para todas las variantes en esta iteración (el espaciado geométrico
  para VE queda como refinamiento futuro).

**Resolución de las 6 decisiones de §5:** (1) `step(x, t, dt, *, generator)` único para los cuatro;
(2) `ScoreFn` callable confirmado; (3) capa propia mínima (`generate.py` + `scripts/sample.py`), no
reuso de `build_run`; (4) grilla uniforme `T→t_eps`, `t_eps≈1e-3`; (5) PC con `snr=0.16`,
`n_corrector=1` (Song et al.), tunables; (6) seam de difusión resuelto vía guarda `is_augmented`
(solo escalar `(B,1)` en esta iteración).
