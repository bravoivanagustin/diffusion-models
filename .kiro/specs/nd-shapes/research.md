# Gap Analysis — `nd-shapes`

_Brecha entre los requisitos de `nd-shapes` (generalizar `sde` familia escalar + `samplers` a event
shapes arbitrarios) y el código actual. Proyecto brownfield maduro: red (`ScoreUNet`), `train`
model+data-agnóstico y fuente de imágenes ya entregados; este es el último eslabón del camino de
imágenes. El cambio es sorprendentemente **contenido** y **por broadcasting**._

## 1. Estado actual (assets y convenciones)

- **`sde/base.py`** — `ForwardSDE`: `data_dim: int` con validación `>= 1` (L47-60); `_expand_t(t) →
  (B,1)` (L173-176) — **el cuello de botella**; `perturb`/`score_target` concretos (L121-169):
  `x_t = mean + std*eps`, `score = -eps/std`, `weight = std**2`, todos con `std` de shape `(B,1)`.
  `is_augmented` ya **no existe** (CLD eliminado del proyecto — la guarda en `samplers/base.py`
  quedó vestigial: `getattr(sde, "is_augmented", False)` siempre `False`).
- **`sde/variants.py`** — VP/VE/sub-VP: los productos coeficiente·estado rompen con `(B,1)` contra
  N-D en L52/62 (VP `-0.5*beta*x`, `alpha*x0`), L172/182 (sub-VP). VE `mean=x0` (agnóstico) pero
  `std` es `(B,1)`. `schedules.py` es matemática pura elementwise sobre `t` (propaga la shape que
  reciba). `prior_sampling` ya es shape-agnóstico (recibe la tupla del caller); el escalado de VE
  (`z * sigma_max`) es escalar → sirve N-D.
- **`samplers/base.py`** — `sample()` arma el prior como `(n_samples, self.sde.data_dim)` (L147) — la
  **única** línea con shape plana en el sampler; `_expand_t → (B,1)` (L213) y `t_batch =
  t_cur.expand(n_samples, 1)` (L159). Los drift helpers (`f - g²s`, `f - ½g²s`) y los cuerpos de los 4
  samplers (`randn(x.shape)`, `reshape(B,-1).norm(...)`, aritmética de drift) **ya generalizan**.
- **`training/losses.py`** — `dsm_loss` (L54): `(weight * (score_pred - score_real).pow(2)).mean()`.
  `weight` es `(B,1)`; **no broadcastea** contra `(B,C,H,W)`. Pero es automático: si la SDE
  generalizada devuelve `std` de rango-emparejado (`(B,1,1,1)`), entonces `weight = std**2` también lo
  es y esta línea broadcastea **sin cambios en `losses.py`**.
- **Plomería `data_dim`** — `trainer.py`: `TrainResult.data_dim: int = 0` (L63), copiado de
  `sde.data_dim` (L142), guardado en meta (L177); `samplers/generate.py`: lee `meta["data_dim"]` →
  `make_sde(sde_name, data_dim=data_dim)`; `config.py:141`: `model_raw.setdefault("data_dim",
  sde.data_dim)` inyecta `data_dim` en la config del **modelo** (dimensiona el MLP por defecto).
- **Fuente de imágenes** — `data_generation.infinite_batches` emite `(B, 3, 64, 64)` en `[-1,1]`
  (contrato confirmado). `models/base.py` ya documenta `x` como `(B,C,H,W)`.
- **Convenciones**: registry/factory `make_sde`; `float32`; `_expand_t`/`_std_eps`; tests
  parametrizados por variante/dim; torch dependencia dura en `sde`/`samplers`/`training`.

## 2. Mapa Requisito → Asset (con brechas)

| Req | Asset reutilizable | Brecha (Missing / Unknown / Constraint) |
|-----|--------------------|------------------------------------------|
| **R1** familia escalar N-D | `variants` coeff products, `base.perturb`/`score_target`, `_expand_t` | **Missing**: broadcasting ndim-aware de los coeficientes (que `std`/`beta`/`sigma`/`alpha` broadcasteen contra `x` de cualquier rango). **Constraint**: 2D idéntico. |
| **R2** samplers N-D | `sample()` prior shape (L147); drift helpers + cuerpos (ya genéricos) | **Missing**: solo el prior shape `(n_samples, *event_shape)`. El resto ya generaliza; `_expand_t` del sampler puede quedar `(B,1)` (la SDE re-expande contra `x`). |
| **R3** event shape param | `data_dim: int` + validación en `ForwardSDE.__init__` | **Missing**: aceptar `int` \| `tuple`, exponer la event shape, validar forma inválida. |
| **R4** generación checkpoint-driven imágenes | meta `data_dim` (trainer), `generate.make_sde(data_dim=)`, `config.setdefault` | **Missing/Constraint**: la forma viaja como `int`\|`tuple` por la meta; `make_sde` la acepta. **Constraint**: `config.py` inyecta `data_dim` en la config del modelo (default MLP) — para imágenes el modelo es U-Net; hay que evitar que una tupla rompa ese path. Seam con el formato de checkpoint de `train-decoupling`. |
| **R5** compat 2D + DSM + tests | `dsm_loss` (genérico), suite existente | **Unknown→resuelto**: `dsm_loss` **no cambia** si `std` queda rango-emparejado (`weight` pasa a `(B,1,1,1)`); **verificar**. **Missing**: tests parametrizados 2D + imagen-chica. **Constraint**: 2D byte-idéntico. |

## 3. Opciones de implementación

### Opción A — Extender en el lugar (recomendada)
Generalizar dentro de los módulos existentes: `_expand_t` ndim-aware (que reshape `t` contra el rango
de `x`), `data_dim` → event shape (`int` \| `tuple`) en `ForwardSDE`, prior shape en `sample()`, y
transportar la forma por meta/`generate`/`config`. La matemática no cambia — solo las shapes.

- ✅ Cambio mínimo, reusa todos los patrones, backward-compatible (2D intacto). ✅ Sin archivos nuevos.
- ❌ Toca varios archivos (sde/samplers/training); exige cuidado con la invariancia 2D y el seam de
  `config.py`.

### Opción B — Utilidad de shapes compartida
Un helper chico (p. ej. `expand_to(t, x)` + utilidades de event-shape) importado por `sde` y
`samplers`, para no duplicar la lógica ndim-aware del `_expand_t` entre las dos bases.

- ✅ Aísla la única primitiva de broadcasting; una sola fuente de verdad. ❌ Indirección extra para 3
  líneas; nuevo archivo.

### Opción C — Híbrido (variante recomendada de A)
Opción A **+** factorizar la primitiva ndim-aware en un solo lugar (helper compartido o método en la
base común) para evitar duplicarla entre `sde/base.py` y `samplers/base.py`. Es A con la única pieza
compartida extraída.

- ✅ Lo mejor de A con cero duplicación del broadcasting. ❌ Decisión menor de dónde vive el helper.

## 4. Esfuerzo y Riesgo

- **R1 (sde N-D) + R2 (samplers prior)** — Esfuerzo **S–M**, Riesgo **Bajo-Medio**: la matemática no
  cambia, solo el broadcasting; el riesgo es un bug sutil de shapes, mitigado porque el sampler ya es
  genérico (solo el prior) y por la suite 2D existente.
- **R3 (event shape param)** — Esfuerzo **S**, Riesgo **Bajo**: generalizar la validación/almacenaje.
- **R4 (plomería meta/config/generate)** — Esfuerzo **S–M**, Riesgo **Medio**: cruza el formato de
  checkpoint de `train-decoupling`; el `setdefault("data_dim", …)` de `config.py` (default MLP vs
  U-Net) necesita cuidado; serializar una tupla en la meta de `torch.save` es trivial.
- **R5 (compat 2D + DSM + tests)** — Esfuerzo **S**, Riesgo **Bajo**: `dsm_loss` probablemente sin
  cambios; el grueso es agregar la parametrización N-D.

**Total estimado**: **M (3–7 días)**, riesgo global **Bajo-Medio**, localizado en la plomería
meta/config y en preservar la invariancia 2D.

## 5. Recomendaciones para la fase de diseño

**Enfoque preferido: Opción C** (extender en el lugar + factorizar la única primitiva ndim-aware).

**Decisiones clave a tomar en `/kiro-spec-design`:**
1. **Firma de `_expand_t` ndim-aware**: reshape `t` a `(B, 1, 1, …)` según el rango de `x` (p. ej.
   `t.reshape(B, *([1] * (x.ndim - 1)))`). ¿Toma `x` como arg, o el rango objetivo? ¿Vive compartido
   (helper) o se implementa en cada base? Nota: el `_expand_t` del **sampler** probablemente **no
   necesita cambiar** — la SDE re-expande `t` contra `x` internamente; el sampler solo cambia el prior.
2. **Cómo la SDE expresa la geometría del dato**: generalizar `data_dim: int` a `int | tuple[int,…]`
   con una propiedad derivada `event_shape` (int `d` → `(d,)`), backward-compatible; o atributo nuevo.
   Definir el nombre y qué expone al sampler para armar `(n_samples, *event_shape)`.
3. **Seam de `config.py`** (Constraint): hoy `setdefault("data_dim", sde.data_dim)` mete `data_dim` en
   la config del **modelo**. Para imágenes el modelo es `ScoreUNet` (no se dimensiona por `data_dim`).
   Decidir cómo separar "la forma del dato para la SDE/prior" de "los hiperparámetros del modelo" para
   que una tupla no rompa el path U-Net. (Posiblemente: `data_dim`/event-shape es de la SDE, no del
   modelo; el default MLP solo aplica cuando el modelo es MLP.)
4. **Meta de checkpoint** (seam con `train-decoupling`): la forma viaja como `int | list/tuple`;
   `TrainResult.data_dim` y la meta widen a `int | tuple`; `generate.make_sde(data_dim=forma)`.

**Items "Research Needed" a arrastrar:**
- **Confirmar que `dsm_loss` no cambia**: con `std` rango-emparejado, `weight = std**2` es `(B,1,1,1)`
  y `weight * error` broadcastea. Verificar con un test N-D en implementación; si por alguna razón el
  peso no queda rango-emparejado, el ajuste es reshape del `weight` (1 línea).
- **Tamaño de la forma tipo-imagen en tests**: elegir algo chico (p. ej. `(3, 8, 8)`) para que los 4
  samplers corran rápido en CPU; opcional un caso `(3, 64, 64)` mínimo de humo.
- **Invariancia 2D**: asegurar salida byte-idéntica en 2D (misma reshape para rango 2 ⇒ `(B,1)`), para
  no tocar la Fase 1.

**Sin investigación externa de dependencias**: sin librerías nuevas; broadcasting puro de torch (2.12
CPU). `torchvision` ya está para el dato; nada que verificar de viabilidad externa.

---

## Síntesis de diseño (lentes aplicadas antes de escribir `design.md`)

**1. Generalización.** Los cinco requisitos son variaciones de un mismo problema: los coeficientes
dependientes de `t` deben broadcastear contra `x` de cualquier rango. Se generaliza con **una sola
primitiva**: `ForwardSDE._expand_t(t, ref)` reshapea `t` a `(B, 1, …, 1)` con `ref.ndim - 1` unos.
Aplicada en `sde`/`marginal_prob`, hace que todos los productos escalar-familia (`α·x0`, `std·ε`,
`-½β·x`, `diffusion`) broadcasteen. Segunda generalización: `data_dim: int` → forma de evento
(`data_dim: int | tuple`, con `data_shape` normalizada). El peso de `dsm_loss` y los drift helpers de
los samplers **vienen gratis** una vez que `std`/`g` quedan rank-matched.

**2. Build vs. Adopt.** Todo se **adopta**: broadcasting nativo de torch; el reshape rank-aware es un
idiom de 1 línea; se reusan registry/factory (`make_sde`), `_std_eps`, `float32` y los patrones de
test. **Cero** dependencias nuevas, cero archivos nuevos.

**3. Simplificación.** Decisiones para mantener el cambio mínimo:
- **El `_expand_t` del sampler NO cambia.** Solo pasa `t` como `(n,1)` a `sde.sde`/`score_fn`, que
  re-expanden contra `x`. El único cambio del sampler es el shape del prior en `sample()`.
- **Sin helper compartido / módulo nuevo.** La primitiva rank-aware se necesita en **un solo lugar**
  (`sde/base.py`); no hay duplicación que factorizar.
- **`data_dim` crudo se conserva** (backward-compat con la meta de checkpoint y el path MLP 2D); se
  agrega `data_shape` normalizada. No se renombra en todo el repo (más invasivo, sin beneficio).
- **`config.py` gate mínimo** (`isinstance(sde.data_dim, int)`): no se rediseña la ergonomía YAML de
  entrenamiento de imágenes (fuera de alcance; el path soportado es la API de Python).
- **CLD**: la guarda vestigial `is_augmented` en `samplers/base.py` se elimina (concepto inexistente).

**Resolución de las 4 decisiones de §5:** (1) `_expand_t(t, ref)` rank-aware en `sde/base.py`, y el del
sampler queda intacto; (2) `data_dim: int | tuple` + `data_shape` normalizada (conservar `data_dim`
crudo); (3) gate `config.py` por `isinstance(int)`; (4) meta widening a `int | tuple`, round-trip por
`trainer → generate`. Invariancia 2D garantizada porque `_expand_t` de rango 2 devuelve `(B,1)` (igual
que hoy).
