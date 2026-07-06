# Gap Analysis — score-unet

Fecha: 2026-07-06. Alcance analizado: requirements.md de `score-unet` (alcance "solo la red")
contra el estado actual del paquete `diffusion` (post-refactor `models-restructure`, suite 242 en
verde).

## 1. Estado actual (assets relevantes)

### Reutilizable directo (módulo `models`)

- `models/layers.py` — `SinusoidalEmbedding` (acepta `t` como `(B,)` o `(B, 1)` vía
  `reshape(-1)`, buffer no aprendible, testeado para escalas `[0,1]`/`[0,T]`/pasos enteros) y
  `_make_activation`/`_ACTIVATIONS` (silu/relu/gelu/tanh, `ValueError` con mensaje claro). Cubre de
  base los requisitos R1.2 (en la rama del embedding), R1.3 (escalas de `t`) y R4.3 (activación
  desconocida).
- `models/base.py` — Protocol `ScoreModel` (`runtime_checkable`): R1.6 es verificable con un
  `isinstance` sin que la red herede de nada.
- `models/mlp.py` — el patrón a espejar: hiperparámetros como argumentos del constructor con
  defaults (R4.1), proyección final sin activación (R1.5), bloque `__main__` de smoke con shape +
  conteo de parámetros (R5.1), docstrings en español con la retórica "variable de control".
- `models/__init__.py` — punto de re-export; la `ScoreUNet` se suma al `__all__` (el docstring ya
  anuncia "la U-Net se sumará como `diffusion.models.unet`").

### Patrones de test reutilizables

- `tests/test_models.py` — la suite del módulo ya contiene los tests espejo que la U-Net necesita:
  determinismo (`torch.equal` en eval), ausencia de dropout/batchnorm recorriendo `.modules()`,
  gradientes finitos, aceptación de `t` en ambas shapes, conteo de parámetros. Se parametrizan o
  duplican para la U-Net (R3.1–R3.4, R1.2).
- No hay `conftest.py`: cada suite es autocontenida con `torch = pytest.importorskip("torch")` al
  tope (R5.3). Convención de steering `testing.md`: imports del paquete después del importorskip.

### Lo que NO existe (el gap central)

`grep` por `Conv2d|GroupNorm|MultiheadAttention|interpolate|avg_pool` en todo `diffusion-models/`:
**cero resultados**. Todo el aparato convolucional es nuevo: bloque residual conv con inyección de
tiempo, bloque de self-attention, down/upsampling, ensamblado encoder–bottleneck–decoder con skips.
No hay código previo del repo que restrinja esas elecciones — las restricciones vienen de steering
(determinismo, float32, CPU) y del brief (GroupNorm, atención en 16×16).

### Fuera del módulo (contexto de frontera, ya auditado)

Confirmado con evidencia a nivel de línea (audit del 2026-07-06, sesión de requirements):

- `sde/base.py:173-176` — `_expand_t` devuelve `(B, 1)`; `perturb`/`score_target` rompen con
  `(B, C, H, W)`. `prior_sampling` sí es shape-agnóstico (recibe `shape` explícito).
- `samplers/base.py:145-148` — el driver muestrea el prior con shape fija `(n_samples, data_dim)`;
  los coeficientes `g` de `sde.sde` heredan el broadcast `(B, 1)`.
- `training/trainer.py:96-102, 151-164, 185-192` — construcción y recarga de checkpoint atadas a
  `ScoreMLP` concreta; `TrainConfig` acarrea sus hiperparámetros. **`dsm_loss` ya es
  shape-agnóstica** (element-wise + `.mean()`).

Nada de esto se toca en esta spec (Boundary Context de requirements.md); se lista porque el diseño
debe garantizar que la U-Net encaje después **sin cambios propios** (contrato `(x, t) -> score`).

## 2. Mapa requisito → asset

| Requisito | Asset existente | Gap |
|---|---|---|
| R1.1 contrato `(B,C,H,W)` | Protocol + patrón MLP | **Missing**: la red completa |
| R1.2 `t` en `(B,)`/`(B,1)` | `SinusoidalEmbedding.reshape(-1)` | Menor: test a nivel red |
| R1.3 finitud / escalas de `t` | embedding testeado | **Missing**: test a nivel red |
| R1.4 condicionamiento efectivo | — | **Missing**: test nuevo |
| R1.5 salida no acotada | patrón MLP (sin activación final) | Menor |
| R1.6 Protocol `ScoreModel` | `base.py` ✓ | Ninguno |
| R2.1 canales C=1 / C=3 | — | **Missing** |
| R2.2 resoluciones 32 y 64 | — | **Missing** + **Constraint**: atención "en 16×16" exige mapear resolución→nivel |
| R2.3 `ValueError` resolución | patrón de mensajes (`layers.py`) | **Missing**: la validación |
| R3.1–3.4 determinismo | tests espejo en `test_models.py` | Menor + **Unknown**: tolerancia numérica de R3.3 (batch-independence en CPU float32) |
| R4.1 defaults de referencia | patrón MLP | **Missing**: elegir los valores (diseño) |
| R4.2 conteo reproducible | trivial | Ninguno |
| R4.3 activación `ValueError` | `_make_activation` ✓ | Ninguno |
| R5.1 smoke `-m` | patrón `mlp.py` `__main__` (`-m` obligatorio por imports relativos) | Menor |
| R5.2 suite CPU rápida | convención de configs chicas | **Constraint**: separar arquitectura de referencia (defaults) de config reducida de test |
| R5.3 importorskip | convención ✓ | Ninguno |
| R5.4 suite del repo en verde | 242 tests actuales | Proceso |
| R6.1 doc del módulo | `models.md` ya lista `unet.py` como pendiente | **Missing**: la sección |
| R6.2 docs de alcance | — | **Constraint**: `ejes.md`, `CLAUDE.md` y steering `product.md` aún dicen "U-Net de librería" |
| R6.3 mitigación fuera de la red | brief lo registra | **Missing**: bajarlo a docs |

## 3. Opciones de implementación

### Opción A — Extender archivos existentes (poner bloques conv en `layers.py`)

Rechazada como estrategia principal: viola la regla de admisión de `layers.py` (solo lo que ambas
redes usan **sin modificar**). El bloque residual conv, la atención y el down/up sampling son
exclusivos de la U-Net.

- ✅ Menos archivos.
- ❌ Rompe la regla documentada en `models.md` y confunde la frontera compartido/propio.

### Opción B — Componente nuevo autocontenido (`models/unet.py`) — **recomendada**

Un solo archivo nuevo con los bloques privados (residual conv + tiempo, atención, down/up) y la
`ScoreUNet` que los ensambla; re-export en `__init__.py`; tests sumados a `tests/test_models.py`
(convención "una suite por módulo"); smoke `__main__` en el propio archivo.

- ✅ Es exactamente la estructura que el brief, `models.md` y el roadmap ya anuncian.
- ✅ Testeable en aislamiento; no toca ningún módulo terminado (R5.4 trivialmente protegido).
- ✅ Los bloques quedan privados: nada se promete como API compartida antes de tiempo.
- ❌ `unet.py` será el archivo más largo del paquete (bloques + ensamblado); mitigable con la misma
  disciplina de secciones que `mlp.py`.
- Variante a decidir en diseño: si `test_models.py` crece demasiado (~22 tests actuales + ~20
  nuevos), separar `test_models_unet.py` es aceptable; la convención de steering es por módulo, no
  por archivo.

### Opción C — Híbrida (B + promover piezas a `layers.py` a posteriori)

Como B, pero si durante el diseño aparece una pieza genuinamente idéntica entre redes (candidata:
la proyección MLP del embedding de tiempo, si ambas usaran la misma), se promueve a `layers.py` en
el mismo cambio. Regla: promover solo con dos consumidores reales, nunca especulativamente.

- ✅ Mantiene la regla de admisión con evidencia.
- ❌ Riesgo menor de churn si se promueve algo que la U-Net luego especializa.

## 4. Esfuerzo y riesgo

- **Esfuerzo: M (3–7 días).** Código nuevo sustancial (bloques + ensamblado + ~20 tests + docs),
  pero PyTorch puro sin integraciones ni dependencias nuevas; los footguns conocidos (contabilidad
  de canales, concatenación de skips, broadcasting del tiempo) están señalados desde discovery.
- **Riesgo: Bajo–Medio.** La arquitectura es terreno trillado (U-Net estilo DDPM, referencias ancla
  en el repo); todo corre y se verifica en CPU. Los puntos medios: divisibilidad grupos/canales de
  GroupNorm, correctitud de la atención, y flexibilidad 32/64 con atención anclada a 16×16.

## 5. Recomendaciones para la fase de diseño

**Enfoque preferido: Opción B** (archivo nuevo autocontenido), con la variante C solo si aparece un
compartido real.

**Decisiones clave que el diseño debe fijar:**

1. Arquitectura de referencia (los defaults de R4.1): canales base y multiplicadores por nivel,
   cantidad de bloques residuales por nivel, niveles de down/up, `num_groups` de GroupNorm
   (debe dividir a los canales en todos los niveles), dimensión de la proyección del embedding de
   tiempo. Anclar en las configs chicas de Ho et al. 2020 / Song et al. 2021, escaladas a CPU.
2. Mapeo resolución→atención: el brief fija atención "en 16×16"; con entradas 32×32 y 64×64 eso
   cae en niveles distintos — definir si la atención se ancla por resolución absoluta o por nivel.
3. Implementación de la atención: QKV por conv 1×1 estilo DDPM vs `F.scaled_dot_product_attention`
   (torch 2.12 CPU) — criterio: determinismo, simplicidad y legibilidad para el TP.
4. Down/upsampling: conv con stride vs pooling; upsample nearest+conv vs `ConvTranspose2d`
   (checkerboard). Ambas familias son determinísticas; decidir por calidad/simplicidad.
5. Config reducida de tests vs arquitectura de referencia (R5.2): p. ej. canales base mínimos en
   tests para que la suite siga en el orden de los ~15–30 s actuales.
6. Tolerancia numérica del test de batch-independence (R3.3): `torch.equal` puede fallar por
   paralelización de convs en CPU; definir `allclose` con tolerancia justificada.
7. Registry/factory: `structure.md` documenta el patrón registry+factory para familias de
   variantes, pero `models` hoy no tiene factory y el consumidor (training) está fuera de alcance.
   Recomendación: **diferir** `make_model` a la spec de desacople del training (YAGNI); dejar la
   decisión registrada en el diseño.

**Research Needed (llevar al diseño, no bloquea):**

- Configuración U-Net mínima que aún genere bien en 32–64 px (literatura EDM/DDPM para presupuestos
  chicos), como referencia para los defaults.
- Comportamiento de `F.scaled_dot_product_attention` en torch 2.12 CPU (disponibilidad y
  determinismo) vs implementación manual.

---

# Discovery y decisiones de diseño — score-unet (fase de diseño, 2026-07-06)

## Summary

- **Feature**: `score-unet`
- **Discovery Scope**: Extension (nueva red dentro del módulo `models` existente; sin dependencias
  nuevas)
- **Key Findings**:
  - `F.scaled_dot_product_attention` está disponible en torch 2.12.0+cpu y es **bitwise
    determinística** en CPU (dos llamadas idénticas → `torch.equal` True). Probado localmente.
  - La equivalencia batch-vs-individual (R3.3) **no es bitwise** en CPU float32 (diff máx
    ~6e-07 por paralelización de convs), pero pasa `allclose(atol=1e-6)`. La tolerancia del test
    queda justificada empíricamente.
  - Timing CPU (probe con stack conv comparable, 64×64): configuración de referencia (~64 canales
    base) ≈ 116 ms/forward (B=2); configuración tiny de test (~8 canales) ≈ 9 ms/forward (B=4).
    La suite puede mantenerse en el orden actual usando configs tiny.
  - El zero-init del conv final (práctica DDPM) haría que la red recién instanciada devuelva
    exactamente 0 para toda entrada → **rompería los tests de R1.4** (condicionamiento temporal
    efectivo) **y R1.5** (salida con ambos signos). Se descarta en esta spec.

## Research Log

### Atención determinística en torch 2.12 CPU
- **Context**: R3 exige red determinística; el brief pide self-attention en 16×16.
- **Sources Consulted**: probe local `probe_unet_design.py` sobre el entorno real
  (torch 2.12.0+cpu, Python 3.14).
- **Findings**: SDPA disponible y bitwise-determinística en CPU con el mismo input.
- **Implications**: se adopta `F.scaled_dot_product_attention` (nativo de la plataforma) con
  proyecciones QKV por conv 1×1 estilo DDPM, single-head. No se agrega dependencia ni se
  reimplementa softmax-attention a mano.

### Batch-independence de GroupNorm + Conv (tolerancia de R3.3)
- **Context**: R3.3 exige que la normalización no dependa del batch; había que fijar el criterio
  de comparación del test.
- **Findings**: `Conv2d+GroupNorm` en eval: muestra sola vs dentro de un batch → no bitwise
  (5.96e-07 máx), sí `allclose(atol=1e-6)`.
- **Implications**: el test de R3.3 usa `torch.allclose(..., atol=1e-6)`; los tests de R3.1
  (mismo input dos veces) sí pueden usar `torch.equal` (mismo grafo, mismas rutas de cómputo).

### Costo CPU y configuración de tests
- **Context**: R5.2 exige la suite en CPU en tiempos del orden del repo (~15–30 s actuales).
- **Findings**: ver Key Findings (116 ms ref / 9 ms tiny por forward).
- **Implications**: los defaults del constructor definen la **arquitectura de referencia**; los
  tests usan una **config tiny** explícita (canales base mínimos, menos niveles). El smoke `-m`
  usa los defaults (~cientos de ms, aceptable).

## Design Decisions (síntesis)

### Decision: bloques privados en `unet.py` (Opción B del gap analysis)
- **Alternatives**: (A) promover bloques a `layers.py`; (C) híbrida con promoción a posteriori.
- **Selected Approach**: todos los bloques nuevos (`TimeMLP`, `ConvResBlock`, `AttentionBlock`,
  `Downsample`, `Upsample`) viven privados en `unet.py`; `layers.py` no se toca.
- **Rationale**: regla de admisión de `layers.py` (solo compartido idéntico con ≥2 consumidores
  reales). La proyección MLP del embedding de tiempo es propia de la U-Net (el MLP concatena el
  embedding crudo, no la comparte).
- **Trade-offs**: `unet.py` es el archivo más largo del paquete; a cambio la frontera
  compartido/propio queda nítida. Promoción futura solo con evidencia (síntesis: generalizar la
  interfaz, no la implementación).

### Decision: atención anclada a resolución absoluta + bottleneck fijo
- **Selected Approach**: `attn_resolutions: tuple[int, ...] = (16,)` — se aplica atención en los
  niveles cuya resolución espacial coincide; el bottleneck lleva atención siempre.
- **Rationale**: generalización de R2.2 (32 y 64 px): con anclaje absoluto, "atención en 16×16"
  (brief) se cumple en ambas resoluciones sin tocar la config; con anclaje por nivel habría que
  reconfigurar por dataset (rompería "misma arquitectura en todas las celdas").
- **Enmienda (validate-design, 06/07/2026)**: la colocación no era computable en construcción sin
  conocer la resolución de entrada. Se agrega `image_size: int = 64` al constructor: las
  resoluciones por nivel se derivan de ahí, la atención se instancia en construcción, y `forward`
  valida `H == W == image_size` (subsume la divisibilidad, que pasa a chequearse sobre
  `image_size` en `__init__`). R2.2 se satisface **por configuración**: una instancia por
  resolución, misma config `attn_resolutions=(16,)` en ambas. Alternativa rechazada: índices de
  nivel (`attn_levels`), que exigiría configs distintas por dataset.

### Decision: down/upsampling determinísticos anti-checkerboard
- **Selected Approach**: downsample = conv 3×3 stride 2 (aprendido); upsample = nearest ×2 +
  conv 3×3.
- **Rationale**: ambos determinísticos; nearest+conv evita los artefactos checkerboard de
  `ConvTranspose2d`. Elección estándar de las U-Nets de difusión (Ho et al. 2020, Song et al. 2021).

### Decision: sin zero-init del conv final
- **Context**: DDPM zero-inicializa la última conv para estabilidad temprana de entrenamiento.
- **Selected Approach**: init estándar de PyTorch en toda la red.
- **Rationale**: con zero-init, la red recién construida es la función constante 0 → los criterios
  R1.4/R1.5 fallarían en una red sin entrenar, que es exactamente lo que la suite testea. La
  estabilidad de entrenamiento pertenece a la futura spec de training de Fase 2.
- **Follow-up**: si esa spec lo necesita, exponerlo allí como opción explícita.

### Decision: diferir el factory/registry `make_model`
- **Selected Approach**: no se crea factory en esta spec; `ScoreUNet` se exporta como clase junto a
  `ScoreMLP`.
- **Rationale**: simplificación (YAGNI): el único consumidor genérico (training) está fuera de
  alcance; el patrón registry+factory de `structure.md` aplica a familias consumidas por nombre,
  cosa que hoy no ocurre con las redes.
- **Follow-up**: la spec de desacople del training decide si introduce `make_model`.

### Decision: configuración de referencia (defaults de R4.1)
- **Selected Approach**: `in_channels=3`, `image_size=64`, `base_channels=64`,
  `channel_mults=(1, 2, 2, 4)`, `num_res_blocks=2`, `embed_dim=128` (sinusoidal),
  `time_embed_dim=256` (4×base, convención DDPM), `attn_resolutions=(16,)`, `groups=8`,
  `activation="silu"`.
- **Rationale**: escala reducida de la U-Net de DDPM (Ho et al. 2020: base 128) apta para el TP;
  3 downsamples → resoluciones 64/32/16/8 con entrada 64 y 32/16/8/4 con entrada 32; `groups=8`
  divide a todos los anchos de la familia (64/128/256) y también a los de configs tiny de test
  (8/16). `embed_dim=128` reusa el default del `SinusoidalEmbedding` compartido.
- **Trade-offs**: defaults pensados para el estudio, no para SOTA; el conteo exacto de parámetros
  lo reporta el smoke (R5.1).

## Risks & Mitigations

- Divisibilidad grupos/canales de GroupNorm — validación fail-fast en el constructor con
  `ValueError` que nombra el nivel infractor; los defaults ya la satisfacen.
- Contabilidad de canales en skips del decoder (footgun de discovery) — el diseño fija el contrato
  de concatenación por canales y los tests de shape por nivel lo cubren indirectamente (forward
  completo en 32 y 64, C=1 y C=3).
- Tiempo de suite (R5.2) — config tiny obligatoria en tests (evidencia: 9 ms/forward), con
  **una excepción deliberada** (validate-design, 06/07/2026): un único test instancia los defaults
  y corre un forward `(1, 3, 64, 64)` para que la arquitectura de referencia quede cubierta por
  pytest y no solo por el smoke manual (~100–200 ms, dentro del presupuesto).

## References

- Ho, Jain & Abbeel (2020), *DDPM* — arquitectura U-Net de referencia (base, mults, atención 16×16).
- Song et al. (2021), *Score-Based Generative Modeling through SDEs* — marco general; DDPM++.
- Probe local: `scratchpad/probe_unet_design.py` (SDPA, batch-independence, timing) — corrido
  2026-07-06 sobre torch 2.12.0+cpu / Python 3.14.
