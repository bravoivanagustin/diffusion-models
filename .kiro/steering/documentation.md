# Convenciones de Documentación

En este proyecto la documentación **manda sobre el alcance y la teoría**: `docs/` es la fuente de
verdad y el código la sigue, no al revés. Por eso las convenciones de docs son parte del contrato.

## `docs/` es la fuente de verdad

- Ante un conflicto entre el código y `docs/` sobre **qué** hay que construir o **por qué**, gana
  `docs/`. El código se alinea a la doc, o se actualiza la doc explícitamente.
- El alcance vive en `docs/project/`, no en docstrings ni en el código.

## Estructura

- **`docs/project/`** — alcance y diseño:
  - `proyecto.md` — overview y objetivo, con la voz del autor.
  - `ejes.md` — diseño experimental completo (dos fases, dos ejes, matriz 4×4, reentrenamiento).
  - `cronica.md` — **bitácora fechada** de avances (decisiones y entregas, módulo a módulo).
  - `to-do.md` — tareas pendientes, derivadas de los "Próximos pasos" de `cronica.md`.
  - `referencias.md` — mapa de literatura del área.
  - `<módulo>.md` — **un doc por módulo de código** (`data_generation.md`, `mlp.md`, `sde.md`,
    `training.md`, …).
- **`docs/knowledge/`** — notas teóricas propias (`ddpm.md`, `score-based.md`).
- **`docs/papers/`** — PDFs ancla.

## Regla por módulo

**Cada módulo de código nuevo suma su `docs/project/<módulo>.md`** y entra en `cronica.md` con fecha.
No se entrega código sin su doc (igual que no se entrega sin tests).

## La crónica

- Append-only y **fechada** (`DD/MM`). Cada entrada: qué se entregó, decisiones tomadas, y
  "Próximos pasos / Follow-ups".
- `to-do.md` se **deriva** de la crónica (consolidando y deduplicando los follow-ups). Al agregar una
  entrada nueva con follow-ups, regenerar o actualizar los estados en `to-do.md`.
- Hay un skill dedicado (`chronicle-writer`) para escribir entradas de crónica: úsalo cuando se pida
  registrar un avance.

## Estilo de escritura

- **Idioma: español** (voz del autor) para todo contenido nuevo, salvo pedido explícito. Términos
  técnicos en su forma convencional: drift, score, sampler, NFE, DSM, kernel, prior, checkpoint.
- **Sin hard-wrap**: una **línea lógica por párrafo**; no cortar oraciones a mitad de línea. El
  editor hace el wrap visual.
- **Markdown** con matemática en LaTeX inline (`$...$`) y en bloque (`$$...$$`).
- Docstrings del código también en español, con `Args:` / `Returns:` / `Raises:` estilo Google.

---
_Convenciones de documentación, no el índice de archivos. La estructura concreta vive en `docs/` mismo._
