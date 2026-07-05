# Matriz de Experimentos (estudio de ablación)

El corazón del TP. Un protocolo, no una lista de corridas: define qué se mantiene fijo, qué varía y
cómo se nombra/reproduce cada celda. Detalle teórico completo en `docs/project/ejes.md`.

## La regla de oro

**La red es la variable de control.** Misma arquitectura, mismos hiperparámetros, mismo dataset en
**todas** las celdas. Lo único que cambia es lo **estocástico** (forward SDE y sampler). Así toda
diferencia medible se atribuye a la matemática, no a la ingeniería. Variar la red rompe el estudio.

## Los dos ejes (matriz 3×4 = 12 celdas)

| | Eje 1 — Forward SDE | Eje 2 — Sampler del reverso |
|---|---|---|
| **Variantes** | `vp`, `ve`, `sub_vp` | Euler–Maruyama, PF-ODE (Euler), Heun, predictor–corrector |
| **Efecto** | define `p_t(x)` → score distinto | integra el mismo score de otra forma |
| **¿Reentrena?** | **Sí** (1 entrenamiento por SDE) | **No** (reusa el checkpoint) |

**Regla de reentrenamiento** (consecuencia directa): cambiar el Eje 1 invalida el checkpoint;
cambiar el Eje 2 no. Las 12 celdas necesitan **3 entrenamientos** y 12 corridas de sampleo.

## Una celda = un archivo de config

Cada celda se describe en un `.yaml` versionable (no en código). El núcleo (`train`, `dsm_loss`) no
sabe de archivos; `load_config` → `build_run` → `RunSpec` lo ensambla reusando los factories
`make_sde` / `make_distribution`. Estructura del YAML (ver `training/config.py`):

```yaml
sde:   { name: vp, beta_min: 0.1 }        # -> make_sde
data:  { shape: mixture, dim: 2, n_samples: 4000, n_components: 8 }
train: { epochs: 300, lr: 0.002, seed: 0 }
model: { hidden_dim: 256 }                 # variable de control: igual en todas las celdas
out:   { checkpoint: models/vp_mixture.pt, loss_curve: models/vp_mixture_loss.png }
```

## Convenciones

- **Naming**: `{sde}_{dataset}` para configs y checkpoints (`vp_mixture.yaml` → `vp_mixture.pt`).
- **Checkpoint = red + metadata reproducible**: `save_checkpoint` guarda `model_state` más `meta`
  (`sde_name`, `data_dim`, hiperparámetros de red, `history` de loss). Suficiente para reconstruir la
  `ScoreMLP` y saber qué SDE la generó.
- **Reproducibilidad por seed**: toda fuente de aleatoriedad acepta `seed`/`generator`
  (`TrainConfig.seed`, `PointDistribution.seed`, `generator=` en `perturb`/`prior_sampling`). Fijar
  el seed para que las celdas sean comparables.
- **`data_dim` por SDE**: la `ScoreMLP` se instancia con `data_dim=sde.data_dim` (2 en Fase 1);
  el seam `sde × mlp` está cubierto por tests.

## Evaluación por fase

- **Fase 1 (2D)**: campo de score, trayectorias de partículas, reconstrucción de densidad; para
  `mixture`, comparar contra el **score analítico**.
- **Fase 2 (imágenes)**: FID / IS a presupuestos de **NFE igualados**, más una grilla cualitativa.

## Estado de las celdas

La matriz 3×4 es **ejecutable completa**: VP/VE/sub-VP convergen en el entrenamiento y los 4
samplers reusan cualquier checkpoint. Falta el módulo de evaluación/visualización para medir las
celdas. **CLD se eliminó del alcance el 05/07/2026** (existió como cuarta SDE, con el pesado HSM
pendiente); no reintroducirlo sin pedido explícito del autor.

---
_Protocolo y convenciones, no el registro de corridas (eso va en `docs/project/cronica.md`)._
