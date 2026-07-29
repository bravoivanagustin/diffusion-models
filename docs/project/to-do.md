# To Do's

Este documento contiene las tareas pendientes a desarrollar para el proyecto.

La tabla consolida los **"Próximos pasos" / Follow-ups** de `cronica.md`, deduplicando ítems repetidos. Los módulos ya entregados figuran como 🟢 Hecho para dar el panorama completo. El orden sigue la secuencia del pipeline (`data_generation → models → sde → training → samplers`), más las tareas transversales. Para el flujo de datos entre módulos y los problemas conocidos cross-módulo, ver `dataflow.md`.

> Los conteos de tests son a la fecha de esta tabla (2026-07-16) y por archivo colectable con `pytest --collect-only`; `test_image_data.py` se saltea si no hay `torchvision` instalado, así que no siempre suma.

| Fecha | Categoría | Tarea | Estado |
|-------|-----------|-------|--------|
| 29/05 | Desarrollo | Módulo `data_generation`: datasets de puntos toy 2D (5 formas) + CLI y preview (`test_data_generation.py`, 24 tests). | 🟢 Hecho |
| 01/06 | Desarrollo | Red de score `ScoreMLP` (MLP determinístico condicionado en el tiempo). Nació como `diffusion.mlp`. | 🟢 Hecho |
| 04/06 | Desarrollo | Módulo `sde`: forward SDE (VP, VE, sub-VP) + target del score (`make_sde`, kernel de perturbación; generalizado a event shapes N-D, `test_sde.py`, 69 tests). | 🟢 Hecho |
| 04/06 | Desarrollo | Módulo `training`: loop de DSM (`dsm_loss`, `train`, `TrainConfig`, checkpoints model-agnósticos, config YAML + CLI). VP/VE/sub-VP convergen. | 🟢 Hecho |
| 23/06 | Desarrollo | Módulo `samplers`: los 4 samplers del reverso (Euler–Maruyama, PF-ODE, Heun, predictor–corrector) + registry/factory + `generate_from_checkpoint` y CLI. Validados con score analítico (`test_samplers.py`, 153 tests). | 🟢 Hecho |
| 05/07 | Diseño | CLD (4ª SDE): **eliminada del alcance** del TP; el Eje 1 queda con VP/VE/sub-VP (matriz **3×4**). | ⚪ Descartado |
| 05/07 | Desarrollo | Reestructuración `diffusion.mlp` → `diffusion.models` (layers/mlp/unet/base) + `make_model`/`REGISTRY`, para hacerle lugar a la U-Net. | 🟢 Hecho |
| 05/07 | Desarrollo | U-Net de Fase 2 `ScoreUNet` (convolucional, a mano; GroupNorm, atención en 16×16 + bottleneck; ~17.2M params por defecto). Suite `test_models.py` (49 tests). | 🟢 Hecho |
| 08/07 | Desarrollo | Fuente de datos de imágenes: `infinite_batches` / `CatImages` / `report_small_images` en `data_generation.images` + `torchvision`. Higiene report-only; dedup en `scripts/limpiar_imagenes.py`. | 🟢 Hecho |
| 09/07 | Desarrollo | Generalización N-D de `sde`/`samplers` a event shapes `(B,C,H,W)` (spec `nd-shapes`), sin romper el toy 2D. | 🟢 Hecho |
| ~15/07 | Desarrollo | Entrenamiento reanudable: `checkpoint_every` + `on_checkpoint`/`TrainSnapshot`, sidecar de resume, `resolve_resume`/`load_resume`/`validate_compatible`, CLI `--force`/`--resume-from`/`--checkpoint-every`. | 🟢 Hecho |
| ~15/07 | Desarrollo | Fase 2 de punta a punta (notebook `06`): entrenar `ScoreUNet` sobre gatos y generar — test de memorización (2 gatos, 32×32, ~1M params, VP + PC). | 🟢 Hecho (prueba) |
| ~15/07 | Análisis | Notebooks de auditoría `audit_01..04`: marginales, score-vs-ε, ruido en el último paso, error por `t` en gatos. | 🟢 Hecho |
| 28/07 | Desarrollo | EMA de los pesos (`ema-weights`): sombra exponencial opt-in (`TrainConfig.ema_decay`), publicada en los checkpoints, hermano de crudos, resume que preserva crudos + sombra; retiro del checkpoint "best". | 🟢 Hecho |
| 29/07 | Desarrollo | **Eficiencia de entrenamiento en GPU** (`gpu-training-efficiency`): AMP opt-in (`TrainConfig.amp`), knobs de dataloader (`data.num_workers`/`pin_memory`) + transferencia no bloqueante, y `cudnn.benchmark` auto-on en CUDA. Todo opt-in y sin regresión (defaults = corrida histórica en CPU). Suite 567 en verde. | 🟢 Hecho |
| 29/07 | Diseño | Multi-GPU / DDP / DataParallel en el entrenamiento: **diferido explícitamente** (fuera de alcance de `gpu-training-efficiency`; reintroducir solo con pedido del autor, igual que CLD). | ⚪ Diferido |
| — | Desarrollo | **Módulo de evaluación / visualización de Fase 1**: campos de score, trayectorias de partículas, reconstrucción de densidad y comparación contra el score analítico de la mezcla (FID / IS en Fase 2). Los samplers ya exponen `return_trajectory`; no existe subpaquete `eval`/`viz` todavía. | 🔴 Pendiente |
| 29/05 | Diseño | Definir el **dataset final de imágenes** (gatos / CIFAR-10 / FashionMNIST). Hoy solo hay `cats-prueba` como sobreajuste. | 🔴 Pendiente |
| — | Desarrollo | **Camino CLI/YAML para imágenes**: `build_run`/`make_distribution` solo cablean la pista toy 2D; la Fase 2 es hoy **notebook-only**. Falta una fuente de imágenes ruteable desde el `.yaml`. | 🔴 Pendiente |
| — | Desarrollo | **Persistir hiperparámetros de la SDE en el checkpoint**: el `meta` guarda solo `sde_name`+`data_dim`; `generate_from_checkpoint` reconstruye con defaults → dinámica equivocada si se entrenó con betas/`sigma_max` no-default (ver `dataflow.md`, footgun #1). | 🔴 Pendiente |
| — | Investigación | **Ceguera de la red a `t` chico** (`audit_04`/`audit_02`): el `SinusoidalEmbedding` usa frecuencias tipo-Transformer y no distingue `t ∈ [1e-4, 1e-2]` → meseta de error ~35% cerca de `t=0`. Evaluar reescalar `t` / cambiar el embedding. | 🔴 Pendiente |
| — | Desarrollo | **EMA + augmentation en el entrenamiento de Fase 2** (mitigación de memorización): no hay EMA en el repo; el flip horizontal existe en la fuente pero hay que activarlo en la corrida. | 🔴 Pendiente |
| 29/05 | Infraestructura | Iniciar git (`git init`) y aplicar el `.gitignore`. | 🟢 Hecho |

> Esta tabla se deriva de `cronica.md`. Al agregar entradas con "Próximos pasos", conviene regenerarla o actualizar los estados. Las fechas `~` son aproximadas (features commiteadas pero sin entrada de crónica fechada al momento de escribir esto).
