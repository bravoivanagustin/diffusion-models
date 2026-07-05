# To Do's

Este documento contiene las tareas pendientes a desarrollar para el proyecto.

La tabla de abajo consolida los **"Próximos Pasos" / Follow-ups** de `cronica.md`, deduplicando los ítems repetidos entre entradas. Los módulos ya entregados figuran como 🟢 Hecho —`data_generation` (29/05), `mlp` (01/06), el forward SDE `sde` (04/06), el loop de entrenamiento `training` (04/06) y los samplers `samplers` (23/06)— para dar el panorama completo. El orden sigue la secuencia implícita de `ejes.md`: forward SDE → entrenamiento → samplers, más las tareas transversales.

| Fecha | Categoría | Tarea | Estado |
|-------|-----------|-------|--------|
| 29/05 | Desarrollo | Módulo `data_generation`: datasets de puntos toy 2D + CLI y preview. | 🟢 Hecho |
| 01/06 | Desarrollo | Módulo `mlp`: red de score `ScoreMLP` (MLP determinístico condicionado en el tiempo). | 🟢 Hecho |
| 04/06 | Desarrollo | Módulo `sde`: forward SDE (VP, VE, sub-VP, CLD) + el target del score (`make_sde`, kernel de perturbación, `data_dim` configurable en cualquier dimensión, 56 tests; CLD validado por Monte Carlo). | 🟢 Hecho |
| 04/06 | Desarrollo | Para CLD, instanciar `ScoreMLP(data_dim=4)` (estado aumentado posición–momento). | 🟢 Hecho (seam `sde × mlp` verificado en tests) |
| 04/06 | Desarrollo | Módulo `training`: loop de entrenamiento por denoising score matching (helper `dsm_loss`, `train`, `TrainConfig`, checkpoints, corridas por config YAML + CLI `scripts/train.py`, 21 tests). VP/VE/sub-VP convergen. | 🟢 Hecho |
| 04/06 | Desarrollo | HSM de CLD: `sde.score_target` devuelve solo el score del momento `(B, spatial_dim)` y `dsm_loss` compara esa mitad de la red (vía `is_augmented`). Estabiliza CLD (pérdida acotada y decreciente, ya no explota a miles). | 🟢 Hecho |
| 04/06 | Desarrollo | (Opcional) Pesado `λ(t)` tipo-verosimilitud para CLD (hoy `weight=1`): afinaría la convergencia, no es bloqueante. | 🔴 Pendiente |
| 23/06 | Desarrollo | Módulo `samplers`: los 4 samplers del reverso (Euler–Maruyama, PF-ODE, Heun, predictor–corrector) + registry/factory + generación checkpoint-driven y CLI. Validados sobre VP/VE/sub-VP (correctitud con score analítico); CLD con guarda (fuera de alcance). | 🟢 Hecho |
| 23/06 | Desarrollo | Dinámica reversa de CLD en los samplers (hoy la guarda rechaza SDEs aumentadas). El target de HSM de arriba ya está; falta la dinámica reversa aumentada y levantar la guarda. | 🔴 Pendiente |
| 23/06 | Desarrollo | Módulo de evaluación / visualización de Fase 1: campos de score, trayectorias de partículas, reconstrucción de densidad y comparación contra el score analítico de la mezcla (FID / IS en Fase 2). Los samplers ya exponen `return_trajectory`. | 🔴 Pendiente |
| 29/05 | Diseño | Definir el dataset final de imágenes (gatos / CIFAR-10 / FashionMNIST). | 🔴 Pendiente |
| 29/05 | Infraestructura | Iniciar git (`git init` en `tp-final/`) y aplicar el `.gitignore`. | 🟢 Hecho |

> Esta tabla se deriva de `cronica.md`. Al agregar nuevas entradas con "Próximos pasos", conviene regenerarla o actualizar los estados de las tareas existentes.
