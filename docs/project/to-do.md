# To Do's

Este documento contiene las tareas pendientes a desarrollar para el proyecto.

La tabla de abajo consolida los **"Próximos Pasos" / Follow-ups** de `cronica.md`, deduplicando los ítems repetidos entre entradas. Los módulos ya entregados figuran como 🟢 Hecho —`data_generation` (29/05), `mlp` (01/06), el forward SDE `sde` (04/06), el loop de entrenamiento `training` (04/06) y los samplers `samplers` (23/06)— para dar el panorama completo. El orden sigue la secuencia implícita de `ejes.md`: forward SDE → entrenamiento → samplers, más las tareas transversales.

| Fecha | Categoría | Tarea | Estado |
|-------|-----------|-------|--------|
| 29/05 | Desarrollo | Módulo `data_generation`: datasets de puntos toy 2D + CLI y preview. | 🟢 Hecho |
| 01/06 | Desarrollo | Módulo `mlp`: red de score `ScoreMLP` (MLP determinístico condicionado en el tiempo). | 🟢 Hecho |
| 04/06 | Desarrollo | Módulo `sde`: forward SDE (VP, VE, sub-VP) + el target del score (`make_sde`, kernel de perturbación, `data_dim` configurable en cualquier dimensión, 47 tests). | 🟢 Hecho |
| 04/06 | Desarrollo | Módulo `training`: loop de entrenamiento por denoising score matching (helper `dsm_loss`, `train`, `TrainConfig`, checkpoints, corridas por config YAML + CLI `scripts/train.py`, 17 tests). VP/VE/sub-VP convergen. | 🟢 Hecho |
| 05/07 | Diseño | CLD (pesado de HSM + dinámica reversa aumentada): se **eliminó del alcance** del TP; el Eje 1 queda con VP/VE/sub-VP (matriz 3×4). | ⚪ Descartado |
| 23/06 | Desarrollo | Módulo `samplers`: los 4 samplers del reverso (Euler–Maruyama, PF-ODE, Heun, predictor–corrector) + registry/factory + generación checkpoint-driven y CLI. Validados sobre VP/VE/sub-VP (correctitud con score analítico). | 🟢 Hecho |
| 23/06 | Desarrollo | Módulo de evaluación / visualización de Fase 1: campos de score, trayectorias de partículas, reconstrucción de densidad y comparación contra el score analítico de la mezcla (FID / IS en Fase 2). Los samplers ya exponen `return_trajectory`. | 🔴 Pendiente |
| 29/05 | Diseño | Definir el dataset final de imágenes (gatos / CIFAR-10 / FashionMNIST). | 🔴 Pendiente |
| 29/05 | Infraestructura | Iniciar git (`git init` en `tp-final/`) y aplicar el `.gitignore`. | 🟢 Hecho |

> Esta tabla se deriva de `cronica.md`. Al agregar nuevas entradas con "Próximos pasos", conviene regenerarla o actualizar los estados de las tareas existentes.
