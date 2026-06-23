# To Do's

Este documento contiene las tareas pendientes a desarrollar para el proyecto.

La tabla de abajo consolida los **"Próximos Pasos" / Follow-ups** de `cronica.md`, deduplicando los ítems repetidos entre entradas. Los módulos ya entregados figuran como 🟢 Hecho —`data_generation` (29/05), `mlp` (01/06), el forward SDE `sde` (04/06) y el loop de entrenamiento `training` (04/06)— para dar el panorama completo. El orden sigue la secuencia implícita de `ejes.md`: forward SDE → entrenamiento → samplers, más las tareas transversales.

| Fecha | Categoría | Tarea | Estado |
|-------|-----------|-------|--------|
| 29/05 | Desarrollo | Módulo `data_generation`: datasets de puntos toy 2D + CLI y preview. | 🟢 Hecho |
| 01/06 | Desarrollo | Módulo `mlp`: red de score `ScoreMLP` (MLP determinístico condicionado en el tiempo). | 🟢 Hecho |
| 04/06 | Desarrollo | Módulo `sde`: forward SDE (VP, VE, sub-VP, CLD) + el target del score (`make_sde`, kernel de perturbación, `data_dim` configurable en cualquier dimensión, 56 tests; CLD validado por Monte Carlo). | 🟢 Hecho |
| 04/06 | Desarrollo | Para CLD, instanciar `ScoreMLP(data_dim=4)` (estado aumentado posición–momento). | 🟢 Hecho (seam `sde × mlp` verificado en tests) |
| 04/06 | Desarrollo | Módulo `training`: loop de entrenamiento por denoising score matching (helper `dsm_loss`, `train`, `TrainConfig`, checkpoints, corridas por config YAML + CLI `scripts/train.py`, 20 tests). VP/VE/sub-VP convergen. | 🟢 Hecho |
| 04/06 | Desarrollo | Pesado de HSM para CLD en el loop (hoy `sde.score_target` devuelve `weight=1`; sin él CLD no converge —el target del momento explota con `t→0`). Decidir la **fórmula del peso** y **dónde vive** (`training` vs `sde`); luego ejercitar las celdas de CLD. | 🔴 Pendiente |
| 01/06 | Desarrollo | Implementar los samplers del reverso: Euler–Maruyama, PF-ODE, Heun y predictor–corrector. | 🔴 Pendiente |
| 29/05 | Diseño | Definir el dataset final de imágenes (gatos / CIFAR-10 / FashionMNIST). | 🔴 Pendiente |
| 29/05 | Infraestructura | Iniciar git (`git init` en `tp-final/`) y aplicar el `.gitignore`. | 🔴 Pendiente |

> Esta tabla se deriva de `cronica.md`. Al agregar nuevas entradas con "Próximos pasos", conviene regenerarla o actualizar los estados de las tareas existentes.
