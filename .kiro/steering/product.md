# Visión del Producto

TP Final de **Cálculo Estocástico** (1er cuatrimestre 2026): un trabajo de investigación
**orientado a la implementación** sobre **modelos de difusión** — modelos generativos que usan
dinámicas estocásticas para generar muestras a partir de ruido. No es solo documentación: hay un
paquete Python (`diffusion-models/`) que se construye módulo a módulo.

## Capacidades centrales

- Generar la fuente de datos `p_data(x_0)`: distribuciones de puntos de juguete 2D y una red de
  score `s_θ(x,t)` determinística para aproximar `∇_x log p_t(x)`.
- Definir procesos *forward* (SDEs: VP, VE, sub-VP) que destruyen los datos hacia ruido.
- Entrenar la red por *denoising score matching* (DSM) para una SDE dada.
- Integrar el proceso reverso con distintos *samplers* (Euler–Maruyama, PF-ODE, Heun,
  predictor–corrector) para generar muestras.

## La idea central: estudio de ablación controlado

Mantener la **red fija** (misma arquitectura, hiperparámetros y dataset) y variar **solo lo
estocástico**. Así toda diferencia medible se atribuye a la matemática —las SDEs y los samplers— y
no a la ingeniería, que es justo lo que importa en un trabajo de cálculo estocástico. Dos ejes
independientes forman una matriz de experimentos **3×4 = 12 celdas**:

- **Eje 1 — Proceso forward (SDE):** VP, VE, sub-VP. Cambiarlo **obliga a reentrenar** (cada
  SDE define una `p_t(x)` distinta, y por ende un score distinto).
- **Eje 2 — Sampler del reverso:** Euler–Maruyama, PF-ODE, Heun, predictor–corrector. Cambiarlo
  **no requiere reentrenar** (todos comparten el mismo score aprendido).

## Las dos fases

- **Fase 1 — toy 2D + MLP (en curso).** Distribuciones 2D con una `ScoreMLP` chica que corre en CPU.
  Permite visualizar campo de score, trayectorias y densidad; para la mezcla de gaussianas se compara
  contra el **score analítico**.
- **Fase 2 — imágenes + U-Net (no empezada).** Escalar el mismo marco a FashionMNIST / CIFAR-10
  reusando una U-Net de librería; medir FID / IS. El dataset final de imágenes está **a definir**.

## Propuesta de valor

La red es la **variable de control**, no el objeto de estudio. El producto vale por aislar el efecto
de las elecciones estocásticas (forward SDE y sampler) sobre la generación, con respaldo teórico
(Song et al. 2021 como referencia ancla) y validación cuantitativa (score analítico, diferencias
finitas).

---
_Foco en el propósito y los patrones, no en la lista exhaustiva de features._
