# Implementation Plan

- [ ] 1. Fundación: base del proceso reverso
- [x] 1.1 Crear el paquete `samplers` y el esqueleto de `ReverseSampler` (ABC)
  - Definir el alias `ScoreFn` y la clase base abstracta con su `__init__` (valida `n_steps >= 1` y `0 < t_eps < T`).
  - Guarda explícita: rechazar SDEs aumentadas (`is_augmented`, CLD) con un error claro de "fuera de alcance".
  - Helpers compartidos: grilla temporal uniforme de `T` a `t_eps`; drift reverso `f - g²s` y drift de PF-ODE `f - ½g²s` (derivados de `sde.sde` y `score_fn`); normalización de `t` a `(B,1)`.
  - Declarar `step()` abstracto.
  - Observable: el paquete queda importable (`from diffusion.samplers.base import ReverseSampler`); instanciar con una SDE escalar funciona y con una SDE aumentada lanza error.
  - _Requirements: 3.2, 3.3, 8.1, 8.2, 8.4_
  - _Boundary: ReverseSampler base_

- [ ] 1.2 Implementar el driver `sample()`
  - Arrancar de `prior_sampling` (o del `init` provisto), integrar hacia atrás (`dt<0`) recorriendo la grilla y llamando a `step()`; capturar la trayectoria cuando se solicita.
  - Ejecutar bajo `no_grad`, en `float32`, sin tocar los parámetros de la red.
  - Observable: `sample(N)` devuelve `x_0` de shape `(N, data_dim)` en `float32` y finito; con `return_trajectory=True` devuelve además la trayectoria `(n_steps+1, N, data_dim)`.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.3, 8.3_
  - _Boundary: ReverseSampler base_

- [ ] 2. Core: los cuatro samplers (un archivo por sampler)
- [ ] 2.1 (P) Sampler Euler–Maruyama
  - Paso de SDE reversa: `x + (f - g²s)·dt + g·√|dt|·Z`, con ruido tomado del `generator` (estocástico).
  - Observable: produce muestras finitas; dos corridas con el mismo `generator` sembrado coinciden y con semillas distintas difieren.
  - _Depends: 1.2_
  - _Requirements: 2.2, 5.2, 5.3, 1.4_
  - _Boundary: EulerMaruyama_

- [ ] 2.2 (P) Sampler Probability-Flow ODE
  - Paso determinístico de la ODE de flujo: `x + (f - ½g²s)·dt`; ignora `generator`.
  - Observable: dos corridas con el mismo `init` producen resultados idénticos; muestras finitas.
  - _Depends: 1.2_
  - _Requirements: 2.3, 5.1, 1.4_
  - _Boundary: ProbabilityFlowODE_

- [ ] 2.3 (P) Sampler Heun
  - Esquema ODE de 2º orden: predictor Euler + corrección promediando el drift de PF-ODE en ambos extremos (dos evaluaciones de score por paso).
  - Observable: dos corridas con el mismo `init` producen resultados idénticos; muestras finitas.
  - _Depends: 1.2_
  - _Requirements: 2.4, 5.1, 1.4_
  - _Boundary: HeunODE_

- [ ] 2.4 (P) Sampler predictor–corrector
  - Paso de Euler–Maruyama (predictor) seguido de `n_corrector` correcciones de Langevin al nivel `t+dt`: `x ← x + ε·s + √(2ε)·Z`, con `ε` derivado de un target de SNR; parámetros propios `snr` y `n_corrector`.
  - Observable: produce muestras finitas; reproducible con `generator` sembrado y distinto con otra semilla.
  - _Depends: 1.2_
  - _Requirements: 2.5, 5.2, 5.3, 1.4_
  - _Boundary: PredictorCorrector_

- [ ] 3. Integración: factory, generación y entrypoints
- [ ] 3.1 Registry y factory por nombre
  - Construir el `REGISTRY` con los cuatro samplers; exponer `available_samplers()` y `make_sampler(name, sde, score_fn, **kwargs)` filtrando kwargs por firma.
  - Observable: `make_sampler` devuelve la instancia correcta; nombre desconocido lanza error enumerando las opciones; kwargs no aplicables se descartan sin fallar.
  - _Requirements: 2.1, 3.1, 4.1, 4.2, 4.3, 4.4_
  - _Boundary: samplers __init___

- [ ] 3.2 Generación desde checkpoint
  - `generate_from_checkpoint`: cargar el checkpoint, reconstruir la SDE desde su `meta` (`sde_name`, `data_dim`) y la red (en `eval`), armar el sampler con la factory, generar y guardar `.npz` (con trayectoria opcional).
  - Errores claros si la ruta/checkpoint no existe o es inválido.
  - Observable: a partir de un checkpoint genera y persiste un `.npz` con las muestras; con `seed` el resultado es reproducible; ruta inexistente produce un error explícito.
  - _Depends: 3.1_
  - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - _Boundary: generate_

- [ ] 3.3 Smoke entrypoint del módulo
  - `__main__` que recorre el registry y corre cada sampler sobre una `ScoreMLP` sin entrenar, reportando shape y estadísticas finitas.
  - Observable: `python -m diffusion.samplers` corre los cuatro samplers sin error e imprime salidas finitas.
  - _Depends: 3.1_
  - _Requirements: 2.1_
  - _Boundary: samplers __main___

- [ ] 3.4 CLI de generación
  - `scripts/sample.py` (argparse) que envuelve `generate_from_checkpoint` (checkpoint, sampler, n_samples, n_steps, seed, salida, flag de trayectoria).
  - Observable: la CLI genera un `.npz` desde un checkpoint dado por argumentos.
  - _Depends: 3.2_
  - _Requirements: 6.1, 6.2, 6.3_
  - _Boundary: scripts/sample.py_

- [ ] 4. Validación: suite de tests (comparten `tests/test_samplers.py`, secuenciales)
- [ ] 4.1 Tests de contrato y factory
  - Parametrizar sobre samplers × SDEs escalares: shape `(N, data_dim)`, `dtype float32`, finitud; grilla arranca en `T` y termina en `t_eps`; `return_trajectory` con shape coherente; `t` como `(B,)` y `(B,1)` da igual resultado; `n_steps` configurable.
  - Factory: `available_samplers()` esperado, tipo correcto, nombre desconocido → error, kwargs filtrados; los parámetros de una `ScoreMLP` no cambian tras `sample()`.
  - Observable: `python -m pytest` corre estos tests en verde.
  - _Requirements: 1.1, 1.3, 1.4, 1.5, 2.1, 4.1, 4.2, 4.3, 4.4, 8.1, 8.3, 8.4, 3.3_

- [ ] 4.2 Tests de determinismo y reproducibilidad
  - PF-ODE y Heun con el mismo `init` → `torch.equal`; EM y PC con el mismo `generator` sembrado → idénticos, con semillas distintas → distintos.
  - Observable: los tests de determinismo/reproducibilidad pasan en verde.
  - _Requirements: 2.2, 2.3, 5.1, 5.2, 5.3_

- [ ] 4.3 Test de correctitud con score analítico
  - Para un target gaussiano `N(μ, Σ_0)`, construir el score analítico de la marginal `p_t` desde `sde.marginal_prob` e inyectarlo como `ScoreFn`; verificar que cada sampler recupera media/covarianza con tolerancia Monte Carlo (cobertura 4 samplers × 3 SDEs escalares).
  - Seam end-to-end: `make_sde` + `ScoreMLP` + `make_sampler` con shapes coherentes (incl. `data_dim` variable).
  - Observable: el test de recuperación gaussiana pasa dentro de la tolerancia para los cuatro samplers.
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 4.4 Tests de generación checkpoint-driven
  - Construir un checkpoint con `save_checkpoint` sobre una `ScoreMLP` sin entrenar; generar vía `generate_from_checkpoint`; verificar shape y archivo `.npz`; ruta inexistente → error.
  - Observable: el test de generación desde checkpoint pasa en verde y produce el `.npz` esperado.
  - _Requirements: 6.1, 6.2, 6.3, 6.4_
