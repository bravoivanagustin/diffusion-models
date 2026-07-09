# Implementation Plan

- [x] 1. Fundación: `sde` sobre event shapes arbitrarios (expansión rank-aware + geometría del dato)
  - Hacer `_expand_t` **rank-aware**: reshape `t` a `(B, 1, …, 1)` con tantos `1` como dimensiones de evento tenga el estado de referencia (para rango 2 devuelve `(B, 1)`, idéntico a hoy).
  - Actualizar **en el mismo paso** las llamadas de la familia escalar (VP/VE/sub-VP, en `sde`/`marginal_prob`) para pasar el tensor de referencia, de modo que la firma nueva y sus usos aterricen juntos y la suite quede en verde. Sin cambios de fórmula.
  - Generalizar el constructor de la SDE para aceptar la geometría del dato como un entero (dato plano) o una forma multidimensional; exponer la forma de evento normalizada para los consumidores y conservar el valor crudo para el round-trip de metadata; validar que toda dimensión sea ≥ 1.
  - Observable: construir una SDE con `2` y con una forma tipo-imagen chica funciona; `perturb`/`score_target`/`sde`/`marginal_prob` sobre `(B, 3, 8, 8)` devuelven/broadcastean shapes correctas y finitas; una forma inválida lanza error claro; la salida en 2D es byte-idéntica a la actual; `test_sde.py` sigue en verde.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2, 3.3, 5.1_
  - _Boundary: sde/base.py, sde/variants.py_

- [x] 2. Prior N-D en los samplers
  - Que el muestreo del prior arme la forma a partir de la forma de evento de la SDE (en vez de una forma plana), de modo que la integración reversa corra sobre cualquier rango. El resto del driver y los pasos de los cuatro samplers no cambian (ya operan sobre la shape del estado); la normalización temporal del sampler tampoco cambia (la SDE re-expande `t` contra el estado).
  - Actualizar la docstring de la geometría del estado inicial/retorno para reflejar la forma de evento.
  - Observable: `sample(n)` devuelve muestras `(n, *forma_de_evento)` en `float32` para una forma tipo-imagen chica en los cuatro samplers; con captura de trayectoria devuelve `(n_steps+1, n, *forma_de_evento)`; el caso 2D no cambia.
  - _Depends: 1_
  - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - _Boundary: samplers/base.py_

- [x] 3. Integración: la forma de evento por la plomería de entrenamiento y generación
  - Ampliar el tipo del campo de geometría del dato en la metadata de checkpoint para que transporte un entero o una forma multidimensional (widening de tipo; la serialización de la tupla es transparente).
  - Confirmar que la generación desde checkpoint reconstruye la SDE con la forma de la metadata (el factory ya acepta la forma tras la task 1); ajustar la docstring de retorno si menciona la forma plana.
  - Gate en la capa de configuración: inyectar la geometría del dato como default del modelo **solo** cuando es un entero (camino MLP 2D); una forma multidimensional (imágenes) no se inyecta como hiperparámetro del modelo (la U-Net trae su propia config).
  - Observable: guardar y recargar un checkpoint de una SDE con forma tipo-imagen conserva la forma en la metadata y `make_sde` la reconstruye; el camino config-driven 2D sigue funcionando sin regresión. (La generación end-to-end a `(n, *forma)` se verifica en 4.2.)
  - _Depends: 1_
  - _Requirements: 4.1, 4.2, 4.3_
  - _Boundary: training/trainer.py, training/config.py, samplers/generate.py_

- [ ] 4. Validación: tests N-D, invariancia 2D y no-regresión
- [x] 4.1 (P) Tests de la familia escalar de SDEs sobre event shapes
  - Parametrizar VP/VE/sub-VP sobre una forma 2D `(2,)` y una forma tipo-imagen chica (p. ej. `(3, 8, 8)`): shapes/dtype/finitud de `perturb`/`score_target`/`sde`/`marginal_prob`; `t` como `(B,)` y `(B,1)` dan el mismo resultado; construcción con entero y con tupla, y forma inválida → error; invariancia 2D (salida byte-idéntica a la referencia con el mismo seed).
  - Observable: la suite parametrizada de `sde` corre en verde, incluyendo el caso tipo-imagen y la invariancia 2D.
  - _Depends: 1_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 3.1, 3.2, 3.3, 5.1, 5.2_
  - _Boundary: tests/test_sde.py_

- [x] 4.2 (P) Tests de samplers N-D y generación checkpoint-driven de imágenes
  - `sample()` sobre una forma tipo-imagen chica → `(n, *forma)` `float32` finito para los cuatro samplers; trayectoria con shape coherente. Round-trip end-to-end: construir un checkpoint (red sin entrenar) con forma tipo-imagen, generar vía la función de generación y verificar shape `(n, *forma)` y el archivo de salida; metadata insuficiente → error claro.
  - Observable: la suite de `samplers` corre en verde, incluyendo generación de muestras con forma de imagen y el round-trip desde checkpoint.
  - _Depends: 2, 3_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 4.1, 4.2, 4.3, 5.2_
  - _Boundary: tests/test_samplers.py_

- [x] 4.3 (P) Tests de la pérdida DSM N-D y no-regresión config-driven 2D
  - Verificar que la pérdida DSM sobre un batch tipo-imagen (p. ej. `(B, 3, 8, 8)`) con una red dummy que devuelve la shape del estado produce un escalar finito sin error de broadcasting (se espera **sin cambios** en la pérdida; si el peso no quedara rank-matched, el ajuste es un reshape de una línea). Confirmar que el camino de entrenamiento config-driven 2D sigue en verde tras el gate de configuración.
  - Observable: los tests de training corren en verde, incluyendo el caso DSM tipo-imagen y el config-driven 2D sin regresión.
  - _Depends: 1, 3_
  - _Requirements: 5.2, 5.3_
  - _Boundary: tests/test_training.py_
