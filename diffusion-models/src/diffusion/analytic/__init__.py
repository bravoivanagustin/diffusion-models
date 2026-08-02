"""Verdad analítica del laboratorio 2D: densidad, score y sesgo exactos.

Este módulo es el **patrón de referencia** contra el que se mide el resto del estudio.
Sobre una mixtura de gaussianas 2D de parámetros conocidos en forma cerrada calcula
``p_t(x)``, ``log p_t(x)`` y ``∇_x log p_t(x)`` **exactos** para las tres SDEs escalares
del Eje 1 (VP, VE y sub-VP), más el sesgo de inicialización de cada una. Con eso el error
de estimación del score se puede *apagar* y separar del error de discretización y del de
truncación.

Acoplamiento con el pipeline:

- **Consume** el contrato marginal ya publicado por :mod:`diffusion.sde`
  (``marginal_prob`` da ``α_t`` y ``σ_t``, y ``T`` el horizonte) y los parámetros
  verdaderos de la mixtura exacta de :mod:`diffusion.data_generation`. No re-deriva los
  schedules ni ramifica por variante: un único camino de código cubre las tres SDEs.
- **No importa** :mod:`diffusion.samplers`: la relación es por contrato de callable
  ``(x, t) -> score``, el mismo que los cuatro samplers ya aceptan de una red entrenada,
  así que samplear con score exacto no requiere tocarlos. La dirección obligatoria de
  dependencias es ``data_generation → sde → analytic``.

``torch`` es dependencia dura del módulo (como en :mod:`diffusion.sde`), no diferida.

Uso típico (las piezas concretas llegan en las tasks siguientes)::

    from diffusion.analytic import MixtureOracle

Nota: el oráculo opera **solo** sobre mixturas de parámetros exactos. Los modelos
entrenados sobre datos con estandarización empírica no quedan descritos por él, porque en
ese caso los parámetros efectivos dependen del sorteo de la muestra.
"""

from __future__ import annotations

__all__: list[str] = []
