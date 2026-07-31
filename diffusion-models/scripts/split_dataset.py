"""Herramienta de split determinístico de un dataset de imágenes en ``train/`` y ``val/``.

Parte una carpeta de imágenes en dos conjuntos disjuntos a partir de un porcentaje y una seed,
de modo que el conjunto held-out quede fijo, reproducible y documentado por esa seed.

El **reparto** (:func:`split_paths`) está separado de la E/S a propósito: es una función pura de
la stdlib —sin torch, sin disco— que depende solo de ``(paths, val_frac, seed)``. El
descubrimiento de las imágenes ordena las rutas antes de llamarla, así que la partición nunca
depende del orden en que el filesystem las entregue.
"""

from __future__ import annotations

import pathlib
import random


def _validar_frac(val_frac: float) -> None:
    """Verifica que el porcentaje de validación caiga en el intervalo abierto (0, 1).

    Args:
        val_frac: Fracción del total que va a ``val/``.

    Raises:
        ValueError: Si ``val_frac`` no está estrictamente entre 0 y 1, informando el valor
            recibido.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(
            f"val_frac debe caer en el intervalo abierto (0, 1); se recibió {val_frac!r}."
        )


def _split_indices(n_total: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Reparte los índices ``0..n_total-1`` en (train, val) permutándolos con un RNG sembrado.

    A ``val`` le toca la cantidad que resulta de redondear ``val_frac`` sobre el total, y a
    ``train`` el resto. La permutación usa una instancia propia de :class:`random.Random`, no el
    RNG global: el resultado depende solo de ``(n_total, val_frac, seed)`` y evaluar el split no
    consume azar de nadie más.

    Args:
        n_total: Cantidad de elementos a repartir.
        val_frac: Fracción del total que va a ``val`` (ya validada).
        seed: Semilla de la permutación.

    Returns:
        Las dos listas de índices, cada una en orden creciente.

    Raises:
        ValueError: Si el reparto dejaría vacía cualquiera de las dos partes, informando el total
            de imágenes disponibles y el porcentaje pedido.
    """
    n_val = round(n_total * val_frac)
    if n_val == 0 or n_val >= n_total:
        raise ValueError(
            f"el reparto de {n_total} imágenes con val_frac={val_frac!r} dejaría vacía una de las "
            f"dos partes (val={n_val}, train={n_total - n_val}); se necesitan más imágenes o otro "
            f"porcentaje."
        )

    orden = list(range(n_total))
    random.Random(seed).shuffle(orden)
    return sorted(orden[n_val:]), sorted(orden[:n_val])


def split_paths(
    paths: list[pathlib.Path], val_frac: float, seed: int
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """Reparte una lista **ordenada** de rutas en (train, val) de forma determinística.

    Función pura: no toca el disco ni muta la lista recibida. ``(train, val)`` es una partición de
    ``paths`` —la unión reconstruye la entrada y la intersección es vacía— y ambas partes quedan
    no vacías. Dos llamadas con la misma seed, el mismo porcentaje y el mismo conjunto de archivos
    devuelven exactamente el mismo reparto.

    Args:
        paths: Rutas a repartir, no vacía y **ya ordenada** (el orden fija el determinismo).
        val_frac: Fracción del total que va a validación, en el intervalo abierto (0, 1).
        seed: Semilla de la permutación.

    Returns:
        La tupla ``(train, val)``, cada lista en el mismo orden relativo que ``paths``, con
        ``len(val) == round(len(paths) * val_frac)``.

    Raises:
        ValueError: Si ``val_frac`` está fuera de (0, 1), o si el reparto dejaría vacía a
            ``train`` o a ``val``.
    """
    _validar_frac(val_frac)
    idx_train, idx_val = _split_indices(len(paths), val_frac, seed)
    return [paths[i] for i in idx_train], [paths[i] for i in idx_val]
