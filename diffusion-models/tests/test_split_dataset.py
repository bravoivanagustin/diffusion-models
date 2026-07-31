"""Tests de la herramienta de split determinístico del dataset (``scripts/split_dataset.py``).

El script no es un módulo del paquete ``diffusion``: se carga **por ruta** con ``importlib``, la
convención ya usada para ``scripts/train.py`` y ``scripts/sample.py`` en ``test_resume.py`` y
``test_samplers.py``.

Esta suite corre sin torch: el reparto (``split_paths``) es una función pura de la stdlib.
"""

from __future__ import annotations

import importlib.util
import pathlib
import random

import pytest

_SPLIT_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "split_dataset.py"


def _load_split_script():
    """Carga ``scripts/split_dataset.py`` por ruta y devuelve el módulo."""
    spec = importlib.util.spec_from_file_location("_split_dataset_under_test", _SPLIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(n: int, root: str = "data/cats") -> list[pathlib.Path]:
    """``n`` rutas sintéticas, **ya ordenadas** (precondición de ``split_paths``)."""
    return sorted(pathlib.Path(root) / f"{i:05d}.jpg" for i in range(n))


# =============================================================================
# task 1.1 — Reparto determinístico como función pura
# =============================================================================
#
# Contrato (design, "Components and Interfaces → Herramienta → DatasetSplitter"):
#     split_paths(paths, val_frac, seed) -> (train, val)
# Precondiciones: ``paths`` no vacío y ya ordenado; 0 < val_frac < 1.
# Postcondiciones: len(val) == round(len(paths) * val_frac) (1.2); (train, val) es una partición
# de ``paths`` con ambas partes no vacías; el resultado depende solo de (paths, val_frac, seed).


# ---------------------------------------------------------------- determinismo (1.3, 1.4)


def test_misma_seed_da_particion_identica():
    # 1.3: dos llamadas con la misma seed, el mismo frac y el mismo conjunto ⇒ mismo reparto.
    mod = _load_split_script()
    paths = _paths(50)

    train_a, val_a = mod.split_paths(paths, 0.2, 7)
    train_b, val_b = mod.split_paths(paths, 0.2, 7)

    assert train_a == train_b
    assert val_a == val_b


def test_seeds_distintas_dan_particiones_distintas():
    # 1.4: con al menos 20 archivos, dos seeds distintas ⇒ particiones distintas.
    mod = _load_split_script()
    paths = _paths(20)

    _, val_0 = mod.split_paths(paths, 0.3, 0)
    _, val_1 = mod.split_paths(paths, 0.3, 1)

    assert set(val_0) != set(val_1)


def test_no_usa_el_rng_global():
    # 1.3 (mecanismo): el azar sale de un RNG sembrado propio, así que el estado del RNG global
    # no se mueve y el reparto no depende de quién sorteó antes.
    mod = _load_split_script()
    paths = _paths(30)

    random.seed(1234)
    estado_previo = random.getstate()
    train, val = mod.split_paths(paths, 0.25, 3)
    assert random.getstate() == estado_previo

    # Y el reparto es el mismo aunque el RNG global esté en otro estado.
    random.seed(9999)
    assert (train, val) == mod.split_paths(paths, 0.25, 3)


def test_no_muta_la_lista_de_entrada():
    # El reparto es una función pura: la lista recibida queda intacta.
    mod = _load_split_script()
    paths = _paths(12)
    copia = list(paths)

    mod.split_paths(paths, 0.25, 0)

    assert paths == copia


# ---------------------------------------------------------------- partición y conteo (1.2)


@pytest.mark.parametrize(
    ("n", "val_frac"),
    [(10, 0.1), (10, 0.2), (100, 0.1), (37, 0.15), (20, 0.5), (7, 0.3), (999, 0.05)],
)
def test_conteo_de_val_por_redondeo(n, val_frac):
    # 1.2: a val le toca la cantidad que resulta de redondear el porcentaje sobre el total,
    # y el resto a train.
    mod = _load_split_script()
    paths = _paths(n)

    train, val = mod.split_paths(paths, val_frac, 0)

    assert len(val) == round(n * val_frac)
    assert len(train) == n - len(val)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_union_reconstruye_la_entrada_sin_interseccion(seed):
    # 1.1 (parte pura): (train, val) es una partición del conjunto de entrada — ningún archivo
    # se duplica y ninguno se pierde.
    mod = _load_split_script()
    paths = _paths(40)

    train, val = mod.split_paths(paths, 0.25, seed)

    assert set(train) | set(val) == set(paths)
    assert not set(train) & set(val)
    assert len(train) + len(val) == len(paths)
    assert train and val


def test_reparto_de_rutas_con_subcarpetas():
    # Las rutas son opacas para el reparto: un dataset con subcarpetas se reparte igual.
    mod = _load_split_script()
    paths = sorted(
        [pathlib.Path("src") / "a" / f"{i}.jpg" for i in range(10)]
        + [pathlib.Path("src") / "b" / f"{i}.png" for i in range(10)]
    )

    train, val = mod.split_paths(paths, 0.2, 5)

    assert len(val) == 4
    assert set(train) | set(val) == set(paths)
    assert all(isinstance(p, pathlib.Path) for p in train + val)


# ---------------------------------------------------------------- rechazos (1.9, 1.10)


@pytest.mark.parametrize("val_frac", [0.0, 1.0, -0.1, 1.5, 2.0])
def test_val_frac_fuera_del_intervalo_abierto_falla(val_frac):
    # 1.9: fuera del intervalo abierto (0, 1) ⇒ ValueError informando el valor recibido.
    mod = _load_split_script()
    paths = _paths(20)

    with pytest.raises(ValueError) as exc:
        mod.split_paths(paths, val_frac, 0)

    assert repr(val_frac) in str(exc.value) or str(val_frac) in str(exc.value)


def test_val_vacia_falla_informando_total_y_porcentaje():
    # 1.10: 3 imágenes con 10% ⇒ round(0.3) == 0 ⇒ val quedaría vacía.
    mod = _load_split_script()

    with pytest.raises(ValueError) as exc:
        mod.split_paths(_paths(3), 0.1, 0)

    mensaje = str(exc.value)
    assert "3" in mensaje
    assert "0.1" in mensaje


def test_train_vacia_falla_informando_total_y_porcentaje():
    # 1.10: 2 imágenes con 90% ⇒ round(1.8) == 2 ⇒ train quedaría vacía.
    mod = _load_split_script()

    with pytest.raises(ValueError) as exc:
        mod.split_paths(_paths(2), 0.9, 0)

    mensaje = str(exc.value)
    assert "2" in mensaje
    assert "0.9" in mensaje


def test_lista_vacia_falla():
    # Precondición "paths no vacío": sin archivos no hay reparto posible (1.10/1.11).
    mod = _load_split_script()

    with pytest.raises(ValueError):
        mod.split_paths([], 0.2, 0)
