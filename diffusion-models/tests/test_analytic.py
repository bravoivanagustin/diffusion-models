"""Tests de la verdad analítica del laboratorio 2D (`diffusion.analytic`)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import diffusion.analytic


# --------------------------------------------------------------- paquete y fronteras


def test_paquete_importable_desde_la_ruta_publica():
    """El módulo se importa como `diffusion.analytic` y publica un `__all__`."""
    assert diffusion.analytic.__name__ == "diffusion.analytic"
    assert isinstance(diffusion.analytic.__all__, list)
    assert all(
        hasattr(diffusion.analytic, nombre) for nombre in diffusion.analytic.__all__
    )


def test_no_importa_samplers_ni_training():
    """Dirección de dependencias `data_generation → sde → analytic` (criterio 2.4).

    Se chequea en un intérprete nuevo porque `sys.modules` del proceso de pytest ya
    tiene cargados los módulos que importan las otras suites.
    """
    src = Path(diffusion.analytic.__file__).resolve().parents[2]
    codigo = (
        "import sys, diffusion.analytic; "
        "print([m for m in ('diffusion.samplers', 'diffusion.training') "
        "if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
