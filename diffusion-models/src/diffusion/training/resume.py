"""Resolución de reanudación (*resume*) de una corrida de entrenamiento.

Helper **puro y testeable** (sin torch, sin entrenar): decide —a partir de la ruta del
checkpoint final del ``.yaml``— si una corrida ya está **completa** (``skip``), si debe
**reanudar** desde un snapshot intermedio (``resume``) o si empieza **de cero** (``fresh``), y
descubre los snapshots intermedios que dejó una corrida previa.

Convención de nombres (la del CLI ``scripts/train.py``): el checkpoint final es ``X.pt`` y sus
snapshots hermanos son ``X_stepNNNNN.pt`` (periódicos) / ``X_best.pt`` (mejor pérdida); cada
snapshot periódico lleva además un *sidecar* de resume ``X_stepNNNNN.resume.pt`` (el estado del
optimizador + paso + azar; ver :mod:`diffusion.training.trainer`).

Este módulo solo mira el filesystem por **existencia/glob** (no lo muta) y no importa torch: es
la **política** de decisión. La carga y validación del punto elegido (``load_resume`` /
``validate_compatible``, que sí tocan torch y el ``trainer``) llegan en una tarea posterior.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

# Sufijo de un snapshot periódico: ``…_stepNNNNN.pt``. El grupo captura el entero del paso. Se
# ancla al final (``$``) para NO matchear los sidecars ``…_stepNNNNN.resume.pt`` (tienen
# ``.resume.pt`` tras los dígitos, no ``.pt``). Cambiar esta convención → revalidar
# ``discover_snapshots`` (ver Revalidation Triggers del diseño).
_STEP_RE = re.compile(r"_step(\d+)\.pt$")

# Sufijo del sidecar de resume hermano de un checkpoint de pesos ``X.pt`` → ``X.resume.pt``.
_RESUME_SUFFIX = ".resume.pt"


@dataclass
class ResumePlan:
    """Decisión de reanudación (DTO puro, sin persistencia).

    Attributes:
        action: ``"skip"`` (corrida ya completa), ``"fresh"`` (desde cero) o ``"resume"``
            (continuar desde ``weights_path``).
        weights_path: Checkpoint de pesos desde el que reanudar (solo en ``"resume"``).
        step: Paso ya completado del checkpoint elegido (solo en ``"resume"``; ``None`` si no se
            pudo parsear del nombre).
    """

    action: str
    weights_path: pathlib.Path | None = None
    step: int | None = None


def resume_sidecar_path(weights_path: pathlib.Path) -> pathlib.Path:
    """Devuelve la ruta del *sidecar* de resume hermano de un checkpoint de pesos.

    Convención: ``X_stepNNNNN.pt`` → ``X_stepNNNNN.resume.pt`` (mismo directorio y stem, con el
    sufijo ``.pt`` reemplazado por ``.resume.pt``).

    Args:
        weights_path: Ruta del checkpoint de pesos (``…_stepNNNNN.pt``).

    Returns:
        La ruta del sidecar ``…_stepNNNNN.resume.pt``.
    """
    return pathlib.Path(weights_path).with_suffix(_RESUME_SUFFIX)


def discover_snapshots(
    final_checkpoint: pathlib.Path,
) -> list[tuple[int, pathlib.Path]]:
    """Descubre los snapshots periódicos hermanos de un checkpoint final, ordenados por paso.

    Busca en el directorio del checkpoint final los archivos ``{stem}_stepNNNNN.pt`` (donde
    ``stem`` es el nombre del final sin extensión) y parsea el entero del paso. **Excluye** el
    checkpoint final mismo, el ``{stem}_best.pt`` y los sidecars ``{stem}_stepNNNNN.resume.pt``;
    y solo considera snapshots del **mismo** ``stem`` (no los de otras corridas del directorio).

    Args:
        final_checkpoint: Ruta del checkpoint final ``X.pt`` de la corrida (puede no existir; se
            usa solo para derivar ``stem`` y el directorio donde buscar).

    Returns:
        Lista de ``(step, path)`` ordenada **ascendente** por ``step``. Vacía si el directorio no
        existe o no hay snapshots.
    """
    final = pathlib.Path(final_checkpoint)
    parent = final.parent
    if not parent.is_dir():
        return []  # no hay dónde buscar

    # Ancla al stem del final: ``{stem}_step(\d+)\.pt`` (fullmatch) → no cuela snapshots de otras
    # corridas del mismo directorio ni el final/best/sidecar.
    pattern = re.compile(re.escape(final.stem) + r"_step(\d+)\.pt$")
    snaps: list[tuple[int, pathlib.Path]] = []
    for entry in parent.iterdir():
        m = pattern.fullmatch(entry.name)
        if m is not None:
            snaps.append((int(m.group(1)), entry))
    snaps.sort(key=lambda pair: pair[0])
    return snaps


def resolve_resume(
    final_checkpoint: pathlib.Path | None,
    *,
    force: bool = False,
    resume_from: str | None = None,
) -> ResumePlan:
    """Decide la acción de resume (``skip`` / ``fresh`` / ``resume``) a partir del config.

    Sigue la resolución del CLI:

    1. Si se pasa ``resume_from`` (ruta o número de paso) se resuelve a un snapshot puntual y se
       reanuda desde ahí (el pedido explícito manda sobre el skip automático) (3.5); si no se
       puede resolver → ``ValueError`` que lista los snapshots disponibles (3.7).
    2. Si no, y el ``final_checkpoint`` existe y no hay ``force`` → ``skip`` (corrida completa)
       (3.1).
    3. Si no (final ausente **o** ``force``): se descubren los snapshots; si hay alguno → ``resume``
       desde el más nuevo (mayor paso) (3.2 con ``force`` / 3.3); si no hay ninguno → ``fresh``
       (3.4).

    ``final_checkpoint=None`` (el ``.yaml`` no define ``out.checkpoint``) no tiene dónde saltear ni
    buscar → ``fresh``.

    Este helper **no muta** el filesystem: solo consulta existencia y lista snapshots.

    Args:
        final_checkpoint: Ruta del checkpoint final del ``.yaml`` (``spec.checkpoint``) o ``None``.
        force: Si es ``True`` se saltea el chequeo del final (se reentrena/reanuda igual) (3.2).
        resume_from: Selector explícito del checkpoint a reanudar: una **ruta** a un snapshot
            existente o un **número de paso** que matchee un snapshot descubierto (3.5).

    Returns:
        El :class:`ResumePlan` con la acción resuelta.

    Raises:
        ValueError: Si ``resume_from`` no resuelve a ningún snapshot (ruta o paso inexistente); el
            mensaje lista los snapshots disponibles (3.7).
    """
    final = pathlib.Path(final_checkpoint) if final_checkpoint is not None else None

    # 1) Selección explícita: manda sobre el skip automático.
    if resume_from is not None:
        weights_path, step = _resolve_resume_from(final, resume_from)
        return ResumePlan("resume", weights_path=weights_path, step=step)

    # 2) Final ya presente (sin force) → corrida completa.
    if final is not None and final.exists() and not force:
        return ResumePlan("skip")

    # 3) Final ausente o force: reanudar desde el más nuevo si hay snapshots; si no, desde cero.
    snaps = discover_snapshots(final) if final is not None else []
    if snaps:
        step, weights_path = snaps[-1]  # más nuevo = mayor paso (lista ASC)
        return ResumePlan("resume", weights_path=weights_path, step=step)
    return ResumePlan("fresh")


def _resolve_resume_from(
    final: pathlib.Path | None, resume_from: str
) -> tuple[pathlib.Path, int | None]:
    """Resuelve ``resume_from`` (ruta o paso) a ``(weights_path, step)``.

    Se intenta primero como **ruta** existente; si no, como **número de paso** contra los
    snapshots descubiertos. Si ninguno resuelve → ``ValueError`` listando los disponibles (3.7).
    """
    snaps = discover_snapshots(final) if final is not None else []

    # (a) ¿es una ruta a un snapshot existente?
    candidate = pathlib.Path(resume_from)
    if candidate.exists():
        return candidate, _parse_step(candidate.name)

    # (b) ¿es un número de paso que matchea un snapshot descubierto?
    if resume_from.isdigit():
        wanted = int(resume_from)
        for step, path in snaps:
            if step == wanted:
                return path, step

    # (c) no resuelve → error accionable que lista los disponibles.
    raise ValueError(_unresolved_message(resume_from, snaps))


def _parse_step(name: str) -> int | None:
    """Extrae el entero del paso de un nombre ``…_stepNNNNN.pt`` (``None`` si no matchea)."""
    m = _STEP_RE.search(name)
    return int(m.group(1)) if m is not None else None


def _unresolved_message(
    resume_from: str, snaps: list[tuple[int, pathlib.Path]]
) -> str:
    """Arma el mensaje de error de ``--resume-from`` inexistente, listando los disponibles (3.7)."""
    if snaps:
        steps = ", ".join(str(step) for step, _ in snaps)
        disponibles = f"Snapshots disponibles (por paso): [{steps}]."
    else:
        disponibles = "No hay snapshots disponibles para reanudar."
    return (
        f"No se pudo resolver --resume-from={resume_from!r} "
        f"(no es una ruta existente ni un paso descubierto). {disponibles}"
    )
