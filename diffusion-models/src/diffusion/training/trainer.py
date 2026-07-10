"""Loop de entrenamiento por denoising score matching y persistencia de la red.

Reúne el muestreo de pares de entrenamiento (:mod:`losses`), una red de score cualquiera
(cualquier :class:`diffusion.models.ScoreModel`: el ``ScoreMLP`` de Fase 1, la ``ScoreUNet``
de Fase 2, …) y un proceso forward (:class:`diffusion.sde.ForwardSDE`) en un :func:`train`
que devuelve la red entrenada y la historia de pérdida.

:func:`train` es **agnóstico a la red y al origen de datos**: recibe la ``model`` ya
construida y un iterador **infinito** de tensores crudos, y corre un loop **por pasos**
(``config.num_steps``). No construye la red ni ramifica por su tipo — esa responsabilidad vive
en el caller (``make_model`` / el config-driven). La **regla del Eje 1** sigue vigente: cambiar
de SDE = una red nueva y un entrenamiento desde cero (los samplers del Eje 2 reusan la misma
red sin reentrenar).

:func:`save_checkpoint` / :func:`load_checkpoint` son **model-agnósticos** (R5-c): guardan el
``state_dict`` de la red junto con una metadata mínima (nombre de la SDE, ``data_dim`` y una
**receta genérica** ``model={name, kwargs}`` opcional) y devuelven ``(state_dict, meta)`` **sin
reconstruir** la red. Es el caller quien reconstruye la red (vía ``make_model`` o una instancia
explícita) y carga el ``state_dict`` — así el mismo checkpoint sirve al ``ScoreMLP`` y a la
``ScoreUNet`` sin que ``training`` importe ninguna red concreta.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import torch

from ..models import ScoreModel
from ..sde import ForwardSDE
from .losses import dsm_loss, sample_timesteps


@dataclass
class TrainConfig:
    """Hiperparámetros del **loop** de entrenamiento (solo optimización y corrida).

    Ya no carga hiperparámetros de red (viven en el constructor de la red / ``make_model``) ni
    de tamaño de dataset (viven en la fuente de datos). Pensado para construirse desde un
    archivo de config (ver :mod:`diffusion.training.config`), pero es un dataclass plano y se
    arma igual de fácil a mano en los tests.
    """

    num_steps: int = 1000  # pasos de optimización (≈ epochs × n_samples/batch_size viejos)
    lr: float = 2e-3
    t_eps: float = 1e-3
    grad_clip: float | None = None
    seed: int | None = 0
    device: str = "cpu"
    log_every: int = 0  # 0 = silencioso; N = imprime (media móvil) cada N pasos. Solo consola.
    # Checkpointing intermedio: 0 = solo el checkpoint final (comportamiento por defecto, sin
    # regresión); N>0 = además, snapshot cada N pasos y un checkpoint "best" (menor pérdida
    # vista). El *cómo/dónde* persistir lo decide el callback ``on_checkpoint`` de :func:`train`
    # (el loop no toca el filesystem); sin ese callback este campo no hace nada.
    checkpoint_every: int = 0


@dataclass
class TrainResult:
    """Resultado de :func:`train`: la red entrenada y la traza de la corrida."""

    net: ScoreModel  # cualquier red de score (ScoreMLP, ScoreUNet, …), no atada a una clase
    history: list[float] = field(default_factory=list)  # pérdida por paso (serie completa: una por paso)
    config: TrainConfig = field(default_factory=TrainConfig)
    sde_name: str = ""
    # = sde.data_dim (valor crudo): un entero para dato plano 2D o una tupla (forma de evento,
    # imágenes). Lo copia save_checkpoint a la meta y lo consume generate.py para reconstruir la SDE.
    data_dim: int | tuple[int, ...] = 0


@dataclass
class ResumeState:
    """Estado en memoria necesario para **reanudar** una corrida desde un checkpoint intermedio.

    Agrupa lo que :func:`train` no puede recuperar de los pesos: el estado del optimizador (los
    momentos de Adam), el número de paso ya alcanzado y el **azar** del loop —el RNG global de
    torch y el estado del ``generator`` del ruido del kernel / muestreo de ``t``—. El ``history``
    viaja acá en memoria (para poder continuar la curva al reanudar), pero **no** se persiste en el
    sidecar: ya vive en el ``meta`` del checkpoint de pesos (:func:`save_checkpoint`) y duplicarlo
    sería redundante (ver :func:`save_resume_state`); al cargar se rellena desde ese ``meta``.

    Attributes:
        optimizer_state: ``optimizer.state_dict()`` (estado de Adam).
        start_step: Pasos ya completados (= ``N`` del nombre ``…_stepNNNNN``).
        torch_rng_state: ``torch.get_rng_state()`` (RNG global de torch).
        generator_state: ``generator.get_state()`` (RNG del ruido del kernel / muestreo de ``t``).
        history: Pérdida per-step hasta ``start_step`` (se rellena desde el ``meta`` al cargar).
    """

    optimizer_state: dict
    start_step: int
    torch_rng_state: torch.Tensor
    generator_state: torch.Tensor
    history: list[float]


@dataclass
class TrainSnapshot:
    """Envoltorio del estado que viaja junto a los pesos en cada checkpoint intermedio.

    Agrupa el :class:`TrainResult` (pesos + history, lo que consume :func:`save_checkpoint`) y el
    :class:`ResumeState` (lo que consume :func:`save_resume_state` para el sidecar), de modo que el
    caller pueda persistir ambos artefactos —el checkpoint de pesos y el sidecar de resume— desde
    un único snapshot del loop.

    Attributes:
        result: Pesos + history del punto de checkpoint (para el checkpoint de pesos).
        resume: Estado para reanudar (para el sidecar ``…_resume.pt``).
    """

    result: TrainResult
    resume: ResumeState


def train(
    sde: ForwardSDE,
    model: ScoreModel,
    data: Iterator[torch.Tensor],
    config: TrainConfig,
    *,
    generator: torch.Generator | None = None,
    on_checkpoint: Callable[[str, TrainSnapshot], None] | None = None,
    resume: ResumeState | None = None,
) -> TrainResult:
    """Entrena la red ``model`` para aproximar el score de ``sde`` por DSM.

    Loop **por pasos** y agnóstico a la red: usa la ``model`` recibida (no construye ninguna ni
    ramifica por su tipo) y consume ``data`` con ``next()`` — un batch por paso.

    Es **reanudable**: sin ``resume`` entrena desde cero (paso 0, optimizador nuevo, azar
    sembrado con ``config.seed``); con un :class:`ResumeState` continúa una corrida previa —
    restaura el optimizador y el azar, arranca en el paso guardado y sigue el ``history``— hasta
    completar ``config.num_steps`` (interpretado como el **total** a alcanzar, no como pasos
    adicionales).

    Args:
        sde: Proceso forward (define el kernel y el target del score). Su ``data_dim`` queda
            registrado en el resultado (lo usa el checkpoint).
        model: Red de score ya construida (cualquier :class:`ScoreModel`). Se mueve al
            dispositivo de forma idempotente y se pone en modo entrenamiento. Al reanudar, el
            caller ya le cargó los pesos del checkpoint elegido.
        data: Iterador/iterable **infinito** que yield-ea tensores crudos ``(B, ...)`` (p. ej.
            ``infinite_bare(distribution.dataloader(...))``). Se le pide un batch por paso.
        config: Hiperparámetros del loop de entrenamiento. ``num_steps`` es el **total** de la
            corrida (al reanudar se corren solo los pasos restantes).
        generator: Generador opcional para el ruido del kernel / muestreo de ``t``. Sin
            ``resume`` se crea (si es ``None``) sembrado con ``config.seed``; con ``resume`` se
            crea si hace falta el objeto pero su estado se **restaura** desde el ``ResumeState``
            (no se re-siembra).
        on_checkpoint: Callback **opcional** de checkpointing intermedio. Se invoca con
            ``(tag, snapshot)`` donde ``tag`` es ``"step{N:05d}"`` (snapshot periódico cada
            ``config.checkpoint_every`` pasos) o ``"best"`` (nueva pérdida mínima de intervalo),
            y ``snapshot`` es un :class:`TrainSnapshot` — el :class:`TrainResult` con la red en
            ese punto **más** el :class:`ResumeState` (optimizador + paso + azar + history) para
            poder reanudar. El loop decide **cuándo** llamar; el callback decide **cómo/dónde**
            persistir — ``train`` no toca el filesystem. Sin este callback (o con
            ``checkpoint_every=0``) solo se entrena; el checkpoint final lo guarda el caller con
            :func:`save_checkpoint`.
        resume: Estado de resume **opcional**. Si es ``None`` (default) el loop entrena desde
            cero (sin regresión). Si se provee, restaura el optimizador (``load_state_dict``) y el
            azar (``torch.set_rng_state`` + ``generator.set_state``) **sin re-sembrar**, arranca
            en ``resume.start_step``, continúa ``resume.history`` e itera
            ``range(start_step, num_steps)``. Si ``start_step >= num_steps`` no corre ningún paso
            y devuelve el resultado ya completo.

    Returns:
        :class:`TrainResult` con la red entrenada, la historia de pérdida (**serie per-step
        completa**: una entrada por paso, ``len(history) == num_steps`` cuando
        ``start_step < num_steps``; el ``history`` previo intacto en el caso no-op), el ``config``
        usado, el nombre de la SDE y su ``data_dim``.
    """
    device = torch.device(config.device)
    if resume is None:
        # Corrida desde cero: siembra el azar con config.seed (comportamiento histórico).
        if config.seed is not None:
            torch.manual_seed(config.seed)
        if generator is None:
            generator = torch.Generator(device=device)
            if config.seed is not None:
                generator.manual_seed(config.seed)
    elif generator is None:
        # Reanudación: NO se re-siembra; el estado del generator se restaura más abajo desde el
        # ResumeState. Igual hay que crear el objeto si el caller no lo pasó.
        generator = torch.Generator(device=device)

    net = model.to(device)  # idempotente: no falla si el caller ya la movió
    net.train()

    data_iter = iter(data)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    # Estado inicial del loop: desde cero o restaurado de un ResumeState.
    if resume is None:
        start_step = 0
        history: list[float] = []
    else:
        # Restaurar optimizador y azar antes de continuar (2.1). El load_state_dict actúa además
        # de guard de compatibilidad: levanta si las shapes del optimizador no corresponden (2.5).
        optimizer.load_state_dict(resume.optimizer_state)
        torch.set_rng_state(resume.torch_rng_state)
        generator.set_state(resume.generator_state)
        start_step = resume.start_step
        history = list(resume.history)  # continuar la curva previa (2.3)

    # ``history`` guarda la pérdida de CADA paso (serie completa, la fuente de verdad). ``log_every``
    # gobierna solo el print de consola, desacoplado. Para el "best" se usa una cadencia interna
    # propia (media de ventana suavizada), también desacoplada de ``log_every``.
    eval_every = max(1, config.num_steps // 100)

    # Checkpointing intermedio (opt-in): activo solo si el caller inyecta un callback y
    # ``checkpoint_every > 0``. El loop decide *cuándo* (cadencia periódica + best por pérdida);
    # el callback decide *cómo/dónde* persistir — ``train`` no toca el filesystem.
    do_checkpoints = on_checkpoint is not None and config.checkpoint_every > 0
    best_loss = float("inf")

    def _result() -> TrainResult:
        # Foto del TrainResult actual (pesos vía la red viva + copia del history).
        return TrainResult(
            net=net,
            history=list(history),
            config=config,
            sde_name=sde.name,
            data_dim=sde.data_dim,
        )

    def _snapshot(completed_steps: int) -> TrainSnapshot:
        # Foto completa para que el callback persista pesos y sidecar: el TrainResult (pesos +
        # history) más el ResumeState del momento (optimizador + paso + azar + history). El azar
        # se lee DESPUÉS del paso, así que reanudar desde acá continúa el mismo stream (2.6).
        return TrainSnapshot(
            result=_result(),
            resume=ResumeState(
                optimizer_state=optimizer.state_dict(),
                start_step=completed_steps,  # pasos ya completados (= N del tag step{N:05d})
                torch_rng_state=torch.get_rng_state(),
                generator_state=generator.get_state(),
                history=list(history),
            ),
        )

    # ``num_steps`` es el TOTAL a alcanzar: se corren solo los pasos restantes (2.2). Si el paso
    # inicial ya lo alcanzó/superó, el rango es vacío y no se ejecuta ningún paso (no-op, 2.4).
    for step in range(start_step, config.num_steps):
        x0 = next(data_iter).to(device)
        t = sample_timesteps(
            x0.shape[0], sde.T, config.t_eps, generator=generator, device=device
        )
        loss = dsm_loss(net, sde, x0, t, generator=generator)

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
        optimizer.step()

        history.append(loss.item())  # serie completa: la pérdida de cada paso
        is_last = step == config.num_steps - 1

        # Snapshot periódico: cadencia propia, chequeada cada paso para que ``checkpoint_every``
        # no tenga que ser múltiplo de nada. El último paso lo cubre el checkpoint final del
        # caller, así que se excluye acá.
        if do_checkpoints and not is_last and (step + 1) % config.checkpoint_every == 0:
            on_checkpoint(f"step{step + 1:05d}", _snapshot(step + 1))

        # Best-so-far sobre una media de ventana (suavizada; la pérdida de un paso suelto es muy
        # ruidosa por el t aleatorio), a una cadencia interna desacoplada de log_every/history.
        if do_checkpoints and ((step + 1) % eval_every == 0 or is_last):
            window = history[-eval_every:]
            window_mean = sum(window) / len(window)
            if window_mean < best_loss:
                best_loss = window_mean
                on_checkpoint("best", _snapshot(step + 1))

        # Print de progreso (solo consola, desacoplado del history): media móvil de los últimos
        # log_every pasos, más legible que un paso suelto.
        if config.log_every > 0 and ((step + 1) % config.log_every == 0 or is_last):
            recent = history[-config.log_every:]
            print(
                f"[{sde.name}] paso {step + 1}/{config.num_steps}  "
                f"pérdida(móvil)={sum(recent) / len(recent):.6f}"
            )

    return TrainResult(
        net=net, history=history, config=config, sde_name=sde.name, data_dim=sde.data_dim
    )


# ----------------------------------------------------------------- persistencia


def save_checkpoint(
    result: TrainResult,
    path: str | pathlib.Path,
    *,
    model_spec: dict | None = None,
) -> pathlib.Path:
    """Guarda la red entrenada y su metadata **model-agnóstica** en ``path`` (``.pt`` de torch).

    El checkpoint es una receta portable: guarda el ``state_dict`` de la red y una ``meta``
    sin hiperparámetros de arquitectura hardcodeados. La receta de red (``model``) es
    **opcional** y la aporta el caller: sin ella el checkpoint sigue siendo válido, pero al
    generar habrá que pasar una red explícita (ver :func:`load_checkpoint` y
    :func:`diffusion.samplers.generate_from_checkpoint`).

    Args:
        result: Resultado de :func:`train`.
        path: Ruta de salida (se crean los directorios intermedios).
        model_spec: Receta genérica de la red ``{"name": str, "kwargs": dict}`` para poder
            reconstruirla vía :func:`diffusion.models.make_model`. Si es ``None`` no se guarda
            la clave ``model`` (la red se pasa aparte al generar).

    Returns:
        La ruta donde se guardó (como :class:`pathlib.Path`).
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "sde_name": result.sde_name,
        # = sde.data_dim (int o tupla; lo registró train); torch.save serializa la tupla sin
        # problema y generate.py la reusa para reconstruir la SDE con su forma de evento.
        "data_dim": result.data_dim,
        "history": list(result.history),
    }
    if model_spec is not None:
        meta["model"] = model_spec  # receta genérica {name, kwargs}, independiente de la clase
    blob = {"model_state": result.net.state_dict(), "meta": meta}
    torch.save(blob, out)
    return out


def load_checkpoint(
    path: str | pathlib.Path, *, map_location: torch.device | str = "cpu"
) -> tuple[dict, dict]:
    """Carga un checkpoint de :func:`save_checkpoint` como ``(state_dict, meta)``.

    **No reconstruye** ninguna red concreta (R5-c): devuelve el ``state_dict`` crudo y la
    metadata, y es el caller quien arma la red (vía :func:`diffusion.models.make_model` con la
    receta ``meta["model"]`` o una instancia propia) y le carga el ``state_dict``. Así
    ``training`` no depende de ninguna clase de red.

    Args:
        path: Ruta del ``.pt`` guardado.
        map_location: Dispositivo donde cargar los pesos (default ``"cpu"``).

    Returns:
        ``(state_dict, meta)``: el ``state_dict`` de la red y el ``dict`` de metadata guardado.

    Raises:
        KeyError: Si el archivo no tiene la forma de un checkpoint de :func:`save_checkpoint`
            (faltan ``"model_state"`` o ``"meta"``).
    """
    # weights_only=False: es nuestro propio checkpoint (incluye un dict de metadata).
    blob = torch.load(path, map_location=map_location, weights_only=False)
    return blob["model_state"], blob["meta"]


# --------------------------------------------------- sidecar de resume (C1)

# Campos obligatorios de un ``ResumeState`` a persistir: sin cualquiera de ellos la reanudación
# es imposible, así que se falla antes de escribir (1.4). El ``history`` queda fuera a propósito
# (vive en el ``meta`` del checkpoint de pesos; no se duplica, 1.3).
_REQUIRED_RESUME_FIELDS = ("optimizer_state", "start_step", "torch_rng_state", "generator_state")


def save_resume_state(
    path: str | pathlib.Path, resume: ResumeState
) -> pathlib.Path:
    """Persiste el **sidecar de resume** de un checkpoint (``torch.save``), sin el ``history``.

    El sidecar es un archivo aparte del checkpoint de pesos: guarda solo lo que ese no tiene —el
    estado del optimizador, el paso alcanzado y el azar (RNG global de torch + estado del
    ``generator``)—. El ``history`` se **omite** a propósito: ya está en el ``meta`` del checkpoint
    de pesos (:func:`save_checkpoint`) y se recupera de ahí al reanudar (1.3). El checkpoint de
    pesos no se toca ni cambia de formato/tamaño (1.2).

    Args:
        path: Ruta de salida del sidecar (se crean los directorios intermedios). Convención de
            nombre: ``X_stepNNNNN.resume.pt`` hermano de ``X_stepNNNNN.pt``.
        resume: Estado a persistir. Debe tener el optimizador, el paso y ambos estados de RNG
            presentes (no ``None``).

    Returns:
        La ruta donde se guardó (como :class:`pathlib.Path`).

    Raises:
        ValueError: Si falta alguno de ``optimizer_state`` / ``start_step`` / ``torch_rng_state``
            / ``generator_state`` (estado incompleto). No se escribe un sidecar parcial (1.4).
    """
    missing = [name for name in _REQUIRED_RESUME_FIELDS if getattr(resume, name) is None]
    if missing:
        raise ValueError(
            "Estado de resume incompleto: falta(n) "
            f"{', '.join(missing)}. No se persiste un sidecar parcial "
            "(se requieren optimizer_state, start_step, torch_rng_state y generator_state)."
        )

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "optimizer_state": resume.optimizer_state,
        "step": resume.start_step,  # el sidecar usa la clave 'step' (= start_step)
        "torch_rng_state": resume.torch_rng_state,
        "generator_state": resume.generator_state,
        # NB: 'history' NO se persiste acá (vive en el meta del checkpoint de pesos; 1.3).
    }
    torch.save(blob, out)
    return out


def load_resume_state(
    path: str | pathlib.Path, *, map_location: torch.device | str = "cpu"
) -> dict:
    """Carga un sidecar de :func:`save_resume_state` como ``dict``.

    Args:
        path: Ruta del sidecar ``…_resume.pt``.
        map_location: Dispositivo donde cargar los tensores (default ``"cpu"``).

    Returns:
        ``{optimizer_state, step, torch_rng_state, generator_state}`` (sin ``history``: se toma del
        ``meta`` del checkpoint de pesos al reanudar).
    """
    # weights_only=False: es nuestro propio archivo (incluye el state_dict del optimizador).
    return torch.load(path, map_location=map_location, weights_only=False)
