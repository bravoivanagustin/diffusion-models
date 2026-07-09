"""CLI para entrenar la red de score por denoising score matching desde un config YAML.

Cada corrida (una celda del estudio de ablación) se describe en un ``.yaml`` con secciones
``sde`` / ``data`` / ``train`` / ``model`` (opcional) / ``out``. Ver ``config/vp_mixture.yaml``.

Ejemplos (correr desde ``diffusion-models/``)::

    python scripts/train.py --config config/vp_mixture.yaml
    python scripts/train.py --config config/vp_mixture.yaml --num-steps 50 --device cpu
    python scripts/train.py --config config/vp_mixture.yaml --checkpoint-every 50

Guarda los pesos entrenados (``.pt`` con ``state_dict`` + metadata) y una curva de pérdida
(``.png``) en las rutas de la sección ``out`` del config (relativas al cwd). Con
``train.checkpoint_every > 0`` (o ``--checkpoint-every``) guarda además, junto al checkpoint
final, un snapshot periódico ``…_stepNNNNN.pt`` cada N pasos y un ``…_best.pt`` con la menor
pérdida vista.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Permitir ejecutar el script sin instalar el paquete (agrega ./src al path).
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diffusion.training import build_run, load_config, save_checkpoint, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entrena la red de score (DSM) a partir de un config YAML."
    )
    p.add_argument("--config", required=True, help="Ruta del .yaml de la corrida.")
    p.add_argument("--num-steps", type=int, default=None,
                   help="Override de la cantidad de pasos de entrenamiento del config.")
    p.add_argument("--device", type=str, default=None,
                   help="Override del dispositivo (p. ej. cpu / cuda).")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="Override de cada cuántos pasos guardar un snapshot intermedio "
                        "(0 = solo el checkpoint final; requiere 'out.checkpoint').")
    p.add_argument("--quiet", action="store_true",
                   help="No imprimir el progreso por paso.")
    return p


def save_loss_curve(path: str | pathlib.Path, history: list[float], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history, linewidth=0.7, alpha=0.8)
    if min(history) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("paso")
    ax.set_ylabel("pérdida DSM")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    # En Windows (py<3.15) la consola no usa UTF-8 por defecto; forzarlo evita
    # mojibake en los acentos de los mensajes.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)

    try:
        spec = build_run(load_config(args.config))
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Overrides de la línea de comandos.
    if args.num_steps is not None:
        spec.config.num_steps = args.num_steps
    if args.device is not None:
        spec.config.device = args.device
    if args.checkpoint_every is not None:
        spec.config.checkpoint_every = args.checkpoint_every
    if args.quiet:
        spec.config.log_every = 0
    elif spec.config.log_every == 0:
        spec.config.log_every = max(1, spec.config.num_steps // 10)

    # Callback de checkpointing intermedio: solo si hay una ruta base de checkpoint. Deriva
    # rutas hermanas (…_stepNNNNN.pt / …_best.pt) del checkpoint final y reusa save_checkpoint;
    # así train() sigue sin tocar el filesystem (decide *cuándo*, esto decide *dónde*).
    on_checkpoint = None
    if spec.checkpoint is not None:
        base = spec.checkpoint

        def on_checkpoint(tag, snapshot):
            tagged = base.with_stem(f"{base.stem}_{tag}")
            save_checkpoint(snapshot, tagged, model_spec=spec.model_spec)
            print(f"Checkpoint ({tag}) -> {tagged}")
    elif spec.config.checkpoint_every > 0:
        print(
            "nota: 'train.checkpoint_every' > 0 pero falta 'out.checkpoint'; "
            "no se guardarán snapshots intermedios."
        )

    print(
        f"Entrenando sde={spec.sde.name} (data_dim={spec.sde.data_dim}) "
        f"con {type(spec.model).__name__}: pasos={spec.config.num_steps} "
        f"device={spec.config.device}"
    )
    result = train(spec.sde, spec.model, spec.data, spec.config, on_checkpoint=on_checkpoint)
    hist = result.history
    k = max(1, len(hist) // 20)  # media de extremos: la pérdida per-step es ruidosa
    ini, fin = sum(hist[:k]) / k, sum(hist[-k:]) / k
    print(
        f"Listo. pérdida inicial≈{ini:.6f} -> final≈{fin:.6f}  "
        f"(medias de {k} pasos; {len(hist)} pasos guardados)"
    )

    if spec.checkpoint:
        save_checkpoint(result, spec.checkpoint, model_spec=spec.model_spec)
        print(f"Checkpoint -> {spec.checkpoint}")
    if spec.loss_curve:
        save_loss_curve(
            spec.loss_curve, result.history, f"{spec.sde.name} · {type(spec.model).__name__}"
        )
        print(f"Curva      -> {spec.loss_curve}")
    if not spec.checkpoint and not spec.loss_curve:
        print("(sin 'out.checkpoint' ni 'out.loss_curve': no se guardó nada)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
