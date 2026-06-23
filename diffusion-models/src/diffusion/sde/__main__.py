"""Smoke test manual del módulo: instancia las 4 SDEs y reporta el kernel.

Correr (desde ``diffusion-models/src/``)::

    python -m diffusion.sde

Para cada SDE corre ``perturb`` sobre un batch dummy e imprime la media/escala del kernel
en ``t ≈ 0`` (debería ser ~``x0`` con escala ~0) y en ``t = T`` (debería tender al prior).
Se usa ``-m`` porque el módulo usa imports relativos y no es ejecutable como script suelto.
"""

from __future__ import annotations

import torch

from diffusion.sde import available_sdes, make_sde


def main() -> None:
    x0 = torch.randn(16, 2)  # posición (B, 2): sirve para la familia escalar y para CLD
    for name in available_sdes():
        sde = make_sde(name)
        t = torch.rand(16)
        x_t, _ = sde.perturb(x0, t)
        print(
            f"{name:7s} data_dim={sde.data_dim} augmented={sde.is_augmented}"
            f"  perturb -> {tuple(x_t.shape)}"
        )
        for tv in (1e-3, sde.T):
            tt = torch.full((16,), float(tv))
            mean, scale = sde.marginal_prob(x0, tt)
            if sde.is_augmented:
                # scale es el Cholesky L (B, 2, 2): reporto los desvíos por dimensión.
                std_x = scale[0, 0, 0].item()
                std_v = (scale[0, 1, 0] ** 2 + scale[0, 1, 1] ** 2).sqrt().item()
                print(
                    f"    t={tv:<6.3g} |mean|~{mean.abs().mean():.3f}"
                    f"  std_x~{std_x:.3f} std_v~{std_v:.3f}"
                )
            else:
                print(
                    f"    t={tv:<6.3g} |mean|~{mean.abs().mean():.3f}"
                    f"  std~{scale.mean():.3f}"
                )


if __name__ == "__main__":
    main()
