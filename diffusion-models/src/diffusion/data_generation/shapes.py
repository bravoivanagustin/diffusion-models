"""Formas concretas de distribuciones de puntos.

Se apoya en scikit-learn donde existe un generador adecuado (``make_moons``,
``make_blobs``, ``make_swiss_roll``) y en numpy para la espiral (sklearn no trae
generador de espiral).
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

from .base import PointDistribution


def _seed_from(rng: np.random.Generator) -> int:
    """Deriva un ``random_state`` int para sklearn, atado al rng reproducible."""
    return int(rng.integers(0, 2**31 - 1))


def _split_counts(n: int, parts: int) -> list[int]:
    """Reparte ``n`` en ``parts`` enteros lo más parejos posible (suman ``n``)."""
    base, rem = divmod(n, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


class Gaussian(PointDistribution):
    """Gaussiana isotrópica N-dim centrada en el origen."""

    name = "gaussian"
    supported_dims = None  # cualquier dim >= 1

    def __init__(self, dim, *, scale=1.0, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=0.0, seed=seed)
        self.scale = float(scale)

    def _sample_raw(self, n, rng):
        return rng.normal(0.0, self.scale, size=(n, self.dim))


class GaussianMixture(PointDistribution):
    """Mezcla de gaussianas isotrópicas.

    En 2D los centros se ubican en un anillo (el clásico "8 gaussianas"); en
    otras dimensiones, en direcciones aleatorias sobre una hiperesfera de radio
    ``radius``.
    """

    name = "mixture"
    supported_dims = None

    def __init__(self, dim, *, n_components=8, cluster_std=0.3, radius=5.0,
                 standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=0.0, seed=seed)
        if n_components < 1:
            raise ValueError(f"n_components debe ser >= 1; recibí {n_components}")
        self.n_components = int(n_components)
        self.cluster_std = float(cluster_std)
        self.radius = float(radius)

    def _centers(self, rng):
        k = self.n_components
        if self.dim == 2:
            ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
            return self.radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        dirs = rng.normal(size=(k, self.dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return self.radius * dirs

    def _sample_raw(self, n, rng):
        centers = self._centers(rng)
        x, y = make_blobs(
            n_samples=n,
            centers=centers,
            cluster_std=self.cluster_std,
            random_state=_seed_from(rng),
        )
        self.color_ = y
        return x


class TwoMoons(PointDistribution):
    """Dos medias lunas entrelazadas (2D). Vía ``sklearn.make_moons``."""

    name = "two_moons"
    supported_dims = frozenset({2})

    def __init__(self, dim=2, *, noise=0.05, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)

    def _sample_raw(self, n, rng):
        x, y = make_moons(n_samples=n, noise=self.noise, random_state=_seed_from(rng))
        self.color_ = y
        return x


class Spiral(PointDistribution):
    """Una o más espirales entrelazadas (2D). Implementada con numpy."""

    name = "spiral"
    supported_dims = frozenset({2})

    def __init__(self, dim=2, *, noise=0.02, n_arms=2, turns=1.5,
                 standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)
        if n_arms < 1:
            raise ValueError(f"n_arms debe ser >= 1; recibí {n_arms}")
        self.n_arms = int(n_arms)
        self.turns = float(turns)

    def _sample_raw(self, n, rng):
        max_t = self.turns * 2.0 * np.pi
        arms, labels = [], []
        for i, k in enumerate(_split_counts(n, self.n_arms)):
            # sqrt(U) => densidad ~uniforme en área a lo largo del radio.
            t = np.sqrt(rng.uniform(0.0, 1.0, size=k)) * max_t
            phase = i * (2.0 * np.pi / self.n_arms)
            xy = np.stack([t * np.cos(t + phase), t * np.sin(t + phase)], axis=1)
            xy = xy / max_t  # normalizar a ~[-1, 1]
            xy = xy + rng.normal(0.0, self.noise, size=(k, 2))
            arms.append(xy)
            labels.append(np.full(k, i))
        out = np.concatenate(arms, axis=0)
        lab = np.concatenate(labels, axis=0)
        perm = rng.permutation(len(out))
        self.color_ = lab[perm]
        return out[perm]


class SwissRoll(PointDistribution):
    """Rollo suizo: variedad 2D embebida en 3D. Vía ``sklearn.make_swiss_roll``."""

    name = "swiss_roll"
    supported_dims = frozenset({3})

    def __init__(self, dim=3, *, noise=0.5, standardize=False, seed=None):
        super().__init__(dim, standardize=standardize, noise=noise, seed=seed)

    def _sample_raw(self, n, rng):
        x, t = make_swiss_roll(
            n_samples=n, noise=self.noise, random_state=_seed_from(rng)
        )
        self.color_ = t
        return x
