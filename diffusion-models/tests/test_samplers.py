"""Tests del proceso reverso (`diffusion.samplers`).

Task 1.1: paquete importable + esqueleto del ABC ``ReverseSampler`` (grilla temporal,
drifts reversos compartidos, guarda contra SDEs aumentadas). El driver ``sample()`` y los
samplers concretos llegan en tasks posteriores; acá solo se valida el contrato base.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.samplers.base import ReverseSampler
from diffusion.sde import make_sde

B = 16


def _zero_score(x, t):
    """``ScoreFn`` trivial: score nulo, shape igual a ``x``."""
    return torch.zeros_like(x)


class _NoOpSampler(ReverseSampler):
    """Subclase mínima para instanciar el ABC: ``step`` no-op."""

    name = "noop"

    def step(self, x, t, dt, *, generator):
        return x


# --------------------------------------------------------------- importabilidad


def test_package_importable():
    # El import a tope del módulo ya prueba el observable principal de 1.1.
    assert ReverseSampler is not None


# ------------------------------------------------------------- construcción / guardas


def test_instantiate_with_scalar_sde():
    sde = make_sde("vp")
    s = _NoOpSampler(sde, _zero_score)
    assert s.sde is sde
    assert s.score_fn is _zero_score


def test_augmented_sde_rejected():
    # CLD (estado aumentado) está fuera de alcance de esta iteración.
    sde = make_sde("cld")
    with pytest.raises((ValueError, NotImplementedError)):
        _NoOpSampler(sde, _zero_score)


def test_n_steps_must_be_positive():
    sde = make_sde("vp")
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, n_steps=0)


def test_t_eps_out_of_range_raises():
    sde = make_sde("vp")  # T = 1.0
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, t_eps=0.0)
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, t_eps=-1e-3)
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, t_eps=sde.T)
    with pytest.raises(ValueError):
        _NoOpSampler(sde, _zero_score, t_eps=sde.T + 1.0)


def test_abstract_cannot_instantiate_directly():
    sde = make_sde("vp")
    with pytest.raises(TypeError):
        ReverseSampler(sde, _zero_score)  # step abstracto sin implementar


# ----------------------------------------------------------------- grilla temporal


def test_time_grid_endpoints_and_length():
    sde = make_sde("vp")
    n_steps = 50
    t_eps = 1e-3
    s = _NoOpSampler(sde, _zero_score, n_steps=n_steps, t_eps=t_eps)
    grid = s._time_grid()
    assert grid.shape == (n_steps + 1,)
    assert grid.dtype == torch.float32
    assert grid[0].item() == pytest.approx(sde.T)
    assert grid[-1].item() == pytest.approx(t_eps)
    # Decreciente de T a t_eps.
    assert torch.all(grid[:-1] > grid[1:])


# ------------------------------------------------------------------ drifts reversos


@pytest.mark.parametrize("dim", [2, 5])
def test_reverse_drift_shape_and_finite(dim):
    sde = make_sde("vp", data_dim=dim)
    s = _NoOpSampler(sde, _zero_score)
    x = torch.randn(B, dim)
    t = torch.rand(B)
    d = s._reverse_drift(x, t)
    assert d.shape == (B, dim)
    assert torch.all(torch.isfinite(d))


@pytest.mark.parametrize("dim", [2, 5])
def test_pfode_drift_shape_and_finite(dim):
    sde = make_sde("vp", data_dim=dim)
    s = _NoOpSampler(sde, _zero_score)
    x = torch.randn(B, dim)
    t = torch.rand(B)
    d = s._pfode_drift(x, t)
    assert d.shape == (B, dim)
    assert torch.all(torch.isfinite(d))


def test_drifts_accept_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) deben dar el mismo resultado.
    sde = make_sde("vp")
    s = _NoOpSampler(sde, _zero_score)
    x = torch.randn(B, 2)
    t = torch.rand(B)
    assert torch.equal(s._reverse_drift(x, t), s._reverse_drift(x, t.reshape(B, 1)))
    assert torch.equal(s._pfode_drift(x, t), s._pfode_drift(x, t.reshape(B, 1)))


def test_pfode_is_half_reverse_relative_to_drift():
    # Con score nulo ambos drifts coinciden con f; con score no nulo, la corrección de
    # PF-ODE es la mitad de la del reverso completo: f - g^2 s vs f - 0.5 g^2 s.
    sde = make_sde("vp")

    def score_fn(x, t):
        return torch.ones_like(x)

    s = _NoOpSampler(sde, score_fn)
    x = torch.randn(B, 2)
    t = torch.rand(B)
    f, g = sde.sde(x, t)
    rev = s._reverse_drift(x, t)
    pf = s._pfode_drift(x, t)
    # rev = f - g^2 s ; pf = f - 0.5 g^2 s  =>  f - pf = 0.5 (f - rev)
    assert torch.allclose(f - pf, 0.5 * (f - rev), atol=1e-5)


# --------------------------------------------------------------- driver sample()


class _PFODESampler(ReverseSampler):
    """Sampler determinístico mínimo: paso de Euler sobre el drift de PF-ODE.

    Sirve para ejercitar el driver ``sample()`` sin depender de los samplers
    concretos (tasks 2.x). Ignora ``generator`` (es determinístico).
    """

    name = "pfode_test"

    def step(self, x, t, dt, *, generator):
        return x + self._pfode_drift(x, t) * dt


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito.
    sde = make_sde(sde_name)
    s = _PFODESampler(sde, _zero_score, n_steps=20)
    n = 32
    x0 = s.sample(n)
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_sample_returns_trajectory_with_correct_shape():
    # 1.5: con return_trajectory devuelve (n_steps+1, N, data_dim) y x_0 aparte.
    sde = make_sde("vp")
    n_steps = 15
    s = _PFODESampler(sde, _zero_score, n_steps=n_steps)
    n = 8
    x0, traj = s.sample(n, return_trajectory=True)
    assert x0.shape == (n, sde.data_dim)
    assert traj.shape == (n_steps + 1, n, sde.data_dim)
    assert traj.dtype == torch.float32
    # El último estado de la trayectoria es x_0.
    assert torch.equal(traj[-1], x0)


def test_trajectory_first_slice_is_start_state():
    # 1.2 / 1.5: la trayectoria incluye el estado inicial x_T como primera capa.
    sde = make_sde("vp")
    s = _PFODESampler(sde, _zero_score, n_steps=10)
    n = 8
    gen = torch.Generator().manual_seed(0)
    x_T = sde.prior_sampling((n, sde.data_dim), generator=gen)
    gen2 = torch.Generator().manual_seed(0)
    _, traj = s.sample(n, generator=gen2, return_trajectory=True)
    assert torch.equal(traj[0], x_T)


def test_sample_uses_init_as_start_state():
    # init provisto se usa como x_T (no se sortea el prior).
    sde = make_sde("vp")
    s = _PFODESampler(sde, _zero_score, n_steps=10)
    n = 8
    init = torch.full((n, sde.data_dim), 3.0)
    x0, traj = s.sample(n, init=init, return_trajectory=True)
    assert torch.equal(traj[0], init)
    # Con score nulo y drift de VP el estado evoluciona, pero arranca en init.
    assert x0.shape == (n, sde.data_dim)


def test_sample_does_not_mutate_network_params():
    # 3.3: los parámetros de una ScoreMLP no cambian tras sample().
    ScoreMLP = pytest.importorskip("diffusion.mlp").ScoreMLP
    sde = make_sde("vp", data_dim=2)
    net = ScoreMLP(data_dim=2)
    net.eval()
    before = [p.detach().clone() for p in net.parameters()]

    def score_fn(x, t):
        return net(x, t)

    s = _PFODESampler(sde, score_fn, n_steps=10)
    s.sample(16)
    after = list(net.parameters())
    assert len(before) == len(after)
    for b, a in zip(before, after):
        assert torch.equal(b, a)


def test_sample_t_eps_floor_respected():
    # 8.2: la integración nunca llega a t = 0 exacto (piso t_eps > 0).
    sde = make_sde("vp")
    t_eps = 5e-3
    s = _PFODESampler(sde, _zero_score, n_steps=20, t_eps=t_eps)
    grid = s._time_grid()
    assert grid[-1].item() == pytest.approx(t_eps)
    assert torch.all(grid > 0.0)


def test_sample_determinism_with_fixed_init():
    # Integrador determinístico con init fijo -> salida idéntica (aísla del prior).
    sde = make_sde("vp")
    s = _PFODESampler(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init)
    b = s.sample(16, init=init)
    assert torch.equal(a, b)
