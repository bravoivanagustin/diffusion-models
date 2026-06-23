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


# ----------------------------------------------------- task 2.1: Euler–Maruyama


def test_euler_name():
    # 2.1: la clave del registry queda fijada en la clase (factory llega en 3.1).
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    assert EulerMaruyama.name == "euler"


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_euler_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito sobre las 3 SDEs escalares.
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde(sde_name)
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    n = 32
    x0 = s.sample(n, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_euler_reproducible_same_seed():
    # 5.2: dos corridas con generadores del MISMO seed coinciden exactamente.
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("vp")
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(123))
    b = s.sample(16, generator=torch.Generator().manual_seed(123))
    assert torch.equal(a, b)


def test_euler_differs_with_different_seed():
    # 5.3: semillas distintas -> muestras distintas.
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("vp")
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


def test_euler_step_injects_noise():
    # 2.2: el paso es genuinamente estocástico — con score y drift nulos (VE: drift 0)
    # el estado igual cambia por la inyección de ruido del término de difusión.
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("ve")  # drift nulo
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    x = torch.zeros(8, sde.data_dim)
    t = torch.full((8, 1), sde.T)
    out = s.step(x, t, dt=-0.05, generator=torch.Generator().manual_seed(0))
    # Con f=0 y s=0, out = g·√|dt|·Z, que no puede ser idénticamente x (=0).
    assert not torch.equal(out, x)
    assert torch.all(torch.isfinite(out))


def test_euler_step_accepts_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) dan el mismo resultado (mismo generador sembrado).
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("vp")
    s = EulerMaruyama(sde, _zero_score, n_steps=20)
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)
    out_flat = s.step(x, t, dt=-0.05, generator=torch.Generator().manual_seed(7))
    out_col = s.step(x, t.reshape(8, 1), dt=-0.05, generator=torch.Generator().manual_seed(7))
    assert torch.equal(out_flat, out_col)


# --------------------------------------------- task 2.2: Probability-Flow ODE


def test_pf_ode_name():
    # 2.3: la clave del registry queda fijada en la clase (factory llega en 3.1).
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    assert ProbabilityFlowODE.name == "pf_ode"


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_pf_ode_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito sobre las 3 SDEs escalares.
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    sde = make_sde(sde_name)
    s = ProbabilityFlowODE(sde, _zero_score, n_steps=20)
    n = 32
    x0 = s.sample(n, init=torch.randn(n, sde.data_dim))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_pf_ode_deterministic_same_init():
    # 5.1: dos corridas con el MISMO init producen resultados idénticos.
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    sde = make_sde("vp")
    s = ProbabilityFlowODE(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init)
    b = s.sample(16, init=init)
    assert torch.equal(a, b)


def test_pf_ode_ignores_generator():
    # 5.1 / 2.3: determinístico — con el mismo init, semillas distintas dan el MISMO
    # resultado (prueba que el sampler ignora la aleatoriedad del generator).
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    sde = make_sde("vp")
    s = ProbabilityFlowODE(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, init=init, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)


def test_pf_ode_step_is_pfode_drift_euler():
    # 2.3: el paso es exactamente x + (f - ½ g² s)·dt (Euler sobre el drift de PF-ODE).
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    def score_fn(x, t):
        return torch.ones_like(x)

    sde = make_sde("vp")
    s = ProbabilityFlowODE(sde, score_fn, n_steps=20)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    dt = -0.05
    expected = x + s._pfode_drift(x, t) * dt
    out = s.step(x, t, dt, generator=None)
    assert torch.equal(out, expected)
    assert torch.all(torch.isfinite(out))


def test_pf_ode_step_accepts_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) dan el mismo resultado (determinístico).
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    sde = make_sde("vp")
    s = ProbabilityFlowODE(sde, _zero_score, n_steps=20)
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)
    out_flat = s.step(x, t, dt=-0.05, generator=None)
    out_col = s.step(x, t.reshape(8, 1), dt=-0.05, generator=None)
    assert torch.equal(out_flat, out_col)


# ----------------------------------------------------- task 2.3: Heun (ODE 2º orden)


def test_heun_name():
    # 2.4: la clave del registry queda fijada en la clase (factory llega en 3.1).
    from diffusion.samplers.heun import HeunODE

    assert HeunODE.name == "heun"


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_heun_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito sobre las 3 SDEs escalares.
    from diffusion.samplers.heun import HeunODE

    sde = make_sde(sde_name)
    s = HeunODE(sde, _zero_score, n_steps=20)
    n = 32
    x0 = s.sample(n, init=torch.randn(n, sde.data_dim))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_heun_deterministic_same_init():
    # 5.1: dos corridas con el MISMO init producen resultados idénticos.
    from diffusion.samplers.heun import HeunODE

    sde = make_sde("vp")
    s = HeunODE(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init)
    b = s.sample(16, init=init)
    assert torch.equal(a, b)


def test_heun_ignores_generator():
    # 5.1 / 2.4: determinístico — con el mismo init, semillas distintas dan el MISMO
    # resultado (prueba que el sampler ignora la aleatoriedad del generator).
    from diffusion.samplers.heun import HeunODE

    sde = make_sde("vp")
    s = HeunODE(sde, _zero_score, n_steps=20)
    init = torch.randn(16, sde.data_dim)
    a = s.sample(16, init=init, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, init=init, generator=torch.Generator().manual_seed(2))
    assert torch.equal(a, b)


def test_heun_step_matches_trapezoidal_formula():
    # 2.4: el paso es exactamente la regla del trapecio (predictor Euler + corrección):
    #   d1 = _pfode_drift(x, t); x̂ = x + d1·dt; d2 = _pfode_drift(x̂, t+dt)
    #   out = x + ½(d1 + d2)·dt
    from diffusion.samplers.heun import HeunODE

    def score_fn(x, t):
        # Score no constante en x para que la 2ª evaluación difiera de la 1ª.
        return x

    sde = make_sde("vp")
    s = HeunODE(sde, score_fn, n_steps=20)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    dt = -0.05
    d1 = s._pfode_drift(x, t)
    x_pred = x + d1 * dt
    d2 = s._pfode_drift(x_pred, t + dt)
    expected = x + 0.5 * (d1 + d2) * dt
    out = s.step(x, t, dt, generator=None)
    assert torch.equal(out, expected)
    assert torch.all(torch.isfinite(out))


def test_heun_does_two_score_evaluations():
    # 2.4: costo observable de DOS evaluaciones de score por paso (predictor + corrector).
    from diffusion.samplers.heun import HeunODE

    calls = {"n": 0}

    def counting_score(x, t):
        calls["n"] += 1
        return torch.ones_like(x)

    sde = make_sde("vp")
    s = HeunODE(sde, counting_score, n_steps=20)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    s.step(x, t, dt=-0.05, generator=None)
    assert calls["n"] == 2


def test_heun_differs_from_single_euler_pfode_step():
    # 2.4: la corrección de 2º orden es real (no un no-op): con drift no constante,
    # un paso de Heun difiere de un único paso de Euler sobre el drift de PF-ODE.
    from diffusion.samplers.heun import HeunODE
    from diffusion.samplers.pf_ode import ProbabilityFlowODE

    def score_fn(x, t):
        return x  # drift depende de x -> d(x̂, t+dt) != d(x, t)

    sde = make_sde("vp")
    heun = HeunODE(sde, score_fn, n_steps=20)
    pf = ProbabilityFlowODE(sde, score_fn, n_steps=20)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    dt = -0.05
    out_heun = heun.step(x, t, dt, generator=None)
    out_euler = pf.step(x, t, dt, generator=None)
    assert not torch.equal(out_heun, out_euler)


def test_heun_step_accepts_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) dan el mismo resultado (determinístico).
    from diffusion.samplers.heun import HeunODE

    sde = make_sde("vp")
    s = HeunODE(sde, _zero_score, n_steps=20)
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)
    out_flat = s.step(x, t, dt=-0.05, generator=None)
    out_col = s.step(x, t.reshape(8, 1), dt=-0.05, generator=None)
    assert torch.equal(out_flat, out_col)


# ------------------------------------------- task 2.4: predictor–corrector


def _linear_score(x, t):
    """Score analítico no nulo (apunta al origen) para ejercitar el corrector."""
    return -x


def test_pc_name():
    # 2.5: la clave del registry queda fijada en la clase (factory llega en 3.1).
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    assert PredictorCorrector.name == "pc"


def test_pc_constructor_accepts_extra_kwargs():
    # 4.4 / 2.5: es el único sampler con kwargs propios (n_corrector, snr).
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _zero_score, n_steps=20, n_corrector=3, snr=0.2)
    assert s.n_corrector == 3
    assert s.snr == pytest.approx(0.2)


def test_pc_constructor_defaults():
    # 2.5: defaults documentados (Song et al.): n_corrector=1, snr=0.16.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _zero_score, n_steps=20)
    assert s.n_corrector == 1
    assert s.snr == pytest.approx(0.16)


@pytest.mark.parametrize("sde_name", ["vp", "ve", "sub_vp"])
def test_pc_sample_shape_dtype_finite(sde_name):
    # 1.1, 1.4, 8.3: x_0 de shape (N, data_dim), float32 y finito sobre las 3 SDEs escalares.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde(sde_name)
    s = PredictorCorrector(sde, _linear_score, n_steps=20, n_corrector=1)
    n = 32
    x0 = s.sample(n, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (n, sde.data_dim)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_pc_sample_finite_with_real_mlp():
    # 1.4: con una ScoreMLP real (sin entrenar) el corrector se ejercita y produce finito.
    ScoreMLP = pytest.importorskip("diffusion.mlp").ScoreMLP
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp", data_dim=2)
    net = ScoreMLP(data_dim=2)
    net.eval()

    def score_fn(x, t):
        return net(x, t)

    s = PredictorCorrector(sde, score_fn, n_steps=20, n_corrector=2)
    x0 = s.sample(16, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (16, 2)
    assert torch.all(torch.isfinite(x0))


def test_pc_reproducible_same_seed():
    # 5.2: dos corridas con generadores del MISMO seed coinciden exactamente.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _linear_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(123))
    b = s.sample(16, generator=torch.Generator().manual_seed(123))
    assert torch.equal(a, b)


def test_pc_differs_with_different_seed():
    # 5.3: semillas distintas -> muestras distintas.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _linear_score, n_steps=20)
    a = s.sample(16, generator=torch.Generator().manual_seed(1))
    b = s.sample(16, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


def test_pc_n_corrector_zero_reduces_to_predictor():
    # 2.5: n_corrector controla el número de correcciones de Langevin. Con n_corrector=0
    # el paso es solo el predictor (Euler–Maruyama); debe diferir de n_corrector=2.
    from diffusion.samplers.predictor_corrector import PredictorCorrector
    from diffusion.samplers.euler_maruyama import EulerMaruyama

    sde = make_sde("vp")
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    dt = -0.05

    pc0 = PredictorCorrector(sde, _linear_score, n_steps=20, n_corrector=0)
    em = EulerMaruyama(sde, _linear_score, n_steps=20)
    out0 = pc0.step(x, t, dt, generator=torch.Generator().manual_seed(5))
    out_em = em.step(x, t, dt, generator=torch.Generator().manual_seed(5))
    # Sin correcciones, el paso PC coincide con el predictor Euler–Maruyama.
    assert torch.equal(out0, out_em)

    pc2 = PredictorCorrector(sde, _linear_score, n_steps=20, n_corrector=2)
    out2 = pc2.step(x, t, dt, generator=torch.Generator().manual_seed(5))
    # Con correcciones de Langevin el resultado difiere del predictor solo.
    assert not torch.equal(out2, out0)


def test_pc_counts_score_calls():
    # 2.5: predictor usa el drift reverso (1 eval de score) + n_corrector evals de Langevin.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    calls = {"n": 0}

    def counting_score(x, t):
        calls["n"] += 1
        return -x

    sde = make_sde("vp")
    s = PredictorCorrector(sde, counting_score, n_steps=20, n_corrector=3)
    x = torch.randn(8, sde.data_dim)
    t = torch.full((8, 1), 0.5)
    s.step(x, t, dt=-0.05, generator=torch.Generator().manual_seed(0))
    assert calls["n"] == 1 + 3  # predictor (reverse drift) + n_corrector Langevin


def test_pc_nan_safe_with_zero_score():
    # 8.2: con score nulo el denominador ‖s‖ es ~0; el piso evita div-by-zero / NaN.
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _zero_score, n_steps=20, n_corrector=2)
    x0 = s.sample(16, generator=torch.Generator().manual_seed(0))
    assert torch.all(torch.isfinite(x0))


def test_pc_step_accepts_t_as_B_and_B1():
    # 8.1: t como (B,) y (B,1) dan el mismo resultado (mismo generador sembrado).
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    sde = make_sde("vp")
    s = PredictorCorrector(sde, _linear_score, n_steps=20)
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)
    out_flat = s.step(x, t, dt=-0.05, generator=torch.Generator().manual_seed(7))
    out_col = s.step(x, t.reshape(8, 1), dt=-0.05, generator=torch.Generator().manual_seed(7))
    assert torch.equal(out_flat, out_col)


# ----------------------------------------------- task 3.1: registry y factory


def test_available_samplers_sorted_names():
    # 4.2: la factory expone la lista de nombres disponibles (ordenados).
    from diffusion.samplers import available_samplers

    assert available_samplers() == ["euler", "heun", "pc", "pf_ode"]


def test_registry_maps_names_to_classes():
    # 4.1/2.1: REGISTRY mapea cada nombre a su clase de sampler.
    from diffusion.samplers import REGISTRY
    from diffusion.samplers.euler_maruyama import EulerMaruyama
    from diffusion.samplers.heun import HeunODE
    from diffusion.samplers.pf_ode import ProbabilityFlowODE
    from diffusion.samplers.predictor_corrector import PredictorCorrector

    assert REGISTRY == {
        "euler": EulerMaruyama,
        "pf_ode": ProbabilityFlowODE,
        "heun": HeunODE,
        "pc": PredictorCorrector,
    }


@pytest.mark.parametrize(
    "name,cls_name",
    [
        ("euler", "EulerMaruyama"),
        ("pf_ode", "ProbabilityFlowODE"),
        ("heun", "HeunODE"),
        ("pc", "PredictorCorrector"),
    ],
)
def test_make_sampler_returns_correct_type(name, cls_name):
    # 4.1/2.1: make_sampler instancia el sampler correcto configurado con sde y score.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler(name, sde, _zero_score)
    assert type(s).__name__ == cls_name
    assert s.sde is sde
    assert s.score_fn is _zero_score


def test_make_sampler_unknown_name_lists_options():
    # 4.3: nombre desconocido -> ValueError enumerando las opciones válidas.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    with pytest.raises(ValueError) as excinfo:
        make_sampler("does_not_exist", sde, _zero_score)
    msg = str(excinfo.value)
    for opt in ("euler", "heun", "pc", "pf_ode"):
        assert opt in msg


def test_make_sampler_discards_inapplicable_kwargs():
    # 4.4: kwargs que no aplican al sampler elegido se descartan sin fallar (caller genérico).
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    # snr/n_corrector son exclusivos de PC; euler debe ignorarlos sin error.
    s = make_sampler("euler", sde, _zero_score, snr=0.5, n_corrector=3)
    assert type(s).__name__ == "EulerMaruyama"
    assert not hasattr(s, "n_corrector")
    assert not hasattr(s, "snr")


def test_make_sampler_applies_pc_kwargs():
    # 4.4: para PC, los kwargs propios SÍ se aplican (no se descartan).
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler("pc", sde, _zero_score, snr=0.5, n_corrector=3)
    assert type(s).__name__ == "PredictorCorrector"
    assert s.n_corrector == 3
    assert s.snr == pytest.approx(0.5)


def test_make_sampler_passes_common_kwargs():
    # 4.1/8.4: kwargs comunes (n_steps, t_eps) llegan al constructor base.
    from diffusion.samplers import make_sampler

    sde = make_sde("vp")
    s = make_sampler("euler", sde, _zero_score, n_steps=33, t_eps=5e-3)
    assert s.n_steps == 33
    assert s.t_eps == pytest.approx(5e-3)


# --------------------------------------- task 3.2: generate_from_checkpoint


def _make_checkpoint(tmp_path, sde_name="vp", data_dim=2):
    """Arma un checkpoint válido SIN entrenar: red sin entrenar -> save_checkpoint.

    Reusa el contrato real de :mod:`diffusion.training` (``TrainResult`` +
    ``save_checkpoint``), de modo que ``load_checkpoint`` lo reconstruya idéntico a uno
    producido por una corrida real. Devuelve la ruta del ``.pt`` guardado.
    """
    ScoreMLP = pytest.importorskip("diffusion.mlp").ScoreMLP
    from diffusion.training import TrainConfig, TrainResult, save_checkpoint

    net = ScoreMLP(data_dim=data_dim)
    result = TrainResult(
        net=net, history=[1.0, 0.5], config=TrainConfig(), sde_name=sde_name
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(result, path)
    return path


def test_generate_from_checkpoint_returns_samples(tmp_path):
    # 6.1: a partir de un checkpoint reconstruye SDE+red y genera (N, data_dim) float32.
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path, sde_name="vp", data_dim=2)
    x0 = generate_from_checkpoint(ckpt, "pf_ode", n_samples=8, n_steps=5, seed=0)
    assert x0.shape == (8, 2)
    assert x0.dtype == torch.float32
    assert torch.all(torch.isfinite(x0))


def test_generate_from_checkpoint_writes_npz(tmp_path):
    # 6.2: con `out` persiste un .npz con la clave `samples`.
    pytest.importorskip("numpy")
    import numpy as np
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path)
    out = tmp_path / "gen" / "out.npz"  # subdir inexistente: debe crearse
    x0 = generate_from_checkpoint(
        ckpt, "pf_ode", n_samples=8, n_steps=5, seed=0, out=out
    )
    assert out.exists()
    with np.load(out) as data:
        assert "samples" in data
        assert data["samples"].shape == (8, 2)
        assert np.allclose(data["samples"], x0.cpu().numpy())


def test_generate_from_checkpoint_saves_trajectory(tmp_path):
    # 6.3: con save_trajectory el .npz también guarda la trayectoria.
    pytest.importorskip("numpy")
    import numpy as np
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path)
    out = tmp_path / "out.npz"
    n_steps = 5
    generate_from_checkpoint(
        ckpt, "pf_ode", n_samples=8, n_steps=n_steps, seed=0, out=out,
        save_trajectory=True,
    )
    with np.load(out) as data:
        assert "samples" in data
        assert "trajectory" in data
        # (n_steps + 1, N, data_dim): incluye el estado inicial x_T.
        assert data["trajectory"].shape == (n_steps + 1, 8, 2)


def test_generate_from_checkpoint_no_trajectory_key_when_disabled(tmp_path):
    # 6.3: sin save_trajectory el .npz NO contiene la clave `trajectory`.
    pytest.importorskip("numpy")
    import numpy as np
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path)
    out = tmp_path / "out.npz"
    generate_from_checkpoint(ckpt, "pf_ode", n_samples=8, n_steps=5, seed=0, out=out)
    with np.load(out) as data:
        assert "trajectory" not in data


def test_generate_from_checkpoint_reproducible_with_seed(tmp_path):
    # 6.1/6.2: misma seed -> resultado reproducible (incluso para un sampler estocástico).
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path)
    a = generate_from_checkpoint(ckpt, "euler", n_samples=8, n_steps=5, seed=42)
    b = generate_from_checkpoint(ckpt, "euler", n_samples=8, n_steps=5, seed=42)
    assert torch.equal(a, b)


def test_generate_from_checkpoint_reconstructs_sde_from_meta(tmp_path):
    # 6.1: la SDE se reconstruye desde meta["sde_name"] (no se pasa por argumento).
    # Un checkpoint con sde_name="ve" debe generar con la SDE VE sin pedir el nombre.
    from diffusion.samplers import generate_from_checkpoint

    ckpt = _make_checkpoint(tmp_path, sde_name="ve", data_dim=2)
    x0 = generate_from_checkpoint(ckpt, "pf_ode", n_samples=8, n_steps=5, seed=0)
    assert x0.shape == (8, 2)
    assert torch.all(torch.isfinite(x0))


def test_generate_from_checkpoint_missing_path_raises(tmp_path):
    # 6.4: ruta inexistente -> error claro que menciona la ruta.
    from diffusion.samplers import generate_from_checkpoint

    missing = tmp_path / "no_existe.pt"
    with pytest.raises((FileNotFoundError, ValueError)) as excinfo:
        generate_from_checkpoint(missing, "pf_ode", n_samples=4, n_steps=3)
    assert "no_existe.pt" in str(excinfo.value)


def test_generate_from_checkpoint_invalid_meta_raises(tmp_path):
    # 6.4: checkpoint sin las claves de meta esperadas -> error claro.
    torch_mod = pytest.importorskip("torch")
    from diffusion.samplers import generate_from_checkpoint

    bad = tmp_path / "bad.pt"
    torch_mod.save({"not": "a checkpoint"}, bad)
    with pytest.raises((KeyError, ValueError)):
        generate_from_checkpoint(bad, "pf_ode", n_samples=4, n_steps=3)


# ------------------------------------------------- task 3.3: smoke entrypoint


def test_main_smoke_runs_all_samplers():
    # 2.1: el smoke entrypoint recorre el registry, corre cada sampler sobre una ScoreMLP
    # sin entrenar y reporta salidas finitas. main() devuelve un resumen assertable
    # {name: is_finite} con los cuatro samplers, todos finitos.
    pytest.importorskip("diffusion.mlp")
    from diffusion.samplers.__main__ import main
    from diffusion.samplers import available_samplers

    summary = main()
    assert set(summary) == set(available_samplers())
    assert len(summary) == 4
    assert all(summary.values())


# ------------------------------------------------------- task 3.4: CLI sample.py


def _load_sample_cli():
    """Importa ``scripts/sample.py`` por ruta (vive fuera del paquete ``diffusion``)."""
    import importlib.util
    import pathlib

    script = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "sample.py"
    )
    spec = importlib.util.spec_from_file_location("_sample_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_sample_writes_npz(tmp_path):
    # 6.1/6.2: la CLI genera un .npz con `samples` (N, data_dim) desde un checkpoint.
    pytest.importorskip("diffusion.mlp")
    pytest.importorskip("numpy")
    import numpy as np

    cli = _load_sample_cli()
    ckpt = _make_checkpoint(tmp_path, sde_name="vp", data_dim=2)
    out = tmp_path / "gen" / "out.npz"  # subdir inexistente: debe crearse
    rc = cli.main(
        [
            str(ckpt),
            "--sampler", "pf_ode",
            "--n-samples", "8",
            "--n-steps", "5",
            "--seed", "0",
            "--out", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    with np.load(out) as data:
        assert "samples" in data
        assert data["samples"].shape == (8, 2)
        assert "trajectory" not in data


def test_cli_sample_trajectory_flag(tmp_path):
    # 6.3: con --trajectory el .npz también guarda la trayectoria.
    pytest.importorskip("diffusion.mlp")
    pytest.importorskip("numpy")
    import numpy as np

    cli = _load_sample_cli()
    ckpt = _make_checkpoint(tmp_path)
    out = tmp_path / "out.npz"
    n_steps = 5
    rc = cli.main(
        [
            str(ckpt),
            "--sampler", "pf_ode",
            "--n-samples", "8",
            "--n-steps", str(n_steps),
            "--seed", "0",
            "--out", str(out),
            "--trajectory",
        ]
    )
    assert rc == 0
    with np.load(out) as data:
        assert "samples" in data
        assert "trajectory" in data
        assert data["trajectory"].shape == (n_steps + 1, 8, 2)


def test_cli_sample_parser_has_sampler_choices():
    # 4.2/6.1: el parser ofrece --sampler con las opciones del registry.
    from diffusion.samplers import available_samplers

    cli = _load_sample_cli()
    parser = cli.build_parser()
    # parse mínimo válido: checkpoint + sampler del registry.
    args = parser.parse_args(["ckpt.pt", "--sampler", available_samplers()[0]])
    assert args.sampler in available_samplers()
    assert args.n_steps == 500  # default de generate_from_checkpoint
