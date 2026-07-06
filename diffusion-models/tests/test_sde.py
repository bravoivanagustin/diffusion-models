"""Tests de los procesos forward (`diffusion.sde`)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.sde import (
    SubVPSDE,
    VESDE,
    VPSDE,
    available_sdes,
    make_sde,
)

SCALAR = ["vp", "ve", "sub_vp"]
B = 16


def _x0_t(n=B, dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, dim, generator=g)
    t = torch.rand(n, generator=g)
    return x0, t


# --------------------------------------------------------------- registry / factory


def test_available_sdes():
    assert set(available_sdes()) == {"vp", "ve", "sub_vp"}


def test_factory_returns_right_type():
    assert isinstance(make_sde("vp"), VPSDE)
    assert isinstance(make_sde("ve"), VESDE)
    assert isinstance(make_sde("sub_vp"), SubVPSDE)


def test_unknown_sde_raises():
    with pytest.raises(ValueError):
        make_sde("no_existe")


def test_factory_filters_kwargs():
    # beta_min no aplica a VE y se descarta; sigma_max sí se aplica.
    sde = make_sde("ve", beta_min=0.1, sigma_max=3.0)
    assert isinstance(sde, VESDE)
    assert sde.sigma_max == 3.0


# ---------------------------------------------------------------- shapes / dtype


@pytest.mark.parametrize("name", SCALAR)
def test_marginal_prob_shapes_dtype(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    mean, std = sde.marginal_prob(x0, t)
    assert mean.shape == (B, 2)
    assert std.shape == (B, 1)
    assert mean.dtype == torch.float32 and std.dtype == torch.float32
    assert torch.all(torch.isfinite(mean)) and torch.all(torch.isfinite(std))


@pytest.mark.parametrize("name", SCALAR)
def test_perturb_shapes_dtype(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    x_t, eps = sde.perturb(x0, t)
    assert x_t.shape == (B, 2) and eps.shape == (B, 2)
    assert x_t.dtype == torch.float32 and eps.dtype == torch.float32


@pytest.mark.parametrize("name", SCALAR)
def test_sde_drift_diffusion_shapes(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    drift, diffusion = sde.sde(x0, t)
    assert drift.shape == (B, 2) and diffusion.shape == (B, 1)
    if name == "ve":
        assert torch.all(drift == 0.0)  # VE no tiene drift


@pytest.mark.parametrize("name", SCALAR)
def test_accepts_t_as_B_and_B1(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    m1, s1 = sde.marginal_prob(x0, t)
    m2, s2 = sde.marginal_prob(x0, t.reshape(B, 1))
    assert torch.equal(m1, m2) and torch.equal(s1, s2)


# ----------------------------------------------------------------- determinismo


@pytest.mark.parametrize("name", SCALAR)
def test_perturb_deterministic_with_generator(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    a, _ = sde.perturb(x0, t, generator=torch.Generator().manual_seed(7))
    b, _ = sde.perturb(x0, t, generator=torch.Generator().manual_seed(7))
    c, _ = sde.perturb(x0, t, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ------------------------------------------------------- límites / forma cerrada


def test_vp_small_t():
    sde = VPSDE()
    x0 = torch.randn(B, 2)
    t = torch.full((B,), 1e-4)
    mean, std = sde.marginal_prob(x0, t)
    assert torch.allclose(mean, x0, atol=1e-2)
    assert torch.all(std < 1e-2)


def test_vp_t_equals_T():
    sde = VPSDE()
    x0 = torch.randn(B, 2)
    t = torch.full((B,), 1.0)
    mean, std = sde.marginal_prob(x0, t)
    assert torch.all(mean.abs() < 0.1)          # mean -> 0
    assert torch.allclose(std, torch.ones_like(std), atol=1e-3)  # std -> 1


def test_ve_no_drift_and_sigma():
    sde = VESDE(sigma_min=0.01, sigma_max=5.0)
    x0 = torch.randn(B, 2)
    t = torch.rand(B)
    mean, std = sde.marginal_prob(x0, t)
    assert torch.equal(mean, x0)                # sin drift: mean == x0 exacto
    expected = 0.01 * (5.0 / 0.01) ** t.reshape(B, 1)
    assert torch.allclose(std, expected)
    # std(T) ~ sigma_max
    _, std_T = sde.marginal_prob(x0, torch.ones(B))
    assert torch.allclose(std_T, torch.full_like(std_T, 5.0))


def test_subvp_std_below_vp_same_mean():
    vp, sub = VPSDE(), SubVPSDE()
    x0 = torch.randn(B, 2)
    t = torch.rand(B) * 0.9 + 0.05  # 0 < t <= T
    m_vp, s_vp = vp.marginal_prob(x0, t)
    m_sub, s_sub = sub.marginal_prob(x0, t)
    assert torch.allclose(m_vp, m_sub)          # mismo alpha_t -> misma media
    assert torch.all(s_sub < s_vp)              # varianza estrictamente por debajo


@pytest.mark.parametrize("name", SCALAR)
def test_variance_ode_consistency(name):
    # Chequeo de cálculo: dSigma/dt debe coincidir con 2 f_coef Sigma + g^2,
    # donde el drift es f_coef * x (lineal) y g = diffusion. Diferencias finitas.
    sde = make_sde(name)
    t = torch.tensor([0.2, 0.4, 0.6, 0.8]).reshape(-1, 1)
    x = torch.ones(t.shape[0], 2)
    drift, diffusion = sde.sde(x, t)
    f_coef = drift[:, :1]                        # f_coef * 1 = drift  (x == 1)
    g2 = diffusion ** 2

    def var(tt):
        return sde.marginal_prob(x, tt)[1] ** 2

    h = 1e-3
    dvar = (var(t + h) - var(t - h)) / (2 * h)
    rhs = 2 * f_coef * var(t) + g2
    assert torch.allclose(dvar, rhs, rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------- score target


@pytest.mark.parametrize("name", SCALAR)
def test_score_target(name):
    sde = make_sde(name)
    x0, t = _x0_t()
    x_t, eps = sde.perturb(x0, t, generator=torch.Generator().manual_seed(3))
    score_real, weight = sde.score_target(x0, t, eps)
    _, std = sde.marginal_prob(x0, t)
    assert torch.allclose(score_real, -eps / std, atol=1e-5)   # -eps/sigma_t
    assert torch.allclose(weight, std ** 2, atol=1e-5)          # lambda(t) = sigma_t^2
    nz = eps.abs() > 1e-3
    assert torch.equal(torch.sign(score_real[nz]), -torch.sign(eps[nz]))


# -------------------------------------------------------------- prior_sampling


@pytest.mark.parametrize("name", SCALAR)
def test_prior_sampling_shape(name):
    sde = make_sde(name)
    z = sde.prior_sampling((10, 2), generator=torch.Generator().manual_seed(0))
    assert z.shape == (10, 2) and z.dtype == torch.float32


def test_prior_variance():
    g = torch.Generator().manual_seed(0)
    n = 40000
    for name in ("vp", "sub_vp"):
        z = make_sde(name).prior_sampling((n, 2), generator=g)
        assert abs(z.var().item() - 1.0) < 0.1
    ve = make_sde("ve", sigma_max=5.0)
    z = ve.prior_sampling((n, 2), generator=g)
    assert abs(z.var().item() - 25.0) < 25.0 * 0.1  # ~ sigma_max^2


# ------------------------------------------------------------ seam sde x mlp


@pytest.mark.parametrize("name", SCALAR)
def test_seam_with_scoremlp(name):
    from diffusion.models import ScoreMLP

    sde = make_sde(name)
    net = ScoreMLP(data_dim=2)
    x0, t = _x0_t()
    x_t, eps = sde.perturb(x0, t, generator=torch.Generator().manual_seed(1))
    pred = net(x_t, t)
    target, _ = sde.score_target(x0, t, eps)
    assert pred.shape == target.shape == (B, 2)
    assert torch.all(torch.isfinite(pred))


# ============================================================ dimensión arbitraria


def test_data_dim_must_be_positive():
    with pytest.raises(ValueError):
        make_sde("vp", data_dim=0)


def test_factory_passes_data_dim():
    assert make_sde("vp", data_dim=7).data_dim == 7


@pytest.mark.parametrize("name", SCALAR)
@pytest.mark.parametrize("dim", [1, 3, 7])
def test_scalar_arbitrary_dim(name, dim):
    sde = make_sde(name, data_dim=dim)
    assert sde.data_dim == dim
    x0, t = torch.randn(B, dim), torch.rand(B)
    x_t, eps = sde.perturb(x0, t, generator=torch.Generator().manual_seed(0))
    assert x_t.shape == (B, dim) and eps.shape == (B, dim)
    mean, std = sde.marginal_prob(x0, t)
    assert mean.shape == (B, dim) and std.shape == (B, 1)
    score, weight = sde.score_target(x0, t, eps)
    assert score.shape == (B, dim) and weight.shape == (B, 1)
    drift, diffusion = sde.sde(x0, t)
    assert drift.shape == (B, dim) and diffusion.shape == (B, 1)
    assert sde.prior_sampling((5, dim)).shape == (5, dim)
