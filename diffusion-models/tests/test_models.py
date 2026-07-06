"""Tests de las redes de score (`diffusion.models`): piezas compartidas + ScoreMLP."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.models import ResidualBlock, ScoreMLP, SinusoidalEmbedding


# --------------------------------------------------------- SinusoidalEmbedding


def test_embedding_odd_dim_raises():
    with pytest.raises(ValueError):
        SinusoidalEmbedding(embed_dim=127)


@pytest.mark.parametrize("embed_dim", [2, 8, 128])
def test_embedding_output_shape(embed_dim):
    emb = SinusoidalEmbedding(embed_dim)
    out = emb(torch.rand(16))
    assert out.shape == (16, embed_dim)


def test_embedding_accepts_both_input_shapes():
    emb = SinusoidalEmbedding(32)
    t = torch.rand(16)
    a = emb(t)
    b = emb(t.reshape(16, 1))
    assert a.shape == b.shape == (16, 32)
    assert torch.equal(a, b)


def test_embedding_is_deterministic():
    emb = SinusoidalEmbedding(64)
    t = torch.rand(10)
    assert torch.equal(emb(t), emb(t))


def test_embedding_values_bounded():
    emb = SinusoidalEmbedding(128)
    out = emb(torch.tensor([0.0, 0.5, 1.0, 7.0, 999.0]))
    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


@pytest.mark.parametrize(
    "t",
    [
        torch.linspace(0.0, 1.0, 8),       # rango [0, 1]
        torch.linspace(0.0, 1000.0, 8),    # rango [0, T]
        torch.arange(8).float(),           # pasos enteros
    ],
)
def test_embedding_any_scale_finite(t):
    out = SinusoidalEmbedding(64)(t)
    assert out.shape == (8, 64)
    assert torch.all(torch.isfinite(out))


def test_embedding_interleaves_sin_cos():
    # Para t -> 0, el primer denominador es 1: sin(0)=0 (índice par),
    # cos(0)=1 (índice impar). Verifica el orden 2i=sin, 2i+1=cos.
    emb = SinusoidalEmbedding(8)
    out = emb(torch.zeros(1))[0]
    assert torch.allclose(out[0::2], torch.zeros(4))   # senos
    assert torch.allclose(out[1::2], torch.ones(4))    # cosenos


def test_embedding_denom_is_buffer_not_param():
    emb = SinusoidalEmbedding(16)
    assert "denom" in dict(emb.named_buffers())
    assert list(emb.parameters()) == []


# ------------------------------------------------------------- ResidualBlock


def test_residual_block_preserves_shape():
    block = ResidualBlock(hidden_dim=32)
    x = torch.randn(8, 32)
    assert block(x).shape == (8, 32)


def test_residual_block_bad_activation_raises():
    with pytest.raises(ValueError):
        ResidualBlock(hidden_dim=16, activation="no_existe")


# ----------------------------------------------------------------- ScoreMLP


@pytest.mark.parametrize("data_dim", [2, 4])
def test_scoremlp_output_shape(data_dim):
    net = ScoreMLP(data_dim=data_dim)
    x = torch.randn(16, data_dim)
    out = net(x, torch.rand(16))
    assert out.shape == (16, data_dim)


def test_scoremlp_accepts_both_t_shapes():
    net = ScoreMLP().eval()
    x = torch.randn(16, 2)
    t = torch.rand(16)
    with torch.no_grad():
        a = net(x, t)              # t de shape (B,)
        b = net(x, t.reshape(16, 1))  # t de shape (B, 1)
    assert a.shape == b.shape == (16, 2)
    assert torch.equal(a, b)       # ambas formas dan el mismo resultado


def test_scoremlp_is_deterministic():
    # La red es la variable de control: mismo (x, t) -> misma salida.
    net = ScoreMLP().eval()
    x, t = torch.randn(16, 2), torch.rand(16)
    with torch.no_grad():
        assert torch.equal(net(x, t), net(x, t))


def test_scoremlp_no_stochastic_layers():
    # Sin dropout ni batchnorm: la red debe ser enteramente determinística.
    net = ScoreMLP()
    for module in net.modules():
        assert not isinstance(module, torch.nn.Dropout)
        assert not isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        )


def test_scoremlp_has_trainable_params():
    net = ScoreMLP()
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert n > 0


def test_scoremlp_num_blocks_configurable():
    assert len(ScoreMLP(num_blocks=2).blocks) == 2
    assert len(ScoreMLP(num_blocks=6).blocks) == 6


def test_scoremlp_gradients_flow():
    net = ScoreMLP()
    x, t = torch.randn(8, 2), torch.rand(8)
    net(x, t).pow(2).sum().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)
