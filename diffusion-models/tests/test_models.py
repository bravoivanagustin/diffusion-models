"""Tests de las redes de score (`diffusion.models`): piezas compartidas + ScoreMLP + ScoreUNet."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.models import (
    REGISTRY,
    ResidualBlock,
    ScoreMLP,
    ScoreModel,
    ScoreUNet,
    SinusoidalEmbedding,
    available_models,
    make_model,
)


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


# ----------------------------------------------------------------- ScoreUNet
# Config tiny del diseño (Testing Strategy): una instancia por resolución de
# trabajo (image_size 32 o 64) con los mismos anchos reducidos, para mantener el
# tiempo de la suite en el orden del resto del repo. El nivel 16×16 ejercita la
# atención. Estos tests fijan el CONTRATO de la red (shape, tiempo, Protocol);
# determinismo, configuración y errores se cubren en su propia sección.


def _tiny_unet(in_channels: int = 3, image_size: int = 32) -> ScoreUNet:
    """Construye una ScoreUNet con la config tiny del diseño para la resolución dada."""
    return ScoreUNet(
        in_channels=in_channels,
        image_size=image_size,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        embed_dim=8,
        time_embed_dim=16,
        groups=4,
        attn_resolutions=(16,),
    )


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("image_size", [32, 64])
def test_scoreunet_output_shape(in_channels, image_size):
    # Contrato 1.1 / 2.1 / 2.2: (B, C, H, W) -> (B, C, H, W) en float32 para los
    # canales candidatos (grises y RGB) y las resoluciones de referencia.
    net = _tiny_unet(in_channels=in_channels, image_size=image_size)
    x = torch.randn(2, in_channels, image_size, image_size)
    out = net(x, torch.rand(2))
    assert out.shape == (2, in_channels, image_size, image_size)
    assert out.dtype == torch.float32


def test_scoreunet_accepts_both_t_shapes():
    # Contrato 1.2: t como (B,) o (B, 1) -> el mismo resultado (lo normaliza el
    # embedding reusado).
    net = _tiny_unet().eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    with torch.no_grad():
        a = net(x, t)                 # t de shape (B,)
        b = net(x, t.reshape(2, 1))   # t de shape (B, 1)
    assert a.shape == b.shape == (2, 3, 32, 32)
    assert torch.equal(a, b)          # ambas formas dan el mismo resultado


@pytest.mark.parametrize(
    "t",
    [
        torch.linspace(0.0, 1.0, 2),       # rango [0, 1]
        torch.linspace(0.0, 1000.0, 2),    # rango [0, T]
        torch.arange(2).float(),           # pasos enteros
    ],
)
def test_scoreunet_any_t_scale_finite(t):
    # Contrato 1.3: salidas finitas para las escalas de tiempo usadas por las SDEs.
    net = _tiny_unet().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = net(x, t)
    assert out.shape == (2, 3, 32, 32)
    assert torch.all(torch.isfinite(out))


def test_scoreunet_time_conditioning_effective():
    # Contrato 1.4: mismo x, dos tiempos distintos -> salidas distintas (el
    # condicionamiento temporal es efectivo).
    net = _tiny_unet().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        a = net(x, torch.zeros(2))
        b = net(x, torch.ones(2))
    assert not torch.allclose(a, b)


def test_scoreunet_output_unbounded_both_signs():
    # Contrato 1.5: salida no acotada -> valores positivos y negativos (ninguna
    # activación final la restringe).
    net = _tiny_unet().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = net(x, torch.rand(2))
    assert (out > 0).any() and (out < 0).any()


def test_scoreunet_satisfies_scoremodel_protocol():
    # Contrato 1.6: satisface el Protocol ScoreModel estructuralmente (sin herencia).
    net = _tiny_unet()
    assert isinstance(net, ScoreModel)


# Determinismo (Req 3): la red es la variable de control -> enteramente determinística.


def test_scoreunet_is_deterministic():
    # Contrato 3.1: mismo (x, t) evaluado dos veces en eval -> salida bitwise idéntica
    # (mismo grafo, mismas rutas de cómputo).
    net = _tiny_unet().eval()
    x, t = torch.randn(2, 3, 32, 32), torch.rand(2)
    with torch.no_grad():
        assert torch.equal(net(x, t), net(x, t))


def test_scoreunet_no_stochastic_layers():
    # Contrato 3.2: sin dropout ni batchnorm (la normalización es GroupNorm,
    # independiente del batch); la red debe ser enteramente determinística.
    net = _tiny_unet()
    for module in net.modules():
        assert not isinstance(module, torch.nn.Dropout)
        assert not isinstance(
            module,
            (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
        )


def test_scoreunet_batch_independence():
    # Contrato 3.3: una misma muestra evaluada sola o dentro de un batch da salidas
    # numéricamente equivalentes (la normalización no depende del resto del batch).
    # No es bitwise en CPU float32 (~6e-07 por paralelización de convs); allclose(atol=1e-6).
    net = _tiny_unet().eval()
    x_single = torch.randn(1, 3, 32, 32)
    t_single = torch.rand(1)
    # La primera fila del batch es exactamente la muestra evaluada sola.
    x_batch = torch.cat([x_single, torch.randn(3, 3, 32, 32)], dim=0)
    t_batch = torch.cat([t_single, torch.rand(3)], dim=0)
    with torch.no_grad():
        out_single = net(x_single, t_single)
        out_batch = net(x_batch, t_batch)
    assert torch.allclose(out_single, out_batch[0:1], atol=1e-6)


def test_scoreunet_gradients_flow():
    # Contrato 3.4: backward sobre una salida -> gradientes finitos en todos los
    # parámetros entrenables (ninguno queda desconectado del grafo).
    net = _tiny_unet()
    x, t = torch.randn(2, 3, 32, 32), torch.rand(2)
    net(x, t).pow(2).sum().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.all(torch.isfinite(g)) for g in grads)


# Configuración y errores (Req 4 + 2.3): defaults como arquitectura de referencia,
# conteo de parámetros reproducible y validaciones fail-fast con ValueError.


def test_scoreunet_param_count_reproducible():
    # Contrato 4.1 / 4.2: dos instancias con los MISMOS argumentos tienen exactamente
    # el mismo conteo de parámetros entrenables (la arquitectura es reproducible, sin
    # números mágicos escondidos que varíen entre construcciones).
    a = _tiny_unet()
    b = _tiny_unet()
    n_a = sum(p.numel() for p in a.parameters())
    n_b = sum(p.numel() for p in b.parameters())
    assert n_a == n_b
    assert n_a > 0


def test_scoreunet_unknown_activation_raises():
    # Contrato 4.3: activación con nombre desconocido -> ValueError (mismo registry de
    # activaciones que el resto del módulo, vía _make_activation). _tiny_unet no expone
    # activation, así que se construye ScoreUNet directo con anchos tiny + config válida.
    with pytest.raises(ValueError):
        ScoreUNet(
            in_channels=3,
            image_size=32,
            base_channels=8,
            channel_mults=(1, 2),
            num_res_blocks=1,
            embed_dim=8,
            time_embed_dim=16,
            groups=4,
            attn_resolutions=(16,),
            activation="no_existe",
        )


def test_scoreunet_incompatible_groups_raises():
    # Contrato 2.3 / 4.3: groups debe dividir a todos los anchos de canal; groups=3
    # contra el nivel base de 8 (8 % 3 != 0) -> ValueError en construcción. image_size
    # y embed_dim se dejan válidos para aislar la infracción de grupos.
    with pytest.raises(ValueError):
        ScoreUNet(
            in_channels=3,
            image_size=32,
            base_channels=8,
            channel_mults=(1, 2),
            num_res_blocks=1,
            embed_dim=8,
            time_embed_dim=16,
            groups=3,
            attn_resolutions=(16,),
        )


def test_scoreunet_indivisible_image_size_raises():
    # Contrato 2.3: la resolución de trabajo debe ser divisible por el factor total de
    # reducción 2**(len(channel_mults)-1); con channel_mults=(1, 2) el factor es 2 y un
    # image_size impar (15) no es divisible -> ValueError en construcción.
    with pytest.raises(ValueError):
        ScoreUNet(
            in_channels=3,
            image_size=15,
            base_channels=8,
            channel_mults=(1, 2),
            num_res_blocks=1,
            embed_dim=8,
            time_embed_dim=16,
            groups=4,
            attn_resolutions=(16,),
        )


def test_scoreunet_wrong_input_size_raises():
    # Contrato 2.3: la arquitectura queda fijada por image_size en construcción; un
    # forward con H/W distintos de image_size (red de 32, entrada de 16) -> ValueError.
    net = _tiny_unet(image_size=32)
    x = torch.randn(2, 3, 16, 16)
    with pytest.raises(ValueError):
        net(x, torch.rand(2))


def test_scoreunet_reference_defaults_forward():
    # Contrato 4.1 / 5.4: ÚNICO test que instancia los defaults (la arquitectura de
    # referencia del estudio) y corre un forward completo bajo pytest. Excepción
    # deliberada a la config tiny (research.md, ~100-200 ms): batch 1 y sin parametrizar
    # para no multiplicar el costo. Cubre el camino de 4 niveles con mult 4.
    net = ScoreUNet()
    x = torch.randn(1, 3, 64, 64)
    out = net(x, torch.rand(1))
    assert out.shape == (1, 3, 64, 64)


# ------------------------------------------------ make_model / registry (Req 5.3, 6.1)
# Factory por nombre additivo, espejo de make_sde / make_distribution: construye la red
# desde una receta (name, kwargs) para el config-driven y la reconstrucción de checkpoints.


def test_make_model_mlp_returns_scoremlp():
    # make_model("mlp", ...) devuelve un ScoreMLP usable (nn.Module con forward válido).
    net = make_model("mlp", data_dim=2)
    assert isinstance(net, ScoreMLP)
    assert isinstance(net, torch.nn.Module)
    out = net(torch.randn(4, 2), torch.rand(4))
    assert out.shape == (4, 2)


def test_make_model_unet_returns_scoreunet():
    # make_model("unet", ...) devuelve un ScoreUNet usable; se pasan los anchos tiny para
    # mantener el costo del test en el orden del resto de la suite.
    net = make_model(
        "unet",
        in_channels=3,
        image_size=32,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        embed_dim=8,
        time_embed_dim=16,
        groups=4,
        attn_resolutions=(16,),
    )
    assert isinstance(net, ScoreUNet)
    assert isinstance(net, torch.nn.Module)
    out = net(torch.randn(2, 3, 32, 32), torch.rand(2))
    assert out.shape == (2, 3, 32, 32)


def test_make_model_satisfies_scoremodel_protocol():
    # Lo que construye el registry satisface el Protocol ScoreModel (contrato (x, t) -> score).
    assert isinstance(make_model("mlp", data_dim=2), ScoreModel)


def test_available_models_expected_set():
    # available_models() == conjunto esperado, y REGISTRY mapea a las clases correctas.
    assert set(available_models()) == {"mlp", "unet"}
    assert REGISTRY["mlp"] is ScoreMLP
    assert REGISTRY["unet"] is ScoreUNet


def test_make_model_unknown_name_raises():
    # Nombre desconocido -> ValueError que nombra las opciones válidas (patrón del repo).
    with pytest.raises(ValueError):
        make_model("no_existe")


def test_make_model_filters_unknown_kwargs():
    # Espejo de make_sde / make_distribution: los kwargs que no aplican a la red se
    # descartan (se filtran por la firma del constructor), así un caller genérico puede
    # pasar siempre el mismo conjunto de parámetros sin que falle la construcción.
    net = make_model("mlp", data_dim=2, no_aplica_a_mlp=123)
    assert isinstance(net, ScoreMLP)
    assert net.data_dim == 2
