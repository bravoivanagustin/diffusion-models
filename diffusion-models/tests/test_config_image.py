"""Tests de la feature ``config-image-training`` (dispatch de fuente por ``kind``).

Torch es dependencia dura del camino de datos, así que se hace ``importorskip`` al tope. Este
archivo cubre el **resolver de fuente** :func:`build_data_source` y su cableado en
:func:`build_run`. La Task 1.1 sólo ejercita el camino de **puntos** (retrocompatible) y el
dispatch por ``kind`` (default ``points``, kind desconocido → error). El camino de imágenes se
agrega en tareas posteriores.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.models import ScoreMLP, ScoreModel
from diffusion.training import RunSpec, build_data_source, build_run, load_config  # noqa: F401


# ------------------------------------------------------------ dispatch / puntos


def test_build_data_source_puntos_sin_kind():
    """Sin ``kind`` la fuente es de puntos: devuelve ``(iterador, None)`` y yield-ea ``(B, dim)``.

    ``event_shape`` es ``None`` para puntos (no hay forma de evento multidimensional). El batch
    respeta el ``batch_size`` del bloque ``data:`` y la dimensión ``dim`` de la distribución.
    """
    data, event_shape = build_data_source(
        {"shape": "gaussian", "dim": 2, "n_samples": 64, "batch_size": 16}
    )

    assert event_shape is None
    batch = next(iter(data))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (16, 2)


def test_build_data_source_puntos_kind_explicito():
    """``kind: points`` explícito recorre el mismo camino que su ausencia (retrocompat)."""
    data, event_shape = build_data_source(
        {"kind": "points", "shape": "mixture", "dim": 2, "n_samples": 64,
         "batch_size": 8, "n_components": 4, "seed": 0}
    )

    assert event_shape is None
    batch = next(iter(data))
    assert batch.shape == (8, 2)


def test_build_data_source_acepta_name_ademas_de_shape():
    """El alias ``name`` para la forma sigue funcionando (como en el ``build_run`` previo)."""
    data, event_shape = build_data_source(
        {"name": "gaussian", "dim": 2, "n_samples": 32, "batch_size": 4}
    )

    assert event_shape is None
    assert next(iter(data)).shape == (4, 2)


def test_build_data_source_falta_forma_es_value_error():
    """Sin ``shape``/``name`` el resolver falla con ``ValueError`` (como hoy en ``build_run``)."""
    with pytest.raises(ValueError):
        build_data_source({"dim": 2})


# ------------------------------------------------------------ kind desconocido


def test_build_data_source_kind_desconocido_lista_validos():
    """Un ``kind`` no reconocido levanta ``ValueError`` que enumera los válidos (1.4)."""
    with pytest.raises(ValueError, match="points.*images|images.*points") as exc:
        build_data_source({"kind": "bogus", "shape": "gaussian", "dim": 2})

    msg = str(exc.value)
    assert "bogus" in msg
    assert "points" in msg
    assert "images" in msg


# ------------------------------------------------------ retrocompat via build_run


def test_build_run_puntos_sigue_armando_runspec_con_mlp():
    """``build_run`` sobre un config de puntos arma el mismo ``RunSpec`` que antes (sin regresión).

    El resolver es ahora el único dueño del parseo de ``data:``; ``build_run`` obtiene la fuente
    vía él. El comportamiento observable del camino de puntos no cambia: red MLP dimensionada
    desde la SDE e iterador infinito de tensores ``(B, dim)``.
    """
    raw = {
        "sde": {"name": "vp", "beta_min": 0.1, "beta_max": 20.0},
        "data": {
            "shape": "mixture", "dim": 2, "n_samples": 512, "batch_size": 128,
            "n_components": 8, "seed": 0,
        },
        "train": {"num_steps": 3, "lr": 1e-3, "seed": 0},
        "out": {"checkpoint": "models/x.pt", "loss_curve": "models/x.png"},
    }
    spec = build_run(raw)

    assert isinstance(spec, RunSpec)
    assert spec.sde.name == "vp"
    assert spec.config.num_steps == 3
    assert isinstance(spec.model, ScoreModel)
    assert isinstance(spec.model, ScoreMLP)
    assert spec.model.data_dim == spec.sde.data_dim
    batch = next(iter(spec.data))
    assert batch.shape == (128, 2)
    assert spec.checkpoint.name == "x.pt"
    assert spec.loss_curve.name == "x.png"


def test_build_data_source_no_muta_dict_del_caller():
    """El resolver no debe mutar el ``dict`` del caller (copia defensiva antes de ``pop``)."""
    raw = {"shape": "gaussian", "dim": 2, "n_samples": 32, "batch_size": 4}
    snapshot = dict(raw)
    build_data_source(raw)
    assert raw == snapshot


# ------------------------------------------------------------------- imágenes

# La fuente de imágenes necesita torchvision (transforms + ImageFolder-like).
torchvision = pytest.importorskip("torchvision")
Image = pytest.importorskip("PIL.Image")

import diffusion.training.config as config_module  # noqa: E402


@pytest.fixture
def carpeta_imagenes(tmp_path):
    """Carpeta temporal con un puñado de imágenes RGB chicas (count ≥ batch_size)."""
    for i in range(8):
        Image.new("RGB", (32, 32), color=(i * 20, 40, 200 - i * 10)).save(
            tmp_path / f"x{i}.png"
        )
    return tmp_path


def test_build_data_source_imagenes_happy_path(carpeta_imagenes):
    """``kind: images`` arma la fuente y deriva ``(3, image_size, image_size)`` (1.3, 2.1, 3.1).

    El iterador devuelto yield-ea un tensor ``(B, 3, image_size, image_size)`` float32; la forma
    de evento fija los canales en 3.
    """
    data, event_shape = build_data_source(
        {"kind": "images", "root": str(carpeta_imagenes), "image_size": 16, "batch_size": 4}
    )

    assert event_shape == (3, 16, 16)
    batch = next(data)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 3, 16, 16)
    assert batch.dtype == torch.float32


def test_build_data_source_imagenes_falta_root_es_value_error():
    """Sin ``root`` el resolver falla con ``ValueError`` claro (2.3)."""
    with pytest.raises(ValueError, match="root"):
        build_data_source({"kind": "images"})


def test_build_data_source_imagenes_root_inexistente_propaga_value_error(tmp_path):
    """Un ``root`` inexistente propaga el ``ValueError`` de ``infinite_batches`` (2.3)."""
    inexistente = tmp_path / "no_existe"
    with pytest.raises(ValueError):
        build_data_source({"kind": "images", "root": str(inexistente), "batch_size": 2})


def test_build_data_source_imagenes_clave_desconocida_es_value_error(carpeta_imagenes):
    """Una clave no mapeada en ``data:`` para imágenes falla enumerándola (rechazo de unknowns)."""
    with pytest.raises(ValueError, match="bogus") as exc:
        build_data_source(
            {"kind": "images", "root": str(carpeta_imagenes), "batch_size": 4, "bogus": 1}
        )

    assert "desconocida" in str(exc.value).lower()


def test_build_data_source_imagenes_forma_inesperada_es_value_error(
    carpeta_imagenes, monkeypatch
):
    """Si la fuente emite una forma distinta de la derivada, el peek levanta ``ValueError`` (3.2).

    La fuente real siempre coacciona al tamaño pedido, así que para ejercitar genuinamente el
    guard del peek se falsea ``infinite_batches`` con una que yield-ea la forma equivocada.
    """

    def fake_infinite_batches(root, batch_size, **kwargs):
        while True:
            yield torch.zeros(batch_size, 3, 8, 8)  # 8x8 ≠ image_size=16 pedido

    monkeypatch.setattr(config_module, "infinite_batches", fake_infinite_batches)

    with pytest.raises(ValueError, match="emite forma|esperaba"):
        build_data_source(
            {"kind": "images", "root": str(carpeta_imagenes), "image_size": 16, "batch_size": 4}
        )
