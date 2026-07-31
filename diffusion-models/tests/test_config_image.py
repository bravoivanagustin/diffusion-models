"""Tests de la feature ``config-image-training`` (dispatch de fuente por ``kind``).

Torch es dependencia dura del camino de datos, así que se hace ``importorskip`` al tope. Este
archivo cubre el **resolver de fuente** :func:`build_data_source` y su cableado en
:func:`build_run`. La Task 1.1 sólo ejercita el camino de **puntos** (retrocompatible) y el
dispatch por ``kind`` (default ``points``, kind desconocido → error). El camino de imágenes se
agrega en tareas posteriores.

La feature ``validation-loss`` (task 6.1) cambió el contrato del resolver: devuelve un
:class:`~diffusion.training.DataSources` en lugar de la 2-tupla ``(data, event_shape)``, y suma
la clave ``data.val_root``. Los tests previos se migraron a la firma nueva en el mismo paso.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.models import ScoreMLP, ScoreModel
from diffusion.training import (  # noqa: F401
    DataSources,
    RunSpec,
    build_data_source,
    build_run,
    load_config,
)


# ------------------------------------------------------------ dispatch / puntos


def test_build_data_source_puntos_sin_kind():
    """Sin ``kind`` la fuente es de puntos: ``DataSources`` con ``event_shape`` nulo y ``(B, dim)``.

    ``event_shape`` es ``None`` para puntos (no hay forma de evento multidimensional). El batch
    respeta el ``batch_size`` del bloque ``data:`` y la dimensión ``dim`` de la distribución.
    """
    sources = build_data_source(
        {"shape": "gaussian", "dim": 2, "n_samples": 64, "batch_size": 16}
    )

    assert isinstance(sources, DataSources)
    assert sources.event_shape is None
    batch = next(iter(sources.train))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (16, 2)


def test_build_data_source_puntos_kind_explicito():
    """``kind: points`` explícito recorre el mismo camino que su ausencia (retrocompat)."""
    sources = build_data_source(
        {"kind": "points", "shape": "mixture", "dim": 2, "n_samples": 64,
         "batch_size": 8, "n_components": 4, "seed": 0}
    )

    assert sources.event_shape is None
    batch = next(iter(sources.train))
    assert batch.shape == (8, 2)


def test_build_data_source_acepta_name_ademas_de_shape():
    """El alias ``name`` para la forma sigue funcionando (como en el ``build_run`` previo)."""
    sources = build_data_source(
        {"name": "gaussian", "dim": 2, "n_samples": 32, "batch_size": 4}
    )

    assert sources.event_shape is None
    assert next(iter(sources.train)).shape == (4, 2)


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
    # validation-loss (2.4): sin 'data.val_root' la corrida no trae fuente de validación.
    assert spec.val_data is None


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
    sources = build_data_source(
        {"kind": "images", "root": str(carpeta_imagenes), "image_size": 16, "batch_size": 4}
    )

    assert sources.event_shape == (3, 16, 16)
    batch = next(sources.train)
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, 3, 16, 16)
    assert batch.dtype == torch.float32
    # validation-loss (2.4): sin 'val_root' las fuentes de examen quedan nulas.
    assert sources.val is None


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


# --------------------------------------------- knobs de carga (num_workers/pin_memory)


def _spy_infinite_batches(captura, image_size):
    """Fábrica de un doble de ``infinite_batches`` que captura kwargs y yield-ea la forma pedida.

    Guarda ``root``/``batch_size`` y todos los kwargs en ``captura`` (dict) y devuelve tensores
    ``(B, 3, image_size, image_size)`` para que el peek de validación de la forma pase. Evita
    spawnear procesos de carga reales en Windows CI.
    """

    def fake(root, batch_size, **kwargs):
        captura["root"] = root
        captura["batch_size"] = batch_size
        captura.update(kwargs)
        while True:
            yield torch.zeros(batch_size, 3, image_size, image_size)

    return fake


def test_build_data_source_imagenes_propaga_knobs_de_carga(carpeta_imagenes, monkeypatch):
    """Declarar ``num_workers``/``pin_memory`` los propaga a la fuente de imágenes (3.1).

    Se espía ``infinite_batches`` en el módulo de config para capturar los kwargs con los que se
    construye la fuente, sin spawnear workers reales.
    """
    captura: dict = {}
    monkeypatch.setattr(
        config_module, "infinite_batches", _spy_infinite_batches(captura, image_size=16)
    )

    sources = build_data_source(
        {
            "kind": "images",
            "root": str(carpeta_imagenes),
            "image_size": 16,
            "batch_size": 4,
            "num_workers": 3,
            "pin_memory": True,
        }
    )

    assert sources.event_shape == (3, 16, 16)
    assert captura["num_workers"] == 3
    assert captura["pin_memory"] is True


def test_build_data_source_imagenes_knobs_defaults(carpeta_imagenes, monkeypatch):
    """Sin declarar los knobs, la fuente se arma con los defaults actuales (3.2).

    ``num_workers=0`` (carga en el proceso principal) y ``pin_memory=False`` (sin memoria fijada):
    sin cambio de comportamiento observable respecto de antes de la feature.
    """
    captura: dict = {}
    monkeypatch.setattr(
        config_module, "infinite_batches", _spy_infinite_batches(captura, image_size=16)
    )

    build_data_source(
        {"kind": "images", "root": str(carpeta_imagenes), "image_size": 16, "batch_size": 4}
    )

    assert captura["num_workers"] == 0
    assert captura["pin_memory"] is False


def test_build_data_source_imagenes_clave_de_carga_desconocida_es_value_error(carpeta_imagenes):
    """Una clave de carga desconocida sigue fallando enumerándola tras sumar los knobs (3.4, 5.3)."""
    with pytest.raises(ValueError, match="num_workerss") as exc:
        build_data_source(
            {
                "kind": "images",
                "root": str(carpeta_imagenes),
                "batch_size": 4,
                "num_workerss": 2,  # typo: clave desconocida, no debe colarse
            }
        )

    assert "desconocida" in str(exc.value).lower()


# --------------------------------------------- data.val_root (validation-loss, 6.1)


def _imagen_asimetrica(path, ancho=32, alto=32):
    """Guarda una imagen con la mitad izquierda negra y la derecha blanca.

    Sirve para detectar espejado horizontal: la fuente de validación se arma con
    ``augment=False``, así que la orientación tiene que ser siempre la canónica (izquierda
    oscura, derecha clara) en todos los recorridos.
    """
    img = Image.new("RGB", (ancho, alto), color=(0, 0, 0))
    for x in range(ancho // 2, ancho):
        for y in range(alto):
            img.putpixel((x, y), (255, 255, 255))
    img.save(path)


@pytest.fixture
def carpetas_train_val(tmp_path):
    """Dos carpetas **hermanas** con imágenes: ``train/`` (8) y ``val/`` (6, asimétricas).

    Hermanas y no anidadas a propósito: el descubrimiento de imágenes es recursivo, así que un
    ``val/`` dentro del ``root`` de entrenamiento contaminaría el set de train.
    """
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    for i in range(8):
        Image.new("RGB", (32, 32), color=(i * 20, 40, 200 - i * 10)).save(
            train_dir / f"t{i}.png"
        )
    for i in range(6):
        _imagen_asimetrica(val_dir / f"v{i}.png")
    return train_dir, val_dir


def _data_raw(train_dir, val_dir=None, **extra):
    """Bloque ``data:`` de imágenes, con ``val_root`` solo si se pasa (clave opt-in)."""
    raw = {"kind": "images", "root": str(train_dir), "image_size": 16, "batch_size": 4}
    if val_dir is not None:
        raw["val_root"] = str(val_dir)
    raw.update(extra)
    return raw


def test_build_data_source_val_root_arma_fuente_de_validacion(carpetas_train_val):
    """``data.val_root`` arma una segunda fuente con la MISMA forma de evento (2.1, 2.3).

    La fuente de validación es finita: recorre las 6 imágenes completas (incluida la cola
    parcial) con el mismo ``batch_size`` y el mismo ``image_size`` que la de entrenamiento.
    """
    train_dir, val_dir = carpetas_train_val
    sources = build_data_source(_data_raw(train_dir, val_dir))

    assert sources.val is not None
    batches = list(sources.val)
    assert [tuple(b.shape) for b in batches] == [(4, 3, 16, 16), (2, 3, 16, 16)]
    # 2.3: la forma de evento de la validación coincide con la derivada para el entrenamiento.
    assert all(tuple(b.shape[1:]) == sources.event_shape for b in batches)
    assert sum(b.shape[0] for b in batches) == 6  # el set completo, sin descartar la cola


def test_build_data_source_val_root_es_reiterable(carpetas_train_val):
    """La fuente de validación es **re-iterable**: dos recorridos dan los mismos tensores (2.2).

    Es la condición que hace reproducible el examen fijo, y de paso descarta cualquier aumento
    de datos aleatorio (un espejado al azar haría diferir los dos recorridos).
    """
    train_dir, val_dir = carpetas_train_val
    sources = build_data_source(_data_raw(train_dir, val_dir))

    primera = [b.clone() for b in sources.val]
    segunda = [b.clone() for b in sources.val]

    assert len(primera) == len(segunda) == 2
    for a, b in zip(primera, segunda):
        assert torch.equal(a, b)


def test_build_data_source_val_root_orientacion_canonica_sin_augmentation(carpetas_train_val):
    """La validación se consume **sin espejado**: orientación canónica en todos los pasos (2.2).

    Las imágenes de validación son asimétricas (mitad izquierda negra, derecha blanca). Con
    ``augment=True`` el volteo horizontal aleatorio invertiría ese patrón en algunas imágenes;
    acá se verifica que **ninguna** aparece invertida, en dos recorridos completos.
    """
    train_dir, val_dir = carpetas_train_val
    sources = build_data_source(_data_raw(train_dir, val_dir))

    for _ in range(2):
        for batch in sources.val:
            izquierda = batch[:, :, :, 0]   # primera columna: negro → normalizado a -1
            derecha = batch[:, :, :, -1]    # última columna: blanco → normalizado a +1
            assert bool((izquierda < 0).all()), "la validación llegó espejada"
            assert bool((derecha > 0).all()), "la validación llegó espejada"


def test_build_data_source_val_root_hereda_params_de_la_fuente_de_train(
    carpetas_train_val, monkeypatch
):
    """La validación se deriva con el mismo batch/imagen/encuadre y carga in-process (2.3).

    Se espía :func:`finite_batches` en el módulo de config para capturar con qué se construye la
    fuente: ``batch_size``/``image_size``/``crop`` copiados de la de entrenamiento, y
    ``num_workers=0``/``pin_memory=False`` forzados (una evaluación no amortiza workers).
    """
    train_dir, val_dir = carpetas_train_val
    captura: dict = {}
    real_finite = config_module.finite_batches

    def spy(root, batch_size, **kwargs):
        captura["root"] = root
        captura["batch_size"] = batch_size
        captura.update(kwargs)
        return real_finite(root, batch_size, **kwargs)

    monkeypatch.setattr(config_module, "finite_batches", spy)

    build_data_source(_data_raw(train_dir, val_dir, image_size=8, batch_size=3, crop=False))

    assert captura["root"] == str(val_dir)
    assert captura["batch_size"] == 3        # mismo batch que la fuente de entrenamiento
    assert captura["image_size"] == 8        # mismo tamaño de imagen
    assert captura["crop"] is False          # mismo modo de encuadre
    assert captura["num_workers"] == 0
    assert captura["pin_memory"] is False


def test_build_data_source_val_root_mas_chico_que_batch_size_es_batch_parcial(tmp_path):
    """Un set de validación más chico que el batch se acepta como un único batch parcial (2.7).

    A diferencia de la fuente de entrenamiento (que con ``drop_last=True`` exige el mínimo), la
    finita no descarta la cola: 3 imágenes con ``batch_size=8`` son un batch parcial válido.
    """
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()
    for i in range(8):
        Image.new("RGB", (32, 32), color=(i * 20, 40, 60)).save(train_dir / f"t{i}.png")
    for i in range(3):
        Image.new("RGB", (32, 32), color=(10, 20, 30)).save(val_dir / f"v{i}.png")

    sources = build_data_source(_data_raw(train_dir, val_dir, batch_size=8))

    batches = list(sources.val)
    assert len(batches) == 1
    assert tuple(batches[0].shape) == (3, 3, 16, 16)


def test_build_data_source_val_root_inexistente_es_value_error(carpetas_train_val, tmp_path):
    """Un ``val_root`` inexistente aborta con ``ValueError`` nombrando la ruta (2.5)."""
    train_dir, _ = carpetas_train_val
    inexistente = tmp_path / "no_existe"

    with pytest.raises(ValueError, match="no_existe"):
        build_data_source(_data_raw(train_dir, inexistente))


def test_build_data_source_val_root_vacio_es_value_error(carpetas_train_val, tmp_path):
    """Un ``val_root`` sin imágenes aborta con ``ValueError`` nombrando la ruta (2.5)."""
    train_dir, _ = carpetas_train_val
    vacio = tmp_path / "vacio"
    vacio.mkdir()

    with pytest.raises(ValueError, match="No se encontraron imágenes"):
        build_data_source(_data_raw(train_dir, vacio))


def test_build_data_source_val_root_con_puntos_es_value_error(tmp_path):
    """``val_root`` sobre una fuente de puntos aborta explícitamente (2.6).

    Sin el chequeo la clave se ignoraría **en silencio**: ``make_distribution`` filtra kwargs por
    firma, así que ``val_root`` se descartaría sin avisar y la corrida entrenaría sin validación.
    """
    with pytest.raises(ValueError, match="val_root") as exc:
        build_data_source(
            {"shape": "gaussian", "dim": 2, "n_samples": 32, "batch_size": 4,
             "val_root": str(tmp_path)}
        )

    msg = str(exc.value)
    assert "images" in msg


def test_build_data_source_val_root_forma_inesperada_es_value_error(
    carpetas_train_val, monkeypatch
):
    """Si la fuente de validación emite otra forma, el peek levanta ``ValueError`` (2.3).

    La fuente real siempre coacciona al tamaño pedido, así que se falsea :func:`finite_batches`
    con una que emite la forma equivocada para ejercitar genuinamente el guard.
    """
    train_dir, val_dir = carpetas_train_val

    def fake_finite_batches(root, batch_size, **kwargs):
        return [torch.zeros(batch_size, 3, 8, 8)]  # 8x8 ≠ image_size=16 pedido

    monkeypatch.setattr(config_module, "finite_batches", fake_finite_batches)

    with pytest.raises(ValueError, match="validación|val_root"):
        build_data_source(_data_raw(train_dir, val_dir))


def test_build_data_source_val_root_no_muta_dict_del_caller(carpetas_train_val):
    """El resolver copia el bloque ``data:`` antes de popear ``val_root`` (no muta al caller)."""
    train_dir, val_dir = carpetas_train_val
    raw = _data_raw(train_dir, val_dir)
    snapshot = dict(raw)

    build_data_source(raw)

    assert raw == snapshot


# ------------------------------------------------- build_run: cableado imágenes

from diffusion.models import ScoreUNet  # noqa: E402


def test_build_run_imagenes_cablea_sde_y_unet(carpeta_imagenes):
    """``build_run`` con ``kind: images`` cablea la SDE con la forma derivada y default-ea U-Net.

    La forma de evento ``(3, H, W)`` sale del bloque ``data:`` (única fuente de verdad) y se pasa
    a ``make_sde`` como ``data_dim``; sin bloque ``model:`` la red default-ea a ``unet`` (2.2). El
    ``RunSpec`` conserva la misma estructura que el camino de puntos (2.4): mismos campos poblados.
    """
    raw = {
        "sde": {"name": "vp"},
        "data": {
            "kind": "images", "root": str(carpeta_imagenes),
            "image_size": 16, "batch_size": 4,
        },
    }
    spec = build_run(raw)

    assert isinstance(spec, RunSpec)
    assert spec.sde.data_dim == (3, 16, 16)  # 3.1: forma derivada → sde.data_dim
    assert isinstance(spec.model, ScoreUNet)  # 2.2: default unet para imágenes
    # 2.4: misma estructura de RunSpec que el camino toy (mismos campos poblados por train).
    assert spec.sde is not None
    assert spec.model is not None
    assert spec.data is not None
    assert spec.config is not None
    assert spec.model_spec is not None
    assert spec.model_spec["name"] == "unet"


def test_build_run_imagenes_data_dim_en_sde_es_value_error(carpeta_imagenes):
    """Declarar ``data_dim`` en ``sde:`` para imágenes falla: única fuente de verdad en ``data:`` (3.3)."""
    raw = {
        "sde": {"name": "vp", "data_dim": [3, 16, 16]},
        "data": {
            "kind": "images", "root": str(carpeta_imagenes),
            "image_size": 16, "batch_size": 4,
        },
    }
    with pytest.raises(ValueError, match="data_dim"):
        build_run(raw)


def test_build_run_imagenes_model_unet_explicito_respetado(carpeta_imagenes):
    """Un ``model: {name: unet, ...}`` explícito se respeta (sigue siendo una ``ScoreUNet``)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {
            "kind": "images", "root": str(carpeta_imagenes),
            "image_size": 16, "batch_size": 4,
        },
        "model": {"name": "unet", "base_channels": 16},
    }
    spec = build_run(raw)

    assert isinstance(spec.model, ScoreUNet)
    assert spec.sde.data_dim == (3, 16, 16)


def test_build_run_imagenes_val_root_cablea_val_data(carpetas_train_val):
    """``build_run`` copia la fuente de validación del resolver al ``RunSpec`` (2.1, 2.3).

    El ``RunSpec`` gana ``val_data``: no nula cuando el config declara ``data.val_root``, con la
    misma forma de evento que la fuente de entrenamiento (la que alimenta la SDE).
    """
    train_dir, val_dir = carpetas_train_val
    raw = {"sde": {"name": "vp"}, "data": _data_raw(train_dir, val_dir)}

    spec = build_run(raw)

    assert spec.val_data is not None
    assert spec.sde.data_dim == (3, 16, 16)
    batch = next(iter(spec.val_data))
    assert tuple(batch.shape[1:]) == spec.sde.data_dim


def test_build_run_imagenes_sin_val_root_val_data_es_none(carpetas_train_val):
    """Sin ``data.val_root`` el ``RunSpec`` sale igual que antes de la feature (2.4).

    ``val_data`` nula y todos los campos previos poblados como siempre: la clave ausente ⇒
    comportamiento sin cambios (la validación es opt-in).
    """
    train_dir, _ = carpetas_train_val
    raw = {"sde": {"name": "vp"}, "data": _data_raw(train_dir)}

    spec = build_run(raw)

    assert spec.val_data is None
    assert spec.sde.data_dim == (3, 16, 16)
    assert isinstance(spec.model, ScoreUNet)
    assert spec.model_spec["name"] == "unet"
    assert next(iter(spec.data)).shape == (4, 3, 16, 16)


def test_build_run_puntos_val_root_es_value_error(tmp_path):
    """``build_run`` propaga el rechazo de ``val_root`` en una corrida de puntos (2.6)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 64, "batch_size": 8,
                 "val_root": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="val_root"):
        build_run(raw)


def test_build_run_puntos_data_dim_entero_y_mlp():
    """Regresión de puntos (2.4): ``sde.data_dim == 2`` y red MLP (sin cambio de comportamiento)."""
    raw = {
        "sde": {"name": "vp"},
        "data": {"shape": "gaussian", "dim": 2, "n_samples": 64, "batch_size": 8},
    }
    spec = build_run(raw)

    assert spec.sde.data_dim == 2
    assert isinstance(spec.model, ScoreMLP)


# --------------------------------------------- config de ejemplo (plantilla)

import pathlib  # noqa: E402

from diffusion.training import train  # noqa: E402

# La config de ejemplo vive en diffusion-models/config/ (hermana de tests/).
_EXAMPLE_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "config" / "vp_cats.yaml"


def test_config_ejemplo_imagenes_parsea_y_tiene_estructura():
    """La config de ejemplo ``vp_cats.yaml`` (kind: images + model: unet) parsea y está bien armada.

    Es un test estructural (lint) de la plantilla: su ``data.root`` puede no existir en CI, así que
    NO se corre ``build_run`` sobre ella (eso escanearía la carpeta). Se afirma que declara una celda
    de imágenes con U-Net: ``kind: images``, ``root`` e ``image_size`` presentes, red ``unet`` y SDE
    nombrada (2.4).
    """
    assert _EXAMPLE_CONFIG.exists(), f"falta la config de ejemplo: {_EXAMPLE_CONFIG}"
    cfg = load_config(str(_EXAMPLE_CONFIG))

    # bloque data: fuente de imágenes con los campos obligatorios del contrato `kind: images`.
    assert cfg["data"]["kind"] == "images"
    assert cfg["data"]["root"]                 # ruta presente (obligatoria para imágenes)
    assert cfg["data"]["image_size"]           # tamaño presente (deriva la forma de evento)
    # bloque model: la red default-eable a unet queda declarada explícitamente en la plantilla.
    assert cfg["model"]["name"] == "unet"
    # bloque sde: nombrada (Eje 1); sin data_dim (se deriva de data:, única fuente de verdad).
    assert cfg["sde"]["name"]
    assert "data_dim" not in cfg["sde"]


# ----------------------------------------------------------- smoke end-to-end


def test_smoke_e2e_imagenes_build_run_y_train(carpeta_imagenes):
    """Smoke integral: ``build_run`` de imágenes → ``train`` unos pocos pasos con la U-Net (2.4).

    Arma su PROPIA config (no la de ejemplo, cuyo ``root`` puede faltar) apuntando ``data.root`` a
    una carpeta temporal de imágenes chicas, con una U-Net MÍNIMA compatible con el ``image_size``
    (canales chicos, sin down-sampling profundo) para correr forward/backward en CPU en segundos.
    Verifica que la corrida se ensambla como la del toy y que ``train`` completa con una historia de
    pérdida finita — el loop de entrenamiento no ramifica por tipo de fuente.
    """
    image_size = 8
    raw = {
        "sde": {"name": "vp"},
        "data": {
            "kind": "images",
            "root": str(carpeta_imagenes),
            "image_size": image_size,
            "batch_size": 4,
            "augment": False,   # determinístico y más rápido para el smoke
            "crop": True,
        },
        # U-Net mínima: 2 niveles (reduction = 2^1 = 2, divide a image_size=8), pocos canales,
        # 1 res-block por nivel, sin atención en encoder/decoder (el bottleneck la lleva igual),
        # embed chico. groups=4 divide a los anchos 8 y 16. image_size = data.image_size.
        "model": {
            "name": "unet",
            "image_size": image_size,
            "base_channels": 8,
            "channel_mults": [1, 2],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "embed_dim": 16,
            "time_embed_dim": 32,
            "groups": 4,
        },
        "train": {"num_steps": 2, "lr": 1e-3, "seed": 0, "device": "cpu"},
    }
    spec = build_run(raw)

    # La corrida quedó cableada como imágenes: forma de evento en la SDE y red U-Net.
    assert spec.sde.data_dim == (3, image_size, image_size)
    assert isinstance(spec.model, ScoreUNet)

    result = train(spec.sde, spec.model, spec.data, spec.config)

    # El loop completó los pasos pedidos con pérdidas finitas (no ramifica por tipo de fuente).
    assert len(result.history) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.history)


# ----------------------------------- smoke end-to-end: gpu-training-efficiency


def test_smoke_e2e_imagenes_amp_y_knobs_build_run_train_resume(carpeta_imagenes):
    """Smoke integral de ``gpu-training-efficiency`` (R5.1/R5.2): config → build_run → train → resume.

    Arma una corrida de IMÁGENES **desde config** con las palancas de la feature activas por el
    camino de config: precisión mixta (``train.amp: true``) y los knobs de dataloader
    (``data.num_workers``/``data.pin_memory``, con la fuente REAL — sin espías — para probar que no
    rompen); ``non_blocking`` (transferencia de batch) y ``cudnn.benchmark`` son incondicionales/auto
    en el loop, así que quedan ejercitados sin flag. Todo en CPU: el autocast usa bfloat16 y el
    escalador va en passthrough — se verifica **no-regresión** (el camino completo no rompe), no el
    speedup (solo observable en GPU). Ejercita el camino de punta a punta: ensamblado por
    :func:`build_run`, loop de :func:`train` con ``amp=True`` y un ciclo de **reanudación** (snapshot
    intermedio → restaurar → completar), con el ``scaler_state`` viajando en el snapshot (R2.1).
    """
    image_size = 8
    total, corte = 4, 2

    def _raw():
        # Config de una celda de imágenes con las tres palancas de la feature activas por config.
        # U-Net mínima (mismos parámetros que el smoke base): corre forward/backward en CPU en
        # segundos. ``num_workers=0`` mantiene la carga in-process (estable en Windows CI).
        return {
            "sde": {"name": "vp"},
            "data": {
                "kind": "images",
                "root": str(carpeta_imagenes),
                "image_size": image_size,
                "batch_size": 4,
                "augment": False,      # determinístico y más rápido para el smoke
                "crop": True,
                "shuffle": False,      # orden fijo de la fuente
                "num_workers": 0,      # ← knob de carga: proceso principal (estable en CI)
                "pin_memory": False,   # ← knob de carga: sin memoria fijada (CPU)
                "seed": 0,
            },
            "model": {
                "name": "unet",
                "image_size": image_size,
                "base_channels": 8,
                "channel_mults": [1, 2],
                "num_res_blocks": 1,
                "attn_resolutions": [],
                "embed_dim": 16,
                "time_embed_dim": 32,
                "groups": 4,
            },
            "train": {
                "num_steps": total,
                "checkpoint_every": corte,  # emite un snapshot intermedio para reanudar
                "lr": 1e-3,
                "seed": 0,
                "device": "cpu",
                "amp": True,           # ← precisión mixta activada desde la config
            },
        }

    # --- config → build_run: la corrida queda cableada como imágenes + U-Net, con AMP en el config. ---
    spec = build_run(_raw())
    assert spec.sde.data_dim == (3, image_size, image_size)
    assert isinstance(spec.model, ScoreUNet)
    assert spec.config.amp is True  # el nuevo campo llegó al TrainConfig por el validador estricto

    # --- loop con AMP: corre en CPU (autocast bf16 + escalador passthrough) y captura un snapshot. ---
    frozen: dict = {}

    def _capture(tag, snap):
        frozen[tag] = snap

    result = train(spec.sde, spec.model, spec.data, spec.config, on_checkpoint=_capture)

    assert len(result.history) == total
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result.history)

    snap = frozen[f"step{corte:05d}"]
    assert snap.resume.start_step == corte
    # El estado del escalador AMP viajó en el snapshot (presente aun en CPU: {} ≠ None) (R2.1).
    assert snap.resume.scaler_state is not None

    # --- resume: red fresca (misma config) con los pesos del corte, reanudada con AMP (R2.2). ---
    spec_b = build_run(_raw())
    spec_b.model.load_state_dict(snap.result.net.state_dict())
    result_b = train(spec_b.sde, spec_b.model, spec_b.data, spec_b.config, resume=snap.resume)

    # La reanudación completó hasta el total sin romper, con la curva previa continuada (R2.2/R2.3).
    assert len(result_b.history) == total
    assert all(torch.isfinite(torch.tensor(loss)) for loss in result_b.history)
