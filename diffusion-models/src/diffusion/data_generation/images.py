"""Fuente de datos de imágenes para la Fase 2 (fotos de gatos, sin labels).

Este módulo convierte una carpeta de imágenes en disco en tensores listos para
alimentar el ``train`` genérico. Su núcleo es :class:`CatImages`, un
``torch.utils.data.Dataset`` **sin labels**: descubre los archivos de imagen
bajo una carpeta raíz, los lee con PIL, los pasa a RGB de 3 canales y les aplica
un ``transform`` inyectado, devolviendo un **tensor pelado** ``(3, H, W)`` (sin
etiqueta), de modo que el ``DataLoader`` yield-ee tensores batcheados y no
1-tuplas.

Los imports pesados (``torch``, ``torchvision``, ``PIL``) son **diferidos** (se
hacen dentro de las funciones/métodos, o de forma perezosa vía ``__getattr__``),
para que ``import diffusion.data_generation`` siga siendo liviano —solo numpy—
para el uso de puntos 2D, sin arrastrar torchvision. Es el mismo criterio que el
torch diferido del core de puntos (ver :mod:`diffusion.data_generation.base`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # solo para type-checkers; no se ejecuta en runtime
    import torch

#: Extensiones reconocidas como imagen (se comparan en minúsculas, así que el
#: descubrimiento es insensible a mayúsculas: ``.JPG`` cuenta igual que ``.jpg``).
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)


def _discover_image_paths(root: str | Path) -> list[Path]:
    """Descubre los archivos de imagen bajo ``root`` en orden determinístico.

    Recorre ``root`` recursivamente (``rglob``) y se queda con los archivos cuya
    extensión (en minúsculas) esté en :data:`IMAGE_EXTENSIONS`. El resultado se
    ordena para que el descubrimiento sea reproducible entre corridas.

    Args:
        root: Carpeta raíz donde buscar las imágenes (se recorre recursivamente).

    Returns:
        Lista ordenada de rutas (:class:`pathlib.Path`) a los archivos de imagen.

    Raises:
        ValueError: Si ``root`` no existe, o si no contiene ninguna imagen (en
            lugar de devolver una lista vacía de forma silenciosa).
    """
    root = Path(root)
    if not root.exists():
        raise ValueError(f"La carpeta raíz de imágenes no existe: {root!s}")
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        exts = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(
            f"No se encontraron imágenes en {root!s} "
            f"(extensiones buscadas: {exts})."
        )
    return paths


def _build_transform(
    image_size: int,
    augment: bool,
    crop: bool = True,
) -> Callable[..., torch.Tensor]:
    """Arma la cadena de transforms PIL→tensor ``(3, image_size, image_size)``.

    Devuelve un ``torchvision.transforms.Compose`` que lleva una ``PIL.Image``
    RGB a un ``torch.Tensor`` float32 de shape ``(3, image_size, image_size)`` con
    valores en ``[-1, 1]``. El orden de la cadena es: (flip opcional) → encuadre →
    ``ToTensor`` → ``Normalize``.

    - **Augmentation:** si ``augment`` es ``True`` se antepone
      ``RandomHorizontalFlip(p=0.5)`` (volteo **solo horizontal**). La cadena
      **nunca** incluye volteos verticales ni rotaciones: un gato al revés no es
      una muestra válida de la distribución.
    - **Encuadre (framing) configurable:** con ``crop=True`` (por defecto) se
      preserva el aspect ratio — ``Resize(image_size)`` escala el lado corto a
      ``image_size`` y ``CenterCrop(image_size)`` recorta el centro al cuadrado.
      Con ``crop=False`` se usa ``Resize((image_size, image_size))``, que deforma
      la imagen al cuadrado sin recortar.
    - **Normalización a ``[-1, 1]``:** ``ToTensor`` lleva la imagen a float32 en
      ``[0, 1]`` (canales primero) y ``Normalize([0.5]*3, [0.5]*3)`` la recentra a
      ``[-1, 1]``.

    El import de ``torchvision`` es **diferido** (dentro de la función), en línea
    con el criterio del resto del módulo: ``import diffusion.data_generation`` no
    debe arrastrar torchvision.

    Args:
        image_size: Lado del cuadrado de salida en píxeles; la imagen resultante
            tiene shape ``(3, image_size, image_size)``.
        augment: Si es ``True``, antepone el volteo horizontal aleatorio
            (``RandomHorizontalFlip(p=0.5)``); si es ``False``, la cadena no
            incluye ningún volteo.
        crop: Modo de encuadre. ``True`` (por defecto) preserva el aspect ratio
            (``Resize`` del lado corto + ``CenterCrop``); ``False`` deforma con
            ``Resize((image_size, image_size))`` sin recortar.

    Returns:
        Un ``torchvision.transforms.Compose`` que mapea una ``PIL.Image`` RGB a un
        ``torch.Tensor`` ``(3, image_size, image_size)`` float32 en ``[-1, 1]``.
    """
    from torchvision import transforms

    steps: list[Callable[..., object]] = []
    if augment:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if crop:
        # Preserva aspect ratio: escala el lado corto y recorta el centro.
        steps.append(transforms.Resize(image_size))
        steps.append(transforms.CenterCrop(image_size))
    else:
        # Deforma al cuadrado sin recortar (encuadre no configurable a mano).
        steps.append(transforms.Resize((image_size, image_size)))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return transforms.Compose(steps)


def _build_cat_images_class() -> type:
    """Construye la clase :class:`CatImages` importando torch de forma diferida.

    Se aísla en una fábrica para que ``torch`` no se importe al cargar el módulo,
    sino recién cuando se accede a ``CatImages`` (vía :func:`__getattr__`). Así el
    ``import`` liviano de ``diffusion.data_generation`` no arrastra torch/torchvision.

    Returns:
        La clase ``CatImages`` (subclase de ``torch.utils.data.Dataset``).
    """
    import torch

    class CatImages(torch.utils.data.Dataset):
        """Dataset de imágenes sin labels: archivo en disco → tensor pelado.

        Descubre los archivos de imagen bajo ``root`` (recursivo, filtrando por
        extensión y en orden determinístico) y, en cada acceso, lee la imagen con
        PIL, la convierte a RGB de 3 canales y le aplica el ``transform``
        inyectado. ``__getitem__`` devuelve el **tensor pelado** (sin etiqueta).

        El ``transform`` se recibe ya armado (la construcción de la cadena de
        transforms es responsabilidad de otra pieza del módulo); acá solo se
        aplica.

        Args:
            root: Carpeta raíz con las imágenes (se recorre recursivamente).
            transform: Callable que recibe una ``PIL.Image`` RGB y devuelve un
                ``torch.Tensor`` (típicamente la cadena resize/crop + ToTensor +
                Normalize).

        Raises:
            ValueError: Si ``root`` no existe o no contiene ninguna imagen.
        """

        def __init__(
            self,
            root: str | Path,
            transform: Callable[..., torch.Tensor],
        ) -> None:
            self.root = Path(root)
            self.transform = transform
            self.paths: list[Path] = _discover_image_paths(root)

        def __len__(self) -> int:
            """Cantidad de imágenes descubiertas bajo ``root``."""
            return len(self.paths)

        def __getitem__(self, index: int) -> torch.Tensor:
            """Lee la imagen ``index``, la pasa a RGB y le aplica el transform.

            La conversión ``.convert("RGB")`` es obligatoria: garantiza 3 canales
            aunque la imagen original sea escala de grises, RGBA o CMYK.

            Args:
                index: Índice de la imagen en la lista ordenada de rutas.

            Returns:
                El ``torch.Tensor`` que produce ``transform`` sobre la imagen en
                RGB (típicamente ``(3, H, W)`` float32). Tensor pelado, sin label.
            """
            from PIL import Image

            with Image.open(self.paths[index]) as img:
                image = img.convert("RGB")
            return self.transform(image)

    return CatImages


def _infinite(loader) -> Iterator[torch.Tensor]:
    """Recorre un ``DataLoader`` finito en bucle y yield-ea el batch crudo.

    Convierte un ``DataLoader`` finito de imágenes en un iterador que **nunca se
    agota**: al terminar de recorrer el loader vuelve a empezar (``while True:
    yield from loader``). A diferencia de :func:`infinite_bare` —que desempaqueta
    la 1-tupla ``(x0,)`` de las fuentes 2D—, acá se yield-ea el tensor batcheado
    **pelado** tal cual lo produce el collate por defecto sobre los tensores
    pelados de :class:`CatImages` (un ``(B, 3, H, W)``, no una tupla).

    No altera el ``loader``; solo lo envuelve.

    Args:
        loader: ``DataLoader`` (u otro iterable) que yield-ea tensores batcheados
            pelados ``(B, 3, H, W)``.

    Yields:
        El tensor batcheado crudo de cada paso, indefinidamente.
    """
    while True:
        yield from loader


def infinite_batches(
    root: str | Path,
    batch_size: int,
    *,
    image_size: int = 64,
    augment: bool = True,
    crop: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int | None = None,
    pin_memory: bool = False,
) -> Iterator[torch.Tensor]:
    """Fuente infinita de batches de imágenes, drop-in para el ``data`` de ``train``.

    Arma el :class:`CatImages` sobre ``root`` con la cadena de transforms de
    :func:`_build_transform`, lo envuelve en un ``DataLoader`` (con ``shuffle``,
    ``drop_last=True`` para que **todos** los batches tengan tamaño exacto, y un
    ``generator`` sembrado desde ``seed`` para barajado reproducible) y lo entrega
    a través de un wrapper infinito (:func:`_infinite`). El iterador resultante
    **nunca se agota**: cada ``next()`` devuelve un tensor batcheado **pelado**
    ``(batch_size, 3, image_size, image_size)`` float32 en ``[-1, 1]`` (no una
    tupla), que es exactamente el contrato de ``data`` que consume el ``train``
    genérico.

    **Fail-fast:** los errores cortan **antes** de construir el iterador (esta
    función no es un generador, así que su cuerpo corre en la llamada). Si ``root``
    no existe o no tiene imágenes, :class:`CatImages` levanta ``ValueError`` al
    construirse; si hay imágenes pero son menos que ``batch_size``, se levanta
    ``ValueError`` acá —porque con ``drop_last=True`` el loader quedaría vacío y el
    ``while True`` del wrapper giraría sin yield-ear nunca (cuelgue silencioso).

    Los imports de ``torch`` y torchvision (vía :func:`_build_transform`) son
    **diferidos**, en línea con el resto del módulo.

    Args:
        root: Carpeta raíz con las imágenes (se recorre recursivamente).
        batch_size: Cantidad de imágenes por batch; todos los batches salen con
            exactamente este tamaño (``drop_last=True``).
        image_size: Lado del cuadrado de salida en píxeles; cada batch tiene shape
            ``(batch_size, 3, image_size, image_size)``.
        augment: Si es ``True``, la cadena aplica volteo horizontal aleatorio
            (ver :func:`_build_transform`); si es ``False``, sin augmentation.
        crop: Modo de encuadre. ``True`` (por defecto) preserva aspect ratio
            (``Resize`` del lado corto + ``CenterCrop``); ``False`` deforma con
            ``Resize((image_size, image_size))`` sin recortar.
        num_workers: Cantidad de procesos de carga del ``DataLoader`` (``0`` corre
            en el proceso principal; funciona en CPU).
        shuffle: Si es ``True``, baraja las imágenes en cada recorrido interno.
        seed: Semilla del ``torch.Generator`` que gobierna el barajado. Con
            ``seed`` fijo, ``augment=False`` y ``num_workers=0`` la secuencia de
            batches es reproducible. Si es ``None``, el barajado usa el RNG global.
        pin_memory: Se pasa tal cual al ``DataLoader`` (útil solo con GPU).

    Returns:
        Un iterador infinito de tensores ``(batch_size, 3, image_size,
        image_size)`` float32 en ``[-1, 1]`` (tensores pelados, sin tupla).

    Raises:
        ValueError: Si ``root`` no existe, no contiene imágenes, o contiene menos
            de ``batch_size`` imágenes (fail-fast, antes de devolver el iterador).
    """
    import torch

    transform = _build_transform(image_size, augment, crop)
    dataset = _build_cat_images_class()(root, transform)

    n_images = len(dataset)
    if n_images < batch_size:
        raise ValueError(
            f"La carpeta {Path(root)!s} tiene {n_images} imagen(es), menos que "
            f"batch_size={batch_size}: con drop_last=True el loader quedaría "
            f"vacío y el iterador infinito no entregaría ningún batch."
        )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    return _infinite(loader)


def report_small_images(
    root: str | Path,
    *,
    min_size: int = 64,
    verbose: bool = False,
) -> list[Path]:
    """Reporta (sin borrar) las imágenes cuyo lado corto es menor que ``min_size``.

    Chequeo de higiene **report-only** y **separado del flujo de carga** (no corre
    en cada batch ni lo dispara :func:`infinite_batches`): recorre las imágenes
    descubiertas bajo ``root`` y devuelve las que tienen ``min(width, height) <
    min_size``. El **lado corto** es el criterio relevante porque es el que el
    ``Resize`` del pipeline escalaría *hacia arriba* (upscale): una imagen con el
    lado corto por debajo de ``min_size`` entra al modelo borrosa y degrada la
    calidad de las muestras.

    **No borra ni modifica ningún archivo**: solo lee las dimensiones (con PIL, sin
    decodificar los píxeles) y devuelve la lista de rutas problemáticas para que el
    autor decida qué descartar. **No** implementa detección de duplicados: esa
    responsabilidad es de ``scripts/limpiar_imagenes.py``.

    El import de ``PIL`` es **diferido** (dentro de la función), en línea con el
    resto del módulo.

    Args:
        root: Carpeta raíz con las imágenes (se recorre recursivamente, igual que
            en la carga).
        min_size: Umbral en píxeles para el lado corto; se reporta toda imagen con
            ``min(width, height) < min_size``. Por defecto ``64`` (el ``image_size``
            habitual de la Fase 2).
        verbose: Si es ``True``, imprime cada ruta reportada con sus dimensiones; si
            es ``False`` (por defecto), solo devuelve la lista sin imprimir nada.

    Returns:
        Lista ordenada de rutas (:class:`pathlib.Path`) a las imágenes con el lado
        corto por debajo de ``min_size`` (subconjunto del orden determinístico de
        :func:`_discover_image_paths`). Lista vacía si no hay ninguna.

    Raises:
        ValueError: Si ``root`` no existe o no contiene ninguna imagen (delegado a
            :func:`_discover_image_paths`).
    """
    from PIL import Image

    small: list[Path] = []
    for path in _discover_image_paths(root):
        with Image.open(path) as img:
            width, height = img.size
        if min(width, height) < min_size:
            small.append(path)
            if verbose:
                print(f"  too-small: {path.name} ({width}x{height})")
    return small


def __getattr__(name: str):
    """Resuelve ``CatImages`` de forma perezosa (PEP 562).

    Permite ``from diffusion.data_generation.images import CatImages`` sin
    importar torch al cargar el módulo: la clase se construye (importando torch)
    solo en el primer acceso y luego se cachea en el namespace del módulo, así
    los accesos siguientes no vuelven a pasar por acá.

    Args:
        name: Nombre del atributo solicitado al módulo.

    Returns:
        La clase ``CatImages`` cuando ``name == "CatImages"``.

    Raises:
        AttributeError: Para cualquier otro nombre (comportamiento estándar).
    """
    if name == "CatImages":
        cls = _build_cat_images_class()
        globals()["CatImages"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    # Smoke test manual: carga las 2 imágenes reales de data/cats-prueba/, arma un
    # batch y verifica el contrato de salida (shape + rango ~[-1, 1]); después corre
    # el chequeo de higiene report-only sobre la misma carpeta. Correr como módulo
    # (imports diferidos):  python -m diffusion.data_generation.images
    # (con diffusion-models/src en PYTHONPATH).
    import torch

    # images.py está en src/diffusion/data_generation/; parents[3] es diffusion-models/.
    cats_prueba = Path(__file__).resolve().parents[3] / "data" / "cats-prueba"

    batch = next(infinite_batches(cats_prueba, batch_size=2, image_size=64))
    assert tuple(batch.shape) == (2, 3, 64, 64), (
        f"shape inesperada: {tuple(batch.shape)}"
    )
    assert batch.dtype == torch.float32, f"dtype inesperado: {batch.dtype}"
    lo, hi = float(batch.min()), float(batch.max())
    assert -1.01 <= lo and hi <= 1.01, f"rango fuera de ~[-1, 1]: [{lo:.3f}, {hi:.3f}]"
    print(
        f"infinite_batches(cats-prueba, batch_size=2): batch {tuple(batch.shape)} "
        f"{batch.dtype}, rango [{lo:.3f}, {hi:.3f}]"
    )

    small = report_small_images(cats_prueba, min_size=64)
    print(
        f"report_small_images(min_size=64): {len(small)} imagen(es) demasiado "
        f"chica(s) (report-only, no se borra ninguna)"
    )
