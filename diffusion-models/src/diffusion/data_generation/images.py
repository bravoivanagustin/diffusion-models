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

from collections.abc import Callable
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
