import functools
import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Union, List, TypeVar

import torch
from medical_image.utils.logging import logger
from medical_image.utils.ErrorHandler import ErrorMessages, DicomDataNotLoadedError
from medical_image.utils.annotation import Annotation
import numpy as np

T = TypeVar("T")


def requires_loaded(func):
    """Decorator that verifies ``pixel_data`` is loaded on every Image argument.

    Inspects all positional and keyword arguments; raises
    ``DicomDataNotLoadedError`` if any :class:`Image` has
    ``pixel_data is None``.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, Image) and arg.pixel_data is None:
                raise DicomDataNotLoadedError(
                    f"{func.__name__}: Image pixel_data is None. Call .load() first."
                )
        return func(*args, **kwargs)

    return wrapper


class Image(ABC):
    """Abstract base class for medical images.

    Supports lazy loading and four mutually-exclusive construction paths
    (file, array, source image, or empty shell).  Width and height are
    computed properties derived from ``pixel_data.shape`` when loaded,
    falling back to cached values before loading.

    The Image optionally holds a list of :class:`~medical_image.utils.annotation.Annotation`
    objects via aggregation -- an image can exist without annotations.

    Attributes:
        file_path (Optional[str]): Path to the image file on disk.
        pixel_data (Optional[torch.Tensor]): Pixel values (``None`` until loaded).
        annotations (Optional[List[Annotation]]): Attached annotations (``None`` by default).
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        array: Optional[Union[np.ndarray, torch.Tensor]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_image: Optional["Image"] = None,
    ):
        """Initialise an Image via one of four construction paths.

        Args:
            file_path: Path to an image file.  Raises ``FileNotFoundError``
                if it does not exist.
            array: Pre-existing numpy array or torch tensor to wrap as pixel data.
            width: Explicit width hint (used before pixel data is loaded).
            height: Explicit height hint (used before pixel data is loaded).
            source_image: Another Image to clone metadata and pixel data from.
        """
        self.file_path: Optional[str] = None
        self._width: Optional[int] = width
        self._height: Optional[int] = height
        self.pixel_data: Optional[torch.Tensor] = None
        self._device: torch.device = torch.device("cpu")
        self.annotations: Optional[List[Annotation]] = None

        if file_path is not None:
            if not os.path.exists(file_path):
                raise ErrorMessages.file_not_found(file_path)
            self.file_path = file_path
        elif array is not None:
            if isinstance(array, np.ndarray):
                self.pixel_data = torch.from_numpy(array).float()
            elif isinstance(array, torch.Tensor):
                self.pixel_data = array.float()
            else:
                raise TypeError("array must be a NumPy array or a PyTorch tensor")
        elif source_image is not None:
            self.file_path = source_image.file_path
            self._width = source_image._width
            self._height = source_image._height
            if source_image.pixel_data is not None:
                self.pixel_data = source_image.pixel_data.clone()
            self.annotations = source_image.annotations
        else:
            self.pixel_data = None

    # ------------------------------------------------------------------
    # Computed width / height — derived from pixel_data when loaded
    # ------------------------------------------------------------------

    @property
    def width(self) -> Optional[int]:
        if self.pixel_data is not None:
            return self.pixel_data.shape[-1]
        return self._width

    @width.setter
    def width(self, value: Optional[int]):
        self._width = value

    @property
    def height(self) -> Optional[int]:
        if self.pixel_data is not None:
            return self.pixel_data.shape[-2]
        return self._height

    @height.setter
    def height(self, value: Optional[int]):
        self._height = value

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        if self.pixel_data is not None:
            return self.pixel_data.device
        return self._device

    def to(self, device: Union[str, torch.device]) -> "Image":
        """Move pixel data to *device* (in-place).

        Args:
            device: Target device (e.g. ``"cuda"``, ``"cpu"``).

        Returns:
            ``self``, for method chaining.
        """
        self._device = torch.device(device)
        if self.pixel_data is not None:
            self.pixel_data = self.pixel_data.to(self._device)
        return self

    def ensure_loaded(self) -> "Image":
        """Raise ``DicomDataNotLoadedError`` if pixel data has not been loaded.

        Returns:
            ``self``, for method chaining.
        """
        if self.pixel_data is None:
            raise DicomDataNotLoadedError("Call .load() first")
        return self

    def pin_memory(self) -> "Image":
        """Pin pixel data to page-locked memory for faster GPU transfers.

        No-op if pixel data is ``None`` or already pinned.

        Returns:
            ``self``, for method chaining.
        """
        if self.pixel_data is not None and not self.pixel_data.is_pinned():
            self.pixel_data = self.pixel_data.pin_memory()
        return self

    # ------------------------------------------------------------------
    # Clone (lightweight alternative to copy.deepcopy)
    # ------------------------------------------------------------------

    def clone(self) -> "Image":
        """Create a lightweight copy of this image.

        Clones the pixel data tensor and shallow-copies the annotation list,
        but does **not** copy heavy objects (DICOM dataset, PIL image).

        Returns:
            A new Image of the same concrete type.
        """
        new = self.__class__.__new__(self.__class__)
        new.file_path = self.file_path
        new._width = self._width
        new._height = self._height
        new._device = self._device
        new.pixel_data = (
            self.pixel_data.clone() if self.pixel_data is not None else None
        )
        new.annotations = list(self.annotations) if self.annotations else None
        # Subclass-specific: don't copy heavy DICOM/PIL objects
        if hasattr(self, "dicom_data"):
            new.dicom_data = None
        if hasattr(self, "_pil_image"):
            new._pil_image = None
        return new

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, file_path: str) -> "Image":
        """Construct an Image from a file path (lazy -- does not load pixels)."""
        return cls(file_path=file_path)

    @classmethod
    def from_image(cls, other_image: "Image") -> "Image":
        """Construct an Image by copying metadata and pixel data from *other_image*."""
        return cls(source_image=other_image)

    @classmethod
    def from_array(cls, array: Union[np.ndarray, torch.Tensor]) -> "Image":
        """Construct an Image from a NumPy array or PyTorch tensor."""
        return cls(array=array)

    @classmethod
    def empty(
        cls, width: Optional[int] = None, height: Optional[int] = None
    ) -> "Image":
        """Construct an empty Image shell with optional width/height hints."""
        return cls(width=width, height=height)

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self):
        """Load pixel data (lazy load). Must be implemented by subclasses."""
        pass

    @abstractmethod
    def save(self):
        """Save pixel data. Must be implemented by subclasses."""
        pass

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def add_annotation(self, annotation: Annotation) -> None:
        """Append an annotation to this image.

        Initialises the annotation list to ``[]`` on first call if it is
        currently ``None``.

        Args:
            annotation: The :class:`Annotation` to attach.
        """
        if self.annotations is None:
            self.annotations = []
        self.annotations.append(annotation)

    def remove_annotation(self, index: int) -> Annotation:
        """Remove and return the annotation at *index*.

        Args:
            index: Zero-based position in the annotation list.

        Returns:
            The removed :class:`Annotation`.

        Raises:
            IndexError: If the annotation list is ``None`` or *index* is
                out of range.
        """
        if self.annotations is None or index >= len(self.annotations):
            raise IndexError(f"Annotation index {index} out of range")
        return self.annotations.pop(index)

    # ------------------------------------------------------------------
    # JSON serialization
    # ------------------------------------------------------------------

    def to_json(self, file_path: Optional[str] = None) -> str:
        """Serialize this image's metadata and annotations to JSON.

        Pixel data is **not** included -- only file path, dimensions,
        image type, and the full annotation list.

        Args:
            file_path: If provided, the JSON string is also written to
                this file path.

        Returns:
            A JSON string with keys ``file_path``, ``width``, ``height``,
            ``image_type``, and ``annotations``.
        """
        data = {
            "file_path": self.file_path,
            "width": self.width,
            "height": self.height,
            "image_type": self.__class__.__name__,
            "annotations": [ann.to_dict() for ann in (self.annotations or [])],
        }

        json_str = json.dumps(data, indent=2)
        if file_path:
            with open(file_path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_input: str) -> "Image":
        """Deserialize an Image from a JSON string or file path.

        Pixel data is **not** loaded -- only metadata and annotations are
        restored.  Call on a concrete subclass (``InMemoryImage``,
        ``DicomImage``, ``PNGImage``).  For automatic subclass dispatch see
        :func:`image_from_json`.

        Args:
            json_input: A JSON string **or** a path to a ``.json`` file.

        Returns:
            A new Image instance with annotations attached.
        """
        if os.path.isfile(json_input):
            with open(json_input, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(json_input)

        image = cls(
            file_path=(
                data.get("file_path")
                if data.get("file_path") and os.path.exists(data["file_path"])
                else None
            ),
            width=data.get("width"),
            height=data.get("height"),
        )

        annotations = data.get("annotations", [])
        if annotations:
            image.annotations = [Annotation.from_dict(ann) for ann in annotations]

        return image

    # ------------------------------------------------------------------
    # Display / repr
    # ------------------------------------------------------------------

    def display_info(self) -> None:
        """Log summary information about this image (path, dimensions, device, annotations)."""
        logger.info("=== Image Info ===")

        if self.file_path:
            logger.info(f"File Path: {self.file_path}")
        else:
            logger.info("File Path: <None>")

        if self.pixel_data is not None:
            logger.info("Pixel Data: Loaded")
            logger.info(f"Pixel Data Type: {self.pixel_data.dtype}")
            if self.pixel_data.ndim == 2:
                logger.info(f"Shape (H x W): {self.pixel_data.shape}")
            else:
                logger.info(f"Shape: {self.pixel_data.shape}")
            logger.info(f"Device: {self.device}")
        else:
            logger.info("Pixel Data: Not loaded")

        if self.width is not None and self.height is not None:
            logger.info(f"Width: {self.width}")
            logger.info(f"Height: {self.height}")
        else:
            logger.info("Width/Height: Not set")

        if self.annotations:
            if isinstance(self.annotations, list):
                logger.info(f"Annotations: {len(self.annotations)} items")
            else:
                logger.info("Annotations: 1 item")
        else:
            logger.info("Annotations: None")

        logger.info("=================")

    def __repr__(self) -> str:
        """Return a one-line summary of the image (class, path, size, status)."""
        status = "loaded" if self.pixel_data is not None else "unloaded"
        dev = str(self.device) if self.pixel_data is not None else "n/a"
        return (
            f"{self.__class__.__name__}("
            f"path='{self.file_path}', "
            f"{self.width}x{self.height}, "
            f"{status}, device={dev})"
        )


def image_from_json(json_input: str) -> Image:
    """Factory function: load any Image subclass from JSON.

    Reads the ``image_type`` field from the JSON payload and dispatches
    to the matching concrete subclass's :meth:`Image.from_json`.

    Dispatch table:
        * ``"DicomImage"``    -> :class:`DicomImage`
        * ``"PNGImage"``      -> :class:`PNGImage`
        * ``"InMemoryImage"`` -> :class:`InMemoryImage` (also the fallback)

    Args:
        json_input: A JSON string **or** a path to a ``.json`` file.

    Returns:
        An instance of the correct concrete Image subclass with annotations
        attached.
    """
    if os.path.isfile(json_input):
        with open(json_input, "r") as f:
            data = json.load(f)
    else:
        data = json.loads(json_input)

    from medical_image.data.dicom_image import DicomImage
    from medical_image.data.png_image import PNGImage
    from medical_image.data.in_memory_image import InMemoryImage

    _registry = {
        "DicomImage": DicomImage,
        "PNGImage": PNGImage,
        "InMemoryImage": InMemoryImage,
    }

    image_type = data.get("image_type", "InMemoryImage")
    cls = _registry.get(image_type, InMemoryImage)
    return cls.from_json(json_input)
