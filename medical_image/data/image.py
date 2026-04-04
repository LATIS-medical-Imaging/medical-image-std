import functools
import os
from abc import ABC, abstractmethod
from typing import Optional, Union, List, TypeVar

import torch
from medical_image.utils.logging import logger
from medical_image.utils.ErrorHandler import ErrorMessages, DicomDataNotLoadedError
import numpy as np

T = TypeVar("T")


def requires_loaded(func):
    """Decorator that checks pixel_data is not None before processing."""

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
    """
    Abstract base class for medical images supporting lazy loading and multiple constructors.

    Width and height are computed properties derived from pixel_data.shape when
    pixel_data is loaded. Before loading, they fall back to cached values set
    during construction or by subclass .load() methods.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        array: Optional[Union[np.ndarray, torch.Tensor]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_image: Optional["Image"] = None,
    ):
        self.file_path: Optional[str] = None
        self._width: Optional[int] = width
        self._height: Optional[int] = height
        self.pixel_data: Optional[torch.Tensor] = None
        self._device: torch.device = torch.device("cpu")

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
            if hasattr(source_image, "annotations"):
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
        """Move pixel_data to target device (in-place). Returns self for chaining."""
        self._device = torch.device(device)
        if self.pixel_data is not None:
            self.pixel_data = self.pixel_data.to(self._device)
        return self

    def ensure_loaded(self) -> "Image":
        """Guard method: raise if pixel_data is None."""
        if self.pixel_data is None:
            raise DicomDataNotLoadedError("Call .load() first")
        return self

    def pin_memory(self) -> "Image":
        """Pin pixel_data to page-locked memory for faster GPU transfers."""
        if self.pixel_data is not None and not self.pixel_data.is_pinned():
            self.pixel_data = self.pixel_data.pin_memory()
        return self

    # ------------------------------------------------------------------
    # Clone (lightweight alternative to copy.deepcopy)
    # ------------------------------------------------------------------

    def clone(self) -> "Image":
        """Lightweight clone: copies pixel_data tensor and metadata, not heavy objects."""
        new = self.__class__.__new__(self.__class__)
        new.file_path = self.file_path
        new._width = self._width
        new._height = self._height
        new._device = self._device
        new.pixel_data = (
            self.pixel_data.clone() if self.pixel_data is not None else None
        )
        if hasattr(self, "annotations"):
            new.annotations = self.annotations
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
        return cls(file_path=file_path)

    @classmethod
    def from_image(cls, other_image: "Image") -> "Image":
        return cls(source_image=other_image)

    @classmethod
    def from_array(cls, array: Union[np.ndarray, torch.Tensor]) -> "Image":
        return cls(array=array)

    @classmethod
    def empty(
        cls, width: Optional[int] = None, height: Optional[int] = None
    ) -> "Image":
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
    # Display / repr
    # ------------------------------------------------------------------

    def display_info(self):
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

        if hasattr(self, "annotations") and self.annotations:
            if isinstance(self.annotations, list):
                logger.info(f"Annotations: {len(self.annotations)} items")
            else:
                logger.info("Annotations: 1 item")
        else:
            logger.info("Annotations: None")

        logger.info("=================")

    def __repr__(self):
        status = "loaded" if self.pixel_data is not None else "unloaded"
        dev = str(self.device) if self.pixel_data is not None else "n/a"
        return (
            f"{self.__class__.__name__}("
            f"path='{self.file_path}', "
            f"{self.width}x{self.height}, "
            f"{status}, device={dev})"
        )
