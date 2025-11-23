import os
from abc import ABC, abstractmethod
from typing import Optional, Union, List, TypeVar

import torch
from log_manager import logger
from medical_image.utils.ErrorHandler import ErrorMessages
from medical_image.utils.annotation import Annotation
import numpy as np

T = TypeVar("T")


class Image(ABC):
    """
    Abstract base class for medical images supporting lazy loading and multiple constructors.
    """

    def __init__(
        self,
        file_path: Optional[str] = None,
        array: Optional[Union[np.ndarray, torch.Tensor]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        source_image: Optional["Image"] = None,
    ):
        """
        Flexible constructor for Image.
        Supports:
            - file_path: lazy-load from a file
            - array: initialize from NumPy array or PyTorch tensor
            - source_image: copy another Image object
            - width/height: empty/default image

        Note: Pixel data is lazy-loaded or copied as appropriate.

        Args:
            file_path (str, optional): Path to image file.
            array (np.ndarray or torch.Tensor, optional): Image pixel data.
            width (int, optional): Image width (for empty image).
            height (int, optional): Image height (for empty image).
            source_image (Image, optional): Another Image to copy.
        """
        self.file_path: Optional[str] = None
        self.width: Optional[int] = width
        self.height: Optional[int] = height
        self.pixel_data: Optional[torch.Tensor] = None
        # TODO: We should Discuss Annotation
        # self.annotations: Optional[Union[Annotation, List[Annotation]]] = None

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
            self.height, self.width = self.pixel_data.shape[:2]
        elif source_image is not None:
            self.file_path = source_image.file_path
            self.width = source_image.width
            self.height = source_image.height
            if source_image.pixel_data is not None:
                self.pixel_data = source_image.pixel_data.clone()
            self.annotations = source_image.annotations
        else:
            # Empty image with optional width/height
            self.pixel_data = None

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

    @abstractmethod
    def load(self):
        """Load pixel data (lazy load). Must be implemented by subclasses."""
        pass

    @abstractmethod
    def save(self):
        """Save pixel data. Must be implemented by subclasses."""
        pass

    def display_info(self):
        """
        Display detailed information about the image object.
        Works for all constructor types: file_path, array, source_image, or empty.
        """
        logger.info("=== Image Info ===")

        # File path info
        if self.file_path:
            logger.info(f"File Path: {self.file_path}")
        else:
            logger.info("File Path: <None>")

        # Pixel data info
        if self.pixel_data is not None:
            logger.info(f"Pixel Data: Loaded")
            logger.info(f"Pixel Data Type: {self.pixel_data.dtype}")
            if self.pixel_data.ndim == 2:
                logger.info(f"Shape (H x W): {self.pixel_data.shape}")
            else:
                logger.info(f"Shape: {self.pixel_data.shape}")
        else:
            logger.info("Pixel Data: Not loaded")

        # Width and height
        if self.width is not None and self.height is not None:
            logger.info(f"Width: {self.width}")
            logger.info(f"Height: {self.height}")
        else:
            logger.info("Width/Height: Not set")

        # Annotations
        if hasattr(self, "annotations") and self.annotations:
            if isinstance(self.annotations, list):
                logger.info(f"Annotations: {len(self.annotations)} items")
            else:
                logger.info("Annotations: 1 item")
        else:
            logger.info("Annotations: None")

        logger.info("=================")
