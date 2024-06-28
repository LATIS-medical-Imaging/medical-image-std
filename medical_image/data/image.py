import os
from abc import ABC, abstractmethod
from typing import TypeVar, Any

import numpy as np

from log_manager import logger

T = TypeVar("T")
from PIL import Image as PILImage


class Image(ABC):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"unable to locate file_path at {file_path}")
        self.file_path = file_path
        self.width = None
        self.height = None
        self.pixel_data = None

    @abstractmethod
    def load(self):
        """Abstract method to load image data. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def save(self):
        """Abstract method to load image data. Must be implemented by subclasses."""
        pass

    def display_info(self):
        """Display basic information about the image."""
        logger.info(f"File Path: {self.file_path}")
        logger.info(f"Width: {self.width}")
        logger.info(f"Height: {self.height}")

    def to_png(self):
        """
        Save a NumPy array as a PNG file.

        Parameters:
        pixel_data (np.ndarray): The NumPy array containing the pixel data.
        file_path (str): The file path where the PNG image will be saved.
        """
        if self.pixel_data is None and not isinstance(self.pixel_data, np.ndarray):
            raise ValueError("pixel_data is not a valid NumPy array.")
        filename, _ = os.path.splitext(self.file_path)
        image = PILImage.fromarray(self.pixel_data)
        image.save(filename + ".png")
        logger.info(f"Image saved successfully at {filename + '.png'}")


from typing import Union, List, Tuple
import numpy as np


class RegionOfInterest:
    """
    This class represents a region of interest (ROI) within an image.

    Parameters:
        image (Image): The Image object that contains the region of interest.
        coordinates (Union[List[int], List[Tuple[int, int]], np.ndarray]): The coordinates defining the region of interest.
            This can be in various forms such as:
            - Bounding box: List of four integers [x_min, y_min, x_max, y_max]
            - Polygon: List of tuples of integers [(x1, y1), (x2, y2), ..., (xn, yn)]
            - Mask: 2D numpy array of the same dimensions as the image, where the region of interest is marked

    Attributes:
        image (Image): The Image object that contains the region of interest.
        coordinates (Union[List[int], List[Tuple[int, int]], np.ndarray]): The coordinates defining the region of interest.

    Examples:
        >>> image = Image("example_path")
        >>> image.load()
        >>> roi_bbox = RegionOfInterest(image, [50, 50, 150, 150])
        >>> roi_polygon = RegionOfInterest(image, [(50, 50), (150, 50), (150, 150), (50, 150)])
        >>> mask = np.zeros((image.height, image.width))
        >>> mask[50:150, 50:150] = 1
        >>> roi_mask = RegionOfInterest(image, mask)
    """

    def __init__(self, image: Image, coordinates: Union[List[int], List[Tuple[int, int]], np.ndarray]):
        self.image = image
        self.coordinates = coordinates

    def load(self) -> Image:
        pass

    pass
