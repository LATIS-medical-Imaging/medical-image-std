import copy
import os
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TypeVar, Union, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from log_manager import logger

T = TypeVar("T")
from PIL import Image as PILImage
class RegionType(Enum):
    BOUNDING_BOX = auto()
    POLYGON = auto()
    MASK = auto()


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

    def plot(self, cmap="gray"):
        """Display the image using matplotlib.

        Parameters:
        cmap (str): Colormap to use for displaying the image. Default is 'gray'.
        """
        if self.pixel_data is None:
            raise ValueError("pixel_data is not loaded.")
        plt.imshow(self.pixel_data, cmap=cmap)
        plt.axis("off")  # Hide axes
        plt.show()


class RegionOfInterest:
    """
    Represents a Region of Interest (ROI) in a medical image, which may be a bounding box,
    a polygon, or a binary mask.

    Parameters:
        image (Image): The Image object the ROI is based on.
        coordinates (Union[List[int], List[Tuple[int, int]], np.ndarray]): The ROI definition.
            - Bounding box: [x_min, y_min, x_max, y_max]
            - Polygon: [(x1, y1), (x2, y2), ..., (xn, yn)]
            - Mask: 2D NumPy array

    Attributes:
        image (Image): The original image.
        coordinates: Coordinates representing the ROI.
        region_type (RegionType): The type of ROI (bounding box, polygon, or mask).
    """

    def __init__(
        self,
        image: Image,
        coordinates: Union[List[int], List[Tuple[int, int]], np.ndarray],
    ):
        self.image = image
        self.coordinates = coordinates
        self.region_type = self._determine_region_type()

    def _determine_region_type(self) -> RegionType:
        if isinstance(self.coordinates, list):
            if len(self.coordinates) == 4 and all(isinstance(c, int) for c in self.coordinates):
                return RegionType.BOUNDING_BOX
            elif all(isinstance(c, tuple) and len(c) == 2 for c in self.coordinates):
                return RegionType.POLYGON
        elif isinstance(self.coordinates, np.ndarray):
            return RegionType.MASK

        raise ValueError("Unsupported or invalid coordinates format for RegionOfInterest")

    def load(self) -> Image:
        """
        Crop the image to the ROI and return a new Image object containing only the ROI.

        Returns:
            Image: A new Image instance cropped to the region of interest.
        """
        pixel_data = self.image.pixel_data

        if self.region_type == RegionType.BOUNDING_BOX:
            x_min, y_min, x_max, y_max = self.coordinates
            cropped_pixel_data = pixel_data[y_min:y_max, x_min:x_max]

        elif self.region_type == RegionType.POLYGON:
            from skimage.draw import polygon  # polygon rasterization

            mask = np.zeros_like(pixel_data, dtype=bool)
            poly_y, poly_x = zip(*self.coordinates)
            rr, cc = polygon(poly_y, poly_x)
            mask[rr, cc] = True
            cropped_pixel_data = pixel_data * mask

            y_indices, x_indices = np.nonzero(mask)
            y_min, y_max = y_indices.min(), y_indices.max() + 1
            x_min, x_max = x_indices.min(), x_indices.max() + 1
            cropped_pixel_data = cropped_pixel_data[y_min:y_max, x_min:x_max]

        elif self.region_type == RegionType.MASK:
            mask = self.coordinates.astype(bool)
            cropped_pixel_data = pixel_data * mask

            y_indices, x_indices = np.nonzero(mask)
            y_min, y_max = y_indices.min(), y_indices.max() + 1
            x_min, x_max = x_indices.min(), x_indices.max() + 1
            cropped_pixel_data = cropped_pixel_data[y_min:y_max, x_min:x_max]

        else:
            raise RuntimeError("Unknown region type.")

        # Create a new Image instance and copy metadata
        cropped_image = copy.deepcopy(self.image)
        cropped_image.pixel_data = cropped_pixel_data
        cropped_image.height, cropped_image.width = cropped_pixel_data.shape[:2]

        return cropped_image

