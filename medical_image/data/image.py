import copy
import os
from abc import ABC, abstractmethod
from typing import TypeVar, Union, List, Tuple
import matplotlib.pyplot as plt
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
    def plot(self, cmap='gray'):
        """Display the image using matplotlib.

        Parameters:
        cmap (str): Colormap to use for displaying the image. Default is 'gray'.
        """
        if self.pixel_data is None:
            raise ValueError("pixel_data is not loaded.")
        plt.imshow(self.pixel_data, cmap=cmap)
        plt.axis('off')  # Hide axes
        plt.show()


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
        >>> image = ExampleImage("example_path")
        >>> image.load()
        >>> roi_bbox = RegionOfInterest(image, [50, 50, 150, 150])
        >>> roi_polygon = RegionOfInterest(image, [(50, 50), (150, 50), (150, 150), (50, 150)])
        >>> mask = np.zeros((image.height, image.width))
        >>> mask[50:150, 50:150] = 1
        >>> roi_mask = RegionOfInterest(image, mask)
    """

    def __init__(
        self,
        image: Image,
        coordinates: Union[List[int], List[Tuple[int, int]], np.ndarray],
    ):
        self.image = image
        self.coordinates = coordinates

    def load(self):
        """
        Crop the image to the ROI and create a new ExampleImage object.

        Returns:
            ExampleImage: A new ExampleImage object cropped to the ROI.
        """
        if isinstance(self.coordinates, list) and len(self.coordinates) == 4:
            # Bounding box: [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = self.coordinates
            cropped_pixel_data = self.image.pixel_data[y_min:y_max, x_min:x_max]

        elif isinstance(self.coordinates, list) and all(
            isinstance(coord, tuple) for coord in self.coordinates
        ):
            # Polygon: [(x1, y1), (x2, y2), ..., (xn, yn)]
            mask = np.zeros_like(self.image.pixel_data, dtype=bool)
            rr, cc = zip(*self.coordinates)
            mask[cc, rr] = True
            cropped_pixel_data = self.image.pixel_data[mask].reshape((len(rr), -1))

        elif isinstance(self.coordinates, np.ndarray):
            # Mask: 2D numpy array of the same dimensions as the image
            mask = self.coordinates.astype(bool)
            cropped_pixel_data = self.image.pixel_data * mask
            y_indices, x_indices = np.nonzero(mask)
            y_min, y_max = y_indices.min(), y_indices.max() + 1
            x_min, x_max = x_indices.min(), x_indices.max() + 1
            cropped_pixel_data = cropped_pixel_data[y_min:y_max, x_min:x_max]

        else:
            raise ValueError("Unsupported coordinates type")

        cropped_image = copy.deepcopy(self.image)
        cropped_image.width = cropped_pixel_data.shape[1]
        cropped_image.height = cropped_pixel_data.shape[0]
        cropped_image.pixel_data = cropped_pixel_data

        return cropped_image

