import os
from abc import ABC, abstractmethod
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np

from log_manager import logger
from medical_image.utils.annotation import Annotation
from medical_image.utils.ErrorHandler import ErrorMessages

T = TypeVar("T")
from PIL import Image as PILImage


class Image(ABC):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise ErrorMessages.file_not_found(file_path)
        self.file_path = file_path
        self.width = None
        self.height = None
        self.pixel_data = None
        self.label: Annotation = None

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

