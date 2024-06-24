import os
from abc import ABC, abstractmethod
from typing import TypeVar

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

    def to_png(self, output_path):
        """
        Save a NumPy array as a PNG file.

        Parameters:
        pixel_data (np.ndarray): The NumPy array containing the pixel data.
        file_path (str): The file path where the PNG image will be saved.
        """
        if self.pixel_data is None and not isinstance(self.pixel_data, np.ndarray):
            raise ValueError("pixel_data is not a valid NumPy array.")

        image = PILImage.fromarray(self.pixel_data)
        image.save(output_path)
        logger.info(f"Image saved successfully at {output_path}")
