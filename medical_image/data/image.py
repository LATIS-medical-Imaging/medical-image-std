import os
from abc import ABC, abstractmethod
from typing import TypeVar

import matplotlib.pyplot as plt
import torch

from log_manager import logger
from medical_image.utils.ErrorHandler import ErrorMessages
from medical_image.utils.annotation import Annotation

T = TypeVar("T")
from PIL import Image as PILImage


class Image(ABC):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise ErrorMessages.file_not_found(file_path)
        self.file_path = file_path
        self.width = None
        self.height = None
        self.pixel_data: torch.Tensor = None
        self.label: Annotation = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        if self.pixel_data is None or not isinstance(self.pixel_data, torch.Tensor):
            raise ErrorMessages.invalid_pixel_data()
        filename, _ = os.path.splitext(self.file_path)
        image = PILImage.fromarray(self.pixel_data.detach().cpu().numpy())
        image.save(filename + ".png")
        logger.info(f"Image saved successfully at {filename + '.png'}")

    def plot(self, cmap="gray"):
        """Display the image using matplotlib.

        Parameters:
        cmap (str): Colormap to use for displaying the image. Default is 'gray'.
        """
        if self.pixel_data is None:
            raise ErrorMessages.invalid_pixel_data()
        plt.imshow(self.to_numpy(), cmap=cmap)
        plt.axis("off")  # Hide axes
        plt.show()

    def to_numpy(self):
        return self.pixel_data.detach().cpu().numpy()

