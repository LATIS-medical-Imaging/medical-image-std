from abc import ABC, abstractmethod

import torch

from medical_image.data.image import Image


class Algorithm(ABC):
    def __init__(self, device: str = None):
        super().__init__()
        self.device = (
            device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        )

    @abstractmethod
    def apply(self, image: Image, output: Image) -> Image:
        """
        Apply the defined operations to the input image.

        Args:
            image: The input image.
            output: The output image to store results.

        Returns:
            The output image after applying the operations.
        """
        pass

    def __call__(self, image: Image, output: Image) -> Image:
        self.apply(image, output)
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}')"
