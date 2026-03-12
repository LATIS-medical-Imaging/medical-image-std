from abc import ABC, abstractmethod
from typing import List

import torch

from medical_image.data.image import Image
from medical_image.utils.device import Precision


class Algorithm(ABC):
    def __init__(self, device: str = None, precision: Precision = Precision.FULL):
        super().__init__()
        self.device = (
            device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.precision = precision

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
        if self.precision != Precision.FULL and self.device != "cpu":
            with torch.cuda.amp.autocast(dtype=self.precision.value):
                self.apply(image, output)
        else:
            self.apply(image, output)
        return output

    def apply_batch(self, images: List[Image], outputs: List[Image]) -> List[Image]:
        """
        Process a batch of images. Default: loop over apply().
        Subclasses can override for truly batched GPU processing.
        """
        for img, out in zip(images, outputs):
            self.apply(img, out)
        return outputs

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}', precision={self.precision.name})"