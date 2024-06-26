from abc import ABC, abstractmethod

from medical_image.data.image import Image


class Algorithm(ABC):
    def __init__(self):
        """
        Constructor for the Algorithm class.
        """
        super().__init__()

    @abstractmethod
    def apply(self, image: Image, output: Image):
        """
        Apply the defined operations to the input image.

        Parameters:
        image: The input image to which the operations will be applied.

        Returns:
        The processed image after applying the operations.
        """
        pass
    # TODO: write unist test for this __call__ method
    def __call__(self, image: Image, output: Image):
        self.apply(image, output)
