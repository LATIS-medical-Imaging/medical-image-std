import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.morphology import MorphologyOperations


class TopHatAlgorithm(Algorithm):
    """
    White Top-Hat enhancement algorithm for microcalcification detection.

    TopHat(I) = I - opening(I, SE)

    Highlights bright structures (MCs) smaller than the structuring element.
    MATLAB reference: SE = strel('disk', 4); ROI = imtophat(I_ROI, SE);

    Args:
        radius: Disk SE radius (default 4 → 9×9 footprint).
        device: Torch device.
    """

    def __init__(self, radius: int = 4, device: str = "cpu"):
        super().__init__(device=device)
        self.radius = radius

    def apply(self, image: Image, output: Image):
        """
        Apply white top-hat to the image.

        Args:
            image: Input Image (2D float, e.g. normalized [0,1]).
            output: Output Image — pixel_data will contain top-hat result.
        """
        MorphologyOperations.white_top_hat(
            image_data=image,
            output=output,
            radius=self.radius,
            device=self.device,
        )
