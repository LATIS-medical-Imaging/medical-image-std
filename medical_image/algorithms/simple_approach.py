# from medical_image import BreastMaskAlgorithm
from medical_image import MammographyPreprocessing
from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image


class SimpleApproach(Algorithm):


    def __init__(self, method: str, device: str = "cpu"):
        super().__init__(device=device)
        # self.breast_mask = BreastMaskAlgorithm(mask_only=True)
        self.intensity_normalization = lambda dicom_image, mask_output: MammographyPreprocessing.robust_intensity_normalization(
            image=dicom_image,
            breast_mask=mask_output,
            device=device,
        )


    def apply(self, image: Image, output: Image) -> Image:
        """Apply the selected algorithm to the input image and store the result in output."""
        return output
