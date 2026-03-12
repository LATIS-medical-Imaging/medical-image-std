from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.filters import Filters
from medical_image.process.threshold import Threshold


class CustomAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.gaussian = lambda img, output: Filters.gaussian_filter(
            image=img, output=output, sigma=2.0, device=self.device
        )
        self.otsu = lambda img, output: Threshold.otsu_threshold(
            image=img, output=output, device=self.device
        )

    def apply(self, image: Image, output: Image) -> Image:
        self.gaussian(img=image, output=output)
        self.gaussian(img=output, output=output)
        self.otsu(img=output, output=output)
        return output
