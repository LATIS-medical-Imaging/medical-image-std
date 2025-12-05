from medical_image.algorithms.algorithm import Algorithm
from medical_image.process.filters import Filters
from medical_image.process.threshold import Threshold


class CustomAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.gaussian = lambda img, output: Filters.gaussian_filter(
            image_data=img, output=output, sigma=2.0, device=self.device
        )
        self.otsu = lambda image_data, output: Threshold.otsu_threshold(
            image_data=image_data, output=output, device=self.device
        )

    #   TODO: check garbage collector of Python
    def apply(self, image, output):
        self.gaussian(img=image, output=output)
        self.gaussian(img=output, output=output)
        self.otsu(image_data=output, output=output)
