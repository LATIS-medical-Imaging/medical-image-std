import numpy as np

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.filters import Filters
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.morphology import MorphologyOperations
from medical_image.process.threshold import Threshold


# TODO: Apply on ROI


class FebdsAlgorithm(Algorithm):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.dog = lambda img, output: Filters.difference_of_gaussian(
            image_data=img, output=output, sigma_1=2.0, sigma_2=1.7, device=self.device
        )
        self.log = lambda img, output: Filters.laplacian_of_gaussian(
            image_data=img, output=output, sigma=2.0, device=self.device
        )
        self.fft = lambda img, output: FrequencyOperations.fft(
            image_data=img, output=output, device=self.device
        )

        self.butter_kernel = lambda img, output: Filters.butterworth_kernel(
            image_data=img, output=output, device=self.device
        )
        self.inverse_fft = lambda img, output: FrequencyOperations.inverse_fft(
            image_data=img, output=output, device=self.device
        )
        self.median = lambda img, output: Filters.median_filter(
            image_data=img, output=output, size=5, device=self.device
        )
        self.gamma = lambda img, output: Filters.gamma_correction(
            image_data=img, output=output, gamma=1.25, device=self.device
        )
        self.binarize = lambda img, output: Threshold.binarize(
            image_data=img, output=output, alpha=1, device=self.device
        )
        self.otsu = Threshold.otsu_threshold
        self.mophology_closing = (
            lambda img, output: MorphologyOperations.morphology_closing(
                image_data=img, output=output, device=self.device
            )
        )
        self.region_fill = lambda img, output: MorphologyOperations.region_fill(
            image_data=img, output=output, device=self.device
        )

    def apply(self, image: Image, output: Image):
        if self.method == "dog":
            self.dog(image, output)
        elif self.method == "log":
            self.log(image, output)
        elif self.method == "fft":
            self.fft(image, output)
            kernel = self.butter_kernel(output)
            output.pixel_data = np.multiply(output.pixel_data, kernel)
            self.inverse_fft(image, output)

        self.median(image, output)
        self.gamma(image, output)
        if self.method == "fft":
            self.binarize(image, output)
        else:
            self.otsu(image, output)
        self.mophology_closing(image, output)
        self.region_fill(image, output)
