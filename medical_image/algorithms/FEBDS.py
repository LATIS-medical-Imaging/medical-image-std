import numpy as np

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.filters import Filters
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.morphology import MorpohologyOperations
from medical_image.process.threshold import Threshold


# TODO: Apply on ROI

class FebdsAlgorithm(Algorithm):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.dog = lambda img, output: Filters.difference_of_gaussian(
            image_data=img, output=output, sigma_1=2.0, sigma_2=1.7
        )
        self.log = lambda img, output: Filters.laplacian_of_gaussian(
            image_data=img, output=output, sigma=2.0
        )
        self.fft = lambda img, output: FrequencyOperations.fft

        self.butter_kernel = Filters.butterworth_kernel
        self.inverse_fft = lambda img, output: FrequencyOperations.inverse_fft
        self.median = lambda img, output: Filters.median_filter(
            image_data=img, output=output, size=5
        )
        self.gamma = lambda img, output: Filters.gamma_correction(
            image_data=img, output=output, gamma=1.25
        )
        self.binarize = lambda img, output: Threshold.binarize(
            image_data=img, output=output, alpha=1
        )
        self.otsu = Threshold.otsu_threshold
        self.mophology_closing = MorpohologyOperations.morphoogy_closing
        self.region_fill = MorpohologyOperations.region_fill

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