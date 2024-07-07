from medical_image.algorithms.algorithm import Algorithm
from medical_image.process.filters import Filters
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.morphology import MorpohologyOperations
from medical_image.process.threshold import Threshold


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
        self.median = lambda img, output: Filters.median_filter(image_data=img, output=output, size=5)
        self.gamma = lambda img, output: Filters.gamma_correction(image_data=img, output=output, gamma=1.25)
        self.binarize = lambda img, output: Threshold.binarize(image_data=img, output=output, alpha=1)
        self.otsu = Threshold.otsu_threshold
        self.mophology_closing = MorpohologyOperations.morphoogy_closing
        self.region_fill = MorpohologyOperations.region_fill

