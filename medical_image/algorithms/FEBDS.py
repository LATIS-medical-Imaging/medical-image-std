from medical_image.algorithms.algorithm import Algorithm
from medical_image.process.filters import Filters
from medical_image.process.frequency import FrequencyOperations


class FebdsAlgorithm(Algorithm):
    # TODO: Check this file:
    #       https://github.com/HamzaGbada/dicomPreProcess/blob/master/Mapper/mathOperation.py#L224
    def __init__(self, method):
        self.method = method
        self.dog = lambda img, output: Filters.difference_of_gaussian(
            image_data=img, output=output, sigma_1=2.0, sigma_2=1.7
        )
        self.log = lambda img, output: Filters.laplacian_of_gaussian(
            image_data=img, output=output, sigma=2.0
        )
        self.fft = lambda img, output: FrequencyOperations.fft(
            image_data=img, output=output
        )

        self.butter_kernel = Filters.butterworth_kernel