import torch
from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.process.filters import Filters
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.morphology import MorphologyOperations
from medical_image.process.threshold import Threshold
from medical_image.utils.image_utils import ImageVisualizer


# TODO: Apply on ROI


class FebdsAlgorithm(Algorithm):
    def __init__(self, method: str, device: str = "cpu"):
        super().__init__()
        self.method = method
        self.device = device

        # Filters
        self.dog = lambda img, out: Filters.difference_of_gaussian(
            image_data=img,
            output=out,
            low_sigma=1.7,
            high_sigma=2.0,
            device=self.device,
        )
        self.log = lambda img, out: Filters.laplacian_of_gaussian(
            image_data=img, output=out, sigma=2.0, device=self.device
        )
        self.median = lambda img, out: Filters.median_filter(
            image_data=img, output=out, size=5, device=self.device
        )
        self.gamma = lambda img, out: Filters.gamma_correction(
            image_data=img, output=out, gamma=1.25, device=self.device
        )

        # Frequency domain
        self.fft = lambda img, out: FrequencyOperations.fft(
            image_data=img, output=out, device=self.device
        )
        self.inverse_fft = lambda img, out: FrequencyOperations.inverse_fft(
            image_data=img, output=out, device=self.device
        )
        self.butter_kernel = lambda img, out: Filters.butterworth_kernel(
            image_data=img, output=out, device=self.device
        )

        # Thresholds
        self.binarize = lambda img, out: Threshold.binarize(
            image_data=img, output=out, alpha=1, device=self.device
        )
        self.otsu = lambda img, out: Threshold.otsu_threshold(
            image_data=img, output=out, device=self.device
        )

        # Morphology
        self.morphology_closing = (
            lambda img, out: MorphologyOperations.morphology_closing(
                image_data=img, output=out, device=self.device
            )
        )
        self.region_fill = lambda img, out: MorphologyOperations.region_fill(
            image_data=img, output=out, device=self.device
        )

    def apply(self, image: Image, output: Image):
        """Applies the selected method pipeline in-place on output"""
        # Step 1: Base filter
        if self.method == "dog":
            self.dog(image, output)
        elif self.method == "log":
            self.log(image, output)
        elif self.method == "fft":
            self.fft(image, output)

            # Frequency band-pass with Butterworth kernel
            kernel_img = Image(array=torch.zeros_like(output.pixel_data))
            self.butter_kernel(output, kernel_img)
            output.pixel_data *= kernel_img.pixel_data

            self.inverse_fft(image, output)

        # Step 2: Denoise and smoothing
        self.median(output, output)

        # Step 3: Show intermediate result
        # ImageVisualizer.show(output)

        # Step 4: Gamma correction
        self.gamma(output, output)

        # Step 5: Thresholding
        if self.method == "fft":
            self.binarize(output, output)
        else:
            self.otsu(output, output)

        # Step 6: Morphological post-processing
        self.morphology_closing(output, output)
        self.region_fill(output, output)
