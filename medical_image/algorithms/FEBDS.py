import torch
from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.filters import Filters
from medical_image.process.frequency import FrequencyOperations
from medical_image.process.morphology import MorphologyOperations
from medical_image.process.threshold import Threshold
from medical_image.utils.image_utils import MathematicalOperations


class FebdsAlgorithm(Algorithm):
    """
    Fourier Enhancement and Band-pass Filtering Algorithm for Microcalcification Segmentation.

    References:
        @article{article,
        author = {Lopez, Elizabeth and Urcid, Gonzalo},
        year = {2016},
        month = {05},
        pages = {},
        title = {Mammograms calcifications segmentation based on band-pass Fourier filtering and adaptive statistical thresholding},
        volume = {5}
        }

    Math and Logic:
        This algorithm aims to enhance microcalcifications by highlighting high-frequency
        components while removing noise and low-frequency background signals.
        It supports filtering in the spatial domain using Difference of Gaussians (DoG)
        or Laplacian of Gaussian (LoG), or in the frequency domain using a Fast
        Fourier Transform (FFT) with a Butterworth band-pass filter.
        After enhancement, the image is denoised via median filtering and gamma correction
        is applied to amplify the calcifications, followed by adaptive thresholding
        (like Otsu's) or binarization, and morphological closing to reconstruct regions.

    Pipeline:
        1. Apply base enhancement filter depending on the method ('dog', 'log', 'fft').
        2. Denoise and smooth by taking the absolute value and applying a median filter.
        3. Apply gamma correction to increase the contrast of microcalcifications.
        4. Apply global thresholding (binarize for 'fft', Otsu for 'dog'/'log').
        5. Apply morphological closing and region filling to restore shape and connectivity.

    Example Usage:
        ```python
        from medical_image.algorithms.FEBDS import FebdsAlgorithm
        from medical_image.data.dicom_image import DicomImage

        img = DicomImage("20527054.dcm")
        img.load()

        algo = FebdsAlgorithm(method="dog", device="cpu")
        output = img.clone()
        algo(img, output)
        ```
    """

    def __init__(self, method: str, device: str = "cpu"):
        super().__init__(device=device)
        self.method = method

        # Filters
        self.dog = lambda img, out: Filters.difference_of_gaussian(
            image=img,
            output=out,
            low_sigma=1.7,
            high_sigma=2.0,
            device=self.device,
        )
        self.log = lambda img, out: Filters.laplacian_of_gaussian(
            image=img, output=out, sigma=2.0, device=self.device
        )
        self.median = lambda img, out: Filters.median_filter(
            image=img, output=out, size=5, device=self.device
        )
        self.gamma = lambda img, out: Filters.gamma_correction(
            image=img, output=out, gamma=1.25, device=self.device
        )
        self.abs = lambda img, out: MathematicalOperations.abs(image=img, out=out)
        # Frequency domain
        self.fft = lambda img, out: FrequencyOperations.fft(
            image=img, output=out, device=self.device
        )
        self.inverse_fft = lambda img, out: FrequencyOperations.inverse_fft(
            image=img, output=out, device=self.device
        )
        self.butter_kernel = lambda img, out: Filters.butterworth_kernel(
            image=img, output=out, device=self.device
        )

        # Thresholds
        self.binarize = lambda img, out: Threshold.binarize(
            image=img, output=out, alpha=1, device=self.device
        )
        self.otsu = lambda img, out: Threshold.otsu_threshold(
            image=img, output=out, device=self.device
        )

        # Morphology
        self.morphology_closing = (
            lambda img, out: MorphologyOperations.morphology_closing(
                image=img, output=out, device=self.device
            )
        )
        self.region_fill = lambda img, out: MorphologyOperations.region_fill(
            image=img, output=out, device=self.device
        )

    def apply(self, image: Image, output: Image) -> Image:
        """Applies the selected method pipeline in-place on output."""
        # Step 1: Base filter
        if self.method == "dog":
            self.dog(image, output)
        elif self.method == "log":
            self.log(image, output)
        elif self.method == "fft":
            self.fft(image, output)

            # Frequency band-pass with Butterworth kernel
            kernel_img = InMemoryImage(array=torch.zeros_like(output.pixel_data))
            self.butter_kernel(output, kernel_img)
            output.pixel_data *= kernel_img.pixel_data

            self.inverse_fft(output, output)

        # Step 2: Denoise and smoothing
        self.abs(output, output)
        self.median(output, output)

        # Step 3: Gamma correction
        self.gamma(output, output)

        # Step 4: Thresholding
        if self.method == "fft":
            self.binarize(output, output)
        else:
            self.otsu(output, output)

        # Step 5: Morphological post-processing
        self.morphology_closing(output, output)
        self.region_fill(output, output)

        return output
