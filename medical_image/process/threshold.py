import copy

import numpy as np

from medical_image.data.image import Image
from medical_image.process.metrics import Metrics


class Threshold:
    @staticmethod
    def otsu_threshold(image_data: Image, output: Image = None):
        """
        Applies Otsu's thresholding method to the given image data.

        Otsu's method automatically determines the optimal threshold value by maximizing the between-class variance.
        This method is particularly useful for bimodal images where the histogram of pixel intensities has two peaks.

        For more information about the Otsu method, see this link:
        https://en.wikipedia.org/wiki/Otsu%27s_method

        Args:
            image_data (Image): The input image data as a Image array with pixel values ranging from 0 to 4095.
            output (np.ndarray, optional): An optional output array to store the binary thresholded image.

        Returns:
            np.ndarray: The binary thresholded image where pixel values are either 0 or 255.
        """
        image = image_data.pixel_data
        # Compute histogram
        hist, bins = np.histogram(image.flatten(), bins=4096, range=(0, 4096))

        # Compute cumulative sums of the histogram
        cumsum = np.cumsum(hist)

        # Compute cumulative means
        cummean = np.cumsum(hist * np.arange(4096))

        # Compute global mean
        global_mean = cummean[-1]

        # Compute between-class variance
        between_class_variance = (global_mean * cumsum - cummean) ** 2 / (
            cumsum * (cumsum[-1] - cumsum)
        )

        # Replace NaNs with zeroes (since division by zero can occur)
        between_class_variance = np.nan_to_num(between_class_variance)

        # Find the threshold value that maximizes between-class variance
        threshold_value = np.argmax(between_class_variance)

        # Apply threshold
        binary_image = image > threshold_value
        binary_image = binary_image.astype(np.uint8) * 255

        # If an output array is provided, copy the result to it
        if output.pixel_data is not None:
            output.pixel_data = binary_image

    @staticmethod
    def sauvola_threshold(
        image_data: Image,
        output: Image = None,
        window_size: int = 10,
        k: float = 0.5,
        r: int = 128,
    ):
        """
        Applies Sauvola thresholding to a grayscale image.

        Sauvola's method calculates the local threshold value for each pixel in a grayscale image based on the mean
        and standard deviation of the pixel values in a local window. This method is particularly useful for images with
        varying illumination.

        Args:
            image (np.ndarray): A 2D grayscale image as a NumPy array.
            output (np.ndarray, optional): An optional output array to store the binary thresholded image.
            window_size (int): The size of the local neighborhood window (must be an odd integer). Default is 10.
            k (float): The weighting factor for the standard deviation. Default is 0.5.
            r (int): The dynamic range parameter. Default is 128.

        Returns:
            np.ndarray: The binary thresholded image where pixel values are either 0 or 255.
        """
        image = image_data.pixel_data
        image_out = output.pixel_data

        # Check for odd window size
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd integer.")

        # Pad the image to handle borders
        pad_width = window_size // 2
        image_padded = np.pad(image, pad_width, mode="edge")

        # Initialize output image
        thresh_image = np.zeros_like(image, dtype=np.uint8)

        # Iterate through each pixel (avoiding padded borders)
        for i in range(pad_width, image.shape[0] + pad_width):
            for j in range(pad_width, image.shape[1] + pad_width):
                # Extract local window
                window = image_padded[
                    i - pad_width : i + pad_width + 1, j - pad_width : j + pad_width + 1
                ]

                # Calculate local mean and standard deviation
                mean = np.mean(window)
                std = np.std(window)

                # Apply Sauvola formula
                threshold = mean * (1 + k * (std / r - 1))

                # Apply threshold
                thresh_image[i - pad_width, j - pad_width] = (
                    255 if image[i - pad_width, j - pad_width] > threshold else 0
                )

        # If an output array is provided, copy the result to it
        if image_out is not None:
            np.copyto(image_out, thresh_image)
        output.pixel_data = image_out

    @staticmethod
    def binarize(image_data: Image, output: Image, alpha: float):
        """
        This function binarize an image using local and global variance.
        For more infomation check this paper:
             https://www.researchgate.net/publication/306253912_Mammograms_calcifications_segmentation_based_on_band-pass_Fourier_filtering_and_adaptive_statistical_thresholding

        Parameters:
            input : a 2D ndarray.
            alpha : float
                 a scaling factor that relates the local and global variances.

        Returns:
            a 2D ndarray with the same size as the input containing 0 or 1 (a binary array)


        Examples:
             >>> a = np.random.randint(0, 5, (9,9))
             >>> a
             array([[3, 1, 0, 1, 1, 3, 4, 2, 2],
                    [0, 3, 3, 2, 3, 4, 0, 1, 1],
                    [0, 4, 4, 4, 3, 3, 1, 0, 3],
                    [4, 2, 3, 2, 2, 4, 2, 3, 4],
                    [2, 1, 3, 0, 0, 1, 4, 3, 1],
                    [2, 0, 0, 2, 0, 4, 0, 3, 1],
                    [4, 4, 4, 0, 4, 4, 1, 4, 2],
                    [2, 1, 3, 1, 2, 3, 1, 2, 0],
                    [4, 1, 3, 2, 3, 2, 3, 3, 0]])
             >>> Threshold.binarize(a, 0.5)
             array([[1, 1, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        """
        # TODO: write unit test for this
        image = image_data.pixel_data
        image_out = output.pixel_data

        local_variance = copy.deepcopy(image)
        global_variance = copy.deepcopy(image)

        Metrics.local_variance(image, output=local_variance, kernel=5)
        Metrics.variance(image_out, output=global_variance)

        binary = local_variance.pixel_data**2 < (alpha * global_variance.pixel_data**2)
        output.pixel_data = np.where(binary, 0, 1)
