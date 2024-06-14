import numpy as np


class Threshold:
    @staticmethod
    def otsu_threshold(image_data: np.ndarray) -> np.ndarray:
        """
        Applies Otsu's thresholding method to the given image data.

        Otsu's method automatically determines the optimal threshold value by maximizing the between-class variance.
        This method is particularly useful for bimodal images where the histogram of pixel intensities has two peaks.

        For more information about the Otsu method, see this link:
        https://en.wikipedia.org/wiki/Otsu%27s_method

        Args:
            image_data (np.ndarray): The input image data as a NumPy array with pixel values ranging from 0 to 4095.

        Returns:
            np.ndarray: The binary thresholded image where pixel values are either 0 or 255.
        """
        # Compute histogram
        hist, bins = np.histogram(image_data.flatten(), bins=4096, range=(0, 4096))

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

        # Find the threshold value that maximizes between-class variance
        threshold_value = np.nanargmax(between_class_variance)

        # Apply threshold
        binary_image = image_data > threshold_value
        return binary_image.astype(np.uint8) * 255

    @staticmethod
    def sauvola_threshold(image, window_size=10, k=0.5, r=128):
        """
        Applies Sauvola thresholding to a grayscale image.

        Args:
            image: A 2D grayscale image as a NumPy array.
            window_size: The size of the local neighborhood window (odd integer).
            k: The weighting factor for the standard deviation.
            r: The dynamic range parameter.

        Returns:
            A 2D binary image (0s and 1s) as a NumPy array.
        """

        # Check for odd window size
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd integer.")

        # Pad the image to handle borders
        pad_width = window_size // 2
        image_padded = np.pad(image, pad_width, mode="edge")

        # Initialize output image
        thresh_image = np.zeros_like(image)

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

        return thresh_image
