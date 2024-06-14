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
        between_class_variance = (global_mean * cumsum - cummean) ** 2 / (cumsum * (cumsum[-1] - cumsum))

        # Find the threshold value that maximizes between-class variance
        threshold_value = np.nanargmax(between_class_variance)

        # Apply threshold
        binary_image = image_data > threshold_value
        return binary_image.astype(np.uint8) * 255

    @staticmethod
    def integral_image(image: np.ndarray) -> np.ndarray:
        """
        Computes the integral image of the given image data.

        The integral image is a data structure used for quick and efficient computation of sum queries over image subregions.
        Each element at (x, y) in the integral image is the sum of all pixels above and to the left of (x, y), inclusive.

        Args:
            image (np.ndarray): The input image data as a NumPy array.

        Returns:
            np.ndarray: The integral image as a NumPy array.
        """
        return np.cumsum(np.cumsum(image, axis=0), axis=1)

    @staticmethod
    def sauvola_threshold(image_data: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
        """
        Applies Sauvola's thresholding method to the given image data.

        Sauvola's method calculates a dynamic threshold for each pixel based on the mean and standard deviation
        of pixel intensities in a local neighborhood around the pixel. This method is particularly effective for images
        with varying illumination conditions.

        Args:
            image_data (np.ndarray): The input image data as a NumPy array with pixel values ranging from 0 to 4095.
            window_size (int, optional): The size of the local neighborhood window. Default is 25.
            k (float, optional): The parameter k that determines the sensitivity of the threshold to variance. Default is 0.2.

        Returns:
            np.ndarray: The binary thresholded image where pixel values are either 0 or 255.
        """
        # Padding to handle borders
        pad_size = window_size // 2
        padded_image = np.pad(image_data, pad_size, mode='reflect')

        # Compute integral images
        integral_image = Threshold.integral_image(padded_image)
        integral_sq_image = Threshold.integral_image(padded_image ** 2)

        # Compute local mean and variance using integral images
        rows, cols = padded_image.shape
        local_mean = np.zeros_like(image_data, dtype=np.float32)
        local_variance = np.zeros_like(image_data, dtype=np.float32)

        for i in range(pad_size, rows - pad_size):
            for j in range(pad_size, cols - pad_size):
                x1, x2 = i - pad_size, i + pad_size + 1
                y1, y2 = j - pad_size, j + pad_size + 1

                area = (x2 - x1) * (y2 - y1)

                sum_ = integral_image[x2, y2] - integral_image[x1, y2] - integral_image[x2, y1] + integral_image[x1, y1]
                sq_sum_ = integral_sq_image[x2, y2] - integral_sq_image[x1, y2] - integral_sq_image[x2, y1] + \
                          integral_sq_image[x1, y1]

                local_mean[i - pad_size, j - pad_size] = sum_ / area
                local_variance[i - pad_size, j - pad_size] = (sq_sum_ / area) - (
                            local_mean[i - pad_size, j - pad_size] ** 2)

        # Compute Sauvola threshold
        sauvola_threshold = local_mean * (1 + k * (np.sqrt(local_variance) / 2048 - 1))

        # Apply threshold
        binary_image = image_data > sauvola_threshold
        return binary_image.astype(np.uint8) * 255

