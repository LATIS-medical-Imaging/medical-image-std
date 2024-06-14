import numpy as np


class Threshold:
    @staticmethod
    def otsu_threshold(image_data: np.ndarray) -> np.ndarray:
        """This function calculates the Otsu thresholding of the image in the Dicom file.
            For more information about the Otsu method see this link:
            https://en.wikipedia.org/wiki/Otsu%27s_method

        Args:
            pixel_array: pixel_array in the Dicom file.

        Returns:
            The return value is the threshold image.

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
        return np.cumsum(np.cumsum(image, axis=0), axis=1)

    @staticmethod
    def sauvola_threshold(image_data: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
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

