import numpy as np

from medical_image.data.image import Image


class Filters:
    @staticmethod
    def convolution(image_data: Image, output: Image, kernel: np.ndarray):
        """
        Applies a convolution filter to the given image using the specified kernel.

        Args:
            image_data (Image): The input image data encapsulated in an Image object.
            output (Image): An Image object to store the convolved image.
            kernel (np.ndarray): The convolution kernel as a 2D NumPy array.

        Returns:
            None
        """
        image = image_data.pixel_data
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad the image
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                              constant_values=0)

        # Initialize the output image
        convolved_image = np.zeros_like(image)

        # Perform the convolution
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + kernel_height, j:j + kernel_width]
                convolved_image[i, j] = np.sum(region * kernel)

        np.copyto(output.pixel_data, convolved_image)

    @staticmethod
    def gaussian_filter(image_data: Image, output: Image, sigma: float):
        """
        Applies a Gaussian filter to the given image.

        Args:
            image_data (Image): The input image data encapsulated in an Image object.
            output (Image): An Image object to store the filtered image.
            sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
            None
        """
        size = int(2 * np.ceil(3 * sigma) + 1)
        kernel = Filters._generate_gaussian_kernel(size, sigma)
        Filters.convolution(image_data, output, kernel)

    @staticmethod
    def _generate_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """
        Generates a Gaussian kernel.

        Args:
            size (int): The size of the kernel.
            sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
            np.ndarray: The Gaussian kernel.
        """
        k = size // 2
        x = np.arange(-k, k + 1)
        y = np.arange(-k, k + 1)
        x, y = np.meshgrid(x, y)
        kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return kernel / np.sum(kernel)

    @staticmethod
    def median_filter(image_data: Image, output: Image, size: int):
        """
        Applies a median filter to the given image.

        Args:
            image_data (Image): The input image data encapsulated in an Image object.
            output (Image): An Image object to store the filtered image.
            size (int): The size of the filter window (must be an odd integer).

        Returns:
            None
        """
        image = image_data.pixel_data
        pad_size = size // 2

        # Pad the image
        padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)

        # Initialize the output image
        filtered_image = np.zeros_like(image)

        # Perform the median filtering
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i + size, j:j + size]
                filtered_image[i, j] = np.median(region)

        np.copyto(output.pixel_data, filtered_image)