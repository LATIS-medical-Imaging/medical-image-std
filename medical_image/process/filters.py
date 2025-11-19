import numpy as np
import torch
from torch.nn import functional as F

from medical_image.data.image import Image


# TODO: Update everything to be in TORCH
class Filters:
    @staticmethod
    def convolution(image_data: Image, output: Image, kernel):
        """
        Applies a convolution filter to the given image using PyTorch.

        Args:
            image_data (Image): Input image object.
            output (Image): Output image object.
            kernel (np.ndarray or torch.Tensor): 2D convolution kernel.

        Returns:
            None
        """
        image = image_data.pixel_data  # Could be uint16

        # Convert to float32 for convolution
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        else:
            image = image.float()

        # Convert kernel to torch.Tensor
        if not isinstance(kernel, torch.Tensor):
            kernel = torch.from_numpy(kernel).float()
        else:
            kernel = kernel.float()

        # Add batch and channel dimensions
        img = image.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        k = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,KH,KW)

        # Compute padding
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2

        # Apply convolution
        convolved = F.conv2d(F.pad(img, (pad_w, pad_w, pad_h, pad_h)), k)

        # Remove batch/channel dimensions
        convolved = convolved.squeeze(0).squeeze(0)

        # Cast back to original dtype if needed
        if output.pixel_data.dtype != torch.float32:
            convolved = convolved.to(output.pixel_data.dtype)

        # Copy to output
        output.pixel_data[:] = convolved

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
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
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
        padded_image = np.pad(image, pad_size, mode="constant", constant_values=0)

        # Initialize the output image
        filtered_image = np.zeros_like(image)

        # Perform the median filtering
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i : i + size, j : j + size]
                filtered_image[i, j] = np.median(region)

        np.copyto(output.pixel_data, filtered_image)

    @staticmethod
    def butterworth_kernel(image_data: Image, D_0=21, W=32, n=3) -> np.ndarray:
        """
        todo: update Docstring
        Apply a Butterworth band-pass filter to enhance frequency features.
        This filter is defined in the Fourier domain.
        For more:
           https://en.wikipedia.org/wiki/Butterworth_filter

        Parameters:
            input: 2d ndarray to process.
            D_0: float
                 cutoff frequency
            W: float
               filter bandwidth
            n: int
               filter order
        Returns:
            band_pass: ndarray
                       The Butterworth-kernel.


        Examples:
            >>> a = np.random.randint(0, 5, (3,3))
            >>> PixelArrayOperation.butterworth_kernel(a)
            array([[0.78760795, 0.3821997 , 0.3821997 ],
                  [0.3821997 , 0.00479278, 0.00479278],
                  [0.3821997 , 0.00479278, 0.00479278]])

        """
        x = image_data.width
        y = image_data.height
        u, v = np.meshgrid(np.arange(x), np.arange(y))
        D = np.sqrt((u - x / 2) ** 2 + (v - y / 2) ** 2)
        band_pass = D**2 - D_0**2
        cuttoff = 8 * W * D
        denom = 1.0 + (band_pass / cuttoff) ** (2 * n)
        band_pass = 1.0 / denom
        return band_pass.transpose()

    @staticmethod
    def difference_of_gaussian(
        image_data: Image, output: Image, sigma_1: float, sigma_2: float
    ):
        """
        Applies the Difference of Gaussian (DoG) filter to the given image.

        Args:
            image_data (Image): The input image data encapsulated in an Image object.
            output (Image): An Image object to store the filtered image.
            sigma_1 (float): The standard deviation of the first Gaussian kernel.
            sigma_2 (float): The standard deviation of the second Gaussian kernel.

        Returns:
            None
        """
        image = image_data.pixel_data
        gaussian1 = Filters.gaussian_filter(image, sigma_1)
        gaussian2 = Filters.gaussian_filter(image, sigma_2)
        dog = gaussian1 - gaussian2
        output.pixel_data = dog

    @staticmethod
    def laplacian_of_gaussian(image_data: Image, output: Image, sigma: float):
        """
        Applies the Laplacian of Gaussian (LoG) filter to the given image.

        Args:
            image_data (Image): The input image data encapsulated in an Image object.
            output (Image): An Image object to store the filtered image.
            sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
            None
        """
        image = image_data.pixel_data
        gaussian = Filters.gaussian_filter(image, sigma)
        laplacian = (
            np.gradient(np.gradient(gaussian)[0])[0]
            + np.gradient(np.gradient(gaussian)[1])[1]
        )
        output.pixel_data = laplacian

    @staticmethod
    def gamma_correction(image_data: Image, output: Image, gamma):
        """This function calculates the Gamma Correction of the image in the Dicom file.
            For more information about the Gamma Correction see this link:
            https://en.wikipedia.org/wiki/Gamma_correction

        Args:
            pixel_array: pixel_array in the Dicom file.
            max: Maximum value that can be assigned to a pixel.

        Returns:
            The return value is the corrected image.

        """
        image = image_data.pixel_data
        image = np.power(image / float(4095), gamma)
        output.pixel_data = image * 4095

    @staticmethod
    def ContrastAdjust(image_data: Image, output: Image, contrast, brightness):
        """This function used to adjust the contrast and the brightness
        the image in the Dicom file.
            This approach is based this equation that can be used to apply
        both contrast and brightness at the same time:
            ****
            new_img = alpha*old_img + beta
            ****
            Where alpha and beta are contrast and brightness coefficient respectively:

            alpha 1  beta 0      --> no change
            0 < alpha < 1        --> lower contrast
            alpha > 1            --> higher contrast
            -2047 < beta < +2047   --> good range for brightness values

            In my case:
                alpha = contrast / 2047 + 1
                beta = brightness - contrast

        Args:
            pixel_array: pixel_array in the Dicom file.
            contrast: the contrast value to be applied.
            brightness: the brightness value to be applied.

        Returns:
            The return value is an image with different contrast and brightness.

        """
        image = image_data.pixel_data
        quotient = 4095 // 2
        alpha = contrast / quotient + 1
        beta = brightness - contrast

        output.pixel_data = np.clip(image * alpha + beta, 0, 4095)
