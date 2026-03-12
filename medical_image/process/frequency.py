import torch

from medical_image.data.image import Image, requires_loaded


class FrequencyOperations:
    @staticmethod
    @requires_loaded
    def fft(image: Image, output: Image, device="cpu") -> Image:
        """
        Computes the 2-dimensional Fast Fourier Transform (FFT) of an image.

        Args:
            image: Input image.
            output: Output image to store the complex FFT result.
            device: Device to perform computation on.

        Returns:
            The output Image.
        """
        img = image.pixel_data.to(device).float()
        fft_result = torch.fft.fft2(img)
        output.pixel_data = fft_result.to(device)
        return output

    @staticmethod
    @requires_loaded
    def inverse_fft(image: Image, output: Image, device="cpu") -> Image:
        """
        Computes the inverse 2-dimensional Fast Fourier Transform (IFFT) of an image.

        Args:
            image: Input image in the frequency domain (complex tensor).
            output: Output image to store the inverse FFT result.
            device: Device to perform computation on.

        Returns:
            The output Image.
        """
        img = image.pixel_data.to(device)
        ifft_result = torch.fft.ifft2(img)
        output.pixel_data = ifft_result.to(device)
        return output
