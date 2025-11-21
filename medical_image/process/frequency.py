import torch

from medical_image.data.image import Image


class FrequencyOperations:
    @staticmethod
    def fft(image_data: Image, output: Image, device="cpu"):
        """
        Computes the 2-dimensional Fast Fourier Transform (FFT) of an image.

        This function transforms the spatial domain image into the frequency domain
        using PyTorch's FFT implementation.

        Args:
            image_data (Image): Input image.
            output (Image): Output image to store the complex FFT result.
            device (str): Device to perform computation on ("cpu" or "cuda").

        Returns:
            None. The result is stored in output.pixel_data as a complex tensor.

        Example:
            >>> from medical_image.data.image import Image
            >>> fft_result = FrequencyOperations.fft(image, output, device="cuda")
        """
        # Move image to specified device
        img = image_data.pixel_data.to(device).float()

        # Compute 2D FFT
        fft_result = torch.fft.fft2(img)

        # Store result in output
        output.pixel_data = fft_result.to(device)
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def inverse_fft(image_data: Image, output: Image, device="cpu"):
        """
        Computes the inverse 2-dimensional Fast Fourier Transform (IFFT) of an image.

        This function transforms a frequency domain image back to the spatial domain
        using PyTorch's IFFT implementation.

        Args:
            image_data (Image): Input image in the frequency domain (complex tensor).
            output (Image): Output image to store the inverse FFT result.
            device (str): Device to perform computation on ("cpu" or "cuda").

        Returns:
            None. The result is stored in output.pixel_data as a complex or float tensor.

        Example:
            >>> from medical_image.data.image import Image
            >>> ifft_result = FrequencyOperations.inverse_fft(image, output, device="cuda")
        """
        # Move image to specified device
        img = image_data.pixel_data.to(device)

        # Compute inverse 2D FFT
        ifft_result = torch.fft.ifft2(img)

        # Store result in output
        output.pixel_data = ifft_result.to(device)
        output.width = image_data.width
        output.height = image_data.height
