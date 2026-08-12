import torch
import torch.nn.functional as F

from medical_image.data.image import Image, requires_loaded
from medical_image.utils.device import resolve_device


class FrequencyOperations:
    @staticmethod
    def high_frequency_energy(
            image: Image,
            output: Image,
            device=None,
    ) -> Image:
        """
        Computes high-frequency energy from SWT coefficients.

            EHF = LH² + HL² + HH²
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        coeffs = image.pixel_data.to(
            device
        ).float()

        if coeffs.ndim != 3 or coeffs.shape[0] != 4:
            raise ValueError(
                "Expected SWT coefficients with shape [4, H, W]."
            )

        LH = coeffs[1]
        HL = coeffs[2]
        HH = coeffs[3]

        EHF = (
                LH.square()
                + HL.square()
                + HH.square()
        )

        output.pixel_data = EHF

        return output

    @staticmethod
    @requires_loaded
    def stationary_wavelet_transform(
            image: Image,
            output: Image,
            device=None,
    ) -> Image:
        """
        One-level 2D Stationary Wavelet Transform (SWT).

        Haar SWT with no spatial downsampling.

        Output:
            [4, H, W]

        Channels:
            0 -> LL
            1 -> LH
            2 -> HL
            3 -> HH
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        img = image.pixel_data.to(device).float()

        while img.ndim > 2:
            img = img.squeeze(0)

        if img.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got shape {img.shape}"
            )

        H, W = img.shape

        sqrt2 = torch.sqrt(
            torch.tensor(
                2.0,
                device=device,
                dtype=img.dtype,
            )
        )

        low = torch.tensor(
            [1.0, 1.0],
            device=device,
            dtype=img.dtype,
        ) / sqrt2

        high = torch.tensor(
            [-1.0, 1.0],
            device=device,
            dtype=img.dtype,
        ) / sqrt2

        LL = torch.outer(low, low)
        LH = torch.outer(low, high)
        HL = torch.outer(high, low)
        HH = torch.outer(high, high)

        kernels = torch.stack(
            [LL, LH, HL, HH],
            dim=0,
        ).unsqueeze(1)

        # [1, 1, H, W]
        img = img.unsqueeze(0).unsqueeze(0)

        # ----------------------------------------------------------
        # Boundary handling
        # ----------------------------------------------------------
        #
        # Kernel = 2x2
        #
        # We need total padding = kernel_size - 1 = 1
        # in each dimension to preserve H x W.
        #
        # F.pad format:
        # (left, right, top, bottom)
        #
        padded = F.pad(
            img,
            (0, 1, 0, 1),
            mode="reflect",
        )

        coeffs = F.conv2d(
            padded,
            kernels,
            stride=1,
            padding=0,
        )

        # Should now be exactly [1, 4, H, W]
        coeffs = coeffs[:, :, :H, :W]

        output.pixel_data = coeffs.squeeze(0)

        return output
    @staticmethod
    @requires_loaded
    def fft(image: Image, output: Image, device=None) -> Image:
        """
        Computes the 2-dimensional Fast Fourier Transform (FFT) of an image.

        Args:
            image: Input image.
            output: Output image to store the complex FFT result.
            device: Device to perform computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()
        fft_result = torch.fft.fft2(img)
        output.pixel_data = fft_result.to(device)
        return output

    @staticmethod
    @requires_loaded
    def inverse_fft(image: Image, output: Image, device=None) -> Image:
        """
        Computes the inverse 2-dimensional Fast Fourier Transform (IFFT) of an image.

        Args:
            image: Input image in the frequency domain (complex tensor).
            output: Output image to store the inverse FFT result.
            device: Device to perform computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device)
        ifft_result = torch.fft.ifft2(img)
        output.pixel_data = ifft_result.to(device)
        return output
