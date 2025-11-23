import numpy as np
import torch
import torch.nn.functional as F

from medical_image.data.image import Image


class Filters:
    @staticmethod
    def convolution(
        image_data: Image, output: Image, kernel: torch.Tensor, device="cpu"
    ):
        """
        Applies a convolution filter to the given image using PyTorch.

        Args:
            image_data (Image): Input image object.
            output (Image): Output image object.
            kernel (torch.Tensor or np.ndarray): 2D convolution kernel.
            device (str): Device to perform computation on ("cpu" or "cuda").

        Returns:
            None
        """
        # Move image to device and ensure float
        image = image_data.pixel_data.to(device).float()

        # Convert kernel to torch tensor and move to device
        if not isinstance(kernel, torch.Tensor):
            kernel = torch.tensor(kernel, dtype=torch.float32, device=device)
        else:
            kernel = kernel.float().to(device)

        # Add batch and channel dimensions
        img = image.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        k = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,KH,KW)

        # Compute padding for 'same' convolution
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2

        # Apply convolution
        convolved = F.conv2d(F.pad(img, (pad_w, pad_w, pad_h, pad_h)), k)

        # Remove batch/channel dims
        output.pixel_data = convolved.squeeze(0).squeeze(0).to(device)
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def gaussian_filter(image_data: Image, output: Image, sigma: float, device="cpu"):
        """
        Applies a Gaussian filter to the given image.

        Args:
            image_data (Image): Input image object.
            output (Image): Output image object.
            sigma (float): Standard deviation of the Gaussian kernel.
            device (str): Device to perform computation on ("cpu" or "cuda").

        Returns:
            None
        """
        # Determine kernel size
        size = int(2 * torch.ceil(torch.tensor(3 * sigma)) + 1)

        # Generate kernel on device
        kernel = Filters._generate_gaussian_kernel(size, sigma, device=device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

        # Prepare image tensor
        img = image_data.pixel_data.to(device).unsqueeze(0).unsqueeze(0).float()

        # Apply convolution with padding
        padding = size // 2
        filtered = F.conv2d(img, kernel, padding=padding)

        # Save result to output
        output.pixel_data = filtered.squeeze(0).squeeze(0).to(device)
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def _generate_gaussian_kernel(
        size: int, sigma: float, device="cpu"
    ) -> torch.Tensor:
        """
        Generates a 2D Gaussian kernel.

        Args:
            size (int): Kernel size.
            sigma (float): Standard deviation.
            device (str): Device to create the kernel on.

        Returns:
            torch.Tensor: 2D Gaussian kernel.
        """
        k = size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
        y = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        return kernel

    @staticmethod
    def median_filter(image_data: Image, output: Image, size: int, device="cpu"):
        """
        Applies a median filter using PyTorch.

        Args:
            image_data (Image): Input image.
            output (Image): Output image.
            size (int): Odd kernel size.
            device (str): Device to run computation on.
        """
        if size % 2 == 0:
            raise ValueError("Median filter size must be an odd integer.")

        img = (
            image_data.pixel_data.to(device).float().unsqueeze(0).unsqueeze(0)
        )  # (1,1,H,W)
        pad = size // 2
        padded = F.pad(img, (pad, pad, pad, pad), mode="constant", value=0)

        patches = padded.unfold(2, size, 1).unfold(3, size, 1)  # (1,1,H,W,size,size)
        patches = patches.contiguous().view(1, 1, img.shape[2], img.shape[3], -1)

        filtered = patches.median(dim=-1).values.squeeze(0).squeeze(0)
        output.pixel_data = filtered.to(device)
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def butterworth_kernel(
        image_data: Image,
        output: Image,
        D_0: float = 21,
        W: float = 32,
        n: int = 3,
        device="cpu",
    ):
        """
        Applies a Butterworth band-pass filter in the frequency domain.

        Args:
            image_data (Image): Input image.
            output (Image): Output image.
            D_0 (float): Cutoff frequency.
            W (float): Bandwidth.
            n (int): Filter order.
            device (str): Device to run computation on.
        """
        H, Wd = image_data.height, image_data.width
        u = torch.arange(Wd, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        D = torch.sqrt((uu - Wd / 2) ** 2 + (vv - H / 2) ** 2)
        band = D**2 - D_0**2
        cutoff = 8 * W * D
        cutoff = torch.where(cutoff == 0, torch.tensor(1e-6, device=device), cutoff)
        kernel = 1.0 / (1 + (band / cutoff).pow(2 * n))

        output.pixel_data = kernel.to(device)
        output.width = Wd
        output.height = H

    @staticmethod
    def difference_of_gaussian(
        image_data: Image, output: Image, sigma_1: float, sigma_2: float, device="cpu"
    ):
        """
        Applies Difference of Gaussian (DoG) filter.

        Args:
            image_data (Image): Input image.
            output (Image): Output image.
            sigma_1 (float): First Gaussian sigma.
            sigma_2 (float): Second Gaussian sigma.
            device (str): Device to run computation on.
        """
        # Temporary images for Gaussian results
        g1 = type(image_data)(image_data.file_path)
        g2 = type(image_data)(image_data.file_path)
        for tmp in [g1, g2]:
            tmp.width = image_data.width
            tmp.height = image_data.height
            tmp.device = device

        Filters.gaussian_filter(image_data, g1, sigma_1, device)
        Filters.gaussian_filter(image_data, g2, sigma_2, device)

        output.pixel_data = (g1.pixel_data - g2.pixel_data).to(device)
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def laplacian_of_gaussian(
        image_data: Image, output: Image, sigma: float, device="cpu"
    ):
        """
        Applies Laplacian of Gaussian (LoG) filter.

        Args:
            image_data (Image): Input image.
            output (Image): Output image.
            sigma (float): Gaussian sigma.
            device (str): Device to run computation on.
        """
        # Step 1: Gaussian blur
        blurred = type(image_data)(image_data.file_path)
        blurred.width = image_data.width
        blurred.height = image_data.height
        blurred.device = device
        Filters.gaussian_filter(image_data, blurred, sigma, device)
        g = blurred.pixel_data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Step 2: Laplacian kernel
        lap_kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        lap = F.conv2d(g, lap_kernel, padding=1).squeeze(0).squeeze(0)

        output.pixel_data = lap.to(device)
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def gamma_correction(image_data: Image, output: Image, gamma: float, device="cpu"):
        """
        Applies Gamma Correction.

        Args:
            image_data (Image): Input image.
            output (Image): Output image.
            gamma (float): Gamma value.
            device (str): Device to run computation on.
        """
        img = image_data.pixel_data.to(device)

        # Choose number of bins
        bins = (
            255 if torch.max(img) <= 255 else 4095
        )  # adapt to 8-bit or higher bit images
        corrected = torch.pow(img / float(bins), gamma) * float(bins)
        output.pixel_data = corrected.to(device)
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def contrast_adjust(
        image_data: Image,
        output: Image,
        contrast: float,
        brightness: float,
        device="cpu",
    ):
        """
        Adjusts contrast and brightness.

        Args:
            image_data (Image): Input image.
            output (Image): Output image.
            contrast (float): Contrast coefficient.
            brightness (float): Brightness coefficient.
            device (str): Device to run computation on.
        """
        img = image_data.pixel_data.to(device)
        bins = (
            255 if torch.max(img) <= 255 else 4095
        )  # adapt to 8-bit or higher bit images
        alpha = contrast / (float(bins) / 2) + 1.0
        beta = brightness - contrast
        adjusted = torch.clamp(img * alpha + beta, 0, bins)
        output.pixel_data = adjusted.to(device)
        output.width = image_data.width
        output.height = image_data.height
