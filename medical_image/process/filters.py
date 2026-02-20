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
    def gaussian_filter(
        image_data: Image,
        output: Image,
        sigma: float,
        device="cpu",
        truncate: float = 4.0,
    ):
        """
        Applies Gaussian filter
        """
        dtype = torch.float32
        output.pixel_data = image_data.pixel_data.clone()
        output.width = image_data.width
        output.height = image_data.height

        img = image_data.pixel_data.to(device).float()
        # --- make kernel ---
        kernel = Filters._generate_gaussian_kernel(
            sigma, truncate=truncate, dtype=dtype, device=device
        )
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # --- convert image to tensor ---
        img_t = (
            torch.tensor(img, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)
        )  # [1,1,H,W]

        # --- padding ---
        pad = kernel.shape[-1] // 2
        img_padded = F.pad(img_t, (pad, pad, pad, pad), mode="replicate")

        # --- convolution ---
        out = F.conv2d(img_padded, kernel)
        output.pixel_data = out.squeeze(0).squeeze(0)
        # return out.squeeze(0).squeeze(0).numpy()  # [H, W]

    @staticmethod
    def _generate_gaussian_kernel(
        sigma: float, dtype=torch.float32, truncate: float = 4.0, device="cpu"
    ) -> torch.Tensor:
        """
        Generates a 2D Gaussian kernel.

        Args:
            sigma (float): Standard deviation of the Gaussian.
            truncate (float): Truncate the filter at this many standard deviations.
            device (str): PyTorch device.

        Returns:
            torch.Tensor: 2D Gaussian kernel (float64 to match skimage).
        """
        # Compute the radius based on sigma and truncate
        radius = int(truncate * sigma + 0.5)
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=dtype, device=device) - radius
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel2d = g[:, None] @ g[None, :]  # outer product
        kernel2d = kernel2d / kernel2d.sum()
        return kernel2d

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
            raise ValueError("Median filter size must be odd")

        img = image_data.pixel_data.to(device).float()
        if img.ndim == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.ndim == 3:
            img = img.unsqueeze(0)

        pad = size // 2
        padded = F.pad(img, (pad, pad, pad, pad), mode="reflect")
        patches = padded.unfold(2, size, 1).unfold(3, size, 1)
        patches = patches.contiguous().view(
            img.shape[0], img.shape[1], img.shape[2], img.shape[3], -1
        )
        filtered = patches.median(dim=-1).values

        output.pixel_data = filtered.squeeze(0)
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
        image_data: Image,
        output: Image,
        low_sigma: float,
        high_sigma: float | None = None,
        device="cpu",
        truncate=4.0,
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
        if low_sigma <= 0:
            raise ValueError("low_sigma must be > 0")

        if high_sigma is None:
            high_sigma = low_sigma * 1.6
        if high_sigma < low_sigma:
            raise ValueError("high_sigma must be >= low_sigma")

        # Move input to device and float
        img = image_data.pixel_data.to(device).float()

        # Temporary buffers (just tensors, no Image object)
        temp_low = img.clone()
        temp_high = img.clone()

        # Apply Gaussian filters directly to the tensors
        Filters.gaussian_filter(
            image_data=image_data,
            output=ImagePlaceholder(temp_low),
            sigma=low_sigma,
            device=device,
            truncate=truncate,
        )
        Filters.gaussian_filter(
            image_data=image_data,
            output=ImagePlaceholder(temp_high),
            sigma=high_sigma,
            device=device,
            truncate=truncate,
        )

        # Compute DoG in-place
        output.pixel_data = (temp_low - temp_high).to(device)
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
        temp = Image(array=image_data.pixel_data.clone())
        temp.width = image_data.width
        temp.height = image_data.height
        Filters.gaussian_filter(image_data, temp, sigma=sigma, device=device)

        g = temp.pixel_data
        if g.ndim == 2:
            g = g.unsqueeze(0).unsqueeze(0)
        elif g.ndim == 3:
            g = g.unsqueeze(0)

        lap_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device
        )
        lap_kernel = lap_kernel.unsqueeze(0).unsqueeze(0).repeat(g.shape[1], 1, 1, 1)
        g = F.pad(g, (1, 1, 1, 1), mode="reflect")
        out = F.conv2d(g, lap_kernel, groups=g.shape[1])

        output.pixel_data = out.squeeze(0)
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
        img = image_data.pixel_data.to(device).float()
        max_val = img.max()
        corrected = (img / max_val) ** gamma * max_val
        output.pixel_data = corrected
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
        img = img.to(torch.int32)
        bins = (
            255 if torch.max(img) <= 255 else 4095
        )  # adapt to 8-bit or higher bit images
        alpha = contrast / (float(bins) / 2) + 1.0
        beta = brightness - contrast
        adjusted = torch.clamp(img * alpha + beta, 0, bins)
        output.pixel_data = adjusted.to(device)
        output.width = image_data.width
        output.height = image_data.height
