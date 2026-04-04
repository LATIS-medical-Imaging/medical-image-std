import torch
import torch.nn.functional as F

from medical_image.data.image import Image, requires_loaded
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.utils.device import resolve_device, get_dtype


class Filters:
    @staticmethod
    @requires_loaded
    def convolution(
        image: Image, output: Image, kernel: torch.Tensor, device=None
    ) -> Image:
        """
        Applies a convolution filter to the given image using PyTorch.

        Args:
            image: Input image object.
            output: Output image object.
            kernel: 2D convolution kernel.
            device: Device to perform computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()

        if not isinstance(kernel, torch.Tensor):
            kernel = torch.tensor(kernel, dtype=torch.float32, device=device)
        else:
            kernel = kernel.float().to(device)

        img = img.unsqueeze(0).unsqueeze(0)
        k = kernel.unsqueeze(0).unsqueeze(0)

        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2

        convolved = F.conv2d(F.pad(img, (pad_w, pad_w, pad_h, pad_h)), k)

        output.pixel_data = convolved.squeeze(0).squeeze(0).to(device)
        return output

    @staticmethod
    @requires_loaded
    def gaussian_filter(
        image: Image,
        output: Image,
        sigma: float,
        device=None,
        truncate: float = 4.0,
    ) -> Image:
        """Applies Gaussian filter."""
        device = resolve_device(image, explicit=device)
        dtype = torch.float32
        output.pixel_data = image.pixel_data.clone()

        img = image.pixel_data.to(device).float()
        kernel = Filters._generate_gaussian_kernel(
            sigma, truncate=truncate, dtype=dtype, device=device
        )
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        img_t = (
            img.detach()
            .clone()
            .to(dtype=dtype, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        pad = kernel.shape[-1] // 2
        img_padded = F.pad(img_t, (pad, pad, pad, pad), mode="replicate")

        out = F.conv2d(img_padded, kernel)
        output.pixel_data = out.squeeze(0).squeeze(0)
        return output

    @staticmethod
    def _generate_gaussian_kernel(
        sigma: float, dtype=torch.float32, truncate: float = 4.0, device=None
    ) -> torch.Tensor:
        """Generates a 2D Gaussian kernel."""
        device = device or torch.device("cpu")
        radius = int(truncate * sigma + 0.5)
        size = 2 * radius + 1
        coords = torch.arange(size, dtype=dtype, device=device) - radius
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel2d = g[:, None] @ g[None, :]
        kernel2d = kernel2d / kernel2d.sum()
        return kernel2d

    @staticmethod
    @requires_loaded
    def median_filter(image: Image, output: Image, size: int, device=None) -> Image:
        """
        Applies a median filter using PyTorch.

        Args:
            image: Input image.
            output: Output image.
            size: Odd kernel size.
            device: Device to run computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        if size % 2 == 0:
            raise ValueError("Median filter size must be odd")

        img = image.pixel_data.to(device).float()
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
        return output

    @staticmethod
    @requires_loaded
    def butterworth_kernel(
        image: Image,
        output: Image,
        D_0: float = 21,
        W: float = 32,
        n: int = 3,
        device=None,
    ) -> Image:
        """
        Applies a Butterworth band-pass filter in the frequency domain.

        Args:
            image: Input image.
            output: Output image.
            D_0: Cutoff frequency.
            W: Bandwidth.
            n: Filter order.
            device: Device to run computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        H, Wd = image.height, image.width
        u = torch.arange(Wd, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        D = torch.sqrt((uu - Wd / 2) ** 2 + (vv - H / 2) ** 2)
        band = D**2 - D_0**2
        cutoff = 8 * W * D
        cutoff = torch.where(cutoff == 0, torch.tensor(1e-6, device=device), cutoff)
        kernel = 1.0 / (1 + (band / cutoff).pow(2 * n))

        output.pixel_data = kernel.to(device)
        return output

    @staticmethod
    @requires_loaded
    def difference_of_gaussian(
        image: Image,
        output: Image,
        low_sigma: float,
        high_sigma: float | None = None,
        device=None,
        truncate=4.0,
    ) -> Image:
        """
        Applies Difference of Gaussian (DoG) filter.

        Args:
            image: Input image.
            output: Output image.
            low_sigma: First Gaussian sigma.
            high_sigma: Second Gaussian sigma.
            device: Device to run computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        if low_sigma <= 0:
            raise ValueError("low_sigma must be > 0")

        if high_sigma is None:
            high_sigma = low_sigma * 1.6
        if high_sigma < low_sigma:
            raise ValueError("high_sigma must be >= low_sigma")

        img1 = InMemoryImage(array=image.pixel_data.clone())
        img2 = InMemoryImage(array=image.pixel_data.clone())

        Filters.gaussian_filter(
            image, img1, low_sigma, truncate=truncate, device=device
        )
        Filters.gaussian_filter(
            image, img2, high_sigma, truncate=truncate, device=device
        )
        output.pixel_data = img1.pixel_data - img2.pixel_data
        return output

    @staticmethod
    @requires_loaded
    def laplacian_of_gaussian(
        image: Image, output: Image, sigma: float, device=None
    ) -> Image:
        """
        Applies Laplacian of Gaussian (LoG) filter.

        Args:
            image: Input image.
            output: Output image.
            sigma: Gaussian sigma.
            device: Device to run computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        temp = InMemoryImage(array=image.pixel_data.clone())
        Filters.gaussian_filter(image, temp, sigma=sigma, device=device)

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
        return output

    @staticmethod
    @requires_loaded
    def gamma_correction(
        image: Image, output: Image, gamma: float, device=None
    ) -> Image:
        """
        Applies Gamma Correction.

        Args:
            image: Input image.
            output: Output image.
            gamma: Gamma value.
            device: Device to run computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()
        corrected = (img / 4095) ** gamma * 4095.0
        output.pixel_data = corrected
        return output

    @staticmethod
    @requires_loaded
    def contrast_adjust(
        image: Image,
        output: Image,
        contrast: float,
        brightness: float,
        device=None,
    ) -> Image:
        """
        Adjusts contrast and brightness.

        Args:
            image: Input image.
            output: Output image.
            contrast: Contrast coefficient.
            brightness: Brightness coefficient.
            device: Device to run computation on (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device)
        img = img.to(torch.int32)
        bins = 255 if torch.max(img) <= 255 else 4095
        alpha = contrast / (float(bins) / 2) + 1.0
        beta = brightness - contrast
        adjusted = torch.clamp(img * alpha + beta, 0, bins)
        output.pixel_data = adjusted.to(device)
        return output

    # ------------------------------------------------------------------
    # Batch variants
    # ------------------------------------------------------------------

    @staticmethod
    def gaussian_filter_batch(
        images: torch.Tensor,
        sigma: float,
        device=None,
        truncate: float = 4.0,
    ) -> torch.Tensor:
        """
        Apply Gaussian filter to a batch of images.

        Args:
            images: Batched tensor (B, C, H, W).
            sigma: Gaussian sigma.
            device: Target device.
            truncate: Kernel truncation factor.

        Returns:
            Filtered batch (B, C, H, W).
        """
        device = device or images.device
        images = images.to(device).float()
        kernel = Filters._generate_gaussian_kernel(
            sigma, truncate=truncate, device=device
        )
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        pad = kernel.shape[-1] // 2
        padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
        return F.conv2d(padded, kernel)
