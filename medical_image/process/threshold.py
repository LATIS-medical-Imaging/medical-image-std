import torch
import torch.nn.functional as F

from medical_image.data.image import Image, requires_loaded
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.metrics import Metrics


class Threshold:
    @staticmethod
    @requires_loaded
    def otsu_threshold(image: Image, output: Image = None, device="cpu") -> Image:
        """
        Applies Otsu's thresholding method to a grayscale image using PyTorch.

        Args:
            image: Input image with pixel_data as torch.Tensor.
            output: Optional output Image object to store the result.
            device: Device to perform computation.

        Returns:
            The output Image (or a new InMemoryImage if output is None).
        """
        img = image.pixel_data.to(device).to(torch.float32)

        min_val = torch.min(img)
        max_val = torch.max(img)
        bins = 256 if max_val <= 255 else 4096

        hist = torch.histc(img, bins=bins, min=min_val.item(), max=max_val.item())
        bin_centers = torch.linspace(min_val, max_val, steps=bins, device=device)

        weight1 = torch.cumsum(hist, dim=0)
        weight2 = hist.sum() - weight1
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / torch.clamp(weight1, min=1e-6)
        mean2 = (hist * bin_centers).sum() - torch.cumsum(hist * bin_centers, dim=0)
        mean2 = mean2 / torch.clamp(weight2, min=1e-6)

        variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
        threshold_idx = torch.argmax(variance_between)
        threshold_value = bin_centers[threshold_idx]

        binary_image = (img > threshold_value).to(torch.uint8)

        if output is None:
            output = InMemoryImage(array=binary_image)
        else:
            output.pixel_data = binary_image
        return output

    @staticmethod
    @requires_loaded
    def sauvola_threshold(
        image: Image,
        output: Image = None,
        window_size: int = 10,
        k: float = 0.5,
        r: int = 128,
        device="cpu",
    ) -> Image:
        """
        Applies Sauvola adaptive thresholding to a grayscale image using PyTorch.

        Args:
            image: Input grayscale image.
            output: Optional Image object for result.
            window_size: Odd size of the local window.
            k: Scaling factor in threshold formula.
            r: Dynamic range of standard deviation.
            device: Device for computation.

        Returns:
            The output Image (or a new InMemoryImage if output is None).
        """
        img = image.pixel_data.to(device).float()
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd integer.")
        pad = window_size // 2

        img4d = img.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, window_size, window_size), device=device) / (
            window_size**2
        )

        mean = F.conv2d(F.pad(img4d, (pad, pad, pad, pad), mode="replicate"), kernel)
        mean_sq = F.conv2d(
            F.pad(img4d**2, (pad, pad, pad, pad), mode="replicate"), kernel
        )
        std = torch.sqrt(mean_sq - mean**2 + 1e-8)

        thresh = mean * (1 + k * (std / r - 1))
        binary = torch.where(
            img > thresh.squeeze(0).squeeze(0),
            torch.tensor(255, device=device, dtype=torch.uint8),
            torch.tensor(0, device=device, dtype=torch.uint8),
        )

        if output is None:
            output = InMemoryImage(array=binary)
        else:
            output.pixel_data = binary
        return output

    @staticmethod
    @requires_loaded
    def binarize(image: Image, output: Image, alpha: float, device="cpu") -> Image:
        """
        Binarizes an image based on local and global variance using PyTorch.

        Formula:
            binary = local_variance^2 < alpha * global_variance^2

        Args:
            image: Input grayscale image.
            output: Output Image object for storing result.
            alpha: Scaling factor relating local and global variances.
            device: Device for computation.

        Returns:
            The output Image.
        """
        img = image.pixel_data.to(device).float()

        # Local variance
        local_var_img = InMemoryImage(array=torch.empty_like(img))
        Metrics.local_variance(image, output=local_var_img, kernel=5)

        # Global variance
        global_var_img = InMemoryImage(array=torch.empty(1, device=device))
        Metrics.variance(image, output=global_var_img)

        # Compute binary mask
        binary = (
            local_var_img.pixel_data**2 >= alpha * global_var_img.pixel_data**2
        ).to(torch.uint8)

        output.pixel_data = binary
        return output
