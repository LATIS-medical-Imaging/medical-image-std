import torch
import torch.nn.functional as F

from medical_image.data.image import Image
from medical_image.process.metrics import Metrics


class Threshold:
    @staticmethod
    def otsu_threshold(image_data: Image, output: Image = None, device="cpu"):
        """
        Applies Otsu's thresholding method to a grayscale image using PyTorch.
        Works on CPU or CUDA.

        This implementation adapts automatically to the min/max pixel values of the input image.

        Args:
            image_data (Image): Input image with pixel_data as torch.Tensor.
            output (Image, optional): Optional output Image object to store the result.
            device (str): Device to perform computation ("cpu" or "cuda").

        Returns:
            torch.Tensor: Thresholded binary image (0 or 255) if output is None.
        """
        # Move image to device
        image = image_data.pixel_data.to(device).to(torch.float32)

        # Determine the actual range of the image
        min_val = torch.min(image)
        max_val = torch.max(image)
        bins = 256 if max_val <= 255 else 4096

        hist = torch.histc(image, bins=bins, min=min_val.item(), max=max_val.item())
        bin_centers = torch.linspace(min_val, max_val, steps=bins, device=device)

        # Cumulative sums
        weight1 = torch.cumsum(hist, dim=0)
        weight2 = hist.sum() - weight1
        mean1 = torch.cumsum(hist * bin_centers, dim=0) / torch.clamp(weight1, min=1e-6)
        mean2 = (hist * bin_centers).sum() - torch.cumsum(hist * bin_centers, dim=0)
        mean2 = mean2 / torch.clamp(weight2, min=1e-6)

        # Between-class variance
        variance_between = weight1 * weight2 * (mean1 - mean2) ** 2
        threshold_idx = torch.argmax(variance_between)
        threshold_value = bin_centers[threshold_idx]

        # Apply threshold → 0/1 like skimage
        binary_image = (image > threshold_value).to(torch.uint8)

        if output is not None:
            output.pixel_data = binary_image
        else:
            return binary_image

    @staticmethod
    def sauvola_threshold(
        image_data: Image,
        output: Image = None,
        window_size: int = 10,
        k: float = 0.5,
        r: int = 128,
        device="cpu",
    ):
        """
        Applies Sauvola adaptive thresholding to a grayscale image using PyTorch.

        Args:
            image_data (Image): Input grayscale image.
            output (Image, optional): Optional Image object for result.
            window_size (int): Odd size of the local window.
            k (float): Scaling factor in threshold formula.
            r (int): Dynamic range of standard deviation (default 128).
            device (str): Device for computation ("cpu" or "cuda").

        Returns:
            torch.Tensor: Thresholded image (0 or 255) if output is None.
        """
        image = image_data.pixel_data.to(device).float()
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd integer.")
        pad = window_size // 2

        img = image.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        kernel = torch.ones((1, 1, window_size, window_size), device=device) / (
            window_size**2
        )

        mean = F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="replicate"), kernel)
        mean_sq = F.conv2d(
            F.pad(img**2, (pad, pad, pad, pad), mode="replicate"), kernel
        )
        std = torch.sqrt(mean_sq - mean**2 + 1e-8)

        thresh = mean * (1 + k * (std / r - 1))
        binary = torch.where(
            image > thresh.squeeze(0).squeeze(0),
            torch.tensor(255, device=device, dtype=torch.uint8),
            torch.tensor(0, device=device, dtype=torch.uint8),
        )

        if output is not None:
            output.pixel_data[:] = binary
        else:
            return binary

    @staticmethod
    def binarize(image_data: Image, output: Image, alpha: float, device="cpu"):
        """
        Binarizes an image based on local and global variance using PyTorch.

        Formula:
            binary = local_variance^2 < alpha * global_variance^2

        Args:
            image_data (Image): Input grayscale image.
            output (Image): Output Image object for storing result.
            alpha (float): Scaling factor relating local and global variances.
            device (str): Device for computation ("cpu" or "cuda").

        Returns:
            torch.Tensor: Binary image (0 or 1) if output is None.
        """
        image = image_data.pixel_data.to(device).float()

        # Local variance
        local_var_img = Image(image_data.file_path)
        local_var_img.pixel_data = torch.empty_like(image)
        Metrics.local_variance(image_data, output=local_var_img, kernel=5)

        # Global variance
        global_var_img = Image(image_data.file_path)
        global_var_img.pixel_data = torch.empty(1, device=device)
        Metrics.variance(image_data, output=global_var_img)

        # Compute binary mask
        binary = (
            local_var_img.pixel_data**2 >= alpha * global_var_img.pixel_data**2
        ).to(torch.uint8)

        output.pixel_data = binary
        output.width = image_data.width
        output.height = image_data.height
