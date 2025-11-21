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

        Args:
            image_data (Image): Input image with pixel_data as torch.Tensor.
            output (Image, optional): Optional output Image object to store the result.
            device (str): Device to perform computation ("cpu" or "cuda").

        Returns:
            torch.Tensor: Thresholded binary image (0 or 255) if output is None.
        """
        image = image_data.pixel_data.to(device).to(torch.int32)

        # Compute histogram
        hist = torch.histc(image.float(), bins=4096, min=0, max=4095)

        # Cumulative sums and means
        cumsum = torch.cumsum(hist, dim=0)
        values = torch.arange(4096, device=device, dtype=torch.float32)
        cummean = torch.cumsum(hist * values, dim=0)
        global_mean = cummean[-1]

        # Between-class variance
        denom = cumsum * (cumsum[-1] - cumsum)
        numer = (global_mean * cumsum - cummean) ** 2
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        variance_between = numer / denom

        # Threshold
        threshold_value = torch.argmax(variance_between).item()
        binary_image = (image > threshold_value).to(torch.uint8) * 255

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
