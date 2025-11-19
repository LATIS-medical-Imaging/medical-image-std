import copy

import numpy as np
import torch
import torch.nn.functional as F

from medical_image.data.image import Image
from medical_image.process.metrics import Metrics


class Threshold:
    @staticmethod
    def otsu_threshold(image_data: Image, output: Image = None):
        """
        Applies Otsu's thresholding method using pure PyTorch operations.
        Works on CPU and CUDA.

        Args:
            image_data (Image): The input image data (torch tensor).
            output (Image, optional): Optional output object.

        Returns:
            torch.Tensor: thresholded binary image (0 or 255)
        """

        # Get torch tensor (assume uint16 → convert to int32)
        image = image_data.pixel_data.to(torch.int32)

        device = image.device

        # 1) Compute histogram using torch.histc
        # bins = 4096 for 12-bit medical CT/MR images
        hist = torch.histc(image.float(), bins=4096, min=0, max=4095)

        # 2) Cumulative sums
        cumsum = torch.cumsum(hist, dim=0)

        # 3) Cumulative means
        values = torch.arange(4096, device=device, dtype=torch.float32)
        cummean = torch.cumsum(hist * values, dim=0)

        # 4) Global mean
        global_mean = cummean[-1]

        # 5) Between-class variance
        denom = cumsum * (cumsum[-1] - cumsum)
        numer = (global_mean * cumsum - cummean) ** 2

        # Avoid division by zero
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        between_class_variance = numer / denom

        # 6) Find threshold (argmax)
        threshold_value = torch.argmax(between_class_variance).item()

        # 7) Apply threshold (comparison works on int32)
        binary_image = (image > threshold_value).to(torch.uint8) * 255

        # 8) Write to output if provided
        if output is not None:
            output.pixel_data = binary_image

    @staticmethod
    def sauvola_threshold(
        image_data: "Image",
        output: "Image" = None,
        window_size: int = 10,
        k: float = 0.5,
        r: int = 128,
    ):
        """
        Applies Sauvola thresholding to a grayscale image using PyTorch tensors.
        """
        image = image_data.pixel_data  # assume PyTorch tensor: shape (H, W)

        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd integer.")

        pad = window_size // 2

        # Add batch and channel dimensions for conv2d
        img = image.unsqueeze(0).unsqueeze(0).float()  # shape (1, 1, H, W)

        # Create averaging kernel
        kernel = torch.ones((1, 1, window_size, window_size), device=image.device) / (
            window_size**2
        )

        # Local mean
        mean = F.conv2d(F.pad(img, (pad, pad, pad, pad), mode="replicate"), kernel)

        # Local squared mean for std
        mean_sq = F.conv2d(
            F.pad(img**2, (pad, pad, pad, pad), mode="replicate"), kernel
        )
        std = torch.sqrt(mean_sq - mean**2 + 1e-8)

        # Sauvola threshold
        thresh = mean * (1 + k * (std / r - 1))

        # Apply threshold
        thresh_image = torch.where(
            image > thresh.squeeze(0).squeeze(0),
            torch.tensor(255, device=image.device, dtype=torch.uint8),
            torch.tensor(0, device=image.device, dtype=torch.uint8),
        )

        # Assign to output
        if output is not None:
            output.pixel_data[:] = thresh_image

    @staticmethod
    def binarize(image_data: Image, output: Image, alpha: float):
        """
        This function binarize an image using local and global variance.
        For more infomation check this paper:
             https://www.researchgate.net/publication/306253912_Mammograms_calcifications_segmentation_based_on_band-pass_Fourier_filtering_and_adaptive_statistical_thresholding

        Parameters:
            input : a 2D ndarray.
            alpha : float
                 a scaling factor that relates the local and global variances.

        Returns:
            a 2D ndarray with the same size as the input containing 0 or 1 (a binary array)


        Examples:
             >>> a = np.random.randint(0, 5, (9,9))
             >>> a
             array([[3, 1, 0, 1, 1, 3, 4, 2, 2],
                    [0, 3, 3, 2, 3, 4, 0, 1, 1],
                    [0, 4, 4, 4, 3, 3, 1, 0, 3],
                    [4, 2, 3, 2, 2, 4, 2, 3, 4],
                    [2, 1, 3, 0, 0, 1, 4, 3, 1],
                    [2, 0, 0, 2, 0, 4, 0, 3, 1],
                    [4, 4, 4, 0, 4, 4, 1, 4, 2],
                    [2, 1, 3, 1, 2, 3, 1, 2, 0],
                    [4, 1, 3, 2, 3, 2, 3, 3, 0]])
             >>> Threshold.binarize(a, 0.5)
             array([[1, 1, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]])

        """
        # TODO: write unit test for this
        image = image_data.pixel_data
        image_out = output.pixel_data

        local_variance = copy.deepcopy(image)
        global_variance = copy.deepcopy(image)

        Metrics.local_variance(image, output=local_variance, kernel=5)
        Metrics.variance(image_out, output=global_variance)

        binary = local_variance.pixel_data**2 < (alpha * global_variance.pixel_data**2)
        output.pixel_data = np.where(binary, 0, 1)
