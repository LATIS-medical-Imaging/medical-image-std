from typing import Union

import torch
import torch.nn.functional as F

from medical_image.data.image import Image, requires_loaded
from medical_image.utils.device import resolve_device


class Metrics:
    @staticmethod
    @requires_loaded
    def entropy(image: Image, decimals=4, device=None) -> float:
        """
        Calculates the Shannon entropy of an image using PyTorch.

        Args:
            image: Input image.
            decimals: Number of decimal places to round to.
            device: Device to perform computation on (None = infer from image).

        Returns:
            Shannon entropy of the image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).flatten()
        hist = torch.histc(
            img,
            bins=int(img.max() - img.min() + 1),
            min=float(img.min()),
            max=float(img.max()),
        )
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy_val = -torch.sum(probs * torch.log2(probs))
        return round(entropy_val.item(), decimals)

    @staticmethod
    @requires_loaded
    def joint_entropy(image1: Image, image2: Image, decimals=4, device=None) -> float:
        """
        Calculates the joint Shannon entropy of two images.

        Args:
            image1: First input image.
            image2: Second input image.
            decimals: Decimal precision.
            device: Device for computation (None = infer from image).

        Returns:
            Joint entropy value.
        """
        device = resolve_device(image1, image2, explicit=device)
        img1 = image1.pixel_data.to(device).flatten()
        img2 = image2.pixel_data.to(device).flatten()
        min1, max1 = float(img1.min()), float(img1.max())
        min2, max2 = float(img2.min()), float(img2.max())
        bins1 = int(max1 - min1 + 1)
        bins2 = int(max2 - min2 + 1)
        joint_hist = torch.histc(
            img1 * bins2 + img2, bins=bins1 * bins2, min=0, max=bins1 * bins2 - 1
        )
        joint_prob = joint_hist / joint_hist.sum()
        joint_prob = joint_prob[joint_prob > 0]
        joint_entropy_val = -torch.sum(joint_prob * torch.log2(joint_prob))
        return round(joint_entropy_val.item(), decimals)

    @staticmethod
    @requires_loaded
    def mutual_information(
        image1: Image, image2: Image, decimals=4, device=None
    ) -> float:
        """
        Computes the mutual information between two images.

        Args:
            image1: First image.
            image2: Second image.
            decimals: Decimal precision.
            device: Device for computation (None = infer from image).

        Returns:
            Mutual information value.
        """
        mi = (
            Metrics.entropy(image1, decimals, device)
            + Metrics.entropy(image2, decimals, device)
            - Metrics.joint_entropy(image1, image2, decimals, device)
        )
        return mi

    @staticmethod
    @requires_loaded
    def local_variance(
        image: Image, output: Image, kernel: Union[int, tuple], device=None
    ) -> Image:
        """
        Computes the local variance for each sub-region of the image.

        Args:
            image: Input image.
            output: Image object to store local variance.
            kernel: Window size for local variance.
            device: Device for computation (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()
        img_unfold = img.unsqueeze(0).unsqueeze(0)
        if isinstance(kernel, int):
            kh, kw = kernel, kernel
        else:
            kh, kw = kernel
        pad_h, pad_w = kh // 2, kw // 2
        img_unfold = F.pad(img_unfold, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        patches = img_unfold.unfold(2, kh, 1).unfold(3, kw, 1)
        patches = patches.contiguous().view(
            1, 1, patches.shape[2], patches.shape[3], -1
        )
        local_var = patches.var(dim=-1)
        output.pixel_data = local_var.squeeze(0).squeeze(0)
        return output

    @staticmethod
    @requires_loaded
    def variance(image: Image, output: Image, device=None) -> Image:
        """
        Computes the global variance of an image.

        Args:
            image: Input image.
            output: Image object to store the variance as a scalar tensor.
            device: Device for computation (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img = image.pixel_data.to(device).float()
        var_val = torch.var(img)
        output.pixel_data = var_val
        return output
