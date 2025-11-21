from typing import Union

import torch

from medical_image.data.image import Image


class Metrics:
    @staticmethod
    def entropy(image: Image, decimals=4, device="cpu") -> float:
        """
        Calculates the Shannon entropy of an image using PyTorch.

        Args:
            image (Image): Input image.
            decimals (int, optional): Number of decimal places to round to. Default is 4.
            device (str): Device to perform computation on ("cpu" or "cuda").

        Returns:
            float: Shannon entropy of the image.

        Notes:
            Uses base-2 logarithm (bits).
        """
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
    def joint_entropy(image1: Image, image2: Image, decimals=4, device="cpu") -> float:
        """
        Calculates the joint Shannon entropy of two images.

        Args:
            image1 (Image): First input image.
            image2 (Image): Second input image.
            decimals (int, optional): Decimal precision. Default is 4.
            device (str): Device for computation ("cpu" or "cuda").

        Returns:
            float: Joint entropy value.
        """
        img1 = image1.pixel_data.to(device).flatten()
        img2 = image2.pixel_data.to(device).flatten()
        # 2D histogram
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
    def mutual_information(
        image1: Image, image2: Image, decimals=4, device="cpu"
    ) -> float:
        """
        Computes the mutual information between two images.

        Args:
            image1 (Image): First image.
            image2 (Image): Second image.
            decimals (int, optional): Decimal precision. Default is 4.
            device (str): Device for computation ("cpu" or "cuda").

        Returns:
            float: Mutual information value.
        """
        mi = (
            Metrics.entropy(image1, decimals, device)
            + Metrics.entropy(image2, decimals, device)
            - Metrics.joint_entropy(image1, image2, decimals, device)
        )
        return mi

    @staticmethod
    def local_variance(
        image: Image, output: Image, kernel: Union[int, tuple], device="cpu"
    ):
        """
        Computes the local variance for each sub-region of the image using a sliding window.

        Args:
            image (Image): Input image.
            output (Image): Image object to store local variance.
            kernel (int or tuple): Window size for local variance.
            device (str): Device for computation ("cpu" or "cuda").
        """
        img = image.pixel_data.to(device).float()
        # Add batch & channel dims for unfolding
        img_unfold = img.unsqueeze(0).unsqueeze(0)
        if isinstance(kernel, int):
            kh, kw = kernel, kernel
        else:
            kh, kw = kernel
        patches = img_unfold.unfold(2, kh, 1).unfold(3, kw, 1)  # (1,1,H,W,kh,kw)
        patches = patches.contiguous().view(1, 1, img.shape[0], img.shape[1], -1)
        local_var = patches.var(dim=-1)
        output.pixel_data = local_var.squeeze(0).squeeze(0)
        output.width = image.width
        output.height = image.height

    @staticmethod
    def variance(image: Image, output: Image, device="cpu"):
        """
        Computes the global variance of an image.

        Args:
            image (Image): Input image.
            output (Image): Image object to store the variance as a scalar tensor.
            device (str): Device for computation ("cpu" or "cuda").
        """
        img = image.pixel_data.to(device).float()
        var_val = torch.var(img)
        output.pixel_data = var_val
