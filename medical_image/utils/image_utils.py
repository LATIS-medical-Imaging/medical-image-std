import os

import numpy as np
import torch
from PIL import Image as PILImage
from matplotlib import pyplot as plt

from medical_image.utils.logging import logger
from medical_image.data.image import Image, requires_loaded
from medical_image.utils.ErrorHandler import ErrorMessages


class TensorConverter:
    @staticmethod
    def to_numpy(image: Image) -> np.ndarray:
        """
        Convert Image.pixel_data (torch tensor) to NumPy array on CPU.

        Args:
            image: Image instance containing pixel_data.

        Returns:
            np.ndarray
        """
        tensor = image.pixel_data

        if tensor is None or not isinstance(tensor, torch.Tensor):
            raise ErrorMessages.invalid_pixel_data()

        return tensor.detach().cpu().numpy()

    @staticmethod
    def ensure_tensor(image: Image, device=None, dtype=None) -> torch.Tensor:
        """
        Move Image.pixel_data to target device and dtype.

        Args:
            image: Image instance.
            device: Target device.
            dtype: Target dtype.

        Returns:
            The updated tensor.
        """
        tensor = image.pixel_data

        if not isinstance(tensor, torch.Tensor):
            raise ErrorMessages.invalid_pixel_data()

        device = device or tensor.device
        dtype = dtype or tensor.dtype

        image.pixel_data = tensor.to(device=device, dtype=dtype)
        return image.pixel_data


class ImageExporter:
    """Export an Image object to PNG/JPG/TIFF."""

    @staticmethod
    def save_as(image: Image, format="PNG") -> str:
        if image.file_path is not None:
            base, _ = os.path.splitext(image.file_path)
        else:
            base = "dummy_data/sample_saved"
        output = f"{base}.{format.lower()}"

        np_img = TensorConverter.to_numpy(image)

        if np_img.dtype == np.float32 or np_img.dtype == np.float64:
            np_img = np.clip(np_img, 0, 255).astype("uint8")

        np_img = np.ascontiguousarray(np_img)

        PILImage.fromarray(np_img).save(output, format=format)

        logger.info(f"Image saved as {output}")
        return output


class ImageVisualizer:
    """Visualization utilities for Image objects."""

    @staticmethod
    def show(image: Image, cmap="gray", title=None):
        if image.pixel_data is None:
            raise ErrorMessages.invalid_pixel_data()

        np_img = TensorConverter.to_numpy(image)

        plt.imshow(np_img, cmap=cmap)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.show()

    @staticmethod
    def compare(
        before: Image, after: Image, title_before="Before", title_after="After"
    ):
        if before.pixel_data is None or after.pixel_data is None:
            raise ErrorMessages.invalid_pixel_data()

        before_np = TensorConverter.to_numpy(before)
        after_np = TensorConverter.to_numpy(after)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(before_np, cmap="gray")
        axes[0].set_title(title_before)
        axes[0].axis("off")

        axes[1].imshow(after_np, cmap="gray")
        axes[1].set_title(title_after)
        axes[1].axis("off")

        plt.show()


class MathematicalOperations:
    @staticmethod
    @requires_loaded
    def abs(image: Image, out: Image) -> Image:
        img = image.pixel_data.float()
        out.pixel_data = torch.abs(img)
        return out

    @staticmethod
    def euclidean_distance_sq(Z: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distances between N data points and c centroids.

        Args:
            Z: (N, d) data matrix.
            V: (c, d) centroid matrix.

        Returns:
            D2: (c, N) squared distances.
        """
        diff = V.unsqueeze(1) - Z.unsqueeze(0)
        return (diff**2).sum(dim=2)

    @staticmethod
    @requires_loaded
    def normalize_12bit(image: Image, out: Image) -> Image:
        """
        Normalize a 12-bit DICOM image to [0, 1] by dividing by 4095.

        Args:
            image: Input Image with raw 12-bit pixel values.
            out: Output Image to store the normalized result.

        Returns:
            The output Image.
        """
        out.pixel_data = torch.clamp(image.pixel_data.float() / 4095.0, 0.0, 1.0)
        return out
