import os
import matplotlib
import numpy as np

# Use a backend that works with PyCharm and plt.show()
matplotlib.use("TkAgg")
import torch
from PIL import Image as PILImage
from matplotlib import pyplot as plt

from log_manager import logger
from medical_image.data.image import Image
from medical_image.utils.ErrorHandler import ErrorMessages


# TODO: perform unit test
class TensorConverter:
    @staticmethod
    def to_numpy(image: Image):
        """
        Convert Image.pixel_data (torch tensor) to NumPy array on CPU.

        Args:
            image (Image): Image instance containing pixel_data.

        Returns:
            np.ndarray
        """
        tensor = image.pixel_data

        if tensor is None or not isinstance(tensor, torch.Tensor):
            raise ErrorMessages.invalid_pixel_data()

        return tensor.detach().cpu().numpy()

    @staticmethod
    def ensure_tensor(image: Image, device=None, dtype=None):
        """
        Move Image.pixel_data to target device and dtype.

        Args:
            image (Image)
            device (str): 'cpu' or 'cuda'. Defaults to image.device.
            dtype: torch dtype.

        Returns:
            torch.Tensor (updated inside image.pixel_data)
        """
        tensor = image.pixel_data

        if not isinstance(tensor, torch.Tensor):
            raise ErrorMessages.invalid_pixel_data()

        device = device or tensor.device
        dtype = dtype or tensor.dtype

        image.pixel_data = tensor.to(device=device, dtype=dtype)
        return image.pixel_data


class ImageExporter:
    """
    Export an Image object to PNG/JPG/TIFF using clean separation of concerns.
    """

    @staticmethod
    def save_as(image: Image, format="PNG"):
        if image.file_path is not None:
            base, _ = os.path.splitext(image.file_path)
        else:
            base = "dummy_data/sample_saved"
        output = f"{base}.{format.lower()}"

        np_img = TensorConverter.to_numpy(image)

        # --- FIX HERE ---
        # Convert float32 images [0..255] to uint8
        if np_img.dtype == np.float32 or np_img.dtype == np.float64:
            np_img = np.clip(np_img, 0, 255).astype("uint8")

        # Pillow requires contiguous array
        np_img = np.ascontiguousarray(np_img)
        # --- END FIX ---

        PILImage.fromarray(np_img).save(output, format=format)

        logger.info(f"Image saved as {output}")
        return output


class ImageVisualizer:
    """
    Visualization utilities for Image objects.
    """

    @staticmethod
    def show(image: Image, cmap="gray", title=None):
        """
        Display Image.pixel_data with matplotlib.

        Args:
            image (Image)
            cmap (str)
            title (str)
        """
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
        """
        Side-by-side comparison of two Image objects.
        """
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
    def abs(image_data: Image, out: Image):
        img = image_data.pixel_data.float()

        out.pixel_data = torch.abs(img)
        out.width = image_data.width
        out.height = image_data.height

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
        # V[:, None, :] → (c, 1, d);  Z[None, :, :] → (1, N, d)
        diff = V.unsqueeze(1) - Z.unsqueeze(0)  # (c, N, d)
        return (diff ** 2).sum(dim=2)  # (c, N)

    @staticmethod
    def normalize_12bit(image_data: Image, out: Image):
        """
        Normalize a 12-bit DICOM image to [0, 1] by dividing by 4095.

        Args:
            image_data: Input Image with raw 12-bit pixel values.
            out: Output Image to store the normalized result.
        """
        out.pixel_data = torch.clamp(image_data.pixel_data.float() / 4095.0, 0.0, 1.0)
        out.width = image_data.width
        out.height = image_data.height
