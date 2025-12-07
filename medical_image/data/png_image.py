import os
from PIL import Image as PILImage
import torch
import numpy as np

from medical_image.data.image import Image
from medical_image.utils.ErrorHandler import ErrorMessages


class PNGImage(Image):
    """
    PNG image implementation compatible with the Image base class.
    Supports lazy loading and saving using Pillow.
    """

    def __init__(self, file_path):
        super().__init__(file_path)
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in [".png"]:
            raise ErrorMessages.unsupported_file_type(ext)

        self._pil_image = None  # Lazy-loaded PIL object

    def load(self):
        """
        Load PNG pixel data using Pillow.
        Converts result to a torch.Tensor.
        """
        self._pil_image = PILImage.open(self.file_path)

        # Convert to grayscale or keep RGB depending on mode
        img = np.array(self._pil_image)

        # Convert to torch tensor
        self.pixel_data = torch.from_numpy(img).float()

        # Set dimensions
        if self.pixel_data.ndim == 2:  # Grayscale
            h, w = self.pixel_data.shape
        else:  # RGB or RGBA (H, W, C)
            h, w = self.pixel_data.shape[:2]

        self.height = h
        self.width = w

    def save(self):
        """
        Save PNG image to <original>_modified.png.
        """
        if self.pixel_data is None:
            raise ErrorMessages.pixel_data_not_loaded()

        # Convert tensor to numpy
        img_np = self.pixel_data.detach().cpu().numpy()

        # Ensure dtype is uint8 for PNG
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # Construct output filename
        filename, _ = os.path.splitext(self.file_path)
        out_path = filename + "_modified.png"

        # Save using PIL
        PILImage.fromarray(img_np).save(out_path)
