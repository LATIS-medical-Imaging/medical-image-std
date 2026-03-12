import os

import numpy as np
import torch
from PIL import Image as PILImage

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

        self._pil_image = None

    def load(self):
        self._pil_image = PILImage.open(self.file_path)
        img = np.array(self._pil_image)
        self.pixel_data = torch.from_numpy(img).float()

    def save(self):
        if self.pixel_data is None:
            raise ErrorMessages.dicom_data_not_loaded()

        img_np = self.pixel_data.detach().cpu().numpy()
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        filename, _ = os.path.splitext(self.file_path)
        out_path = filename + "_modified.png"
        PILImage.fromarray(img_np).save(out_path)
