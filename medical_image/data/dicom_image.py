import os
from typing import Callable

import pydicom
from PIL import Image as PILImage

from log_manager import logger
from medical_image.data.image import Image, T


# TODO: - Implement Plotting method
#       - Implement similar to threshold, apply_filter method
class DicomImage(Image):
    def __init__(self, file_path):

        super().__init__(file_path)
        if os.path.splitext(self.file_path)[1].lower() != ".dcm":
            raise RuntimeError("Only .dcm-files are supported")
        self.dicom_data = None

    def load(self):
        self.dicom_data = pydicom.dcmread(self.file_path)
        self.pixel_data = self.dicom_data.pixel_array
        self.width = self.dicom_data.Columns
        self.height = self.dicom_data.Rows

    def apply_threshold(self, threshold_func: Callable[[T], T]) -> None:
        """Apply a thresholding function to the DICOM image."""
        if self.pixel_data is None:
            raise RuntimeError("No pixel data loaded. Cannot apply threshold.")

        try:
            # Apply the thresholding function to the pixel data
            self.pixel_data = threshold_func(self.pixel_data)
            logger.info("Threshold applied successfully.")

        except Exception as e:
            logger.error(f"Error applying threshold: {e}")
