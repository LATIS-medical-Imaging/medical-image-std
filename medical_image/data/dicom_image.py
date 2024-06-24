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

    def to_png(self, output_path):
        """Convert DICOM image to PNG format."""
        if self.pixel_data is None:
            raise RuntimeError("No pixel data loaded. Cannot convert to PNG.")

        try:
            # Convert pixel data to PIL Image format
            img = PILImage.fromarray(self.pixel_data)

            # Save as PNG
            png_path = os.path.join(output_path, "converted_image.png")
            img.save(png_path)
            logger.info(f"Converted image saved to {png_path}")

        except Exception as e:
            logger.error(f"Error converting image to PNG: {e}")

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
