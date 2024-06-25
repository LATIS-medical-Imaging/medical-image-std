import os

import pydicom

from medical_image.data.image import Image


# TODO: - Implement Plotting method
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

    def save(self):
        if self.dicom_data is None:
            raise RuntimeError(
                "DICOM data has not been loaded yet. Call load() method first."
            )
        filename, extension = os.path.splitext(self.file_path)
        # Update DICOM data pixel array
        self.dicom_data.PixelData = self.pixel_data.tobytes()

        # Save DICOM data back to file
        self.dicom_data.save_as(filename + "_modified.dcm")
