import os

import pydicom
import torch

from medical_image.data.image import Image
from medical_image.utils.ErrorHandler import ErrorMessages


class DicomImage(Image):
    def __init__(self, file_path):
        super().__init__(file_path)
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext != ".dcm":
            raise ErrorMessages.unsupported_file_type(ext)
        self.dicom_data = None

    def load(self):
        self.dicom_data = pydicom.dcmread(self.file_path)
        self.pixel_data = torch.Tensor(self.dicom_data.pixel_array, device=self.device)
        self.width = self.dicom_data.Columns
        self.height = self.dicom_data.Rows

    def save(self):
        if self.dicom_data is None:
            raise ErrorMessages.dicom_data_not_loaded()
        filename, extension = os.path.splitext(self.file_path)
        # Update DICOM data pixel array
        self.dicom_data.pixel_array = self.to_numpy().tobytes()

        # Save DICOM data back to file
        self.dicom_data.save_as(filename + "_modified.dcm")
