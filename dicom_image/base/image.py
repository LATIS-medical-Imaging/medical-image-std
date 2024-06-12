import os
from abc import ABC, abstractmethod

import pydicom

from log_manager import logger


class Image(ABC):
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"unable to locate file_path at {file_path}")
        self.file_path = file_path
        self.width = None
        self.height = None
        self.pixel_data = None

    @abstractmethod
    def load(self):
        """Abstract method to load image data. Must be implemented by subclasses."""
        pass

    def display_info(self):
        """Display basic information about the image."""
        logger.info(f"File Path: {self.file_path}")
        logger.info(f"Width: {self.width}")
        logger.info(f"Height: {self.height}")


class DicomImage(Image):
    def __init__(self, file_path):
        if  os.path.splitext(self.file_path)[1].lower() != ".dcm":
            raise RuntimeError("Only .dcm-files are supported")

        super().__init__(file_path)
        self.dicom_data = None

    def load(self):
        self.dicom_data = pydicom.dcmread(self.file_path)
        self.pixel_data = self.dicom_data.pixel_array
        self.width = self.dicom_data.Columns
        self.height = self.dicom_data.Rows

    def display_info(self):
        """Method to display DICOM image information."""
        super().display_info()
        logger.info(f"Patient ID: {self.patient_id}")
        logger.info(f"Patient Name: {self.patient_name}")
        logger.info(f"Study ID: {self.study_id}")
        logger.info(f"Series ID: {self.series_id}")
        logger.info(f"Modality: {self.modality}")
