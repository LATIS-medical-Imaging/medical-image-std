import pydicom


class Image(ABC):
    def __init__(self, file_path):
        if not os.path.exists(img_path):
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
        print(f"File Path: {self.file_path}")
        print(f"Width: {self.width}")
        print(f"Height: {self.height}")


class DicomImage(Image):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.dicom_data = None

    def load(self):
        self.dicom_data = pydicom.dcmread(file_path)
        self.pixel_data = self.dicom_data.pixel_array
        self.width = self.dicom_data.Columns
        self.height = self.dicom_data.Rows

    def display_info(self):
        """Method to display DICOM image information."""
        super().display_info()
        print(f"Patient ID: {self.patient_id}")
        print(f"Patient Name: {self.patient_name}")
        print(f"Study ID: {self.study_id}")
        print(f"Series ID: {self.series_id}")
        print(f"Modality: {self.modality}")
