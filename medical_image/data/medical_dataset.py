import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Union, Callable, Dict, Optional
from medical_image.data.image import RegionOfInterest, Image


class MedicalDataset(Dataset, ABC):
    """
    Abstract base class for medical image datasets using your Image abstraction.
    Supports bounding box or mask labels.
    """

    def __init__(
        self,
        base_path: str,
        file_format: str = "dcm",
        transform: Optional[Callable] = None,
        label_type: Optional[str] = None,  # "bbox", "mask", or None
        label_data: Optional[Union[Dict[str, Union[list, "np.ndarray"]], str]] = None,
        train: bool = True,
        test: bool = False,
    ):
        self.base_path = base_path
        self.file_format = file_format.lower()
        self.transform = transform
        self.label_type = label_type
        self.label_data = label_data
        self.train = train
        self.test = test

        self.image_paths = [
            os.path.join(base_path, f)
            for f in os.listdir(base_path)
            if f.lower().endswith(self.file_format)
        ]
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_obj = self.load_image(image_path)  # Returns subclass of Image
        image_obj.load()  # Load pixel data into image_obj

        pixel_data = image_obj.pixel_data
        label = None

        # Load label if defined
        if self.label_type in {"bbox", "polygon", "mask"}:
            filename = os.path.basename(image_path)

            # Coordinate source: either dict (for bbox/polygon) or mask directory path (for mask)
            if self.label_type in {"bbox", "polygon"}:
                if self.label_data and filename in self.label_data:
                    coordinates = self.label_data[filename]
                    roi = RegionOfInterest(image_obj, coordinates)
                    cropped_image = roi.load()
                    pixel_data = cropped_image.pixel_data
                    label = roi.coordinates  # Optionally, return the raw coordinates as label

            elif self.label_type == "mask":
                mask_image = self.load_mask(image_path)  # Should return Image subclass
                mask_image.load()
                roi = RegionOfInterest(image_obj, mask_image.pixel_data)
                cropped_image = roi.load()
                pixel_data = cropped_image.pixel_data
                label = mask_image.pixel_data  # Full original mask (optional)

        # Apply transform
        if self.transform:
            pixel_data = self.transform(pixel_data)
            if label is not None and isinstance(label, np.ndarray):  # e.g., mask
                label = self.transform(label)

        return (pixel_data, label) if label is not None else pixel_data

    @abstractmethod
    def load_image(self, path: str) -> Image:
        """
        Should return an instance of Image (e.g., DicomImage).
        """
        pass

    def load_mask(self, image_path: str) -> Image:
        """
        Loads a corresponding mask using the same file name from mask directory.
        """
        if isinstance(self.label_data, str):
            filename = os.path.basename(image_path)
            mask_path = os.path.join(self.label_data, filename)
            mask = self.load_image(mask_path)
            mask.load()
            return mask
        else:
            raise ValueError("label_data must be a directory path when label_type is 'mask'")
