import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, List

from torch.utils.data import Dataset

from medical_image.data.dicom_image import DicomImage
from medical_image.data.image import Image
from medical_image.data.region_of_interest import RegionOfInterest
from medical_image.utils.ErrorHandler import ErrorMessages
from medical_image.utils.annotation import GeometryType


class MedicalDataset(Dataset, ABC):
    """
    Abstract base class for medical image datasets using your Image abstraction.
    Supports bounding box or mask labels.
    """

    def __init__(
        self,
        base_path: str,
        file_format: str = ".dcm",
        transform: Optional[Callable] = None,
        train: bool = True,
        test: bool = False,
    ):
        self.base_path = base_path
        self.file_format = file_format.lower()
        self.transform = transform
        self.images_path: List[str] = None
        self.image_labels = None
        self.current_image = None
        self.train = train
        self.test = test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.images_path is None:
            raise ErrorMessages.empty_dataset()
        image_path = self.images_path[idx]
        if self.file_format == ".dcm":
            self.current_image = DicomImage(image_path)
        # FIXME: the load should be done in externel way to load all images or check with load batch
        #       The load should be external and the image should be already loaded (first load 10 then remove then reload the next 10 (add method destroy)
        # image_obj.load()  # Load pixel data into image_obj

        pixel_data = self.current_image.pixel_data
        label = self.current_image.label

        # Load label if defined
        if label.annotation_type in {
            GeometryType.BOUNDING_BOX,
            GeometryType.POLYGON,
            GeometryType.MASK,
        }:
            # tODO: Stopped Here continue
            filename = os.path.basename(image_path)

            # Coordinate source: either dict (for bbox/polygon) or mask directory path (for mask)
            if self.label_type in {"bbox", "polygon"}:
                if self.label_data and filename in self.label_data:
                    coordinates = self.label_data[filename]
                    roi = RegionOfInterest(image_obj, coordinates)
                    cropped_image = roi.load()
                    pixel_data = cropped_image.pixel_data
                    label = (
                        roi.coordinates
                    )  # Optionally, return the raw coordinates as label

            elif self.label_type == "mask":
                mask_image = self.load_mask(image_path)  # Should return Image subclass
                mask_image.load()
                roi = RegionOfInterest(image_obj, mask_image.pixel_data)
                cropped_image = roi.load()
                pixel_data = cropped_image.pixel_data
                label = mask_image.pixel_data  # Full original mask (optional)
        else:
            raise ErrorMessages.annotation_type_not_recognized(label.annotation_type)
        # Apply transform
        if self.transform:
            self.apply_transform(self.transform, pixel_data, label)

        return (pixel_data, label) if label is not None else pixel_data

    @abstractmethod
    def load_batch(self, batch_size) -> Image:
        """
        Should return an instance of Image (e.g., DicomImage).
        """
        pass

    @abstractmethod
    def destroy_batch(self) -> Image:
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
            raise ValueError(
                "label_data must be a directory path when label_type is 'mask'"
            )

    @abstractmethod
    def apply_transform(self, transform, pixel_data, label):
        pass
