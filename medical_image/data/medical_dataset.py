import os
from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class MedicalDataset(Dataset, ABC):
    """
    Abstract base class for medical image datasets supporting bounding boxes and masks.
    """

    def __init__(
            self,
            base_path,
            file_format="dcm",
            transform=None,
            label_type=None,  # "bbox", "mask", or None
            label_data=None,  # dict or base path for masks
            train=True,
            test=False
    ):
        """
        Args:
            base_path (str): Directory with medical images.
            file_format (str): Image file extension (e.g., 'dcm', 'png').
            transform (Optional[Callable]): Image transform.
            label_type (Optional[str]): 'bbox', 'mask', or None.
            label_data (dict or str): 
                - If 'bbox', dict mapping filenames to bbox coords.
                - If 'mask', path to corresponding mask files.
            train (bool): Training flag.
            test (bool): Testing flag.
        """
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
        image = self.load_image(image_path)

        label = None
        if self.label_type == "bbox":
            filename = os.path.basename(image_path)
            label = self.label_data.get(filename, None)

        elif self.label_type == "mask":
            label = self.load_mask(image_path)

        if self.transform:
            image = self.transform(image)
            if label is not None and self.label_type == "mask":
                label = self.transform(label)  # Optionally transform mask too

        return (image, label) if label is not None else image

    @abstractmethod
    def load_image(self, path):
        pass

    def load_mask(self, image_path):
        """
        Load a segmentation mask corresponding to the image.
        By default, assumes same filename in label_data dir.
        Override if needed.
        """
        if isinstance(self.label_data, str):
            filename = os.path.basename(image_path)
            mask_path = os.path.join(self.label_data, filename)
            return self.load_image(mask_path)  # Reuse image loader
        else:
            raise ValueError("label_data must be a directory path when label_type is 'mask'")
