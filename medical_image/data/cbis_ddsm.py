import os
from typing import Optional, Callable

from torch.utils.data import Dataset


# TODO: implemnt it with Tree check, download the required csv for train test, and Label, match with the required image
class CbisDDsmDataset(Dataset):
    def __init__(
            self,
            base_path: str,
            file_format: str = ".dcm",
            transform: Optional[Callable] = None,
            train: bool = True,
            test: bool = False,
            task: str = "calcification"
    ):
        self.base_path = base_path
        self.file_format = file_format
        self.transform = transform
        self.train = train
        self.test = test
        self.images_path = os.path.join(base_path, '"CBIS-DDSM"')
        if self.train and task == "calcification":
            self.image_labels_path = "data/calc_case_description_train_set.csv"
        elif self.test and task == "calcification":
            self.image_labels_path = "data/calc_case_description_test_set.csv"
        elif self.train and task == "mass":
            self.image_labels_path = "data/mass_case_description_train_set.csv"
        else:
            self.image_labels_path = "data/mass_case_description_test_set.csv"
        # TODO: complete from here