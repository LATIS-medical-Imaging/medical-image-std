from enum import Enum, auto
from typing import Union, List, Tuple

import numpy as np

from medical_image.utils.ErrorHandler import ErrorMessages


class AnnotationType(Enum):
    BOUNDING_BOX = auto()
    POLYGON = auto()
    MASK = auto()


class Annotation:

    def __init__(self, annotation_type: AnnotationType,
                 coordinates: List[Union[List[int], List[Tuple[int, int]], np.ndarray]], classes: List[str]):
        self.annotation_type = annotation_type
        assert len(coordinates) == len(classes), ErrorMessages.length_mismatch(
            len(coordinates), len(classes), "coordinates", "classes"
        )
        self.coordinates = coordinates
        self.classes = classes



