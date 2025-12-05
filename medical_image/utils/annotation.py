from enum import Enum, auto

from medical_image.utils.ErrorHandler import ErrorMessages


class GeometryType(Enum):
    BOUNDING_BOX = auto()
    POLYGON = auto()
    MASK = auto()


from typing import List, Union, Tuple
import numpy as np


# TODO: check Dicom Annotation


from enum import Enum, auto
from typing import List, Tuple, Union, Optional
import numpy as np


class GeometryType(Enum):
    BOUNDING_BOX = auto()
    POLYGON = auto()
    MASK = auto()
    POINT = auto()


class Annotation:
    """
    A simple, general, and extensible annotation class for medical imaging.
    """

    def __init__(
        self,
        geometry_type: GeometryType,
        coordinates: Union[
            List[int],
            List[Tuple[int, int]],
            np.ndarray,
        ],
        label: str,
        pathology: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Args:
            geometry_type (GeometryType): Type of geometric shape.
            coordinates: The geometric data (bbox, polygon, mask, point)
            label (str): Annotation label (e.g., mass, calcification, nodule)
            pathology (str, optional): Status (e.g., benign, malignant).
            metadata (dict, optional): Extra info (shape, margins, BI-RADS…)
        """
        self.geometry_type = geometry_type
        self.coordinates = coordinates
        self.label = label
        self.pathology = pathology
        self.metadata = metadata or {}

        self._validate()

    def _validate(self):
        """Light validation to keep the class simple and safe."""

        if self.geometry_type == GeometryType.BOUNDING_BOX:
            if not (isinstance(self.coordinates, list) and len(self.coordinates) == 4):
                raise ValueError("Bounding box must be [x_min, y_min, x_max, y_max]")

        elif self.geometry_type == GeometryType.POLYGON:
            if not (isinstance(self.coordinates, list) and len(self.coordinates) >= 3):
                raise ValueError("Polygon must be a list of (x, y) points")

        elif self.geometry_type == GeometryType.MASK:
            if not isinstance(self.coordinates, np.ndarray):
                raise ValueError("Mask must be a NumPy 2D array")

    def __repr__(self):
        return (
            f"Annotation(label={self.label}, geometry={self.geometry_type.name}, "
            f"pathology={self.pathology}, metadata={self.metadata})"
        )
