from enum import Enum, auto

from medical_image.utils.ErrorHandler import ErrorMessages


class AnnotationType(Enum):
    BOUNDING_BOX = auto()
    POLYGON = auto()
    MASK = auto()


from typing import List, Union, Tuple
import numpy as np


# TODO: move to Torch, and  check Dicom Annotation


class Annotation:
    """
    Represents an annotation for an image, which could correspond to different
    abnormalities such as calcifications or masses. The class stores information
    about the annotation type, coordinates, associated classes, and pathology information.
    """

    def __init__(
        self,
        annotation_type: str,  # Assuming AnnotationType is a string enum or constant
        coordinates: List[Union[List[int], List[Tuple[int, int]], np.ndarray]],
        classes: List[str],
        image_view: str,
        abnormality_type: str,
        pathology: str,
        calcification_type: str = None,  # Optional, only for calcifications
        calcification_distribution: str = None,  # Optional, only for calcifications
        mass_shape: str = None,  # Optional, only for masses
        mass_margin: str = None,  # Optional, only for masses
    ):
        """
        Initializes an Annotation object with all necessary attributes.

        Args:
            annotation_type (str): The type of annotation (e.g., "calcification" or "mass").
            coordinates (List[Union[List[int], List[Tuple[int, int]], np.ndarray]]):
                List of coordinates for the annotation (varies by annotation type).
            classes (List[str]): List of class labels corresponding to the coordinates.
            image_view (str): The view or perspective from which the image was taken (e.g., "front", "side").
            abnormality_type (str): The type of abnormality (e.g., "calcification", "mass").
            pathology (str): Pathology associated with the annotation (e.g., "benign", "malignant").
            calcification_type (str, optional): The type of calcification (if abnormality_type is "calcification").
            calcification_distribution (str, optional): Distribution of calcification (if abnormality_type is "calcification").
            mass_shape (str, optional): The shape of the mass (if abnormality_type is "mass").
            mass_margin (str, optional): The margin of the mass (if abnormality_type is "mass").

        Raises:
            AssertionError: If lengths of coordinates and classes do not match.
        """

        self.annotation_type = annotation_type
        self.image_view = image_view
        self.abnormality_type = abnormality_type
        self.pathology = pathology

        # Validate coordinates and classes match in length
        # if len(coordinates) != len(classes):
        #     raise ValueError(
        #         f"Length mismatch: Coordinates ({len(coordinates)}) and classes ({len(classes)}) must have the same length.")

        self.coordinates = coordinates
        self.classes = classes

        # Conditional handling of abnormality type
        if self.abnormality_type == "calcification":
            self.calcification_type = calcification_type
            self.calcification_distribution = calcification_distribution
            if not self.calcification_type or not self.calcification_distribution:
                raise ErrorMessages.input_none(
                    "`calcification_type` and `calcification_distribution`"
                )
        elif self.abnormality_type == "mass":
            self.mass_shape = mass_shape
            self.mass_margin = mass_margin
            if not self.mass_shape or not self.mass_margin:
                raise ErrorMessages.input_none("`mass_shape` and `mass_margin`")
        else:
            raise ValueError(
                f"Unknown abnormality type: {self.abnormality_type}. Supported types are 'calcification' and 'mass'."
            )

    def __repr__(self):
        return f"Annotation(annotation_type={self.annotation_type}, abnormality_type={self.abnormality_type}, coordinates={self.coordinates}, classes={self.classes})"
