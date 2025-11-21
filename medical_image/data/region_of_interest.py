import copy
from typing import Union, List, Tuple

import numpy as np

from medical_image.data.image import Image
from medical_image.utils.annotation import AnnotationType


# TODO:
class RegionOfInterest:
    """
    Represents a Region of Interest (ROI) in a medical image, which may be a bounding box,
    a polygon, or a binary mask.

    Parameters:
        image (Image): The Image object the ROI is based on.
        coordinates (Union[List[int], List[Tuple[int, int]], np.ndarray]): The ROI definition.
            - Bounding box: [x_min, y_min, x_max, y_max]
            - Polygon: [(x1, y1), (x2, y2), ..., (xn, yn)]
            - Mask: 2D NumPy array

    Attributes:
        image (Image): The original image.
        coordinates: Coordinates representing the ROI.
        annotation_type (AnnotationType): The type of ROI (bounding box, polygon, or mask).
    """

    def __init__(
        self,
        image: Image,
        coordinates: Union[List[int], List[Tuple[int, int]], np.ndarray],
    ):
        self.image = image
        self.coordinates = coordinates
        self.annotation_type = self._determine_annotation_type()

    def _determine_annotation_type(self) -> AnnotationType:
        if isinstance(self.coordinates, list):
            if len(self.coordinates) == 4 and all(
                isinstance(c, int) for c in self.coordinates
            ):
                return AnnotationType.BOUNDING_BOX
            elif all(isinstance(c, tuple) and len(c) == 2 for c in self.coordinates):
                return AnnotationType.POLYGON
        elif isinstance(self.coordinates, np.ndarray):
            return AnnotationType.MASK

        raise ValueError(
            "Unsupported or invalid coordinates format for RegionOfInterest"
        )

    def load(self) -> Image:
        """
        Crop the image to the ROI and return a new Image object containing only the ROI.

        Returns:
            Image: A new Image instance cropped to the region of interest.
        """
        pixel_data = self.image.pixel_data

        if self.annotation_type == AnnotationType.BOUNDING_BOX:
            x_min, y_min, x_max, y_max = self.coordinates
            cropped_pixel_data = pixel_data[y_min:y_max, x_min:x_max]

        elif self.annotation_type == AnnotationType.POLYGON:
            from skimage.draw import polygon  # polygon rasterization

            mask = np.zeros_like(pixel_data, dtype=bool)
            poly_y, poly_x = zip(*self.coordinates)
            rr, cc = polygon(poly_y, poly_x)
            mask[rr, cc] = True
            cropped_pixel_data = pixel_data * mask

            y_indices, x_indices = np.nonzero(mask)
            y_min, y_max = y_indices.min(), y_indices.max() + 1
            x_min, x_max = x_indices.min(), x_indices.max() + 1
            cropped_pixel_data = cropped_pixel_data[y_min:y_max, x_min:x_max]

        elif self.annotation_type == AnnotationType.MASK:
            mask = self.coordinates.astype(bool)
            cropped_pixel_data = pixel_data * mask

            y_indices, x_indices = np.nonzero(mask)
            y_min, y_max = y_indices.min(), y_indices.max() + 1
            x_min, x_max = x_indices.min(), x_indices.max() + 1
            cropped_pixel_data = cropped_pixel_data[y_min:y_max, x_min:x_max]

        else:
            raise RuntimeError("Unknown region type.")

        # Create a new Image instance and copy metadata
        cropped_image = copy.deepcopy(self.image)
        cropped_image.pixel_data = cropped_pixel_data
        cropped_image.height, cropped_image.width = cropped_pixel_data.shape[:2]

        return cropped_image
