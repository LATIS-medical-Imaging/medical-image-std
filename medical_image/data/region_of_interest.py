import copy
from typing import Union, List, Tuple

import numpy as np
import torch
from skimage.draw import polygon

from medical_image.data.image import Image
from medical_image.utils.annotation import GeometryType


class RegionOfInterest:
    """
    PyTorch-compatible Region of Interest (ROI) extractor.
    Coordinates remain NumPy for all ROI types.

    Supports:
        - Bounding Box: [x_min, y_min, x_max, y_max]
        - Polygon: [(x1, y1), ..., (xn, yn)]
        - Mask: 2D NumPy array
    """

    def __init__(
        self,
        image: Image,
        coordinates: Union[List[int], List[Tuple[int, int]], np.ndarray],
    ):
        self.image = image
        self.coordinates = coordinates  # MUST remain numpy
        self.annotation_type = self._determine_annotation_type()

    # -------------------------------------------------------------------------
    # Determine annotation type
    # -------------------------------------------------------------------------
    def _determine_annotation_type(self) -> GeometryType:
        if isinstance(self.coordinates, list):
            # Bounding Box
            if len(self.coordinates) == 4 and all(
                isinstance(c, int) for c in self.coordinates
            ):
                return GeometryType.BOUNDING_BOX

            # Polygon
            if all(isinstance(pt, tuple) and len(pt) == 2 for pt in self.coordinates):
                return GeometryType.POLYGON

        # Mask
        if isinstance(self.coordinates, np.ndarray):
            return GeometryType.MASK

        raise ValueError("Unsupported ROI coordinates format.")

    # -------------------------------------------------------------------------
    # Load ROI from image
    # -------------------------------------------------------------------------
    def load(self) -> Image:
        """
        Crop the image using the ROI definition and return a new Image object.
        Pixel data will be a torch.Tensor.
        """
        # Ensure pixel data is loaded and torch.Tensor
        if self.image.pixel_data is None:
            self.image.load()

        pixel_t: torch.Tensor = self.image.pixel_data

        # Convert to numpy temporarily for ROI operations requiring numpy
        pixel_np = pixel_t.cpu().numpy()

        # =====================================================================
        # BOUNDING BOX
        # =====================================================================
        if self.annotation_type == GeometryType.BOUNDING_BOX:
            x_min, y_min, x_max, y_max = self.coordinates
            cropped_np = pixel_np[y_min:y_max, x_min:x_max]

        # =====================================================================
        # POLYGON
        # =====================================================================
        elif self.annotation_type == GeometryType.POLYGON:
            mask = np.zeros(pixel_np.shape[:2], dtype=bool)

            poly_y, poly_x = zip(*self.coordinates)
            rr, cc = polygon(poly_y, poly_x)
            mask[rr, cc] = True

            cropped_np = pixel_np * mask

            # Auto-crop to bounding box of polygon
            ys, xs = np.nonzero(mask)
            y_min, y_max = ys.min(), ys.max() + 1
            x_min, x_max = xs.min(), xs.max() + 1

            cropped_np = cropped_np[y_min:y_max, x_min:x_max]

        # =====================================================================
        # MASK
        # =====================================================================
        elif self.annotation_type == GeometryType.MASK:
            mask = self.coordinates.astype(bool)

            cropped_np = pixel_np * mask

            ys, xs = np.nonzero(mask)
            y_min, y_max = ys.min(), ys.max() + 1
            x_min, x_max = xs.min(), xs.max() + 1

            cropped_np = cropped_np[y_min:y_max, x_min:x_max]

        else:
            raise RuntimeError("Unknown ROI annotation type.")

        # Convert NumPy -> torch
        cropped_tensor = torch.from_numpy(cropped_np).float()

        # Build new Image object (deep copy metadata)
        cropped_image = copy.deepcopy(self.image)
        cropped_image.pixel_data = cropped_tensor
        cropped_image.height, cropped_image.width = cropped_tensor.shape[:2]

        return cropped_image
