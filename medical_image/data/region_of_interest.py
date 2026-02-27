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

    @classmethod
    def from_center(
        cls,
        image: Image,
        cx: int,
        cy: int,
        half_size: int,
    ) -> "RegionOfInterest":
        """
        Create a bounding-box ROI from center coordinates and half-size.

        The bounding box is [cx - half_size, cy - half_size,
                             cx + half_size + 1, cy + half_size + 1],
        clamped to image dimensions.

        Args:
            image: Source Image.
            cx: Center row (y-axis in image space).
            cy: Center column (x-axis in image space).
            half_size: Half-size of the square ROI.

        Returns:
            RegionOfInterest with bounding box coordinates.
        """
        if image.pixel_data is None:
            image.load()

        H, W = image.pixel_data.shape[:2]

        x_min = max(0, cy - half_size)
        y_min = max(0, cx - half_size)
        x_max = min(W, cy + half_size + 1)
        y_max = min(H, cx + half_size + 1)

        return cls(image=image, coordinates=[x_min, y_min, x_max, y_max])

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

    @staticmethod
    def normalize(image: Image, divisor: float = 4095.0) -> Image:
        """
        Normalize pixel values by dividing by a constant (e.g. 4095 for 12-bit).

        Modifies the image in-place and returns it.

        Args:
            image: Image to normalize.
            divisor: Value to divide by (default: 4095.0 for 12-bit DICOM).

        Returns:
            The same Image with normalized pixel_data.
        """
        image.pixel_data = torch.clamp(image.pixel_data.float() / divisor, 0.0, 1.0)
        return image
