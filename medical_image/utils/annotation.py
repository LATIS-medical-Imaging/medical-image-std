import math
from enum import Enum, auto
from typing import List, Tuple, Union, Optional

import numpy as np

from medical_image.utils.ErrorHandler import ErrorMessages


class GeometryType(Enum):
    RECTANGLE = auto()    # [x_min, y_min, x_max, y_max]
    ELLIPSE = auto()      # [cx, cy, rx, ry]
    POLYGON = auto()      # [(x1, y1), (x2, y2), ...]
    MASK = auto()
    POINT = auto()

    # Backward-compatible alias
    BOUNDING_BOX = RECTANGLE


class Annotation:
    """
    A simple, general, and extensible annotation class for medical imaging.
    """

    def __init__(
        self,
        shape: GeometryType,
        coordinates: Union[
            List[int],
            List[Tuple[int, int]],
            np.ndarray,
        ],
        label: str,
        metadata: Optional[dict] = None,
    ):
        """
        Args:
            shape (GeometryType): Type of geometric shape.
            coordinates: The geometric data (bbox, polygon, mask, point)
            label (str): Annotation label (e.g., mass, calcification, nodule)
            metadata (dict, optional): Extra info (shape, margins, BI-RADS...)
        """
        self.shape = shape
        self.coordinates = coordinates
        self.label = label
        self.metadata = metadata or {}

        self._validate()
        self.center: Tuple[float, float] = self._compute_center()

    def _validate(self):
        """Light validation to keep the class simple and safe."""

        if self.shape == GeometryType.RECTANGLE:
            if not (isinstance(self.coordinates, list) and len(self.coordinates) == 4):
                raise ValueError("Rectangle must be [x_min, y_min, x_max, y_max]")

        elif self.shape == GeometryType.ELLIPSE:
            if not (isinstance(self.coordinates, list) and len(self.coordinates) == 4):
                raise ValueError("Ellipse must be [cx, cy, rx, ry]")

        elif self.shape == GeometryType.POLYGON:
            if not (isinstance(self.coordinates, list) and len(self.coordinates) >= 3):
                raise ValueError("Polygon must be a list of >= 3 (x, y) points")

        elif self.shape == GeometryType.MASK:
            if not isinstance(self.coordinates, np.ndarray):
                raise ValueError("Mask must be a NumPy 2D array")

    def _compute_center(self) -> Tuple[float, float]:
        """Compute the centroid of the annotation geometry."""
        if self.shape == GeometryType.RECTANGLE:
            x_min, y_min, x_max, y_max = self.coordinates
            return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

        elif self.shape == GeometryType.ELLIPSE:
            cx, cy, rx, ry = self.coordinates
            return (float(cx), float(cy))

        elif self.shape == GeometryType.POLYGON:
            xs = [p[0] for p in self.coordinates]
            ys = [p[1] for p in self.coordinates]
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        return (0.0, 0.0)

    def get_bounding_box(self) -> List[int]:
        """Return [x_min, y_min, x_max, y_max] bounding box of the annotation."""
        if self.shape == GeometryType.RECTANGLE:
            return list(self.coordinates)

        elif self.shape == GeometryType.ELLIPSE:
            cx, cy, rx, ry = self.coordinates
            return [int(cx - rx), int(cy - ry), int(cx + rx), int(cy + ry)]

        elif self.shape == GeometryType.POLYGON:
            xs = [p[0] for p in self.coordinates]
            ys = [p[1] for p in self.coordinates]
            return [min(xs), min(ys), max(xs), max(ys)]

        raise ValueError(f"Unsupported shape: {self.shape}")

    def get_roi(
        self,
        padding: int = 0,
        roi_type: str = "bbox",
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> dict:
        """
        Get a region of interest around the annotation.

        Args:
            padding: Extra pixels to add around the bounding box on each side.
            roi_type: Shape of the ROI output:
                - "bbox" / "rectangle": [x_min, y_min, x_max, y_max] with padding
                - "ellipse": {"center": (cx, cy), "radii": (rx, ry)} with padding
            image_shape: (height, width) to clamp ROI within image bounds.

        Returns:
            dict with keys "type" and "coordinates".
        """
        bbox = self.get_bounding_box()
        x_min, y_min, x_max, y_max = bbox

        x_min -= padding
        y_min -= padding
        x_max += padding
        y_max += padding

        if image_shape is not None:
            h, w = image_shape
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

        if roi_type in ("bbox", "rectangle"):
            return {
                "type": roi_type,
                "coordinates": [x_min, y_min, x_max, y_max],
            }
        elif roi_type == "ellipse":
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            rx = (x_max - x_min) / 2.0
            ry = (y_max - y_min) / 2.0
            return {
                "type": "ellipse",
                "coordinates": {"center": (cx, cy), "radii": (rx, ry)},
            }
        else:
            raise ValueError(
                f"Unknown roi_type: {roi_type}. Use 'bbox', 'rectangle', or 'ellipse'"
            )

    def to_dict(self) -> dict:
        """Serialize annotation to a JSON-compatible dict."""
        coords = self.coordinates
        if self.shape == GeometryType.POLYGON:
            coords = [list(p) for p in self.coordinates]
        return {
            "shape": self.shape.name,
            "coordinates": coords,
            "label": self.label,
            "center": list(self.center),
            "bounding_box": self.get_bounding_box(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Annotation":
        """Deserialize annotation from a dict."""
        shape = GeometryType[data["shape"]]
        coordinates = data["coordinates"]
        if shape == GeometryType.POLYGON:
            coordinates = [tuple(p) for p in coordinates]
        return cls(
            shape=shape,
            coordinates=coordinates,
            label=data["label"],
            metadata=data.get("metadata", {}),
        )

    def __repr__(self):
        return (
            f"Annotation(label={self.label}, geometry={self.shape.name}, "
            f"center={self.center}, metadata={self.metadata})"
        )