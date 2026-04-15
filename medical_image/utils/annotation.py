import math
from enum import Enum, auto
from typing import List, Tuple, Union, Optional

import numpy as np

from medical_image.utils.ErrorHandler import ErrorMessages


class GeometryType(Enum):
    """Supported geometric shapes for annotations.

    Each member defines the coordinate format expected by :class:`Annotation`:

    * ``RECTANGLE``  -- ``[x_min, y_min, x_max, y_max]``
    * ``ELLIPSE``    -- ``[cx, cy, rx, ry]`` (center + radii)
    * ``POLYGON``    -- ``[(x1, y1), (x2, y2), ...]`` (>= 3 vertices)
    * ``BOUNDING_BOX`` -- backward-compatible alias for ``RECTANGLE``
    """

    RECTANGLE = auto()  # [x_min, y_min, x_max, y_max]
    ELLIPSE = auto()  # [cx, cy, rx, ry]
    POLYGON = auto()  # [(x1, y1), (x2, y2), ...]

    # Backward-compatible alias
    BOUNDING_BOX = RECTANGLE


class Annotation:
    """A single annotation on a medical image.

    Represents a geometric region (rectangle, ellipse, or polygon) with a
    label and optional metadata.  The centroid is computed automatically in
    the constructor and exposed as :pyattr:`center`.

    Attributes:
        shape (GeometryType): Geometry type of the annotation.
        coordinates: Shape-specific coordinate data.
        label (str): Human-readable annotation label.
        metadata (dict): Arbitrary extra information (BI-RADS, pathology, …).
        center (Tuple[float, float]): Computed centroid ``(cx, cy)``.
    """

    def __init__(
        self,
        shape: GeometryType,
        coordinates: Union[
            List[int],
            List[Tuple[int, int]],
        ],
        label: str,
        metadata: Optional[dict] = None,
    ):
        """Initialise an annotation and compute its center.

        Args:
            shape: Geometry type (``RECTANGLE``, ``ELLIPSE``, or ``POLYGON``).
            coordinates: Coordinate data whose format depends on *shape*:
                - RECTANGLE: ``[x_min, y_min, x_max, y_max]``
                - ELLIPSE:   ``[cx, cy, rx, ry]``
                - POLYGON:   ``[(x1, y1), (x2, y2), ...]`` (>= 3 points)
            label: Annotation label (e.g. ``"mass"``, ``"calcification"``).
            metadata: Optional extra info. Defaults to ``{}``.

        Raises:
            ValueError: If *coordinates* do not match the *shape* contract.
        """
        self.shape = shape
        self.coordinates = coordinates
        self.label = label
        self.metadata = metadata or {}

        self._validate()
        self.center: Tuple[float, float] = self._compute_center()

    def _validate(self):
        """Validate that *coordinates* match the declared *shape*.

        Raises:
            ValueError: If the coordinate format is invalid for the shape.
        """

        if self.shape == GeometryType.RECTANGLE:
            if not (isinstance(self.coordinates, list) and len(self.coordinates) == 4):
                raise ValueError("Rectangle must be [x_min, y_min, x_max, y_max]")

        elif self.shape == GeometryType.ELLIPSE:
            if not (isinstance(self.coordinates, list) and len(self.coordinates) == 4):
                raise ValueError("Ellipse must be [cx, cy, rx, ry]")

        elif self.shape == GeometryType.POLYGON:
            if not (isinstance(self.coordinates, list) and len(self.coordinates) >= 3):
                raise ValueError("Polygon must be a list of >= 3 (x, y) points")

    def _compute_center(self) -> Tuple[float, float]:
        """Compute the centroid of the annotation geometry.

        Returns:
            Tuple of ``(cx, cy)`` floats.  For ``RECTANGLE`` this is the
            midpoint; for ``ELLIPSE`` it is the center directly; for
            ``POLYGON`` it is the arithmetic mean of all vertices.
        """
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
        """Return the axis-aligned bounding box enclosing the annotation.

        Returns:
            ``[x_min, y_min, x_max, y_max]`` in pixel coordinates.

        Raises:
            ValueError: For unsupported geometry types.
        """
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
        """Return a region of interest around the annotation.

        Computes the bounding box, applies *padding*, optionally clamps to
        image bounds, and returns the result in the requested shape.

        Args:
            padding: Extra pixels added on **each** side of the bounding box.
            roi_type: Output shape format:
                - ``"bbox"`` or ``"rectangle"``: returns
                  ``{"type": ..., "coordinates": [x_min, y_min, x_max, y_max]}``
                - ``"ellipse"``: returns
                  ``{"type": "ellipse", "coordinates": {"center": (cx, cy), "radii": (rx, ry)}}``
            image_shape: ``(height, width)`` used to clamp coordinates so the
                ROI stays within image bounds.  ``None`` means no clamping.

        Returns:
            dict with keys ``"type"`` (str) and ``"coordinates"``.

        Raises:
            ValueError: If *roi_type* is not ``"bbox"``, ``"rectangle"``, or
                ``"ellipse"``.
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
        """Serialize the annotation to a JSON-compatible dictionary.

        The output includes computed fields (``center``, ``bounding_box``)
        so that consumers do not need to recompute them.  Polygon coordinates
        are converted from tuples to nested lists for JSON compatibility.

        Returns:
            dict with keys ``shape``, ``coordinates``, ``label``, ``center``,
            ``bounding_box``, and ``metadata``.
        """
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
        """Deserialize an annotation from a dictionary.

        Inverse of :meth:`to_dict`.  The ``center`` and ``bounding_box``
        fields in *data* are ignored (they are recomputed from coordinates).

        Args:
            data: Dictionary with at least ``shape``, ``coordinates``, and
                ``label`` keys.  ``metadata`` is optional (defaults to ``{}``).

        Returns:
            A new :class:`Annotation` instance.
        """
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

    def __repr__(self) -> str:
        """Return a human-readable summary of the annotation."""
        return (
            f"Annotation(label={self.label}, geometry={self.shape.name}, "
            f"center={self.center}, metadata={self.metadata})"
        )
