"""
Abstract base dataset for medical image PyTorch dataloaders.

Provides the common interface and behavior shared by all medical image dataset
implementations (INbreast, Custom INbreast, CBIS-DDSM, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List, Tuple, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from medical_image.utils.logging import logger
from medical_image.utils.downloader import download
from medical_image.data.annotation import Annotation, GeometryType


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for PyTorch-compatible medical image datasets.

    Enforces lazy loading: images are never pre-loaded into memory.
    Each sample is loaded on-the-fly in ``__getitem__``.

    Subclasses must implement:
        - ``_build_sample_list()``: scan the dataset directory and build
          an internal list of samples (paths, metadata).
        - ``_load_sample(idx)``: load a single sample by index, returning
          a dict with ``"image"``, ``"mask"`` (or ``"label"``), and ``"metadata"``.

    Args:
        root_dir: Root directory of the dataset.
        transform: Optional callable transform applied to the image tensor.
        target_transform: Optional callable transform applied to the mask/label.
        target_size: Optional (H, W) to resize images and masks on load.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size

        self._samples: List[Any] = []
        self._build_sample_list()

        logger.info(
            f"{self.__class__.__name__}: {len(self._samples)} samples "
            f"from {root_dir}"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_sample_list(self) -> None:
        """
        Scan the dataset directory and populate ``self._samples``.

        Each entry should contain enough information (paths, IDs) to
        lazily load a single sample in ``_load_sample``.
        """
        ...

    @abstractmethod
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """
        Load a single sample by index.

        Must return a dictionary with at least:
            - ``"image"``: ``torch.Tensor`` of shape ``(C, H, W)``
            - ``"mask"``: ``torch.Tensor`` of shape ``(1, H, W)`` or ``"label"``: ``int``
            - ``"metadata"``: ``dict`` with case_id and other info

        The image and mask should NOT have transforms applied yet —
        that is handled by the base class.
        """
        ...

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load sample ``idx`` and apply transforms.

        Returns:
            Dict with ``"image"``, ``"mask"`` (or ``"label"``), and ``"metadata"``.
        """
        sample = self._load_sample(idx)

        image = sample["image"]
        mask = sample.get("mask")
        label = sample.get("label")

        # Optional resize
        if self.target_size is not None:
            image = self._resize(image, self.target_size)
            if mask is not None:
                mask = self._resize(mask, self.target_size, mode="nearest")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None and mask is not None:
            mask = self.target_transform(mask)

        result: Dict[str, Any] = {
            "image": image,
            "metadata": sample.get("metadata", {}),
        }
        if mask is not None:
            result["mask"] = mask
        if label is not None:
            result["label"] = label

        return result

    # ------------------------------------------------------------------
    # Download classmethod
    # ------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        source: str,
        destination: str,
        method: Literal["local", "http", "ftp"] = "local",
        percentage: Optional[float] = None,
    ) -> str:
        """
        Download the dataset from a source.

        Args:
            source: Source path or URL.
            destination: Local destination directory.
            method: ``'local'``, ``'http'``, or ``'ftp'``.
            percentage: Optional subset percentage (0–1] for large datasets.

        Returns:
            Absolute path to the downloaded dataset.
        """
        return download(source, destination, method=method, percentage=percentage)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(
        tensor: torch.Tensor,
        size: Tuple[int, int],
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Resize a tensor to ``(H, W)`` using interpolation.

        Handles both 3D ``(C, H, W)`` and 2D ``(H, W)`` inputs.
        """
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            align = mode != "nearest"
            out = F.interpolate(
                tensor.float(),
                size=size,
                mode=mode,
                align_corners=align if align else None,
            )
            return out.squeeze(0).squeeze(0)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            align = mode != "nearest"
            out = F.interpolate(
                tensor.float(),
                size=size,
                mode=mode,
                align_corners=align if align else None,
            )
            return out.squeeze(0)

        return tensor

    # ------------------------------------------------------------------
    # Annotation support
    # ------------------------------------------------------------------

    def _get_annotations(self, idx: int) -> List[Annotation]:
        """Return annotations for the sample at *idx*.

        Override this in subclasses to provide actual
        :class:`~medical_image.utils.annotation.Annotation` objects
        used by :meth:`to_coco_json`.

        Args:
            idx: Sample index (same as for ``_load_sample``).

        Returns:
            List of annotations.  The default implementation returns ``[]``.
        """
        return []

    @staticmethod
    def _annotation_to_coco_segmentation(ann: Annotation) -> list:
        """Convert an Annotation to COCO segmentation format.

        Args:
            ann: The annotation to convert.

        Returns:
            A nested list ``[[x1, y1, x2, y2, ...]]`` suitable for the
            COCO ``segmentation`` field.  ``POLYGON`` coordinates are
            flattened; ``RECTANGLE`` is expanded to four corners;
            ``ELLIPSE`` is approximated as a 36-point polygon.
            Returns ``[]`` for unsupported shapes.
        """
        if ann.shape == GeometryType.POLYGON:
            flat = []
            for x, y in ann.coordinates:
                flat.extend([x, y])
            return [flat]

        elif ann.shape in (GeometryType.RECTANGLE,):
            x_min, y_min, x_max, y_max = ann.coordinates
            return [[x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]]

        elif ann.shape == GeometryType.ELLIPSE:
            import math

            cx, cy, rx, ry = ann.coordinates
            n_points = 36
            poly = []
            for i in range(n_points):
                theta = 2 * math.pi * i / n_points
                px = cx + rx * math.cos(theta)
                py = cy + ry * math.sin(theta)
                poly.extend([px, py])
            return [poly]

        return []

    # ------------------------------------------------------------------
    # COCO JSON export / import
    # ------------------------------------------------------------------

    def to_coco_json(
        self,
        output_path: Optional[str] = None,
        description: str = "Medical Image Dataset",
    ) -> dict:
        """Export the entire dataset as a COCO-format JSON.

        Iterates over every sample, calls :meth:`_get_annotations` for each,
        and builds the standard COCO structure (``info``, ``images``,
        ``annotations``, ``categories``).

        Each annotation entry includes a custom ``"center"`` field with the
        annotation centroid -- this is an extension to the official COCO spec.

        Note:
            COCO ``bbox`` format is ``[x, y, width, height]``, **not**
            ``[x_min, y_min, x_max, y_max]``.  The conversion is handled
            automatically.

        Args:
            output_path: If provided, the JSON dict is also written to this
                file path.
            description: Free-text description for the COCO ``info`` block.

        Returns:
            The full COCO JSON structure as a ``dict``.
        """
        import json
        from datetime import datetime

        coco: Dict[str, Any] = {
            "info": {
                "description": description,
                "version": "1.0",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": [],
        }

        category_map: Dict[str, int] = {}
        ann_id = 1

        for img_id, idx in enumerate(range(len(self)), start=1):
            sample = self._load_sample(idx)
            metadata = sample.get("metadata", {})
            image_tensor = sample["image"]

            if image_tensor.ndim == 3:
                _, h, w = image_tensor.shape
            else:
                h, w = image_tensor.shape

            coco["images"].append(
                {
                    "id": img_id,
                    "file_name": metadata.get(
                        "file_name", metadata.get("case_id", f"image_{idx}")
                    ),
                    "width": int(w),
                    "height": int(h),
                }
            )

            annotations = self._get_annotations(idx)

            for ann in annotations:
                if ann.label not in category_map:
                    cat_id = len(category_map) + 1
                    category_map[ann.label] = cat_id
                    coco["categories"].append(
                        {
                            "id": cat_id,
                            "name": ann.label,
                            "supercategory": "lesion",
                        }
                    )

                segmentation = self._annotation_to_coco_segmentation(ann)

                bbox = ann.get_bounding_box()
                coco_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                area = coco_bbox[2] * coco_bbox[3]

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": category_map[ann.label],
                        "segmentation": segmentation,
                        "bbox": coco_bbox,
                        "area": float(area),
                        "iscrowd": 0,
                        "center": list(ann.center),
                    }
                )
                ann_id += 1

        if output_path:
            with open(output_path, "w") as f:
                json.dump(coco, f, indent=2)

        return coco

    @classmethod
    def from_coco_json(cls, json_path: str) -> dict:
        """Load dataset metadata and annotations from a COCO JSON file.

        Reconstructs :class:`~medical_image.utils.annotation.Annotation`
        objects from COCO segmentation polygons (preferred) or bounding boxes
        (fallback when segmentation is empty).

        Args:
            json_path: Path to a COCO-format ``.json`` file.

        Returns:
            A dict with three keys:

            * ``"images"``      -- list of COCO image-entry dicts.
            * ``"annotations"`` -- ``Dict[int, List[Annotation]]`` mapping
              each COCO image ID to its reconstructed annotations.
            * ``"categories"``  -- ``Dict[int, str]`` mapping each COCO
              category ID to its label name.
        """
        import json

        with open(json_path, "r") as f:
            coco = json.load(f)

        categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

        annotations_by_image: Dict[int, List[Annotation]] = {}
        for ann_data in coco.get("annotations", []):
            img_id = ann_data["image_id"]

            seg = ann_data.get("segmentation", [[]])
            if seg and seg[0]:
                flat = seg[0]
                coords = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
                shape = GeometryType.POLYGON
            else:
                bbox = ann_data.get("bbox", [0, 0, 0, 0])
                coords = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                shape = GeometryType.RECTANGLE

            label = categories.get(ann_data.get("category_id", -1), "unknown")

            annotation = Annotation(
                shape=shape,
                coordinates=coords,
                label=label,
                metadata={"coco_id": ann_data.get("id")},
            )

            annotations_by_image.setdefault(img_id, []).append(annotation)

        return {
            "images": coco.get("images", []),
            "annotations": annotations_by_image,
            "categories": categories,
        }

    @staticmethod
    def _to_chw(tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is in ``(C, H, W)`` format.

        - ``(H, W)`` → ``(1, H, W)``
        - ``(H, W, C)`` → ``(C, H, W)``
        - ``(C, H, W)`` → unchanged
        """
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[2] in (1, 3, 4):
            return tensor.permute(2, 0, 1)
        return tensor
