"""
INbreast mammography dataset.

PyTorch-compatible dataset that loads DICOM mammograms with either:
- COCO-format JSON annotations (polygon segmentation masks), or
- XML annotations converted to binary masks on-the-fly.

Integrates with the existing ``DicomImage`` framework for lazy loading.
"""

import json
import csv
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List

import numpy as np
import torch

from medical_image.utils.logging import logger
from medical_image.data.dicom_image import DicomImage
from medical_image.datasets.base_dataset import BaseDataset


class INbreastDataset(BaseDataset):
    """
    PyTorch Dataset for the INbreast mammography database.

    Supports two directory layouts:

    **Layout A — COCO annotations (preferred):**

    .. code-block:: text

        root_dir/
        ├── annotations.json     # COCO format
        └── images/
            ├── 20586934.dcm
            └── ...

    **Layout B — XML annotations (legacy):**

    .. code-block:: text

        root_dir/
        ├── AllDICOMs/
        ├── AllXML/
        ├── AllROI/       (optional)
        └── INbreast.csv  (optional)

    Each sample returns:

    .. code-block:: python

        {
            "image": Tensor[1, H, W],
            "mask":  Tensor[1, H, W],
            "metadata": {
                "case_id": str,
                "file_name": str,
                "num_annotations": int,
            }
        }

    Args:
        root_dir: Path to the dataset root directory.
        transform: Optional transform for the image tensor.
        target_transform: Optional transform for the mask tensor.
        target_size: Optional (H, W) resize target.
        point_radius: Radius for rendering single-point ROIs (XML mode only).

    Example:

    .. code-block:: python

        dataset = INbreastDataset("data/Inbreast")
        sample = dataset[0]
        print(sample["image"].shape)     # torch.Size([1, 512, 512])
        print(sample["mask"].shape)      # torch.Size([1, 512, 512])
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_size: Optional[Tuple[int, int]] = None,
        point_radius: int = 3,
    ):
        self.point_radius = point_radius
        self._coco: Optional[dict] = None
        self._img_to_anns: Dict[int, List[dict]] = {}
        self._csv_metadata: Dict[str, Dict[str, Any]] = {}
        self._mode: str = "unknown"  # "coco" or "xml"
        super().__init__(root_dir, transform, target_transform, target_size)

    # ------------------------------------------------------------------
    # Build sample list
    # ------------------------------------------------------------------

    def _build_sample_list(self) -> None:
        root = Path(self.root_dir)

        # Detect layout: COCO (annotations.json + images/) or XML (AllDICOMs/ + AllXML/)
        coco_path = root / "annotations.json"
        if coco_path.exists():
            self._build_from_coco(root, coco_path)
        else:
            self._build_from_xml(root)

    def _build_from_coco(self, root: Path, coco_path: Path) -> None:
        """Build sample list from COCO-format annotations."""
        self._mode = "coco"

        with open(coco_path, "r") as f:
            self._coco = json.load(f)

        # Group annotations by image_id
        for ann in self._coco.get("annotations", []):
            self._img_to_anns.setdefault(ann["image_id"], []).append(ann)

        images_dir = root / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"images/ directory not found at {images_dir}")

        # Build samples — only include images that exist on disk
        for img_info in self._coco["images"]:
            dcm_path = images_dir / img_info["file_name"]
            if dcm_path.exists():
                self._samples.append(img_info)
            else:
                logger.warning(f"DICOM not found, skipping: {dcm_path}")

        logger.info(
            f"INbreastDataset (COCO): {len(self._samples)} samples, "
            f"{len(self._coco.get('annotations', []))} annotations"
        )

    def _build_from_xml(self, root: Path) -> None:
        """Build sample list from XML annotations (legacy layout)."""
        self._mode = "xml"

        from medical_image.utils.pairing import pair_inbreast, INbreastSample

        dicoms_dir = root / "AllDICOMs"
        xml_dir = root / "AllXML"
        roi_dir = root / "AllROI"

        if not dicoms_dir.exists():
            raise FileNotFoundError(
                f"Neither annotations.json nor AllDICOMs/ found at {root}"
            )

        self._csv_metadata = self._parse_csv(root)
        self._samples = pair_inbreast(
            str(dicoms_dir),
            str(xml_dir),
            str(roi_dir) if roi_dir.exists() else None,
        )

        logger.info(
            f"INbreastDataset (XML): {len(self._samples)} samples, "
            f"{len(self._csv_metadata)} CSV entries"
        )

    # ------------------------------------------------------------------
    # Load single sample
    # ------------------------------------------------------------------

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        if self._mode == "coco":
            return self._load_coco_sample(idx)
        else:
            return self._load_xml_sample(idx)

    def _load_coco_sample(self, idx: int) -> Dict[str, Any]:
        img_info = self._samples[idx]
        img_id = img_info["id"]
        h, w = img_info["height"], img_info["width"]

        # Load DICOM
        images_dir = Path(self.root_dir) / "images"
        dcm = DicomImage(file_path=str(images_dir / img_info["file_name"]))
        dcm.load()
        image_tensor = self._to_chw(dcm.pixel_data.float())

        # Render mask from polygon annotations
        anns = self._img_to_anns.get(img_id, [])
        mask = self._render_coco_mask(anns, h, w)

        metadata = {
            "case_id": img_info["file_name"].replace(".dcm", ""),
            "file_name": img_info["file_name"],
            "num_annotations": len(anns),
        }

        return {"image": image_tensor, "mask": mask, "metadata": metadata}

    def _load_xml_sample(self, idx: int) -> Dict[str, Any]:
        from medical_image.utils.mask_utils import xml_to_binary_mask
        from medical_image.utils.pairing import INbreastSample

        sample: INbreastSample = self._samples[idx]

        dcm = DicomImage(file_path=sample.dicom_path)
        dcm.load()
        image_tensor = self._to_chw(dcm.pixel_data.float())

        if sample.xml_path is not None:
            h, w = dcm.height, dcm.width
            mask_np = xml_to_binary_mask(
                sample.xml_path, (h, w), point_radius=self.point_radius
            )
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        else:
            mask_tensor = torch.zeros(1, dcm.height, dcm.width, dtype=torch.float32)
            logger.warning(f"No XML annotation for case {sample.case_id}")

        csv_meta = self._csv_metadata.get(sample.case_id, {})
        metadata = {
            "case_id": sample.case_id,
            "file_name": Path(sample.dicom_path).name,
            "num_annotations": 1 if sample.xml_path else 0,
            "laterality": csv_meta.get("laterality", ""),
            "view": csv_meta.get("view", ""),
            "birads": csv_meta.get("birads", -1),
        }

        return {"image": image_tensor, "mask": mask_tensor, "metadata": metadata}

    # ------------------------------------------------------------------
    # COCO mask rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _render_coco_mask(anns: List[dict], h: int, w: int) -> torch.Tensor:
        """Render polygon annotations as a binary mask tensor (1, H, W)."""
        from skimage.draw import polygon as draw_polygon

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                pts = np.array(seg).reshape(-1, 2)
                rr, cc = draw_polygon(pts[:, 1], pts[:, 0], shape=(h, w))
                mask[rr, cc] = 1

        return torch.from_numpy(mask).unsqueeze(0).float()

    # ------------------------------------------------------------------
    # CSV parsing (XML mode only)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_csv(root: Path) -> Dict[str, Dict[str, Any]]:
        """Parse ``INbreast.csv`` for metadata (semicolon-delimited)."""
        csv_path = root / "INbreast.csv"
        if not csv_path.exists():
            logger.warning(f"INbreast.csv not found at {csv_path}")
            return {}

        metadata = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                case_id = row.get("File Name", "").strip()
                if not case_id:
                    continue
                try:
                    birads = int(row.get("Bi-Rads", "-1").strip())
                except ValueError:
                    birads = -1

                metadata[case_id] = {
                    "laterality": row.get("Laterality", "").strip(),
                    "view": row.get("View", "").strip(),
                    "birads": birads,
                    "acr": row.get("ACR", "").strip(),
                }

        logger.debug(f"Parsed {len(metadata)} entries from INbreast.csv")
        return metadata
