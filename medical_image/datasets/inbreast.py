"""
INbreast mammography dataset.

PyTorch-compatible dataset that loads DICOM mammograms with XML annotations
converted to binary segmentation masks on-the-fly. Integrates with the
existing ``DicomImage`` framework for lazy loading.
"""

import csv
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List

import torch

from medical_image.utils.logging import logger
from medical_image.data.dicom_image import DicomImage
from medical_image.datasets.base_dataset import BaseDataset
from medical_image.utils.mask_utils import xml_to_binary_mask
from medical_image.utils.pairing import pair_inbreast, INbreastSample


class INbreastDataset(BaseDataset):
    """
    PyTorch Dataset for the INbreast mammography database.

    Each sample returns:

    .. code-block:: python

        {
            "image": Tensor[1, H, W],    # DICOM pixel data
            "mask":  Tensor[1, H, W],     # Binary mask from XML annotations
            "metadata": {
                "case_id": str,
                "laterality": str,        # from CSV
                "view": str,              # from CSV
                "birads": int,            # from CSV
                "file_name": str,
            }
        }

    Args:
        root_dir: Path to ``INbreast Release 1.0/`` directory.
        transform: Optional transform for the image tensor.
        target_transform: Optional transform for the mask tensor.
        target_size: Optional (H, W) resize target.
        point_radius: Radius for rendering single-point ROIs (default: 3).

    Example:

    .. code-block:: python

        dataset = INbreastDataset("path/to/INbreast Release 1.0")
        sample = dataset[0]
        print(sample["image"].shape)     # torch.Size([1, H, W])
        print(sample["mask"].shape)      # torch.Size([1, H, W])
        print(sample["metadata"])        # {'case_id': '20586908', ...}
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
        self._csv_metadata: Dict[str, Dict[str, Any]] = {}
        super().__init__(root_dir, transform, target_transform, target_size)

    # ------------------------------------------------------------------
    # Build sample list
    # ------------------------------------------------------------------

    def _build_sample_list(self) -> None:
        root = Path(self.root_dir)

        dicoms_dir = root / "AllDICOMs"
        xml_dir = root / "AllXML"
        roi_dir = root / "AllROI"

        if not dicoms_dir.exists():
            raise FileNotFoundError(f"AllDICOMs not found at {dicoms_dir}")

        # Parse CSV metadata
        self._csv_metadata = self._parse_csv(root)

        # Pair DICOMs ↔ XMLs
        self._samples: List[INbreastSample] = pair_inbreast(
            str(dicoms_dir),
            str(xml_dir),
            str(roi_dir) if roi_dir.exists() else None,
        )

        logger.info(
            f"INbreastDataset: {len(self._samples)} samples, "
            f"{len(self._csv_metadata)} CSV entries"
        )

    # ------------------------------------------------------------------
    # Load single sample
    # ------------------------------------------------------------------

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        sample: INbreastSample = self._samples[idx]

        # Load DICOM via framework
        dcm = DicomImage(file_path=sample.dicom_path)
        dcm.load()

        image_tensor = self._to_chw(dcm.pixel_data.float())

        # Generate binary mask from XML
        if sample.xml_path is not None:
            h, w = dcm.height, dcm.width
            mask_np = xml_to_binary_mask(
                sample.xml_path, (h, w), point_radius=self.point_radius
            )
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        else:
            # No annotation → empty mask
            mask_tensor = torch.zeros(1, dcm.height, dcm.width, dtype=torch.float32)
            logger.warning(f"No XML annotation for case {sample.case_id}")

        # Build metadata
        csv_meta = self._csv_metadata.get(sample.case_id, {})
        metadata = {
            "case_id": sample.case_id,
            "laterality": csv_meta.get("laterality", ""),
            "view": csv_meta.get("view", ""),
            "birads": csv_meta.get("birads", -1),
            "file_name": Path(sample.dicom_path).name,
        }

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_csv(root: Path) -> Dict[str, Dict[str, Any]]:
        """
        Parse ``INbreast.csv`` for metadata.

        CSV format (semicolon-delimited):
            Patient ID;Patient age;Laterality;View;Acquisition date;File Name;ACR;Bi-Rads

        Returns:
            Dict mapping ``file_name`` (case ID) → metadata dict.
        """
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
