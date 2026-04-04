"""
Custom INbreast dataset with TIF segmentation masks.

Extends the INbreast structure with pre-computed TIF masks in an ``AllMasks/``
directory. Falls back to XML-generated masks when TIF is unavailable.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List

import torch

from medical_image.utils.logging import logger
from medical_image.data.dicom_image import DicomImage
from medical_image.datasets.base_dataset import BaseDataset
from medical_image.datasets.inbreast import INbreastDataset
from medical_image.utils.mask_utils import load_tif_mask, xml_to_binary_mask
from medical_image.utils.pairing import pair_custom_inbreast, CustomINbreastSample


class CustomINbreastDataset(BaseDataset):
    """
    PyTorch Dataset for a custom INbreast-based dataset with TIF masks.

    Directory structure::

        root_dir/
        ├── AllMasks/
        │   ├── 20586934_mask.tif
        │   └── ...
        └── INbreast Release 1.0/
            ├── AllDICOMs/
            ├── AllXML/
            └── AllROI/

    Each sample returns:

    .. code-block:: python

        {
            "image": Tensor[1, H, W],
            "mask":  Tensor[1, H, W],     # from TIF or XML fallback
            "metadata": {"case_id": str, "mask_source": "tif"|"xml"|"empty", ...}
        }

    Args:
        root_dir: Path to the custom dataset root containing ``AllMasks/``
                  and ``INbreast Release 1.0/``.
        transform: Optional image transform.
        target_transform: Optional mask transform.
        target_size: Optional (H, W) resize target.
        point_radius: Radius for XML single-point ROI rendering.
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

        # Locate directories
        masks_dir = root / "AllMasks"
        inbreast_dir = root / "INbreast Release 1.0"

        if not inbreast_dir.exists():
            raise FileNotFoundError(f"'INbreast Release 1.0' not found in {root}")

        dicoms_dir = inbreast_dir / "AllDICOMs"
        xml_dir = inbreast_dir / "AllXML"
        roi_dir = inbreast_dir / "AllROI"

        if not dicoms_dir.exists():
            raise FileNotFoundError(f"AllDICOMs not found at {dicoms_dir}")

        # Parse CSV metadata (reuse INbreast logic)
        self._csv_metadata = INbreastDataset._parse_csv(inbreast_dir)

        # Pair DICOMs ↔ XMLs ↔ Masks
        self._samples: List[CustomINbreastSample] = pair_custom_inbreast(
            str(dicoms_dir),
            str(xml_dir),
            str(masks_dir) if masks_dir.exists() else "",
            str(roi_dir) if roi_dir.exists() else None,
        )

        logger.info(
            f"CustomINbreastDataset: {len(self._samples)} samples, "
            f"{sum(1 for s in self._samples if s.mask_path)} with TIF masks"
        )

    # ------------------------------------------------------------------
    # Load single sample
    # ------------------------------------------------------------------

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        sample: CustomINbreastSample = self._samples[idx]

        # Load DICOM
        dcm = DicomImage(file_path=sample.dicom_path)
        dcm.load()
        image_tensor = self._to_chw(dcm.pixel_data.float())

        h, w = dcm.height, dcm.width
        mask_source = "empty"

        # Priority 1: TIF mask
        if sample.mask_path is not None:
            try:
                mask_np = load_tif_mask(sample.mask_path)
                # Resize mask if dimensions don't match DICOM
                if mask_np.shape != (h, w):
                    logger.debug(
                        f"Resizing mask {mask_np.shape} → ({h}, {w}) "
                        f"for case {sample.case_id}"
                    )
                    mask_tensor = (
                        torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
                    )
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor, size=(h, w), mode="nearest"
                    ).squeeze(0)
                else:
                    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
                mask_source = "tif"
            except Exception as e:
                logger.warning(
                    f"Failed to load TIF mask for case {sample.case_id}: {e}"
                )
                mask_tensor = None

        # Priority 2: XML fallback
        if mask_source != "tif" and sample.xml_path is not None:
            try:
                mask_np = xml_to_binary_mask(
                    sample.xml_path, (h, w), point_radius=self.point_radius
                )
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
                mask_source = "xml"
            except Exception as e:
                logger.warning(
                    f"Failed to generate XML mask for case {sample.case_id}: {e}"
                )
                mask_tensor = None

        # Fallback: empty mask
        if mask_source == "empty" or mask_tensor is None:
            mask_tensor = torch.zeros(1, h, w, dtype=torch.float32)
            mask_source = "empty"
            logger.warning(f"No mask available for case {sample.case_id}")

        # Build metadata
        csv_meta = self._csv_metadata.get(sample.case_id, {})
        metadata = {
            "case_id": sample.case_id,
            "mask_source": mask_source,
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
