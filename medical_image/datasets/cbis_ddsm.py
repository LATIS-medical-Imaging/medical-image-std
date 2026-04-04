"""
CBIS-DDSM mammography dataset with full-image and patch-based loading.

Supports the TCIA CBIS-DDSM directory layout with automatic pairing of
full mammogram images and their corresponding ROI mask DICOMs.
"""

import random
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List, Literal

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve
from skimage.feature import match_template
from skimage.filters import sobel

from medical_image.utils.logging import logger
from medical_image.data.dicom_image import DicomImage
from medical_image.datasets.base_dataset import BaseDataset
from medical_image.utils.mask_utils import stack_dicom_masks
from medical_image.utils.pairing import pair_cbis_ddsm, CBISDDSMSample


class CBISDDSMDataset(BaseDataset):
    """
    PyTorch Dataset for the CBIS-DDSM mammography database.

    Supports two loading modes:

    - ``"full_image"``: Load the entire mammogram (optionally resized).
    - ``"patch"``: Extract sliding-window patches from each mammogram.

    Each sample via ``__getitem__`` returns:

    .. code-block:: python

        {
            "image": Tensor[1, H, W],      # mammogram (or patch)
            "mask":  Tensor[1, H, W],       # OR-merged ROI masks
            "metadata": {
                "case_id": str,
                "patient_id": str,
                "side": str,
                "view": str,
                "num_masks": int,
                "patch_idx": int,           # only in patch mode
                "patch_position": (y, x),   # only in patch mode
            }
        }

    Use :meth:`get_detailed_sample` for the rich format with per-ROI
    bounding boxes, crops, and masks.

    Args:
        root_dir: Path to the manifest directory containing ``CBIS-DDSM/``.
        mode: ``"full_image"`` or ``"patch"``.
        patch_size: Patch side length (used when ``mode="patch"``).
        stride: Stride between patches (used when ``mode="patch"``).
        transform: Optional image transform.
        target_transform: Optional mask transform.
        target_size: Optional (H, W) resize for full_image mode.
        percentage: Optional float (0-1] to use a random subset of cases.
        seed: Random seed for reproducible subset selection.

    Example:

    .. code-block:: python

        dataset = CBISDDSMDataset(
            root_dir="data/ddsm",
            mode="patch",
            patch_size=512,
            stride=256,
            percentage=0.2,
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            collate_fn=dataset.collate_fn,
        )
    """

    def __init__(
        self,
        root_dir: str,
        mode: Literal["full_image", "patch"] = "full_image",
        patch_size: int = 512,
        stride: int = 256,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_size: Optional[Tuple[int, int]] = None,
        percentage: Optional[float] = None,
        seed: int = 42,
    ):
        self.mode = mode
        self.patch_size = patch_size
        self.stride = stride
        self.percentage = percentage
        self.seed = seed

        # Will be populated by _build_sample_list
        self._case_samples: List[CBISDDSMSample] = []
        self._patch_index: List[Tuple[int, int, int]] = []  # (case_idx, y, x)

        super().__init__(root_dir, transform, target_transform, target_size)

    # ------------------------------------------------------------------
    # Build sample list
    # ------------------------------------------------------------------

    def _build_sample_list(self) -> None:
        self._case_samples = pair_cbis_ddsm(self.root_dir)

        if not self._case_samples:
            logger.warning(f"No CBIS-DDSM cases found in {self.root_dir}")
            self._samples = []
            return

        # Apply percentage filtering
        if self.percentage is not None:
            n_select = max(1, int(len(self._case_samples) * self.percentage))
            rng = random.Random(self.seed)
            self._case_samples = sorted(
                rng.sample(self._case_samples, n_select),
                key=lambda s: s.case_id,
            )
            logger.info(
                f"CBIS-DDSM: selected {n_select} cases "
                f"({self.percentage * 100:.0f}%)"
            )

        if self.mode == "full_image":
            # One sample per case
            self._samples = list(range(len(self._case_samples)))

        elif self.mode == "patch":
            # Pre-compute patch positions for each case
            self._samples = []
            self._patch_index = []

            for case_idx, case in enumerate(self._case_samples):
                # Need to peek at image dimensions
                h, w = self._peek_image_size(case.mammogram_path)
                positions = self._compute_patch_positions(h, w)

                for y, x in positions:
                    sample_idx = len(self._samples)
                    self._samples.append(sample_idx)
                    self._patch_index.append((case_idx, y, x))

            logger.info(
                f"CBIS-DDSM patch mode: {len(self._samples)} patches "
                f"from {len(self._case_samples)} cases "
                f"(size={self.patch_size}, stride={self.stride})"
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'full_image' or 'patch'.")

    # ------------------------------------------------------------------
    # Load single sample (BaseDataset contract)
    # ------------------------------------------------------------------

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        if self.mode == "full_image":
            return self._load_full_image(idx)
        else:
            return self._load_patch(idx)

    def _load_full_image(self, idx: int) -> Dict[str, Any]:
        case = self._case_samples[idx]

        # Load mammogram
        dcm = DicomImage(file_path=case.mammogram_path)
        dcm.load()
        image_tensor = self._to_chw(dcm.pixel_data.float())

        # Load and merge masks
        h, w = dcm.height, dcm.width
        if case.mask_paths:
            try:
                mask_tensor = stack_dicom_masks(case.mask_paths)
                # Resize mask to match mammogram if needed
                if mask_tensor.shape[1:] != (h, w):
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0),
                        size=(h, w),
                        mode="nearest",
                    ).squeeze(0)
            except Exception as e:
                logger.warning(f"Failed to load masks for {case.case_id}: {e}")
                mask_tensor = torch.zeros(1, h, w, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros(1, h, w, dtype=torch.float32)

        metadata = {
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "side": case.side,
            "view": case.view,
            "num_masks": len(case.mask_paths),
            "file_name": Path(case.mammogram_path).name,
        }

        return {"image": image_tensor, "mask": mask_tensor, "metadata": metadata}

    def _load_patch(self, idx: int) -> Dict[str, Any]:
        case_idx, py, px = self._patch_index[idx]
        case = self._case_samples[case_idx]

        # Load mammogram
        dcm = DicomImage(file_path=case.mammogram_path)
        dcm.load()
        image_tensor = self._to_chw(dcm.pixel_data.float())

        # Extract patch from image
        image_patch = image_tensor[
            :, py : py + self.patch_size, px : px + self.patch_size
        ]

        # Load mask and extract corresponding patch
        h, w = dcm.height, dcm.width
        if case.mask_paths:
            try:
                mask_tensor = stack_dicom_masks(case.mask_paths)
                if mask_tensor.shape[1:] != (h, w):
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0),
                        size=(h, w),
                        mode="nearest",
                    ).squeeze(0)
            except Exception as e:
                logger.warning(f"Failed to load masks for {case.case_id}: {e}")
                mask_tensor = torch.zeros(1, h, w, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros(1, h, w, dtype=torch.float32)

        mask_patch = mask_tensor[
            :, py : py + self.patch_size, px : px + self.patch_size
        ]

        # Pad if patch is at the edge
        _, ph, pw = image_patch.shape
        if ph < self.patch_size or pw < self.patch_size:
            image_patch = F.pad(
                image_patch, (0, self.patch_size - pw, 0, self.patch_size - ph)
            )
            mask_patch = F.pad(
                mask_patch, (0, self.patch_size - pw, 0, self.patch_size - ph)
            )

        metadata = {
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "side": case.side,
            "view": case.view,
            "num_masks": len(case.mask_paths),
            "patch_idx": idx,
            "patch_position": (py, px),
        }

        return {"image": image_patch, "mask": mask_patch, "metadata": metadata}

    # ------------------------------------------------------------------
    # Detailed sample with per-ROI bboxes, crops, and masks
    # ------------------------------------------------------------------

    def get_detailed_sample(self, idx: int) -> Dict[str, Any]:
        """
        Load a full-image sample with per-ROI bounding boxes, crops, and masks.

        Only available in ``"full_image"`` mode.

        Returns:
            Dict with:
                - ``"image"``: ``Tensor[1, H, W]`` — full mammogram
                - ``"bboxes"``: ``Tensor[N, 4]`` — ``(x_min, y_min, x_max, y_max)``
                  in full mammogram coordinates
                - ``"rois"``: ``List[Tensor]`` — ROI crop tensors (from 1-1.dcm)
                - ``"masks"``: ``List[Tensor]`` — ROI mask tensors (from 1-2.dcm)
                - ``"label"``: task string (e.g. ``"Calc-Test"``)
                - ``"meta"``: dict with patient_id, view, side, task
        """
        if self.mode != "full_image":
            raise ValueError("get_detailed_sample is only available in 'full_image' mode")

        case = self._case_samples[idx]

        # Load full mammogram
        dcm = DicomImage(file_path=case.mammogram_path)
        dcm.load()
        full_image = self._to_chw(dcm.pixel_data.float())
        full_array = dcm.pixel_data.numpy().astype(np.float64)

        rois: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        bboxes: List[List[int]] = []

        for entry in case.roi_entries:
            # Load ROI crop
            roi_tensor: Optional[torch.Tensor] = None
            roi_array: Optional[np.ndarray] = None
            if entry.roi_path:
                try:
                    roi_ds = pydicom.dcmread(entry.roi_path)
                    roi_array = roi_ds.pixel_array.astype(np.float64)
                    roi_tensor = self._to_chw(
                        torch.from_numpy(roi_array.astype(np.float32))
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load ROI crop {entry.roi_path}: {e}"
                    )

            # Load mask
            mask_tensor: Optional[torch.Tensor] = None
            if entry.mask_path:
                try:
                    mask_ds = pydicom.dcmread(entry.mask_path)
                    mask_arr = (mask_ds.pixel_array > 0).astype(np.float32)
                    mask_tensor = self._to_chw(torch.from_numpy(mask_arr))
                except Exception as e:
                    logger.warning(
                        f"Failed to load ROI mask {entry.mask_path}: {e}"
                    )

            if roi_tensor is not None:
                rois.append(roi_tensor)
            if mask_tensor is not None:
                masks.append(mask_tensor)

            # Compute bounding box from ROI crop position in full mammogram
            if roi_array is not None:
                try:
                    bbox = self._locate_roi_in_mammogram(full_array, roi_array)
                    bboxes.append(bbox)
                except Exception as e:
                    logger.warning(
                        f"Failed to compute bbox for {entry.roi_path}: {e}"
                    )

        bbox_tensor = (
            torch.tensor(bboxes, dtype=torch.int64)
            if bboxes
            else torch.zeros(0, 4, dtype=torch.int64)
        )

        return {
            "image": full_image,
            "bboxes": bbox_tensor,
            "rois": rois,
            "masks": masks,
            "label": case.task,
            "meta": {
                "patient_id": case.patient_id,
                "view": case.view,
                "side": case.side,
                "task": case.task,
            },
        }

    # ------------------------------------------------------------------
    # Bounding box computation
    # ------------------------------------------------------------------

    @staticmethod
    def get_bounding_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Compute bounding boxes from a binary mask array.

        Finds connected components of non-zero pixels and returns
        the bounding box of each component.

        Args:
            mask: 2D numpy array (H, W) with non-zero pixels marking ROIs.

        Returns:
            List of ``(x_min, y_min, x_max, y_max)`` tuples.
        """
        from scipy.ndimage import label as ndimage_label

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        binary = (mask > 0).astype(np.uint8)
        if binary.sum() == 0:
            return []

        labeled, n_components = ndimage_label(binary)
        boxes = []
        for comp_id in range(1, n_components + 1):
            ys, xs = np.where(labeled == comp_id)
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            boxes.append((x_min, y_min, x_max, y_max))

        return boxes

    @staticmethod
    def _locate_roi_in_mammogram(full_image: np.ndarray, roi_crop: np.ndarray):
        th, tw = roi_crop.shape

        # --- Step 1: Normalize (z-score) ---
        full_norm = (full_image - np.mean(full_image)) / (np.std(full_image) + 1e-8)
        roi_norm = (roi_crop - np.mean(roi_crop)) / (np.std(roi_crop) + 1e-8)

        # --- Step 2: Edge enhancement (VERY important for mammograms) ---
        full_edges = sobel(full_norm)
        roi_edges = sobel(roi_norm)

        # --- Step 3: Normalized cross-correlation ---
        result = match_template(full_edges, roi_edges, pad_input=False)

        # --- Step 4: Locate best match ---
        y_off, x_off = np.unravel_index(np.argmax(result), result.shape)

        return [int(x_off), int(y_off), int(x_off + tw), int(y_off + th)]
    # ------------------------------------------------------------------
    # Custom collate_fn
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function that handles variable-sized masks.

        Pads all images and masks to the maximum spatial dimensions in the batch.

        Returns:
            Dict with stacked ``"image"`` and ``"mask"`` tensors,
            and a list of ``"metadata"`` dicts.
        """
        # Find max spatial dimensions
        max_h = max(s["image"].shape[1] for s in batch)
        max_w = max(s["image"].shape[2] for s in batch)

        images = []
        masks = []
        metadata = []

        for s in batch:
            img = s["image"]
            msk = s["mask"]

            # Pad to max dimensions
            _, h, w = img.shape
            pad_h = max_h - h
            pad_w = max_w - w

            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, pad_w, 0, pad_h))
                msk = F.pad(msk, (0, pad_w, 0, pad_h))

            images.append(img)
            masks.append(msk)
            metadata.append(s["metadata"])

        return {
            "image": torch.stack(images),
            "mask": torch.stack(masks),
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _peek_image_size(dcm_path: str) -> Tuple[int, int]:
        """
        Read DICOM header to get image dimensions without loading pixel data.
        """
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return int(ds.Rows), int(ds.Columns)

    def _compute_patch_positions(
        self, h: int, w: int
    ) -> List[Tuple[int, int]]:
        """
        Compute top-left (y, x) positions for sliding-window patches.
        """
        positions = []
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                positions.append((y, x))
        return positions