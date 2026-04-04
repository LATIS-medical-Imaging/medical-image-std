"""
Unit tests for CBIS-DDSM dataset implementation.

Tests pairing logic, bounding box extraction, detailed sample loading,
and dataset contract compliance against real CBIS-DDSM data.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pytest
import torch

# -------------------------------------------------------------------------
# Test data paths
# -------------------------------------------------------------------------

CBIS_ROOT = "data/dataset"
CBIS_DDSM_DIR = os.path.join(CBIS_ROOT, "CBIS-DDSM")

# Skip all tests if data is not available
pytestmark = pytest.mark.skipif(
    not os.path.exists(CBIS_DDSM_DIR),
    reason="CBIS-DDSM test data not available",
)

# Use a tiny subset for integration tests to avoid OOM on large DICOMs
_SUBSET_PCT = 0.5


# =========================================================================
# Pairing Tests
# =========================================================================


class TestCBISDDSMPairing:
    """Tests for CBIS-DDSM pairing logic in utils/pairing.py."""

    def test_pair_cbis_ddsm_finds_samples(self):
        from medical_image.utils.pairing import pair_cbis_ddsm

        samples = pair_cbis_ddsm(CBIS_ROOT)
        assert len(samples) > 0, "Should find at least one case"

    def test_sample_has_mammogram_path(self):
        from medical_image.utils.pairing import pair_cbis_ddsm

        samples = pair_cbis_ddsm(CBIS_ROOT)
        for s in samples:
            assert os.path.isfile(s.mammogram_path), (
                f"Mammogram not found: {s.mammogram_path}"
            )

    def test_sample_has_required_fields(self):
        from medical_image.utils.pairing import pair_cbis_ddsm

        samples = pair_cbis_ddsm(CBIS_ROOT)
        s = samples[0]
        assert s.case_id
        assert s.patient_id
        assert s.side in ("LEFT", "RIGHT")
        assert s.view in ("CC", "MLO")
        assert s.task in ("Calc-Test", "Calc-Training", "Mass-Test", "Mass-Training")

    def test_roi_entries_populated(self):
        from medical_image.utils.pairing import pair_cbis_ddsm

        samples = pair_cbis_ddsm(CBIS_ROOT)
        with_rois = [s for s in samples if s.roi_entries]
        assert len(with_rois) > 0, "At least some cases should have ROI entries"

    def test_roi_entry_has_roi_or_mask(self):
        from medical_image.utils.pairing import pair_cbis_ddsm

        samples = pair_cbis_ddsm(CBIS_ROOT)
        for s in samples:
            for entry in s.roi_entries:
                assert entry.roi_path or entry.mask_path, (
                    f"ROI entry for {s.case_id} has neither roi_path nor mask_path"
                )

    def test_mask_paths_backward_compat(self):
        from medical_image.utils.pairing import pair_cbis_ddsm

        samples = pair_cbis_ddsm(CBIS_ROOT)
        with_masks = [s for s in samples if s.mask_paths]
        assert len(with_masks) > 0, "mask_paths should still be populated"

    def test_parse_roi_folder_standard_layout(self):
        from medical_image.utils.pairing import _parse_roi_folder

        for entry in sorted(os.listdir(CBIS_DDSM_DIR)):
            if entry.endswith("_1"):
                roi_folder = os.path.join(CBIS_DDSM_DIR, entry)
                result = _parse_roi_folder(roi_folder)
                assert result.roi_path or result.mask_path, (
                    f"Failed to parse ROI folder: {entry}"
                )
                break


# =========================================================================
# Bounding Box Tests
# =========================================================================


class TestBoundingBoxes:
    """Tests for bounding box extraction logic."""

    def test_get_bounding_boxes_single_component(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1

        boxes = CBISDDSMDataset.get_bounding_boxes(mask)
        assert len(boxes) == 1
        x_min, y_min, x_max, y_max = boxes[0]
        assert x_min == 30
        assert y_min == 20
        assert x_max == 59
        assert y_max == 39

    def test_get_bounding_boxes_multiple_components(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[60:70, 70:80] = 1

        boxes = CBISDDSMDataset.get_bounding_boxes(mask)
        assert len(boxes) == 2

    def test_get_bounding_boxes_empty_mask(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        mask = np.zeros((100, 100), dtype=np.uint8)
        boxes = CBISDDSMDataset.get_bounding_boxes(mask)
        assert len(boxes) == 0

    def test_get_bounding_boxes_rejects_3d(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        mask = np.zeros((1, 100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D"):
            CBISDDSMDataset.get_bounding_boxes(mask)

    def test_locate_roi_in_mammogram(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        rng = np.random.RandomState(42)
        full = rng.rand(200, 200).astype(np.float64) * 100

        y_off, x_off = 50, 70
        crop = full[y_off : y_off + 30, x_off : x_off + 40].copy()

        bbox = CBISDDSMDataset._locate_roi_in_mammogram(full, crop)
        assert bbox[0] == x_off
        assert bbox[1] == y_off
        assert bbox[2] == x_off + 40
        assert bbox[3] == y_off + 30


# =========================================================================
# Dataset Tests (use percentage to limit RAM)
# =========================================================================


class TestCBISDDSMDataset:
    """Tests for the CBISDDSMDataset class."""

    def test_dataset_creation(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        assert len(ds) > 0

    def test_getitem_returns_dict(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds[0]

        assert "image" in sample
        assert "mask" in sample
        assert "metadata" in sample

    def test_getitem_tensor_shapes(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds[0]

        img = sample["image"]
        mask = sample["mask"]

        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert img.ndim == 3  # (C, H, W)
        assert mask.ndim == 3  # (1, H, W)
        assert img.shape[0] == 1  # Grayscale
        assert mask.shape[0] == 1

    def test_metadata_has_required_keys(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds[0]
        meta = sample["metadata"]

        assert "case_id" in meta
        assert "patient_id" in meta
        assert "side" in meta
        assert "view" in meta

    def test_with_resize(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        target = (256, 256)
        ds = CBISDDSMDataset(CBIS_ROOT, target_size=target, percentage=_SUBSET_PCT)
        sample = ds[0]

        assert sample["image"].shape[1:] == target
        assert sample["mask"].shape[1:] == target

    def test_percentage_filtering(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds_full = CBISDDSMDataset(CBIS_ROOT)
        ds_half = CBISDDSMDataset(CBIS_ROOT, percentage=0.5)

        assert len(ds_half) < len(ds_full)
        assert len(ds_half) > 0


# =========================================================================
# Detailed Sample Tests (use percentage to limit RAM)
# =========================================================================


class TestDetailedSample:
    """Tests for get_detailed_sample() rich output format."""

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def test_detailed_sample_structure(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds.get_detailed_sample(5)

        # Convert image (C, H, W) → (H, W)
        img = sample["image"].squeeze().cpu().numpy()

        # Normalize for visualization (important!)
        img = (img - img.min()) / (img.max() - img.min())

        # ---------------------------
        # 1. Full image with bboxes
        # ---------------------------
        fig, ax = plt.subplots(1, figsize=(8, 10))
        ax.imshow(img, cmap="gray")

        for bbox in sample["bboxes"]:
            x1, y1, x2, y2 = bbox.tolist()
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title("Full Image with Bounding Boxes")
        ax.axis("off")
        plt.show()

        # ------------------------------------------
        # 2. ROI vs BBox crop (VERY IMPORTANT DEBUG)
        # ------------------------------------------
        bboxes = sample["bboxes"]
        rois = sample["rois"]

        for i, (bbox, roi) in enumerate(zip(bboxes, rois)):
            x1, y1, x2, y2 = map(int, bbox.tolist())

            # Crop manually from full image
            cropped = img[y1:y2, x1:x2]

            # Dataset ROI
            roi_img = roi.squeeze().cpu().numpy()
            roi_img = (roi_img - roi_img.min()) / (roi_img.max() - roi_img.min())

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Left: manual crop
            axes[0].imshow(cropped, cmap="gray")
            axes[0].set_title(f"Manual Crop (BBox {i})")
            axes[0].axis("off")

            # Right: dataset ROI
            axes[1].imshow(roi_img, cmap="gray")
            axes[1].set_title(f"Dataset ROI {i}")
            axes[1].axis("off")

            plt.suptitle(f"ROI vs BBox Check #{i}")
            plt.show()

        assert "image" in sample
        assert "bboxes" in sample
        assert "rois" in sample
        assert "masks" in sample
        assert "label" in sample
        assert "meta" in sample

    def test_detailed_sample_image_is_tensor(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds.get_detailed_sample(0)

        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].ndim == 3
        assert sample["image"].shape[0] == 1

    def test_detailed_sample_bboxes_shape(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds.get_detailed_sample(0)

        bboxes = sample["bboxes"]
        assert isinstance(bboxes, torch.Tensor)
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4

    def test_detailed_sample_rois_are_tensors(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds.get_detailed_sample(0)

        assert isinstance(sample["rois"], list)
        for roi in sample["rois"]:
            assert isinstance(roi, torch.Tensor)
            assert roi.ndim == 3

    def test_detailed_sample_masks_are_tensors(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds.get_detailed_sample(0)

        assert isinstance(sample["masks"], list)
        for mask in sample["masks"]:
            assert isinstance(mask, torch.Tensor)
            assert mask.ndim == 3

    def test_detailed_sample_meta_fields(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds.get_detailed_sample(0)

        meta = sample["meta"]
        assert "patient_id" in meta
        assert "view" in meta
        assert "side" in meta
        assert "task" in meta

    def test_detailed_sample_bboxes_valid_coordinates(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds.get_detailed_sample(0)

        _, h, w = sample["image"].shape
        for bbox in sample["bboxes"]:
            x_min, y_min, x_max, y_max = bbox.tolist()
            assert 0 <= x_min < x_max <= w
            assert 0 <= y_min < y_max <= h

    def test_detailed_sample_rejects_patch_mode(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, mode="patch", patch_size=256, stride=256,
                             percentage=_SUBSET_PCT)
        with pytest.raises(ValueError, match="full_image"):
            ds.get_detailed_sample(0)

    def test_detailed_sample_label_is_task(self):
        from medical_image.datasets.cbis_ddsm import CBISDDSMDataset

        ds = CBISDDSMDataset(CBIS_ROOT, percentage=_SUBSET_PCT)
        sample = ds.get_detailed_sample(0)

        assert sample["label"] in (
            "Calc-Test", "Calc-Training", "Mass-Test", "Mass-Training"
        )