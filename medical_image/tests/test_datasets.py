"""
Unit tests for medical image datasets.

Tests pairing logic, mask utilities, and dataset classes against
real INbreast data.
"""

import os
import pytest
import torch
import numpy as np

# -------------------------------------------------------------------------
# Test data paths
# -------------------------------------------------------------------------

CUSTOM_ROOT = "/home/bobmarley/PycharmProjects/graph-micro-calcification/custom_data"
INBREAST_ROOT = os.path.join(CUSTOM_ROOT, "INbreast Release 1.0")

# Skip all tests if data is not available
pytestmark = pytest.mark.skipif(
    not os.path.exists(INBREAST_ROOT),
    reason="INbreast test data not available",
)


# =========================================================================
# Pairing Tests
# =========================================================================


class TestPairing:
    """Tests for utils/pairing.py matching logic."""

    def test_pair_inbreast_finds_samples(self):
        from medical_image.utils.pairing import pair_inbreast

        samples = pair_inbreast(
            os.path.join(INBREAST_ROOT, "AllDICOMs"),
            os.path.join(INBREAST_ROOT, "AllXML"),
        )
        assert len(samples) > 0, "Should find at least one DICOM"

    def test_pair_inbreast_has_xml(self):
        from medical_image.utils.pairing import pair_inbreast

        samples = pair_inbreast(
            os.path.join(INBREAST_ROOT, "AllDICOMs"),
            os.path.join(INBREAST_ROOT, "AllXML"),
        )
        with_xml = [s for s in samples if s.xml_path is not None]
        assert len(with_xml) > 0, "At least some samples should have XML"

    def test_pair_inbreast_case_ids_are_numeric(self):
        from medical_image.utils.pairing import pair_inbreast

        samples = pair_inbreast(
            os.path.join(INBREAST_ROOT, "AllDICOMs"),
            os.path.join(INBREAST_ROOT, "AllXML"),
        )
        for s in samples:
            assert s.case_id.isdigit(), f"case_id should be numeric: {s.case_id}"

    def test_pair_custom_inbreast_finds_masks(self):
        from medical_image.utils.pairing import pair_custom_inbreast

        samples = pair_custom_inbreast(
            os.path.join(INBREAST_ROOT, "AllDICOMs"),
            os.path.join(INBREAST_ROOT, "AllXML"),
            os.path.join(CUSTOM_ROOT, "AllMasks"),
        )
        with_mask = [s for s in samples if s.mask_path is not None]
        assert len(with_mask) > 0, "Should find at least one mask match"

    def test_pair_custom_inbreast_mask_id_consistency(self):
        from medical_image.utils.pairing import pair_custom_inbreast

        samples = pair_custom_inbreast(
            os.path.join(INBREAST_ROOT, "AllDICOMs"),
            os.path.join(INBREAST_ROOT, "AllXML"),
            os.path.join(CUSTOM_ROOT, "AllMasks"),
        )
        for s in samples:
            if s.mask_path is not None:
                mask_name = os.path.basename(s.mask_path)
                assert mask_name.startswith(
                    s.case_id
                ), f"Mask {mask_name} should start with case_id {s.case_id}"


# =========================================================================
# Mask Utilities Tests
# =========================================================================


class TestMaskUtils:
    """Tests for utils/mask_utils.py."""

    def test_parse_inbreast_xml(self):
        from medical_image.utils.mask_utils import parse_inbreast_xml

        xml_dir = os.path.join(INBREAST_ROOT, "AllXML")
        xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
        assert len(xml_files) > 0

        rois = parse_inbreast_xml(os.path.join(xml_dir, xml_files[0]))
        assert isinstance(rois, list)

    def test_xml_to_binary_mask_shape(self):
        from medical_image.utils.mask_utils import xml_to_binary_mask

        xml_dir = os.path.join(INBREAST_ROOT, "AllXML")
        xml_files = sorted(f for f in os.listdir(xml_dir) if f.endswith(".xml"))
        xml_path = os.path.join(xml_dir, xml_files[0])

        shape = (4084, 3328)  # Typical INbreast dimensions
        mask = xml_to_binary_mask(xml_path, shape)

        assert mask.shape == shape
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})

    def test_load_tif_mask(self):
        from medical_image.utils.mask_utils import load_tif_mask

        masks_dir = os.path.join(CUSTOM_ROOT, "AllMasks")
        if not os.path.exists(masks_dir):
            pytest.skip("AllMasks directory not available")

        tif_files = [f for f in os.listdir(masks_dir) if f.endswith(".tif")]
        assert len(tif_files) > 0

        mask = load_tif_mask(os.path.join(masks_dir, tif_files[0]))
        assert mask.ndim == 2
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})


# =========================================================================
# INbreast Dataset Tests
# =========================================================================


class TestINbreastDataset:
    """Tests for datasets/inbreast.py."""

    def test_dataset_creation(self):
        from medical_image.datasets.inbreast import INbreastDataset

        ds = INbreastDataset(INBREAST_ROOT)
        assert len(ds) > 0

    def test_getitem_returns_dict(self):
        from medical_image.datasets.inbreast import INbreastDataset

        ds = INbreastDataset(INBREAST_ROOT)
        sample = ds[0]

        assert "image" in sample
        assert "mask" in sample
        assert "metadata" in sample

    def test_getitem_tensor_shapes(self):
        from medical_image.datasets.inbreast import INbreastDataset

        ds = INbreastDataset(INBREAST_ROOT)
        sample = ds[0]

        img = sample["image"]
        mask = sample["mask"]

        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert img.ndim == 3  # (C, H, W)
        assert mask.ndim == 3  # (1, H, W)
        assert img.shape[0] == 1  # Grayscale
        assert mask.shape[0] == 1
        assert img.shape[1:] == mask.shape[1:]  # Same spatial dims

    def test_metadata_has_case_id(self):
        from medical_image.datasets.inbreast import INbreastDataset

        ds = INbreastDataset(INBREAST_ROOT)
        sample = ds[0]

        meta = sample["metadata"]
        assert "case_id" in meta
        assert meta["case_id"].isdigit()

    def test_with_resize(self):
        from medical_image.datasets.inbreast import INbreastDataset

        target = (256, 256)
        ds = INbreastDataset(INBREAST_ROOT, target_size=target)
        sample = ds[0]

        assert sample["image"].shape[1:] == target
        assert sample["mask"].shape[1:] == target


# =========================================================================
# Custom INbreast Dataset Tests
# =========================================================================


class TestCustomINbreastDataset:
    """Tests for datasets/custom_inbreast.py."""

    def test_dataset_creation(self):
        from medical_image.datasets.custom_inbreast import CustomINbreastDataset

        ds = CustomINbreastDataset(CUSTOM_ROOT)
        assert len(ds) > 0

    def test_getitem_returns_dict(self):
        from medical_image.datasets.custom_inbreast import CustomINbreastDataset

        ds = CustomINbreastDataset(CUSTOM_ROOT)
        sample = ds[0]

        assert "image" in sample
        assert "mask" in sample
        assert "metadata" in sample

    def test_mask_source_is_set(self):
        from medical_image.datasets.custom_inbreast import CustomINbreastDataset

        ds = CustomINbreastDataset(CUSTOM_ROOT)
        sample = ds[0]

        assert sample["metadata"]["mask_source"] in {"tif", "xml", "empty"}

    def test_tensor_shapes_match(self):
        from medical_image.datasets.custom_inbreast import CustomINbreastDataset

        ds = CustomINbreastDataset(CUSTOM_ROOT)
        sample = ds[0]

        img = sample["image"]
        mask = sample["mask"]

        assert img.shape[1:] == mask.shape[1:]


# =========================================================================
# BaseDataset Contract Tests
# =========================================================================


class TestBaseDatasetContract:
    """Tests that the abstract interface is properly enforced."""

    def test_cannot_instantiate_directly(self):
        from medical_image.datasets.base_dataset import BaseDataset

        with pytest.raises(TypeError):
            BaseDataset("/tmp")

    def test_resize_helper(self):
        from medical_image.datasets.base_dataset import BaseDataset

        # 2D tensor
        t = torch.randn(100, 80)
        resized = BaseDataset._resize(t, (50, 40))
        assert resized.shape == (50, 40)

        # 3D (C, H, W) tensor
        t3 = torch.randn(1, 100, 80)
        resized3 = BaseDataset._resize(t3, (50, 40))
        assert resized3.shape == (1, 50, 40)

    def test_to_chw_helper(self):
        from medical_image.datasets.base_dataset import BaseDataset

        # (H, W) → (1, H, W)
        t2 = torch.randn(100, 80)
        assert BaseDataset._to_chw(t2).shape == (1, 100, 80)

        # (H, W, 1) → (1, H, W)
        t_hwc = torch.randn(100, 80, 1)
        assert BaseDataset._to_chw(t_hwc).shape == (1, 100, 80)
