"""
Unit tests for DeepSegmentationAlgorithm.

Tests cover:
  - Synthetic model with controlled output → binary mask + annotations
  - CLAHE toggle
  - min_lesion_area filtering
  - Annotation geometry and metadata
  - Integration with real DICOM via mock_dicom_image
  - Real checkpoint loading from medical-image-baseline/results
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm
from medical_image.data.annotation import Annotation, GeometryType
from medical_image.data.dicom_image import DicomImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.tests.mock_sample import mock_dicom_image

# Path to trained checkpoints
RESULTS_DIR = Path(__file__).resolve().parents[2] / "medical-image-baseline" / "results"


# =========================================================================
# Helpers: fake model
# =========================================================================


class _FakeSegModel(nn.Module):
    """Deterministic model that returns high logits in a fixed region."""

    def __init__(self, hot_y=10, hot_x=10, hot_size=6):
        super().__init__()
        self.hot_y = hot_y
        self.hot_x = hot_x
        self.hot_size = hot_size

    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.full((b, 1, h, w), -5.0)  # sigmoid(-5) ≈ 0
        y, hy = self.hot_y, self.hot_size
        x_, hx = self.hot_x, self.hot_size
        out[:, :, y : y + hy, x_ : x_ + hx] = 5.0  # sigmoid(5) ≈ 1
        return out


def _make_algo(patch_size=64, stride=64, clahe=False, **kwargs):
    """Build algorithm with a fake model — no checkpoint needed."""
    model = _FakeSegModel()
    return DeepSegmentationAlgorithm(
        model=model,
        use_clahe=clahe,
        patch_size=patch_size,
        stride=stride,
        device="cpu",
        **kwargs,
    )


# =========================================================================
# Tests — Synthetic
# =========================================================================


class TestDeepSegmentationSynthetic:
    """Tests using a fake model that produces deterministic masks."""

    def test_output_is_binary(self):
        algo = _make_algo()
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_shape_preserved(self):
        algo = _make_algo()
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        assert out.pixel_data.shape == img.pixel_data.shape

    def test_annotations_created(self):
        algo = _make_algo()
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        assert out.annotations is not None
        assert len(out.annotations) >= 1

    def test_annotation_geometry(self):
        algo = _make_algo()
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        for ann in out.annotations:
            assert ann.shape in (GeometryType.POLYGON, GeometryType.RECTANGLE)
            assert ann.label == "microcalcification"
            assert "confidence" in ann.metadata
            assert "area" in ann.metadata
            assert 0.0 <= ann.metadata["confidence"] <= 1.0
            assert ann.metadata["area"] > 0

    def test_annotation_bbox_metadata(self):
        algo = _make_algo()
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        for ann in out.annotations:
            if ann.shape == GeometryType.POLYGON:
                assert "bbox" in ann.metadata
                bbox = ann.metadata["bbox"]
                assert len(bbox) == 4
                assert bbox[0] <= bbox[2]  # x_min <= x_max
                assert bbox[1] <= bbox[3]  # y_min <= y_max

    def test_probability_map_populated(self):
        algo = _make_algo()
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        assert algo.probability_map is not None
        assert algo.probability_map.shape == img.pixel_data.shape
        assert float(algo.probability_map.min()) >= 0.0
        assert float(algo.probability_map.max()) <= 1.0

    def test_lesion_count(self):
        algo = _make_algo()
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        assert algo.lesion_count == len(out.annotations)
        assert algo.lesion_count >= 1

    def test_min_lesion_area_filtering(self):
        """With a very large min_lesion_area, small lesions are excluded."""
        algo = _make_algo(min_lesion_area=10000)
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        # The fake model produces a 6x6 = 36 pixel lesion, filtered by min=10000
        assert algo.lesion_count == 0
        assert out.annotations == []

    def test_clahe_toggle(self):
        """Algorithm applies CLAHE when configured."""
        algo_no = _make_algo(clahe=False)
        algo_yes = _make_algo(clahe=True)

        assert not algo_no.use_clahe
        assert algo_yes.use_clahe

        # Both should produce valid binary output
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        for algo in [algo_no, algo_yes]:
            out = img.clone()
            algo(img, out)
            unique = torch.unique(out.pixel_data)
            assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_non_square_image(self):
        model = _FakeSegModel(hot_y=5, hot_x=5, hot_size=4)
        algo = DeepSegmentationAlgorithm(
            model=model, patch_size=32, stride=32, device="cpu"
        )
        img = InMemoryImage(array=np.random.rand(48, 80).astype(np.float32))
        out = img.clone()
        algo(img, out)

        assert out.pixel_data.shape == (48, 80)

    def test_annotation_serialization(self):
        """Annotations can round-trip through to_dict / from_dict."""
        algo = _make_algo()
        img = InMemoryImage(array=np.random.rand(64, 64).astype(np.float32))
        out = img.clone()
        algo(img, out)

        for ann in out.annotations:
            d = ann.to_dict()
            restored = Annotation.from_dict(d)
            assert restored.label == ann.label
            assert restored.shape == ann.shape
            assert restored.metadata == ann.metadata

    def test_12bit_input(self):
        """12-bit range input is auto-normalized."""
        algo = _make_algo()
        arr = np.random.rand(64, 64).astype(np.float32) * 4095
        arr[10:16, 10:16] = 3800
        img = InMemoryImage(array=arr)
        out = img.clone()
        algo(img, out)

        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_constructor_requires_model_or_checkpoint(self):
        with pytest.raises(ValueError, match="checkpoint_path or model"):
            DeepSegmentationAlgorithm(device="cpu")


# =========================================================================
# Tests — Real checkpoint from medical-image-baseline/results
# =========================================================================

# Pick one non-CLAHE and one CLAHE checkpoint
_CKPT_NO_CLAHE = RESULTS_DIR / "unet_bce_dice_128_inbreast" / "best_model.pt"
_CKPT_CLAHE = RESULTS_DIR / "unet_bce_dice_128_inbreast_clahe" / "best_model.pt"


@pytest.mark.skipif(
    not _CKPT_NO_CLAHE.exists(),
    reason="Trained checkpoint not available",
)
class TestRealCheckpoint:
    """Load real trained models from medical-image-baseline/results and
    run inference.  The checkpoint carries its own config (patch_size,
    clahe, threshold, etc.) so the algorithm auto-configures from it —
    just like MammogramPredictor in the baseline does."""

    def test_load_checkpoint_auto_config(self):
        """Checkpoint config is correctly read: patch_size, stride, clahe."""
        algo = DeepSegmentationAlgorithm(
            checkpoint_path=str(_CKPT_NO_CLAHE), device="cpu"
        )

        # Config says patch_size=128, stride_ratio=0.5, clahe=False
        assert algo.patch_size == 128
        assert algo.stride == 64  # 128 * 0.5
        assert algo.threshold == 0.5
        assert algo.min_lesion_area == 4
        assert algo.use_clahe is False

    def test_load_checkpoint_clahe(self):
        """CLAHE variant checkpoint sets use_clahe=True."""
        algo = DeepSegmentationAlgorithm(checkpoint_path=str(_CKPT_CLAHE), device="cpu")

        assert algo.use_clahe is True
        assert algo.patch_size == 128

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_dicom_inference_no_clahe(self, dicom_image):
        """Real U-Net (no CLAHE) on real DICOM.

        patch_size=128 is read from the checkpoint — each sliding-window
        patch is (1, 1, 128, 128), matching training resolution."""
        algo = DeepSegmentationAlgorithm(
            checkpoint_path=str(_CKPT_NO_CLAHE), device="cpu"
        )
        assert algo.patch_size == 128

        img = dicom_image
        img.pixel_data = img.pixel_data.float()
        if img.pixel_data.max() > 1.0:
            img.pixel_data = img.pixel_data / img.pixel_data.max()

        out = img.clone()
        algo(img, out)
        image_output = (
            out.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((dicom_image.height, dicom_image.width))
        )
        # Binary mask
        assert out.pixel_data.shape == img.pixel_data.shape
        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

        # Probability map
        assert algo.probability_map is not None
        assert algo.probability_map.shape == img.pixel_data.shape
        assert float(algo.probability_map.min()) >= 0.0
        assert float(algo.probability_map.max()) <= 1.0

        # Annotations are well-formed
        assert out.annotations is not None
        assert algo.lesion_count == len(out.annotations)
        for ann in out.annotations:
            assert ann.label == "microcalcification"
            assert ann.metadata["area"] >= algo.min_lesion_area
            assert 0.0 <= ann.metadata["confidence"] <= 1.0
            bbox = ann.get_bounding_box()
            assert bbox[0] <= bbox[2]
            assert bbox[1] <= bbox[3]

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_dicom_inference_clahe(self, dicom_image):
        """Real U-Net (CLAHE) on real DICOM — CLAHE applied before patching."""
        algo = DeepSegmentationAlgorithm(checkpoint_path=str(_CKPT_CLAHE), device="cpu")
        assert algo.use_clahe is True
        assert algo.patch_size == 128

        img = dicom_image
        img.pixel_data = img.pixel_data.float()
        if img.pixel_data.max() > 1.0:
            img.pixel_data = img.pixel_data / img.pixel_data.max()

        out = img.clone()
        algo(img, out)

        image_output = (
            out.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((dicom_image.height, dicom_image.width))
        )
        assert out.pixel_data.shape == img.pixel_data.shape
        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

        assert out.annotations is not None
        assert algo.lesion_count == len(out.annotations)
