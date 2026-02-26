"""
Unit tests for micro-calcification segmentation algorithms.

Tests use synthetic data (no DICOM dependency) to validate:
  - MathematicalOperations.euclidean_distance_sq
  - MorphologyOperations.white_top_hat / erosion / dilation
  - RegionOfInterest.from_center / normalize
  - TopHatAlgorithm
  - KMeansAlgorithm
  - FCMAlgorithm
  - PFCMAlgorithm
  - Full pipeline integration
"""

import copy

import numpy as np
import pytest
import torch

from medical_image.algorithms.fcm import FCMAlgorithm
from medical_image.algorithms.kmeans import KMeansAlgorithm
from medical_image.algorithms.pfcm import PFCMAlgorithm
from medical_image.algorithms.top_hat import TopHatAlgorithm
from medical_image.data.dicom_image import DicomImage
from medical_image.data.region_of_interest import RegionOfInterest
from medical_image.process.morphology import MorphologyOperations
from medical_image.tests.mock_sample import mock_dicom_image
from medical_image.utils.image_utils import MathematicalOperations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_image(size: int = 64) -> DicomImage:
    """
    Create a synthetic 2D image with a few bright spots (simulating MCs).
    Values in [0, 1] range (already normalized).
    """
    arr = np.random.RandomState(42).rand(size, size).astype(np.float32) * 0.3
    # Add bright spots (microcalcifications)
    arr[20:23, 20:23] = 0.9
    arr[40:42, 45:47] = 0.85
    arr[10:12, 50:52] = 0.95
    return DicomImage(array=arr)


def _make_12bit_image(size: int = 64) -> DicomImage:
    """Create a synthetic 12-bit image (values 0–4095)."""
    arr = np.random.RandomState(42).rand(size, size).astype(np.float32) * 1500
    arr[20:23, 20:23] = 3800
    arr[40:42, 45:47] = 3600
    return DicomImage(array=arr)


# ===========================================================================
# MathematicalOperations
# ===========================================================================


class TestMathematicalOperations:
    def test_euclidean_distance_sq_shape(self):
        Z = torch.rand(100, 3)  # 100 samples, 3 features
        V = torch.rand(5, 3)   # 5 centroids
        D2 = MathematicalOperations.euclidean_distance_sq(Z, V)
        assert D2.shape == (5, 100)

    def test_euclidean_distance_sq_values(self):
        Z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 2 samples
        V = torch.tensor([[0.0, 0.0]])                # 1 centroid
        D2 = MathematicalOperations.euclidean_distance_sq(Z, V)
        # d²(Z[0], V[0]) = 1² + 0² = 1
        # d²(Z[1], V[0]) = 0² + 1² = 1
        assert D2.shape == (1, 2)
        torch.testing.assert_close(D2, torch.tensor([[1.0, 1.0]]))

    def test_euclidean_distance_sq_zero(self):
        Z = torch.tensor([[3.0, 4.0]])
        V = torch.tensor([[3.0, 4.0]])
        D2 = MathematicalOperations.euclidean_distance_sq(Z, V)
        assert float(D2[0, 0]) == pytest.approx(0.0)

    def test_normalize_12bit(self):
        img = _make_12bit_image()
        out = copy.deepcopy(img)
        MathematicalOperations.normalize_12bit(img, out)
        assert float(out.pixel_data.min()) >= 0.0
        assert float(out.pixel_data.max()) <= 1.0


# ===========================================================================
# MorphologyOperations
# ===========================================================================


class TestMorphologyOperations:
    def test_erosion_reduces_values(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        MorphologyOperations.erosion(img, out, radius=2)
        # Erosion should generally reduce or maintain values
        assert float(out.pixel_data.max()) <= float(img.pixel_data.max()) + 1e-5

    def test_dilation_increases_values(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        MorphologyOperations.dilation(img, out, radius=2)
        # Dilation should generally increase or maintain values
        assert float(out.pixel_data.max()) >= float(img.pixel_data.max()) - 1e-5

    def test_white_top_hat_non_negative(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        MorphologyOperations.white_top_hat(img, out, radius=4)
        assert float(out.pixel_data.min()) >= 0.0

    def test_white_top_hat_highlights_bright_spots(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        MorphologyOperations.white_top_hat(img, out, radius=4)
        # The bright spots should have high top-hat values
        assert float(out.pixel_data[20:23, 20:23].mean()) > float(out.pixel_data.mean())

    def test_white_top_hat_shape_preserved(self):
        img = _make_synthetic_image(size=32)
        out = copy.deepcopy(img)
        MorphologyOperations.white_top_hat(img, out, radius=3)
        assert out.pixel_data.shape == (32, 32)
        assert out.width == 32
        assert out.height == 32

    def test_disk_footprint(self):
        fp = MorphologyOperations._disk_footprint(3)
        assert fp.shape == (7, 7)
        # Center must be 1
        assert float(fp[3, 3]) == 1.0
        # Corners of 7×7 should be 0 for disk of radius 3
        assert float(fp[0, 0]) == 0.0


# ===========================================================================
# RegionOfInterest
# ===========================================================================


class TestRegionOfInterest:
    def test_from_center(self):
        img = _make_synthetic_image(size=100)
        roi = RegionOfInterest.from_center(img, cx=50, cy=50, half_size=10)
        loaded = roi.load()
        assert loaded.pixel_data.shape == (21, 21)

    def test_from_center_clamped(self):
        img = _make_synthetic_image(size=64)
        # Center near edge — should clamp
        roi = RegionOfInterest.from_center(img, cx=2, cy=2, half_size=10)
        loaded = roi.load()
        assert loaded.pixel_data.shape[0] <= 21
        assert loaded.pixel_data.shape[1] <= 21

    def test_normalize(self):
        img = _make_12bit_image()
        normalized = RegionOfInterest.normalize(img, divisor=4095.0)
        assert float(normalized.pixel_data.min()) >= 0.0
        assert float(normalized.pixel_data.max()) <= 1.0


# ===========================================================================
# TopHatAlgorithm
# ===========================================================================


class TestTopHatAlgorithm:
    def test_apply(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        algo = TopHatAlgorithm(radius=4, device="cpu")
        algo(img, out)
        assert out.pixel_data is not None
        assert out.pixel_data.shape == img.pixel_data.shape
        assert float(out.pixel_data.min()) >= 0.0

    def test_highlights_mc(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        algo = TopHatAlgorithm(radius=3, device="cpu")
        algo(img, out)
        # MC region should be brighter than background in top-hat
        mc_mean = float(out.pixel_data[20:23, 20:23].mean())
        bg_mean = float(out.pixel_data[30:35, 30:35].mean())
        assert mc_mean > bg_mean


# ===========================================================================
# KMeansAlgorithm
# ===========================================================================


class TestKMeansAlgorithm:
    def test_apply_basic(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        km = KMeansAlgorithm(k=2, device="cpu")
        km(img, out)

        assert out.pixel_data is not None
        assert out.pixel_data.shape == img.pixel_data.shape
        assert km.centroids.shape == (2, 1)
        assert km.labels.shape == img.pixel_data.shape
        assert km.converged or km.n_iter == km.max_iter

    def test_binary_output(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        km = KMeansAlgorithm(k=2, device="cpu")
        km(img, out)

        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_mc_in_brightest_cluster(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        km = KMeansAlgorithm(k=3, device="cpu")
        km(img, out)

        assert float(out.pixel_data.sum()) > 0
        assert float(out.pixel_data[20:23, 20:23].sum()) > 0

    def test_quantized_range(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        km = KMeansAlgorithm(k=4, device="cpu")
        km(img, out)

        assert float(km.quantized.min()) >= 0.0
        assert float(km.quantized.max()) <= 1.0

    def test_stats_populated(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        km = KMeansAlgorithm(k=3, device="cpu")
        km(img, out)

        assert len(km.stats) == 3
        total_pixels = sum(s["pixels"] for s in km.stats)
        assert total_pixels == 64 * 64
        assert any(s["is_mc"] for s in km.stats)


# ===========================================================================
# FCMAlgorithm
# ===========================================================================


class TestFCMAlgorithm:
    def test_apply_basic(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        fcm = FCMAlgorithm(c=2, device="cpu")
        fcm(img, out)

        assert out.pixel_data is not None
        assert out.pixel_data.shape == img.pixel_data.shape
        assert fcm.centroids.shape == (2, 1)
        assert fcm.labels.shape == img.pixel_data.shape
        assert fcm.converged or fcm.n_iter == fcm.max_iter

    def test_binary_output(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        fcm = FCMAlgorithm(c=2, device="cpu")
        fcm(img, out)

        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_mc_in_brightest_cluster(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        fcm = FCMAlgorithm(c=3, device="cpu")
        fcm(img, out)

        # At least one MC pixel should be detected
        assert float(out.pixel_data.sum()) > 0
        # The MC region should be in the MC cluster
        assert float(out.pixel_data[20:23, 20:23].sum()) > 0

    def test_quantized_range(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        fcm = FCMAlgorithm(c=4, device="cpu")
        fcm(img, out)

        assert float(fcm.quantized.min()) >= 0.0
        assert float(fcm.quantized.max()) <= 1.0

    def test_stats_populated(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        fcm = FCMAlgorithm(c=3, device="cpu")
        fcm(img, out)

        assert len(fcm.stats) == 3
        total_pixels = sum(s["pixels"] for s in fcm.stats)
        assert total_pixels == 64 * 64  # all pixels accounted for
        assert any(s["is_mc"] for s in fcm.stats)


# ===========================================================================
# PFCMAlgorithm
# ===========================================================================


class TestPFCMAlgorithm:
    def test_apply_basic(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        pfcm = PFCMAlgorithm(c=2, tau=0.04, device="cpu")
        pfcm(img, out)

        assert out.pixel_data is not None
        assert out.pixel_data.shape == img.pixel_data.shape
        assert pfcm.T_max_map is not None
        assert pfcm.T_max_map.shape == img.pixel_data.shape

    def test_binary_output(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        pfcm = PFCMAlgorithm(c=2, tau=0.04, device="cpu")
        pfcm(img, out)

        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_typicality_range(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        pfcm = PFCMAlgorithm(c=2, tau=0.1, device="cpu")
        pfcm(img, out)

        T_max = pfcm.T_max_map
        assert float(T_max.min()) >= 0.0
        assert float(T_max.max()) <= 1.0

    def test_atypical_detection(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        pfcm = PFCMAlgorithm(c=2, tau=0.5, device="cpu")
        pfcm(img, out)

        # With high tau, more pixels should be detected
        n_mc = float(out.pixel_data.sum())
        assert n_mc > 0  # at least some atypical pixels

    def test_centroids_shape(self):
        img = _make_synthetic_image()
        out = copy.deepcopy(img)
        pfcm = PFCMAlgorithm(c=2, device="cpu")
        pfcm(img, out)

        assert pfcm.centroids.shape == (2, 1)
        assert pfcm.gamma.shape == (2,)


# ===========================================================================
# Integration: Full Pipeline
# ===========================================================================


class TestFullPipeline:
    def test_top_hat_then_fcm(self):
        """TopHat → FCM: typical MC detection pipeline."""
        img = _make_synthetic_image()

        # Step 1: Top-Hat
        th_out = copy.deepcopy(img)
        top_hat = TopHatAlgorithm(radius=3, device="cpu")
        top_hat(img, th_out)

        # Step 2: FCM on top-hat result
        fcm_out = copy.deepcopy(th_out)
        fcm = FCMAlgorithm(c=3, device="cpu")
        fcm(th_out, fcm_out)

        assert float(fcm_out.pixel_data.sum()) > 0
        assert fcm.converged or fcm.n_iter == fcm.max_iter

    def test_top_hat_then_kmeans(self):
        """TopHat → KMeans: typical MC detection pipeline."""
        img = _make_synthetic_image()

        th_out = copy.deepcopy(img)
        top_hat = TopHatAlgorithm(radius=3, device="cpu")
        top_hat(img, th_out)

        km_out = copy.deepcopy(th_out)
        km = KMeansAlgorithm(k=3, device="cpu")
        km(th_out, km_out)

        assert float(km_out.pixel_data.sum()) > 0
        assert km.converged or km.n_iter == km.max_iter

    def test_top_hat_then_pfcm(self):
        """TopHat → PFCM: full pipeline with atypicality detection."""
        img = _make_synthetic_image()

        # Step 1: Top-Hat
        th_out = copy.deepcopy(img)
        top_hat = TopHatAlgorithm(radius=3, device="cpu")
        top_hat(img, th_out)

        # Step 2: PFCM on top-hat result
        pfcm_out = copy.deepcopy(th_out)
        pfcm = PFCMAlgorithm(c=2, tau=0.1, device="cpu")
        pfcm(th_out, pfcm_out)

        assert pfcm_out.pixel_data is not None
        assert pfcm.T_max_map is not None
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_roi_then_pipeline(self, dicom_image):
        """ROI extraction → normalize → TopHat → FCM."""
        # Simulate a large image
        # big_arr = np.random.RandomState(42).rand(256, 256).astype(np.float32) * 2000
        # big_arr[100:105, 100:105] = 3800
        big_img = dicom_image

        # Extract ROI
        roi = RegionOfInterest.from_center(big_img, cx=1250, cy=2000, half_size=127)
        roi_img = roi.load()

        # Normalize
        RegionOfInterest.normalize(roi_img, divisor=4095.0)

        roi_out_numpy = roi_img.pixel_data.cpu().numpy()

        # TopHat
        th_out = copy.deepcopy(roi_img)
        top_hat = TopHatAlgorithm(radius=3, device="cpu")
        top_hat(roi_img, th_out)
        th_out_numpy = th_out.pixel_data.cpu().numpy()


        # FCM
        fcm_out = copy.deepcopy(th_out)
        fcm = FCMAlgorithm(c=3, device="cpu")
        fcm(th_out, fcm_out)
        fcm_out_numpy = fcm_out.pixel_data.cpu().numpy()

        # PFCM
        pfcm_out = copy.deepcopy(th_out)
        pfcm = PFCMAlgorithm(c=3, tau=0.1 ,device="cpu")
        pfcm(th_out, pfcm_out)
        pfcm_out_numpy = pfcm_out.pixel_data.cpu().numpy()

        # KMEANS
        kmean_out = copy.deepcopy(th_out)
        kmean = KMeansAlgorithm( k = 5,device="cpu")
        kmean(th_out, kmean_out)
        kmean_out_numpy = kmean_out.pixel_data.cpu().numpy()

        assert float(fcm_out.pixel_data.sum()) > 0
