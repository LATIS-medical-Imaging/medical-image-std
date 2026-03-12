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
from medical_image.tests.mock_sample import (
    mock_synthetic_image,
    mock_12bit_image,
    mock_tophat_radius,
    mock_kmeans_k,
    mock_fcm_c,
    mock_pfcm_params,
    mock_roi_center,
    mock_dicom_image,
)
from medical_image.utils.image_utils import MathematicalOperations


# ===========================================================================
# MathematicalOperations
# ===========================================================================


class TestMathematicalOperations:
    def test_euclidean_distance_sq_shape(self):
        Z = torch.rand(100, 3)
        V = torch.rand(5, 3)
        D2 = MathematicalOperations.euclidean_distance_sq(Z, V)
        assert D2.shape == (5, 100)

    def test_euclidean_distance_sq_values(self):
        Z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        V = torch.tensor([[0.0, 0.0]])
        D2 = MathematicalOperations.euclidean_distance_sq(Z, V)
        assert D2.shape == (1, 2)
        torch.testing.assert_close(D2, torch.tensor([[1.0, 1.0]]))

    def test_euclidean_distance_sq_zero(self):
        Z = torch.tensor([[3.0, 4.0]])
        V = torch.tensor([[3.0, 4.0]])
        D2 = MathematicalOperations.euclidean_distance_sq(Z, V)
        assert float(D2[0, 0]) == pytest.approx(0.0)

    @pytest.mark.parametrize("image_12bit", mock_12bit_image())
    def test_normalize_12bit(self, image_12bit):
        out = image_12bit.clone()
        MathematicalOperations.normalize_12bit(image_12bit, out)
        assert float(out.pixel_data.min()) >= 0.0
        assert float(out.pixel_data.max()) <= 1.0


# ===========================================================================
# MorphologyOperations
# ===========================================================================


class TestMorphologyOperations:
    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_erosion_reduces_values(self, image):
        out = image.clone()
        MorphologyOperations.erosion(image, out, radius=2)
        assert float(out.pixel_data.max()) <= float(image.pixel_data.max()) + 1e-5

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_dilation_increases_values(self, image):
        out = image.clone()
        MorphologyOperations.dilation(image, out, radius=2)
        assert float(out.pixel_data.max()) >= float(image.pixel_data.max()) - 1e-5

    @pytest.mark.parametrize("radius", mock_tophat_radius())
    def test_white_top_hat_non_negative(self, radius):
        img = mock_synthetic_image()[0]
        out = img.clone()
        MorphologyOperations.white_top_hat(img, out, radius=radius[0])
        assert float(out.pixel_data.min()) >= 0.0

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_white_top_hat_highlights_bright_spots(self, image):
        out = image.clone()
        MorphologyOperations.white_top_hat(image, out, radius=4)
        assert float(out.pixel_data[20:23, 20:23].mean()) > float(out.pixel_data.mean())

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_white_top_hat_shape_preserved(self, image):
        out = image.clone()
        MorphologyOperations.white_top_hat(image, out, radius=3)
        assert out.pixel_data.shape == image.pixel_data.shape
        assert out.width == image.width
        assert out.height == image.height

    def test_disk_footprint(self):
        fp = MorphologyOperations._disk_footprint(3)
        assert fp.shape == (7, 7)
        assert float(fp[3, 3]) == 1.0
        assert float(fp[0, 0]) == 0.0


# ===========================================================================
# RegionOfInterest
# ===========================================================================


class TestRegionOfInterest:
    @pytest.mark.parametrize("cx, cy, half_size", mock_roi_center())
    def test_from_center(self, cx, cy, half_size):
        img = DicomImage(array=np.random.rand(100, 100).astype(np.float32))
        roi = RegionOfInterest.from_center(img, cx=cx, cy=cy, half_size=half_size)
        loaded = roi.load()
        assert loaded.pixel_data.shape[0] <= 2 * half_size + 1
        assert loaded.pixel_data.shape[1] <= 2 * half_size + 1

    @pytest.mark.parametrize("image_12bit", mock_12bit_image())
    def test_normalize(self, image_12bit):
        normalized = RegionOfInterest.normalize(image_12bit, divisor=4095.0)
        assert float(normalized.pixel_data.min()) >= 0.0
        assert float(normalized.pixel_data.max()) <= 1.0


# ===========================================================================
# TopHatAlgorithm
# ===========================================================================


class TestTopHatAlgorithm:
    @pytest.mark.parametrize("radius", mock_tophat_radius())
    def test_apply(self, radius):
        img = mock_synthetic_image()[0]
        out = img.clone()
        algo = TopHatAlgorithm(radius=radius[0], device="cpu")
        algo(img, out)
        assert out.pixel_data is not None
        assert out.pixel_data.shape == img.pixel_data.shape
        assert float(out.pixel_data.min()) >= 0.0

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_highlights_mc(self, image):
        out = image.clone()
        algo = TopHatAlgorithm(radius=3, device="cpu")
        algo(image, out)
        mc_mean = float(out.pixel_data[20:23, 20:23].mean())
        bg_mean = float(out.pixel_data[30:35, 30:35].mean())
        assert mc_mean > bg_mean


# ===========================================================================
# KMeansAlgorithm
# ===========================================================================


class TestKMeansAlgorithm:
    @pytest.mark.parametrize("k", mock_kmeans_k())
    def test_apply_basic(self, k):
        img = mock_synthetic_image()[0]
        out = img.clone()
        km = KMeansAlgorithm(k=k[0], device="cpu")
        km(img, out)

        assert out.pixel_data is not None
        assert out.pixel_data.shape == img.pixel_data.shape
        assert km.centroids.shape == (k[0], 1)
        assert km.labels.shape == img.pixel_data.shape
        assert km.converged or km.n_iter == km.max_iter

    @pytest.mark.parametrize("k", mock_kmeans_k())
    def test_binary_output(self, k):
        img = mock_synthetic_image()[0]
        out = img.clone()
        km = KMeansAlgorithm(k=k[0], device="cpu")
        km(img, out)

        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    @pytest.mark.parametrize("k", mock_kmeans_k())
    def test_stats_populated(self, k):
        img = mock_synthetic_image()[0]
        out = img.clone()
        km = KMeansAlgorithm(k=k[0], device="cpu")
        km(img, out)

        assert len(km.stats) == k[0]
        total = sum(s["pixels"] for s in km.stats)
        assert total == 64 * 64
        assert any(s["is_mc"] for s in km.stats)

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_mc_in_brightest_cluster(self, image):
        out = image.clone()
        km = KMeansAlgorithm(k=3, device="cpu")
        km(image, out)

        assert float(out.pixel_data.sum()) > 0
        assert float(out.pixel_data[20:23, 20:23].sum()) > 0

    @pytest.mark.parametrize("k", mock_kmeans_k())
    def test_quantized_range(self, k):
        img = mock_synthetic_image()[0]
        out = img.clone()
        km = KMeansAlgorithm(k=k[0], device="cpu")
        km(img, out)

        assert float(km.quantized.min()) >= 0.0
        assert float(km.quantized.max()) <= 1.0


# ===========================================================================
# FCMAlgorithm
# ===========================================================================


class TestFCMAlgorithm:
    @pytest.mark.parametrize("c", mock_fcm_c())
    def test_apply_basic(self, c):
        img = mock_synthetic_image()[0]
        out = img.clone()
        fcm = FCMAlgorithm(c=c[0], device="cpu")
        fcm(img, out)

        assert out.pixel_data is not None
        assert out.pixel_data.shape == img.pixel_data.shape
        assert fcm.centroids.shape == (c[0], 1)
        assert fcm.converged or fcm.n_iter == fcm.max_iter

    @pytest.mark.parametrize("c", mock_fcm_c())
    def test_binary_output(self, c):
        img = mock_synthetic_image()[0]
        out = img.clone()
        fcm = FCMAlgorithm(c=c[0], device="cpu")
        fcm(img, out)

        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_mc_in_brightest_cluster(self, image):
        out = image.clone()
        fcm = FCMAlgorithm(c=3, device="cpu")
        fcm(image, out)

        assert float(out.pixel_data.sum()) > 0
        assert float(out.pixel_data[20:23, 20:23].sum()) > 0

    @pytest.mark.parametrize("c", mock_fcm_c())
    def test_quantized_range(self, c):
        img = mock_synthetic_image()[0]
        out = img.clone()
        fcm = FCMAlgorithm(c=c[0], device="cpu")
        fcm(img, out)

        assert float(fcm.quantized.min()) >= 0.0
        assert float(fcm.quantized.max()) <= 1.0

    @pytest.mark.parametrize("c", mock_fcm_c())
    def test_stats_populated(self, c):
        img = mock_synthetic_image()[0]
        out = img.clone()
        fcm = FCMAlgorithm(c=c[0], device="cpu")
        fcm(img, out)

        assert len(fcm.stats) == c[0]
        total = sum(s["pixels"] for s in fcm.stats)
        assert total == 64 * 64
        assert any(s["is_mc"] for s in fcm.stats)


# ===========================================================================
# PFCMAlgorithm
# ===========================================================================


class TestPFCMAlgorithm:
    @pytest.mark.parametrize("c, tau", mock_pfcm_params())
    def test_apply_basic(self, c, tau):
        img = mock_synthetic_image()[0]
        out = img.clone()
        pfcm = PFCMAlgorithm(c=c, tau=tau, device="cpu")
        pfcm(img, out)

        assert out.pixel_data is not None
        assert out.pixel_data.shape == img.pixel_data.shape
        assert pfcm.T_max_map is not None
        assert pfcm.T_max_map.shape == img.pixel_data.shape

    @pytest.mark.parametrize("c, tau", mock_pfcm_params())
    def test_binary_output(self, c, tau):
        img = mock_synthetic_image()[0]
        out = img.clone()
        pfcm = PFCMAlgorithm(c=c, tau=tau, device="cpu")
        pfcm(img, out)

        unique = torch.unique(out.pixel_data)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    @pytest.mark.parametrize("c, tau", mock_pfcm_params())
    def test_typicality_range(self, c, tau):
        img = mock_synthetic_image()[0]
        out = img.clone()
        pfcm = PFCMAlgorithm(c=c, tau=tau, device="cpu")
        pfcm(img, out)

        assert float(pfcm.T_max_map.min()) >= 0.0
        assert float(pfcm.T_max_map.max()) <= 1.0

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_atypical_detection(self, image):
        out = image.clone()
        pfcm = PFCMAlgorithm(c=2, tau=0.5, device="cpu")
        pfcm(image, out)
        assert float(out.pixel_data.sum()) > 0

    @pytest.mark.parametrize("c, tau", mock_pfcm_params())
    def test_centroids_shape(self, c, tau):
        img = mock_synthetic_image()[0]
        out = img.clone()
        pfcm = PFCMAlgorithm(c=c, tau=tau, device="cpu")
        pfcm(img, out)

        assert pfcm.centroids.shape == (c, 1)
        assert pfcm.gamma.shape == (c,)


# ===========================================================================
# Integration: Full Pipeline
# ===========================================================================


class TestFullPipeline:
    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_top_hat_then_fcm(self, image):
        """TopHat -> FCM pipeline."""
        th_out = image.clone()
        TopHatAlgorithm(radius=3, device="cpu")(image, th_out)

        fcm_out = th_out.clone()
        fcm = FCMAlgorithm(c=3, device="cpu")
        fcm(th_out, fcm_out)

        assert float(fcm_out.pixel_data.sum()) > 0
        assert fcm.converged or fcm.n_iter == fcm.max_iter

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_top_hat_then_kmeans(self, image):
        """TopHat -> KMeans pipeline."""
        th_out = image.clone()
        TopHatAlgorithm(radius=3, device="cpu")(image, th_out)

        km_out = th_out.clone()
        km = KMeansAlgorithm(k=3, device="cpu")
        km(th_out, km_out)

        assert float(km_out.pixel_data.sum()) > 0
        assert km.converged or km.n_iter == km.max_iter

    @pytest.mark.parametrize("image", mock_synthetic_image())
    def test_top_hat_then_pfcm(self, image):
        """TopHat -> PFCM pipeline."""
        th_out = image.clone()
        TopHatAlgorithm(radius=3, device="cpu")(image, th_out)

        pfcm_out = th_out.clone()
        pfcm = PFCMAlgorithm(c=2, tau=0.1, device="cpu")
        pfcm(th_out, pfcm_out)

        assert pfcm_out.pixel_data is not None
        assert pfcm.T_max_map is not None

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_roi_then_pipeline(self, dicom_image):
        """ROI extraction -> normalize -> TopHat -> FCM on real DICOM."""
        roi = RegionOfInterest.from_center(dicom_image, cx=1250, cy=2000, half_size=127)
        roi_img = roi.load()
        RegionOfInterest.normalize(roi_img, divisor=4095.0)

        th_out = roi_img.clone()
        TopHatAlgorithm(radius=3, device="cpu")(roi_img, th_out)

        fcm_out = th_out.clone()
        fcm = FCMAlgorithm(c=3, device="cpu")
        fcm(th_out, fcm_out)

        assert float(fcm_out.pixel_data.sum()) > 0
