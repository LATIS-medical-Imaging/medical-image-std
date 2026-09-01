"""
Unit tests for DeepSegmentationAlgorithm remote model support.

Tests cover:
  - Model discovery from server
  - Model download and caching
  - Inference on DICOM images with remote models
  - CLAHE model configuration
  - Probability map output
  - Configurable server URL
"""

import numpy as np
import pytest
import torch

from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.tests.mock_sample import mock_dicom_image


@pytest.mark.slow
class TestDeepSegmentationRemote:

    # --- Model Discovery ---

    def test_list_available_models(self):
        """list_available_models() returns a non-empty list of model metadata dicts."""
        models = DeepSegmentationAlgorithm.list_available_models()
        assert isinstance(models, list)
        assert len(models) > 0
        for model_info in models:
            assert "name" in model_info
            assert "architecture" in model_info
            assert "patch_size" in model_info
            assert "uses_clahe" in model_info
            assert "url" in model_info

        print()

    def test_list_available_models_custom_url(self):
        """list_available_models() accepts a custom server URL."""
        models = DeepSegmentationAlgorithm.list_available_models(
            server_url="http://mcdmodels.ptm.tn:555/"
        )
        assert len(models) > 0

    def test_known_models_present(self):
        """The 10 known models are discoverable."""
        models = DeepSegmentationAlgorithm.list_available_models()
        names = {m["name"] for m in models}
        expected = {
            "attention_unet_bce_dice_256_inbreast_clahe",
            "attention_unet_focal_dice_64_inbreast",
            "unet_bce_dice_64_inbreast_clahe",
            "unet_focal_dice_256_inbreast",
            "unet_topk_bce_dice_128_inbreast",
            "unetpp_bce_dice_256_inbreast",
            "unetpp_bce_dice_32_inbreast",
            "unetpp_focal_dice_128_inbreast",
            "unetpp_topk_bce_dice_32_inbreast_clahe",
            "unetpp_topk_bce_dice_64_inbreast",
        }
        assert expected.issubset(names), f"Missing models: {expected - names}"

    # --- Model Download & Loading ---

    @pytest.mark.parametrize(
        "model_name",
        [
            "unetpp_bce_dice_32_inbreast",
            "unet_bce_dice_64_inbreast_clahe",
        ],
    )
    def test_from_pretrained(self, model_name):
        """from_pretrained() downloads, caches, and loads a model."""
        algo = DeepSegmentationAlgorithm.from_pretrained(model_name, device="cuda")
        assert algo.model is not None
        assert isinstance(algo.patch_size, int)
        assert algo.patch_size > 0

    # --- Inference on DICOM ---

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_inference_on_dicom(self, dicom_image):
        """Run inference on a real DICOM image -> output is binary numpy mask."""
        algo = DeepSegmentationAlgorithm.from_pretrained(
            "unetpp_bce_dice_32_inbreast", device="cuda"
        )

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        algo(image=dicom_image, output=output)

        mask_np = output.pixel_data.detach().cpu().numpy()

        assert isinstance(mask_np, np.ndarray)
        assert mask_np.shape == (dicom_image.height, dicom_image.width)
        unique_vals = np.unique(mask_np)
        assert set(unique_vals).issubset({0.0, 1.0})

        assert output.annotations is not None
        assert isinstance(output.annotations, list)

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_inference_clahe_model(self, dicom_image):
        """CLAHE model correctly applies CLAHE before inference."""
        algo = DeepSegmentationAlgorithm.from_pretrained(
            "unet_bce_dice_64_inbreast_clahe", device="cuda"
        )
        assert algo.use_clahe is True

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        algo(image=dicom_image, output=output)

        mask_np = output.pixel_data.detach().cpu().numpy()
        assert isinstance(mask_np, np.ndarray)
        unique_vals = np.unique(mask_np)
        assert set(unique_vals).issubset({0.0, 1.0})

    # --- Probability Map ---

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_probability_map(self, dicom_image):
        """Probability map is populated after inference."""
        algo = DeepSegmentationAlgorithm.from_pretrained(
            "unetpp_bce_dice_32_inbreast", device="cuda"
        )

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        algo(image=dicom_image, output=output)

        assert algo.probability_map is not None
        prob_np = algo.probability_map.numpy()
        assert isinstance(prob_np, np.ndarray)
        assert 0.0 <= prob_np.min()
        assert prob_np.max() <= 1.0

    # --- Configurable URL ---

    def test_configurable_server_url(self):
        """Server URL can be overridden in from_pretrained()."""
        algo = DeepSegmentationAlgorithm.from_pretrained(
            "unetpp_bce_dice_32_inbreast",
            server_url="http://mcdmodels.ptm.tn:555/",
            device="cpu",
        )
        assert algo.model is not None

    # --- Model Info ---

    def test_model_info(self):
        """model_info property returns parsed metadata."""
        algo = DeepSegmentationAlgorithm.from_pretrained(
            "unetpp_bce_dice_32_inbreast", device="cuda"
        )
        info = algo.model_info
        assert info is not None
        assert info["architecture"] == "unetpp"
        assert info["loss"] == "bce_dice"
        assert info["patch_size"] == 32
        assert info["dataset"] == "inbreast"
        assert info["uses_clahe"] is False
