import copy

import numpy as np
import pytest
import torch

from medical_image.algorithms.custom_algorithm import CustomAlgorithm
from medical_image.process.threshold import Threshold
from medical_image.tests.mock_sample import mock_dicom_image, mock_sauvola_threshold


class TestDicom:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_dicom_image(self, dicom_image):
        dicom_image.to_png()
        assert dicom_image.pixel_data is not None
        assert dicom_image.width == 512
        assert dicom_image.height == 512

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_otsu_threshold(self, dicom_image):
        # Apply Otsu's threshold to the mock DICOM image
        output = copy.deepcopy(dicom_image)
        Threshold.otsu_threshold(dicom_image, output)

        # Check that the pixel data has been modified
        assert not np.array_equal(
            dicom_image.pixel_data.cpu().numpy(), output.pixel_data.cpu().numpy()
        )

        assert torch.all((output.pixel_data == 0) | (output.pixel_data == 255))

    @pytest.mark.parametrize("dicom_image, window_size, k", mock_sauvola_threshold())
    def test_sauvola_threshold(self, dicom_image, window_size, k):
        output = copy.deepcopy(dicom_image)
        Threshold.sauvola_threshold(dicom_image, output, window_size, k)

        # Check that pixel data has changed
        assert not torch.equal(dicom_image.pixel_data, output.pixel_data)

        # Check that output is binary (0 or 255)
        out = output.pixel_data
        assert torch.all((out == 0) | (out == 255))

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_custom_algorithm(self, dicom_image):
        # Convert input pixel data to torch.Tensor if not already
        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        # Prepare output image
        output = copy.deepcopy(dicom_image)
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        # Apply the custom algorithm
        algorithm = CustomAlgorithm()
        algorithm(image=dicom_image, output=output)

        # Check that the pixel data has been modified
        assert not torch.allclose(
            dicom_image.pixel_data.float(), output.pixel_data.float()
        )

        # Check that the output is a binary image (0 or 255)
        unique_vals = torch.unique(output.pixel_data)
        assert set(unique_vals.tolist()).issubset({0, 255})
