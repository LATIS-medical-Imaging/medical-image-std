import numpy as np
import pytest

import os

from log_manager import logger
from medical_image.tests.mock_sample import mock_dicom_image, mock_sauvola_threshold
from medical_image.process.threshold import Threshold


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
        output = dicom_image
        # TODO: FIX THIS: it seems output is not set after update (or applying the threshold)
        Threshold.otsu_threshold(dicom_image, output)
        # Check that the pixel data has been modified
        assert not np.array_equal(dicom_image.pixel_data, output.pixel_data)

        # Check that the output is a binary image (0 or 255)
        assert np.all(np.logical_or(output.pixel_data == 0, output.pixel_data == 255))

    @pytest.mark.parametrize("dicom_image, window_size, k", mock_sauvola_threshold())
    def test_sauvola_threshold(self, dicom_image, window_size, k):
        # Apply Sauvola's threshold to the mock DICOM image
        output = dicom_image
        Threshold.sauvola_threshold(dicom_image, output, window_size, k)
        # Check that the pixel data has been modified
        assert not np.array_equal(dicom_image.pixel_data, output.pixel_data)

        # Check that the output is a binary image (0 or 255)
        assert np.all(np.logical_or(output.pixel_data == 0, output.pixel_data == 255))
