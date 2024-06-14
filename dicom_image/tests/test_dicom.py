import numpy as np
import pytest

from dicom_image.tests.mock_sample import *
import os

from dicom_image.utils.threshold import Threshold
from log_manager import logger


class TestDicom:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_dicom_image(self, dicom_image):
        dicom_image.to_png("dummy_data/")
        assert dicom_image.pixel_data is not None
        assert dicom_image.width == 512
        assert dicom_image.height == 512
        assert os.path.splitext("dummy_data/converted_image.png")[1].lower() == ".png"

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_otsu_threshold(self, dicom_image):
        # Apply Otsu's threshold to the mock DICOM image
        original_pixel_data = (
            dicom_image.pixel_data.copy()
        )  # Make a copy for comparison
        # Threshold.otsu_threshold(dicom_image.pixel_data)
        dicom_image.apply_threshold(Threshold.otsu_threshold)
        # Check that the pixel data has been modified
        assert not np.array_equal(dicom_image.pixel_data, original_pixel_data)

        # Check that the output is a binary image (0 or 255)
        assert np.all(
            np.logical_or(dicom_image.pixel_data == 0, dicom_image.pixel_data == 255)
        )

    @pytest.mark.parametrize("dicom_image, window_size, k", mock_sauvola_threshold())
    def test_sauvola_threshold(self, dicom_image, window_size, k):
        # Apply Sauvola's threshold to the mock DICOM image
        dicom_image.apply_threshold(
            lambda data: Threshold.sauvola_threshold(data, window_size=window_size, k=k)
        )

        # Check that the output is a binary image (0 or 255)
        logger.info(dicom_image.pixel_data)

        condition = np.logical_or(
            dicom_image.pixel_data == 0, dicom_image.pixel_data == 255
        )
        logger.info(f"Condition array: {condition}")
        logger.info(f"False elements: {np.where(~condition)}")

        assert np.all(condition)
