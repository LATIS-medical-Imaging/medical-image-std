import pytest

from dicom_image.tests.mock_sample import *
import os


class TestDicom:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_dicom_image(self, dicom_image):
        dicom_image.to_png("dummy_data/")
        assert dicom_image.pixel_data is not None
        assert dicom_image.width == 512
        assert dicom_image.height == 512
        assert os.path.splitext("dummy_data/converted_image.png")[1].lower() == ".png"

