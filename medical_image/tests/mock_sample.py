import os
from pathlib import Path

import requests

from medical_image.data.dicom_image import DicomImage
from medical_image.data.png_image import PNGImage

BASE_DIR = Path(__file__).parent
DUMMY_DIR = BASE_DIR / "dummy_data"


def mock_dicom_image():
    mock_dicom()

    primary = DUMMY_DIR / "20587054.dcm"
    fallback = DUMMY_DIR / "sample.dcm"

    path = primary if primary.exists() else fallback

    dicom_image = DicomImage(str(path))
    dicom_image.load()
    return [dicom_image]


def mock_png_image():
    download_png()

    png_path = DUMMY_DIR / "sample.png"

    png_image = PNGImage(str(png_path))
    png_image.load()
    return [png_image]


def mock_sauvola_threshold():
    mock_dicom()

    dicom_path = DUMMY_DIR / "sample.dcm"

    dicom_image = DicomImage(str(dicom_path))
    dicom_image.load()

    return [
        (dicom_image, 15, 0.2),
        (dicom_image, 35, 0.3),
        (dicom_image, 25, 0.1),
    ]


def mock_dicom():
    DUMMY_DIR.mkdir(exist_ok=True)

    dicom_path = DUMMY_DIR / "sample.dcm"

    if dicom_path.exists():
        print("DICOM file already exists at:", dicom_path)
        return

    dicom_url = "https://marketing.webassets.siemens-healthineers.com/1800000004191561/771cc0fe7509/swi_tra_p2_448_1800000004191561.dcm"

    try:
        response = requests.get(dicom_url)
        response.raise_for_status()

        with open(dicom_path, "wb") as f:
            f.write(response.content)

        print("DICOM file downloaded successfully.")
        print("Saved at:", dicom_path)

    except requests.exceptions.RequestException as e:
        print("Error downloading the DICOM file:", e)


def download_png():
    img_dir = os.path.join("dummy_data")
    os.makedirs(img_dir, exist_ok=True)

    png_path = os.path.join(img_dir, "sample.png")

    if os.path.exists(png_path):
        print("PNG file already exists at:", png_path)
        return

    png_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Steuben_-_Bataille_de_Poitiers.png/1280px-Steuben_-_Bataille_de_Poitiers.png"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(png_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad response status codes

        with open(png_path, "wb") as f:
            f.write(response.content)

        print("PNG file downloaded successfully.")
        print("Saved at:", png_path)

    except requests.exceptions.RequestException as e:
        print("Error downloading the PNG file:", e)


def mock_kernel_sizes():
    return [(3,), (5,), (7,)]


def mock_kernel():
    return [
        (3, 1.5),
        (5, 1.6),
        # (11, 1.7),
        (4, 1.8),
    ]


def mock_median_size():
    return [
        5,
        4,
        10,
        15,
    ]


def mock_two_sigmas():
    return [
        (1, 1.5),
        (1.2, 1.6),
        # (11, 1.7),
        (1.5, 1.8),
    ]
