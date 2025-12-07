import os

import requests

from medical_image.data.dicom_image import DicomImage
from medical_image.data.png_image import PNGImage


def mock_dicom_image():
    mock_dicom()
    dicom_image = DicomImage("dummy_data/sample.dcm")
    dicom_image.load()
    # dicom_image.save()
    return [dicom_image]


def mock_png_image():
    download_png()
    png_image = PNGImage("dummy_data/sample.png")
    png_image.load()
    # dicom_image.save()
    return [png_image]


def mock_sauvola_threshold():
    mock_dicom()
    dicom_image = DicomImage("dummy_data/sample.dcm")
    dicom_image.load()

    # dicom_image.save()
    return [(dicom_image, 15, 0.2), (dicom_image, 35, 0.3), (dicom_image, 25, 0.1)]


def mock_dicom():
    img_dir = os.path.join("dummy_data")
    os.makedirs(img_dir, exist_ok=True)

    dicom_path = os.path.join(img_dir, "sample.dcm")

    if os.path.exists(dicom_path):
        print("DICOM file already exists at:", dicom_path)
        return

    dicom_url = "https://marketing.webassets.siemens-healthineers.com/1800000004191561/771cc0fe7509/swi_tra_p2_448_1800000004191561.dcm"

    try:
        response = requests.get(dicom_url)
        response.raise_for_status()  # Raise an error for bad response status codes

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
