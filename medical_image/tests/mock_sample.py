import os

import requests

from medical_image.data.dicom_image import DicomImage


def mock_dicom_image():
    mock_dicom()
    dicom_image = DicomImage("dummy_data/sample.dcm")
    dicom_image.save()
    return [dicom_image]


def mock_sauvola_threshold():
    mock_dicom()
    dicom_image = DicomImage("dummy_data/sample.dcm")
    dicom_image.save()
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
