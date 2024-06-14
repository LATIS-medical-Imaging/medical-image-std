import os
import zipfile
from io import BytesIO

import requests


def mock_dicom():
    img_dir = os.path.join("dummy_data")
    os.makedirs(img_dir, exist_ok=True)

    dicom_path = os.path.join(img_dir, "sample.dcm")

    if os.path.exists(dicom_path):
        print("DICOM file already exists at:", dicom_path)
        return
    # TODO: The url does not work
    dicom_url = "https://www.dicomlibrary.com/?requestType=WADO&studyUID=1.2.826.0.1.3680043.8.1055.1.20111103112244831.40200514.30965937&manage=feb6447a72c9a0a31e1bb4459e547964&token=51057a667c38114c33415f14d55b26b379ca1e0b4c1e0be4ec"

    try:
        response = requests.get(dicom_url)
        response.raise_for_status()  # Raise an error for bad response status codes

        zip_path = os.path.join(img_dir, "downloaded.zip")

        with open(zip_path, "wb") as f:
            f.write(response.content)
        print(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Extract all the contents of the zip file to the directory specified
            z.extractall(img_dir)

        # Path to the desired DICOM file
        extracted_dicom_path = os.path.join(img_dir, "series-000001", "image-000001.dcm")

        if os.path.exists(extracted_dicom_path):
            os.rename(extracted_dicom_path, dicom_path)
            print("DICOM file downloaded and saved successfully.")
            print("Saved at:", dicom_path)
        else:
            print("DICOM file not found in the extracted contents.")

        # Remove the downloaded zip file
        os.remove(zip_path)

    except requests.exceptions.RequestException as e:
        print("Error downloading the DICOM file:", e)
    except zipfile.BadZipFile as e:
        print("Error processing the zip file:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)


# Example usage
mock_dicom()
