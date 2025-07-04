import os
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import torch

# Initialize MedGemma
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it", device=device)

def dcm_to_raw_array(dcm_path):
    """Return raw DICOM pixel data as 3-channel image, without normalization."""
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array  # Keep original dtype and range

    # Convert grayscale to 3-channel by repeating channels
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)

    return img

def ask_medgemma(image_np):
    """Query MedGemma with raw DICOM image array (no normalization)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_np},
            {"type": "text", "text": "Does this image contain microcalcification? Answer with only 'yes' or 'no'."}
        ]
    }]
    result = pipe(messages)[0]["generated_text"]
    return "yes" if "yes" in result.lower() else "no"

def find_dcm_files(root_dir):
    """Find all .dcm files recursively in the dataset."""
    dcm_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(root, file))
    return dcm_files

def evaluate_dataset(dataset_root, output_csv="medgemma_results.csv"):
    dcm_files = find_dcm_files(dataset_root)
    results = []

    for dcm_path in tqdm(dcm_files, desc="Evaluating DICOM files"):
        try:
            image_np = dcm_to_raw_array(dcm_path)
            prediction = ask_medgemma(image_np)
            results.append({"file_path": dcm_path, "microcalcification": prediction})
        except Exception as e:
            print(f"Error processing {dcm_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")

if __name__ == "__main__":
    evaluate_dataset('/home/bobmarley/PycharmProjects/medical-image-std/data/dcm_data/manifest-1751047304477/"CBIS-DDSM"')
