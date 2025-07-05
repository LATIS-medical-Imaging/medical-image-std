import os
import re
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
    img = ds.pixel_array
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

def extract_id_from_filename(filename):
    """Extract leading digits before '_mask.tif'."""
    match = re.match(r"(\d+)_mask\.tif", filename)
    return match.group(1) if match else None

def find_matching_dcm(tif_id, dcm_root):
    """Search for a DICOM file starting with the tif_id in a given directory."""
    for root, _, files in os.walk(dcm_root):
        for file in files:
            if file.lower().endswith(".dcm") and file.startswith(tif_id):
                return os.path.join(root, file)
    return None

def evaluate_from_masks(tif_dir, dcm_root, output_csv="medgemma_results.csv"):
    results = []
    tif_files = [f for f in os.listdir(tif_dir) if f.lower().endswith(".tif")]

    for tif_file in tqdm(tif_files, desc="Evaluating from mask files"):
        tif_id = extract_id_from_filename(tif_file)
        if not tif_id:
            print(f"Skipping unrecognized file name: {tif_file}")
            continue

        dcm_path = find_matching_dcm(tif_id, dcm_root)
        if not dcm_path:
            print(f"No matching DICOM found for: {tif_file}")
            continue

        try:
            image_np = dcm_to_raw_array(dcm_path)
            prediction = ask_medgemma(image_np)
            results.append({
                "mask_file": tif_file,
                "dcm_file": os.path.basename(dcm_path),
                "microcalcification": prediction
            })
        except Exception as e:
            print(f"Error processing {dcm_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")

if __name__ == "__main__":
    tif_dir = "/home/latis/asma_data/AllMasks/"
    dcm_dir = "/home/latis/asma_data/INbreast Release 1.0/AllDICOMs/"
    evaluate_from_masks(tif_dir, dcm_dir)
