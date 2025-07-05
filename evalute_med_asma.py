import os
import re
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import torch
from PIL import Image
# Initialize MedGemma
device = 0 if torch.cuda.is_available() else -1
print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")
pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it", device=device)
print("[INFO] MedGemma pipeline initialized.")

def dcm_to_raw_array(dcm_path):
    """Return raw DICOM pixel data as 3-channel image, without normalization."""
    print(f"[DEBUG] Reading DICOM file: {dcm_path}")
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    print(f"[DEBUG] DICOM shape: {img.shape}, dtype: {img.dtype}")
    # if len(img.shape) == 2:
    #     img = np.stack([img]*3, axis=-1)
    #     print("[DEBUG] Converted grayscale to 3-channel image.")
    image = Image.fromarray(img.astype(np.uint8))

    return image

def ask_medgemma(image_np):
    """Query MedGemma with raw DICOM image array (no normalization)."""
    print("[INFO] Querying MedGemma...")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_np},
            {"type": "text", "text": "Does this image contain microcalcification? Answer with only 'yes' or 'no'."}
        ]
    }]
    result = pipe(messages)[0]["generated_text"]
    print(f"[RESULT] MedGemma response: {result}")
    return "yes" if "yes" in result.lower() else "no"

def extract_id_from_filename(filename):
    """Extract leading digits before '_mask.tif'."""
    match = re.match(r"(\d+)_mask\.tif", filename)
    tif_id = match.group(1) if match else None
    print(f"[DEBUG] Extracted ID from {filename}: {tif_id}")
    return tif_id

def find_matching_dcm(tif_id, dcm_root):
    """Search for a DICOM file starting with the tif_id in a given directory."""
    print(f"[DEBUG] Searching for matching DICOM for ID: {tif_id}")
    for root, _, files in os.walk(dcm_root):
        for file in files:
            if file.lower().endswith(".dcm") and file.startswith(tif_id):
                matched_path = os.path.join(root, file)
                print(f"[INFO] Match found: {matched_path}")
                return matched_path
    print(f"[WARNING] No match found for ID: {tif_id}")
    return None

def evaluate_from_masks(tif_dir, dcm_root, output_csv="medgemma_results.csv"):
    results = []
    tif_files = [f for f in os.listdir(tif_dir) if f.lower().endswith(".tif")]
    print(f"[INFO] Found {len(tif_files)} .tif mask files.")

    for tif_file in tqdm(tif_files, desc="Evaluating from mask files"):
        print(f"\n[PROCESSING] {tif_file}")
        tif_id = extract_id_from_filename(tif_file)
        if not tif_id:
            print(f"[WARNING] Skipping unrecognized file name: {tif_file}")
            continue

        dcm_path = find_matching_dcm(tif_id, dcm_root)
        if not dcm_path:
            print(f"[WARNING] No matching DICOM found for: {tif_file}")
            continue

        try:
            image_np = dcm_to_raw_array(dcm_path)
            prediction = ask_medgemma(image_np)
            results.append({
                "mask_file": tif_file,
                "dcm_file": os.path.basename(dcm_path),
                "microcalcification": prediction
            })
            print(f"[SUCCESS] Prediction saved for {tif_file}")
        except Exception as e:
            print(f"[ERROR] Failed to process {dcm_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to: {output_csv}")

if __name__ == "__main__":
    tif_dir = "/home/latis/asma_data/AllMasks/"
    dcm_dir = "/home/latis/asma_data/INbreast Release 1.0/AllDICOMs/"
    print("[START] Beginning evaluation from masks...")
    evaluate_from_masks(tif_dir, dcm_dir)
    print("[DONE] Evaluation complete.")
