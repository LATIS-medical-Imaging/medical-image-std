import os
import re
import pydicom
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from PIL import Image
import json

# Initialize MedGemma model and processor
device = 0 if torch.cuda.is_available() else -1
print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")

model_id = "google/medgemma-4b-it"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

print("[INFO] MedGemma model and processor initialized.")


def dcm_to_raw_array(dcm_path):
    """Return raw DICOM pixel data as 3-channel image, without normalization."""
    print(f"[DEBUG] Reading DICOM file: {dcm_path}")
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array

    print(f"[DEBUG] DICOM shape: {img.shape}, dtype: {img.dtype}")
    # Convert to PIL Image
    image = Image.fromarray(img.astype(np.uint8))

    return image


def ask_medgemma(image_pil):
    """Query MedGemma with raw DICOM image array (no normalization)."""
    print("[INFO] Querying MedGemma...")

    # Define the message structure
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                Analyze the provided mammography images for the presence of microcalcifications.

                    1. Identify and highlight all suspicious microcalcifications.
                    
                    2. For each detected microcalcification cluster or individual calcification, provide precise anatomical localization (e.g., quadrant, clock position, distance from nipple, depth).
                    
                    3. Describe the morphology and distribution of the calcifications using BI-RADS descriptors (e.g., punctate, amorphous, pleomorphic, fine linear, branching, clustered, diffuse, regional, segmental).
                    
                    4. Based on the detected features, provide a preliminary BI-RADS assessment category for the calcifications (e.g., BI-RADS 2, 3, 4A, 4B, 4C, 5).
                    
                    5. Suggest potential differential diagnoses based on the characteristics of the calcifications.
                    
                    6. Propose next steps or further imaging/biopsy recommendations based on the assessment.
                    
                    7. Indicate the confidence level of the detection and assessment (e.g., High, Moderate, Low).
                    
                    Present the findings in a structured report format, clearly separating sections for 'Detection', 'Localization', 'Characterization', 'Assessment', 'Differential Diagnosis', and 'Recommendations'. Use bullet points for lists and clear, concise language. If possible, visually annotate the image with bounding boxes or overlays highlighting the detected calcifications.
                    
                    * Focus solely on microcalcifications; ignore other findings unless they are directly relevant to the calcification assessment.
                    
                    * Prioritize clinically significant findings.
                    
                    * Acknowledge any limitations in the image quality or specific areas that are challenging to assess

                """,
                },
                {"type": "image", "image": image_pil},
            ],
        },
    ]

    # Process the input with the processor
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    # Generate the output
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]

    # Decode the result
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(f"[RESULT] MedGemma response: {decoded}")

    # Check if the response contains 'yes' or 'no'
    # return "yes" if "yes" in decoded.lower() else "no"
    return decoded


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


def evaluate_from_masks(tif_dir, dcm_root, output_json="medgemma_results.json"):
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
            image_pil = dcm_to_raw_array(dcm_path)
            prediction = ask_medgemma(image_pil)
            results.append(
                {
                    "mask_file": tif_file,
                    "dcm_file": os.path.basename(dcm_path),
                    "microcalcification": prediction,
                }
            )
            print(f"[SUCCESS] Prediction saved for {tif_file}")
        except Exception as e:
            print(f"[ERROR] Failed to process {dcm_path}: {e}")

    # Save results as JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Results saved to: {output_json}")


# def evaluate_from_masks(tif_dir, dcm_root, output_csv="medgemma_results.csv"):
#     results = []
#     tif_files = [f for f in os.listdir(tif_dir) if f.lower().endswith(".tif")]
#     print(f"[INFO] Found {len(tif_files)} .tif mask files.")
#
#     for tif_file in tqdm(tif_files, desc="Evaluating from mask files"):
#         print(f"\n[PROCESSING] {tif_file}")
#         tif_id = extract_id_from_filename(tif_file)
#         if not tif_id:
#             print(f"[WARNING] Skipping unrecognized file name: {tif_file}")
#             continue
#
#         dcm_path = find_matching_dcm(tif_id, dcm_root)
#         if not dcm_path:
#             print(f"[WARNING] No matching DICOM found for: {tif_file}")
#             continue
#
#         try:
#             image_pil = dcm_to_raw_array(dcm_path)
#             prediction = ask_medgemma(image_pil)
#             results.append({
#                 "mask_file": tif_file,
#                 "dcm_file": os.path.basename(dcm_path),
#                 "microcalcification": prediction
#             })
#             print(f"[SUCCESS] Prediction saved for {tif_file}")
#         except Exception as e:
#             print(f"[ERROR] Failed to process {dcm_path}: {e}")
#
#     df = pd.DataFrame(results)
#     df.to_csv(output_csv, index=False)
#     print(f"\n✅ Results saved to: {output_csv}")


if __name__ == "__main__":
    tif_dir = "/home/latis/asma_data/AllMasks/"
    dcm_dir = "/home/latis/asma_data/INbreast Release 1.0/AllDICOMs/"
    print("[START] Beginning evaluation from masks...")
    evaluate_from_masks(tif_dir, dcm_dir)
    print("[DONE] Evaluation complete.")
