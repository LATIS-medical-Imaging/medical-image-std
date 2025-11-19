# import pandas as pd
# import pydicom
#
# df = pd.read_csv("medgemma_results_classification.csv")
#
# print(df.head())


# from transformers import AutoProcessor, AutoModelForImageTextToText
# from PIL import Image
import torch
import pydicom
import numpy as np

# import os
# os.environ["TORCHDYNAMO_DISABLE"] = "1"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# Load the processor and model
# processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
# model = AutoModelForImageTextToText.from_pretrained("google/medgemma-4b-it")

# Load and preprocess DICOM image
# dicom_path = "/home/latis/Downloads/MRBRAIN.DCM"  # Replace with your actual path
# dicom_data = pydicom.dcmread(dicom_path)

# Convert the DICOM pixel data to a PIL Image
# pixel_array = dicom_data.pixel_array

# Normalize the pixel values to 0-255 and convert to uint8
# pixel_array = pixel_array.astype(float)
# pixel_array -= pixel_array.min()
# pixel_array /= pixel_array.max()
# pixel_array *= 255.0
# pixel_array = pixel_array.astype(np.uint8)

# Convert to RGB if grayscale
# if len(pixel_array.shape) == 2:
#   image = Image.fromarray(pixel_array).convert("RGB")
# else:
#    image = Image.fromarray(pixel_array)

# Define the prompt
# prompt = "Describe the medical condition in the image."

# Preprocess and run the model
# inputs = processor(images=image, text=prompt, return_tensors="pt")

# with torch.no_grad():
#    generated_ids = model.generate(**inputs, max_new_tokens=128)

# Decode the output
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("Generated Text:", generated_text)


# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch

dicom_url = "https://marketing.webassets.siemens-healthineers.com/fcc5ee5afaaf9c51/b73cfcb2da62/Vida_Head.MR.Comp_DR-Gain_DR.1005.1.2021.04.27.14.20.13.818.14380335.dcm"  # Add your DICOM file URL here
# dcm_path="/home/latis/Downloads/Altea_t1_sag_tse_DR_77233508.dcm"
dcm_path = "/home/latis/Downloads/1.3.6.1.4.1.5962.99.1.2280943358.716200484.1363785608958.256.0.dcm"
# Fetch DICOM image from URL
dicom_data = requests.get(dicom_url, stream=True).content
dicom_file = pydicom.dcmread(dcm_path)

# Convert DICOM pixel data to a PIL image
# Depending on the DICOM image type, you may need to adjust the conversion.
image_data = dicom_file.pixel_array
image = Image.fromarray(image_data.astype(np.uint8))

# Initialize model and processor
model_id = "google/medgemma-4b-it"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

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
                "text": "Does the image contains some anomalies, and describe it",
            },
            {"type": "image", "image": image},
        ],
    },
]

# Process the input
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

# Decode and print the result
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
print(type(decoded))
