"""Helpers for loading images and converting tensors for Streamlit display."""

import time

import numpy as np
import torch
from PIL import Image as PILImage


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a uint8 numpy array [0, 255]."""
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim > 2:
        arr = arr.squeeze()
    if arr.ndim > 2:
        arr = arr[0]

    mn, mx = arr.min(), arr.max()
    if mx - mn > 0:
        arr = (arr - mn) / (mx - mn) * 255.0
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


def numpy_to_pil(arr: np.ndarray) -> PILImage.Image:
    """Convert a uint8 numpy array to a PIL RGB Image."""
    if arr.ndim == 2:
        return PILImage.fromarray(arr, mode="L").convert("RGB")
    return PILImage.fromarray(arr, mode="RGB")


def draw_roi_rectangle(image_arr: np.ndarray, x_min: int, y_min: int,
                        x_max: int, y_max: int,
                        color=(255, 0, 0), thickness: int = 3) -> np.ndarray:
    """Draw a rectangle on a grayscale or RGB image array.
    Returns an RGB uint8 array.
    """
    if image_arr.ndim == 2:
        rgb = np.stack([image_arr] * 3, axis=-1)
    else:
        rgb = image_arr.copy()

    rgb = rgb.astype(np.uint8)
    h, w = rgb.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    t = thickness

    rgb[y_min:y_min + t, x_min:x_max] = color
    rgb[y_max - t:y_max, x_min:x_max] = color
    rgb[y_min:y_max, x_min:x_min + t] = color
    rgb[y_min:y_max, x_max - t:x_max] = color
    return rgb


def overlay_roi_result(original_arr: np.ndarray, result_arr: np.ndarray,
                       x_min: int, y_min: int, x_max: int, y_max: int,
                       overlay_color=(255, 50, 50), alpha: float = 0.5) -> np.ndarray:
    """Overlay a binary/grayscale ROI result back onto the original image.
    Returns an RGB uint8 array with a green border.
    """
    if original_arr.ndim == 2:
        canvas = np.stack([original_arr] * 3, axis=-1).astype(np.float32)
    else:
        canvas = original_arr.astype(np.float32).copy()

    result = result_arr.astype(np.float32)
    rh = min(result.shape[0], y_max - y_min)
    rw = min(result.shape[1], x_max - x_min)
    result = result[:rh, :rw]

    is_binary = len(np.unique(result)) <= 2

    if is_binary:
        mask = result > 0
        for ch in range(3):
            region = canvas[y_min:y_min + rh, x_min:x_min + rw, ch]
            region[mask] = region[mask] * (1 - alpha) + overlay_color[ch] * alpha
    else:
        norm = result / max(result.max(), 1e-6) * 255.0
        for ch in range(3):
            region = canvas[y_min:y_min + rh, x_min:x_min + rw, ch]
            canvas[y_min:y_min + rh, x_min:x_min + rw, ch] = (
                region * (1 - alpha) + norm * alpha
            )

    return draw_roi_rectangle(canvas.astype(np.uint8), x_min, y_min, x_max, y_max,
                               color=(0, 255, 0), thickness=3)


def get_dicom_metadata(image) -> dict:
    """Extract useful DICOM metadata if available."""
    info = {}
    if hasattr(image, "dicom_data") and image.dicom_data is not None:
        ds = image.dicom_data
        for tag in ["PatientID", "Modality", "BitsStored", "BitsAllocated",
                     "PhotometricInterpretation", "Rows", "Columns",
                     "WindowCenter", "WindowWidth", "PixelRepresentation"]:
            val = getattr(ds, tag, None)
            if val is not None:
                info[tag] = str(val)
    return info


def timed_execution(func, *args, **kwargs):
    """Execute func and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
