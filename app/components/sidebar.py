"""Sidebar UI components for the Streamlit app."""

import torch
import streamlit as st


OPERATION_CATEGORIES = {
    "Filters": ["Gaussian Blur", "Median Filter", "Difference of Gaussians",
                 "Laplacian of Gaussian", "Gamma Correction"],
    "Mammography": ["Breast Mask", "Apply Breast Mask", "DICOM Window",
                     "GRAIL Window", "Bit Depth Normalization"],
    "Algorithms": ["Top-Hat", "K-Means", "FCM", "PFCM", "FEBDS"],
}


def render_sidebar(img_height: int = 0, img_width: int = 0):
    """Render sidebar controls (file uploader lives in app.py).

    Returns (_, category, operation, params, device, roi_config).
    """
    st.sidebar.header("Medical Image Processing")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    device = st.sidebar.selectbox("Device", devices, index=len(devices) - 1)

    category = st.sidebar.selectbox("Category", list(OPERATION_CATEGORIES.keys()))
    operation = st.sidebar.selectbox("Operation", OPERATION_CATEGORIES[category])

    params = _render_params(operation)

    # ── ROI ───────────────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Region of Interest")
    roi_enabled = st.sidebar.toggle("Enable ROI", value=False)

    roi_config = None
    if roi_enabled and img_height > 0 and img_width > 0:
        roi_config = _render_roi_controls(img_height, img_width)

    return None, category, operation, params, device, roi_config


def _render_roi_controls(img_h: int, img_w: int) -> dict:
    """ROI coordinate controls using sliders for intuitive selection."""
    default_w = min(255, img_w // 2)
    default_h = min(255, img_h // 2)

    roi_w = st.sidebar.slider("Width", 16, img_w, default_w, step=8, key="roi_w_input")
    roi_h = st.sidebar.slider("Height", 16, img_h, default_h, step=8, key="roi_h_input")

    x_min = st.sidebar.slider("X position", 0, max(0, img_w - roi_w), min(100, img_w // 4),
                               step=8, key="roi_x_input")
    y_min = st.sidebar.slider("Y position", 0, max(0, img_h - roi_h), min(100, img_h // 4),
                               step=8, key="roi_y_input")

    x_max = min(img_w, x_min + roi_w)
    y_max = min(img_h, y_min + roi_h)

    normalize = st.sidebar.toggle("Normalize ROI (÷ 4095)", value=True)

    return {
        "x_min": x_min, "y_min": y_min,
        "x_max": x_max, "y_max": y_max,
        "roi_w": roi_w, "roi_h": roi_h,
        "normalize": normalize, "normalize_divisor": 4095.0,
    }


def _render_params(operation: str) -> dict:
    """Render parameter controls based on the selected operation."""
    p = {}

    if operation == "Gaussian Blur":
        p["sigma"] = st.sidebar.slider("Sigma", 0.1, 10.0, 2.0, 0.1)
    elif operation == "Median Filter":
        p["size"] = st.sidebar.slider("Kernel Size (odd)", 3, 21, 5, 2)
    elif operation == "Difference of Gaussians":
        p["low_sigma"] = st.sidebar.slider("Low Sigma", 0.1, 10.0, 1.0, 0.1)
        p["high_sigma"] = st.sidebar.slider("High Sigma", 0.1, 20.0, 1.6, 0.1)
    elif operation == "Laplacian of Gaussian":
        p["sigma"] = st.sidebar.slider("Sigma", 0.1, 10.0, 2.0, 0.1)
    elif operation == "Gamma Correction":
        p["gamma"] = st.sidebar.slider("Gamma", 0.1, 5.0, 1.25, 0.05)
    elif operation == "Breast Mask":
        p["mask_only"] = True
    elif operation == "Apply Breast Mask":
        p["mask_only"] = False
    elif operation == "DICOM Window":
        p["auto_wl"] = st.sidebar.checkbox("Auto (from DICOM header)", value=True)
        if not p["auto_wl"]:
            p["window_center"] = st.sidebar.number_input("Window Center", value=2000.0)
            p["window_width"] = st.sidebar.number_input("Window Width", value=3000.0, min_value=1.0)
    elif operation == "GRAIL Window":
        p["n_scales"] = st.sidebar.slider("Gabor Scales", 1, 6, 3)
        p["n_orientations"] = st.sidebar.slider("Gabor Orientations", 2, 12, 6)
        p["delta"] = st.sidebar.slider("Delta (grid spacing)", 50, 500, 300, 50)
        p["k_max"] = st.sidebar.slider("Max Iterations", 1, 10, 3)
    elif operation == "Bit Depth Normalization":
        p["auto_bits"] = st.sidebar.checkbox("Auto-detect bit depth", value=True)
        if not p["auto_bits"]:
            p["bits_stored"] = st.sidebar.selectbox("Bits Stored", [8, 12, 16], index=1)
        p["target_max"] = st.sidebar.number_input("Target Max", value=255.0, min_value=1.0)
    elif operation == "Top-Hat":
        p["radius"] = st.sidebar.slider("Radius", 1, 20, 4)
    elif operation == "K-Means":
        p["k"] = st.sidebar.slider("Clusters (k)", 2, 10, 2)
        p["max_iter"] = st.sidebar.slider("Max Iterations", 10, 500, 100, 10)
        p["tol"] = st.sidebar.number_input("Tolerance", value=1e-4, format="%.6f")
    elif operation == "FCM":
        p["c"] = st.sidebar.slider("Clusters (c)", 2, 10, 2)
        p["m"] = st.sidebar.slider("Fuzziness (m)", 1.1, 5.0, 2.0, 0.1)
        p["max_iter"] = st.sidebar.slider("Max Iterations", 10, 500, 100, 10)
        p["tol"] = st.sidebar.number_input("Tolerance", value=1e-3, format="%.6f")
    elif operation == "PFCM":
        p["c"] = st.sidebar.slider("Clusters (c)", 2, 10, 2)
        p["m"] = st.sidebar.slider("Fuzziness (m)", 1.1, 5.0, 2.0, 0.1)
        p["eta"] = st.sidebar.slider("Typicality (eta)", 1.1, 5.0, 2.0, 0.1)
        p["a"] = st.sidebar.slider("Membership weight (a)", 0.1, 5.0, 1.0, 0.1)
        p["b"] = st.sidebar.slider("Typicality weight (b)", 0.1, 10.0, 4.0, 0.1)
        p["tau"] = st.sidebar.number_input("Atypicality threshold (tau)", value=0.04, format="%.4f")
        p["max_iter"] = st.sidebar.slider("Max Iterations", 10, 500, 100, 10)
    elif operation == "FEBDS":
        p["method"] = st.sidebar.selectbox("Enhancement Method", ["dog", "log", "fft"])

    return p
