"""Medical Image Processing — Streamlit Demo Application.

Run with:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st
import torch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from medical_image import (
    Filters,
    RegionOfInterest,
    TopHatAlgorithm,
    KMeansAlgorithm,
    FCMAlgorithm,
    PFCMAlgorithm,
    FebdsAlgorithm,
    BreastMaskAlgorithm,
    DicomWindowAlgorithm,
    GrailWindowAlgorithm,
    BitDepthNormAlgorithm,
)

from components.image_utils import (
    tensor_to_display,
    get_dicom_metadata,
    timed_execution,
    draw_roi_rectangle,
    overlay_roi_result,
)
from components.sidebar import render_sidebar

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Medical Image Processing", layout="wide")

st.markdown(
    "<h2 style='margin-bottom:0'>Medical Image Processing</h2>"
    "<p style='color:gray;margin-top:0'>Interactive demo for the "
    "<code>medical-image-std</code> framework</p>",
    unsafe_allow_html=True,
)

# ── Load image ───────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading image…")
def _load_image(file_bytes: bytes, file_name: str):
    import tempfile
    from medical_image import DicomImage, PNGImage

    suffix = Path(file_name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
    img = DicomImage(tmp.name) if suffix == ".dcm" else PNGImage(tmp.name)
    img.load()
    return img


uploaded_file = st.sidebar.file_uploader(
    "Upload a medical image", type=["dcm", "png", "jpg", "jpeg"], key="file_upload",
)

if uploaded_file is None:
    render_sidebar(0, 0)
    st.info("Upload a medical image (.dcm, .png, .jpg) from the sidebar to get started.")
    st.stop()

image = _load_image(uploaded_file.getvalue(), uploaded_file.name)
img_h, img_w = image.pixel_data.shape[:2]
orig_display = tensor_to_display(image.pixel_data)

# ── Sidebar ──────────────────────────────────────────────────────────────────

_, category, operation, params, device, roi_config = render_sidebar(img_h, img_w)

run = st.sidebar.button("Apply", type="primary", use_container_width=True)

meta = get_dicom_metadata(image)
if meta:
    with st.sidebar.expander("DICOM Metadata"):
        for k, v in meta.items():
            st.sidebar.text(f"{k}: {v}")

# ── Core dispatcher ──────────────────────────────────────────────────────────


def _apply_operation(img, op, par, dev):
    out = img.clone()
    algo = None

    if op == "Gaussian Blur":
        Filters.gaussian_filter(img, out, sigma=par["sigma"], device=dev)
    elif op == "Median Filter":
        Filters.median_filter(img, out, size=par["size"], device=dev)
    elif op == "Difference of Gaussians":
        Filters.difference_of_gaussian(img, out, low_sigma=par["low_sigma"],
                                       high_sigma=par["high_sigma"], device=dev)
    elif op == "Laplacian of Gaussian":
        Filters.laplacian_of_gaussian(img, out, sigma=par["sigma"], device=dev)
    elif op == "Gamma Correction":
        Filters.gamma_correction(img, out, gamma=par["gamma"], device=dev)
    elif op == "Breast Mask":
        algo = BreastMaskAlgorithm(mask_only=True, device=dev); algo(img, out)
    elif op == "Apply Breast Mask":
        algo = BreastMaskAlgorithm(mask_only=False, device=dev); algo(img, out)
    elif op == "DICOM Window":
        wc = None if par.get("auto_wl") else par.get("window_center")
        ww = None if par.get("auto_wl") else par.get("window_width")
        algo = DicomWindowAlgorithm(window_center=wc, window_width=ww, device=dev)
        algo(img, out)
    elif op == "GRAIL Window":
        algo = GrailWindowAlgorithm(n_scales=par["n_scales"],
                                    n_orientations=par["n_orientations"],
                                    delta=par["delta"], k_max=par["k_max"], device=dev)
        algo(img, out)
    elif op == "Bit Depth Normalization":
        bits = None if par.get("auto_bits") else par.get("bits_stored")
        algo = BitDepthNormAlgorithm(bits_stored=bits, target_max=par["target_max"], device=dev)
        algo(img, out)
    elif op == "Top-Hat":
        algo = TopHatAlgorithm(radius=par["radius"], device=dev); algo(img, out)
    elif op == "K-Means":
        algo = KMeansAlgorithm(k=par["k"], max_iter=par["max_iter"],
                               tol=par["tol"], device=dev)
        algo(img, out)
    elif op == "FCM":
        algo = FCMAlgorithm(c=par["c"], m=par["m"], max_iter=par["max_iter"],
                            tol=par["tol"], device=dev)
        algo(img, out)
    elif op == "PFCM":
        algo = PFCMAlgorithm(c=par["c"], m=par["m"], eta=par["eta"],
                             a=par["a"], b=par["b"], tau=par["tau"],
                             max_iter=par["max_iter"], device=dev)
        algo(img, out)
    elif op == "FEBDS":
        algo = FebdsAlgorithm(method=par["method"], device=dev); algo(img, out)

    return out, algo


def _show_algo_details(algo):
    if algo is None:
        return
    if hasattr(algo, "stats") and algo.stats:
        with st.expander("Cluster Statistics"):
            for i, s in enumerate(algo.stats):
                st.text(f"Cluster {i}: {s}")
    if hasattr(algo, "centroids") and algo.centroids is not None:
        with st.expander("Centroids"):
            st.text(algo.centroids.cpu().numpy())


def _show_metrics(elapsed, device, algo):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processing Time", f"{elapsed:.3f} s")
    m2.metric("Device", device)
    if torch.cuda.is_available() and device != "cpu":
        m3.metric("Peak GPU Memory", f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    else:
        m3.metric("Peak GPU Memory", "N/A")
    if algo and hasattr(algo, "n_iter"):
        m4.metric("Iterations", algo.n_iter)
    elif algo and hasattr(algo, "converged"):
        m4.metric("Converged", str(algo.converged))


# ══════════════════════════════════════════════════════════════════════════════
#  ROI WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

if roi_config is not None:
    x_min, y_min = roi_config["x_min"], roi_config["y_min"]
    x_max, y_max = roi_config["x_max"], roi_config["y_max"]

    orig_with_roi = draw_roi_rectangle(orig_display, x_min, y_min, x_max, y_max)

    col_orig, col_proc = st.columns(2)

    with col_orig:
        st.markdown("**Original**")
        st.image(orig_with_roi, use_container_width=True)
        st.caption(
            f"ROI: ({x_min}, {y_min}) to ({x_max}, {y_max})  —  "
            f"{x_max - x_min} x {y_max - y_min} px"
        )

    if not run:
        with col_proc:
            st.markdown("**Processed**")
            st.info("Configure parameters and click **Apply**.")
        st.stop()

    # ── Process ROI ──────────────────────────────────────────────────────

    def _apply_on_roi(image, operation, params, device, roi_config):
        xn, yn = roi_config["x_min"], roi_config["y_min"]
        xx, yx = roi_config["x_max"], roi_config["y_max"]

        if operation == "FEBDS":
            full_out, algo = _apply_operation(image, operation, params, device)
            roi = RegionOfInterest(full_out, coordinates=[xn, yn, xx, yx])
            return roi.load(), algo

        roi = RegionOfInterest(image, coordinates=[xn, yn, xx, yx])
        roi_img = roi.load()
        if roi_config.get("normalize"):
            RegionOfInterest.normalize(roi_img, divisor=roi_config.get("normalize_divisor", 4095.0))
        return _apply_operation(roi_img, operation, params, device)

    with st.spinner("Processing ROI…"):
        (result, algo), elapsed = timed_execution(
            _apply_on_roi, image, operation, params, device, roi_config,
        )

    result_display = tensor_to_display(result.pixel_data)

    with col_proc:
        st.markdown("**Processed**")
        overlay = overlay_roi_result(orig_display, result_display,
                                      x_min, y_min, x_max, y_max)
        st.image(overlay, use_container_width=True)

    # ── Metrics ──────────────────────────────────────────────────────────

    st.divider()
    _show_metrics(elapsed, device, algo)

    # ── Bottom row: ROI before | ROI after ───────────────────────────────

    st.divider()
    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**ROI — before**")
        cropped_display = orig_display[y_min:y_max, x_min:x_max]
        st.image(cropped_display, use_container_width=True)
    with col_after:
        st.markdown("**ROI — after**")
        st.image(result_display, use_container_width=True)
        rpd_f = result.pixel_data.float()
        st.caption(
            f"{tuple(result.pixel_data.shape)}  |  "
            f"min {rpd_f.min().item():.4f}  |  max {rpd_f.max().item():.4f}"
        )

    _show_algo_details(algo)

# ══════════════════════════════════════════════════════════════════════════════
#  STANDARD (full-image) WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

else:
    col_orig, col_proc = st.columns(2)

    with col_orig:
        st.markdown("**Original**")
        st.image(orig_display, use_container_width=True)
        pd_f = image.pixel_data.float()
        st.caption(
            f"{tuple(image.pixel_data.shape)}  |  {image.pixel_data.dtype}  |  "
            f"min {pd_f.min().item():.1f}  |  max {pd_f.max().item():.1f}"
        )

    if not run:
        with col_proc:
            st.markdown("**Processed**")
            st.info("Configure parameters and click **Apply**.")
        st.stop()

    with st.spinner("Processing…"):
        (result, algo), elapsed = timed_execution(
            _apply_operation, image, operation, params, device,
        )

    with col_proc:
        st.markdown("**Processed**")
        st.image(tensor_to_display(result.pixel_data), use_container_width=True)
        rpd_f = result.pixel_data.float()
        st.caption(
            f"{tuple(result.pixel_data.shape)}  |  {result.pixel_data.dtype}  |  "
            f"min {rpd_f.min().item():.1f}  |  max {rpd_f.max().item():.1f}"
        )

    st.divider()
    _show_metrics(elapsed, device, algo)
    _show_algo_details(algo)
