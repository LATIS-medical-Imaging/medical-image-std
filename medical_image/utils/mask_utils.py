"""
Mask utilities for medical image datasets.

Provides XML annotation parsing (INbreast plist format), binary mask generation
from polygon/point ROIs, TIF mask loading, and DICOM mask stacking.
"""

import plistlib
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image as PILImage
from skimage.draw import polygon as draw_polygon, disk as draw_disk

from log_manager import logger


# ---------------------------------------------------------------------------
# INbreast XML (Apple plist) parsing
# ---------------------------------------------------------------------------


def parse_inbreast_xml(xml_path: str) -> List[Dict[str, Any]]:
    """
    Parse an INbreast XML annotation file (Apple plist format).

    Each ROI entry contains:
        - name: str (e.g. 'Calcification', 'Mass')
        - roi_type: int (15 = polygon, 19 = single point, etc.)
        - num_points: int
        - points_px: list of (x, y) tuples in pixel coordinates

    Args:
        xml_path: Absolute path to the XML file.

    Returns:
        List of ROI dictionaries.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML annotation not found: {xml_path}")

    with open(xml_path, "rb") as f:
        plist_data = plistlib.load(f)

    rois_out: List[Dict[str, Any]] = []

    images = plist_data.get("Images", [])
    for image_entry in images:
        for roi in image_entry.get("ROIs", []):
            points_raw = roi.get("Point_px", [])
            points_px = _parse_point_strings(points_raw)

            rois_out.append(
                {
                    "name": roi.get("Name", "Unknown"),
                    "roi_type": roi.get("Type", -1),
                    "num_points": roi.get("NumberOfPoints", 0),
                    "points_px": points_px,
                }
            )

    logger.debug(f"Parsed {len(rois_out)} ROIs from {xml_path.name}")
    return rois_out


def _parse_point_strings(point_strings: List[str]) -> List[Tuple[float, float]]:
    """
    Parse INbreast point strings like '(716.030029, 3177.629883)' into (x, y) tuples.
    """
    pattern = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)")
    points = []
    for s in point_strings:
        match = pattern.search(s)
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            points.append((x, y))
    return points


# ---------------------------------------------------------------------------
# Binary mask generation from XML annotations
# ---------------------------------------------------------------------------


def xml_to_binary_mask(
    xml_path: str,
    image_shape: Tuple[int, int],
    point_radius: int = 3,
) -> np.ndarray:
    """
    Convert INbreast XML annotations into a single binary mask.

    - Polygon ROIs (multiple points) → filled polygons
    - Single-point ROIs → small disks of ``point_radius``

    Args:
        xml_path: Path to the INbreast XML annotation file.
        image_shape: (height, width) of the corresponding DICOM image.
        point_radius: Radius for single-point ROIs (default: 3 pixels).

    Returns:
        Binary numpy array of shape ``image_shape`` with dtype ``np.uint8``
        (values 0 or 1).
    """
    rois = parse_inbreast_xml(xml_path)
    mask = np.zeros(image_shape, dtype=np.uint8)
    h, w = image_shape

    for roi in rois:
        points = roi["points_px"]
        if not points:
            continue

        if len(points) == 1:
            # Single point → disk
            cx, cy = points[0]
            rr, cc = draw_disk((int(cy), int(cx)), point_radius, shape=(h, w))
            mask[rr, cc] = 1

        elif len(points) >= 3:
            # Polygon → filled region
            rows = [int(p[1]) for p in points]
            cols = [int(p[0]) for p in points]
            rr, cc = draw_polygon(rows, cols, shape=(h, w))
            mask[rr, cc] = 1

        else:
            # 2 points — treat as two separate disk markers
            for px, py in points:
                rr, cc = draw_disk((int(py), int(px)), point_radius, shape=(h, w))
                mask[rr, cc] = 1

    logger.debug(
        f"Generated binary mask {image_shape} with "
        f"{mask.sum()} positive pixels from {len(rois)} ROIs"
    )
    return mask


# ---------------------------------------------------------------------------
# TIF mask loading
# ---------------------------------------------------------------------------


def load_tif_mask(tif_path: str) -> np.ndarray:
    """
    Load a TIF segmentation mask as a binary numpy array.

    Args:
        tif_path: Path to the ``.tif`` mask file.

    Returns:
        Binary numpy array (0/1) of shape (H, W), dtype ``np.uint8``.
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"Mask file not found: {tif_path}")

    img = PILImage.open(tif_path)
    mask = np.array(img)

    # Binarize: anything > 0 is foreground
    mask = (mask > 0).astype(np.uint8)

    logger.debug(f"Loaded TIF mask {tif_path.name}: shape={mask.shape}")
    return mask


# ---------------------------------------------------------------------------
# DICOM mask stacking (for CBIS-DDSM)
# ---------------------------------------------------------------------------


def stack_dicom_masks(dcm_paths: List[str]) -> torch.Tensor:
    """
    Load multiple DICOM mask files and combine into a single binary mask tensor.

    All masks are OR-merged into one mask of the maximum spatial dimensions
    found across the inputs.

    Args:
        dcm_paths: List of paths to DICOM mask files.

    Returns:
        Tensor of shape (1, H, W) with dtype ``torch.float32``, values 0 or 1.
    """
    import pydicom

    if not dcm_paths:
        raise ValueError("No DICOM mask paths provided")

    masks = []
    for p in dcm_paths:
        ds = pydicom.dcmread(p)
        arr = ds.pixel_array.astype(np.float32)
        # Binarize
        arr = (arr > 0).astype(np.float32)
        masks.append(arr)

    # Find max dimensions
    max_h = max(m.shape[0] for m in masks)
    max_w = max(m.shape[1] for m in masks)

    # OR-merge into single mask
    combined = np.zeros((max_h, max_w), dtype=np.float32)
    for m in masks:
        h, w = m.shape
        combined[:h, :w] = np.maximum(combined[:h, :w], m)

    tensor = torch.from_numpy(combined).unsqueeze(0)  # (1, H, W)
    logger.debug(
        f"Stacked {len(dcm_paths)} DICOM masks → shape={tuple(tensor.shape)}"
    )
    return tensor
