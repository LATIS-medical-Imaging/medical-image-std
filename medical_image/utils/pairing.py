"""
Pairing logic for matching medical images with their annotations/masks.

Supports INbreast, Custom INbreast, and CBIS-DDSM directory structures.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from medical_image.utils.logging import logger

# ---------------------------------------------------------------------------
# Data classes for paired samples
# ---------------------------------------------------------------------------


@dataclass
class INbreastSample:
    """A paired INbreast sample: DICOM + XML annotation + optional ROI file."""

    case_id: str
    dicom_path: str
    xml_path: Optional[str] = None
    roi_path: Optional[str] = None


@dataclass
class CustomINbreastSample:
    """A paired Custom INbreast sample: DICOM + XML + TIF mask."""

    case_id: str
    dicom_path: str
    xml_path: Optional[str] = None
    mask_path: Optional[str] = None
    roi_path: Optional[str] = None


@dataclass
class CBISDDSMROIEntry:
    """A single ROI annotation for a CBIS-DDSM case."""

    roi_path: Optional[str] = None  # ROI crop DICOM (from 1-1.dcm or cropped images)
    mask_path: Optional[str] = (
        None  # Binary mask DICOM (from 1-2.dcm or ROI mask images)
    )


@dataclass
class CBISDDSMSample:
    """A CBIS-DDSM case: full mammogram DICOM + all ROI mask DICOMs."""

    case_id: str
    patient_id: str
    side: str  # LEFT or RIGHT
    view: str  # CC or MLO
    task: str  # e.g. Calc-Test, Mass-Training
    mammogram_path: str
    roi_entries: List[CBISDDSMROIEntry] = field(default_factory=list)
    mask_paths: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# INbreast pairing
# ---------------------------------------------------------------------------

_NUMERIC_PREFIX = re.compile(r"^(\d+)")


def _extract_numeric_id(filename: str) -> Optional[str]:
    """Extract leading numeric ID from a filename."""
    match = _NUMERIC_PREFIX.match(filename)
    return match.group(1) if match else None


def pair_inbreast(
    dicoms_dir: str,
    xml_dir: str,
    roi_dir: Optional[str] = None,
) -> List[INbreastSample]:
    """
    Pair INbreast DICOM files with their XML annotations and ROI files.

    Matching is by numeric prefix: ``20586908_...dcm`` ↔ ``20586908.xml``.

    Args:
        dicoms_dir: Path to ``AllDICOMs/`` directory.
        xml_dir: Path to ``AllXML/`` directory.
        roi_dir: Optional path to ``AllROI/`` directory.

    Returns:
        List of :class:`INbreastSample` with matched paths.
    """
    dicoms_dir = Path(dicoms_dir)
    xml_dir = Path(xml_dir)

    # Build lookup: numeric_id → xml path
    xml_lookup = {}
    if xml_dir.exists():
        for f in xml_dir.iterdir():
            if f.suffix.lower() == ".xml":
                nid = _extract_numeric_id(f.stem)
                if nid:
                    xml_lookup[nid] = str(f)

    # Build lookup: numeric_id → roi path
    roi_lookup = {}
    if roi_dir and Path(roi_dir).exists():
        for f in Path(roi_dir).iterdir():
            if f.suffix.lower() == ".roi":
                nid = _extract_numeric_id(f.stem)
                if nid:
                    roi_lookup[nid] = str(f)

    samples = []
    for dcm_file in sorted(dicoms_dir.iterdir()):
        if dcm_file.suffix.lower() != ".dcm":
            continue
        nid = _extract_numeric_id(dcm_file.name)
        if nid is None:
            logger.warning(f"Could not extract ID from DICOM: {dcm_file.name}")
            continue

        sample = INbreastSample(
            case_id=nid,
            dicom_path=str(dcm_file),
            xml_path=xml_lookup.get(nid),
            roi_path=roi_lookup.get(nid),
        )
        samples.append(sample)

    logger.info(
        f"INbreast pairing: {len(samples)} DICOMs, "
        f"{sum(1 for s in samples if s.xml_path)} with XML"
    )
    return samples


# ---------------------------------------------------------------------------
# Custom INbreast pairing (adds TIF masks)
# ---------------------------------------------------------------------------


def pair_custom_inbreast(
    dicoms_dir: str,
    xml_dir: str,
    masks_dir: str,
    roi_dir: Optional[str] = None,
) -> List[CustomINbreastSample]:
    """
    Pair Custom INbreast DICOMs with XML annotations and TIF masks.

    Mask naming convention: ``<case_id>_mask.tif``

    Args:
        dicoms_dir: Path to ``AllDICOMs/`` directory.
        xml_dir: Path to ``AllXML/`` directory.
        masks_dir: Path to ``AllMasks/`` directory.
        roi_dir: Optional path to ``AllROI/`` directory.

    Returns:
        List of :class:`CustomINbreastSample`.
    """
    dicoms_dir = Path(dicoms_dir)
    xml_dir = Path(xml_dir)
    masks_dir = Path(masks_dir)

    # Build lookups
    xml_lookup = {}
    if xml_dir.exists():
        for f in xml_dir.iterdir():
            if f.suffix.lower() == ".xml":
                nid = _extract_numeric_id(f.stem)
                if nid:
                    xml_lookup[nid] = str(f)

    mask_lookup = {}
    if masks_dir.exists():
        for f in masks_dir.iterdir():
            if f.suffix.lower() == ".tif":
                nid = _extract_numeric_id(f.stem)
                if nid:
                    mask_lookup[nid] = str(f)

    roi_lookup = {}
    if roi_dir and Path(roi_dir).exists():
        for f in Path(roi_dir).iterdir():
            if f.suffix.lower() == ".roi":
                nid = _extract_numeric_id(f.stem)
                if nid:
                    roi_lookup[nid] = str(f)

    samples = []
    for dcm_file in sorted(dicoms_dir.iterdir()):
        if dcm_file.suffix.lower() != ".dcm":
            continue
        nid = _extract_numeric_id(dcm_file.name)
        if nid is None:
            logger.warning(f"Could not extract ID from DICOM: {dcm_file.name}")
            continue

        sample = CustomINbreastSample(
            case_id=nid,
            dicom_path=str(dcm_file),
            xml_path=xml_lookup.get(nid),
            mask_path=mask_lookup.get(nid),
            roi_path=roi_lookup.get(nid),
        )
        samples.append(sample)

    logger.info(
        f"CustomINbreast pairing: {len(samples)} DICOMs, "
        f"{sum(1 for s in samples if s.mask_path)} with masks, "
        f"{sum(1 for s in samples if s.xml_path)} with XML"
    )
    return samples


# ---------------------------------------------------------------------------
# CBIS-DDSM pairing
# ---------------------------------------------------------------------------

# Folder pattern: Calc-Test_P_00038_LEFT_CC  (mammogram)
#                 Calc-Test_P_00038_LEFT_CC_1 (mask 1)
_CBIS_FOLDER_PATTERN = re.compile(
    r"^((?:Calc|Mass)-(?:Test|Training))_"  # group 1: task prefix
    r"(P_\d+)_"  # group 2: patient ID
    r"(LEFT|RIGHT)_"  # group 3: side
    r"(CC|MLO)"  # group 4: view
    r"(?:_(\d+))?$"  # group 5: optional mask index
)


def _find_dcm_in_tree(root: str) -> Optional[str]:
    """
    Walk a CBIS-DDSM case directory tree and return the first ``.dcm`` file found.

    The structure is typically:
        ``<case_folder>/<study-id>/<series>/1-1.dcm``
    """
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith(".dcm"):
                return os.path.join(dirpath, fn)
    return None


def _find_all_dcm_in_tree(root: str) -> List[str]:
    """Return all ``.dcm`` files under a directory tree."""
    dcms = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.lower().endswith(".dcm"):
                dcms.append(os.path.join(dirpath, fn))
    return dcms


def _parse_roi_folder(roi_folder: str) -> CBISDDSMROIEntry:
    """
    Parse a single CBIS-DDSM ROI folder and identify the ROI crop and mask DICOMs.

    Handles two layouts:

    - **Standard**: ``ROI mask images/`` contains ``1-1.dcm`` (crop) and ``1-2.dcm`` (mask).
    - **Split**: ``cropped images/`` has the crop, ``ROI mask images/`` has the mask.
    """
    cropped_dcm: Optional[str] = None
    mask_dcm: Optional[str] = None

    for dirpath, _, filenames in os.walk(roi_folder):
        parent_name = os.path.basename(dirpath).lower()
        dcms = sorted(f for f in filenames if f.lower().endswith(".dcm"))

        if "cropped images" in parent_name:
            # The cropped images folder contains the ROI crop
            if dcms:
                cropped_dcm = os.path.join(dirpath, dcms[0])

        elif "roi mask images" in parent_name:
            if cropped_dcm is not None or len(dcms) == 1:
                # Cropped image found separately, or only one file → it's the mask
                mask_dcm = os.path.join(dirpath, dcms[0]) if dcms else None
            elif len(dcms) >= 2:
                # 1-1.dcm = ROI crop, 1-2.dcm = binary mask
                cropped_dcm = os.path.join(dirpath, dcms[0])
                mask_dcm = os.path.join(dirpath, dcms[1])

    return CBISDDSMROIEntry(roi_path=cropped_dcm, mask_path=mask_dcm)


def pair_cbis_ddsm(root_dir: str) -> List[CBISDDSMSample]:
    """
    Pair CBIS-DDSM full mammogram images with their ROI mask images.

    Expects the standard TCIA directory layout::

        root_dir/
        └── CBIS-DDSM/
            ├── Calc-Test_P_00038_LEFT_CC/       ← mammogram
            ├── Calc-Test_P_00038_LEFT_CC_1/     ← mask 1
            ├── Calc-Test_P_00038_LEFT_CC_2/     ← mask 2
            ...

    Args:
        root_dir: Path to the manifest directory containing ``CBIS-DDSM/``.

    Returns:
        List of :class:`CBISDDSMSample` with paired mammogram and mask paths.
    """
    cbis_root = Path(root_dir)

    # Try to find CBIS-DDSM subfolder
    cbis_ddsm = cbis_root / "CBIS-DDSM"
    if not cbis_ddsm.exists():
        # Maybe root_dir IS the CBIS-DDSM folder
        cbis_ddsm = cbis_root

    if not cbis_ddsm.exists():
        raise FileNotFoundError(f"CBIS-DDSM directory not found at {cbis_root}")

    # Group folders by base case key
    mammogram_folders = {}  # key → folder path
    mask_folders = {}  # key → [folder paths]

    for entry in sorted(cbis_ddsm.iterdir()):
        if not entry.is_dir():
            continue
        match = _CBIS_FOLDER_PATTERN.match(entry.name)
        if not match:
            continue

        task_prefix = match.group(1)
        patient_id = match.group(2)
        side = match.group(3)
        view = match.group(4)
        mask_index = match.group(5)

        base_key = f"{task_prefix}_{patient_id}_{side}_{view}"

        if mask_index is None:
            # This is the full mammogram folder
            mammogram_folders[base_key] = str(entry)
        else:
            # This is a mask folder
            mask_folders.setdefault(base_key, []).append(str(entry))

    # Build paired samples
    samples = []
    for base_key, mammo_folder in mammogram_folders.items():
        match = _CBIS_FOLDER_PATTERN.match(os.path.basename(mammo_folder))
        if not match:
            continue

        mammo_dcm = _find_dcm_in_tree(mammo_folder)
        if mammo_dcm is None:
            logger.warning(f"No DICOM found for mammogram: {mammo_folder}")
            continue

        # Parse each ROI folder into structured entries
        roi_entries: List[CBISDDSMROIEntry] = []
        mask_dcms: List[str] = []
        for roi_folder in sorted(mask_folders.get(base_key, [])):
            entry = _parse_roi_folder(roi_folder)
            roi_entries.append(entry)
            # Backward-compatible flat mask list
            if entry.mask_path:
                mask_dcms.append(entry.mask_path)

        sample = CBISDDSMSample(
            case_id=base_key,
            patient_id=match.group(2),
            side=match.group(3),
            view=match.group(4),
            task=match.group(1),
            mammogram_path=mammo_dcm,
            roi_entries=roi_entries,
            mask_paths=mask_dcms,
        )
        samples.append(sample)

    logger.info(
        f"CBIS-DDSM pairing: {len(samples)} cases, "
        f"{len([e for s in samples for e in s.roi_entries])} ROI entries"
    )
    return samples
