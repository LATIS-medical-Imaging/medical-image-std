"""Time the current sliding-window inference against the previous implementation.

Runs both on a real mammogram with a real checkpoint, so the numbers include the
breast crop, CLAHE and the stitch — not just the model call.

The "before" path is the golden reference in
``medical_image/tests/legacy_inference.py`` — the implementation that shipped
prior to the optimisation work, kept verbatim so this compares against the real
thing rather than a description of it.

Usage::

    python scripts/benchmark_inference.py                       # auto-pick a checkpoint
    python scripts/benchmark_inference.py --patch-sizes 32 256
    python scripts/benchmark_inference.py --checkpoint path/to/best_model.pt

Output agreement is reported alongside the timings.  Expect the interior to match
to float precision and the bottom/right edges to differ: the previous code
clamped its final strides onto duplicate patch corners and then averaged over the
duplicates, over-weighting a band ``patch_size`` wide.  That is a fix, not drift,
which is why this script prints both the overall and interior-only differences.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm
from medical_image.data.dicom_image import DicomImage
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.tests.legacy_inference import (
    interior_mask,
    legacy_clahe,
    legacy_infer_patches,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO / "medical_image" / "tests" / "dummy_data" / "20587054.dcm"
RESULTS_DIR = REPO / "medical-image-baseline" / "results"
CACHE_DIR = Path.home() / ".cache" / "medical-std" / "models"


def legacy_apply(algo, image):
    """The whole previous pipeline: normalise, CLAHE, patch, stitch.

    Deliberately no breast crop — that is one of the things being measured.
    Returns the probability map, the patch count, the preprocessed input (so the
    stitch can be compared without the CLAHE change confounding it) and the time
    CLAHE alone took.
    """
    img = image.float()
    if img.ndim == 3:
        img = img.squeeze(0)
    if img.max() > 1.0:
        img = img / img.max()

    clahe_s = 0.0
    if algo.use_clahe:
        start = time.perf_counter()
        img = legacy_clahe(img)
        clahe_s = time.perf_counter() - start

    prob_map, patches = legacy_infer_patches(
        algo.model, img, algo.patch_size, algo.stride, algo.device
    )
    return prob_map, patches, img, clahe_s


# ---------------------------------------------------------------------------


def find_checkpoint(explicit: str, patch_size: int, prefer_clahe: bool) -> Path:
    """Locate a checkpoint whose config matches the requested patch size."""
    if explicit:
        return Path(explicit)

    suffix = "_clahe" if prefer_clahe else ""
    wanted = [
        f"unet_focal_dice_{patch_size}_inbreast{suffix}",
        f"unet_bce_dice_{patch_size}_inbreast{suffix}",
        f"unetpp_bce_dice_{patch_size}_inbreast{suffix}",
        f"attention_unet_bce_dice_{patch_size}_inbreast{suffix}",
    ]
    for root in (RESULTS_DIR, CACHE_DIR):
        for name in wanted:
            candidate = root / name / "best_model.pt"
            if candidate.exists():
                return candidate
        # Fall back to any checkpoint at this patch size.
        if root.is_dir():
            for entry in sorted(root.iterdir()):
                if (
                    f"_{patch_size}_" in entry.name
                    and (entry / "best_model.pt").exists()
                ):
                    return entry / "best_model.pt"
    raise SystemExit(
        f"No checkpoint found for patch size {patch_size} in {RESULTS_DIR} or {CACHE_DIR}. "
        "Pass --checkpoint explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Prefer a CLAHE checkpoint, so the preprocessing fix is included.",
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    dicom = DicomImage(file_path=args.image)
    dicom.load()
    pixels = dicom.pixel_data.float()
    if pixels.ndim == 3:
        pixels = pixels.squeeze(0)

    print(f"Image   : {Path(args.image).name}  {tuple(pixels.shape)}")
    print(f"Device  : {args.device}   torch threads: {torch.get_num_threads()}\n")

    for patch_size in args.patch_sizes:
        checkpoint = find_checkpoint(args.checkpoint, patch_size, args.clahe)
        algo = DeepSegmentationAlgorithm(
            checkpoint_path=str(checkpoint),
            device=args.device,
        )
        print(f"--- patch {algo.patch_size}, stride {algo.stride} ---")
        print(f"checkpoint: {checkpoint.parent.name}  (CLAHE: {algo.use_clahe})")

        start = time.perf_counter()
        legacy_map, legacy_patches, legacy_input, legacy_clahe_s = legacy_apply(
            algo, pixels
        )
        legacy_s = time.perf_counter() - start

        if algo.use_clahe:
            normalised = pixels / pixels.max() if pixels.max() > 1.0 else pixels
            start = time.perf_counter()
            algo._apply_clahe(normalised)
            current_clahe_s = time.perf_counter() - start

        src = InMemoryImage(array=pixels)
        out = InMemoryImage(source_image=src)
        start = time.perf_counter()
        algo.apply(src, out)
        current_s = time.perf_counter() - start

        print(
            f"  patches   : {legacy_patches:>8,}  ->  {algo.patches_inferred:>8,}   "
            f"({legacy_patches / max(1, algo.patches_inferred):.2f}x fewer)"
        )
        print(
            f"  time      : {legacy_s:>8.2f}s  ->  {current_s:>8.2f}s   "
            f"({legacy_s / current_s:.2f}x faster)"
        )
        print(
            f"  breast box: {algo.breast_bbox}, "
            f"background patches skipped: {algo.patches_skipped:,}"
        )
        if algo.use_clahe:
            print(
                f"  CLAHE     : {legacy_clahe_s:>8.2f}s  ->  {current_clahe_s:>8.2f}s   "
                f"({legacy_clahe_s / current_clahe_s:.2f}x faster, and now matches training)"
            )
        print(f"  lesions   : {algo.lesion_count}")

        # Agreement has to be checked like for like.  The run above crops to the
        # breast, so most of its frame is a hard zero the legacy path never
        # produced — comparing the two directly measures the crop, not the
        # stitch.  Feed the legacy path's own preprocessed input in, so CLAHE is
        # held constant too, and switch off everything else that changed.
        matched = DeepSegmentationAlgorithm(
            checkpoint_path=str(checkpoint),
            device=args.device,
            use_clahe=False,
            crop_to_breast=False,
            skip_background_patches=False,
            blend="uniform",
        )
        matched.use_clahe = False
        src2 = InMemoryImage(array=legacy_input)
        out2 = InMemoryImage(source_image=src2)
        matched.apply(src2, out2)

        legacy_np = legacy_map.numpy()
        matched_np = matched.probability_map.numpy()
        keep = interior_mask(legacy_np.shape, algo.patch_size)
        overall = float(np.abs(legacy_np - matched_np).max())
        interior = (
            float(np.abs(legacy_np[keep] - matched_np[keep]).max())
            if keep.any()
            else float("nan")
        )
        edge_frac = 100.0 * (~keep).mean()

        print("  stitch check (crop off, uniform blend, no skipping):")
        print(f"    interior max |diff| : {interior:.3e}   <- expected ~0")
        print(
            f"    overall  max |diff| : {overall:.3e}   "
            f"<- edge band, {edge_frac:.0f}% of the frame, where the previous"
        )
        print("                          code over-weighted duplicate patch corners\n")


if __name__ == "__main__":
    main()
