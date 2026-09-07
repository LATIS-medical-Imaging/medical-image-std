"""Sweep full-image inference settings and report Dice against latency.

``medical-image-baseline/scripts/evaluate.py`` scores *patches*, so it says
nothing about stride or overlap blending — those only exist once patches are
stitched back into a whole mammogram.  This sweeps them on full images.

Usage::

    python scripts/sweep_inference.py \
        --model unet_focal_dice_256_inbreast \
        --dataset-root data/Inbreast \
        --limit 20 \
        --output sweep.json

Every configuration is scored on the same cases, so the Dice column is
comparable across rows even on a small ``--limit``.  Treat ``stride_ratio=0.5``
with ``blend=uniform`` as the reference: it is what the pipeline did before
these knobs existed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm
from medical_image.data.in_memory_image import InMemoryImage


def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    """Dice over a whole image; 1.0 when both are empty."""
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, truth).sum() / total)


def load_cases(root: str, limit: int) -> list:
    from medical_image.datasets import INbreastDataset

    dataset = INbreastDataset(root_dir=root)
    cases = []
    for i in range(min(limit, len(dataset))):
        sample = dataset[i]
        image = sample["image"]
        mask = sample["mask"]
        if image.ndim == 3:
            image = image.squeeze(0)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        cases.append((image.float(), mask.numpy() > 0))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Checkpoint name on the model server"
    )
    parser.add_argument("--server-url", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stride-ratios",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0],
    )
    parser.add_argument("--blends", nargs="+", default=["uniform", "gaussian"])
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable the breast crop, to isolate its effect on Dice.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Loading up to {args.limit} cases from {args.dataset_root} …")
    cases = load_cases(args.dataset_root, args.limit)
    if not cases:
        raise SystemExit("No cases loaded — check --dataset-root.")
    print(f"Loaded {len(cases)} cases.\n")

    rows = []
    header = f"{'stride':>7} {'blend':>9} {'patches':>9} {'Dice':>8} {'s/case':>8} {'vs ref':>8}"
    print(header)
    print("-" * len(header))

    reference = None
    for ratio in args.stride_ratios:
        for blend in args.blends:
            algo = DeepSegmentationAlgorithm.from_pretrained(
                args.model,
                server_url=args.server_url,
                cache_dir=args.cache_dir,
                device=args.device,
                stride_ratio=ratio,
                blend=blend,
                crop_to_breast=not args.no_crop,
            )

            scores, elapsed, patches = [], 0.0, 0
            for image, truth in cases:
                src = InMemoryImage(array=image)
                out = InMemoryImage(source_image=src)
                start = time.perf_counter()
                algo.apply(src, out)
                elapsed += time.perf_counter() - start
                patches += algo.patches_inferred
                scores.append(dice(out.pixel_data.numpy() > 0.5, truth))

            mean_dice = float(np.mean(scores))
            per_case = elapsed / len(cases)
            if reference is None:
                reference = per_case
            row = {
                "stride_ratio": ratio,
                "blend": blend,
                "mean_patches": patches / len(cases),
                "mean_dice": mean_dice,
                "seconds_per_case": per_case,
                "speedup_vs_reference": reference / per_case,
            }
            rows.append(row)
            print(
                f"{ratio:>7.2f} {blend:>9} {row['mean_patches']:>9.0f} "
                f"{mean_dice:>8.4f} {per_case:>8.2f} {row['speedup_vs_reference']:>7.2f}x"
            )

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {"model": args.model, "cases": len(cases), "results": rows}, indent=2
            )
        )
        print(f"\nWrote {args.output}")

    best = max(rows, key=lambda r: r["mean_dice"])
    print(
        f"\nHighest Dice: stride_ratio={best['stride_ratio']} blend={best['blend']} "
        f"({best['mean_dice']:.4f}, {best['speedup_vs_reference']:.2f}x)"
    )
    print(
        "Pick the knee, not the maximum — a 0.001 Dice gain is not worth 4x the "
        "latency, and with few cases that difference is inside the noise."
    )


if __name__ == "__main__":
    main()
