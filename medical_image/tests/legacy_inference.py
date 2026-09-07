# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 [YOUR NAME / ORGANIZATION]
#
# This file is part of medical-image-std.
#
# medical-image-std is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# medical-image-std is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with medical-image-std. If not, see
# <https://www.gnu.org/licenses/>.

"""The sliding-window inference that shipped before the optimisation work.

Kept as a golden reference so the current implementation can be checked against
it rather than against a remembered description of it.  Two callers:
``test_inference_optimization.py`` asserts they agree, and
``scripts/benchmark_inference.py`` times one against the other.

Do not "fix" anything in here.  The duplicate-corner over-weighting in
:func:`legacy_infer_patches` is a faithful reproduction of the bug this module
exists to demonstrate.
"""

import numpy as np
import torch


@torch.no_grad()
def legacy_infer_patches(model, image, patch_size, stride, device="cpu"):
    """``DeepSegmentationAlgorithm._infer_patches`` as it was.

    Returns the probability map and the number of patches it ran, which is
    larger than the number of *distinct* patches: clamping the final strides to
    ``h - patch_size`` maps several of them onto the same corner, and the
    averaging below then counts those patches once per duplicate.
    """
    h, w = image.shape
    ps = patch_size

    prediction_sum = torch.zeros(h, w)
    count = torch.zeros(h, w)

    patches = []
    positions = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + ps, h)
            x_end = min(x + ps, w)
            y_start = max(0, y_end - ps)
            x_start = max(0, x_end - ps)

            patch = image[y_start : y_start + ps, x_start : x_start + ps]

            if patch.shape[0] < ps or patch.shape[1] < ps:
                padded = torch.zeros(ps, ps)
                padded[: patch.shape[0], : patch.shape[1]] = patch
                patch = padded

            patches.append(patch)
            positions.append((y_start, x_start))

    batch_size = 8
    pred_patches = []
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i : i + batch_size]).unsqueeze(1).to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).squeeze(1).cpu()
        pred_patches.extend([p for p in probs])

    for patch_pred, (y, x) in zip(pred_patches, positions):
        ph, pw = patch_pred.shape
        ey = min(y + ph, h)
        ex = min(x + pw, w)
        prediction_sum[y:ey, x:ex] += patch_pred[: ey - y, : ex - x]
        count[y:ey, x:ex] += 1

    return prediction_sum / count.clamp(min=1), len(patches)


def legacy_clahe(image):
    """The previous CLAHE call, with its 8x8-*pixel* kernel.

    scikit-image reads ``kernel_size`` as a size in pixels, so this asks for
    roughly 213,000 tiles on a full mammogram where training (OpenCV
    ``tileGridSize=(8, 8)``) used 64.
    """
    from skimage.exposure import equalize_adapthist

    img_np = image.numpy()
    img_min, img_max = img_np.min(), img_np.max()
    if img_max - img_min > 0:
        img_norm = (img_np - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_np)

    enhanced = equalize_adapthist(
        img_norm,
        kernel_size=((8, 8) if min(img_np.shape) >= 8 else None),
        clip_limit=0.02,
    )
    return torch.from_numpy(enhanced.astype(np.float32))


def legacy_preprocess(image, use_clahe):
    """Normalise to [0, 1] and optionally CLAHE — no breast crop, as before."""
    img = image.float()
    if img.ndim == 3:
        img = img.squeeze(0)
    if img.max() > 1.0:
        img = img / img.max()
    if use_clahe:
        img = legacy_clahe(img)
    return img


def interior_mask(shape, patch_size):
    """The region the duplicate-corner over-weighting never touched.

    Duplicates only arise where a stride was clamped, which is a band
    ``patch_size`` wide along the bottom and right edges.
    """
    h, w = shape
    keep = np.zeros((h, w), dtype=bool)
    keep[: max(0, h - patch_size), : max(0, w - patch_size)] = True
    return keep
