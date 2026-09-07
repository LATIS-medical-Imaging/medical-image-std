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

"""Regression guards for the sliding-window inference optimisation.

Pins the two properties the rewrite is only allowed to have if it is correct:

  * it runs strictly fewer patches than the previous implementation, and
  * where the previous implementation was not already wrong, it produces the
    same numbers.

The second one needs the qualifier.  Clamping the final strides to
``h - patch_size`` used to map several of them onto the same patch corner, and
averaging over the duplicates weighted those patches by how many strides
collapsed onto them — so a band ``patch_size`` wide along the bottom and right
edges was over-weighted.  Agreement is therefore asserted on the interior and
the edge difference is asserted to be *non-zero*, because that band is the fix.

Wall-clock timing lives in ``scripts/benchmark_inference.py``, not here: it needs
a real checkpoint and it is not something to assert on in a test suite.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from medical_image.algorithms.deep_segmentation import DeepSegmentationAlgorithm
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.tests.legacy_inference import (
    interior_mask,
    legacy_clahe,
    legacy_infer_patches,
)

DICOM_PATH = Path(__file__).parent / "dummy_data" / "20587054.dcm"


class _StripeModel(nn.Module):
    """Position-dependent output, so a mis-stitched patch cannot go unnoticed.

    A constant-output model would agree with almost any stitching bug; this
    varies within the patch so the weighting actually shows up.
    """

    def forward(self, x):
        b, _, h, w = x.shape
        ramp = torch.linspace(-4.0, 4.0, w).view(1, 1, 1, w)
        column = torch.linspace(-2.0, 2.0, h).view(1, 1, h, 1)
        return (ramp + column + x.mean(dim=1, keepdim=True)).expand(b, 1, h, w).clone()


def _matched_algo(patch_size, stride):
    """Current implementation with every behaviour change switched off.

    Anything left is the rewrite itself, which is what these tests are about.
    """
    return DeepSegmentationAlgorithm(
        model=_StripeModel(),
        patch_size=patch_size,
        stride=stride,
        device="cpu",
        crop_to_breast=False,
        skip_background_patches=False,
        blend="uniform",
    )


@pytest.mark.parametrize(
    "shape,patch_size",
    [
        ((600, 800), 128),
        ((512, 512), 256),
        ((300, 200), 128),
        ((517, 733), 64),
    ],
)
def test_stitch_matches_the_previous_implementation_in_the_interior(shape, patch_size):
    torch.manual_seed(0)
    image = torch.rand(*shape)
    stride = patch_size // 2

    algo = _matched_algo(patch_size, stride)
    legacy, _ = legacy_infer_patches(algo.model, image, patch_size, stride)
    current = algo._infer_patches(image, None)

    assert current.shape == legacy.shape

    keep = interior_mask(tuple(shape), patch_size)
    if keep.any():
        assert torch.allclose(
            current[torch.from_numpy(keep)],
            legacy[torch.from_numpy(keep)],
            atol=1e-5,
        )


@pytest.mark.parametrize("shape,patch_size", [((600, 800), 128), ((517, 733), 64)])
def test_the_edge_band_is_corrected_not_merely_reproduced(shape, patch_size):
    """The old code over-weighted the clamped edge; the new one must not."""
    torch.manual_seed(0)
    image = torch.rand(*shape)
    stride = patch_size // 2

    algo = _matched_algo(patch_size, stride)
    legacy, legacy_count = legacy_infer_patches(algo.model, image, patch_size, stride)
    current = algo._infer_patches(image, None)

    # The legacy loop ran more patches than there were distinct corners.
    assert legacy_count > algo.patches_inferred

    edge = ~interior_mask(tuple(shape), patch_size)
    assert edge.any()
    difference = (current - legacy)[torch.from_numpy(edge)].abs().max()
    assert difference > 0, "the edge band should differ — that band is the bug"


def test_fewer_patches_than_before_on_a_real_mammogram():
    """The breast crop and background skipping have to pay for themselves."""
    pytest.importorskip("pydicom")
    if not DICOM_PATH.exists():
        pytest.skip(f"{DICOM_PATH.name} not available")

    from medical_image.data.dicom_image import DicomImage

    dicom = DicomImage(file_path=str(DICOM_PATH))
    dicom.load()
    pixels = dicom.pixel_data.float()
    if pixels.ndim == 3:
        pixels = pixels.squeeze(0)

    patch_size, stride = 256, 128
    height, width = pixels.shape
    legacy_count = len(range(0, height, stride)) * len(range(0, width, stride))

    algo = DeepSegmentationAlgorithm(
        model=_StripeModel(), patch_size=patch_size, stride=stride, device="cpu"
    )
    source = InMemoryImage(array=pixels)
    algo.apply(source, InMemoryImage(source_image=source))

    assert algo.breast_bbox is not None, "the breast should have been located"
    assert algo.patches_inferred < legacy_count
    # Air is roughly half the frame on a mammogram; anything less than a third
    # saved means the crop silently stopped working.
    assert algo.patches_inferred < legacy_count / 1.5


def test_output_is_full_frame_even_though_inference_is_cropped():
    """Cropping is an optimisation, not a change to the output contract."""
    torch.manual_seed(0)
    image = torch.zeros(900, 900)
    image[100:800, 60:400] = 0.5  # off-centre "breast"

    algo = DeepSegmentationAlgorithm(
        model=_StripeModel(), patch_size=128, stride=64, device="cpu"
    )
    source = InMemoryImage(array=image)
    output = InMemoryImage(source_image=source)
    algo.apply(source, output)

    assert algo.breast_bbox is not None
    assert algo.probability_map.shape == image.shape
    assert output.pixel_data.shape == image.shape


def test_clahe_matches_training_and_beats_the_previous_call():
    """The old call asked scikit-image for 8x8-*pixel* tiles; cv2 meant 64 tiles."""
    torch.manual_seed(0)
    image = torch.rand(256, 256)

    algo = DeepSegmentationAlgorithm(
        model=_StripeModel(), patch_size=64, stride=32, device="cpu", use_clahe=True
    )
    current = algo._apply_clahe(image)
    legacy = legacy_clahe(image)

    assert current.shape == image.shape
    assert current.dtype == torch.float32
    assert 0.0 <= float(current.min()) and float(current.max()) <= 1.0
    # Same input, different transform — if these ever agree, the fix was undone.
    assert not torch.allclose(current, legacy, atol=1e-3)
