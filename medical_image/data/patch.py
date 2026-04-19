from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch

from medical_image.data.image import Image


class Patch:
    """Represents a patch extracted from an image.

    Attributes:
        parent (Image): The parent image from which this patch is extracted.
        row_idx (int): Vertical index in the patch grid.
        col_idx (int): Horizontal index in the patch grid.
        x (int): Top-left pixel x-coordinate in the original image.
        y (int): Top-left pixel y-coordinate in the original image.
        pixel_data (torch.Tensor): Tensor representing patch pixels.
        is_padded (bool): Whether the patch contains padding.
    """

    def __init__(
        self,
        parent: Image,
        row_idx: int,
        col_idx: int,
        x: int,
        y: int,
        pixel_data: torch.Tensor,
        is_padded: bool = False,
    ):
        self.parent = parent
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.x = x
        self.y = y
        self.pixel_data = pixel_data
        self.is_padded = is_padded

    @property
    def height(self) -> int:
        return self.pixel_data.shape[-2]

    @property
    def width(self) -> int:
        return self.pixel_data.shape[-1]

    def grid_id(self) -> Tuple[int, int]:
        """Return patch position as (row, col) in the grid."""
        return self.row_idx, self.col_idx

    def pixel_position(self) -> Tuple[int, int]:
        """Return top-left (x, y) pixel coordinates in the original image."""
        return self.x, self.y

    def to_numpy(self) -> np.ndarray:
        """Convert patch pixel data to a NumPy array."""
        return self.pixel_data.detach().cpu().numpy()

    def to_image(self) -> Image:
        """Convert this patch into a new Image instance.

        Clones the parent image and replaces its pixel data with the patch
        tensor.  Follows the same pattern as
        :meth:`~medical_image.data.region_of_interest.RegionOfInterest.load`.

        Returns:
            Image: A new Image object containing only the patch pixels.
        """
        patch_image = self.parent.clone()
        patch_image.pixel_data = self.pixel_data.clone()
        patch_image.file_path = None
        return patch_image

    def load(self) -> Image:
        """Convert this patch into a new Image instance.

        .. deprecated::
            Use :meth:`to_image` instead for consistency with the ROI API.

        Returns:
            Image: A new Image object containing only the patch pixels.
        """
        return self.to_image()

    def __repr__(self):
        return (
            f"Patch[{self.row_idx},{self.col_idx}] "
            f"({self.x},{self.y}) size={self.width}x{self.height}"
        )


class PatchGrid:
    """Divides an :class:`Image` into a regular grid of rectangular patches.

    Automatically pads the image with zeros when its dimensions are not
    evenly divisible by the requested patch size.

    Attributes:
        parent (Image): The source image.
        patch_h (int): Height of each patch in pixels.
        patch_w (int): Width of each patch in pixels.
        patches (list[Patch]): Flat list of all patches (row-major order).
        grid (list[list[Patch]]): 2-D grid indexed as ``grid[row][col]``.
        pad_bottom (int): Number of rows of zero-padding added at the bottom.
        pad_right (int): Number of columns of zero-padding added on the right.
    """

    def __init__(self, parent_image: Image, patch_size: Union[Tuple[int, int], int]):
        """Create a PatchGrid by splitting *parent_image*.

        Args:
            parent_image: The image to divide.  Must have ``pixel_data``
                loaded (i.e. not ``None``).
            patch_size: ``(patch_height, patch_width)`` or a single int for
                square patches.
        """
        self.parent = parent_image
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_h, self.patch_w = patch_size
        self.patches: list[Patch] = []
        self.grid: list[list[Patch]] = []
        self.pad_bottom = 0
        self.pad_right = 0

        self._split()

    @classmethod
    def from_image(
        cls,
        image: Image,
        patch_size: Union[Tuple[int, int], int],
    ) -> PatchGrid:
        """Create a PatchGrid from an :class:`Image`.

        Loads the image lazily if its ``pixel_data`` is ``None``.

        Args:
            image: Source image.
            patch_size: ``(patch_height, patch_width)`` or a single int for
                square patches.

        Returns:
            A new :class:`PatchGrid` covering the entire image.
        """
        if image.pixel_data is None:
            image.load()
        return cls(parent_image=image, patch_size=patch_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_layout(self, img):
        if img.ndim == 2:
            return "HW"

        if img.ndim == 3:
            H, W, C_last = img.shape
            C_first = img.shape[0]

            if C_first in (1, 3, 4):
                return "CHW"
            if C_last in (1, 3, 4):
                return "HWC"

        raise ValueError(f"Cannot determine channel layout for shape {img.shape}")

    def _split(self):
        img = self.parent.pixel_data
        layout = self._infer_layout(img)

        if layout == "HW":
            H, W = img.shape
        elif layout == "CHW":
            _, H, W = img.shape
        elif layout == "HWC":
            H, W, _ = img.shape

        if H % self.patch_h != 0:
            self.pad_bottom = self.patch_h - (H % self.patch_h)
        if W % self.patch_w != 0:
            self.pad_right = self.patch_w - (W % self.patch_w)

        if self.pad_bottom or self.pad_right:
            if layout in ("HW", "CHW"):
                img = torch.nn.functional.pad(
                    img,
                    (0, self.pad_right, 0, self.pad_bottom),
                    mode="constant",
                    value=0,
                )
            elif layout == "HWC":
                img = torch.nn.functional.pad(
                    img,
                    (0, 0, 0, self.pad_right, 0, self.pad_bottom),
                    mode="constant",
                    value=0,
                )

        if layout == "HW":
            H, W = img.shape
        elif layout == "CHW":
            _, H, W = img.shape
        else:
            H, W, _ = img.shape

        num_rows = H // self.patch_h
        num_cols = W // self.patch_w

        for r in range(num_rows):
            row_list = []
            for c in range(num_cols):
                x = r * self.patch_h
                y = c * self.patch_w

                if layout == "HW":
                    patch_tensor = img[x : x + self.patch_h, y : y + self.patch_w]
                elif layout == "CHW":
                    patch_tensor = img[:, x : x + self.patch_h, y : y + self.patch_w]
                elif layout == "HWC":
                    patch_tensor = img[x : x + self.patch_h, y : y + self.patch_w, :]

                is_padded = (r == num_rows - 1 and self.pad_bottom > 0) or (
                    c == num_cols - 1 and self.pad_right > 0
                )

                patch = Patch(
                    parent=self.parent,
                    row_idx=r,
                    col_idx=c,
                    x=x,
                    y=y,
                    pixel_data=patch_tensor,
                    is_padded=is_padded,
                )

                row_list.append(patch)
                self.patches.append(patch)

            self.grid.append(row_list)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def reconstruct(self) -> torch.Tensor:
        """Reassemble the full image tensor from patches (removing padding).

        Returns:
            The reconstructed ``torch.Tensor`` with the same layout as the
            original image.
        """
        layout = self._infer_layout(self.patches[0].pixel_data)

        rows = []
        for row in self.grid:
            rows.append(torch.cat([p.pixel_data for p in row], dim=-1))
        reconstructed = torch.cat(rows, dim=-2)

        if layout == "HW":
            if self.pad_bottom:
                reconstructed = reconstructed[: -self.pad_bottom, :]
            if self.pad_right:
                reconstructed = reconstructed[:, : -self.pad_right]
        elif layout == "CHW":
            if self.pad_bottom:
                reconstructed = reconstructed[:, : -self.pad_bottom, :]
            if self.pad_right:
                reconstructed = reconstructed[:, :, : -self.pad_right]
        elif layout == "HWC":
            if self.pad_bottom:
                reconstructed = reconstructed[: -self.pad_bottom, :, :]
            if self.pad_right:
                reconstructed = reconstructed[:, : -self.pad_right, :]

        return reconstructed

    def to_image(self) -> Image:
        """Reconstruct the full image from patches and return a new Image.

        Removes any padding that was added during splitting, clones the
        parent image, and replaces its pixel data with the reconstructed
        tensor.  Follows the same pattern as
        :meth:`~medical_image.data.region_of_interest.RegionOfInterest.load`.

        Returns:
            A new :class:`Image` containing the reassembled pixel data.
        """
        reconstructed = self.reconstruct()
        result = self.parent.clone()
        result.pixel_data = reconstructed
        return result
