import torch

from medical_image.data.image import Image


class Patch:
    def __init__(
        self,
        parent: Image,
        row_idx: int,
        col_idx: int,
        x: int,
        y: int,
        pixel_data: int,
        is_padded=False,
    ):
        """
        Represents a patch extracted from an image.

        Args:
            parent (Image): Reference to the parent Image object
            row_idx (int): Patch index in the vertical grid
            col_idx (int): Patch index in the horizontal grid
            x (int): Top-left pixel x-coordinate in original image
            y (int): Top-left pixel y-coordinate in original image
            pixel_data (torch.Tensor): Patch pixels
            is_padded (bool): True if patch includes padding
        """
        self.parent = parent
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.x = x
        self.y = y
        self.pixel_data = pixel_data
        self.is_padded = is_padded

        self.height = pixel_data.shape[-2]
        self.width = pixel_data.shape[-1]

    def grid_id(self):
        """Return patch position as (row, col)."""
        return (self.row_idx, self.col_idx)

    def pixel_position(self):
        """Return (x, y) pixel coordinates in original image."""
        return (self.x, self.y)

    def to_numpy(self):
        return self.pixel_data.detach().cpu().numpy()

    def __repr__(self):
        return f"Patch[{self.row_idx},{self.col_idx}] ({self.x},{self.y}) size={self.width}x{self.height}"


class PatchGrid:
    def __init__(self, parent_image, patch_size):
        """
        Manages a full grid of patches.

        Args:
            parent_image (Image): The image to divide
            patch_size (tuple): (patch_height, patch_width)
        """
        self.parent = parent_image
        self.patch_h, self.patch_w = patch_size
        self.patches = []  # flat list
        self.grid = []  # 2D grid-like structure
        self.pad_bottom = 0
        self.pad_right = 0

        self._split()

    def _split(self):
        img = self.parent.pixel_data

        H, W = img.shape[-2], img.shape[-1]

        # Compute padding (same logic as divide_raster_band)
        if H % self.patch_h != 0:
            self.pad_bottom = self.patch_h - (H % self.patch_h)
        if W % self.patch_w != 0:
            self.pad_right = self.patch_w - (W % self.patch_w)

        if self.pad_bottom or self.pad_right:
            img = torch.nn.functional.pad(
                img, (0, self.pad_right, 0, self.pad_bottom), mode="constant", value=0
            )

        new_H, new_W = img.shape[-2], img.shape[-1]
        num_rows = new_H // self.patch_h
        num_cols = new_W // self.patch_w

        # Extract patches with coordinates
        for r in range(num_rows):
            row_list = []
            for c in range(num_cols):
                x = r * self.patch_h
                y = c * self.patch_w

                patch_tensor = img[:, x : x + self.patch_h, y : y + self.patch_w]

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

    def reconstruct(self):
        """Reassemble the full image from patches (removing padding)."""
        rows = []
        for row in self.grid:
            rows.append(torch.cat([p.pixel_data for p in row], dim=-1))
        reconstructed = torch.cat(rows, dim=-2)

        # remove padding
        if self.pad_bottom:
            reconstructed = reconstructed[:, : -self.pad_bottom, :]
        if self.pad_right:
            reconstructed = reconstructed[:, :, : -self.pad_right]

        return reconstructed


# TODO: add unit test with plots for this patch detection

# TODO: add these to Patch class
#   is_empty() (useful for padded patches)
#   to_mask() if patch includes segmentation labels
#   reproject_to_parent() → maps patch coordinates back to original image
#   and produce:
#   A SlidingWindowPatchGrid for overlapping patches (stride < patch_size)
#   and GPU-accelerated patch extraction using torch.unfold
