import torch
import torch.nn.functional as F

from medical_image.data.image import Image


class MorphologyOperations:
    @staticmethod
    def morphology_closing(
        image_data: Image, output: Image, kernel_size: int = 7, device="cpu"
    ):
        """
        Performs 2D binary closing on a given image using PyTorch.

        Closing = Dilation followed by Erosion with the same structuring element.
        Works for single-channel images (H,W) or (C,H,W).

        Args:
            image_data (Image): Input binary image (0/1).
            output (Image): Output Image object to store the result.
            kernel_size (int): Size of the square structuring element.
            device (str): Device for computation ("cpu" or "cuda").

        Returns:
            None: Result stored in `output.pixel_data`.
        """
        # Ensure image is float tensor on the correct device
        img = image_data.pixel_data.to(device).float()

        # Make sure img is 4D for conv2d: (N=1, C=1, H, W) or (N=1, C, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif img.ndim == 3:
            img = img.unsqueeze(0)  # (1,C,H,W)

        # Ensure binary (0/1)
        img = (img > 0).float()

        # Structuring element
        se = torch.ones((img.shape[1], 1, kernel_size, kernel_size), device=device)

        pad = kernel_size // 2

        # Dilation
        dilated = F.conv2d(F.pad(img, (pad, pad, pad, pad)), se, groups=img.shape[1])
        dilated = (dilated > 0).float()

        # Erosion (using complement trick)
        eroded = F.conv2d(
            F.pad(1 - dilated, (pad, pad, pad, pad)), se, groups=img.shape[1]
        )
        eroded = (eroded < kernel_size * kernel_size).float()
        closed = 1 - eroded

        # Remove batch dimension, keep channels if multi-channel
        if closed.shape[1] == 1:
            closed = closed.squeeze(0).squeeze(0)  # (H,W)
        else:
            closed = closed.squeeze(0)  # (C,H,W)

        output.pixel_data = closed
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def region_fill(image_data: Image, output: Image, device="cpu"):
        """
        Fills holes in a binary image using PyTorch.

        The algorithm performs binary dilation of the complement from the image boundary
        to fill all interior holes not connected to the border.

        Args:
            image_data (Image): Input binary image (0/1).
            output (Image): Output Image object to store the filled result.
            device (str): Device for computation ("cpu" or "cuda").

        Returns:
            None: The result is stored in `output.pixel_data`.
        """
        img = image_data.pixel_data.to(device).float()
        h, w = img.shape
        # Create mask for flood fill: start from boundary
        mask = torch.zeros((h + 2, w + 2), device=device)
        mask[1:-1, 1:-1] = 1 - img  # complement

        prev_mask = torch.zeros_like(mask)
        struct_elem = torch.ones((3, 3), device=device)

        # Iterative flood fill until no change
        while not torch.equal(mask, prev_mask):
            prev_mask = mask.clone()
            dilated = F.conv2d(
                mask.unsqueeze(0).unsqueeze(0),
                struct_elem.unsqueeze(0).unsqueeze(0),
                padding=1,
            )
            mask = (dilated > 0).float().squeeze(0).squeeze(0)

        # Filled image: original OR complement of invaded region
        filled = img + (1 - mask[1:-1, 1:-1])
        filled = (filled > 0).float()

        output.pixel_data = filled
        output.width = image_data.width
        output.height = image_data.height
