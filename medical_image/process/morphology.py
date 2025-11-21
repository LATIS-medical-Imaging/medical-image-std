import torch
import torch.nn.functional as F

from medical_image.data.image import Image


class MorphologyOperations:
    @staticmethod
    def morphology_closing(
        image: Image, output: Image, kernel_size: int = 7, device="cpu"
    ):
        """
        Performs 2D binary closing on a given image using PyTorch.

        Closing is defined as a *dilation followed by an erosion* with the same structuring element.
        This operation fills small holes and smooths boundaries in binary images.

        Args:
            image (Image): Input binary image (0/1).
            output (Image): Output Image object to store the result.
            kernel_size (int): Size of the square structuring element. Default is 7.
            device (str): Device for computation ("cpu" or "cuda").

        Returns:
            None: The result is stored in `output.pixel_data`.
        """
        img = image.pixel_data.to(device).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        se = torch.ones((1, 1, kernel_size, kernel_size), device=device)

        # Dilation
        pad = kernel_size // 2
        dilated = F.conv2d(F.pad(img, (pad, pad, pad, pad)), se, padding=0)
        dilated = (dilated > 0).float()

        # Erosion
        # Use complement trick: erosion = 1 - dilation(1 - img)
        dilated_padded = F.pad(1 - dilated, (pad, pad, pad, pad))
        eroded = F.conv2d(dilated_padded, se, padding=0)
        eroded = (eroded < kernel_size * kernel_size).float()
        closed = 1 - eroded

        output.pixel_data = closed.squeeze(0).squeeze(0)
        output.width = image.width
        output.height = image.height

    @staticmethod
    def region_fill(image: Image, output: Image, device="cpu"):
        """
        Fills holes in a binary image using PyTorch.

        The algorithm performs binary dilation of the complement from the image boundary
        to fill all interior holes not connected to the border.

        Args:
            image (Image): Input binary image (0/1).
            output (Image): Output Image object to store the filled result.
            device (str): Device for computation ("cpu" or "cuda").

        Returns:
            None: The result is stored in `output.pixel_data`.
        """
        img = image.pixel_data.to(device).float()
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
        output.width = image.width
        output.height = image.height
