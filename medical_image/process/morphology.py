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
        while img.ndim > 2:
            img = img.squeeze(0)

        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {img.shape}")
        H, W = image_data.height, image_data.width

        # Ensure shape (1,1,H,W)
        img = img.unsqueeze(0).unsqueeze(0)

        pad = kernel_size // 2

        # -----------------------
        # DILATION (max pooling)
        # -----------------------
        dilated = F.max_pool2d(
            F.pad(img, (pad, pad, pad, pad), mode="constant", value=0),
            kernel_size,
            stride=1,
        )

        # -----------------------
        # EROSION (true min pooling)
        # -----------------------
        eroded = -F.max_pool2d(
            F.pad(-dilated, (pad, pad, pad, pad), mode="constant", value=0),
            kernel_size,
            stride=1,
        )

        # Crop back to original size
        closed = eroded[:, :, :H, :W]

        # Remove batch/channel dims
        closed = closed.squeeze(0).squeeze(0)

        # Store result as int64
        output.pixel_data = closed.to(torch.int64)
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

    @staticmethod
    def _disk_footprint(radius: int, device: str = "cpu") -> torch.Tensor:
        """
        Create a flat circular disk structuring element.

        Args:
            radius: Radius of the disk.
            device: Torch device.

        Returns:
            (2*radius+1, 2*radius+1) bool tensor.
        """
        size = 2 * radius + 1
        y, x = torch.meshgrid(
            torch.arange(size, device=device) - radius,
            torch.arange(size, device=device) - radius,
            indexing="ij",
        )
        return (x ** 2 + y ** 2 <= radius ** 2).float()

    @staticmethod
    def erosion(
        image_data: Image, output: Image, radius: int = 4, device: str = "cpu"
    ):
        """
        Grayscale erosion using a flat disk SE.

        Erosion = local minimum under the structuring element.
        Implemented as: -max_pool2d(-img, footprint).

        Args:
            image_data: Input Image (2D float).
            output: Output Image to store result.
            radius: Disk SE radius.
            device: Torch device.
        """
        img = image_data.pixel_data.to(device).float()
        while img.ndim > 2:
            img = img.squeeze(0)
        H, W = img.shape

        kernel_size = 2 * radius + 1
        pad = radius

        # Erosion = -max_pool(-img)
        neg_img = (-img).unsqueeze(0).unsqueeze(0)
        neg_padded = F.pad(neg_img, (pad, pad, pad, pad), mode="constant", value=0)
        neg_max = F.max_pool2d(neg_padded, kernel_size, stride=1)
        eroded = (-neg_max).squeeze(0).squeeze(0)[:H, :W]

        output.pixel_data = eroded
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def dilation(
        image_data: Image, output: Image, radius: int = 4, device: str = "cpu"
    ):
        """
        Grayscale dilation using a flat disk SE.

        Dilation = local maximum under the structuring element.

        Args:
            image_data: Input Image (2D float).
            output: Output Image to store result.
            radius: Disk SE radius.
            device: Torch device.
        """
        img = image_data.pixel_data.to(device).float()
        while img.ndim > 2:
            img = img.squeeze(0)
        H, W = img.shape

        kernel_size = 2 * radius + 1
        pad = radius

        img4d = img.unsqueeze(0).unsqueeze(0)
        padded = F.pad(img4d, (pad, pad, pad, pad), mode="constant", value=0)
        dilated = F.max_pool2d(padded, kernel_size, stride=1)
        dilated = dilated.squeeze(0).squeeze(0)[:H, :W]

        output.pixel_data = dilated
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def white_top_hat(
        image_data: Image, output: Image, radius: int = 4, device: str = "cpu"
    ):
        """
        White Top-Hat transform: TopHat(I) = I - opening(I).

        Opening = dilation(erosion(I)). Highlights bright structures smaller
        than the structuring element (microcalcifications).

        MATLAB reference: SE = strel('disk', 4); ROI = imtophat(I_ROI, SE);

        Args:
            image_data: Input Image (2D float, e.g. normalized to [0,1]).
            output: Output Image to store result.
            radius: Disk SE radius (default 4 → 9×9, matching MATLAB).
            device: Torch device.
        """
        import copy
        # Step 1: Erosion
        eroded = copy.deepcopy(image_data)
        MorphologyOperations.erosion(image_data, eroded, radius=radius, device=device)

        # Step 2: Dilation of eroded → opening
        opened = copy.deepcopy(eroded)
        MorphologyOperations.dilation(eroded, opened, radius=radius, device=device)

        # Step 3: Top-Hat = I - opening
        img = image_data.pixel_data.to(device).float()
        while img.ndim > 2:
            img = img.squeeze(0)

        th = torch.clamp(img - opened.pixel_data, min=0.0)

        output.pixel_data = th
        output.width = image_data.width
        output.height = image_data.height
