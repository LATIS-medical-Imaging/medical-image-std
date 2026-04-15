import torch
import torch.nn.functional as F

from medical_image.data.image import Image, requires_loaded
from medical_image.utils.device import resolve_device
from medical_image.utils.logging import logger


def _safe_to_device(tensor: torch.Tensor, device: torch.device) -> tuple:
    """Move tensor to device with CUDA error fallback to CPU.

    Returns:
        (tensor_on_device, actual_device)
    """
    try:
        return tensor.to(device).float(), device
    except (RuntimeError, torch.cuda.CudaError, torch.AcceleratorError) as e:
        if device.type != "cpu":
            logger.warning(
                "CUDA error moving tensor to %s, falling back to CPU: %s",
                device,
                e,
            )
            torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            return tensor.to(cpu).float(), cpu
        raise


class MorphologyOperations:
    @staticmethod
    @requires_loaded
    def morphology_closing(
        image: Image, output: Image, kernel_size: int = 7, device=None
    ) -> Image:
        """
        Performs 2D binary closing on a given image using PyTorch.

        Closing = Dilation followed by Erosion with the same structuring element.

        Args:
            image: Input binary image (0/1).
            output: Output Image object to store the result.
            kernel_size: Size of the square structuring element.
            device: Device for computation (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img, device = _safe_to_device(image.pixel_data, device)
        while img.ndim > 2:
            img = img.squeeze(0)

        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {img.shape}")
        H, W = image.height, image.width

        img = img.unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2

        # DILATION (max pooling)
        dilated = F.max_pool2d(
            F.pad(img, (pad, pad, pad, pad), mode="constant", value=0),
            kernel_size,
            stride=1,
        )

        # EROSION (true min pooling)
        eroded = -F.max_pool2d(
            F.pad(-dilated, (pad, pad, pad, pad), mode="constant", value=0),
            kernel_size,
            stride=1,
        )

        closed = eroded[:, :, :H, :W]
        closed = closed.squeeze(0).squeeze(0)

        output.pixel_data = closed.to(torch.int64)
        return output

    @staticmethod
    @requires_loaded
    def region_fill(image: Image, output: Image, device=None) -> Image:
        """
        Fills holes in a binary image using PyTorch.

        Args:
            image: Input binary image (0/1).
            output: Output Image object to store the filled result.
            device: Device for computation (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img, device = _safe_to_device(image.pixel_data, device)
        h, w = img.shape
        mask = torch.zeros((h + 2, w + 2), device=device)
        mask[1:-1, 1:-1] = 1 - img

        prev_mask = torch.zeros_like(mask)
        struct_elem = torch.ones((3, 3), device=device)

        while not torch.equal(mask, prev_mask):
            prev_mask = mask.clone()
            dilated = F.conv2d(
                mask.unsqueeze(0).unsqueeze(0),
                struct_elem.unsqueeze(0).unsqueeze(0),
                padding=1,
            )
            mask = (dilated > 0).float().squeeze(0).squeeze(0)

        filled = img + (1 - mask[1:-1, 1:-1])
        filled = (filled > 0).float()

        output.pixel_data = filled
        return output

    @staticmethod
    def _disk_footprint(radius: int, device=None) -> torch.Tensor:
        """
        Create a flat circular disk structuring element.

        Args:
            radius: Radius of the disk.
            device: Torch device.

        Returns:
            (2*radius+1, 2*radius+1) float tensor.
        """
        device = device or torch.device("cpu")
        size = 2 * radius + 1
        y, x = torch.meshgrid(
            torch.arange(size, device=device) - radius,
            torch.arange(size, device=device) - radius,
            indexing="ij",
        )
        return (x**2 + y**2 <= radius**2).float()

    @staticmethod
    @requires_loaded
    def erosion(image: Image, output: Image, radius: int = 4, device=None) -> Image:
        """
        Grayscale erosion using a flat disk SE.

        Args:
            image: Input Image (2D float).
            output: Output Image to store result.
            radius: Disk SE radius.
            device: Torch device (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img, device = _safe_to_device(image.pixel_data, device)
        while img.ndim > 2:
            img = img.squeeze(0)
        H, W = img.shape

        kernel_size = 2 * radius + 1
        pad = radius

        try:
            neg_img = (-img).unsqueeze(0).unsqueeze(0)
            neg_padded = F.pad(neg_img, (pad, pad, pad, pad), mode="constant", value=0)
            neg_max = F.max_pool2d(neg_padded, kernel_size, stride=1)
            eroded = (-neg_max).squeeze(0).squeeze(0)[:H, :W]
        except (RuntimeError, torch.cuda.CudaError, torch.AcceleratorError):
            if device.type != "cpu":
                logger.warning("CUDA error in erosion, falling back to CPU")
                torch.cuda.empty_cache()
                img = img.cpu()
                neg_img = (-img).unsqueeze(0).unsqueeze(0)
                neg_padded = F.pad(
                    neg_img, (pad, pad, pad, pad), mode="constant", value=0
                )
                neg_max = F.max_pool2d(neg_padded, kernel_size, stride=1)
                eroded = (-neg_max).squeeze(0).squeeze(0)[:H, :W]
            else:
                raise

        output.pixel_data = eroded
        return output

    @staticmethod
    @requires_loaded
    def dilation(image: Image, output: Image, radius: int = 4, device=None) -> Image:
        """
        Grayscale dilation using a flat disk SE.

        Args:
            image: Input Image (2D float).
            output: Output Image to store result.
            radius: Disk SE radius.
            device: Torch device (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        img, device = _safe_to_device(image.pixel_data, device)
        while img.ndim > 2:
            img = img.squeeze(0)
        H, W = img.shape

        kernel_size = 2 * radius + 1
        pad = radius

        try:
            img4d = img.unsqueeze(0).unsqueeze(0)
            padded = F.pad(img4d, (pad, pad, pad, pad), mode="constant", value=0)
            dilated = F.max_pool2d(padded, kernel_size, stride=1)
            dilated = dilated.squeeze(0).squeeze(0)[:H, :W]
        except (RuntimeError, torch.cuda.CudaError, torch.AcceleratorError):
            if device.type != "cpu":
                logger.warning("CUDA error in dilation, falling back to CPU")
                torch.cuda.empty_cache()
                img = img.cpu()
                img4d = img.unsqueeze(0).unsqueeze(0)
                padded = F.pad(img4d, (pad, pad, pad, pad), mode="constant", value=0)
                dilated = F.max_pool2d(padded, kernel_size, stride=1)
                dilated = dilated.squeeze(0).squeeze(0)[:H, :W]
            else:
                raise

        output.pixel_data = dilated
        return output

    @staticmethod
    @requires_loaded
    def white_top_hat(
        image: Image, output: Image, radius: int = 4, device=None
    ) -> Image:
        """
        White Top-Hat transform: TopHat(I) = I - opening(I).

        Opening = dilation(erosion(I)). Highlights bright structures smaller
        than the structuring element (microcalcifications).

        Args:
            image: Input Image (2D float, e.g. normalized to [0,1]).
            output: Output Image to store result.
            radius: Disk SE radius (default 4 -> 9x9, matching MATLAB).
            device: Torch device (None = infer from image).

        Returns:
            The output Image.
        """
        device = resolve_device(image, explicit=device)
        # Step 1: Erosion
        eroded = image.clone()
        MorphologyOperations.erosion(image, eroded, radius=radius, device=device)

        # Step 2: Dilation of eroded -> opening
        opened = eroded.clone()
        MorphologyOperations.dilation(eroded, opened, radius=radius, device=device)

        # Step 3: Top-Hat = I - opening
        img, device = _safe_to_device(image.pixel_data, device)
        while img.ndim > 2:
            img = img.squeeze(0)

        opened_data = opened.pixel_data.to(img.device)
        th = torch.clamp(img - opened_data, min=0.0)

        output.pixel_data = th
        return output
