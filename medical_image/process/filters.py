import numpy as np
import torch
from torch.nn import functional as F

from medical_image.data.image import Image


class Filters:
    @staticmethod
    def convolution(image_data: Image, output: Image, kernel, device="cup"):
        """
        Applies a convolution filter to the given image using PyTorch.

        Args:
            image_data (Image): Input image object.
            output (Image): Output image object.
            kernel (np.ndarray or torch.Tensor): 2D convolution kernel.

        Returns:
            None
        """
        image = image_data.pixel_data  # Could be uint16

        # Convert to float32 for convolution
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        else:
            image = image.float()

        # Convert kernel to torch.Tensor
        if not isinstance(kernel, torch.Tensor):
            kernel = torch.from_numpy(kernel).float()
        else:
            kernel = kernel.float()

        # Add batch and channel dimensions
        img = image.unsqueeze(0).unsqueeze(0)  # shape (1,1,H,W)
        k = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,KH,KW)

        # Compute padding
        pad_h = kernel.shape[0] // 2
        pad_w = kernel.shape[1] // 2

        # Apply convolution
        convolved = F.conv2d(F.pad(img, (pad_w, pad_w, pad_h, pad_h)), k)

        # Remove batch/channel dimensions
        convolved = convolved.squeeze(0).squeeze(0)

        # Cast back to original dtype if needed
        if output.pixel_data.dtype != torch.float32:
            convolved = convolved.to(output.pixel_data.dtype)

        # Copy to output
        output.pixel_data[:] = convolved

    @staticmethod
    def gaussian_filter(image_data: Image, output: Image, sigma: float, device="cpu"):
        # Determine kernel size
        size = int(2 * torch.ceil(torch.tensor(3 * sigma)) + 1)

        # Generate kernel
        kernel = Filters._generate_gaussian_kernel(size, sigma, device=device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

        # Prepare input: (1, 1, H, W)
        img = image_data.pixel_data.unsqueeze(0).unsqueeze(0).float()

        # Apply convolution (padding='same')
        padding = size // 2
        filtered = F.conv2d(img, kernel, padding=padding)

        # Remove batch/channel dims
        output.pixel_data = filtered.squeeze(0).squeeze(0)
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def _generate_gaussian_kernel(size: int, sigma: float, device="cpu") -> torch.Tensor:
        k = size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
        y = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()

        return kernel


    @staticmethod
    def median_filter(image_data: Image, output: Image, size: int, device="cpu"):
        """
        Applies a median filter using PyTorch.

        Args:
            image_data (Image): The input image.
            output (Image): Image object to store the filtered result.
            size (int): Odd kernel/window size.
        """
        if size % 2 == 0:
            raise ValueError("Median filter size must be an odd integer.")

        img = image_data.pixel_data.to(device).float()  # (H, W)
        img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        pad = size // 2

        # Pad image (same behavior as NumPy zero padding)
        padded = F.pad(img, (pad, pad, pad, pad), mode="constant", value=0)

        # Extract sliding windows:
        # Unfold height → shape (1, 1, H, W*size)
        # Then unfold width → shape (1, 1, H, W, size*size)
        patches = padded.unfold(2, size, 1).unfold(3, size, 1)
        # patches shape: (1, 1, H, W, size, size)

        # Flatten each window to vector (size*size)
        patches = patches.contiguous().view(1, 1, img.shape[2], img.shape[3], -1)

        # Compute median along last dimension
        filtered = patches.median(dim=-1).values  # shape: (1, 1, H, W)

        # Remove dummy dimensions
        filtered = filtered.squeeze(0).squeeze(0)

        # Save to output image
        output.pixel_data = filtered
        output.width = image_data.width
        output.height = image_data.height


    @staticmethod
    def butterworth_kernel(image_data: Image, output: Image, D_0: float = 21, W: float = 32, n: int = 3, device="cpu"):
        """
        Applies a Butterworth band-pass filter in the frequency domain.

        Parameters:
            image_data (Image): The input image (width/height define coordinate grid).
            output (Image): Image object where the kernel will be stored.
            D_0 (float): Cutoff frequency.
            W (float): Bandwidth of the filter.
            n (int): Filter order.
            device (String): device to run in (cpu or cuda).

        Produces:
            output.pixel_data: (H, W) Butterworth filter tensor.
        """

        height, width = image_data.height, image_data.width

        # Create coordinate grid centered at image center
        u = torch.arange(width, device=device, dtype=torch.float32)
        v = torch.arange(height, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        # Distance from center in the frequency domain
        D = torch.sqrt((uu - width / 2) ** 2 + (vv - height / 2) ** 2)

        # Butterworth band-pass formula
        band = D ** 2 - D_0 ** 2
        cutoff = 8 * W * D

        # Avoid division by zero (for D = 0)
        cutoff = torch.where(cutoff == 0, torch.tensor(1e-6, device=device), cutoff)

        denominator = 1 + (band / cutoff).pow(2 * n)

        # Final filter kernel (H, W)
        kernel = 1.0 / denominator

        # Save result
        output.pixel_data = kernel
        output.width = width
        output.height = height

    import torch

    @staticmethod
    def difference_of_gaussian(
            image_data: Image,
            output: Image,
            sigma_1: float,
            sigma_2: float,
            device="cpu",
    ):
        """
        Applies the Difference of Gaussian (DoG) filter to an image.

        Args:
            image_data (Image): Input image.
            output (Image): Image object to store the result.
            sigma_1 (float): Standard deviation of the first Gaussian kernel.
            sigma_2 (float): Standard deviation of the second Gaussian kernel.
        """


        # Prepare temporary images
        g1_img = type(image_data)(image_data.file_path)
        g2_img = type(image_data)(image_data.file_path)

        # Copy metadata (width/height are needed)
        g1_img.width = image_data.width
        g1_img.height = image_data.height
        g1_img.device = device

        g2_img.width = image_data.width
        g2_img.height = image_data.height
        g2_img.device = device

        # Apply Gaussian filters (your PyTorch version writes directly into pixel_data)
        Filters.gaussian_filter(image_data, g1_img, sigma_1, device)
        Filters.gaussian_filter(image_data, g2_img, sigma_2, device)

        # Difference of Gaussians
        dog = g1_img.pixel_data - g2_img.pixel_data

        # Save to output
        output.pixel_data = dog
        output.width = image_data.width
        output.height = image_data.height

    import torch
    import torch.nn.functional as F

    @staticmethod
    def laplacian_of_gaussian(image_data: Image, output: Image, sigma: float, device="cpu"):
        """
        Applies the Laplacian of Gaussian (LoG) filter using PyTorch.

        Args:
            image_data (Image): The input image.
            output (Image): Image object to store the LoG result.
            sigma (float): Standard deviation of the Gaussian kernel.
        """

        # ----- Step 1: Gaussian Blur (same API as your Gaussian filter) -----
        blurred_img = type(image_data)(image_data.file_path)
        blurred_img.width = image_data.width
        blurred_img.height = image_data.height
        blurred_img.device = device

        Filters.gaussian_filter(image_data, blurred_img, sigma)

        # Blurred pixel data (H, W)
        g = blurred_img.pixel_data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # ----- Step 2: Laplacian Kernel -----
        # Standard 3×3 Laplacian
        lap_kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)

        # Convolution gives the second derivative ∇²(g)
        lap = F.conv2d(g, lap_kernel, padding=1)

        # Back to (H, W)
        lap = lap.squeeze(0).squeeze(0)

        # ----- Step 3: Save result -----
        output.pixel_data = lap
        output.width = image_data.width
        output.height = image_data.height

    @staticmethod
    def gamma_correction(image_data: Image, output: Image, gamma: float, device="cpu"):
        """
        Applies Gamma Correction to the given image.

        Args:
            image_data (Image): The input image data encapsulated in an Image object.
            output (Image): An Image object to store the corrected image.
            gamma (float): The gamma correction value.
            device (str): The device to perform computation on ("cpu" or "cuda").

        Returns:
            None
        """
        # Move image to the specified device
        image = image_data.pixel_data.to(device)

        # Normalize, apply gamma, and rescale
        corrected = torch.pow(image / 4095.0, gamma) * 4095.0

        # Store result in output, keeping device consistent
        output.pixel_data = corrected.to(image_data.device)

    @staticmethod
    def ContrastAdjust(image_data: Image, output: Image, contrast, brightness):
        """This function used to adjust the contrast and the brightness
        the image in the Dicom file.
            This approach is based this equation that can be used to apply
        both contrast and brightness at the same time:
            ****
            new_img = alpha*old_img + beta
            ****
            Where alpha and beta are contrast and brightness coefficient respectively:

            alpha 1  beta 0      --> no change
            0 < alpha < 1        --> lower contrast
            alpha > 1            --> higher contrast
            -2047 < beta < +2047   --> good range for brightness values

            In my case:
                alpha = contrast / 2047 + 1
                beta = brightness - contrast

        Args:
            pixel_array: pixel_array in the Dicom file.
            contrast: the contrast value to be applied.
            brightness: the brightness value to be applied.

        Returns:
            The return value is an image with different contrast and brightness.

        """
        image = image_data.pixel_data
        quotient = 4095 // 2
        alpha = contrast / quotient + 1
        beta = brightness - contrast

        output.pixel_data = np.clip(image * alpha + beta, 0, 4095)
