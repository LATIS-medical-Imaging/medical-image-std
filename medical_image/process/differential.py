import torch
import torch.nn.functional as F

from medical_image.data.image import Image
from medical_image.data.image import requires_loaded
from medical_image.utils.device import resolve_device


class DifferentialOperations:

    @staticmethod
    @requires_loaded
    def hessian(
        image: Image,
        output: Image,
        device=None,
    ) -> Image:
        """
        Compute the 2D Hessian matrix.

        Returns:

            [3, H, W]

        Channels:

            0 -> Ixx
            1 -> Ixy
            2 -> Iyy
        """

        device = resolve_device(
            image,
            explicit=device,
        )

        img = image.pixel_data.to(
            device
        ).float()

        while img.ndim > 2:
            img = img.squeeze(0)

        if img.ndim != 2:
            raise ValueError(
                f"Expected 2D image, got {img.shape}"
            )

        # Second derivative kernels.
        #
        # Ixx:
        #     [1, -2, 1]
        #
        # Iyy:
        #     transpose
        #
        # Ixy:
        #     d/dx(d/dy)
        #

        dtype = img.dtype

        k_xx = torch.tensor(
            [
                [0., 0., 0.],
                [1., -2., 1.],
                [0., 0., 0.],
            ],
            device=device,
            dtype=dtype,
        )

        k_yy = torch.tensor(
            [
                [0., 1., 0.],
                [0., -2., 0.],
                [0., 1., 0.],
            ],
            device=device,
            dtype=dtype,
        )

        k_xy = torch.tensor(
            [
                [1., 0., -1.],
                [0., 0., 0.],
                [-1., 0., 1.],
            ],
            device=device,
            dtype=dtype,
        ) / 4.0

        kernels = torch.stack(
            [
                k_xx,
                k_xy,
                k_yy,
            ],
            dim=0,
        ).unsqueeze(1)

        img4d = img.unsqueeze(0).unsqueeze(0)

        padded = F.pad(
            img4d,
            (1, 1, 1, 1),
            mode="reflect",
        )

        hessian = F.conv2d(
            padded,
            kernels,
            stride=1,
        )

        output.pixel_data = (
            hessian
            .squeeze(0)
        )

        return output

    @staticmethod
    def hessian_eigenvalues(
            hessian: torch.Tensor,
    ):
        """
        Compute eigenvalues of a 2D symmetric Hessian.

        Args:
            hessian:
                Tensor [3, H, W]

        Returns:
            lambda1:
                [H, W]

            lambda2:
                [H, W]
        """

        if hessian.ndim != 3:
            raise ValueError(
                "Expected Hessian shape [3,H,W]."
            )

        Ixx = hessian[0]
        Ixy = hessian[1]
        Iyy = hessian[2]

        trace = (
                        Ixx + Iyy
                ) / 2.0

        discriminant = torch.sqrt(
            (
                    (Ixx - Iyy) / 2.0
            ).square()
            + Ixy.square()
            + 1e-12
        )

        lambda1 = trace + discriminant
        lambda2 = trace - discriminant

        return lambda1, lambda2