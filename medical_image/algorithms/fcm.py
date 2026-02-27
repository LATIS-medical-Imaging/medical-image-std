import copy
from typing import Optional, List

import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.utils.image_utils import MathematicalOperations


class FCMAlgorithm(Algorithm):
    """
    Fuzzy C-Means (FCM) clustering algorithm for microcalcification segmentation.

    References:
        @article{quintanilla2011image,
          title={Image segmentation by fuzzy and possibilistic clustering algorithms for the identification of microcalcifications},
          author={Quintanilla-Dom{\\'i}nguez, Joel and Ojeda-Maga{\\~n}a, Benjam{\\'i}n and Cortina-Januchs, Maria Guadalupe and Ruelas, Rub{\\'e}n and Vega-Corona, Antonio and Andina, Diego},
          journal={Scientia Iranica},
          volume={18},
          number={3},
          pages={580--589},
          year={2011},
          publisher={Elsevier}
        }

    Math and Logic:
        FCM clusters data points by assigning a fuzzy membership degree to each cluster.
        It minimizes an objective function based on the distance between pixels and
        cluster centroids, weighted by their membership degree.
        The output is a binary mask where pixels assigned to the brightest cluster
        (highest intensity centroid) are marked as microcalcification candidates.

    Pipeline:
        1. Flatten the input image into a 1D feature matrix.
        2. Randomly initialize the fuzzy membership matrix.
        3. Iteratively compute distances, update membership probabilities, and update cluster centroids.
        4. Stop when the membership changes fall below a tolerance or max iterations are reached.
        5. Build a quantized output image and isolate the brightest cluster as the mask.

    Example Usage:
        ```python
        import copy
        import numpy as np
        import matplotlib.pyplot as plt
        from medical_image.algorithms.fcm import FCMAlgorithm
        from medical_image.data.dicom_image import DicomImage

        # Load and prepare image
        img = DicomImage("20527054.dcm")
        img.load()

        # Initialize algorithm and output
        algo = FCMAlgorithm(c=3, device="cpu")
        output = copy.deepcopy(img)

        # Apply algorithm
        algo(img, output)

        # Plot output
        plt.imshow(output.pixel_data.numpy(), cmap='gray')
        plt.title('FCM Output')
        plt.show()
        ```

    Attributes after apply():
        centroids:    (c, d) cluster centroids.
        membership:   (c, N) fuzzy membership matrix U.
        labels:       (H, W) int hard cluster assignments.
        quantized:    (H, W) float quantized image (pixel -> centroid / max).
        stats:        List of dicts with cluster statistics.
    """

    def __init__(
        self,
        c: int = 2,
        m: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-3,
        random_state: int = 42,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.c = c
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Operations (FEBDS-style lambdas)
        self.compute_distances = (
            lambda Z, V: MathematicalOperations.euclidean_distance_sq(Z=Z, V=V)
        )
        self.update_membership = lambda D2: FCMAlgorithm._update_membership(
            D2=D2, m=self.m
        )
        self.update_centroids = lambda Z, U: FCMAlgorithm._update_centroids(
            Z=Z, U=U, m=self.m
        )

        # Results (populated by apply)
        self.centroids: Optional[torch.Tensor] = None
        self.membership: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.quantized: Optional[torch.Tensor] = None
        self.n_iter: int = 0
        self.converged: bool = False
        self.stats: Optional[List[dict]] = None
        self.mc_label: int = -1

    # ------------------------------------------------------------------
    # Core FCM update steps
    # ------------------------------------------------------------------

    @staticmethod
    def _update_membership(D2: torch.Tensor, m: float) -> torch.Tensor:
        """FCM membership update: U_{ij} ∝ 1/d_{ij}^{2/(m-1)}. → (c, N)"""
        eps = 1e-10
        exp = 1.0 / (m - 1.0)
        inv_D = (D2 + eps) ** (-exp)
        return inv_D / (inv_D.sum(dim=0, keepdim=True) + eps)

    @staticmethod
    def _update_centroids(Z: torch.Tensor, U: torch.Tensor, m: float) -> torch.Tensor:
        """FCM centroid update: V_i = Σ u^m_ij z_j / Σ u^m_ij. → (c, d)"""
        Um = U**m
        denom = Um.sum(dim=1, keepdim=True) + 1e-10
        return (Um @ Z) / denom

    @staticmethod
    def _build_quantized(
        labels: torch.Tensor, centroids: torch.Tensor, image_shape: tuple
    ) -> torch.Tensor:
        """Build quantized greyscale image: pixel → centroid / max_centroid."""
        vals = centroids[:, 0].clone()
        mx = vals.max()
        if mx > 0:
            vals = vals / mx
        return vals[labels.reshape(image_shape)]

    # ------------------------------------------------------------------
    # Main apply
    # ------------------------------------------------------------------

    def apply(self, image: Image, output: Image):
        """
        Apply FCM clustering.

        Pipeline:
            1. Flatten image → feature matrix Z (N, 1)
            2. Random initialisation of membership U
            3. Iterate: distances → membership → centroids
            4. Build quantized image + binary MC mask

        Args:
            image: Input Image (2D float tensor, e.g. top-hat result).
            output: Output Image — pixel_data = binary MC mask.
        """
        device = self.device
        img = image.pixel_data.to(device).float()
        while img.ndim > 2:
            img = img.squeeze(0)

        H, W = img.shape
        image_shape = (H, W)
        N = H * W

        # Step 1: Flatten
        Z = img.reshape(N, 1)

        # Step 2: Random initialisation
        torch.manual_seed(self.random_state)
        U = torch.rand(self.c, N, device=device)
        U = U / (U.sum(dim=0, keepdim=True) + 1e-10)

        V = self.update_centroids(Z, U)

        # Step 3: Iterate
        converged = False
        n_iter = 0

        for iteration in range(self.max_iter):
            D2 = self.compute_distances(Z, V)
            U_new = self.update_membership(D2)
            V = self.update_centroids(Z, U_new)

            diff = float(torch.norm(U_new - U))
            U = U_new
            n_iter = iteration + 1

            if diff < self.tol:
                converged = True
                break

        # Step 4: Build outputs
        self.centroids = V
        self.membership = U
        self.n_iter = n_iter
        self.converged = converged

        labels = torch.argmax(U, dim=0).to(torch.int64)
        self.labels = labels.reshape(image_shape)
        self.quantized = self._build_quantized(labels, V, image_shape)
        self.mc_label = int(torch.argmax(V[:, 0]))

        mc_mask = (self.quantized == self.quantized.max()).float()

        self.stats = []
        for i in range(self.c):
            cluster_mask = self.labels == i
            self.stats.append(
                {
                    "id": i,
                    "centroid": float(V[i, 0]),
                    "pixels": int(cluster_mask.sum()),
                    "is_mc": (i == self.mc_label),
                }
            )

        output.pixel_data = mc_mask
        output.width = W
        output.height = H
