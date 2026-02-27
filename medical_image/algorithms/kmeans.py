import copy
from typing import Optional, List

import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.utils.image_utils import MathematicalOperations


class KMeansAlgorithm(Algorithm):
    """
    K-Means clustering algorithm for microcalcification segmentation.

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
        K-Means partitions image pixels into K distinct, non-overlapping clusters based on
        pixel intensity. It iteratively assigns each pixel to the nearest cluster centroid
        and updates the centroids based on the mean of the assigned pixels.
        The output is a binary mask where pixels in the brightest cluster are marked
        as microcalcification candidates.

    Pipeline:
        1. Flatten the input image into a 1D feature matrix.
        2. Initialize centroids using k-means++ for better convergence.
        3. Iteratively assign pixels to the nearest centroid and update centroids.
        4. Stop when the shift in centroids falls below a tolerance or max iterations are reached.
        5. Build a quantized output image and isolate the brightest cluster as the mask.

    Example Usage:
        ```python
        import copy
        import numpy as np
        import matplotlib.pyplot as plt
        from medical_image.algorithms.kmeans import KMeansAlgorithm
        from medical_image.data.dicom_image import DicomImage

        # Load and prepare image
        img = DicomImage("20527054.dcm")
        img.load()

        # Initialize algorithm and output
        algo = KMeansAlgorithm(k=3, device="cpu")
        output = copy.deepcopy(img)

        # Apply algorithm
        algo(img, output)

        # Plot output
        plt.imshow(output.pixel_data.numpy(), cmap='gray')
        plt.title('KMeans Output')
        plt.show()
        ```

    Attributes after apply():
        centroids:    (k, d) cluster centroids.
        labels:       (H, W) int hard cluster assignments.
        quantized:    (H, W) float quantized image (pixel -> centroid / max).
        stats:        List of dicts with cluster statistics.
        mc_label:     int index of the brightest (MC) cluster.
    """

    def __init__(
        self,
        k: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int = 42,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Operations (FEBDS-style lambdas)
        self.compute_distances = (
            lambda Z, V: MathematicalOperations.euclidean_distance_sq(Z=Z, V=V)
        )

        # Results (populated by apply)
        self.centroids: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.quantized: Optional[torch.Tensor] = None
        self.n_iter: int = 0
        self.converged: bool = False
        self.stats: Optional[List[dict]] = None
        self.mc_label: int = -1

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

    def apply(self, image: Image, output: Image):
        """
        Apply k-Means clustering.

        Pipeline:
            1. Flatten image → feature matrix Z (N, 1)
            2. k-means++ initialisation
            3. Iterate: assign → update centroids
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

        # Step 2: k-means++ initialisation
        torch.manual_seed(self.random_state)
        indices = [torch.randint(0, N, (1,)).item()]
        for _ in range(1, self.k):
            D2 = self.compute_distances(Z, Z[indices])
            min_D2 = D2.min(dim=0).values
            probs = min_D2 / (min_D2.sum() + 1e-10)
            indices.append(torch.multinomial(probs, 1).item())

        V = Z[indices].clone()

        # Step 3: Iterate assign → update
        labels = torch.zeros(N, dtype=torch.int64, device=device)
        converged = False
        n_iter = 0

        for iteration in range(self.max_iter):
            D2 = self.compute_distances(Z, V)
            new_labels = torch.argmin(D2, dim=0)

            V_new = torch.zeros_like(V)
            for i in range(self.k):
                mask = new_labels == i
                V_new[i] = Z[mask].mean(dim=0) if mask.sum() > 0 else V[i]

            n_iter = iteration + 1
            shift = float(torch.norm(V_new - V))
            V = V_new
            labels = new_labels

            if shift < self.tol:
                converged = True
                break

        # Step 4: Build outputs
        self.centroids = V
        self.labels = labels.reshape(image_shape)
        self.n_iter = n_iter
        self.converged = converged
        self.quantized = self._build_quantized(labels, V, image_shape)
        self.mc_label = int(torch.argmax(V[:, 0]))

        mc_mask = (self.quantized == self.quantized.max()).float()

        self.stats = []
        for i in range(self.k):
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
