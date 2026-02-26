import copy
from typing import Optional, List

import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.image import Image
from medical_image.utils.image_utils import MathematicalOperations


class KMeansAlgorithm(Algorithm):
    """
    K-Means clustering algorithm for microcalcification segmentation.

    Follows the MATLAB reference:
        RKMeans = KMeans(IterQuintanilla, ROI, C);
        logical(RKMeans == max(max(RKMeans)))

    Clusters 1D pixel values from a top-hat image. The output is a binary
    MC mask where pixels in the brightest cluster are marked as
    microcalcification candidates.

    Attributes after apply():
        centroids:    (k, d)  cluster centroids.
        labels:       (H, W) int  hard cluster assignments.
        quantized:    (H, W) float  quantized image (pixel → centroid / max).
        stats:        List of dicts with cluster statistics.
        mc_label:     int  index of the brightest (MC) cluster.
        n_iter:       int  number of iterations run.
        converged:    bool  whether convergence criterion was met.
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
        """
        Build quantized greyscale image: each pixel → centroid / max_centroid.
        MATLAB: imshow(uint8(RKMeans.*255))
        """
        vals = centroids[:, 0].clone()
        mx = vals.max()
        if mx > 0:
            vals = vals / mx
        labels_2d = labels.reshape(image_shape)
        return vals[labels_2d]

    def apply(self, image: Image, output: Image):
        """
        Apply k-Means clustering.

        Args:
            image: Input Image (2D float tensor, e.g. top-hat result).
            output: Output Image — pixel_data will be the binary MC mask.
        """
        device = self.device
        img = image.pixel_data.to(device).float()
        while img.ndim > 2:
            img = img.squeeze(0)

        H, W = img.shape
        image_shape = (H, W)
        N = H * W

        # Flatten to feature matrix Z (N, 1)
        Z = img.reshape(N, 1)

        # --- Initialize centroids (k-means++) ---
        torch.manual_seed(self.random_state)
        indices = [torch.randint(0, N, (1,)).item()]
        for _ in range(1, self.k):
            D2 = MathematicalOperations.euclidean_distance_sq(
                Z, Z[indices]
            )  # (len(indices), N)
            min_D2 = D2.min(dim=0).values  # (N,)
            probs = min_D2 / (min_D2.sum() + 1e-10)
            idx = torch.multinomial(probs, 1).item()
            indices.append(idx)

        V = Z[indices].clone()  # (k, 1)

        # --- Iterate ---
        converged = False
        n_iter = 0
        labels = torch.zeros(N, dtype=torch.int64, device=device)

        for iteration in range(self.max_iter):
            # Assignment step
            D2 = MathematicalOperations.euclidean_distance_sq(Z, V)  # (k, N)
            new_labels = torch.argmin(D2, dim=0)  # (N,)

            # Update step
            V_new = torch.zeros_like(V)
            for i in range(self.k):
                mask = (new_labels == i)
                if mask.sum() > 0:
                    V_new[i] = Z[mask].mean(dim=0)
                else:
                    V_new[i] = V[i]  # keep empty cluster centroid

            n_iter = iteration + 1
            shift = float(torch.norm(V_new - V))
            V = V_new
            labels = new_labels

            if shift < self.tol:
                converged = True
                break

        # --- Store results ---
        self.centroids = V
        self.labels = labels.reshape(image_shape)
        self.n_iter = n_iter
        self.converged = converged

        # Quantized image
        self.quantized = self._build_quantized(labels, V, image_shape)

        # MC label = brightest cluster
        self.mc_label = int(torch.argmax(V[:, 0]))

        # Binary mask: pixels in brightest cluster
        mc_mask = (self.quantized == self.quantized.max()).float()

        # Statistics
        self.stats = []
        for i in range(self.k):
            cluster_mask = (self.labels == i)
            self.stats.append({
                "id": i,
                "centroid": float(V[i, 0]),
                "pixels": int(cluster_mask.sum()),
                "is_mc": (i == self.mc_label),
            })

        # Output
        output.pixel_data = mc_mask
        output.width = W
        output.height = H
