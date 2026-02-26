import copy
from typing import Optional, List

import torch

from medical_image.algorithms.algorithm import Algorithm
from medical_image.algorithms.fcm import FCMAlgorithm
from medical_image.data.image import Image
from medical_image.utils.image_utils import MathematicalOperations


class PFCMAlgorithm(Algorithm):
    """
    Possibilistic Fuzzy C-Means (PFCM) algorithm for microcalcification detection.

    PFCM extends FCM by adding typicality values that measure how "typical"
    a sample is for each cluster. Microcalcifications are detected as
    **atypical** pixels — those with low maximum typicality across all clusters.

    MATLAB reference:
        T = Tpfcm;
        Tmax = max(T');
        [T2, PosT2] = find(Tmax < tau);    % atypical → MC candidates
        S1 = logical(Rpfcm == min(min(Rpfcm)));
        IAtypique(S1) = 0;                  % exclude darkest cluster

    Attributes after apply():
        typicality:   (c, N) typicality matrix T.
        T_max_map:    (H, W) max typicality per pixel (low = atypical = MC).
        centroids:    (c, d) cluster centroids.
        membership:   (c, N) fuzzy membership matrix.
        labels:       (H, W) int hard cluster assignments.
        quantized:    (H, W) float quantized image.
        gamma:        (c,) gamma values per cluster.
    """

    def __init__(
        self,
        c: int = 2,
        m: float = 2.0,
        eta: float = 2.0,
        a: float = 1.0,
        b: float = 4.0,
        tau: float = 0.04,
        max_iter: int = 100,
        tol: float = 1e-3,
        fcm_max_iter: int = 100,
        random_state: int = 42,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.c = c
        self.m = m
        self.eta = eta
        self.a = a
        self.b = b
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.fcm_max_iter = fcm_max_iter
        self.random_state = random_state

        # Results (populated by apply)
        self.typicality: Optional[torch.Tensor] = None
        self.T_max_map: Optional[torch.Tensor] = None
        self.centroids: Optional[torch.Tensor] = None
        self.membership: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.quantized: Optional[torch.Tensor] = None
        self.gamma: Optional[torch.Tensor] = None
        self.n_iter: int = 0
        self.converged: bool = False

    # ------------------------------------------------------------------
    # PFCM-specific update rules
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gamma(U: torch.Tensor, D2: torch.Tensor, m: float) -> torch.Tensor:
        """γ_i = Σ_k μ_ik^m D²_ik / Σ_k μ_ik^m → (c,)"""
        Um = U ** m
        num = (Um * D2).sum(dim=1)
        den = Um.sum(dim=1) + 1e-10
        return num / den

    @staticmethod
    def _update_typicality(
        D2: torch.Tensor, gamma: torch.Tensor, b: float, eta: float,
    ) -> torch.Tensor:
        """t_ik = 1 / (1 + (b/γ_i · D²_ik)^{1/(η-1)}) → (c, N)"""
        eps = 1e-10
        exp = 1.0 / (eta - 1.0)
        ratio = (b / (gamma.unsqueeze(1) + eps)) * D2
        return 1.0 / (1.0 + (ratio + eps) ** exp)

    @staticmethod
    def _update_prototypes(
        Z: torch.Tensor, U: torch.Tensor, T: torch.Tensor,
        m: float, eta: float, a: float, b: float,
    ) -> torch.Tensor:
        """v_i = Σ(a μ^m + b t^η) z / Σ(a μ^m + b t^η) → (c, d)"""
        W = a * (U ** m) + b * (T ** eta)  # (c, N)
        denom = W.sum(dim=1, keepdim=True) + 1e-10
        return (W @ Z) / denom

    # ------------------------------------------------------------------
    # Main apply
    # ------------------------------------------------------------------

    def apply(self, image: Image, output: Image):
        """
        Apply PFCM: warm-start from FCM, iterate PFCM, detect MCs by atypicality.

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

        # Flatten
        Z = img.reshape(N, 1)

        # ----- Step 1: Warm-start from FCM (c=2) -----
        fcm = FCMAlgorithm(
            c=self.c, m=self.m, max_iter=self.fcm_max_iter,
            tol=self.tol, random_state=self.random_state, device=device,
        )
        fcm_output = copy.deepcopy(image)
        fcm.apply(image, fcm_output)

        U = fcm.membership.clone()
        V = fcm.centroids.clone()

        # ----- Step 2: PFCM iterations -----
        D2 = MathematicalOperations.euclidean_distance_sq(Z, V)
        gamma = self._compute_gamma(U, D2, self.m)
        T = self._update_typicality(D2, gamma, self.b, self.eta)

        converged = False
        n_iter = 0

        for iteration in range(self.max_iter):
            V = self._update_prototypes(Z, U, T, self.m, self.eta, self.a, self.b)
            D2 = MathematicalOperations.euclidean_distance_sq(Z, V)
            U = FCMAlgorithm._update_membership(D2, self.m)
            gamma = self._compute_gamma(U, D2, self.m)
            T_new = self._update_typicality(D2, gamma, self.b, self.eta)

            diff = float(torch.norm(T_new - T))
            T = T_new
            n_iter = iteration + 1
            if diff < self.tol:
                converged = True
                break

        # ----- Step 3: MC detection via atypicality (MATLAB logic) -----
        # Tmax = max typicality across ALL clusters per pixel
        T_max = T.max(dim=0).values  # (N,)
        T_max_map = T_max.reshape(image_shape)

        # Atypical mask: pixels with Tmax < tau
        atypical_mask = (T_max_map < self.tau)

        # Build quantized image
        labels = torch.argmax(U, dim=0).to(torch.int64)
        labels_2d = labels.reshape(image_shape)

        centroid_vals = V[:, 0]
        darkest_cluster = int(torch.argmin(centroid_vals))

        # Remove atypical pixels from darkest cluster (background noise)
        darkest_mask = (labels_2d == darkest_cluster)
        binary_mask = atypical_mask & (~darkest_mask)

        # Build quantized
        mx = centroid_vals.max()
        quant_lut = centroid_vals / (mx + 1e-10)
        quantized = quant_lut[labels_2d]

        # ----- Store results -----
        self.typicality = T
        self.T_max_map = T_max_map
        self.centroids = V
        self.membership = U
        self.labels = labels_2d
        self.quantized = quantized
        self.gamma = gamma
        self.n_iter = n_iter
        self.converged = converged

        # Output
        output.pixel_data = binary_mask.float()
        output.width = W
        output.height = H
