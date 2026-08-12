import torch

# TODO: change this from here

class CandidateScaleSignature:
    """
    Extract multi-scale response signatures for candidate
    particles.

    The candidate position is used as the sampling anchor.
    """

    def __init__(
            self,
            neighborhood_radius: int = 1,
            device: str = "cpu",
    ):
        self.radius = neighborhood_radius
        self.device = torch.device(device)

    @staticmethod
    def _normalize_scale_maps(
            maps: torch.Tensor,
    ) -> torch.Tensor:

        if maps.ndim == 2:
            maps = maps.unsqueeze(0)

        if maps.ndim != 3:
            raise ValueError(
                f"Expected [S,H,W], got {maps.shape}"
            )

        flat = maps.flatten(
            start_dim=1
        )

        low = torch.quantile(
            flat,
            0.01,
            dim=1,
            keepdim=True,
        )

        high = torch.quantile(
            flat,
            0.99,
            dim=1,
            keepdim=True,
        )

        normalized = (
                             flat - low
                     ) / (
                             high - low
                             + 1e-8
                     )

        normalized = torch.clamp(
            normalized,
            0.0,
            1.0,
        )

        return normalized.reshape_as(
            maps
        )

    def extract(
            self,
            maps: torch.Tensor,
            candidates: list,
    ) -> list[list[float]]:

        maps = maps.to(
            device=self.device,
            dtype=torch.float32,
        )

        if maps.ndim == 2:
            maps = maps.unsqueeze(0)

        maps = self._normalize_scale_maps(
            maps
        )

        _, height, width = maps.shape

        signatures = []

        for candidate in candidates:
            y = int(round(
                candidate.center_y
            ))

            x = int(round(
                candidate.center_x
            ))

            y0 = max(
                0,
                y - self.radius,
            )

            y1 = min(
                height,
                y + self.radius + 1,
            )

            x0 = max(
                0,
                x - self.radius,
            )

            x1 = min(
                width,
                x + self.radius + 1,
            )

            local = maps[
                :,
                y0:y1,
                x0:x1,
            ]

            values = local.amax(
                dim=(1, 2)
            )

            signatures.append(
                values.detach()
                .cpu()
                .tolist()
            )

        return signatures