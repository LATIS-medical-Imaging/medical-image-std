import torch

# TODO: change this from here

class ScaleProfileAnalyzer:

    def __init__(
        self,
        persistence_threshold: float = 0.60,
        decay_weight: float = 0.5,
    ):
        self.persistence_threshold = (
            persistence_threshold
        )

        self.decay_weight = decay_weight

    def analyze(
        self,
        signature: list[float],
    ) -> dict:

        if not signature:
            return {
                "scale_score": 0.0,
                "dominant_scale": 0.0,
                "scale_width": 0.0,
            }

        values = torch.tensor(
            signature,
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # Dominant scale
        # --------------------------------------------------

        peak_index = int(
            values.argmax().item()
        )

        peak_value = float(
            values[peak_index]
        )

        # --------------------------------------------------
        # Persistence
        #
        # How much of the profile remains strong after
        # the dominant scale?
        # --------------------------------------------------

        if peak_index < len(values) - 1:

            tail = values[
                peak_index + 1:
            ]

            persistence = float(
                (
                    tail
                    >= (
                        self.persistence_threshold
                        * peak_value
                    )
                )
                .float()
                .mean()
                .item()
            )

        else:

            persistence = 0.0

        # --------------------------------------------------
        # Decay
        # --------------------------------------------------

        if peak_index < len(values) - 1:

            tail = values[
                peak_index + 1:
            ]

            if len(tail) > 0:

                decay = float(
                    torch.clamp(
                        (
                            peak_value
                            - tail.mean()
                        )
                        / (
                            peak_value
                            + 1e-8
                        ),
                        0.0,
                        1.0,
                    )
                    .item()
                )

            else:

                decay = 1.0

        else:

            decay = 1.0

        # --------------------------------------------------
        # Width at half maximum
        # --------------------------------------------------

        half = (
            0.5
            * peak_value
        )

        above_half = (
            values >= half
        )

        indices = torch.where(
            above_half
        )[0]

        if len(indices) > 0:

            scale_width = float(
                (
                    indices[-1]
                    - indices[0]
                    + 1
                )
                .item()
            )

        else:

            scale_width = 0.0

        # --------------------------------------------------
        # Score
        # --------------------------------------------------

        strength = peak_value

        persistence_penalty = (
            1.0 - persistence
        )

        scale_score = (
            strength
            * (
                (1.0 - self.decay_weight)
                * persistence_penalty
                +
                self.decay_weight
                * decay
            )
        )

        return {
            "scale_score": float(
                torch.clamp(
                    torch.tensor(
                        scale_score
                    ),
                    0.0,
                    1.0,
                ).item()
            ),

            "dominant_scale": float(
                peak_index
            ),

            "scale_width": scale_width,
        }