# TODO: change this from here
import torch
import numpy as np

from medical_image.algorithms.candidate_scale_signature import CandidateScaleSignature
from medical_image.algorithms.local_physical_analysis_algorithm import CandidatePhysicalFeatures
from medical_image.algorithms.scale_profile_analyzer import ScaleProfileAnalyzer
from medical_image.data.physical_features import Particle


class ParticleBuilder:

    def __init__(
        self,
        device: str = "cpu",
        scale_neighborhood_radius: int = 1,
    ):
        self.device = device

        self.scale_extractor = (
            CandidateScaleSignature(
                neighborhood_radius=(
                    scale_neighborhood_radius
                ),
                device=device,
            )
        )

        self.scale_analyzer = (
            ScaleProfileAnalyzer()
        )

    def build(
        self,
        physical_candidates: list[
            CandidatePhysicalFeatures
        ],
        top_hat_maps: torch.Tensor,
        log_maps: torch.Tensor,
        hf_maps: torch.Tensor,
    ) -> tuple[
        list[Particle],
        list[CandidatePhysicalFeatures],
    ]:
        """
        Build particles from physical candidates.

        Returns
        -------
        particles:
            Filtered list of Particle objects.

        filtered_candidates:
            The corresponding CandidatePhysicalFeatures
            for each accepted particle (same length,
            same order).
        """

        if not physical_candidates:
            return [], []

        # ==================================================
        # 1. Extract scale signatures
        # ==================================================

        top_hat_signatures = (
            self.scale_extractor.extract(
                top_hat_maps,
                physical_candidates,
            )
        )

        log_signatures = (
            self.scale_extractor.extract(
                log_maps,
                physical_candidates,
            )
        )

        hf_signatures = (
            self.scale_extractor.extract(
                hf_maps,
                physical_candidates,
            )
        )

        # ==================================================
        # 2. Build particles
        # ==================================================

        particles = []
        filtered_candidates = []
        accepted_count = 0

        for index, candidate in enumerate(
            physical_candidates
        ):

            top_hat = (
                top_hat_signatures[index]
            )

            log = (
                log_signatures[index]
            )

            hf = (
                hf_signatures[index]
            )

            # ----------------------------------------------
            # Analyze scale signatures
            # ----------------------------------------------

            top_hat_scale = (
                self.scale_analyzer.analyze(
                    top_hat
                )
            )

            log_scale = (
                self.scale_analyzer.analyze(
                    log
                )
            )

            hf_scale = (
                self.scale_analyzer.analyze(
                    hf
                )
            )

            # ----------------------------------------------
            # Combine scale evidence
            # ----------------------------------------------

            scale_score = (
                0.4
                * top_hat_scale[
                    "scale_score"
                ]
                +
                0.35
                * log_scale[
                    "scale_score"
                ]
                +
                0.25
                * hf_scale[
                    "scale_score"
                ]
            )

            # ----------------------------------------------
            # Dominant scale
            # ----------------------------------------------

            dominant_scale = (
                0.4
                * top_hat_scale[
                    "dominant_scale"
                ]
                +
                0.35
                * log_scale[
                    "dominant_scale"
                ]
                +
                0.25
                * hf_scale[
                    "dominant_scale"
                ]
            )

            # ----------------------------------------------
            # Radial score
            # ----------------------------------------------

            radial_score = (
                0.5
                * candidate.radial_symmetry
                +
                0.5
                * self._normalize_radial_decay(
                    candidate.radial_decay
                )
            )

            # ------ Particle Filtering Gates ------
            # Reject candidates that are clearly not microcalcifications
            if candidate.area < 2 or candidate.area > 150:
                continue
            if candidate.circularity < 0.3:
                continue
            if candidate.eccentricity > 0.9:
                continue
            if candidate.solidity < 0.4:
                continue
            if candidate.robust_peak_z < 2.0:
                continue

            # ---- Compute particle_score ----
            # Normalize peak prominence via logistic
            peak_prominence_norm = 1.0 / (1.0 + np.exp(-candidate.robust_peak_z))

            # Normalize center-ring contrast
            contrast_norm = float(min(1.0, max(0.0, candidate.center_ring_contrast / (candidate.neighborhood_mad + 1e-8))))

            # Normalize top-hat max from signature
            tophat_max_norm = float(max(top_hat)) if top_hat else 0.0

            # Normalize high-frequency max
            hf_max_norm = float(max(hf)) if hf else 0.0

            # Shape score from circularity, solidity, eccentricity
            shape_score = (
                0.4 * candidate.circularity
                + 0.3 * candidate.solidity
                + 0.3 * (1.0 - candidate.eccentricity)
            )

            particle_score = (
                0.25 * peak_prominence_norm
                + 0.20 * radial_score
                + 0.15 * shape_score
                + 0.15 * scale_score
                + 0.10 * contrast_norm
                + 0.10 * tophat_max_norm
                + 0.05 * hf_max_norm
            )

            particle_score = float(min(1.0, max(0.0, particle_score)))

            particle = Particle(

                id=accepted_count,

                particle_score=particle_score,

                label=candidate.label,

                x=candidate.center_x,

                y=candidate.center_y,

                area=candidate.area,

                diameter=(
                    candidate.equivalent_diameter
                ),

                perimeter=candidate.perimeter,

                intensity=candidate.peak_value,

                neighborhood_median=(
                    candidate.neighborhood_median
                ),

                neighborhood_mad=(
                    candidate.neighborhood_mad
                ),

                peak_prominence=(
                    candidate.robust_peak_z
                ),

                circularity=(
                    candidate.circularity
                ),

                eccentricity=(
                    candidate.eccentricity
                ),

                solidity=(
                    candidate.solidity
                ),

                aspect_ratio=(
                    candidate.aspect_ratio
                ),

                radial_score=radial_score,

                radial_decay=(
                    candidate.radial_decay
                ),

                radial_peak_width=(
                    candidate.radial_peak_width
                ),

                radial_symmetry=(
                    candidate.radial_symmetry
                ),

                center_ring_contrast=(
                    candidate.center_ring_contrast
                ),

                top_hat_signature=top_hat,

                log_signature=log,

                hf_signature=hf,

                scale_score=scale_score,

                dominant_scale=dominant_scale,

                scale_width=(
                    0.4
                    * top_hat_scale[
                        "scale_width"
                    ]
                    +
                    0.35
                    * log_scale[
                        "scale_width"
                    ]
                    +
                    0.25
                    * hf_scale[
                        "scale_width"
                    ]
                ),
            )

            particles.append(
                particle
            )
            filtered_candidates.append(
                candidate
            )
            accepted_count += 1

        print(
            f"ParticleBuilder: {len(physical_candidates)} candidates -> "
            f"{len(particles)} particles "
            f"({len(physical_candidates) - len(particles)} filtered)"
        )

        return particles, filtered_candidates

    @staticmethod
    def _normalize_radial_decay(
        decay: float,
    ) -> float:

        # Negative slope = intensity decreases
        # as radius increases.
        #
        # Map negative decay to [0,1].

        value = max(
            0.0,
            min(
                1.0,
                -decay,
            ),
        )

        return value