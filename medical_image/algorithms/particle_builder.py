# TODO: change this from here
import torch

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
    ) -> list[Particle]:

        if not physical_candidates:
            return []

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

            particle = Particle(

                id=index,

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

        return particles

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