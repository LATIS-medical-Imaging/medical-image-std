from typing import List, Sequence

import numpy as np

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.mammogram_feature import (
    GlobalMammogramFeatures,
)
from medical_image.data.physical_features import (
    CandidatePhysicalFeatures,
    SpatialParticleFeatures,
)
from medical_image.data.cluster import ClusterFeatures


class GlobalMammogramReasoningAlgorithm(Algorithm):
    """
    Global mammogram-level reasoning.

    Step 11 of the non-deep-learning microcalcification
    detection pipeline.

    Inputs:

        CandidatePhysicalFeatures
        SpatialParticleFeatures
        ClusterFeatures

    Outputs:

        GlobalMammogramFeatures

    This algorithm does NOT inspect image pixels.

    It interprets the global population of candidate
    particles and spatial clusters.
    """

    def __init__(
        self,
        min_significant_cluster_score: float = 0.50,
        min_significant_cluster_particles: int = 3,
        device: str = "cpu",
    ):
        super().__init__(
            device=device
        )

        if (
            min_significant_cluster_score
            < 0.0
        ):
            raise ValueError(
                "min_significant_cluster_score "
                "must be >= 0."
            )

        if (
            min_significant_cluster_score
            > 1.0
        ):
            raise ValueError(
                "min_significant_cluster_score "
                "must be <= 1."
            )

        if (
            min_significant_cluster_particles
            < 1
        ):
            raise ValueError(
                "min_significant_cluster_particles "
                "must be >= 1."
            )

        self.min_significant_cluster_score = float(
            min_significant_cluster_score
        )

        self.min_significant_cluster_particles = int(
            min_significant_cluster_particles
        )

    # ======================================================
    # Utilities
    # ======================================================

    @staticmethod
    def _safe_mean(
        values,
    ) -> float:

        if len(values) == 0:
            return 0.0

        return float(
            np.mean(values)
        )

    @staticmethod
    def _safe_median(
        values,
    ) -> float:

        if len(values) == 0:
            return 0.0

        return float(
            np.median(values)
        )

    @staticmethod
    def _safe_std(
        values,
    ) -> float:

        if len(values) == 0:
            return 0.0

        return float(
            np.std(values)
        )

    @staticmethod
    def _safe_max(
        values,
    ) -> float:

        if len(values) == 0:
            return 0.0

        return float(
            np.max(values)
        )

    # ======================================================
    # Cluster presence
    # ======================================================

    @staticmethod
    def _cluster_presence_score(
        valid_cluster_count: int,
        total_particles: int,
    ) -> float:
        """
        Estimate how strongly the mammogram exhibits
        meaningful cluster structure.

        Saturates smoothly instead of increasing
        indefinitely with the number of clusters.
        """

        if (
            valid_cluster_count <= 0
            or total_particles <= 0
        ):
            return 0.0

        ratio = (
            valid_cluster_count
            / np.sqrt(
                total_particles
            )
        )

        score = (
            ratio
            / (
                ratio + 1.0
            )
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ======================================================
    # Cluster strength
    # ======================================================

    @staticmethod
    def _cluster_strength_score(
        cluster_scores: np.ndarray,
    ) -> float:
        """
        Global strength of cluster evidence.

        Uses a combination of the strongest cluster
        and the average cluster strength.
        """

        if len(
            cluster_scores
        ) == 0:
            return 0.0

        max_score = float(
            np.max(
                cluster_scores
            )
        )

        mean_score = float(
            np.mean(
                cluster_scores
            )
        )

        score = (
            0.7 * max_score
            + 0.3 * mean_score
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ======================================================
    # Particle support
    # ======================================================

    @staticmethod
    def _particle_support_score(
        clustered_particle_ratio: float,
        clustered_particle_evidence: float,
    ) -> float:
        """
        Estimate how strongly the particle population
        supports the cluster interpretation.
        """

        score = (
            0.5
            * clustered_particle_ratio
            +
            0.5
            * clustered_particle_evidence
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ======================================================
    # Spatial concentration
    # ======================================================

    @staticmethod
    def _spatial_concentration_score(
        dominant_cluster_fraction: float,
    ) -> float:
        """
        Measure whether meaningful particles are
        concentrated in a dominant spatial region.

        A dominant cluster is treated as stronger global
        evidence than a completely diffuse distribution.
        """

        return float(
            np.clip(
                dominant_cluster_fraction,
                0.0,
                1.0,
            )
        )

    # ======================================================
    # Global score
    # ======================================================

    @staticmethod
    def _global_evidence_score(
        cluster_presence_score: float,
        cluster_strength_score: float,
        particle_support_score: float,
        spatial_concentration_score: float,
    ) -> float:
        """
        Combine global mammogram-level evidence.

        Current weighting:

            cluster presence       20%
            cluster strength       35%
            particle support       30%
            spatial concentration  15%
        """

        score = (
            0.20
            * cluster_presence_score

            + 0.35
            * cluster_strength_score

            + 0.30
            * particle_support_score

            + 0.15
            * spatial_concentration_score
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ======================================================
    # Main
    # ======================================================

    def apply(
        self,
        physical_features: Sequence[
            CandidatePhysicalFeatures
        ],
        spatial_features: Sequence[
            SpatialParticleFeatures
        ],
        clusters: Sequence[
            ClusterFeatures
        ],
    ) -> GlobalMammogramFeatures:

        if physical_features is None:
            raise ValueError(
                "physical_features must not be None."
            )

        if spatial_features is None:
            raise ValueError(
                "spatial_features must not be None."
            )

        if clusters is None:
            raise ValueError(
                "clusters must not be None."
            )

        # --------------------------------------------------
        # Validate particle representations
        # --------------------------------------------------

        if len(
            physical_features
        ) != len(
            spatial_features
        ):
            raise ValueError(
                "physical_features and "
                "spatial_features must have "
                "the same number of particles."
            )

        total_particles = len(
            physical_features
        )

        if total_particles == 0:

            return GlobalMammogramFeatures(
                total_particles=0,
                clustered_particles=0,
                isolated_particles=0,
                clustered_particle_ratio=0.0,
                isolated_particle_ratio=0.0,

                total_clusters=0,
                valid_clusters=0,
                cluster_ratio=0.0,

                mean_cluster_size=0.0,
                median_cluster_size=0.0,
                max_cluster_size=0,
                std_cluster_size=0.0,

                mean_cluster_score=0.0,
                median_cluster_score=0.0,
                max_cluster_score=0.0,
                std_cluster_score=0.0,

                mean_cluster_density=0.0,
                max_cluster_density=0.0,

                mean_cluster_compactness=0.0,
                max_cluster_compactness=0.0,
                mean_cluster_eccentricity=0.0,

                mean_particle_score=0.0,
                max_particle_score=0.0,
                clustered_particle_evidence=0.0,
                global_particle_evidence=0.0,

                global_center_y=0.0,
                global_center_x=0.0,

                spatial_spread_y=0.0,
                spatial_spread_x=0.0,
                global_spatial_spread=0.0,

                dominant_cluster_id=-1,
                dominant_cluster_score=0.0,
                dominant_cluster_size=0,
                dominant_cluster_fraction=0.0,

                cluster_score_concentration=0.0,
                particle_concentration=0.0,

                cluster_presence_score=0.0,
                cluster_strength_score=0.0,
                particle_support_score=0.0,
                spatial_concentration_score=0.0,
                global_evidence_score=0.0,

                has_significant_cluster=False,
            )

        # --------------------------------------------------
        # Validate labels
        # --------------------------------------------------

        physical_labels = [
            feature.label
            for feature in physical_features
        ]

        spatial_labels = [
            feature.label
            for feature in spatial_features
        ]

        if physical_labels != spatial_labels:
            raise ValueError(
                "physical_features and "
                "spatial_features must have "
                "identical particle ordering."
            )

        # --------------------------------------------------
        # Particle evidence
        # --------------------------------------------------

        particle_scores = np.asarray(
            [
                float(
                    getattr(
                        feature,
                        "particle_score",
                        feature.robust_peak_z,
                    )
                )
                for feature in physical_features
            ],
            dtype=np.float64,
        )

        mean_particle_score = (
            self._safe_mean(
                particle_scores
            )
        )

        max_particle_score = (
            self._safe_max(
                particle_scores
            )
        )

        # --------------------------------------------------
        # Cluster filtering
        # --------------------------------------------------

        valid_clusters = [
            cluster
            for cluster in clusters
            if (
                cluster.is_cluster
                and cluster.particle_count
                >= self.min_significant_cluster_particles
                and cluster.cluster_score
                >= self.min_significant_cluster_score
            )
        ]

        total_clusters = len(
            clusters
        )

        valid_cluster_count = len(
            valid_clusters
        )

        cluster_ratio = (
            valid_cluster_count
            / max(
                total_clusters,
                1,
            )
        )

        # --------------------------------------------------
        # Cluster statistics
        # --------------------------------------------------

        if valid_cluster_count > 0:

            cluster_sizes = np.asarray(
                [
                    cluster.particle_count
                    for cluster in valid_clusters
                ],
                dtype=np.float64,
            )

            cluster_scores = np.asarray(
                [
                    cluster.cluster_score
                    for cluster in valid_clusters
                ],
                dtype=np.float64,
            )

            cluster_densities = np.asarray(
                [
                    cluster.density
                    for cluster in valid_clusters
                ],
                dtype=np.float64,
            )

            cluster_compactness = np.asarray(
                [
                    cluster.spatial_compactness
                    for cluster in valid_clusters
                ],
                dtype=np.float64,
            )

            cluster_eccentricities = np.asarray(
                [
                    cluster.spatial_eccentricity
                    for cluster in valid_clusters
                ],
                dtype=np.float64,
            )

            mean_cluster_size = (
                self._safe_mean(
                    cluster_sizes
                )
            )

            median_cluster_size = (
                self._safe_median(
                    cluster_sizes
                )
            )

            max_cluster_size = int(
                np.max(
                    cluster_sizes
                )
            )

            std_cluster_size = (
                self._safe_std(
                    cluster_sizes
                )
            )

            mean_cluster_score = (
                self._safe_mean(
                    cluster_scores
                )
            )

            median_cluster_score = (
                self._safe_median(
                    cluster_scores
                )
            )

            max_cluster_score = (
                self._safe_max(
                    cluster_scores
                )
            )

            std_cluster_score = (
                self._safe_std(
                    cluster_scores
                )
            )

            mean_cluster_density = (
                self._safe_mean(
                    cluster_densities
                )
            )

            max_cluster_density = (
                self._safe_max(
                    cluster_densities
                )
            )

            mean_cluster_compactness = (
                self._safe_mean(
                    cluster_compactness
                )
            )

            max_cluster_compactness = (
                self._safe_max(
                    cluster_compactness
                )
            )

            mean_cluster_eccentricity = (
                self._safe_mean(
                    cluster_eccentricities
                )
            )

        else:

            cluster_sizes = np.empty(
                0,
                dtype=np.float64,
            )

            cluster_scores = np.empty(
                0,
                dtype=np.float64,
            )

            mean_cluster_size = 0.0
            median_cluster_size = 0.0
            max_cluster_size = 0
            std_cluster_size = 0.0

            mean_cluster_score = 0.0
            median_cluster_score = 0.0
            max_cluster_score = 0.0
            std_cluster_score = 0.0

            mean_cluster_density = 0.0
            max_cluster_density = 0.0

            mean_cluster_compactness = 0.0
            max_cluster_compactness = 0.0

            mean_cluster_eccentricity = 0.0

        # --------------------------------------------------
        # Cluster membership
        # --------------------------------------------------

        clustered_labels = set()

        for cluster in valid_clusters:

            clustered_labels.update(
                cluster.particle_labels
            )

        clustered_particles = len(
            clustered_labels
        )

        isolated_particles = (
            total_particles
            - clustered_particles
        )

        clustered_particle_ratio = (
            clustered_particles
            / max(
                total_particles,
                1,
            )
        )

        isolated_particle_ratio = (
            isolated_particles
            / max(
                total_particles,
                1,
            )
        )

        # --------------------------------------------------
        # Clustered particle evidence
        # --------------------------------------------------

        if clustered_particles > 0:

            clustered_particle_scores = np.asarray(
                [
                    self._particle_score(
                        feature
                    )
                    for feature in physical_features
                    if feature.label
                    in clustered_labels
                ],
                dtype=np.float64,
            )

            clustered_particle_evidence = (
                float(
                    np.mean(
                        clustered_particle_scores
                    )
                )
            )

        else:

            clustered_particle_evidence = 0.0

        # --------------------------------------------------
        # Global particle evidence
        # --------------------------------------------------

        global_particle_evidence = float(
            np.mean(
                particle_scores
            )
        )

        # --------------------------------------------------
        # Global spatial distribution
        # --------------------------------------------------

        positions = np.asarray(
            [
                (
                    feature.center_x,
                    feature.center_y,
                )
                for feature in physical_features
            ],
            dtype=np.float64,
        )

        global_center_x = float(
            np.mean(
                positions[:, 0]
            )
        )

        global_center_y = float(
            np.mean(
                positions[:, 1]
            )
        )

        spatial_spread_x = float(
            np.std(
                positions[:, 0]
            )
        )

        spatial_spread_y = float(
            np.std(
                positions[:, 1]
            )
        )

        global_spatial_spread = float(
            np.sqrt(
                spatial_spread_x ** 2
                +
                spatial_spread_y ** 2
            )
        )

        # --------------------------------------------------
        # Dominant cluster
        # --------------------------------------------------

        if valid_cluster_count > 0:

            dominant_index = int(
                np.argmax(
                    cluster_scores
                )
            )

            dominant_cluster = (
                valid_clusters[
                    dominant_index
                ]
            )

            dominant_cluster_id = int(
                dominant_cluster.cluster_id
            )

            dominant_cluster_score = float(
                dominant_cluster.cluster_score
            )

            dominant_cluster_size = int(
                dominant_cluster.particle_count
            )

            dominant_cluster_fraction = (
                dominant_cluster_size
                / max(
                    clustered_particles,
                    1,
                )
            )

        else:

            dominant_cluster_id = -1
            dominant_cluster_score = 0.0
            dominant_cluster_size = 0
            dominant_cluster_fraction = 0.0

        # --------------------------------------------------
        # Concentration
        # --------------------------------------------------

        if (
            valid_cluster_count > 0
            and max_cluster_score > 1e-8
        ):

            cluster_score_concentration = float(
                dominant_cluster_score
                / max_cluster_score
            )

        else:

            cluster_score_concentration = 0.0

        particle_concentration = (
            dominant_cluster_fraction
        )

        # --------------------------------------------------
        # Global evidence components
        # --------------------------------------------------

        cluster_presence_score = (
            self._cluster_presence_score(
                valid_cluster_count,
                total_particles,
            )
        )

        cluster_strength_score = (
            self._cluster_strength_score(
                cluster_scores
            )
        )

        particle_support_score = (
            self._particle_support_score(
                clustered_particle_ratio,
                clustered_particle_evidence,
            )
        )

        spatial_concentration_score = (
            self._spatial_concentration_score(
                dominant_cluster_fraction
            )
        )

        global_evidence_score = (
            self._global_evidence_score(
                cluster_presence_score,
                cluster_strength_score,
                particle_support_score,
                spatial_concentration_score,
            )
        )

        # --------------------------------------------------
        # Final global decision
        # --------------------------------------------------

        has_significant_cluster = (
            valid_cluster_count > 0
            and max_cluster_score
            >= self.min_significant_cluster_score
        )

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        return GlobalMammogramFeatures(

            total_particles=(
                total_particles
            ),

            clustered_particles=(
                clustered_particles
            ),

            isolated_particles=(
                isolated_particles
            ),

            clustered_particle_ratio=(
                clustered_particle_ratio
            ),

            isolated_particle_ratio=(
                isolated_particle_ratio
            ),

            total_clusters=(
                total_clusters
            ),

            valid_clusters=(
                valid_cluster_count
            ),

            cluster_ratio=(
                cluster_ratio
            ),

            mean_cluster_size=(
                mean_cluster_size
            ),

            median_cluster_size=(
                median_cluster_size
            ),

            max_cluster_size=(
                max_cluster_size
            ),

            std_cluster_size=(
                std_cluster_size
            ),

            mean_cluster_score=(
                mean_cluster_score
            ),

            median_cluster_score=(
                median_cluster_score
            ),

            max_cluster_score=(
                max_cluster_score
            ),

            std_cluster_score=(
                std_cluster_score
            ),

            mean_cluster_density=(
                mean_cluster_density
            ),

            max_cluster_density=(
                max_cluster_density
            ),

            mean_cluster_compactness=(
                mean_cluster_compactness
            ),

            max_cluster_compactness=(
                max_cluster_compactness
            ),

            mean_cluster_eccentricity=(
                mean_cluster_eccentricity
            ),

            mean_particle_score=(
                mean_particle_score
            ),

            max_particle_score=(
                max_particle_score
            ),

            clustered_particle_evidence=(
                clustered_particle_evidence
            ),

            global_particle_evidence=(
                global_particle_evidence
            ),

            global_center_y=(
                global_center_y
            ),

            global_center_x=(
                global_center_x
            ),

            spatial_spread_y=(
                spatial_spread_y
            ),

            spatial_spread_x=(
                spatial_spread_x
            ),

            global_spatial_spread=(
                global_spatial_spread
            ),

            dominant_cluster_id=(
                dominant_cluster_id
            ),

            dominant_cluster_score=(
                dominant_cluster_score
            ),

            dominant_cluster_size=(
                dominant_cluster_size
            ),

            dominant_cluster_fraction=(
                dominant_cluster_fraction
            ),

            cluster_score_concentration=(
                cluster_score_concentration
            ),

            particle_concentration=(
                particle_concentration
            ),

            cluster_presence_score=(
                cluster_presence_score
            ),

            cluster_strength_score=(
                cluster_strength_score
            ),

            particle_support_score=(
                particle_support_score
            ),

            spatial_concentration_score=(
                spatial_concentration_score
            ),

            global_evidence_score=(
                global_evidence_score
            ),

            has_significant_cluster=(
                has_significant_cluster
            ),
        )

    # ======================================================
    # Particle score helper
    # ======================================================

    @staticmethod
    def _particle_score(
        particle: CandidatePhysicalFeatures,
    ) -> float:

        if hasattr(
            particle,
            "particle_score",
        ):
            return float(
                particle.particle_score
            )

        return float(
            particle.robust_peak_z
        )