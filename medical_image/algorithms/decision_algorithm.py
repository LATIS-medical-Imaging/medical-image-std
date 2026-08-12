import numpy as np

from medical_image.data.decision import FinalMCDecision, FinalParticleDecision, FinalClusterDecision
from medical_image.algorithms.algorithm import Algorithm


class FinalMCDecisionAlgorithm(Algorithm):
    """
    Step 12:
    Final microcalcification evidence fusion.

    This algorithm does not inspect mammogram pixels.

    It combines:

        1. Particle-level evidence
        2. Cluster-level evidence
        3. Global mammogram evidence

    into final continuous particle, cluster, and
    mammogram-level decisions.
    """

    def __init__(
        self,
        particle_weight: float = 0.40,
        cluster_weight: float = 0.40,
        global_weight: float = 0.20,
        particle_threshold: float = 0.50,
        cluster_threshold: float = 0.50,
        final_threshold: float = 0.50,
        device: str = "cpu",
    ):
        super().__init__(
            device=device
        )

        if particle_weight < 0:
            raise ValueError(
                "particle_weight must be >= 0."
            )

        if cluster_weight < 0:
            raise ValueError(
                "cluster_weight must be >= 0."
            )

        if global_weight < 0:
            raise ValueError(
                "global_weight must be >= 0."
            )

        total_weight = (
            particle_weight
            + cluster_weight
            + global_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "At least one evidence weight "
                "must be > 0."
            )

        self.particle_weight = (
            particle_weight / total_weight
        )

        self.cluster_weight = (
            cluster_weight / total_weight
        )

        self.global_weight = (
            global_weight / total_weight
        )

        self.particle_threshold = float(
            particle_threshold
        )

        self.cluster_threshold = float(
            cluster_threshold
        )

        self.final_threshold = float(
            final_threshold
        )

    # ==========================================================
    # Utilities
    # ==========================================================

    @staticmethod
    def _clip(
        value: float,
    ) -> float:

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _normalize_particle_score(
        particle,
    ) -> float:
        """
        Convert particle evidence to [0,1].

        CandidatePhysicalFeatures currently exposes
        robust_peak_z as its main particle evidence.

        If a final particle_score exists, use it.
        """

        if hasattr(
            particle,
            "particle_score",
        ):

            score = float(
                particle.particle_score
            )

            return float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )

        # ------------------------------------------------------
        # robust_peak_z is not naturally bounded.
        #
        # Convert it smoothly to [0,1].
        # ------------------------------------------------------

        z = float(
            particle.robust_peak_z
        )

        if not np.isfinite(z):
            return 0.0

        # Logistic mapping.
        #
        # z = 0 -> 0.5
        # z = 2 -> ~0.88
        # z = 4 -> ~0.98
        #
        score = 1.0 / (
            1.0
            + np.exp(
                -z
            )
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ==========================================================
    # Cluster lookup
    # ==========================================================

    @staticmethod
    def _build_particle_cluster_map(
        clusters,
    ):
        """
        Map particle label -> strongest cluster.

        A particle normally belongs to one spatial group,
        but taking the strongest cluster makes the method
        robust to future overlapping grouping strategies.
        """

        particle_to_cluster = {}

        for cluster in clusters:

            for label in (
                cluster.particle_labels
            ):

                previous = (
                    particle_to_cluster.get(
                        label
                    )
                )

                if (
                    previous is None
                    or cluster.cluster_score
                    > previous.cluster_score
                ):

                    particle_to_cluster[
                        label
                    ] = cluster

        return particle_to_cluster

    # ==========================================================
    # Cluster context
    # ==========================================================

    @staticmethod
    def _cluster_context_score(
        cluster,
    ) -> float:
        """
        Cluster-level contextual evidence.

        Cluster score already contains:

            density
            compactness
            particle evidence
            geometry

        Therefore Step 12 should not reconstruct those
        components.

        It only applies the cluster validity state.
        """

        if cluster is None:
            return 0.0

        if not cluster.is_cluster:
            return (
                0.25
                * float(
                    np.clip(
                        cluster.cluster_score,
                        0.0,
                        1.0,
                    )
                )
            )

        return float(
            np.clip(
                cluster.cluster_score,
                0.0,
                1.0,
            )
        )

    # ==========================================================
    # Particle final score
    # ==========================================================

    def _particle_final_score(
        self,
        particle_score: float,
        cluster_score: float,
        global_score: float,
    ) -> float:
        """
        Fuse particle, cluster and global evidence.
        """

        score = (
            self.particle_weight
            * particle_score

            + self.cluster_weight
            * cluster_score

            + self.global_weight
            * global_score
        )

        return self._clip(
            score
        )

    # ==========================================================
    # Cluster final score
    # ==========================================================

    def _cluster_final_score(
        self,
        cluster_score: float,
        global_score: float,
    ) -> float:
        """
        Cluster-level fusion.

        Cluster evidence receives more weight because
        this is the natural spatial unit for MC reasoning.
        """

        score = (
            0.70
            * cluster_score

            + 0.30
            * global_score
        )

        return self._clip(
            score
        )

    # ==========================================================
    # Main
    # ==========================================================

    def apply(
        self,
        particles,
        physical_features,
        clusters,
        global_features,
    ) -> FinalMCDecision:

        if particles is None:
            raise ValueError(
                "particles must not be None."
            )

        if physical_features is None:
            raise ValueError(
                "physical_features must not be None."
            )

        if clusters is None:
            raise ValueError(
                "clusters must not be None."
            )

        if global_features is None:
            raise ValueError(
                "global_features must not be None."
            )

        if len(particles) != len(
            physical_features
        ):
            raise ValueError(
                "particles and physical_features "
                "must contain the same number "
                "of objects."
            )

        # ------------------------------------------------------
        # Empty case
        # ------------------------------------------------------

        if len(
            physical_features
        ) == 0:

            return FinalMCDecision(
                final_score=0.0,

                is_microcalcification_present=False,

                particle_evidence=0.0,
                cluster_evidence=0.0,
                global_evidence=0.0,

                total_particles=0,
                positive_particles=0,

                total_clusters=len(
                    clusters
                ),
                positive_clusters=0,

                dominant_particle_id=-1,
                dominant_particle_label=-1,

                dominant_cluster_id=-1,
                dominant_cluster_score=0.0,
                dominant_cluster_size=0,

                particle_decisions=[],
                cluster_decisions=[],
            )

        # ------------------------------------------------------
        # Build cluster lookup
        # ------------------------------------------------------

        particle_to_cluster = (
            self._build_particle_cluster_map(
                clusters
            )
        )

        global_score = float(
            np.clip(
                global_features.global_evidence_score,
                0.0,
                1.0,
            )
        )

        # ------------------------------------------------------
        # Particle decisions
        # ------------------------------------------------------

        particle_decisions = []

        for particle_index, (
            particle,
            physical,
        ) in enumerate(
            zip(
                particles,
                physical_features,
            )
        ):

            label = int(
                physical.label
            )

            # --------------------------------------------------
            # Particle evidence
            # --------------------------------------------------

            particle_score = (
                self._normalize_particle_score(
                    physical
                )
            )

            # --------------------------------------------------
            # Cluster context
            # --------------------------------------------------

            cluster = (
                particle_to_cluster.get(
                    label
                )
            )

            if cluster is None:

                cluster_id = -1

                cluster_score = 0.0

                cluster_context_score = 0.0

            else:

                cluster_id = int(
                    cluster.cluster_id
                )

                cluster_score = float(
                    np.clip(
                        cluster.cluster_score,
                        0.0,
                        1.0,
                    )
                )

                cluster_context_score = (
                    self._cluster_context_score(
                        cluster
                    )
                )

            # --------------------------------------------------
            # Global context
            # --------------------------------------------------

            # Do not give isolated particles the full global
            # score. Global evidence is contextual support,
            # not direct particle evidence.
            if cluster is None:

                global_context_score = (
                    0.50
                    * global_score
                )

            else:

                global_context_score = (
                    global_score
                )

            # --------------------------------------------------
            # Final particle score
            # --------------------------------------------------

            final_score = (
                self._particle_final_score(
                    particle_score,
                    cluster_context_score,
                    global_context_score,
                )
            )

            # --------------------------------------------------
            # Final particle decision
            # --------------------------------------------------

            is_mc = (
                final_score
                >= self.final_threshold
                and particle_score
                >= self.particle_threshold
            )

            particle_decisions.append(
                FinalParticleDecision(
                    particle_id=int(
                        particle_index
                    ),

                    label=label,

                    particle_score=(
                        particle_score
                    ),

                    cluster_id=(
                        cluster_id
                    ),

                    cluster_score=(
                        cluster_score
                    ),

                    cluster_context_score=(
                        cluster_context_score
                    ),

                    global_score=(
                        global_score
                    ),

                    global_context_score=(
                        global_context_score
                    ),

                    final_score=(
                        final_score
                    ),

                    is_microcalcification=(
                        is_mc
                    ),
                )
            )

        # ------------------------------------------------------
        # Cluster decisions
        # ------------------------------------------------------

        cluster_decisions = []

        for cluster in clusters:

            cluster_score = float(
                np.clip(
                    cluster.cluster_score,
                    0.0,
                    1.0,
                )
            )

            final_cluster_score = (
                self._cluster_final_score(
                    cluster_score,
                    global_score,
                )
            )

            is_positive_cluster = (
                cluster.is_cluster
                and cluster_score
                >= self.cluster_threshold
                and final_cluster_score
                >= self.final_threshold
            )

            cluster_decisions.append(
                FinalClusterDecision(

                    cluster_id=int(
                        cluster.cluster_id
                    ),

                    particle_count=int(
                        cluster.particle_count
                    ),

                    cluster_score=(
                        cluster_score
                    ),

                    global_context_score=(
                        global_score
                    ),

                    final_score=(
                        final_cluster_score
                    ),

                    is_microcalcification_cluster=(
                        is_positive_cluster
                    ),

                    particle_labels=list(
                        cluster.particle_labels
                    ),
                )
            )

        # ------------------------------------------------------
        # Positive particles
        # ------------------------------------------------------

        positive_particles = [
            decision
            for decision
            in particle_decisions
            if decision.is_microcalcification
        ]

        # ------------------------------------------------------
        # Positive clusters
        # ------------------------------------------------------

        positive_clusters = [
            decision
            for decision
            in cluster_decisions
            if decision.is_microcalcification_cluster
        ]

        # ------------------------------------------------------
        # Dominant particle
        # ------------------------------------------------------

        if particle_decisions:

            dominant_particle = max(
                particle_decisions,
                key=lambda x: x.final_score,
            )

            dominant_particle_id = (
                dominant_particle.particle_id
            )

            dominant_particle_label = (
                dominant_particle.label
            )

            particle_evidence = (
                dominant_particle.final_score
            )

        else:

            dominant_particle_id = -1
            dominant_particle_label = -1
            particle_evidence = 0.0

        # ------------------------------------------------------
        # Dominant cluster
        # ------------------------------------------------------

        if cluster_decisions:

            dominant_cluster = max(
                cluster_decisions,
                key=lambda x: x.final_score,
            )

            dominant_cluster_id = (
                dominant_cluster.cluster_id
            )

            dominant_cluster_score = (
                dominant_cluster.final_score
            )

            dominant_cluster_size = (
                dominant_cluster.particle_count
            )

            cluster_evidence = (
                dominant_cluster_score
            )

        else:

            dominant_cluster_id = -1
            dominant_cluster_score = 0.0
            dominant_cluster_size = 0
            cluster_evidence = 0.0

        # ------------------------------------------------------
        # Final mammogram score
        # ------------------------------------------------------

        # The mammogram should primarily be supported by
        # actual positive particles/clusters.
        #
        # Global evidence acts as contextual evidence.

        positive_particle_fraction = (
            len(
                positive_particles
            )
            / max(
                len(particle_decisions),
                1,
            )
        )

        positive_cluster_fraction = (
            len(
                positive_clusters
            )
            / max(
                len(cluster_decisions),
                1,
            )
        )

        final_score = (
            0.45
            * particle_evidence

            + 0.35
            * cluster_evidence

            + 0.20
            * global_score
        )

        # ------------------------------------------------------
        # Prevent global evidence alone from producing
        # a positive final decision.
        #
        # This is important.
        # ------------------------------------------------------

        is_microcalcification_present = (
            (
                len(
                    positive_clusters
                ) > 0
            )
            and
            (
                len(
                    positive_particles
                ) > 0
            )
            and
            (
                final_score
                >= self.final_threshold
            )
        )

        return FinalMCDecision(

            final_score=float(
                np.clip(
                    final_score,
                    0.0,
                    1.0,
                )
            ),

            is_microcalcification_present=(
                is_microcalcification_present
            ),

            particle_evidence=float(
                np.clip(
                    particle_evidence,
                    0.0,
                    1.0,
                )
            ),

            cluster_evidence=float(
                np.clip(
                    cluster_evidence,
                    0.0,
                    1.0,
                )
            ),

            global_evidence=(
                global_score
            ),

            total_particles=len(
                particle_decisions
            ),

            positive_particles=len(
                positive_particles
            ),

            total_clusters=len(
                cluster_decisions
            ),

            positive_clusters=len(
                positive_clusters
            ),

            dominant_particle_id=(
                dominant_particle_id
            ),

            dominant_particle_label=(
                dominant_particle_label
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

            particle_decisions=(
                particle_decisions
            ),

            cluster_decisions=(
                cluster_decisions
            ),
        )