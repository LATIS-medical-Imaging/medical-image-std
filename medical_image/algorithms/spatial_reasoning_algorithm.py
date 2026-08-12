from typing import List, Sequence

import numpy as np
from scipy.spatial import cKDTree

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.physical_features import (
    CandidatePhysicalFeatures,
    SpatialParticleFeatures,
)


class SpatialReasoningAlgorithm(Algorithm):
    """
    Native-resolution spatial reasoning over candidate particles.

    This algorithm does not inspect image pixels.

    It operates exclusively on candidate object centers and
    their previously computed physical evidence.

    Each candidate is represented as a point:

        p_i = (x_i, y_i)

    A spatial neighborhood is defined as:

        N_i = {j : ||p_i - p_j|| <= radius}

    The algorithm computes:

        - nearest-neighbor distance
        - second-nearest-neighbor distance
        - number of neighbors
        - local particle density
        - mean / median / std neighbor distance
        - minimum / maximum neighbor distance
        - neighbor evidence statistics
        - distance quantiles
        - preliminary spatial connected components

    Important:

        This is NOT the final cluster reasoning stage.

        Step 9 constructs the spatial representation.

        Step 10 will reason about whether those spatial
        structures are actually compatible with a
        microcalcification cluster.
    """

    def __init__(
        self,
        neighborhood_radius: float = 20.0,
        min_cluster_neighbors: int = 1,
        include_self: bool = False,
        device: str = "cpu",
    ):
        super().__init__(device=device)

        if neighborhood_radius <= 0:
            raise ValueError(
                "neighborhood_radius must be > 0."
            )

        if min_cluster_neighbors < 1:
            raise ValueError(
                "min_cluster_neighbors must be >= 1."
            )

        self.neighborhood_radius = float(
            neighborhood_radius
        )

        self.min_cluster_neighbors = int(
            min_cluster_neighbors
        )

        self.include_self = include_self

    # ======================================================
    # Score extraction
    # ======================================================

    @staticmethod
    def _get_particle_score(
        particle: CandidatePhysicalFeatures,
    ) -> float:
        """
        Obtain the current local evidence score.

        Step 8 does not yet have a final Particle Score,
        therefore robust_peak_z is used as the current
        local evidence proxy.

        When the final Particle object is introduced,
        this method can use particle_score instead without
        changing the spatial reasoning logic.
        """

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

    # ======================================================
    # Input validation
    # ======================================================

    @staticmethod
    def _validate_particles(
        particles: Sequence[
            CandidatePhysicalFeatures
        ],
    ):

        for particle in particles:

            if not np.isfinite(
                particle.center_x
            ):
                raise ValueError(
                    f"Particle {particle.label} "
                    "has invalid center_x."
                )

            if not np.isfinite(
                particle.center_y
            ):
                raise ValueError(
                    f"Particle {particle.label} "
                    "has invalid center_y."
                )

    # ======================================================
    # Spatial graph
    # ======================================================

    def _build_spatial_graph(
        self,
        positions: np.ndarray,
    ):
        """
        Build the radius-neighborhood graph.

        positions:
            [N, 2]

        Returns:
            neighbors:
                list of neighboring particle indices

        Edge rule:

            distance(i, j) <= radius
        """

        tree = cKDTree(
            positions
        )

        neighbors = tree.query_ball_point(
            positions,
            r=self.neighborhood_radius,
        )

        if not self.include_self:

            for i in range(
                len(neighbors)
            ):
                neighbors[i] = [
                    j
                    for j in neighbors[i]
                    if j != i
                ]

        return neighbors

    # ======================================================
    # Connected spatial groups
    # ======================================================

    @staticmethod
    def _spatial_groups(
        neighbors,
    ):
        """
        Find connected components of the radius graph.

        This is intentionally a simple spatial grouping.

        It is NOT the final cluster classifier.

        If:

            A -- B -- C

        then A, B and C belong to the same preliminary
        spatial group even if A and C are farther apart
        than the neighborhood radius.
        """

        n = len(neighbors)

        group_ids = np.full(
            n,
            -1,
            dtype=np.int32,
        )

        current_group = 0

        for start in range(n):

            if group_ids[start] != -1:
                continue

            stack = [start]

            group_ids[start] = (
                current_group
            )

            while stack:

                current = stack.pop()

                for neighbor in neighbors[
                    current
                ]:

                    if (
                        group_ids[neighbor]
                        == -1
                    ):
                        group_ids[neighbor] = (
                            current_group
                        )

                        stack.append(
                            neighbor
                        )

            current_group += 1

        return group_ids

    # ======================================================
    # Main
    # ======================================================

    def apply(
        self,
        particles: List[
            CandidatePhysicalFeatures
        ],
    ) -> List[
        SpatialParticleFeatures
    ]:

        if particles is None:
            raise ValueError(
                "particles must not be None."
            )

        if len(particles) == 0:
            return []

        self._validate_particles(
            particles
        )

        n = len(particles)

        # --------------------------------------------------
        # Convert candidate centers into spatial points.
        #
        # IMPORTANT:
        #
        # This is still native mammogram coordinates.
        #
        # No resizing.
        # No normalization.
        # No image processing.
        # --------------------------------------------------

        positions = np.asarray(
            [
                (
                    particle.center_x,
                    particle.center_y,
                )
                for particle in particles
            ],
            dtype=np.float64,
        )

        scores = np.asarray(
            [
                self._get_particle_score(
                    particle
                )
                for particle in particles
            ],
            dtype=np.float64,
        )

        # --------------------------------------------------
        # Build spatial neighborhood graph.
        # --------------------------------------------------

        neighbors = (
            self._build_spatial_graph(
                positions
            )
        )

        # --------------------------------------------------
        # Preliminary spatial groups.
        # --------------------------------------------------

        group_ids = (
            self._spatial_groups(
                neighbors
            )
        )

        results = []

        # Area of circular neighborhood.
        neighborhood_area = (
            np.pi
            * self.neighborhood_radius ** 2
        )

        # --------------------------------------------------
        # Analyze each particle.
        # --------------------------------------------------

        for i, particle in enumerate(
            particles
        ):

            neighbor_indices = (
                neighbors[i]
            )

            # ----------------------------------------------
            # Isolated particle
            # ----------------------------------------------

            if len(
                neighbor_indices
            ) == 0:

                results.append(
                    SpatialParticleFeatures(
                        label=particle.label,

                        center_y=particle.center_y,
                        center_x=particle.center_x,

                        nearest_neighbor_distance=float(
                            "inf"
                        ),

                        second_nearest_neighbor_distance=float(
                            "inf"
                        ),

                        neighbor_count=0,

                        local_density=0.0,

                        mean_neighbor_distance=float(
                            "inf"
                        ),

                        median_neighbor_distance=float(
                            "inf"
                        ),

                        std_neighbor_distance=0.0,

                        min_neighbor_distance=float(
                            "inf"
                        ),

                        max_neighbor_distance=float(
                            "inf"
                        ),

                        mean_neighbor_score=0.0,

                        max_neighbor_score=0.0,

                        distance_q25=float(
                            "inf"
                        ),

                        distance_q75=float(
                            "inf"
                        ),

                        spatial_group_id=int(
                            group_ids[i]
                        ),
                    )
                )

                continue

            # ----------------------------------------------
            # Distances to neighbors
            # ----------------------------------------------

            neighbor_positions = (
                positions[
                    neighbor_indices
                ]
            )

            delta = (
                neighbor_positions
                - positions[i]
            )

            distances = np.linalg.norm(
                delta,
                axis=1,
            )

            distances.sort()

            # ----------------------------------------------
            # Basic neighborhood statistics
            # ----------------------------------------------

            neighbor_count = len(
                neighbor_indices
            )

            local_density = (
                neighbor_count
                / neighborhood_area
            )

            # ----------------------------------------------
            # Distance statistics
            # ----------------------------------------------

            nearest_neighbor_distance = (
                float(
                    distances[0]
                )
            )

            if len(distances) >= 2:

                second_nearest_neighbor_distance = (
                    float(
                        distances[1]
                    )
                )

            else:

                second_nearest_neighbor_distance = (
                    float(
                        "inf"
                    )
                )

            mean_neighbor_distance = (
                float(
                    np.mean(distances)
                )
            )

            median_neighbor_distance = (
                float(
                    np.median(distances)
                )
            )

            std_neighbor_distance = (
                float(
                    np.std(distances)
                )
            )

            min_neighbor_distance = (
                float(
                    np.min(distances)
                )
            )

            max_neighbor_distance = (
                float(
                    np.max(distances)
                )
            )

            distance_q25 = float(
                np.percentile(
                    distances,
                    25,
                )
            )

            distance_q75 = float(
                np.percentile(
                    distances,
                    75,
                )
            )

            # ----------------------------------------------
            # Neighbor evidence
            # ----------------------------------------------

            neighbor_scores = scores[
                neighbor_indices
            ]

            mean_neighbor_score = (
                float(
                    np.mean(
                        neighbor_scores
                    )
                )
            )

            max_neighbor_score = (
                float(
                    np.max(
                        neighbor_scores
                    )
                )
            )

            # ----------------------------------------------
            # Result
            # ----------------------------------------------

            results.append(
                SpatialParticleFeatures(
                    label=particle.label,

                    center_y=particle.center_y,
                    center_x=particle.center_x,

                    nearest_neighbor_distance=(
                        nearest_neighbor_distance
                    ),

                    second_nearest_neighbor_distance=(
                        second_nearest_neighbor_distance
                    ),

                    neighbor_count=(
                        neighbor_count
                    ),

                    local_density=(
                        local_density
                    ),

                    mean_neighbor_distance=(
                        mean_neighbor_distance
                    ),

                    median_neighbor_distance=(
                        median_neighbor_distance
                    ),

                    std_neighbor_distance=(
                        std_neighbor_distance
                    ),

                    min_neighbor_distance=(
                        min_neighbor_distance
                    ),

                    max_neighbor_distance=(
                        max_neighbor_distance
                    ),

                    mean_neighbor_score=(
                        mean_neighbor_score
                    ),

                    max_neighbor_score=(
                        max_neighbor_score
                    ),

                    distance_q25=(
                        distance_q25
                    ),

                    distance_q75=(
                        distance_q75
                    ),

                    spatial_group_id=int(
                        group_ids[i]
                    ),
                )
            )

        return results