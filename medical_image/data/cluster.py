from dataclasses import dataclass
from typing import List


@dataclass
class ClusterFeatures:
    """
    Cluster-level physical and spatial descriptors.

    A cluster is a group of candidate particles connected
    through the spatial neighborhood graph constructed
    during Step 9.

    This object deliberately contains information about
    the GROUP rather than an individual particle.
    """

    # ------------------------------------------------------
    # Identity
    # ------------------------------------------------------

    cluster_id: int

    # ------------------------------------------------------
    # Particles
    # ------------------------------------------------------

    particle_labels: List[int]

    particle_count: int

    # ------------------------------------------------------
    # Spatial extent
    # ------------------------------------------------------

    center_y: float
    center_x: float

    extent_y: float
    extent_x: float

    spatial_extent: float

    # ------------------------------------------------------
    # Density
    # ------------------------------------------------------

    density: float

    # ------------------------------------------------------
    # Nearest-neighbor structure
    # ------------------------------------------------------

    mean_nearest_neighbor_distance: float

    median_nearest_neighbor_distance: float

    std_nearest_neighbor_distance: float

    min_nearest_neighbor_distance: float

    max_nearest_neighbor_distance: float

    # ------------------------------------------------------
    # Inter-particle distance structure
    # ------------------------------------------------------

    mean_pairwise_distance: float

    median_pairwise_distance: float

    max_pairwise_distance: float

    distance_q25: float

    distance_q75: float

    # ------------------------------------------------------
    # Particle evidence
    # ------------------------------------------------------

    mean_particle_score: float

    median_particle_score: float

    std_particle_score: float

    min_particle_score: float

    max_particle_score: float

    # ------------------------------------------------------
    # Particle shape statistics
    # ------------------------------------------------------

    mean_circularity: float

    mean_eccentricity: float

    mean_solidity: float

    mean_equivalent_diameter: float

    # ------------------------------------------------------
    # Spatial geometry
    # ------------------------------------------------------

    spatial_eccentricity: float

    spatial_compactness: float

    spatial_regularity: float

    # ------------------------------------------------------
    # Cluster evidence
    # ------------------------------------------------------

    density_score: float

    compactness_score: float

    particle_evidence_score: float

    geometry_score: float

    cluster_score: float

    # ------------------------------------------------------
    # Classification metadata
    # ------------------------------------------------------

    is_cluster: bool