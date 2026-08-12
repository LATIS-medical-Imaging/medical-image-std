from dataclasses import dataclass
from typing import List


@dataclass
class GlobalMammogramFeatures:
    """
    Global mammogram-level reasoning features.

    This object summarizes the complete candidate/cluster
    population of one mammogram.

    It does not inspect image pixels.

    It operates exclusively on:

        - CandidatePhysicalFeatures
        - SpatialParticleFeatures
        - ClusterFeatures
    """

    # ==========================================================
    # Candidate population
    # ==========================================================

    total_particles: int

    clustered_particles: int

    isolated_particles: int

    clustered_particle_ratio: float

    isolated_particle_ratio: float

    # ==========================================================
    # Cluster population
    # ==========================================================

    total_clusters: int

    valid_clusters: int

    cluster_ratio: float

    # ==========================================================
    # Cluster size statistics
    # ==========================================================

    mean_cluster_size: float

    median_cluster_size: float

    max_cluster_size: int

    std_cluster_size: float

    # ==========================================================
    # Cluster score statistics
    # ==========================================================

    mean_cluster_score: float

    median_cluster_score: float

    max_cluster_score: float

    std_cluster_score: float

    # ==========================================================
    # Cluster density statistics
    # ==========================================================

    mean_cluster_density: float

    max_cluster_density: float

    # ==========================================================
    # Cluster geometry
    # ==========================================================

    mean_cluster_compactness: float

    max_cluster_compactness: float

    mean_cluster_eccentricity: float

    # ==========================================================
    # Particle evidence
    # ==========================================================

    mean_particle_score: float

    max_particle_score: float

    clustered_particle_evidence: float

    global_particle_evidence: float

    # ==========================================================
    # Spatial distribution
    # ==========================================================

    global_center_y: float

    global_center_x: float

    spatial_spread_y: float

    spatial_spread_x: float

    global_spatial_spread: float

    # ==========================================================
    # Dominant cluster
    # ==========================================================

    dominant_cluster_id: int

    dominant_cluster_score: float

    dominant_cluster_size: int

    dominant_cluster_fraction: float

    # ==========================================================
    # Cluster concentration
    # ==========================================================

    cluster_score_concentration: float

    particle_concentration: float

    # ==========================================================
    # Global evidence components
    # ==========================================================

    cluster_presence_score: float

    cluster_strength_score: float

    particle_support_score: float

    spatial_concentration_score: float

    global_evidence_score: float

    # ==========================================================
    # Final interpretation
    # ==========================================================

    has_significant_cluster: bool