from dataclasses import dataclass, asdict


@dataclass
class CandidatePhysicalFeatures:
    """
    Physical descriptors of one candidate object.
    """

    label: int

    # ----------------------------------------------------------
    # Position
    # ----------------------------------------------------------

    center_y: float
    center_x: float

    # ----------------------------------------------------------
    # Peak prominence
    # ----------------------------------------------------------

    peak_value: float
    neighborhood_median: float
    neighborhood_mad: float
    robust_peak_z: float

    # ----------------------------------------------------------
    # Radial profile
    # ----------------------------------------------------------

    radial_center: float
    radial_ring_mean: float
    center_ring_contrast: float
    radial_decay: float
    radial_peak_width: float
    radial_symmetry: float

    # ----------------------------------------------------------
    # Shape
    # ----------------------------------------------------------

    area: float
    perimeter: float
    circularity: float
    eccentricity: float
    solidity: float
    aspect_ratio: float
    equivalent_diameter: float


from dataclasses import dataclass, field
from typing import List


from dataclasses import dataclass
from typing import List


@dataclass
class SpatialParticleFeatures:
    """
    Spatial descriptors of one candidate particle.

    These features describe the relationship of a particle
    to the other candidate particles in the mammogram.

    Pixel/intensity/shape properties are intentionally not
    duplicated here. They remain in CandidatePhysicalFeatures.
    """

    # ------------------------------------------------------
    # Identity
    # ------------------------------------------------------

    label: int

    center_y: float
    center_x: float

    # ------------------------------------------------------
    # Nearest-neighbor statistics
    # ------------------------------------------------------

    nearest_neighbor_distance: float

    second_nearest_neighbor_distance: float

    # ------------------------------------------------------
    # Neighborhood statistics
    # ------------------------------------------------------

    neighbor_count: int

    local_density: float

    mean_neighbor_distance: float

    median_neighbor_distance: float

    std_neighbor_distance: float

    min_neighbor_distance: float

    max_neighbor_distance: float

    # ------------------------------------------------------
    # Neighbor evidence
    # ------------------------------------------------------

    mean_neighbor_score: float

    max_neighbor_score: float

    # ------------------------------------------------------
    # Distance distribution
    # ------------------------------------------------------

    distance_q25: float
    distance_q75: float

    # ------------------------------------------------------
    # Spatial grouping
    # ------------------------------------------------------

    spatial_group_id: int

@dataclass
class Particle:
    """
    Final physical representation of one candidate
    microcalcification particle.

    A Particle is built from the physical descriptors
    extracted by LocalPhysicalAnalysis and enriched with
    multi-scale evidence signatures.
    """

    # ==========================================================
    # Identity
    # ==========================================================

    id: int
    label: int

    # ==========================================================
    # Position
    # ==========================================================

    x: float
    y: float

    # ==========================================================
    # Basic physical properties
    # ==========================================================

    area: float
    diameter: float
    perimeter: float

    # ==========================================================
    # Intensity / peak
    # ==========================================================

    intensity: float
    neighborhood_median: float
    neighborhood_mad: float
    peak_prominence: float

    # ==========================================================
    # Shape
    # ==========================================================

    circularity: float
    eccentricity: float
    solidity: float
    aspect_ratio: float

    # ==========================================================
    # Radial structure
    # ==========================================================

    radial_score: float
    radial_decay: float
    radial_peak_width: float
    radial_symmetry: float
    center_ring_contrast: float

    # ==========================================================
    # Multi-scale signatures
    # ==========================================================

    top_hat_signature: List[float] = field(
        default_factory=list
    )

    log_signature: List[float] = field(
        default_factory=list
    )

    hf_signature: List[float] = field(
        default_factory=list
    )

    # ==========================================================
    # Scale characterization
    # ==========================================================

    scale_score: float = 0.0
    dominant_scale: float = 0.0
    scale_width: float = 0.0

    # ==========================================================
    # Other physical evidence
    # ==========================================================

    hessian_score: float = 0.0
    frequency_score: float = 0.0
    orientation_score: float = 0.0

    # ==========================================================
    # Final score
    # ==========================================================

    particle_score: float = 0.0
