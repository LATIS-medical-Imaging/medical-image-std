from dataclasses import dataclass, field
from typing import List


from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class FinalMCOutput:
    """
    Final native-resolution output of the complete
    microcalcification reasoning pipeline.

    All raster maps have exactly the same spatial
    resolution as the original mammogram.
    """

    # ==========================================================
    # Native-resolution segmentation
    # ==========================================================

    mc_mask: torch.Tensor

    # ==========================================================
    # Continuous evidence
    # ==========================================================

    mc_evidence: torch.Tensor

    # ==========================================================
    # Object maps
    # ==========================================================

    particle_map: torch.Tensor

    cluster_map: torch.Tensor

    # ==========================================================
    # Structured detections
    # ==========================================================

    particles: List = field(
        default_factory=list
    )

    clusters: List = field(
        default_factory=list
    )

    # ==========================================================
    # Global confidence
    # ==========================================================

    global_confidence: float = 0.0

    final_confidence: float = 0.0

    is_microcalcification_present: bool = False

    # ==========================================================
    # Resolution
    # ==========================================================

    height: int = 0
    width: int = 0
@dataclass
class FinalParticleDecision:
    """
    Final decision for one candidate particle.

    The score combines:
        - intrinsic particle evidence
        - cluster context
        - global mammogram context
    """

    particle_id: int
    label: int

    particle_score: float

    cluster_id: int

    cluster_score: float
    cluster_context_score: float

    global_score: float
    global_context_score: float

    final_score: float

    is_microcalcification: bool


@dataclass
class FinalClusterDecision:
    """
    Final decision for one spatial cluster.
    """

    cluster_id: int

    particle_count: int

    cluster_score: float
    global_context_score: float

    final_score: float

    is_microcalcification_cluster: bool

    particle_labels: List[int] = field(
        default_factory=list
    )


@dataclass
class FinalMCDecision:
    """
    Final mammogram-level MC decision.

    This is the output of Step 12.
    """

    # ==========================================================
    # Global decision
    # ==========================================================

    final_score: float

    is_microcalcification_present: bool

    # ==========================================================
    # Evidence components
    # ==========================================================

    particle_evidence: float

    cluster_evidence: float

    global_evidence: float

    # ==========================================================
    # Population
    # ==========================================================

    total_particles: int

    positive_particles: int

    total_clusters: int

    positive_clusters: int

    # ==========================================================
    # Dominant finding
    # ==========================================================

    dominant_particle_id: int

    dominant_particle_label: int

    dominant_cluster_id: int

    dominant_cluster_score: float

    dominant_cluster_size: int

    # ==========================================================
    # Final objects
    # ==========================================================

    particle_decisions: List[
        FinalParticleDecision
    ] = field(
        default_factory=list
    )

    cluster_decisions: List[
        FinalClusterDecision
    ] = field(
        default_factory=list
    )