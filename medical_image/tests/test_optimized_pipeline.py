import numpy as np
import pytest

from medical_image import (
    BreastMaskAlgorithm,
    MammographyPreprocessing,
    MultiScaleTopHatAlgorithm,
    DoG,
    CandidateGenerationAlgorithm,
    InMemoryImage,
    DifferentialBlobAlgorithm,
    CandidateEvidenceAlgorithm,
    ParticleBuilder,
    SpatialReasoningAlgorithm,
)
from medical_image.algorithms.cluster_reasoning_algorithm import (
    ClusterReasoningAlgorithm,
)
from medical_image.algorithms.decision_algorithm import FinalMCDecisionAlgorithm
from medical_image.algorithms.global_mammogram_reasoning_algorithm import (
    GlobalMammogramReasoningAlgorithm,
)
from medical_image.algorithms.local_physical_analysis_algorithm import (
    LocalPhysicalAnalysisAlgorithm,
)
from medical_image.algorithms.native_resolution_segmentation import (
    NativeResolutionSegmentationAlgorithm,
)
from medical_image.tests.mock_sample import (
    mock_dicom_image,
)


class TestOptimizedPipeline:
    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_optimized_NativeResolutionSegmentationAlgorithm(
        self,
        dicom_image,
    ):
        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================
        print("\n\n" + "=" * 80)
        print("STAGE 1: Breast Mask")
        print("=" * 80)

        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )

        breast_mask_tensor = breast_mask.pixel_data.to(device=device).bool()
        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        print(f"Breast mask created. Active pixels: {breast_mask_tensor.sum().item()}")

        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 2: Intensity Normalization")
        print("=" * 80)

        I_raw, I_norm = MammographyPreprocessing.robust_intensity_normalization(
            image=dicom_image,
            breast_mask=breast_mask,
            device=device,
        )

        print(f"Normalized image shape: {I_norm.pixel_data.shape}")

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 3: Multi-scale Top-Hat")
        print("=" * 80)

        top_hat = MultiScaleTopHatAlgorithm(
            device=device,
        )

        top_hat_output = dicom_image.clone()

        top_hat_out = top_hat.apply(
            I_norm,
            top_hat_output,
        )

        print(
            "Top-Hat shape:",
            top_hat_out.pixel_data.shape,
        )

        print(
            f"Top-Hat min: {top_hat_out.pixel_data.min().item()}, max: {top_hat_out.pixel_data.max().item()}"
        )

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 4: DoG (Frequency)")
        print("=" * 80)

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )

        # ENABLE REAL DoG - NO ZEROING

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        print(
            f"DoG min: {frequency_out.pixel_data.min().item()}, max: {frequency_out.pixel_data.max().item()}"
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 5: Multi-scale LoG")
        print("=" * 80)

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[: len(differential.sigmas)]

        # ENABLE REAL LoG
        log_image = InMemoryImage(
            array=log_features,
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        print(f"LoG min: {log_features.min().item()}, max: {log_features.max().item()}")

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 6: Candidate Evidence")
        print("=" * 80)

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=0.5,
            w_dog=1.0,
            device=device,
        )

        evidence_output = dicom_image.clone()

        evidence = evidence_algorithm.apply(
            breast_mask=breast_mask,
            top_hat=top_hat_out,
            log=log_image,
            dog=frequency_out,
            output=evidence_output,
        )

        evidence_data = evidence.pixel_data.float()

        # ==============================================================
        # 7. Evidence validation
        # ==============================================================

        print(
            "Evidence min:",
            evidence_data.min().item(),
        )

        print(
            "Evidence max:",
            evidence_data.max().item(),
        )

        print(
            "Evidence mean:",
            evidence_data.mean().item(),
        )

        # ==============================================================
        # 8. Breast evidence statistics
        # ==============================================================

        breast_evidence = evidence_data[breast_mask_tensor]

        print(
            "Breast evidence min:",
            breast_evidence.min().item(),
        )

        print(
            "Breast evidence max:",
            breast_evidence.max().item(),
        )

        print(
            "Breast evidence mean:",
            breast_evidence.mean().item(),
        )

        # ==============================================================
        # 9. Candidate Generation
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 9: Candidate Generation")
        print("=" * 80)

        # OPTIMIZATION: Percentile instead of Otsu
        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="percentile",
            percentile=0.9,
            device=device,
        )

        candidate_output = dicom_image.clone()

        candidates = candidate_algorithm.apply(
            evidence=evidence,
            breast_mask=breast_mask,
            output=candidate_output,
        )

        # ==============================================================
        # 10. Candidate validation
        # ==============================================================

        candidate_mask = candidates.pixel_data.bool()

        num_candidates = candidate_mask.sum().item()

        total_pixels = dicom_image.height * dicom_image.width

        candidate_ratio = num_candidates / total_pixels

        breast_candidate_count = (candidate_mask & breast_mask_tensor).sum().item()

        breast_pixel_count = breast_mask_tensor.sum().item()

        breast_candidate_ratio = breast_candidate_count / breast_pixel_count

        print(f"Total candidates: " f"{num_candidates}")

        print(f"Candidate ratio: " f"{candidate_ratio:.6%}")

        print(f"Breast candidate pixels: " f"{breast_candidate_count}")

        print(f"Candidate ratio in breast: " f"{breast_candidate_ratio:.6%}")

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[candidate_mask]

        non_candidate_evidence = evidence_data[~candidate_mask & breast_mask_tensor]

        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )

        initial_candidate_pixels = num_candidates

        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = I_raw.pixel_data.detach().cpu().numpy()

        candidate_np = candidate_mask.detach().cpu().numpy()

        evidence_np = evidence_data.detach().cpu().numpy()

        image_vis = image_np.astype(np.float32)

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1) / (p99 - p1 + 1e-8),
            0.0,
            1.0,
        )

        candidate_overlay = np.ma.masked_where(
            ~candidate_np,
            candidate_np,
        )

        print(
            "Candidate output shape:",
            candidates.pixel_data.shape,
        )

        print(
            "Evidence output shape:",
            evidence.pixel_data.shape,
        )

        # ==============================================================
        # 13. Local Physical Analysis
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 13: Local Physical Analysis")
        print("=" * 80)

        print("physical_algorithm")

        # OPTIMIZATION: Tighter physical analysis params
        physical_algorithm = LocalPhysicalAnalysisAlgorithm(
            window_size=31,
            radial_radius=15,
            shape_threshold_k=2.75,
            min_area=1,
            max_area=50,
            connectivity=2,
            device=device,
        )
        print("physical_algorithm init")

        physical_candidates = physical_algorithm.apply(
            image=I_norm,
            candidates=candidates,
            breast_mask=breast_mask,
        )
        print("physical_algorithm apply")

        physical_candidate_count = len(physical_candidates)
        print(
            "Physical candidates:",
            physical_candidate_count,
        )

        # ==============================================================
        # 14. Candidate → Particle conversion
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 14: Particle Builder")
        print("=" * 80)

        particle_builder = ParticleBuilder(
            device=device,
            scale_neighborhood_radius=1,
        )

        particles, filtered_physical = particle_builder.build(
            physical_candidates=physical_candidates,
            top_hat_maps=top_hat_out.pixel_data,
            log_maps=log_features,
            hf_maps=frequency_out.pixel_data,
        )

        particle_count = len(particles)
        filtered_particles = physical_candidate_count - particle_count

        print(
            "Particles:",
            particle_count,
        )

        # ==============================================================
        # 15. Spatial Reasoning
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 15: Spatial Reasoning")
        print("=" * 80)

        # OPTIMIZATION: Larger spatial neighborhood
        spatial_algorithm = SpatialReasoningAlgorithm(
            neighborhood_radius=50.0,
            min_cluster_neighbors=2,
            device=device,
        )

        spatial_features = spatial_algorithm.apply(physical_candidates)
        print("Spatial features computed")

        # ==============================================================
        # 16. Cluster Reasoning
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 16: Cluster Reasoning")
        print("=" * 80)

        # OPTIMIZATION: Fixed cluster reasoning
        cluster_algorithm = ClusterReasoningAlgorithm(
            min_particles=1,
            density_radius=20.0,
            min_density=0.005,
            particle_score_threshold=0.5,
            device=device,
        )

        clusters = cluster_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
        )

        cluster_count = len(clusters)
        print(f"Found {cluster_count} clusters")

        # ==============================================================
        # 17. Global Reasoning
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 17: Global Reasoning")
        print("=" * 80)

        global_algorithm = GlobalMammogramReasoningAlgorithm(
            min_significant_cluster_score=0.50,
            min_significant_cluster_particles=5,
            device=device,
        )

        global_features = global_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
            clusters=clusters,
        )
        print("Global features computed")

        # ==============================================================
        # 18. Decision Algorithm
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 18: Decision Algorithm")
        print("=" * 80)

        # OPTIMIZATION: Higher decision thresholds
        final_algorithm = FinalMCDecisionAlgorithm(
            particle_weight=0.40,
            cluster_weight=0.40,
            global_weight=0.20,
            particle_threshold=0.3,
            cluster_threshold=0.3,
            final_threshold=0.3,
            device=device,
        )

        final_decision = final_algorithm.apply(
            particles=particles,
            physical_features=filtered_physical,
            clusters=clusters,
            global_features=global_features,
        )

        positive_particles = final_decision.positive_particles
        positive_clusters = final_decision.positive_clusters

        filtered_positive_particles = particle_count - positive_particles
        filtered_positive_clusters = cluster_count - positive_clusters

        print(f"Final score: {final_decision.final_score:.4f}")
        print(f"Is MC present: {final_decision.is_microcalcification_present}")
        print(f"Particle evidence: {final_decision.particle_evidence:.4f}")
        print(f"Cluster evidence: {final_decision.cluster_evidence:.4f}")
        print(f"Global evidence: {final_decision.global_evidence:.4f}")
        print(f"Total particles: {final_decision.total_particles}")
        print(f"Positive particles: {positive_particles}")
        print(f"Total clusters: {final_decision.total_clusters}")
        print(f"Positive clusters: {positive_clusters}")
        print(f"Dominant particle id: {final_decision.dominant_particle_id}")
        print(f"Dominant cluster id: {final_decision.dominant_cluster_id}")
        print(f"Dominant cluster score: {final_decision.dominant_cluster_score:.4f}")

        # Print top-5 particle decisions by score
        if final_decision.particle_decisions:
            sorted_particles = sorted(
                final_decision.particle_decisions,
                key=lambda d: d.final_score,
                reverse=True,
            )
            print("\nTop-5 particle decisions:")
            for pd in sorted_particles[:5]:
                print(
                    f"  particle={pd.particle_id} "
                    f"label={pd.label} "
                    f"p_score={pd.particle_score:.4f} "
                    f"c_ctx={pd.cluster_context_score:.4f} "
                    f"g_ctx={pd.global_context_score:.4f} "
                    f"final={pd.final_score:.4f} "
                    f"is_mc={pd.is_microcalcification}"
                )

        # ==============================================================
        # 19. Segmentation
        # ==============================================================
        print("\n" + "=" * 80)
        print("STAGE 19: Native Resolution Segmentation")
        print("=" * 80)

        segmentation_algorithm = NativeResolutionSegmentationAlgorithm(
            connectivity=2,
            device=device,
        )

        final_output = segmentation_algorithm.apply(
            candidates=candidates,
            physical_features=filtered_physical,
            clusters=clusters,
            final_decision=final_decision,
        )
        mask_fiinal = final_output.mc_mask.cpu().detach().numpy()

        print(f"Final mask active pixels: {mask_fiinal.sum()}")
        print("==============================================")

        print("\n\n")
        print("============== PIPELINE SUMMARY ==============")
        print(f"{'Stage':<22} | {'Count':<8} | {'Filtered':<8}")
        print("-" * 47)
        print(f"{'Candidate pixels':<22} | {initial_candidate_pixels:<8} | {'-':<8}")
        print(f"{'Physical candidates':<22} | {physical_candidate_count:<8} | {'-':<8}")
        print(
            f"{'Particles (after filter)':<22} | {particle_count:<8} | {filtered_particles:<8}"
        )
        print(f"{'Clusters':<22} | {cluster_count:<8} | {'-':<8}")
        print(
            f"{'Positive particles':<22} | {positive_particles:<8} | {filtered_positive_particles:<8}"
        )
        print(
            f"{'Positive clusters':<22} | {positive_clusters:<8} | {filtered_positive_clusters:<8}"
        )
        print("==============================================\n")

        num_true = np.count_nonzero(mask_fiinal)
        num_false = np.count_nonzero(~mask_fiinal)

        print("True:", num_true)
        print("False:", num_false)
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
