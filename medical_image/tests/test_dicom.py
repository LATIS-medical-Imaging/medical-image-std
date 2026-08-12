import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage._shared.filters import gaussian
from skimage.filters import difference_of_gaussians, threshold_otsu

from  medical_image.algorithms.global_mammogram_reasoning_algorithm import GlobalMammogramReasoningAlgorithm
from  medical_image.algorithms.cluster_reasoning_algorithm import ClusterReasoningAlgorithm
from  medical_image.algorithms.decision_algorithm import FinalMCDecisionAlgorithm
from medical_image.algorithms.native_resolution_segmentation import NativeResolutionSegmentationAlgorithm
from medical_image import RegionOfInterest, SimpleApproach, BreastMaskAlgorithm, MammographyPreprocessing, \
    MultiScaleTopHatAlgorithm, DoG, GaborOrientationAlgorithm, CandidateGenerationAlgorithm, \
    InMemoryImage, DifferentialBlobAlgorithm, CandidateEvidenceAlgorithm, ParticleBuilder, SpatialReasoningAlgorithm
from medical_image.utils.logging import logger
from medical_image.algorithms.FEBDS import FebdsAlgorithm
from medical_image.algorithms.local_physical_analysis_algorithm import LocalPhysicalAnalysisAlgorithm
from medical_image.algorithms.sbrg import SbrgAlgorithm
from medical_image.algorithms.custom_algorithm import CustomAlgorithm
from medical_image.data.dicom_image import DicomImage
from medical_image.data.patch import PatchGrid
from medical_image.process.filters import Filters
from medical_image.process.morphology import MorphologyOperations
from medical_image.process.threshold import Threshold
from medical_image.tests.mock_sample import (
    mock_dicom_image,
    mock_sauvola_threshold,
    mock_png_image,
    mock_kernel,
    mock_two_sigmas,
    mock_kernel_sizes,
)
from medical_image.utils.image_utils import ImageExporter


def morphoogy_closing(input):
    return ndimage.binary_closing(input, structure=np.ones((7, 7))).astype(np.int64)


def region_fill(input):
    return ndimage.binary_fill_holes(input, structure=np.ones((7, 7))).astype(int)


class TestDicom:
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_dicom_image(self, dicom_image):
        ImageExporter.save_as(dicom_image)
        assert dicom_image.pixel_data is not None
        assert dicom_image.width > 0
        assert dicom_image.height > 0

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_otsu_threshold(self, dicom_image):
        output = dicom_image.clone()
        Threshold.otsu_threshold(dicom_image, output)

        assert not np.array_equal(
            dicom_image.pixel_data.cpu().numpy(), output.pixel_data.cpu().numpy()
        )

        unique_vals = torch.unique(output.pixel_data)
        assert torch.all((unique_vals == 0) | (unique_vals == 1))

    @pytest.mark.parametrize("dicom_image, window_size, k", mock_sauvola_threshold())
    def test_sauvola_threshold(self, dicom_image, window_size, k):
        output = dicom_image.clone()
        Threshold.sauvola_threshold(dicom_image, output, window_size, k)

        assert not torch.equal(
            dicom_image.pixel_data.float(), output.pixel_data.float()
        )

        out = output.pixel_data
        assert torch.all((out == 0) | (out == 255))

    @pytest.mark.parametrize("size, sigma", mock_kernel())
    def test_gaussian_kernel_matches_skimage(self, size, sigma):
        image = np.random.rand(8, 8).astype(np.float32)
        sigma = 1.5
        truncate = 4.0
        image_object = DicomImage.from_array(image)
        output = image_object.clone()

        Filters.gaussian_filter(image_object, output, sigma, truncate=truncate)

        skimage_result = gaussian(
            image, sigma=sigma, mode="nearest", truncate=truncate, preserve_range=False
        ).astype(np.float32)

        np.testing.assert_allclose(
            output.pixel_data, skimage_result, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_febds_fft(self, dicom_image):
        image = dicom_image.pixel_data
        sigma1, sigma2 = 1.7, 2.0
        skimage_result = difference_of_gaussians(image, sigma1, sigma2)
        skimage_result_finished = ndimage.median_filter(
            np.abs(skimage_result), size=(5, 5)
        )
        fi = np.power(skimage_result_finished / 4095.0, 1.25)
        fi = fi * 4095
        x = threshold_otsu(fi)
        out = np.zeros_like(fi)
        out[fi > x] = 1
        I = morphoogy_closing(out)
        fill = region_fill(I)

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()
        print("image.pixel_data.shape")
        print(image.shape)
        algorithm = FebdsAlgorithm("fft")
        algorithm(image=dicom_image, output=output)

        image_output = (
            output.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((dicom_image.height, dicom_image.width))
        )
        assert not torch.allclose(
            torch.tensor(I).float(), output.pixel_data.detach().cpu()
        )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_febds(self, dicom_image):
        image = dicom_image.pixel_data
        sigma1, sigma2 = 1.7, 2.0
        skimage_result = difference_of_gaussians(image, sigma1, sigma2)
        skimage_result_finished = ndimage.median_filter(
            np.abs(skimage_result), size=(5, 5)
        )
        fi = np.power(skimage_result_finished / 4095.0, 1.25)
        fi = fi * 4095
        x = threshold_otsu(fi)
        out = np.zeros_like(fi)
        out[fi > x] = 1
        I = morphoogy_closing(out)
        fill = region_fill(I)

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        algorithm = FebdsAlgorithm("dog")
        algorithm(image=dicom_image, output=output)

        image_output = (
            output.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((dicom_image.height, dicom_image.width))
        )
        assert not torch.allclose(
            torch.tensor(I).float(), output.pixel_data.detach().cpu()
        )
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_simple_algorithm(self, dicom_image):
        image = dicom_image.pixel_data
        sigma1, sigma2 = 1.7, 2.0
        skimage_result = difference_of_gaussians(image, sigma1, sigma2)
        skimage_result_finished = ndimage.median_filter(
            np.abs(skimage_result), size=(5, 5)
        )
        fi = np.power(skimage_result_finished / 4095.0, 1.25)
        fi = fi * 4095
        x = threshold_otsu(fi)
        out = np.zeros_like(fi)
        out[fi > x] = 1
        I = morphoogy_closing(out)
        fill = region_fill(I)

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        algorithm = SimpleApproach()
        algorithm(image=dicom_image, output=output)

        image_output = (
            output.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((dicom_image.height, dicom_image.width))
        )
        assert not torch.allclose(
            torch.tensor(I).float(), output.pixel_data.detach().cpu()
        )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_breast_mask_algorithm(self, dicom_image):
        image = dicom_image.pixel_data

        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        algorithm = BreastMaskAlgorithm(mask_only=True)
        algorithm(image=dicom_image, output=output)

        image_output = (
            output.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((dicom_image.height, dicom_image.width))
        )
        print(image_output)

    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_local_physical_GlobalMammogramAlgorithm(
            self,
            dicom_image,
    ):

        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )


        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================

        I_raw, I_norm = (
            MammographyPreprocessing
            .robust_intensity_normalization(
                image=dicom_image,
                breast_mask=breast_mask,
                device=device,
            )
        )

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================

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

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        log_image = InMemoryImage(
            array=log_features,
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=1.0,
            w_dog=0.75,
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

        breast_mask_tensor = (
            breast_mask.pixel_data
            .to(device=device)
            .bool()
        )

        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        breast_evidence = evidence_data[
            breast_mask_tensor
        ]


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

        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="otsu",
            percentile=0.995,
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

        candidate_mask = (
            candidates.pixel_data.bool()
        )



        num_candidates = (
            candidate_mask.sum().item()
        )

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )

        breast_candidate_count = (
                candidate_mask & breast_mask_tensor
        ).sum().item()

        breast_pixel_count = (
            breast_mask_tensor.sum().item()
        )

        breast_candidate_ratio = (
                breast_candidate_count
                / breast_pixel_count
        )

        print(
            f"Total candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )

        print(
            f"Breast candidate pixels: "
            f"{breast_candidate_count}"
        )

        print(
            f"Candidate ratio in breast: "
            f"{breast_candidate_ratio:.6%}"
        )

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[
            candidate_mask
        ]

        non_candidate_evidence = evidence_data[
            ~candidate_mask & breast_mask_tensor
            ]


        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )



        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidate_mask
            .detach()
            .cpu()
            .numpy()
        )

        evidence_np = (
            evidence_data
            .detach()
            .cpu()
            .numpy()
        )

        image_vis = image_np.astype(
            np.float32
        )

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1)
            / (p99 - p1 + 1e-8),
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
        print("physical_algorithm")
        physical_algorithm = LocalPhysicalAnalysisAlgorithm(
            window_size=31,
            radial_radius=15,
            shape_threshold_k=1.5,
            min_area=1,
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


        print(
            "Physical candidates:",
            len(physical_candidates),
        )
        # ==============================================================
        # 14. Candidate → Particle conversion
        # ==============================================================

        particle_builder = ParticleBuilder(
            device=device,
            scale_neighborhood_radius=1,
        )

        particles = particle_builder.build(
            physical_candidates=physical_candidates,

            top_hat_maps=top_hat_out.pixel_data,

            log_maps=log_features,

            hf_maps=frequency_out.pixel_data,
        )

        print(
            "Particles:",
            len(particles),
        )


        spatial_algorithm = SpatialReasoningAlgorithm(
            neighborhood_radius=20.0,
            min_cluster_neighbors=1,
            device=device,
        )

        spatial_features = spatial_algorithm.apply(
            physical_candidates
        )
        cluster_algorithm = ClusterReasoningAlgorithm(
            min_particles=3,
            density_radius=20.0,
            min_density=0.001,
            particle_score_threshold=2.0,
            device=device,
        )

        clusters = cluster_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
        )
        print("Particle builder")

        global_algorithm = (
            GlobalMammogramReasoningAlgorithm(
                min_significant_cluster_score=0.50,
                min_significant_cluster_particles=3,
                device=device,
            )
        )

        global_features = global_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
            clusters=clusters,
        )

        print(
            "=============================================="
        )

    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_local_physical_NativeResolutionSegmentationAlgorithm(
            self,
            dicom_image,
    ):

        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )


        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================

        I_raw, I_norm = (
            MammographyPreprocessing
            .robust_intensity_normalization(
                image=dicom_image,
                breast_mask=breast_mask,
                device=device,
            )
        )

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================

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

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )
        # TODO:
        frequency_out.pixel_data = torch.zeros_like(frequency_out.pixel_data)

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        log_image = InMemoryImage(
            # TODO array=log_features,
            array=torch.zeros_like(log_features),
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=1.0,
            w_dog=0.75,
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

        breast_mask_tensor = (
            breast_mask.pixel_data
            .to(device=device)
            .bool()
        )

        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        breast_evidence = evidence_data[
            breast_mask_tensor
        ]


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

        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="otsu",
            percentile=0.995,
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

        candidate_mask = (
            candidates.pixel_data.bool()
        )



        num_candidates = (
            candidate_mask.sum().item()
        )

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )

        breast_candidate_count = (
                candidate_mask & breast_mask_tensor
        ).sum().item()

        breast_pixel_count = (
            breast_mask_tensor.sum().item()
        )

        breast_candidate_ratio = (
                breast_candidate_count
                / breast_pixel_count
        )

        print(
            f"Total candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )

        print(
            f"Breast candidate pixels: "
            f"{breast_candidate_count}"
        )

        print(
            f"Candidate ratio in breast: "
            f"{breast_candidate_ratio:.6%}"
        )

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[
            candidate_mask
        ]

        non_candidate_evidence = evidence_data[
            ~candidate_mask & breast_mask_tensor
            ]


        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )



        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidate_mask
            .detach()
            .cpu()
            .numpy()
        )

        evidence_np = (
            evidence_data
            .detach()
            .cpu()
            .numpy()
        )

        image_vis = image_np.astype(
            np.float32
        )

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1)
            / (p99 - p1 + 1e-8),
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
        print("physical_algorithm")
        physical_algorithm = LocalPhysicalAnalysisAlgorithm(
            window_size=31,
            radial_radius=15,
            shape_threshold_k=1.5,
            min_area=1,
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


        print(
            "Physical candidates:",
            len(physical_candidates),
        )
        # ==============================================================
        # 14. Candidate → Particle conversion
        # ==============================================================

        particle_builder = ParticleBuilder(
            device=device,
            scale_neighborhood_radius=1,
        )

        particles = particle_builder.build(
            physical_candidates=physical_candidates,

            top_hat_maps=top_hat_out.pixel_data,

            log_maps=log_features,

            hf_maps=frequency_out.pixel_data,
        )

        print(
            "Particles:",
            len(particles),
        )


        spatial_algorithm = SpatialReasoningAlgorithm(
            neighborhood_radius=20.0,
            min_cluster_neighbors=1,
            device=device,
        )

        spatial_features = spatial_algorithm.apply(
            physical_candidates
        )
        cluster_algorithm = ClusterReasoningAlgorithm(
            min_particles=3,
            density_radius=20.0,
            min_density=0.001,
            particle_score_threshold=2.0,
            device=device,
        )

        clusters = cluster_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
        )
        print("Particle builder")

        global_algorithm = (
            GlobalMammogramReasoningAlgorithm(
                min_significant_cluster_score=0.50,
                min_significant_cluster_particles=5,
                device=device,
            )
        )

        global_features = global_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
            clusters=clusters,
        )
        final_algorithm = FinalMCDecisionAlgorithm(
            particle_weight=0.40,
            cluster_weight=0.40,
            global_weight=0.20,

            particle_threshold=0.40,
            cluster_threshold=0.30,
            final_threshold=0.30,

            device=device,
        )

        final_decision = final_algorithm.apply(
            particles=particles,

            physical_features=physical_candidates,

            clusters=clusters,

            global_features=global_features,
        )
        segmentation_algorithm = (
            NativeResolutionSegmentationAlgorithm(
                connectivity=2,
                device=device,
            )
        )

        final_output = (
            segmentation_algorithm.apply(
                candidates=candidates,
                physical_features=physical_candidates,
                clusters=clusters,
                final_decision=final_decision,
            )
        )
        mask_fiinal = final_output.mc_mask.cpu().detach().numpy()
        print(
            "=============================================="
        )

    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_local_physical_FinalMCDecisionAlgorithm(
            self,
            dicom_image,
    ):

        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )


        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================

        I_raw, I_norm = (
            MammographyPreprocessing
            .robust_intensity_normalization(
                image=dicom_image,
                breast_mask=breast_mask,
                device=device,
            )
        )

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================

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

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        log_image = InMemoryImage(
            array=log_features,
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=1.0,
            w_dog=0.75,
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

        breast_mask_tensor = (
            breast_mask.pixel_data
            .to(device=device)
            .bool()
        )

        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        breast_evidence = evidence_data[
            breast_mask_tensor
        ]


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

        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="otsu",
            percentile=0.995,
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

        candidate_mask = (
            candidates.pixel_data.bool()
        )



        num_candidates = (
            candidate_mask.sum().item()
        )

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )

        breast_candidate_count = (
                candidate_mask & breast_mask_tensor
        ).sum().item()

        breast_pixel_count = (
            breast_mask_tensor.sum().item()
        )

        breast_candidate_ratio = (
                breast_candidate_count
                / breast_pixel_count
        )

        print(
            f"Total candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )

        print(
            f"Breast candidate pixels: "
            f"{breast_candidate_count}"
        )

        print(
            f"Candidate ratio in breast: "
            f"{breast_candidate_ratio:.6%}"
        )

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[
            candidate_mask
        ]

        non_candidate_evidence = evidence_data[
            ~candidate_mask & breast_mask_tensor
            ]


        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )



        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidate_mask
            .detach()
            .cpu()
            .numpy()
        )

        evidence_np = (
            evidence_data
            .detach()
            .cpu()
            .numpy()
        )

        image_vis = image_np.astype(
            np.float32
        )

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1)
            / (p99 - p1 + 1e-8),
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
        print("physical_algorithm")
        physical_algorithm = LocalPhysicalAnalysisAlgorithm(
            window_size=31,
            radial_radius=15,
            shape_threshold_k=1.5,
            min_area=1,
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


        print(
            "Physical candidates:",
            len(physical_candidates),
        )
        # ==============================================================
        # 14. Candidate → Particle conversion
        # ==============================================================

        particle_builder = ParticleBuilder(
            device=device,
            scale_neighborhood_radius=1,
        )

        particles = particle_builder.build(
            physical_candidates=physical_candidates,

            top_hat_maps=top_hat_out.pixel_data,

            log_maps=log_features,

            hf_maps=frequency_out.pixel_data,
        )

        print(
            "Particles:",
            len(particles),
        )


        spatial_algorithm = SpatialReasoningAlgorithm(
            neighborhood_radius=20.0,
            min_cluster_neighbors=1,
            device=device,
        )

        spatial_features = spatial_algorithm.apply(
            physical_candidates
        )
        cluster_algorithm = ClusterReasoningAlgorithm(
            min_particles=3,
            density_radius=20.0,
            min_density=0.001,
            particle_score_threshold=2.0,
            device=device,
        )

        clusters = cluster_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
        )
        print("Particle builder")

        global_algorithm = (
            GlobalMammogramReasoningAlgorithm(
                min_significant_cluster_score=0.50,
                min_significant_cluster_particles=3,
                device=device,
            )
        )

        global_features = global_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
            clusters=clusters,
        )
        final_algorithm = FinalMCDecisionAlgorithm(
            particle_weight=0.40,
            cluster_weight=0.40,
            global_weight=0.20,

            particle_threshold=0.50,
            cluster_threshold=0.50,
            final_threshold=0.50,

            device=device,
        )

        final_decision = final_algorithm.apply(
            particles=particles,

            physical_features=physical_candidates,

            clusters=clusters,

            global_features=global_features,
        )
        print(
            "=============================================="
        )

    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_local_physical_ClusterReasoningAlgorithm(
            self,
            dicom_image,
    ):

        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )


        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================

        I_raw, I_norm = (
            MammographyPreprocessing
            .robust_intensity_normalization(
                image=dicom_image,
                breast_mask=breast_mask,
                device=device,
            )
        )

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================

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

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        log_image = InMemoryImage(
            array=log_features,
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=1.0,
            w_dog=0.75,
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

        breast_mask_tensor = (
            breast_mask.pixel_data
            .to(device=device)
            .bool()
        )

        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        breast_evidence = evidence_data[
            breast_mask_tensor
        ]


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

        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="otsu",
            percentile=0.995,
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

        candidate_mask = (
            candidates.pixel_data.bool()
        )



        num_candidates = (
            candidate_mask.sum().item()
        )

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )

        breast_candidate_count = (
                candidate_mask & breast_mask_tensor
        ).sum().item()

        breast_pixel_count = (
            breast_mask_tensor.sum().item()
        )

        breast_candidate_ratio = (
                breast_candidate_count
                / breast_pixel_count
        )

        print(
            f"Total candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )

        print(
            f"Breast candidate pixels: "
            f"{breast_candidate_count}"
        )

        print(
            f"Candidate ratio in breast: "
            f"{breast_candidate_ratio:.6%}"
        )

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[
            candidate_mask
        ]

        non_candidate_evidence = evidence_data[
            ~candidate_mask & breast_mask_tensor
            ]


        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )



        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidate_mask
            .detach()
            .cpu()
            .numpy()
        )

        evidence_np = (
            evidence_data
            .detach()
            .cpu()
            .numpy()
        )

        image_vis = image_np.astype(
            np.float32
        )

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1)
            / (p99 - p1 + 1e-8),
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
        print("physical_algorithm")
        physical_algorithm = LocalPhysicalAnalysisAlgorithm(
            window_size=31,
            radial_radius=15,
            shape_threshold_k=1.5,
            min_area=1,
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


        print(
            "Physical candidates:",
            len(physical_candidates),
        )
        # ==============================================================
        # 14. Candidate → Particle conversion
        # ==============================================================

        particle_builder = ParticleBuilder(
            device=device,
            scale_neighborhood_radius=1,
        )

        particles = particle_builder.build(
            physical_candidates=physical_candidates,

            top_hat_maps=top_hat_out.pixel_data,

            log_maps=log_features,

            hf_maps=frequency_out.pixel_data,
        )

        print(
            "Particles:",
            len(particles),
        )


        spatial_algorithm = SpatialReasoningAlgorithm(
            neighborhood_radius=20.0,
            min_cluster_neighbors=1,
            device=device,
        )

        spatial_features = spatial_algorithm.apply(
            physical_candidates
        )
        cluster_algorithm = ClusterReasoningAlgorithm(
            min_particles=3,
            density_radius=20.0,
            min_density=0.001,
            particle_score_threshold=2.0,
            device=device,
        )

        clusters = cluster_algorithm.apply(
            physical_features=physical_candidates,
            spatial_features=spatial_features,
        )
        print("Particle builder")


    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_local_physical_SpatialReasoningAlgorithm(
            self,
            dicom_image,
    ):

        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )


        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================

        I_raw, I_norm = (
            MammographyPreprocessing
            .robust_intensity_normalization(
                image=dicom_image,
                breast_mask=breast_mask,
                device=device,
            )
        )

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================

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

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        log_image = InMemoryImage(
            array=log_features,
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=1.0,
            w_dog=0.75,
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

        breast_mask_tensor = (
            breast_mask.pixel_data
            .to(device=device)
            .bool()
        )

        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        breast_evidence = evidence_data[
            breast_mask_tensor
        ]


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

        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="otsu",
            percentile=0.995,
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

        candidate_mask = (
            candidates.pixel_data.bool()
        )



        num_candidates = (
            candidate_mask.sum().item()
        )

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )

        breast_candidate_count = (
                candidate_mask & breast_mask_tensor
        ).sum().item()

        breast_pixel_count = (
            breast_mask_tensor.sum().item()
        )

        breast_candidate_ratio = (
                breast_candidate_count
                / breast_pixel_count
        )

        print(
            f"Total candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )

        print(
            f"Breast candidate pixels: "
            f"{breast_candidate_count}"
        )

        print(
            f"Candidate ratio in breast: "
            f"{breast_candidate_ratio:.6%}"
        )

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[
            candidate_mask
        ]

        non_candidate_evidence = evidence_data[
            ~candidate_mask & breast_mask_tensor
            ]


        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )



        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidate_mask
            .detach()
            .cpu()
            .numpy()
        )

        evidence_np = (
            evidence_data
            .detach()
            .cpu()
            .numpy()
        )

        image_vis = image_np.astype(
            np.float32
        )

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1)
            / (p99 - p1 + 1e-8),
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
        print("physical_algorithm")
        physical_algorithm = LocalPhysicalAnalysisAlgorithm(
            window_size=31,
            radial_radius=15,
            shape_threshold_k=1.5,
            min_area=1,
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


        print(
            "Physical candidates:",
            len(physical_candidates),
        )
        # ==============================================================
        # 14. Candidate → Particle conversion
        # ==============================================================

        particle_builder = ParticleBuilder(
            device=device,
            scale_neighborhood_radius=1,
        )

        particles = particle_builder.build(
            physical_candidates=physical_candidates,

            top_hat_maps=top_hat_out.pixel_data,

            log_maps=log_features,

            hf_maps=frequency_out.pixel_data,
        )

        print(
            "Particles:",
            len(particles),
        )


        spatial_algorithm = SpatialReasoningAlgorithm(
            neighborhood_radius=20.0,
            min_cluster_neighbors=1,
            device=device,
        )

        spatial_features = spatial_algorithm.apply(
            physical_candidates
        )
        print("Particle builder")

    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_local_physical_particle(
            self,
            dicom_image,
    ):

        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )


        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================

        I_raw, I_norm = (
            MammographyPreprocessing
            .robust_intensity_normalization(
                image=dicom_image,
                breast_mask=breast_mask,
                device=device,
            )
        )

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================

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

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        log_image = InMemoryImage(
            array=log_features,
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=1.0,
            w_dog=0.75,
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

        breast_mask_tensor = (
            breast_mask.pixel_data
            .to(device=device)
            .bool()
        )

        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        breast_evidence = evidence_data[
            breast_mask_tensor
        ]


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

        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="otsu",
            percentile=0.995,
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

        candidate_mask = (
            candidates.pixel_data.bool()
        )



        num_candidates = (
            candidate_mask.sum().item()
        )

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )

        breast_candidate_count = (
                candidate_mask & breast_mask_tensor
        ).sum().item()

        breast_pixel_count = (
            breast_mask_tensor.sum().item()
        )

        breast_candidate_ratio = (
                breast_candidate_count
                / breast_pixel_count
        )

        print(
            f"Total candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )

        print(
            f"Breast candidate pixels: "
            f"{breast_candidate_count}"
        )

        print(
            f"Candidate ratio in breast: "
            f"{breast_candidate_ratio:.6%}"
        )

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[
            candidate_mask
        ]

        non_candidate_evidence = evidence_data[
            ~candidate_mask & breast_mask_tensor
            ]


        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )



        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidate_mask
            .detach()
            .cpu()
            .numpy()
        )

        evidence_np = (
            evidence_data
            .detach()
            .cpu()
            .numpy()
        )

        image_vis = image_np.astype(
            np.float32
        )

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1)
            / (p99 - p1 + 1e-8),
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
        print("physical_algorithm")
        physical_algorithm = LocalPhysicalAnalysisAlgorithm(
            window_size=31,
            radial_radius=15,
            shape_threshold_k=1.5,
            min_area=1,
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


        print(
            "Physical candidates:",
            len(physical_candidates),
        )
        # ==============================================================
        # 14. Candidate → Particle conversion
        # ==============================================================

        particle_builder = ParticleBuilder(
            device=device,
            scale_neighborhood_radius=1,
        )

        particles = particle_builder.build(
            physical_candidates=physical_candidates,

            top_hat_maps=top_hat_out.pixel_data,

            log_maps=log_features,

            hf_maps=frequency_out.pixel_data,
        )

        print(
            "Particles:",
            len(particles),
        )

        for particle in particles[:10]:
            print(
                f"Particle {particle.id}: "
                f"label={particle.label}, "
                f"position=({particle.x:.1f}, {particle.y:.1f}), "
                f"area={particle.area:.1f}, "
                f"diameter={particle.diameter:.2f}, "
                f"peak={particle.peak_prominence:.2f}, "
                f"radial={particle.radial_score:.3f}, "
                f"scale={particle.scale_score:.3f}"
            )

            print(
                "  TopHat:",
                particle.top_hat_signature,
            )

            print(
                "  LoG:",
                particle.log_signature,
            )

            print(
                "  HF:",
                particle.hf_signature,
            )
        print("Particle builder")

    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_local_physical(
            self,
            dicom_image,
    ):

        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )


        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================

        I_raw, I_norm = (
            MammographyPreprocessing
            .robust_intensity_normalization(
                image=dicom_image,
                breast_mask=breast_mask,
                device=device,
            )
        )

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================

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

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        log_image = InMemoryImage(
            array=log_features,
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=1.0,
            w_dog=0.75,
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

        breast_mask_tensor = (
            breast_mask.pixel_data
            .to(device=device)
            .bool()
        )

        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        breast_evidence = evidence_data[
            breast_mask_tensor
        ]


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

        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="otsu",
            percentile=0.995,
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

        candidate_mask = (
            candidates.pixel_data.bool()
        )



        num_candidates = (
            candidate_mask.sum().item()
        )

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )

        breast_candidate_count = (
                candidate_mask & breast_mask_tensor
        ).sum().item()

        breast_pixel_count = (
            breast_mask_tensor.sum().item()
        )

        breast_candidate_ratio = (
                breast_candidate_count
                / breast_pixel_count
        )

        print(
            f"Total candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )

        print(
            f"Breast candidate pixels: "
            f"{breast_candidate_count}"
        )

        print(
            f"Candidate ratio in breast: "
            f"{breast_candidate_ratio:.6%}"
        )

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[
            candidate_mask
        ]

        non_candidate_evidence = evidence_data[
            ~candidate_mask & breast_mask_tensor
            ]


        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )



        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidate_mask
            .detach()
            .cpu()
            .numpy()
        )

        evidence_np = (
            evidence_data
            .detach()
            .cpu()
            .numpy()
        )

        image_vis = image_np.astype(
            np.float32
        )

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1)
            / (p99 - p1 + 1e-8),
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
        print("physical_algorithm")
        physical_algorithm = LocalPhysicalAnalysisAlgorithm(
            window_size=31,
            radial_radius=15,
            shape_threshold_k=1.5,
            min_area=1,
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


        print(
            "Physical candidates:",
            len(physical_candidates),
        )

    @pytest.mark.parametrize(
        "dicom_image",
        mock_dicom_image(),
    )
    def test_candidate_evidence_generation(
            self,
            dicom_image,
    ):

        device = "cuda"

        # ==============================================================
        # 1. Breast mask
        # ==============================================================

        breast_mask = dicom_image.clone()

        breast_mask_algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device=device,
        )

        breast_mask_algorithm(
            image=dicom_image,
            output=breast_mask,
        )


        # ==============================================================
        # 2. Intensity normalization
        # ==============================================================

        I_raw, I_norm = (
            MammographyPreprocessing
            .robust_intensity_normalization(
                image=dicom_image,
                breast_mask=breast_mask,
                device=device,
            )
        )

        # ==============================================================
        # 3. Multi-scale Top-Hat
        # ==============================================================

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

        # ==============================================================
        # 4. DoG / frequency
        # ==============================================================

        frequency = DoG(
            device=device,
        )

        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )

        print(
            "DoG output shape:",
            frequency_out.pixel_data.shape,
        )

        # ==============================================================
        # 5. Multi-scale LoG
        # ==============================================================

        differential = DifferentialBlobAlgorithm(
            device=device,
        )

        differential_output = dicom_image.clone()

        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        log_image = InMemoryImage(
            array=log_features,
        )

        print(
            "LoG shape:",
            log_features.shape,
        )

        # ==============================================================
        # 6. Candidate Evidence
        # ==============================================================

        evidence_algorithm = CandidateEvidenceAlgorithm(
            w_top_hat=1.0,
            w_log=1.0,
            w_dog=0.75,
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

        breast_mask_tensor = (
            breast_mask.pixel_data
            .to(device=device)
            .bool()
        )

        while breast_mask_tensor.ndim > 2:
            breast_mask_tensor = breast_mask_tensor.squeeze(0)

        breast_evidence = evidence_data[
            breast_mask_tensor
        ]


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

        candidate_algorithm = CandidateGenerationAlgorithm(
            threshold_method="otsu",
            percentile=0.995,
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

        candidate_mask = (
            candidates.pixel_data.bool()
        )



        num_candidates = (
            candidate_mask.sum().item()
        )

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )

        breast_candidate_count = (
                candidate_mask & breast_mask_tensor
        ).sum().item()

        breast_pixel_count = (
            breast_mask_tensor.sum().item()
        )

        breast_candidate_ratio = (
                breast_candidate_count
                / breast_pixel_count
        )

        print(
            f"Total candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )

        print(
            f"Breast candidate pixels: "
            f"{breast_candidate_count}"
        )

        print(
            f"Candidate ratio in breast: "
            f"{breast_candidate_ratio:.6%}"
        )

        # ==============================================================
        # 11. Verify candidates correspond to high evidence
        # ==============================================================

        candidate_evidence = evidence_data[
            candidate_mask
        ]

        non_candidate_evidence = evidence_data[
            ~candidate_mask & breast_mask_tensor
            ]


        print(
            "Mean candidate evidence:",
            candidate_evidence.mean().item(),
        )

        if non_candidate_evidence.numel() > 0:
            print(
                "Mean non-candidate evidence:",
                non_candidate_evidence.mean().item(),
            )



        # ==============================================================
        # 12. Visualization data
        # ==============================================================

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidate_mask
            .detach()
            .cpu()
            .numpy()
        )

        evidence_np = (
            evidence_data
            .detach()
            .cpu()
            .numpy()
        )

        image_vis = image_np.astype(
            np.float32
        )

        p1 = np.percentile(
            image_vis,
            1,
        )

        p99 = np.percentile(
            image_vis,
            99,
        )

        image_vis = np.clip(
            (image_vis - p1)
            / (p99 - p1 + 1e-8),
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
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_candidate_generation(self, dicom_image):

        # --------------------------------------------------------------
        # 1. Generate breast mask
        # --------------------------------------------------------------

        mask_output = dicom_image.clone()
        print("candidate generation completedcandidate generation completedcandidate generation completedcandidate generation completedcandidate generation completedcandidate generation completed")
        algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device="cuda",
        )
        print("init breast mask algorithm")
        algorithm(
            image=dicom_image,
            output=mask_output,
        )
        print("breast mask algorithm completed")
        # --------------------------------------------------------------
        # 2. Robust intensity normalization
        # --------------------------------------------------------------

        I_raw, I_norm = (
            MammographyPreprocessing.robust_intensity_normalization(
                image=dicom_image,
                breast_mask=mask_output,
                device="cuda",
            )
        )
        print("robust intensity normalization completed")
        # --------------------------------------------------------------
        # 3. Generate Top-Hat features
        # --------------------------------------------------------------

        top_hat = MultiScaleTopHatAlgorithm(
            device="cuda",
        )
        print("init top hat algorithm")
        top_hat_output = dicom_image.clone()

        top_hat_out = top_hat.apply(
            I_norm,
            top_hat_output,
        )
        # top_hat_out.pixel_data[0] = torch.zeros_like(top_hat_out.pixel_data[0])
        # top_hat_out.pixel_data[1] = torch.zeros_like(top_hat_out.pixel_data[1])
        # top_hat_out.pixel_data[2] = torch.zeros_like(top_hat_out.pixel_data[2])
        # top_hat_out.pixel_data[3] = torch.zeros_like(top_hat_out.pixel_data[3])
        # top_hat_out.pixel_data[5] = torch.zeros_like(top_hat_out.pixel_data[5])
        # top_hat_out_np = top_hat_out.pixel_data[5].cpu().detach().numpy()
        print("top hat algorithm completed")
        # --------------------------------------------------------------
        # 4. Generate Frequency features
        # --------------------------------------------------------------

        frequency = DoG(
            device="cuda",
        )
        print("init frequency analysis algorithm")
        frequency_output = dicom_image.clone()

        frequency_out = frequency.apply(
            I_norm,
            frequency_output,
        )
        # TODO: for the frequency analysis algorithm keep only DOG
        # frequency_out.pixel_data[1] = torch.zeros_like(frequency_out.pixel_data[1])
        # frequency_out.pixel_data[2] = torch.zeros_like(frequency_out.pixel_data[2])
        # frequency_out.pixel_data[3] = torch.zeros_like(frequency_out.pixel_data[3])

        print("frequency analysis algorithm completed")
        # --------------------------------------------------------------
        # 5. Generate Differential / LoG features
        # --------------------------------------------------------------

        print("init differential blob algorithm")
        differential = DifferentialBlobAlgorithm(
            device="cuda",
        )

        differential_output = dicom_image.clone()
        print("applying differential blob algorithm")
        differential_out = differential.apply(
            I_norm,
            differential_output,
        )

        # differential_out.pixel_data = torch.zeros_like(differential_out.pixel_data)
        print("applying differential blob algorithm completed")

        # --------------------------------------------------------------
        # 6. Extract required feature maps
        # --------------------------------------------------------------

        # Top-Hat:
        #   [num_scales, H, W]
        #
        # Candidate generation handles the scales independently.

        top_hat_features = top_hat_out.pixel_data

        # Frequency output:
        #
        #   0 -> DoG
        #   1 -> LH
        #   2 -> HL
        #   3 -> HH
        #   4 -> EHF

        dog = frequency_out.pixel_data[0]

        # Differential output:
        #
        #   0 -> LoG sigma 1
        #   1 -> LoG sigma 2
        #   2 -> LoG sigma 3
        #   ...

        log_features = differential_out.pixel_data[
            :len(differential.sigmas)
        ]

        # --------------------------------------------------------------
        # 7. Candidate generation
        # --------------------------------------------------------------
        print("init candidate generation algorithm")
        candidate_algorithm = CandidateGenerationAlgorithm(
            # threshold_method="otsu",
            device="cuda",
        )

        candidate_output = dicom_image.clone()

        candidates = candidate_algorithm.apply(
            top_hat=top_hat_out,
            log=InMemoryImage(
                array=log_features,
            ),
            dog=InMemoryImage(
                array=dog,
            ),
            # high_frequency=InMemoryImage(
            #     array=high_frequency,
            # ),
            breast_mask=mask_output,
            output=candidate_output,
        )
        print("candidate generation algorithm completed")
        # --------------------------------------------------------------
        # 8. Basic output validation
        # --------------------------------------------------------------

        # assert isinstance(
        #     candidates.pixel_data,
        #     torch.Tensor,
        # )
        #
        # assert candidates.pixel_data.shape == (
        #     dicom_image.height,
        #     dicom_image.width,
        # )
        #
        # # --------------------------------------------------------------
        # # 9. Candidate map must be binary
        # # --------------------------------------------------------------
        #
        # assert torch.all(
        #     (candidates.pixel_data == 0)
        #     | (candidates.pixel_data == 1)
        # )
        #
        # # --------------------------------------------------------------
        # # 10. Candidate map should be integer/binary
        # # --------------------------------------------------------------
        #
        # assert candidates.pixel_data.dtype in (
        #     torch.uint8,
        #     torch.bool,
        # )
        #
        # # --------------------------------------------------------------
        # # 11. Candidate map must remain full resolution
        # # --------------------------------------------------------------
        #
        # assert candidates.pixel_data.shape[-2:] == (
        #     dicom_image.height,
        #     dicom_image.width,
        # )

        # --------------------------------------------------------------
        # 12. Candidate map must not contain NaN / Inf
        # --------------------------------------------------------------

        candidate_float = (
            candidates.pixel_data.float()
        )

        # assert torch.isfinite(
        #     candidate_float
        # ).all()

        # --------------------------------------------------------------
        # 13. Candidate generation should produce candidates
        # --------------------------------------------------------------

        num_candidates = (
            candidates.pixel_data
            .sum()
            .item()
        )

        # assert num_candidates > 0

        # --------------------------------------------------------------
        # 14. Candidate ratio
        # --------------------------------------------------------------

        total_pixels = (
                dicom_image.height
                * dicom_image.width
        )

        candidate_ratio = (
                num_candidates
                / total_pixels
        )



        print(
            f"Number of candidates: "
            f"{num_candidates}"
        )

        print(
            f"Candidate ratio: "
            f"{candidate_ratio:.6%}"
        )
        # --------------------------------------------------------------
        # 15. Visualize candidate generation
        # --------------------------------------------------------------

        image_np = (
            I_raw.pixel_data
            .detach()
            .cpu()
            .numpy()
        )

        candidate_np = (
            candidates.pixel_data
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

        # Normalize only for visualization.
        # This does NOT modify I_raw or the algorithm input.
        image_vis = image_np.astype(np.float32)

        p1 = np.percentile(image_vis, 1)
        p99 = np.percentile(image_vis, 99)

        image_vis = np.clip(
            (image_vis - p1) / (p99 - p1 + 1e-8),
            0.0,
            1.0,
        )


        # Only draw candidate pixels
        candidate_overlay = np.ma.masked_where(
            ~candidate_np,
            candidate_np,
        )
        x = top_hat_out.pixel_data[5].float()

        # For now, use the breast mask
        breast_mask = mask_output.pixel_data.bool()

        # Only values inside breast
        breast_values = x[breast_mask]

        threshold = torch.quantile(
            breast_values,
            0.995,
        )

        candidate = (
                (x > threshold)
                & breast_mask
        )

        print("Top-Hat min:", x.min().item())
        print("Top-Hat max:", x.max().item())
        print("Top-Hat threshold:", threshold.item())

        print(
            "Candidate pixels:",
            candidate.sum().item(),
        )

        print(
            "Candidate ratio in breast:",
            candidate.sum().item()
            / breast_mask.sum().item(),
        )
        print(
            f"Candidate output shape: "
            f"{candidates.pixel_data.shape}"
        )
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_multi_scale_tophat(self, dicom_image):

        # --------------------------------------------------------------
        # 1. Generate breast mask
        # --------------------------------------------------------------

        mask_output = dicom_image.clone()

        algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device="cuda",
        )

        algorithm(
            image=dicom_image,
            output=mask_output,
        )


        # --------------------------------------------------------------
        # 2. Robust intensity normalization
        # --------------------------------------------------------------

        I_raw, I_norm = MammographyPreprocessing.robust_intensity_normalization(
            image=dicom_image,
            breast_mask=mask_output,
            device="cuda",
        )
        top_hat = MultiScaleTopHatAlgorithm()
        t_output = dicom_image.clone()
        top_hat_out = top_hat.apply(I_norm, t_output)

        top_hat_out0 = top_hat_out.pixel_data[0].detach().cpu().numpy()
        top_hat_out1 = top_hat_out.pixel_data[1].detach().cpu().numpy()
        top_hat_out2 = top_hat_out.pixel_data[2].detach().cpu().numpy()
        top_hat_out4 = top_hat_out.pixel_data[4].detach().cpu().numpy()
        top_hat_out5 = top_hat_out.pixel_data[5].detach().cpu().numpy()
        top_hat_out3 = top_hat_out.pixel_data[3].detach().cpu().numpy()
        print(f"Top-Hat output shape: {top_hat_out2.shape}")
    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_gabor_orientation(self, dicom_image):

        # --------------------------------------------------------------
        # 1. Generate breast mask
        # --------------------------------------------------------------

        mask_output = dicom_image.clone()

        algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device="cuda",
        )

        algorithm(
            image=dicom_image,
            output=mask_output,
        )


        # --------------------------------------------------------------
        # 2. Robust intensity normalization
        # --------------------------------------------------------------

        I_raw, I_norm = MammographyPreprocessing.robust_intensity_normalization(
            image=dicom_image,
            breast_mask=mask_output,
            device="cuda",
        )
        differential = GaborOrientationAlgorithm()
        t_output = dicom_image.clone()
        diff = differential.apply(I_norm, t_output)

        diff0 = diff.pixel_data[0].detach().cpu().numpy()
        diff1 = diff.pixel_data[1].detach().cpu().numpy()
        diff2 = diff.pixel_data[2].detach().cpu().numpy()
        print(f"Gabor Orientation output shape: {diff2.shape}")

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_differential_blob(self, dicom_image):

        # --------------------------------------------------------------
        # 1. Generate breast mask
        # --------------------------------------------------------------

        mask_output = dicom_image.clone()

        algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device="cuda",
        )

        algorithm(
            image=dicom_image,
            output=mask_output,
        )


        # --------------------------------------------------------------
        # 2. Robust intensity normalization
        # --------------------------------------------------------------

        I_raw, I_norm = MammographyPreprocessing.robust_intensity_normalization(
            image=dicom_image,
            breast_mask=mask_output,
            device="cuda",
        )
        differential = DifferentialBlobAlgorithm()
        t_output = dicom_image.clone()
        diff = differential.apply(I_norm, t_output)

        diff0 = diff.pixel_data[0].detach().cpu().numpy()
        diff1 = diff.pixel_data[1].detach().cpu().numpy()
        diff2 = diff.pixel_data[2].detach().cpu().numpy()
        diff3 = diff.pixel_data[3].detach().cpu().numpy()
        print(f"Differential output shape: {diff2.shape}")

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_frequency_analysis(self, dicom_image):

        # --------------------------------------------------------------
        # 1. Generate breast mask
        # --------------------------------------------------------------

        mask_output = dicom_image.clone()

        algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device="cuda",
        )

        algorithm(
            image=dicom_image,
            output=mask_output,
        )


        # --------------------------------------------------------------
        # 2. Robust intensity normalization
        # --------------------------------------------------------------

        I_raw, I_norm = MammographyPreprocessing.robust_intensity_normalization(
            image=dicom_image,
            breast_mask=mask_output,
            device="cuda",
        )
        frequency = DoG()
        t_output = dicom_image.clone()
        frequency_out = frequency.apply(I_norm, t_output)

        frequency_out0 = frequency_out.pixel_data[0].detach().cpu().numpy()
        frequency_out1 = frequency_out.pixel_data[1].detach().cpu().numpy()
        frequency_out2 = frequency_out.pixel_data[2].detach().cpu().numpy()
        frequency_out3 = frequency_out.pixel_data[3].detach().cpu().numpy()
        frequency_out4 = frequency_out.pixel_data[4].detach().cpu().numpy()
        print(f"Frequency output shape: {frequency_out2.shape}")

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_robust_intensity_normalization(self, dicom_image):

        # --------------------------------------------------------------
        # 1. Generate breast mask
        # --------------------------------------------------------------

        mask_output = dicom_image.clone()

        algorithm = BreastMaskAlgorithm(
            mask_only=True,
            device="cuda",
        )

        algorithm(
            image=dicom_image,
            output=mask_output,
        )

        # mask_output.pixel_data contains 0/1 breast mask
        assert isinstance(mask_output.pixel_data, torch.Tensor)

        assert mask_output.pixel_data.shape == dicom_image.pixel_data.shape

        assert torch.all(
            (mask_output.pixel_data == 0)
            | (mask_output.pixel_data == 1)
        )

        # --------------------------------------------------------------
        # 2. Robust intensity normalization
        # --------------------------------------------------------------

        I_raw, I_norm = MammographyPreprocessing.robust_intensity_normalization(
            image=dicom_image,
            breast_mask=mask_output,
            device="cuda",
        )

        # --------------------------------------------------------------
        # 3. Basic output validation
        # --------------------------------------------------------------


        assert isinstance(I_raw.pixel_data, torch.Tensor)
        assert isinstance(I_norm.pixel_data, torch.Tensor)

        assert I_raw.pixel_data.shape == dicom_image.pixel_data.shape
        assert I_norm.pixel_data.shape == dicom_image.pixel_data.shape

        # --------------------------------------------------------------
        # 4. Normalized image must be in [0, 1]
        # --------------------------------------------------------------

        assert torch.all(I_norm.pixel_data >= 0.0)
        assert torch.all(I_norm.pixel_data <= 1.0)

        # --------------------------------------------------------------
        # 5. Background must be zero in I_norm
        # --------------------------------------------------------------

        background = mask_output.pixel_data == 0

        assert torch.all(
            I_norm.pixel_data[background] == 0
        )

        # --------------------------------------------------------------
        # 6. Raw image must remain unchanged
        # --------------------------------------------------------------

        assert torch.equal(
            I_raw.pixel_data.cpu(),
            dicom_image.pixel_data.cpu(),
        )

        # --------------------------------------------------------------
        # 7. Normalization parameters
        # --------------------------------------------------------------

        assert hasattr(I_norm, "normalization_p_low")
        assert hasattr(I_norm, "normalization_p_high")

        assert I_norm.normalization_p_low < I_norm.normalization_p_high
        I_norm_2 = I_norm.pixel_data.detach().cpu().numpy()
        I_raw_2 = I_raw.pixel_data.detach().cpu().numpy()

        print(
            f"P1  = {I_norm.normalization_p_low:.4f}"
        )

        print(
            f"P99 = {I_norm.normalization_p_high:.4f}"
        )

    @pytest.mark.parametrize("kernel_size", mock_kernel_sizes())
    def test_morphology_closing_matches_ndimage(self, kernel_size):
        image = (np.random.rand(16, 16) > 0.5).astype(np.int64)

        image_object = DicomImage.from_array(image)
        output_object = image_object.clone()

        MorphologyOperations.morphology_closing(
            image_object, output_object, kernel_size=kernel_size[0], device="cpu"
        )

        output_object.pixel_data = output_object.pixel_data.to(torch.int64)
        ndimage_result = ndimage.binary_closing(
            image, structure=np.ones((kernel_size[0], kernel_size[0]))
        ).astype(np.int64)

        np.testing.assert_array_equal(output_object.pixel_data.numpy(), ndimage_result)

    @pytest.mark.parametrize("sigma1, sigma2", mock_two_sigmas())
    def test_DoG_matches_skimage(self, sigma1, sigma2):
        image = np.random.rand(8, 8).astype(np.float32)
        truncate = 4.0
        image_object = DicomImage.from_array(image)
        output = image_object.clone()

        Filters.difference_of_gaussian(
            image_object, output, sigma1, sigma2, truncate=truncate
        )

        skimage_result = difference_of_gaussians(image, sigma1, sigma2)

        np.testing.assert_allclose(
            output.pixel_data, skimage_result, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_custom_algorithm(self, dicom_image):
        if not isinstance(dicom_image.pixel_data, torch.Tensor):
            dicom_image.pixel_data = torch.from_numpy(dicom_image.pixel_data).float()

        output = dicom_image.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        algorithm = CustomAlgorithm()
        algorithm(image=dicom_image, output=output)

        assert not torch.allclose(
            dicom_image.pixel_data.float(), output.pixel_data.detach().cpu().float()
        )

        unique_vals = torch.unique(output.pixel_data)
        assert set(unique_vals.tolist()).issubset({0, 1})

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_patches(self, dicom_image):
        patchs = PatchGrid(dicom_image, (15, 15))

    @pytest.mark.parametrize("png_image", mock_png_image())
    def test_patches_png(self, png_image):
        logger.info(f"PNG image shape: {png_image.pixel_data.shape}")

        patch_grid = PatchGrid(png_image, (100, 100))

        total_patches = len(patch_grid.patches)
        num_rows = len(patch_grid.grid)
        num_cols = len(patch_grid.grid[0]) if num_rows > 0 else 0

        logger.info(f"Total patches: {total_patches}")
        logger.info(f"Grid rows: {num_rows}, Grid cols: {num_cols}")

        assert total_patches == num_rows * num_cols
        assert num_rows > 0 and num_cols > 0

        patch1 = patch_grid.patches[0].load()
        patch2 = patch_grid.patches[1].load()

        assert patch1.pixel_data.shape[0] <= 100
        assert patch1.pixel_data.shape[1] <= 100

    @pytest.mark.parametrize("dicom_image", mock_dicom_image())
    def test_sbrg(self, dicom_image):
        # --- Reference implementation (pure numpy/skimage/scipy, no framework) ---
        coordinates = [1958, 1177, 2165, 1310]
        region_of_interest = RegionOfInterest(dicom_image, coordinates).load()
        roi_np = region_of_interest.pixel_data.detach().numpy()
        image_np = (
            region_of_interest.pixel_data.detach().numpy()
            if isinstance(region_of_interest.pixel_data, torch.Tensor)
            else region_of_interest.pixel_data
        ).reshape(region_of_interest.height, region_of_interest.width)

        # Stage 1: Seed-based region growing reference
        from skimage.morphology import local_maxima

        regional_max = local_maxima(image_np)
        seed_coords = np.argwhere(regional_max)
        seed_values = image_np[seed_coords[:, 0], seed_coords[:, 1]]
        seed_threshold = np.mean(seed_values)
        ref_region = (image_np >= seed_threshold).astype(np.float32)

        # Stage 2: Boundary segmentation reference
        from skimage.filters import sobel

        gradient = sobel(ref_region)
        binary_mask = (gradient > 0).astype(np.float32)
        I = morphoogy_closing(binary_mask)
        fill = region_fill(I)

        # --- Framework call ---
        if not isinstance(region_of_interest.pixel_data, torch.Tensor):
            region_of_interest.pixel_data = torch.from_numpy(
                region_of_interest.pixel_data
            ).float()

        output = region_of_interest.clone()
        if not isinstance(output.pixel_data, torch.Tensor):
            output.pixel_data = torch.from_numpy(output.pixel_data).float()

        algorithm = SbrgAlgorithm()
        algorithm(image=region_of_interest, output=output)

        image_output = (
            output.pixel_data.detach()
            .cpu()
            .numpy()
            .reshape((region_of_interest.height, region_of_interest.width))
        )

        # Output must be a valid binary segmentation mask
        assert image_output is not None
        assert image_output.shape == (
            region_of_interest.height,
            region_of_interest.width,
        )
        unique_vals = np.unique(image_output)
        assert len(unique_vals) <= 2, "Output should be binary (0 and 1 only)"

        # Framework output must differ from the trivial all-zeros or all-ones result
        assert not np.all(
            image_output == 0
        ), "Output is all zeros — algorithm produced no segmentation"
        assert not np.all(
            image_output == 1
        ), "Output is all ones — algorithm over-segmented"

        # Framework result should not be identical to the raw reference
        assert not np.allclose(
            image_output,
            fill.reshape(region_of_interest.height, region_of_interest.width),
        ), "Framework output is identical to naive reference — boundary post-processing had no effect"
