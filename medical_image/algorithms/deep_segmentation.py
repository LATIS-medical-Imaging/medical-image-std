"""
Generic deep-learning segmentation algorithm.

Wraps any trained segmentation model as an :class:`Algorithm` subclass.
Output follows Option 2: binary mask on ``output.pixel_data`` + per-lesion
:class:`Annotation` objects on ``output.annotations``.
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.annotation import Annotation, GeometryType
from medical_image.data.image import Image
from medical_image.utils.device import Precision


class DeepSegmentationAlgorithm(Algorithm):
    """Run a trained segmentation model as a framework Algorithm.

    After :meth:`apply`, the following attributes are populated:

    * ``probability_map`` — ``(H, W)`` float tensor in [0, 1].
    * ``lesion_count`` — number of detected lesions after filtering.

    The ``output`` image receives:

    * ``pixel_data`` — ``(H, W)`` binary mask (0.0 / 1.0).
    * ``annotations`` — one :class:`Annotation` per detected lesion
      (``POLYGON`` contour + metadata with confidence, area, bbox).

    Construction
    ------------
    Pass **either** ``checkpoint_path`` to load from a saved checkpoint
    (requires ``segmentation_models_pytorch``), **or** ``model`` to supply
    any ``nn.Module`` directly.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint (must contain
            ``model_state_dict`` and optionally ``config``).
        model: A pre-built ``nn.Module`` (mutually exclusive with
            *checkpoint_path*).
        use_clahe: Whether to apply CLAHE before inference.  When loading
            from checkpoint this is read from ``config.preprocessing.clahe``.
        patch_size: Sliding-window patch size for inference.
        stride: Stride between patches (default: ``patch_size // 2``).
        threshold: Probability threshold for binarisation.
        min_lesion_area: Minimum connected-component area (pixels) to keep.
        device: ``"cuda"`` or ``"cpu"`` (auto-detected if ``None``).
        precision: Mixed-precision mode.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        use_clahe: bool = False,
        patch_size: int = 512,
        stride: Optional[int] = None,
        threshold: float = 0.5,
        min_lesion_area: int = 4,
        device: str = None,
        precision: Precision = Precision.FULL,
    ):
        super().__init__(device=device, precision=precision)

        if checkpoint_path is None and model is None:
            raise ValueError("Provide either checkpoint_path or model")

        self.patch_size = patch_size
        self.stride = stride or patch_size // 2
        self.threshold = threshold
        self.min_lesion_area = min_lesion_area
        self.use_clahe = use_clahe

        if checkpoint_path is not None:
            self._load_from_checkpoint(checkpoint_path)
        else:
            self.model = model.to(self.device)
            self.model.eval()

        # Populated after apply()
        self.probability_map: Optional[torch.Tensor] = None
        self.lesion_count: int = 0

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load model architecture + weights from a training checkpoint.

        Reads ``config`` from the checkpoint and auto-configures:
        ``patch_size``, ``use_clahe``, ``threshold``, ``min_lesion_area``,
        and ``stride`` when they were not explicitly overridden by the caller.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        config = checkpoint.get("config", {})

        # Auto-configure from checkpoint when constructor used defaults
        self.use_clahe = config.get("preprocessing", {}).get("clahe", False)

        patching_cfg = config.get("patching", {})
        if patching_cfg.get("patch_size"):
            self.patch_size = patching_cfg["patch_size"]

        inference_cfg = config.get("inference", {})
        if inference_cfg.get("threshold"):
            self.threshold = inference_cfg["threshold"]
        if inference_cfg.get("min_lesion_area"):
            self.min_lesion_area = inference_cfg["min_lesion_area"]

        stride_ratio = inference_cfg.get("stride_ratio", 0.5)
        self.stride = int(self.patch_size * stride_ratio)

        # Defer smp import to here — only needed for checkpoint loading
        import segmentation_models_pytorch as smp

        model_cfg = config.get("model", {})
        name = model_cfg.get("name", "unet")
        encoder = model_cfg.get("encoder", "resnet34")
        in_channels = model_cfg.get("in_channels", 1)

        common = dict(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

        if name == "unet":
            self.model = smp.Unet(**common)
        elif name == "attention_unet":
            self.model = smp.Unet(**common, decoder_attention_type="scse")
        elif name == "unetpp":
            self.model = smp.UnetPlusPlus(**common)
        elif name == "deeplabv3p":
            self.model = smp.DeepLabV3Plus(**common)
        else:
            raise ValueError(f"Unknown model: {name}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Algorithm interface
    # ------------------------------------------------------------------

    def apply(self, image: Image, output: Image) -> Image:
        img = image.pixel_data.float()
        if img.ndim == 3:
            img = img.squeeze(0)

        # Normalize to [0, 1]
        img_max = img.max()
        if img_max > 1.0:
            img = img / img_max

        # Optional CLAHE
        if self.use_clahe:
            img = self._apply_clahe(img)

        # Patch-based inference
        prob_map = self._infer_patches(img)

        # Store probability map
        self.probability_map = prob_map

        # Binarize
        binary_mask = (prob_map > self.threshold).float()

        # Extract per-lesion annotations (filter by min_lesion_area)
        annotations = self._extract_annotations(
            binary_mask, prob_map, self.min_lesion_area
        )
        self.lesion_count = len(annotations)

        # Set output
        output.pixel_data = binary_mask
        output.annotations = []
        for ann in annotations:
            output.add_annotation(ann)

        return output

    # ------------------------------------------------------------------
    # Patch inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Sliding-window inference with overlap averaging."""
        h, w = image.shape
        ps = self.patch_size
        stride = self.stride

        prediction_sum = torch.zeros(h, w)
        count = torch.zeros(h, w)

        patches = []
        positions = []

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + ps, h)
                x_end = min(x + ps, w)
                y_start = max(0, y_end - ps)
                x_start = max(0, x_end - ps)

                patch = image[y_start : y_start + ps, x_start : x_start + ps]

                if patch.shape[0] < ps or patch.shape[1] < ps:
                    padded = torch.zeros(ps, ps)
                    padded[: patch.shape[0], : patch.shape[1]] = patch
                    patch = padded

                patches.append(patch)
                positions.append((y_start, x_start))

        # Batch inference
        batch_size = 8
        pred_patches = []
        for i in range(0, len(patches), batch_size):
            batch = (
                torch.stack(patches[i : i + batch_size])
                .unsqueeze(1)
                .to(self.device)
            )
            logits = self.model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu()
            pred_patches.extend([p for p in probs])

        # Stitch
        for patch_pred, (y, x) in zip(pred_patches, positions):
            ph, pw = patch_pred.shape
            ey = min(y + ph, h)
            ex = min(x + pw, w)
            prediction_sum[y:ey, x:ex] += patch_pred[: ey - y, : ex - x]
            count[y:ey, x:ex] += 1

        return prediction_sum / count.clamp(min=1)

    # ------------------------------------------------------------------
    # Annotation extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_annotations(
        binary_mask: torch.Tensor,
        prob_map: torch.Tensor,
        min_lesion_area: int = 4,
    ) -> list[Annotation]:
        """Extract per-lesion Annotations from binary mask via connected components."""
        mask_np = binary_mask.numpy().astype(np.uint8)
        prob_np = prob_map.numpy()

        labeled, num_components = ndimage.label(mask_np)
        annotations = []

        for i in range(1, num_components + 1):
            component = (labeled == i).astype(np.uint8)
            area = int(component.sum())

            if area < min_lesion_area:
                continue

            # Find contour for polygon
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            contour_pts = [(int(p[0][0]), int(p[0][1])) for p in contour]

            # Need >= 3 points for a polygon
            if len(contour_pts) < 3:
                ys, xs = np.where(component)
                x_min, y_min = int(xs.min()), int(ys.min())
                x_max, y_max = int(xs.max()), int(ys.max())
                ann = Annotation(
                    shape=GeometryType.RECTANGLE,
                    coordinates=[x_min, y_min, x_max, y_max],
                    label="microcalcification",
                    metadata={
                        "confidence": float(prob_np[component == 1].mean()),
                        "area": area,
                    },
                )
            else:
                confidence = float(prob_np[component == 1].mean())
                ys, xs = np.where(component)
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

                ann = Annotation(
                    shape=GeometryType.POLYGON,
                    coordinates=contour_pts,
                    label="microcalcification",
                    metadata={
                        "confidence": confidence,
                        "area": area,
                        "bbox": bbox,
                    },
                )

            annotations.append(ann)

        return annotations

    # ------------------------------------------------------------------
    # CLAHE
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_clahe(
        image: torch.Tensor,
        clip_limit: float = 2.0,
        grid_size: int = 8,
    ) -> torch.Tensor:
        """Apply CLAHE preprocessing. Returns float tensor in [0, 1]."""
        img_np = image.numpy()
        img_min, img_max = img_np.min(), img_np.max()
        if img_max - img_min > 0:
            img_u8 = ((img_np - img_min) / (img_max - img_min) * 255).astype(
                np.uint8
            )
        else:
            img_u8 = np.zeros_like(img_np, dtype=np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
        )
        enhanced = clahe.apply(img_u8)
        return torch.from_numpy(enhanced.astype(np.float32) / 255.0)