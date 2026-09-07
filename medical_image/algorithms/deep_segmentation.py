"""
Generic deep-learning segmentation algorithm.

Wraps any trained segmentation model as an :class:`Algorithm` subclass.
Output follows Option 2: binary mask on ``output.pixel_data`` + per-lesion
:class:`Annotation` objects on ``output.annotations``.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.exposure import equalize_adapthist
from skimage.measure import find_contours

from medical_image.algorithms.algorithm import Algorithm
from medical_image.data.annotation import Annotation, GeometryType
from medical_image.data.image import Image
from medical_image.data.in_memory_image import InMemoryImage
from medical_image.process.mammography import MammographyPreprocessing
from medical_image.utils.device import Precision

try:
    import cv2
except ImportError:  # pragma: no cover - optional at import time
    cv2 = None

logger = logging.getLogger(__name__)

# Patches per forward call are sized to a fixed pixel budget rather than a fixed
# count: the served checkpoints use patch sizes from 32 to 256, and a batch of
# eight 32x32 patches is a 32 KB tensor issued thousands of times, where
# per-call overhead dwarfs the arithmetic.
DEFAULT_BATCH_PIXELS = 8 * 512 * 512

DEFAULT_MODEL_SERVER_URL = "http://mcdmodels.ptm.tn:555/"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "medical-std" / "models"

KNOWN_ARCHITECTURES = {"unet", "attention_unet", "unetpp", "deeplabv3p"}
KNOWN_LOSSES = {"bce_dice", "focal_dice", "topk_bce_dice", "focal_tversky"}
KNOWN_DATASETS = {"inbreast", "cbis_ddsm_new"}

_MODEL_NAME_PATTERN = re.compile(
    r"^(attention_unet|deeplabv3p|unetpp|unet)"
    r"_(bce_dice|focal_dice|topk_bce_dice|focal_tversky)"
    r"_(\d+)"
    r"_(inbreast|cbis_ddsm_new)"
    r"(_clahe)?$"
)


class _OnnxModel:
    """Adapter presenting an ONNX Runtime session as a callable torch module.

    Keeps the inference loop free of backend branching: it still calls
    ``self.model(tensor)`` and still gets logits back.
    """

    def __init__(self, path: str, providers: Optional[list] = None, threads: int = 0):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            path, options, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        array = x.detach().cpu().contiguous().numpy()
        return torch.from_numpy(self.session.run(None, {self.input_name: array})[0])

    def to(self, *args, **kwargs) -> "_OnnxModel":
        return self

    def eval(self) -> "_OnnxModel":
        return self


def _parse_model_name(name: str) -> Optional[dict]:
    """Parse a model directory name into metadata dict, or None if invalid."""
    m = _MODEL_NAME_PATTERN.match(name)
    if not m:
        return None
    return {
        "name": name,
        "architecture": m.group(1),
        "loss": m.group(2),
        "patch_size": int(m.group(3)),
        "dataset": m.group(4),
        "uses_clahe": m.group(5) is not None,
    }


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
        crop_to_breast: Restrict inference to the breast bounding box, as the
            training/validation pipeline does.  A mammogram is mostly air, and
            the model never saw that region during training.
        breast_margin: Pixels of slack around the breast bounding box.
        skip_background_patches: Drop patches that lie entirely outside the
            breast mask.  Free once the mask has been computed for the crop.
        stride_ratio: Overrides the checkpoint's own value.  ``0.5`` infers every
            pixel four times; see ``blend``.
        blend: ``"gaussian"`` weights a patch's centre above its edges when
            averaging overlaps, which suppresses seams better than ``"uniform"``
            at the same stride — so it is what makes a larger stride viable.
        batch_pixels: Pixel budget per forward call.
        channels_last: Use the channels-last memory format, usually faster for
            convolutions on both oneDNN CPU and CUDA tensor cores.
        compile_model: Wrap the model in :func:`torch.compile`.  The cost is
            paid once per process and per input shape, which suits a long-lived
            worker with a fixed patch size.
        onnx_path: Run inference through ONNX Runtime instead of torch.  The
            checkpoint is still loaded, because it carries the patch size,
            threshold and CLAHE flag.  See :meth:`export_onnx`.
        onnx_providers: ONNX Runtime execution providers, defaulting to CPU.
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
        crop_to_breast: bool = True,
        breast_margin: int = 30,
        skip_background_patches: bool = True,
        stride_ratio: Optional[float] = None,
        blend: str = "gaussian",
        batch_pixels: int = DEFAULT_BATCH_PIXELS,
        channels_last: bool = False,
        compile_model: bool = False,
        onnx_path: Optional[str] = None,
        onnx_providers: Optional[list] = None,
    ):
        super().__init__(device=device, precision=precision)

        if checkpoint_path is None and model is None:
            raise ValueError("Provide either checkpoint_path or model")
        if blend not in ("gaussian", "uniform"):
            raise ValueError(f"blend must be 'gaussian' or 'uniform', got {blend!r}")

        self.patch_size = patch_size
        self.stride = stride or patch_size // 2
        self.threshold = threshold
        self.min_lesion_area = min_lesion_area
        self.use_clahe = use_clahe
        self.crop_to_breast = crop_to_breast
        self.breast_margin = breast_margin
        self.skip_background_patches = skip_background_patches
        self.blend = blend
        self.batch_pixels = batch_pixels

        self._explicit_stride = stride is not None
        self._stride_ratio_override = stride_ratio
        self._model_name: Optional[str] = None

        if checkpoint_path is not None:
            self._load_from_checkpoint(checkpoint_path)
        else:
            self.model = model.to(self.device)
            self.model.eval()

        if stride_ratio is not None:
            self.stride = max(1, int(self.patch_size * stride_ratio))

        self.channels_last = channels_last and onnx_path is None
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if compile_model and onnx_path is None:
            self.model = torch.compile(self.model)

        # Swapped in last: the checkpoint is still what supplies patch_size,
        # threshold and the CLAHE flag, so it has to be loaded either way.
        if onnx_path is not None:
            self.model = _OnnxModel(onnx_path, providers=onnx_providers)

        # Populated after apply()
        self.probability_map: Optional[torch.Tensor] = None
        self.lesion_count: int = 0
        self.breast_bbox: Optional[tuple] = None
        self.patches_inferred: int = 0
        self.patches_skipped: int = 0

    # ------------------------------------------------------------------
    # Remote model support
    # ------------------------------------------------------------------

    @classmethod
    def list_available_models(cls, server_url: str = None) -> list[dict]:
        """Query the model server and return metadata for each available model.

        Returns a list of dicts with keys: name, architecture, loss,
        patch_size, dataset, uses_clahe, url.
        """
        url = (server_url or DEFAULT_MODEL_SERVER_URL).rstrip("/") + "/"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        html = resp.text

        # Extract href links ending with /
        dirs = re.findall(r'href="([^"]+/)"', html)
        models = []
        for d in dirs:
            dirname = d.rstrip("/")
            if dirname in (".", "..") or dirname.startswith("?"):
                continue
            info = _parse_model_name(dirname)
            if info is not None:
                info["url"] = url + dirname + "/"
                models.append(info)
        return models

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        server_url: str = None,
        cache_dir: str = None,
        device: str = None,
        precision: Precision = Precision.FULL,
        force_download: bool = False,
        **kwargs,
    ) -> "DeepSegmentationAlgorithm":
        """Download a pretrained model from the server and return a ready-to-use algorithm.

        Extra keyword arguments are forwarded to the constructor, so inference
        tuning (``stride_ratio``, ``blend``, ``crop_to_breast``, …) survives the
        checkpoint round-trip.
        """
        base_url = (server_url or DEFAULT_MODEL_SERVER_URL).rstrip("/")
        checkpoint_url = f"{base_url}/{model_name}/best_model.pt"

        cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        local_path = cache / model_name / "best_model.pt"

        if not local_path.exists() or force_download:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s -> %s", checkpoint_url, local_path)
            resp = requests.get(checkpoint_url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download complete: %s", local_path)

        algo = cls(
            checkpoint_path=str(local_path),
            device=device,
            precision=precision,
            **kwargs,
        )
        algo._model_name = model_name
        return algo

    def export_onnx(self, path: str, opset: int = 17) -> str:
        """Export the loaded model to ONNX with a dynamic batch axis.

        The batch axis is dynamic because the pipeline sizes batches from a
        pixel budget, and the final batch of a run is usually short.  Height and
        width stay static — they are the checkpoint's patch size and never vary.
        """
        if isinstance(self.model, _OnnxModel):
            raise ValueError(
                "This instance already runs ONNX; export from a torch one."
            )

        ps = self.patch_size
        dummy = torch.zeros(1, 1, ps, ps, device=self.device)
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )
        logger.info("Exported ONNX model to %s (patch size %d)", path, ps)
        return path

    @property
    def model_info(self) -> Optional[dict]:
        """Return metadata about the loaded model, or None if name unknown."""
        if self._model_name is None:
            return None
        return _parse_model_name(self._model_name)

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

        if not self._explicit_stride and self._stride_ratio_override is None:
            stride_ratio = inference_cfg.get("stride_ratio", 0.5)
            self.stride = max(1, int(self.patch_size * stride_ratio))

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
        full_shape = img.shape

        # Normalize to [0, 1].  The reference pipeline's normalize_bit_depth step
        # is redundant here: it scales by 2^bits and the caller then divides by
        # the result's own max, which is algebraically this same division.
        img_max = img.max()
        if img_max > 1.0:
            img = img / img_max

        breast_mask = self._breast_mask(img) if self.crop_to_breast else None
        bbox = (
            self._mask_bbox(breast_mask, full_shape)
            if breast_mask is not None
            else None
        )
        self.breast_bbox = bbox
        if bbox is not None:
            y0, x0, y1, x1 = bbox
            img = img[y0:y1, x0:x1]
            breast_mask = breast_mask[y0:y1, x0:x1]

        # CLAHE after the crop, matching the reference pipeline: the transform is
        # tile-local, so including the air region shifts every tile's histogram.
        if self.use_clahe:
            img = self._apply_clahe(img)

        prob_map = self._infer_patches(
            img, breast_mask if self.skip_background_patches else None
        )

        if bbox is not None:
            full = torch.zeros(full_shape, dtype=prob_map.dtype)
            full[bbox[0] : bbox[2], bbox[1] : bbox[3]] = prob_map
            prob_map = full

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

    @torch.inference_mode()
    def _infer_patches(
        self,
        image: torch.Tensor,
        breast_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sliding-window inference with weighted overlap averaging.

        Patches are gathered, run and stitched as whole batches rather than one
        at a time — at a 32 px patch size a full mammogram is over 50,000 of
        them, and the per-patch Python was costing seconds before any of the
        arithmetic happened.
        """
        h, w = image.shape
        ps = self.patch_size

        # Padding the image up to at least one patch removes the ragged-edge
        # case entirely, so every patch below is exactly ps x ps and the
        # gather/scatter can be a single indexing op.
        pad_h, pad_w = max(0, ps - h), max(0, ps - w)
        if pad_h or pad_w:
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
            if breast_mask is not None:
                breast_mask = torch.nn.functional.pad(breast_mask, (0, pad_w, 0, pad_h))
        ph, pw = image.shape

        ys, xs = self._patch_positions(ph, pw, ps, self.stride, breast_mask)
        self.patches_inferred = len(ys)

        prediction_sum = torch.zeros(ph, pw)
        weight_sum = torch.zeros(ph, pw)
        if len(ys) == 0:
            return prediction_sum[:h, :w]

        window = self._blend_window(ps)
        batch = max(1, self.batch_pixels // (ps * ps))

        # Two ways to accumulate, and which is cheaper flips with patch size.
        # A slice assignment per patch costs ~36 us of Python but touches each
        # pixel once; a batched index_add_ has no per-patch Python but has to
        # materialise an int64 index per pixel.  Trading ~36 us against ps^2
        # index writes puts the break-even near ps = 110.
        scatter = ps <= 128
        if scatter:
            rows = torch.arange(ps)
            offsets = (rows.view(1, ps, 1) * pw + rows.view(1, 1, ps)).reshape(1, -1)
            flat_sum = prediction_sum.view(-1)
            flat_weight = weight_sum.view(-1)

        for i in range(0, len(ys), batch):
            by, bx = ys[i : i + batch], xs[i : i + batch]
            patches = self._gather_patches(image, by, bx, ps, scatter)
            inp = patches.unsqueeze(1).to(self.device)
            if self.channels_last:
                inp = inp.contiguous(memory_format=torch.channels_last)
            probs = torch.sigmoid(self.model(inp)).squeeze(1).float().cpu()
            weighted = probs * window

            if scatter:
                idx = ((by * pw + bx).view(-1, 1) + offsets).reshape(-1)
                flat_sum.index_add_(0, idx, weighted.reshape(-1))
                flat_weight.index_add_(0, idx, window.expand_as(probs).reshape(-1))
            else:
                for k in range(len(by)):
                    y, x = int(by[k]), int(bx[k])
                    prediction_sum[y : y + ps, x : x + ps] += weighted[k]
                    weight_sum[y : y + ps, x : x + ps] += window

        return (prediction_sum / weight_sum.clamp(min=1e-8))[:h, :w]

    def _patch_positions(
        self,
        h: int,
        w: int,
        ps: int,
        stride: int,
        breast_mask: Optional[torch.Tensor],
    ) -> tuple:
        """Top-left corners of every patch, dropping all-background ones.

        ``unique()`` is load-bearing.  Clamping the last few strides to ``h - ps``
        maps several of them onto the same corner, and averaging over the
        resulting duplicates weights those patches by how many strides happened
        to collapse onto them — a band ``ps`` wide along the bottom and right
        edges came out over-weighted, reaching an overlap count of 16 where the
        true maximum is 9.  De-duplicating makes every distinct patch count once.
        """
        ys = torch.arange(0, h, stride).clamp(max=h - ps).unique()
        xs = torch.arange(0, w, stride).clamp(max=w - ps).unique()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y, grid_x = grid_y.reshape(-1), grid_x.reshape(-1)

        if breast_mask is None:
            self.patches_skipped = 0
            return grid_y, grid_x

        # Summed-area table: one O(1) lookup per patch instead of slicing each.
        integral = torch.zeros(h + 1, w + 1)
        integral[1:, 1:] = (breast_mask > 0).float().cumsum(0).cumsum(1)
        covered = (
            integral[grid_y + ps, grid_x + ps]
            - integral[grid_y, grid_x + ps]
            - integral[grid_y + ps, grid_x]
            + integral[grid_y, grid_x]
        )
        keep = covered > 0
        self.patches_skipped = int((~keep).sum())
        return grid_y[keep], grid_x[keep]

    @staticmethod
    def _gather_patches(
        image: torch.Tensor,
        ys: torch.Tensor,
        xs: torch.Tensor,
        ps: int,
        vectorised: bool,
    ) -> torch.Tensor:
        """Stack one batch of patches. Same size/shape either way.

        Mirrors the accumulation trade-off: advanced indexing avoids per-patch
        Python but builds an int64 index per pixel, which is the losing side of
        the deal once a patch is large enough for the Python to disappear.
        """
        if not vectorised:
            return torch.stack(
                [
                    image[int(y) : int(y) + ps, int(x) : int(x) + ps]
                    for y, x in zip(ys, xs)
                ]
            )
        rows = ys.view(-1, 1, 1) + torch.arange(ps).view(1, ps, 1)
        cols = xs.view(-1, 1, 1) + torch.arange(ps).view(1, 1, ps)
        return image[rows, cols]

    def _blend_window(self, ps: int) -> torch.Tensor:
        """Per-pixel weight applied to a patch when averaging overlaps.

        A patch's border pixels are the ones the model had least context for, so
        weighting them down suppresses seams better than a plain mean — which is
        what makes a coarser stride viable at the same quality.
        """
        if self.blend == "uniform":
            return torch.ones(ps, ps)
        coords = torch.arange(ps, dtype=torch.float32) - (ps - 1) / 2.0
        sigma = ps / 8.0
        line = torch.exp(-(coords**2) / (2 * sigma**2))
        window = torch.outer(line, line)
        # Never let a weight reach zero: a pixel covered by exactly one patch
        # would otherwise divide by ~0 and come back as noise.
        return window.clamp(min=window.max() * 0.1)

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

            # Find contour for polygon using skimage
            contours = find_contours(component, level=0.5)
            if not contours:
                continue

            # Pick the longest contour
            contour = max(contours, key=len)
            # find_contours returns (row, col) — convert to (x, y)
            contour_pts = [(int(round(c[1])), int(round(c[0]))) for c in contour]

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
        """Apply CLAHE preprocessing. Returns float tensor in [0, 1].

        Uses OpenCV to match training bit for bit — the checkpoints were built
        with ``cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))``, where the
        grid size is a *tile count*.  The scikit-image fallback takes its own
        default ``kernel_size``, which is height/8 x width/8 and so the same 64
        tiles; passing ``(8, 8)`` there would instead ask for 8x8-*pixel* tiles —
        roughly 213,000 of them on a full mammogram, which is both far slower and
        a different transform from the one the models were trained on.
        """
        img_np = image.detach().cpu().numpy()
        img_min, img_max = img_np.min(), img_np.max()
        if img_max - img_min > 0:
            img_norm = (img_np - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img_np)

        if cv2 is not None:
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(grid_size, grid_size)
            )
            enhanced = clahe.apply((img_norm * 255).astype(np.uint8))
            return torch.from_numpy(enhanced.astype(np.float32) / 255.0)

        logger.warning(
            "OpenCV unavailable — falling back to scikit-image CLAHE, which is "
            "an approximation of the transform used during training."
        )
        enhanced = equalize_adapthist(img_norm, clip_limit=0.02)
        return torch.from_numpy(enhanced.astype(np.float32))

    # ------------------------------------------------------------------
    # Breast region
    # ------------------------------------------------------------------

    def _breast_mask(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """Binary breast mask, or None if it could not be computed.

        A failure here must not fail the request: the mask is an optimisation
        plus a training-parity step, and falling back to the whole frame is the
        old behaviour rather than an error.
        """
        try:
            src = InMemoryImage(array=image)
            out = MammographyPreprocessing.breast_mask(src, device="cpu")
            mask = out.pixel_data
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            return mask.detach().cpu()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Breast mask failed, using the full frame: %s", exc)
            return None

    def _mask_bbox(self, mask: torch.Tensor, shape: tuple) -> Optional[tuple]:
        """Bounding box of the mask as (y0, x0, y1, x1), with margin applied."""
        nonzero = torch.nonzero(mask > 0)
        if nonzero.numel() == 0:
            return None

        h, w = shape
        margin = self.breast_margin
        y_min, x_min = nonzero.min(dim=0).values.tolist()
        y_max, x_max = nonzero.max(dim=0).values.tolist()
        bbox = (
            max(0, y_min - margin),
            max(0, x_min - margin),
            min(h, y_max + 1 + margin),
            min(w, x_max + 1 + margin),
        )
        if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
            return None
        return bbox
