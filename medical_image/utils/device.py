"""
GPU device management, memory handling, mixed precision, and multi-GPU utilities.
"""

import functools
from enum import Enum
from typing import List, Optional, Union

import torch

from medical_image.utils.logging import logger


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(
    *images, explicit: Union[str, torch.device, None] = None
) -> torch.device:
    """
    Determine the target device for a processing operation.

    Priority:
        1. Explicit device parameter (if provided)
        2. Device of the first loaded image
        3. Fallback to CPU
    """
    if explicit is not None:
        return torch.device(explicit)
    for img in images:
        if hasattr(img, "pixel_data") and img.pixel_data is not None:
            return img.pixel_data.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

class Precision(Enum):
    FULL = torch.float32
    HALF = torch.float16
    BFLOAT16 = torch.bfloat16


_default_precision: Precision = Precision.FULL


def set_default_precision(precision: Precision) -> None:
    global _default_precision
    _default_precision = precision


def get_default_precision() -> Precision:
    return _default_precision


def get_dtype() -> torch.dtype:
    return _default_precision.value


# ---------------------------------------------------------------------------
# DeviceContext — GPU-aware context manager
# ---------------------------------------------------------------------------

class DeviceContext:
    """
    Context manager for GPU-aware processing with automatic memory management.

    Features:
        - Clears GPU cache on entry and exit
        - Provides memory usage tracking
        - Automatic CPU fallback when CUDA is unavailable
    """

    def __init__(
        self,
        device: str = "cuda",
        fallback: str = "cpu",
        verbose: bool = False,
    ):
        self.primary = torch.device(device)
        self.fallback = torch.device(fallback)
        self.active_device = self.primary
        self.verbose = verbose

    def __enter__(self) -> "DeviceContext":
        if self.primary.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.verbose:
                free, total = torch.cuda.mem_get_info(self.primary)
                logger.info(
                    f"GPU memory: {free / 1e9:.1f} / {total / 1e9:.1f} GB free"
                )
        elif self.primary.type == "cuda":
            logger.warning("CUDA requested but not available — falling back to CPU")
            self.active_device = self.fallback
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active_device.type == "cuda":
            torch.cuda.empty_cache()
        if exc_type is torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM — retrying on CPU")
            self.active_device = self.fallback
            return True  # suppress exception
        return False

    @property
    def device(self) -> torch.device:
        return self.active_device

    def memory_stats(self) -> dict:
        """Return current GPU memory usage."""
        if self.active_device.type != "cuda":
            return {"device": "cpu"}
        free, total = torch.cuda.mem_get_info(self.active_device)
        return {
            "device": str(self.active_device),
            "allocated_gb": torch.cuda.memory_allocated(self.active_device) / 1e9,
            "free_gb": free / 1e9,
            "total_gb": total / 1e9,
        }


# ---------------------------------------------------------------------------
# @gpu_safe — OOM fallback decorator
# ---------------------------------------------------------------------------

def gpu_safe(func):
    """Decorator: catches CUDA OOM and retries the operation on CPU."""

    @functools.wraps(func)
    def wrapper(*args, device=None, **kwargs):
        device = resolve_device(
            *[a for a in args if hasattr(a, "pixel_data")], explicit=device
        )
        try:
            return func(*args, device=device, **kwargs)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"{func.__name__}: GPU OOM — retrying on CPU")
            torch.cuda.empty_cache()
            return func(*args, device=torch.device("cpu"), **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# AsyncGPUPipeline — overlapped I/O + compute with CUDA streams
# ---------------------------------------------------------------------------

class AsyncGPUPipeline:
    """
    Overlap disk I/O, CPU→GPU transfer, and GPU compute using CUDA streams.

    Only usable when CUDA is available.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("AsyncGPUPipeline requires CUDA")
        self.compute_stream = torch.cuda.Stream(self.device)
        self.transfer_stream = torch.cuda.Stream(self.device)

    def process_images(
        self, images: list, algorithm
    ) -> list:
        """
        Process pre-loaded Image objects with overlapped transfer and compute.

        Args:
            images: List of Image objects (already loaded).
            algorithm: An Algorithm instance.

        Returns:
            List of output Image objects.
        """
        results = []
        for img in images:
            # Transfer to GPU on transfer stream
            with torch.cuda.stream(self.transfer_stream):
                gpu_data = img.pixel_data.pin_memory().to(
                    self.device, non_blocking=True
                )

            # Compute on compute stream
            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_stream(self.transfer_stream)
                img.pixel_data = gpu_data
                output = img.clone()
                algorithm(img, output)
                results.append(output)

        torch.cuda.synchronize()
        return results


# ---------------------------------------------------------------------------
# MultiGPUAlgorithm — data-parallel across GPUs
# ---------------------------------------------------------------------------

class MultiGPUAlgorithm:
    """
    Distribute algorithm execution across available GPUs (data-parallel).
    """

    def __init__(
        self,
        algorithm_cls: type,
        gpu_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("MultiGPUAlgorithm requires CUDA")
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))

        self.gpu_ids = gpu_ids
        self.algorithms = {
            gpu_id: algorithm_cls(device=f"cuda:{gpu_id}", **kwargs)
            for gpu_id in gpu_ids
        }

    def apply_batch(self, images: list, outputs: list) -> list:
        """Distribute images across GPUs round-robin."""
        n_gpus = len(self.gpu_ids)
        results = [None] * len(images)

        for i, (img, out) in enumerate(zip(images, outputs)):
            gpu_id = self.gpu_ids[i % n_gpus]
            algo = self.algorithms[gpu_id]
            img.to(f"cuda:{gpu_id}")
            out.to(f"cuda:{gpu_id}")
            algo.apply(img, out)
            results[i] = out

        return results