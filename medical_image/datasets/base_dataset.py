"""
Abstract base dataset for medical image PyTorch dataloaders.

Provides the common interface and behavior shared by all medical image dataset
implementations (INbreast, Custom INbreast, CBIS-DDSM, etc.).
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List, Tuple, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from log_manager import logger
from medical_image.utils.downloader import download


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for PyTorch-compatible medical image datasets.

    Enforces lazy loading: images are never pre-loaded into memory.
    Each sample is loaded on-the-fly in ``__getitem__``.

    Subclasses must implement:
        - ``_build_sample_list()``: scan the dataset directory and build
          an internal list of samples (paths, metadata).
        - ``_load_sample(idx)``: load a single sample by index, returning
          a dict with ``"image"``, ``"mask"`` (or ``"label"``), and ``"metadata"``.

    Args:
        root_dir: Root directory of the dataset.
        transform: Optional callable transform applied to the image tensor.
        target_transform: Optional callable transform applied to the mask/label.
        target_size: Optional (H, W) to resize images and masks on load.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_size = target_size

        self._samples: List[Any] = []
        self._build_sample_list()

        logger.info(
            f"{self.__class__.__name__}: {len(self._samples)} samples "
            f"from {root_dir}"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_sample_list(self) -> None:
        """
        Scan the dataset directory and populate ``self._samples``.

        Each entry should contain enough information (paths, IDs) to
        lazily load a single sample in ``_load_sample``.
        """
        ...

    @abstractmethod
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """
        Load a single sample by index.

        Must return a dictionary with at least:
            - ``"image"``: ``torch.Tensor`` of shape ``(C, H, W)``
            - ``"mask"``: ``torch.Tensor`` of shape ``(1, H, W)`` or ``"label"``: ``int``
            - ``"metadata"``: ``dict`` with case_id and other info

        The image and mask should NOT have transforms applied yet —
        that is handled by the base class.
        """
        ...

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load sample ``idx`` and apply transforms.

        Returns:
            Dict with ``"image"``, ``"mask"`` (or ``"label"``), and ``"metadata"``.
        """
        sample = self._load_sample(idx)

        image = sample["image"]
        mask = sample.get("mask")
        label = sample.get("label")

        # Optional resize
        if self.target_size is not None:
            image = self._resize(image, self.target_size)
            if mask is not None:
                mask = self._resize(mask, self.target_size, mode="nearest")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None and mask is not None:
            mask = self.target_transform(mask)

        result: Dict[str, Any] = {
            "image": image,
            "metadata": sample.get("metadata", {}),
        }
        if mask is not None:
            result["mask"] = mask
        if label is not None:
            result["label"] = label

        return result

    # ------------------------------------------------------------------
    # Download classmethod
    # ------------------------------------------------------------------

    @classmethod
    def download(
        cls,
        source: str,
        destination: str,
        method: Literal["local", "http", "ftp"] = "local",
        percentage: Optional[float] = None,
    ) -> str:
        """
        Download the dataset from a source.

        Args:
            source: Source path or URL.
            destination: Local destination directory.
            method: ``'local'``, ``'http'``, or ``'ftp'``.
            percentage: Optional subset percentage (0–1] for large datasets.

        Returns:
            Absolute path to the downloaded dataset.
        """
        return download(source, destination, method=method, percentage=percentage)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(
        tensor: torch.Tensor,
        size: Tuple[int, int],
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """
        Resize a tensor to ``(H, W)`` using interpolation.

        Handles both 3D ``(C, H, W)`` and 2D ``(H, W)`` inputs.
        """
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            align = mode != "nearest"
            out = F.interpolate(
                tensor.float(), size=size, mode=mode,
                align_corners=align if align else None,
            )
            return out.squeeze(0).squeeze(0)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # (1, C, H, W)
            align = mode != "nearest"
            out = F.interpolate(
                tensor.float(), size=size, mode=mode,
                align_corners=align if align else None,
            )
            return out.squeeze(0)

        return tensor

    @staticmethod
    def _to_chw(tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor is in ``(C, H, W)`` format.

        - ``(H, W)`` → ``(1, H, W)``
        - ``(H, W, C)`` → ``(C, H, W)``
        - ``(C, H, W)`` → unchanged
        """
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        if tensor.ndim == 3 and tensor.shape[2] in (1, 3, 4):
            return tensor.permute(2, 0, 1)
        return tensor
