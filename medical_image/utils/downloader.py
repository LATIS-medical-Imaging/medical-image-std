"""
Universal download system for medical image datasets.

Supports local copy, HTTP download, and FTP transfer with optional
partial-download support for large datasets (e.g. CBIS-DDSM).
"""

import ftplib
import os
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Literal, Optional, List

from medical_image.utils.logging import logger


def download(
    source: str,
    destination: str,
    method: Literal["local", "http", "ftp"] = "local",
    percentage: Optional[float] = None,
    seed: int = 42,
) -> str:
    """
    Download or copy a dataset from a source to a destination directory.

    Args:
        source: Source path (local path, HTTP URL, or FTP URL).
        destination: Local directory to store the downloaded data.
        method: Download method — ``'local'``, ``'http'``, or ``'ftp'``.
        percentage: Optional float in (0, 1] to download only a random subset
                    of top-level items (e.g. 0.2 = 20%). Useful for CBIS-DDSM.
                    Maintains case integrity by selecting at the folder level.
        seed: Random seed for reproducible subset selection.

    Returns:
        Absolute path to the destination directory.
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    if method == "local":
        _download_local(source, dest, percentage, seed)
    elif method == "http":
        _download_http(source, dest)
    elif method == "ftp":
        _download_ftp(source, dest)
    else:
        raise ValueError(f"Unsupported download method: {method}")

    logger.info(f"Download complete → {dest}")
    return str(dest.resolve())


# ---------------------------------------------------------------------------
# Local copy
# ---------------------------------------------------------------------------


def _download_local(
    source: str,
    dest: Path,
    percentage: Optional[float],
    seed: int,
) -> None:
    """Copy from a local directory, optionally selecting a random subset."""
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    items = sorted(src.iterdir())

    if percentage is not None:
        items = _select_subset(items, percentage, seed)
        logger.info(
            f"Subset selection: {len(items)} items "
            f"({percentage * 100:.0f}% of total)"
        )

    for item in items:
        dst_item = dest / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)

    logger.info(f"Copied {len(items)} items from {src}")


# ---------------------------------------------------------------------------
# HTTP download
# ---------------------------------------------------------------------------


def _download_http(source: str, dest: Path) -> None:
    """Download a file via HTTP/HTTPS with progress logging."""
    filename = source.split("/")[-1] or "download"
    output_path = dest / filename

    logger.info(f"HTTP download: {source} → {output_path}")

    def _report_hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(downloaded / total_size * 100, 100)
            if block_num % 100 == 0:
                logger.debug(f"  Progress: {pct:.1f}%")

    urllib.request.urlretrieve(source, str(output_path), reporthook=_report_hook)
    logger.info(f"Downloaded {output_path.name}")


# ---------------------------------------------------------------------------
# FTP download
# ---------------------------------------------------------------------------


def _download_ftp(source: str, dest: Path) -> None:
    """
    Download files via FTP.

    Expects source format: ``ftp://host/path/to/directory``
    """
    # Parse FTP URL
    if source.startswith("ftp://"):
        source = source[6:]

    parts = source.split("/", 1)
    host = parts[0]
    remote_path = "/" + parts[1] if len(parts) > 1 else "/"

    logger.info(f"FTP download: {host}{remote_path} → {dest}")

    ftp = ftplib.FTP(host)
    ftp.login()
    ftp.cwd(remote_path)

    filenames = ftp.nlst()
    for fn in filenames:
        local_path = dest / fn
        logger.debug(f"  Downloading {fn}...")
        with open(local_path, "wb") as f:
            ftp.retrbinary(f"RETR {fn}", f.write)

    ftp.quit()
    logger.info(f"FTP: downloaded {len(filenames)} files")


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def _select_subset(
    items: List[Path],
    percentage: float,
    seed: int,
) -> List[Path]:
    """
    Randomly select a percentage of items for partial download.

    Used for large datasets like CBIS-DDSM to maintain case integrity
    (selection at the folder level, never splitting image/masks).

    Args:
        items: List of paths (files or directories).
        percentage: Float in (0, 1].
        seed: Random seed for reproducibility.

    Returns:
        Sorted subset of items.
    """
    if not 0 < percentage <= 1:
        raise ValueError(f"percentage must be in (0, 1], got {percentage}")

    n_select = max(1, int(len(items) * percentage))

    rng = random.Random(seed)
    selected = sorted(rng.sample(items, n_select), key=lambda p: p.name)

    logger.info(f"Selected {n_select}/{len(items)} items (seed={seed})")
    for item in selected:
        logger.debug(f"  → {item.name}")

    return selected
