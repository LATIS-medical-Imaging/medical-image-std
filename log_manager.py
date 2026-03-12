"""
Backward-compatible shim. New code should use:
    from medical_image.utils.logging import logger
"""

from medical_image.utils.logging import logger

__all__ = ["logger"]
