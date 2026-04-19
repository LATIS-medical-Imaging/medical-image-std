"""Backward-compatible re-export.

The canonical location for :class:`Annotation` and :class:`GeometryType` is
now :mod:`medical_image.data.annotation`.  This module re-exports them so
that existing ``from medical_image.utils.annotation import ...`` statements
continue to work.
"""

from medical_image.data.annotation import Annotation, GeometryType

__all__ = ["Annotation", "GeometryType"]
