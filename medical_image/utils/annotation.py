# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 [YOUR NAME / ORGANIZATION]
#
# This file is part of medical-image-std.
#
# medical-image-std is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# medical-image-std is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with medical-image-std. If not, see
# <https://www.gnu.org/licenses/>.

"""Backward-compatible re-export.

The canonical location for :class:`Annotation` and :class:`GeometryType` is
now :mod:`medical_image.data.annotation`.  This module re-exports them so
that existing ``from medical_image.utils.annotation import ...`` statements
continue to work.
"""

from medical_image.data.annotation import Annotation, GeometryType

__all__ = ["Annotation", "GeometryType"]
