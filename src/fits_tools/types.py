#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Package-wide type aliases and enumerations for ``fits_tools``.

Defines shared types used across the package in function signatures,
type annotations, and region specifications.

Types
-----
StrOrPath
    Union of ``str`` and ``pathlib.Path`` for flexible path arguments.
RegionType
    Enumeration of named image region types for section extraction.
"""
from enum import Enum
from pathlib import Path
from typing import TypeAlias

StrOrPath: TypeAlias = str | Path


class RegionType(Enum):
    """Named region types for image section extraction."""
    FULL = "full"
    DATA = "data"
    BIAS = "bias"
    TRIM = "trim"
    DARK = "dark"


__all__ = ["StrOrPath", "RegionType"]
