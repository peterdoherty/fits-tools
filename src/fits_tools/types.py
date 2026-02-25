"""
Package-wide type aliases for fits_tools.

Import from here rather than redefining StrOrPath in each module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

StrOrPath: TypeAlias = Union[str, Path]