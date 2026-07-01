#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""
Custom exception classes for the fits_tools package.

This module defines a hierarchy of exceptions for handling errors that occur
during FITS file operations, image processing, validation, and calibration
frame generation for CCD/CMOS imager test data.

All package-specific exceptions inherit from :class:`FitsToolsError`, so
callers can catch the entire hierarchy with a single ``except`` clause when
desired, or target specific sub-types for fine-grained handling.

Exception Hierarchy
-------------------
::

    FitsToolsError (base)
    ├── FileHandlingError
    │   ├── InvalidHDUIndexError
    │   ├── FileOverwriteError
    │   └── HDUStructureError
    ├── ImageProcessingError
    │   ├── DataShapeError
    │   │   └── DimensionalityError
    │   ├── WCSError
    │   └── RegionError
    ├── HeaderKeywordError
    ├── SectionParsingError
    ├── SelectionError
    ├── ValidationError
    ├── InvalidParameterError
    ├── UnsupportedOperationError
    └── MasterFrameGenerationError
        ├── NoInputFilesFoundError
        ├── MasterBiasCreationError
        ├── MasterDarkCreationError
        └── MasterFlatCreationError
            └── NormalizationError
"""

from __future__ import annotations

import inspect
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .types import StrOrPath

__all__ = [
    # Base
    "FitsToolsError",
    # File handling
    "FileHandlingError",
    "InvalidHDUIndexError",
    "FileOverwriteError",
    "HDUStructureError",
    # Image processing
    "ImageProcessingError",
    "DataShapeError",
    "DimensionalityError",
    "WCSError",
    "RegionError",
    # Headers / keywords
    "HeaderKeywordError",
    "SectionParsingError",
    # Selection / validation
    "SelectionError",
    "ValidationError",
    # Parameters / operations
    "InvalidParameterError",
    "UnsupportedOperationError",
    # Master frame generation
    "MasterFrameGenerationError",
    "NoInputFilesFoundError",
    "MasterBiasCreationError",
    "MasterDarkCreationError",
    "MasterFlatCreationError",
    "NormalizationError",
    # Utilities
    "fits_operation_context",
    "get_exception_hierarchy",
]


# =============================================================================
# Base Exception
# =============================================================================


class FitsToolsError(Exception):
    """Base exception for the fits_tools package.

    All custom exceptions in fits_tools inherit from this class, allowing
    callers to catch all package-specific errors with a single ``except``
    clause.
    """

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        message: str | None = None,
        **kwargs: Any,
    ) -> FitsToolsError:
        """Create an instance of this exception class wrapping *exc*.

        Parameters
        ----------
        exc : Exception
            The original exception to wrap.
        message : str, optional
            Human-readable message for the new exception. Defaults to
            ``str(exc)`` when not provided.
        **kwargs
            Additional keyword arguments forwarded to the class constructor.
            Useful for subclasses that accept structured fields such as
            ``filepath`` or ``keyword``.

        Returns
        -------
        FitsToolsError
            A new instance of this class (or subclass). Not raised; the
            caller is responsible for raising it. The returned exception
            has *exc* set as its ``__cause__`` so that chained tracebacks
            are preserved even if the caller raises without
            ``raise ... from ...``.
        """
        if message is None:
            message = str(exc)

        try:
            new_exc = cls(message, **kwargs)
        except TypeError:
            new_exc = cls(message)

        new_exc.__cause__ = exc
        return new_exc


# =============================================================================
# File Handling Exceptions
# =============================================================================


class FileHandlingError(FitsToolsError):
    """Raised for errors during FITS file reading, writing, or access.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    filepath : str or Path, optional
        Path to the file that caused the error.

    Attributes
    ----------
    filepath : Path or None
        Path to the problematic file.
    """

    def __init__(self, message: str, filepath: StrOrPath | None = None) -> None:
        self.filepath = Path(filepath) if filepath is not None else None
        super().__init__(message)

    def __repr__(self) -> str:
        base = f"{self.__class__.__name__}({self.args[0]!r}"
        if self.filepath is not None:
            base += f", filepath={self.filepath!r}"
        return base + ")"


class InvalidHDUIndexError(FileHandlingError):
    """Raised when a requested HDU index is out of bounds.

    Parameters
    ----------
    path : str or Path
        Path to the FITS file.
    hdu_index : int
        The HDU index that was requested.
    max_hdu : int
        The number of HDUs present in the file (valid indices are
        ``0`` through ``max_hdu - 1``).
    message : str, optional
        Custom error message. A default is generated if not supplied.

    Attributes
    ----------
    path : Path
        Path to the FITS file.
    hdu_index : int
        The requested (invalid) HDU index.
    max_hdu : int
        Total number of HDUs in the file.
    """

    def __init__(
        self,
        path: StrOrPath,
        hdu_index: int,
        max_hdu: int,
        message: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.hdu_index = hdu_index
        self.max_hdu = max_hdu

        if message is None:
            message = (
                f"HDU index {hdu_index} is out of bounds for '{path}'. "
                f"File contains {max_hdu} HDU(s) "
                f"(valid indices: 0–{max_hdu - 1})."
            )

        super().__init__(message, filepath=path)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"path={self.path!r}, "
            f"hdu_index={self.hdu_index!r}, "
            f"max_hdu={self.max_hdu!r})"
        )


class FileOverwriteError(FileHandlingError):
    """Raised when a file would be overwritten but overwriting is disabled.

    Parameters
    ----------
    filepath : str or Path
        Path to the file that would be overwritten.
    message : str, optional
        Custom error message. A default is generated if not supplied.
    """

    def __init__(self, filepath: StrOrPath, message: str | None = None) -> None:
        if message is None:
            message = (
                f"File '{filepath}' already exists. "
                "Pass overwrite=True to allow overwriting."
            )
        super().__init__(message, filepath=filepath)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath={self.filepath!r})"


class HDUStructureError(FileHandlingError):
    """Raised when a FITS HDU structure is invalid or non-standard.

    Parameters
    ----------
    message : str
        Human-readable description of the structural problem.
    filepath : str or Path, optional
        Path to the problematic file.
    issue : str, optional
        Short machine-readable label describing the structural problem,
        e.g. ``'missing_primary_hdu'``.

    Attributes
    ----------
    issue : str or None
        Short label describing the structural problem.
    """

    def __init__(
        self,
        message: str,
        filepath: StrOrPath | None = None,
        issue: str | None = None,
    ) -> None:
        self.issue = issue
        super().__init__(message, filepath=filepath)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        if self.filepath is not None:
            parts.append(f"filepath={self.filepath!r}")
        if self.issue is not None:
            parts.append(f"issue={self.issue!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# Image Processing Exceptions
# =============================================================================


class ImageProcessingError(FitsToolsError):
    """Raised for errors during image processing operations.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    operation : str, optional
        Name of the operation that failed.
    """

    def __init__(self, message: str, operation: str | None = None) -> None:
        self.operation = operation
        super().__init__(message)

    def __repr__(self) -> str:
        base = f"{self.__class__.__name__}({self.args[0]!r}"
        if self.operation is not None:
            base += f", operation={self.operation!r}"
        return base + ")"


class DataShapeError(ImageProcessingError):
    """Raised when array shapes are incompatible for an operation.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    expected_shape : tuple, optional
        The shape that was expected.
    actual_shape : tuple, optional
        The shape that was encountered.
    operation : str, optional
        Name of the operation that failed.
    """

    def __init__(
        self,
        message: str,
        expected_shape: tuple | None = None,
        actual_shape: tuple | None = None,
        operation: str | None = None,
    ) -> None:
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        super().__init__(message, operation=operation)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        if self.expected_shape is not None:
            parts.append(f"expected_shape={self.expected_shape!r}")
        if self.actual_shape is not None:
            parts.append(f"actual_shape={self.actual_shape!r}")
        if self.operation is not None:
            parts.append(f"operation={self.operation!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class DimensionalityError(DataShapeError):
    """Raised when array dimensionality is wrong for an operation.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    ndim : int, optional
        The actual number of dimensions encountered.
    required_ndim : int or sequence of int, optional
        The required number of dimensions.
    operation : str, optional
        Name of the operation that failed.
    """

    def __init__(
        self,
        message: str,
        ndim: int | None = None,
        required_ndim: Any = None,
        operation: str | None = None,
    ) -> None:
        self.ndim = ndim
        self.required_ndim = required_ndim
        super().__init__(message, operation=operation)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        if self.ndim is not None:
            parts.append(f"ndim={self.ndim!r}")
        if self.required_ndim is not None:
            parts.append(f"required_ndim={self.required_ndim!r}")
        if self.operation is not None:
            parts.append(f"operation={self.operation!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class WCSError(ImageProcessingError):
    """Raised for World Coordinate System (WCS) related errors.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    filepath : str or Path, optional
        Path to the FITS file with the problematic WCS.
    wcs_operation : str, optional
        Name of the WCS operation that failed.
    """

    def __init__(
        self,
        message: str,
        filepath: StrOrPath | None = None,
        wcs_operation: str | None = None,
    ) -> None:
        self.filepath = Path(filepath) if filepath is not None else None
        self.wcs_operation = wcs_operation
        super().__init__(message, operation=wcs_operation)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        if self.filepath is not None:
            parts.append(f"filepath={self.filepath!r}")
        if self.wcs_operation is not None:
            parts.append(f"wcs_operation={self.wcs_operation!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class RegionError(ImageProcessingError):
    """Raised when a region specification is invalid.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    region : tuple, optional
        The invalid region specification.
    image_shape : tuple, optional
        Shape of the image the region was applied to.
    operation : str, optional
        Name of the operation that failed.
    """

    def __init__(
        self,
        message: str,
        region: tuple | None = None,
        image_shape: tuple | None = None,
        operation: str | None = None,
    ) -> None:
        self.region = region
        self.image_shape = image_shape
        super().__init__(message, operation=operation)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        if self.region is not None:
            parts.append(f"region={self.region!r}")
        if self.image_shape is not None:
            parts.append(f"image_shape={self.image_shape!r}")
        if self.operation is not None:
            parts.append(f"operation={self.operation!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# Header and Keyword Exceptions
# =============================================================================


class HeaderKeywordError(FitsToolsError):
    """Raised for errors related to FITS header keywords.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    keyword : str, optional
        The FITS keyword that caused the error.
    filepath : str or Path, optional
        Path to the FITS file with the problematic keyword.
    """

    def __init__(
        self,
        message: str,
        keyword: str | None = None,
        filepath: StrOrPath | None = None,
    ) -> None:
        self.keyword = keyword
        self.filepath = Path(filepath) if filepath is not None else None
        super().__init__(message)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        if self.keyword is not None:
            parts.append(f"keyword={self.keyword!r}")
        if self.filepath is not None:
            parts.append(f"filepath={self.filepath!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class SectionParsingError(FitsToolsError):
    """Raised when a FITS section string cannot be parsed.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    section_string : str
        The section string that failed to parse.
    keyword : str, optional
        The FITS keyword from which the section string was read.
    """

    def __init__(
        self,
        message: str,
        section_string: str,
        keyword: str | None = None,
    ) -> None:
        self.section_string = section_string
        self.keyword = keyword
        super().__init__(message)

    def __repr__(self) -> str:
        parts = [repr(self.args[0]), f"section_string={self.section_string!r}"]
        if self.keyword is not None:
            parts.append(f"keyword={self.keyword!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# Selection and Validation Exceptions
# =============================================================================


class SelectionError(FitsToolsError):
    """Raised for errors during file or HDU selection operations."""


class ValidationError(FitsToolsError):
    """Raised when a FITS file fails a validation check.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    filepath : str or Path, optional
        Path to the file that failed validation.
    validation_type : str, optional
        Short label identifying the type of validation that failed.
    errors : list of str, optional
        Individual validation error messages.

    Attributes
    ----------
    errors : list of str
        Individual validation error messages. Empty list if none provided.
    """

    def __init__(
        self,
        message: str,
        filepath: StrOrPath | None = None,
        validation_type: str | None = None,
        errors: list[str] | None = None,
    ) -> None:
        self.filepath = Path(filepath) if filepath is not None else None
        self.validation_type = validation_type
        self.errors: list[str] = list(errors) if errors is not None else []
        super().__init__(message)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        if self.filepath is not None:
            parts.append(f"filepath={self.filepath!r}")
        if self.validation_type is not None:
            parts.append(f"validation_type={self.validation_type!r}")
        if self.errors:
            parts.append(f"errors=[{len(self.errors)} item(s)]")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# Parameter and Operation Exceptions
# =============================================================================


class InvalidParameterError(FitsToolsError):
    """Raised when function parameters are invalid or incompatible.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    parameter : str, optional
        Name of the invalid parameter.
    value : Any, optional
        The invalid value that was supplied.
    expected : str, optional
        Description of the expected value or range.
    """

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        value: Any = None,
        expected: str | None = None,
    ) -> None:
        self.parameter = parameter
        self.value = value
        self.expected = expected
        super().__init__(message)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        if self.parameter is not None:
            parts.append(f"parameter={self.parameter!r}")
        if self.value is not None:
            parts.append(f"value={self.value!r}")
        if self.expected is not None:
            parts.append(f"expected={self.expected!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class UnsupportedOperationError(FitsToolsError):
    """Raised when an operation is not supported for the given data or context.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    operation : str
        Name of the unsupported operation.
    reason : str, optional
        Explanation of why the operation is unsupported.
    """

    def __init__(
        self,
        message: str,
        operation: str,
        reason: str | None = None,
    ) -> None:
        self.operation = operation
        self.reason = reason
        super().__init__(message)

    def __repr__(self) -> str:
        parts = [repr(self.args[0]), f"operation={self.operation!r}"]
        if self.reason is not None:
            parts.append(f"reason={self.reason!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# =============================================================================
# Master Frame Generation Exceptions
# =============================================================================


class MasterFrameGenerationError(FitsToolsError):
    """Base exception for master calibration frame generation failures.

    Parameters
    ----------
    message : str
        Human-readable description of the error.
    frame_type : str, optional
        Type of calibration frame being generated, e.g. ``'BIAS'``.
    """

    def __init__(self, message: str, frame_type: str | None = None) -> None:
        self.frame_type = frame_type
        super().__init__(message)

    def __repr__(self) -> str:
        base = f"{self.__class__.__name__}({self.args[0]!r}"
        if self.frame_type is not None:
            base += f", frame_type={self.frame_type!r}"
        return base + ")"


class NoInputFilesFoundError(MasterFrameGenerationError):
    """Raised when no suitable input files are found for master frame creation.

    Parameters
    ----------
    file_type : str
        Type of file being searched for, e.g. ``'BIAS'``.
    search_param : str
        The search path, pattern, or criterion used.
    message : str, optional
        Custom error message. A default is generated if not supplied.
    """

    def __init__(
        self,
        file_type: str,
        search_param: str,
        message: str | None = None,
    ) -> None:
        self.file_type = file_type
        self.search_param = search_param

        if message is None:
            message = f"No suitable {file_type} files found matching '{search_param}'."

        super().__init__(message, frame_type=file_type)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"file_type={self.file_type!r}, "
            f"search_param={self.search_param!r})"
        )


class MasterBiasCreationError(MasterFrameGenerationError):
    """Raised when master bias frame creation fails.

    Parameters
    ----------
    message : str, optional
        Custom error message. A default is generated if not supplied.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Failed to create master bias frame.", frame_type="BIAS")


class MasterDarkCreationError(MasterFrameGenerationError):
    """Raised when master dark frame creation fails.

    Parameters
    ----------
    message : str, optional
        Custom error message. A default is generated if not supplied.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Failed to create master dark frame.", frame_type="DARK")


class MasterFlatCreationError(MasterFrameGenerationError):
    """Raised when master flat field frame creation fails.

    Parameters
    ----------
    message : str, optional
        Custom error message. A default is generated if not supplied.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Failed to create master flat frame.", frame_type="FLAT")


class NormalizationError(MasterFlatCreationError):
    """Raised when master flat normalisation fails.

    Parameters
    ----------
    message : str, optional
        Custom error message. A default is generated if not supplied.
    mean_value : float, optional
        The mean signal value that caused the failure. Included in the
        default message when provided.

    Attributes
    ----------
    mean_value : float or None
        The mean signal value that caused the failure.
    """

    def __init__(
        self,
        message: str | None = None,
        mean_value: float | None = None,
    ) -> None:
        self.mean_value = mean_value

        if message is None:
            if mean_value is not None:
                message = (
                    f"Cannot normalise master flat: mean signal is {mean_value:.6g} "
                    "(must be positive)."
                )
            else:
                message = "Cannot normalise master flat: mean signal is zero or non-positive."

        super().__init__(message)

    def __repr__(self) -> str:
        if self.mean_value is not None:
            return f"{self.__class__.__name__}(mean_value={self.mean_value!r})"
        return f"{self.__class__.__name__}({self.args[0]!r})"


# =============================================================================
# Context Manager
# =============================================================================


@contextmanager
def fits_operation_context(
    operation: str,
    filepath: StrOrPath | None = None,
    reraise_as: type[FitsToolsError] = FitsToolsError,
) -> Generator[None, None, None]:
    """Context manager that converts non-fits_tools exceptions to fits_tools ones.

    Catches any exception that is not already a :class:`FitsToolsError`,
    wraps it in an instance of *reraise_as*, and re-raises with the original
    exception set as ``__cause__``.

    Parameters
    ----------
    operation : str
        Human-readable description of the operation being performed.
        Included in the error message on failure.
    filepath : str or Path, optional
        Path to the FITS file being operated on. Included in the error
        message when provided.
    reraise_as : type[FitsToolsError], optional
        The exception class to raise on failure. Must be
        :class:`FitsToolsError` or a subclass. Default is
        :class:`FitsToolsError`.

    Yields
    ------
    None

    Raises
    ------
    FitsToolsError
        Or an instance of *reraise_as* if a non-fits_tools exception is
        caught. Existing :class:`FitsToolsError` exceptions pass through
        unchanged.

    Examples
    --------
    ::

        with fits_operation_context("reading header", path, FileHandlingError):
            header = fits.getheader(path)
    """
    try:
        yield
    except FitsToolsError:
        raise
    except Exception as exc:
        msg = f"Error during {operation}"
        if filepath is not None:
            msg += f" on '{filepath}'"
        msg += f": {exc}"

        kwargs: dict[str, Any] = {}
        try:
            params = inspect.signature(reraise_as.__init__).parameters
            if filepath is not None:
                if "filepath" in params:
                    kwargs["filepath"] = filepath
                elif "path" in params:
                    kwargs["path"] = filepath
            if "operation" in params:
                kwargs["operation"] = operation
            if "wcs_operation" in params:
                kwargs["wcs_operation"] = operation
        except Exception:
            kwargs = {}

        try:
            raise reraise_as(msg, **kwargs) from exc
        except TypeError:
            raise reraise_as(msg) from exc


# =============================================================================
# Introspection Helper
# =============================================================================


def get_exception_hierarchy() -> str:
    """Return a formatted string showing the full exception hierarchy.

    Returns
    -------
    str
        Multi-line string representing the exception class tree.

    Notes
    -----
    This string duplicates the hierarchy shown in the module docstring.
    If the hierarchy changes, both locations must be updated.
    """
    return """FitsToolsError (base)
├── FileHandlingError
│   ├── InvalidHDUIndexError
│   ├── FileOverwriteError
│   └── HDUStructureError
├── ImageProcessingError
│   ├── DataShapeError
│   │   └── DimensionalityError
│   ├── WCSError
│   └── RegionError
├── HeaderKeywordError
├── SectionParsingError
├── SelectionError
├── ValidationError
├── InvalidParameterError
├── UnsupportedOperationError
└── MasterFrameGenerationError
    ├── NoInputFilesFoundError
    ├── MasterBiasCreationError
    ├── MasterDarkCreationError
    └── MasterFlatCreationError
        └── NormalizationError"""


# =============================================================================
# Self-Test / Demo
# =============================================================================

if __name__ == "__main__":
    import sys

    print(get_exception_hierarchy())
    print("\n" + "=" * 70)
    print("Exception Examples")
    print("=" * 70)

    def _demo_hdu_index() -> None:
        try:
            raise InvalidHDUIndexError("test.fits", 5, 3)
        except InvalidHDUIndexError as e:
            print(f"\n1. {type(e).__name__}")
            print(f"   msg:       {e}")
            print(f"   path:      {e.path}")
            print(f"   hdu_index: {e.hdu_index}")
            print(f"   max_hdu:   {e.max_hdu}")
            print(f"   repr:      {e!r}")

    def _demo_context_manager() -> None:
        print("\n2. fits_operation_context")
        try:
            with fits_operation_context("test operation", "test.fits", ImageProcessingError):
                raise ValueError("Something went wrong internally")
        except ImageProcessingError as e:
            print(f"   caught: {type(e).__name__}: {e}")
            print(f"   cause:  {e.__cause__!r}")

    _demo_hdu_index()
    _demo_context_manager()

    print("\n" + "=" * 70)
    print("All demos completed.")
    print("=" * 70)
    sys.exit(0)
