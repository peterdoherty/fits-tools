#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

Usage Examples
--------------
Catching the entire hierarchy::

    from fits_tools.exceptions import FitsToolsError
    try:
        result = some_fits_operation()
    except FitsToolsError as e:
        logger.error("fits_tools error: %s", e)

Catching and inspecting a specific exception::

    from fits_tools.exceptions import InvalidHDUIndexError
    try:
        data = get_data('image.fits', ext=99)
    except InvalidHDUIndexError as e:
        print(f"Requested HDU {e.hdu_index}, file has {e.max_hdu}")

Using the context manager to wrap third-party errors::

    from fits_tools.exceptions import fits_operation_context, ImageProcessingError
    with fits_operation_context('median combine', 'stack.fits', ImageProcessingError):
        result = np.median(cube, axis=0)   # any error becomes ImageProcessingError

Version History
---------------
0.1.0
    Initial release.
0.2.0
    Added ``ValidationError``, improved message handling, added ``__repr__``
    methods, consistent type hints, and better documentation.
0.3.0
    Added exceptions for edit.py operations (``InvalidParameterError``,
    ``FileOverwriteError``, ``DataShapeError``, ``DimensionalityError``,
    ``WCSError``, ``RegionError``, ``SectionParsingError``,
    ``HDUStructureError``, ``UnsupportedOperationError``), added context
    manager, improved hierarchy.
0.4.0
    Fixed ``from_exception`` return/raise semantics; tightened ``__repr__``
    guards to use ``is not None`` for shape tuples; general cleanup.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Type, Union

from .types import StrOrPath

__version__ = "0.4.0"
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

    Class Method
    ------------
    :meth:`from_exception` creates a new instance wrapping an existing
    exception.  Unlike the previous design, it *returns* the new exception
    and lets the caller raise it so that normal exception chaining and
    tracebacks work predictably::

        try:
            do_something()
        except OSError as e:
            raise FitsToolsError.from_exception(e, "Could not access file")
    """

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        message: Optional[str] = None,
        **kwargs: Any,
    ) -> "FitsToolsError":
        """Create an instance of this exception class wrapping *exc*.

        The returned exception has *exc* set as its ``__cause__`` so that
        Python's normal ``raise … from …`` chaining is preserved when the
        caller re-raises it.

        Parameters
        ----------
        exc:
            The original exception to wrap.
        message:
            Human-readable message for the new exception.  Defaults to
            ``str(exc)`` when not provided.
        **kwargs:
            Additional keyword arguments forwarded to the class constructor.
            Useful for subclasses that accept structured fields such as
            ``filepath`` or ``keyword``.

        Returns
        -------
        FitsToolsError
            A new instance of this class (or subclass).  **Not raised** —
            the caller is responsible for raising the return value so that
            normal ``raise … from …`` semantics apply.

        Example
        -------
        ::

            try:
                int("not-a-number")
            except ValueError as e:
                raise FitsToolsError.from_exception(e, "Invalid header value")
        """
        if message is None:
            message = str(exc)
        new_exc = cls(message, **kwargs)
        new_exc.__cause__ = exc
        return new_exc


# =============================================================================
# File Handling Exceptions
# =============================================================================

class FileHandlingError(FitsToolsError):
    """Raised for errors during FITS file reading, writing, or access.

    Covers file-not-found, permission errors, corrupted files, and similar
    I/O failures.

    Parameters
    ----------
    message:
        Human-readable description of the error.
    filepath:
        Path to the file that caused the error (optional).

    Attributes
    ----------
    filepath : Path or None
        Path to the problematic file.
    """

    def __init__(self, message: str, filepath: Optional[StrOrPath] = None) -> None:
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
    path:
        Path to the FITS file.
    hdu_index:
        The HDU index that was requested.
    max_hdu:
        The number of HDUs present in the file (so valid indices are
        ``0`` through ``max_hdu - 1``).
    message:
        Custom error message.  A default is generated if not supplied.

    Attributes
    ----------
    path : str or Path
        Path to the FITS file.
    hdu_index : int
        The requested (invalid) HDU index.
    max_hdu : int
        Total number of HDUs in the file.
    """

    def __init__(self, 
                 path: StrOrPath, 
                 hdu_index: int, 
                 max_hdu: int,
                message: Optional[str] = None,
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
    filepath:
        Path to the file that would be overwritten.
    message:
        Custom error message.  A default is generated if not supplied.

    Attributes
    ----------
    filepath : Path
        Path to the file that already exists.
    """

    def __init__(self, filepath: StrOrPath, message: Optional[str] = None) -> None:
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

    Examples include a missing PRIMARY HDU, invalid HDU ordering, or an
    operation that would produce a non-conformant FITS file.

    Parameters
    ----------
    message:
        Human-readable description of the structural problem.
    filepath:
        Path to the problematic file (optional).
    issue:
        Short description of the structural issue (optional).  Intended to
        carry a concise, machine-readable label such as
        ``'missing_primary_hdu'``.

    Attributes
    ----------
    issue : Path or None
        Short label describing the structural problem.
    """

    def __init__(
        self,
        message: str,
        filepath: Optional[StrOrPath] = None,
        issue: Optional[str] = None,
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

    Covers arithmetic, combining, statistical calculations, and similar
    data-manipulation failures.

    Parameters
    ----------
    message:
        Human-readable description of the error.
    operation:
        Name or short description of the failed operation (optional).

    Attributes
    ----------
    operation : str or None
        The operation that failed.
    """

    def __init__(self, message: str, operation: Optional[str] = None) -> None:
        self.operation = operation
        super().__init__(message)

    def __repr__(self) -> str:
        base = f"{self.__class__.__name__}({self.args[0]!r}"
        if self.operation is not None:
            base += f", operation={self.operation!r}"
        return base + ")"


class DataShapeError(ImageProcessingError):
    """Raised when array shapes are incompatible for an operation.

    Common situations: combining images with different sizes, applying a
    mask whose shape does not match the data, stacking unequal arrays.

    Parameters
    ----------
    message:
        Human-readable description of the mismatch.
    expected_shape:
        The shape that was required (optional).
    actual_shape:
        The shape that was encountered (optional).
    operation:
        The operation that failed (optional).

    Attributes
    ----------
    expected_shape : tuple or None
    actual_shape : tuple or None
    """

    def __init__(
        self,
        message: str,
        expected_shape: Optional[tuple] = None,
        actual_shape: Optional[tuple] = None,
        operation: Optional[str] = None,
    ) -> None:
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        super().__init__(message, operation=operation)

    def __repr__(self) -> str:
        parts = [repr(self.args[0])]
        # Use `is not None` so that (0, 0) or empty tuples are still reported.
        if self.expected_shape is not None:
            parts.append(f"expected_shape={self.expected_shape!r}")
        if self.actual_shape is not None:
            parts.append(f"actual_shape={self.actual_shape!r}")
        if self.operation is not None:
            parts.append(f"operation={self.operation!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class DimensionalityError(DataShapeError):
    """Raised when array dimensionality is wrong for an operation.

    For example: cropping a 1-D array, or applying a 2-D operation to a
    3-D data-cube without specifying how to handle the extra axis.

    Parameters
    ----------
    message:
        Human-readable description of the problem.
    ndim:
        Actual number of dimensions encountered (optional).
    required_ndim:
        Required number of dimensions.  May be an ``int`` (exactly) or a
        tuple of ints (acceptable values) (optional).
    operation:
        The operation that failed (optional).

    Attributes
    ----------
    ndim : int or None
    required_ndim : int, tuple, or None
    """

    def __init__(
        self,
        message: str,
        ndim: Optional[int] = None,
        required_ndim: Optional[Any] = None,
        operation: Optional[str] = None,
    ) -> None:
        self.ndim = ndim
        self.required_ndim = required_ndim
        super().__init__(
            message,
            expected_shape=None,
            actual_shape=None,
            operation=operation,
        )

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

    Covers missing WCS information, invalid WCS headers, and failures
    during WCS transformations or adjustments.

    Parameters
    ----------
    message:
        Human-readable description of the WCS problem.
    filepath:
        Path to the FITS file with WCS issues (optional).
    wcs_operation:
        The specific WCS operation that failed (optional).

    Attributes
    ----------
    filepath : Path or None
    wcs_operation : str or None
    """

    def __init__(
        self,
        message: str,
        filepath: Optional[StrOrPath] = None,
        wcs_operation: Optional[str] = None,
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

    Covers out-of-bounds regions, negative dimensions, zero-area regions,
    and regions that do not fit within the image boundaries.

    Parameters
    ----------
    message:
        Human-readable description of the problem.
    region:
        The invalid region specification as a tuple, e.g.
        ``(start_col, end_col, start_row, end_row)`` (optional).
    image_shape:
        Shape of the image array being accessed, as ``(rows, cols)``
        (optional).
    operation:
        The operation that failed (optional).

    Attributes
    ----------
    region : tuple or None
    image_shape : tuple or None
    """

    def __init__(
        self,
        message: str,
        region: Optional[tuple] = None,
        image_shape: Optional[tuple] = None,
        operation: Optional[str] = None,
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

    Covers missing required keywords, invalid keyword values, type
    mismatches, and malformed keyword content.

    Parameters
    ----------
    message:
        Human-readable description of the problem.
    keyword:
        The problematic keyword name (optional).
    filepath:
        Path to the FITS file (optional).

    Attributes
    ----------
    keyword : str or None
    filepath : Path or None
    """

    def __init__(
        self,
        message: str,
        keyword: Optional[str] = None,
        filepath: Optional[StrOrPath] = None,
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

    FITS sections such as ``[1:512,1:512]`` define image regions used by
    keywords like ``BIASSEC``, ``DATASEC``, and ``TRIMSEC``.  This
    exception is raised when the string does not conform to the expected
    format.

    Parameters
    ----------
    message:
        Human-readable description of the parse failure.
    section_string:
        The raw string that could not be parsed.
    keyword:
        The header keyword that contained the section string (optional).

    Attributes
    ----------
    section_string : str
        The unparseable section string.
    keyword : str or None
        Header keyword, e.g. ``'BIASSEC'``, ``'DATASEC'``.
    """

    def __init__(
        self,
        message: str,
        section_string: str,
        keyword: Optional[str] = None,
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
    """Raised for errors during file or HDU selection operations.

    Covers pattern-matching failures, empty result sets, and ambiguous
    selections where a unique result was required.
    """


class ValidationError(FitsToolsError):
    """Raised when a FITS file fails a validation check.

    Used by the validation framework when a file does not meet required
    specifications or quality criteria.

    Parameters
    ----------
    message:
        Human-readable summary of the validation failure.
    filepath:
        Path to the file that failed validation (optional).
    validation_type:
        Short label for the type of check that failed, e.g. ``'header'``,
        ``'data_range'``, ``'frame_type'`` (optional).
    errors:
        List of individual error messages for fine-grained diagnostics
        (optional).

    Attributes
    ----------
    filepath : Path or None
    validation_type : str or None
    errors : list[str]
        Always a list; never ``None``.
    """

    def __init__(
        self,
        message: str,
        filepath: Optional[StrOrPath] = None,
        validation_type: Optional[str] = None,
        errors: Optional[list] = None,
    ) -> None:
        self.filepath = Path(filepath) if filepath is not None else None
        self.validation_type = validation_type
        self.errors: list = errors if errors is not None else []
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

    Covers wrong types, out-of-range values, mutually exclusive parameters
    specified together, and similar contract violations.

    Parameters
    ----------
    message:
        Human-readable description of the problem.
    parameter:
        Name of the offending parameter (optional).
    value:
        The invalid value that was supplied (optional).
    expected:
        Description of what was expected, e.g. ``'positive integer'``
        or ``"one of ('full', 'data', 'bias')"`` (optional).

    Attributes
    ----------
    parameter : str or None
    value : Any
    expected : str or None
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Any = None,
        expected: Optional[str] = None,
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

    Examples: requesting an operation that requires a specific data type,
    HDU structure, or external library that is not present.

    Parameters
    ----------
    message:
        Human-readable description of the problem.
    operation:
        The name of the unsupported operation.
    reason:
        Explanation of why the operation is not supported (optional).

    Attributes
    ----------
    operation : str
    reason : str or None
    """

    def __init__(
        self,
        message: str,
        operation: str,
        reason: Optional[str] = None,
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

    Master frames are combined calibration exposures (bias, dark, flat)
    used in CCD/CMOS image calibration pipelines.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    frame_type:
        Type of master frame being generated, e.g. ``'BIAS'``, ``'DARK'``,
        ``'FLAT'`` (optional).

    Attributes
    ----------
    frame_type : str or None
    """

    def __init__(self, message: str, frame_type: Optional[str] = None) -> None:
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
    file_type:
        Type of files that were searched for, e.g. ``'BIAS'``.
    search_param:
        Directory path or glob pattern used for the search.
    message:
        Custom error message.  A default is generated if not supplied.

    Attributes
    ----------
    file_type : str
    search_param : str
    """

    def __init__(
        self,
        file_type: str,
        search_param: str,
        message: Optional[str] = None,
    ) -> None:
        self.file_type = file_type
        self.search_param = search_param

        if message is None:
            message = (
                f"No suitable {file_type} files found matching '{search_param}'."
            )

        super().__init__(message, frame_type=file_type)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"file_type={self.file_type!r}, "
            f"search_param={self.search_param!r})"
        )


class MasterBiasCreationError(MasterFrameGenerationError):
    """Raised when master bias frame creation fails.

    Master bias frames are produced by combining multiple zero-length
    exposures to characterise the detector's readout offset.

    Parameters
    ----------
    message:
        Error message.  Defaults to a standard description.
    """

    def __init__(self, message: Optional[str] = None) -> None:
        if message is None:
            message = "Failed to create master bias frame."
        super().__init__(message, frame_type="BIAS")


class MasterDarkCreationError(MasterFrameGenerationError):
    """Raised when master dark frame creation fails.

    Master dark frames are produced by combining multiple dark exposures
    to characterise the detector's thermally-generated dark current.

    Parameters
    ----------
    message:
        Error message.  Defaults to a standard description.
    """

    def __init__(self, message: Optional[str] = None) -> None:
        if message is None:
            message = "Failed to create master dark frame."
        super().__init__(message, frame_type="DARK")


class MasterFlatCreationError(MasterFrameGenerationError):
    """Raised when master flat field frame creation fails.

    Master flat frames are produced by combining multiple uniformly
    illuminated exposures to correct for pixel-to-pixel sensitivity
    variations.

    Parameters
    ----------
    message:
        Error message.  Defaults to a standard description.
    """

    def __init__(self, message: Optional[str] = None) -> None:
        if message is None:
            message = "Failed to create master flat frame."
        super().__init__(message, frame_type="FLAT")


class NormalizationError(MasterFlatCreationError):
    """Raised when master flat normalisation fails.

    Flat fields must be normalised to a reference level (typically 1.0 or
    the median signal) before use.  Normalisation fails when the mean or
    median signal is zero or non-positive, which would produce a
    division-by-zero or sign inversion.

    Parameters
    ----------
    message:
        Error message.  A default is generated using *mean_value* when
        not supplied.
    mean_value:
        The mean pixel value that caused the failure (optional).

    Attributes
    ----------
    mean_value : float or None
    """

    def __init__(
        self,
        message: Optional[str] = None,
        mean_value: Optional[float] = None,
    ) -> None:
        self.mean_value = mean_value

        if message is None:
            if mean_value is not None:
                message = (
                    f"Cannot normalise master flat: mean signal is {mean_value:.6g} "
                    "(must be positive)."
                )
            else:
                message = (
                    "Cannot normalise master flat: mean signal is zero or non-positive."
                )

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
    filepath: Optional[str] = None,
    reraise_as: Type[FitsToolsError] = FitsToolsError,
) -> Generator[None, None, None]:
    """Context manager that wraps non-fits_tools exceptions as fits_tools ones.

    Any exception that is *not* already a :class:`FitsToolsError` is caught,
    wrapped in *reraise_as* with an informative message, and re-raised with
    the original exception as its ``__cause__``.

    The wrapper inspects the signature of *reraise_as* and passes optional
    keyword arguments (``filepath``, ``path``, ``operation``,
    ``wcs_operation``) when the target constructor accepts them, providing
    structured context wherever possible.

    Parameters
    ----------
    operation:
        Short description of what is being attempted, e.g.
        ``'median combine'``, ``'cropping'``.
    filepath:
        Path to the file being processed (optional).
    reraise_as:
        The fits_tools exception class to wrap caught exceptions in.
        Defaults to :class:`FitsToolsError`.

    Raises
    ------
    FitsToolsError
        The wrapped exception (or a subclass thereof).

    Example
    -------
    ::

        with fits_operation_context('sigma clip', 'stack.fits', ImageProcessingError):
            result = sigma_clip(data)   # OSError, ValueError, etc. → ImageProcessingError
    """
    try:
        yield
    except FitsToolsError:
        raise  # Pass through; already one of ours.
    except Exception as exc:
        # Build the message.
        msg = f"Error during {operation}"
        if filepath:
            msg += f" on '{filepath}'"
        msg += f": {exc}"

        # Attempt to forward structured context fields to the constructor.
        kwargs: dict = {}
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
            # Fallback: the subclass constructor does not accept **kwargs.
            raise reraise_as(msg) from exc


# =============================================================================
# Introspection Helper
# =============================================================================

def get_exception_hierarchy() -> str:
    """Return a formatted string showing the full exception hierarchy.

    Returns
    -------
    str
        A tree-structured representation of all custom exception classes.

    Example
    -------
    ::

        >>> print(get_exception_hierarchy())
        FitsToolsError (base)
        ├── FileHandlingError
        ...
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

    demos = []

    # 1. InvalidHDUIndexError
    def _demo_hdu_index():
        try:
            raise InvalidHDUIndexError("test.fits", 5, 3)
        except InvalidHDUIndexError as e:
            print(f"\n1. {type(e).__name__}")
            print(f"   msg:       {e}")
            print(f"   path:      {e.path}")
            print(f"   hdu_index: {e.hdu_index}")
            print(f"   max_hdu:   {e.max_hdu}")
            print(f"   repr:      {e!r}")

    # 2. HeaderKeywordError
    def _demo_header_keyword():
        try:
            raise HeaderKeywordError(
                "Missing required keyword",
                keyword="EXPTIME",
                filepath="image.fits",
            )
        except HeaderKeywordError as e:
            print(f"\n2. {type(e).__name__}")
            print(f"   msg:      {e}")
            print(f"   keyword:  {e.keyword}")
            print(f"   filepath: {e.filepath}")
            print(f"   repr:     {e!r}")

    # 3. ValidationError
    def _demo_validation():
        try:
            raise ValidationError(
                "File failed validation",
                filepath="bad_data.fits",
                validation_type="data_range",
                errors=["Values exceed maximum", "Contains NaN values"],
            )
        except ValidationError as e:
            print(f"\n3. {type(e).__name__}")
            print(f"   msg:             {e}")
            print(f"   filepath:        {e.filepath}")
            print(f"   validation_type: {e.validation_type}")
            print(f"   errors:          {e.errors}")
            print(f"   repr:            {e!r}")

    # 4. NormalizationError
    def _demo_normalization():
        try:
            raise NormalizationError(mean_value=0.0)
        except NormalizationError as e:
            print(f"\n4. {type(e).__name__}")
            print(f"   msg:        {e}")
            print(f"   mean_value: {e.mean_value}")
            print(f"   repr:       {e!r}")

    # 5. DataShapeError — (0, 0) shape should still appear in repr
    def _demo_data_shape():
        try:
            raise DataShapeError(
                "Shape mismatch",
                expected_shape=(0, 0),
                actual_shape=(100, 100),
                operation="stack",
            )
        except DataShapeError as e:
            print(f"\n5. {type(e).__name__} (zero-size expected_shape still shown)")
            print(f"   msg:            {e}")
            print(f"   expected_shape: {e.expected_shape}")
            print(f"   repr:           {e!r}")

    # 6. from_exception — returns, does not raise; caller raises
    def _demo_from_exception():
        print(f"\n6. FitsToolsError.from_exception (caller raises)")
        try:
            try:
                int("bad")
            except ValueError as orig:
                raise FitsToolsError.from_exception(orig, "Invalid header value")
        except FitsToolsError as e:
            print(f"   msg:    {e}")
            print(f"   cause:  {e.__cause__}")

    # 7. Context manager
    def _demo_context_manager():
        print(f"\n7. fits_operation_context")
        try:
            with fits_operation_context("test operation", "test.fits", ImageProcessingError):
                raise ValueError("Something went wrong internally")
        except ImageProcessingError as e:
            print(f"   caught: {type(e).__name__}: {e}")
            print(f"   cause:  {e.__cause__!r}")

    # 8. Catching all via base class
    def _demo_base_catch():
        print(f"\n8. Catching all via FitsToolsError")
        for err in [
            InvalidHDUIndexError("f.fits", 10, 5),
            HeaderKeywordError("Missing", keyword="EXPTIME"),
            RegionError("Bad region", region=(0, 0, -10, -10)),
        ]:
            try:
                raise err
            except FitsToolsError as e:
                print(f"   caught {type(e).__name__}: {e}")

    for demo in [
        _demo_hdu_index,
        _demo_header_keyword,
        _demo_validation,
        _demo_normalization,
        _demo_data_shape,
        _demo_from_exception,
        _demo_context_manager,
        _demo_base_catch,
    ]:
        demo()

    print("\n" + "=" * 70)
    print("All demos completed.")
    print("=" * 70)
    sys.exit(0)
