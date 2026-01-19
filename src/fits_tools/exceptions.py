#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom exception classes for the fits_tools package.

This module defines a hierarchy of exceptions for handling errors that occur
during FITS file operations, image processing, validation, and calibration
frame generation for CCD/CMOS imager test data.

Exception Hierarchy:
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

Usage Example:
    >>> from fits_tools.exceptions import InvalidHDUIndexError, RegionError
    >>> try:
    ...     # Some FITS operation
    ...     raise InvalidHDUIndexError('test.fits', 5, 3)
    ... except InvalidHDUIndexError as e:
    ...     print(f"Error: {e}")
    ...     print(f"File: {e.path}, Requested: {e.hdu_index}, Max: {e.max_hdu}")
    
    Context manager usage:
    >>> from fits_tools.exceptions import fits_operation_context, ImageProcessingError
    >>> with fits_operation_context('cropping', 'image.fits', ImageProcessingError):
    ...     # Perform operations; errors are wrapped automatically
    ...     pass

Version History:
    0.1.0 - Initial release
    0.2.0 - Added ValidationError, improved message handling, added __repr__ methods,
            consistent type hints, and better documentation
    0.3.0 - Added exceptions for edit.py operations (InvalidParameterError,
            FileOverwriteError, DataShapeError, DimensionalityError, WCSError,
            RegionError, SectionParsingError, HDUStructureError,
            UnsupportedOperationError), added context manager, improved hierarchy
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional, Type
import inspect

__version__ = "0.3.0"


# ============================================================================
# Base Exception
# ============================================================================

class FitsToolsError(Exception):
    """
    Base exception for fits_tools package.
    
    All custom exceptions in the fits_tools package inherit from this class,
    allowing users to catch all package-specific exceptions with a single
    except clause.
    
    Example:
        >>> try:
        ...     # fits_tools operations
        ...     pass
        ... except FitsToolsError as e:
        ...     print(f"A fits_tools error occurred: {e}")
    """
    
    @classmethod
    def from_exception(
        cls, 
        exc: Exception, 
        message: Optional[str] = None,
        **kwargs
    ) -> 'FitsToolsError':
        """
        Create a FitsToolsError from another exception with chaining.
        
        Args:
            exc: Original exception.
            message: Optional override message. If None, uses str(exc).
            **kwargs: Additional attributes for the exception.
        
        Returns:
            Instance of this exception class with original exception chained.
        
        Example:
            >>> try:
            ...     int("invalid")
            ... except ValueError as e:
            ...     raise FitsToolsError.from_exception(e, "Invalid value")
        """
        if message is None:
            message = str(exc)
        new_exc = cls(message, **kwargs)
        raise new_exc from exc


# ============================================================================
# File Handling Exceptions
# ============================================================================

class FileHandlingError(FitsToolsError):
    """
    Raised for errors during FITS file reading, writing, or access.
    
    This includes file not found, permission errors, corrupted files, etc.
    
    Attributes:
        filepath (str, optional): Path to the file that caused the error.
    """
    
    def __init__(self, message: str, filepath: Optional[str] = None):
        """
        Initialize FileHandlingError.
        
        Args:
            message: Error message.
            filepath: Path to the problematic file.
        """
        self.filepath = filepath
        super().__init__(message)
    
    def __repr__(self) -> str:
        if self.filepath:
            return f"{self.__class__.__name__}('{self.args[0]}', filepath='{self.filepath}')"
        return f"{self.__class__.__name__}('{self.args[0]}')"


class InvalidHDUIndexError(FileHandlingError):
    """
    Raised when a specified FITS HDU index is out of bounds or invalid.
    
    Attributes:
        path (str): Path to the FITS file.
        hdu_index (int): The invalid HDU index that was requested.
        max_hdu (int): The maximum valid HDU index (number of HDUs in file).
    """
    
    def __init__(
        self, 
        path: str, 
        hdu_index: int, 
        max_hdu: int, 
        message: Optional[str] = None
    ):
        """
        Initialize InvalidHDUIndexError.
        
        Args:
            path: Path to the FITS file.
            hdu_index: The requested HDU index.
            max_hdu: The actual number of HDUs in the file.
            message: Custom error message. If None, a default message is generated.
        """
        self.path = path
        self.hdu_index = hdu_index
        self.max_hdu = max_hdu
        
        if message is None:
            message = (
                f"HDU index {hdu_index} is out of bounds for FITS file '{path}'. "
                f"File has {max_hdu} HDUs (valid indices: 0-{max_hdu-1})."
            )
        
        super().__init__(message, filepath=path)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(path='{self.path}', "
            f"hdu_index={self.hdu_index}, max_hdu={self.max_hdu})"
        )


class FileOverwriteError(FileHandlingError):
    """
    Raised when attempting to overwrite an existing file without permission.
    
    Attributes:
        filepath (str): Path to the file that would be overwritten.
    """
    
    def __init__(self, filepath: str, message: Optional[str] = None):
        """
        Initialize FileOverwriteError.
        
        Args:
            filepath: Path to the file that would be overwritten.
            message: Custom error message. If None, a default message is generated.
        """
        if message is None:
            message = (
                f"File '{filepath}' already exists. "
                f"Use --overwrite flag to allow overwriting."
            )
        super().__init__(message, filepath=filepath)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath='{self.filepath}')"


class HDUStructureError(FileHandlingError):
    """
    Raised when FITS HDU structure is invalid or violates FITS standard.
    
    For example, missing PRIMARY HDU, invalid HDU ordering, or attempting
    operations that would create non-standard FITS files.
    
    Attributes:
        issue (str, optional): Description of the structural issue.
    """
    
    def __init__(
        self,
        message: str,
        filepath: Optional[str] = None,
        issue: Optional[str] = None
    ):
        """
        Initialize HDUStructureError.
        
        Args:
            message: Error message.
            filepath: Path to the problematic file.
            issue: Description of the structural problem.
        """
        self.issue = issue
        super().__init__(message, filepath=filepath)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'"]
        if self.filepath:
            parts.append(f"filepath='{self.filepath}'")
        if self.issue:
            parts.append(f"issue='{self.issue}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# ============================================================================
# Processing and Data Exceptions
# ============================================================================

class ImageProcessingError(FitsToolsError):
    """
    Raised for errors during image processing operations.
    
    This includes arithmetic operations, combining images, statistical
    calculations, etc.
    
    Attributes:
        operation (str, optional): Description of the operation that failed.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None):
        """
        Initialize ImageProcessingError.
        
        Args:
            message: Error message.
            operation: Name/description of the failed operation.
        """
        self.operation = operation
        super().__init__(message)
    
    def __repr__(self) -> str:
        if self.operation:
            return f"{self.__class__.__name__}('{self.args[0]}', operation='{self.operation}')"
        return f"{self.__class__.__name__}('{self.args[0]}')"


class DataShapeError(ImageProcessingError):
    """
    Raised when array shapes are incompatible for an operation.
    
    This is used when combining images with different shapes, applying masks
    that don't match, etc.
    
    Attributes:
        expected_shape (tuple, optional): Expected array shape.
        actual_shape (tuple, optional): Actual array shape.
        operation (str, optional): Operation that failed.
    """
    
    def __init__(
        self, 
        message: str,
        expected_shape: Optional[tuple] = None,
        actual_shape: Optional[tuple] = None,
        operation: Optional[str] = None
    ):
        """
        Initialize DataShapeError.
        
        Args:
            message: Error message.
            expected_shape: Expected shape of the array.
            actual_shape: Actual shape encountered.
            operation: Operation being performed.
        """
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        super().__init__(message, operation=operation)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'"]
        if self.expected_shape:
            parts.append(f"expected_shape={self.expected_shape}")
        if self.actual_shape:
            parts.append(f"actual_shape={self.actual_shape}")
        if self.operation:
            parts.append(f"operation='{self.operation}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class DimensionalityError(DataShapeError):
    """
    Raised when array dimensionality is invalid for an operation.
    
    For example, trying to crop a 1D array, or attempting 2D operations
    on 3D datacubes without specifying how to handle the extra dimensions.
    
    Attributes:
        ndim (int, optional): Actual number of dimensions.
        required_ndim (int or tuple, optional): Required number of dimensions.
    """
    
    def __init__(
        self,
        message: str,
        ndim: Optional[int] = None,
        required_ndim: Optional[int] = None,
        operation: Optional[str] = None
    ):
        """
        Initialize DimensionalityError.
        
        Args:
            message: Error message.
            ndim: Actual dimensionality encountered.
            required_ndim: Required dimensionality.
            operation: Operation being performed.
        """
        self.ndim = ndim
        self.required_ndim = required_ndim
        super().__init__(
            message,
            actual_shape=None,
            expected_shape=None,
            operation=operation
        )
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'"]
        if self.ndim is not None:
            parts.append(f"ndim={self.ndim}")
        if self.required_ndim is not None:
            parts.append(f"required_ndim={self.required_ndim}")
        if self.operation:
            parts.append(f"operation='{self.operation}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class WCSError(ImageProcessingError):
    """
    Raised for WCS (World Coordinate System) related errors.
    
    This includes missing WCS information, invalid WCS headers, or failures
    during WCS transformations and adjustments.
    
    Attributes:
        filepath (str, optional): Path to the FITS file.
        wcs_operation (str, optional): WCS operation that failed.
    """
    
    def __init__(
        self,
        message: str,
        filepath: Optional[str] = None,
        wcs_operation: Optional[str] = None
    ):
        """
        Initialize WCSError.
        
        Args:
            message: Error message.
            filepath: Path to the file with WCS issues.
            wcs_operation: Specific WCS operation that failed.
        """
        self.filepath = filepath
        self.wcs_operation = wcs_operation
        super().__init__(message, operation=wcs_operation)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'"]
        if self.filepath:
            parts.append(f"filepath='{self.filepath}'")
        if self.wcs_operation:
            parts.append(f"wcs_operation='{self.wcs_operation}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class RegionError(ImageProcessingError):
    """
    Raised when a region (ROI, crop box, section) is invalid.
    
    This includes out-of-bounds regions, negative dimensions, or regions
    that don't fit within the image boundaries.
    
    Attributes:
        region (tuple, optional): The region specification (x0, y0, width, height).
        image_shape (tuple, optional): Shape of the image being accessed.
    """
    
    def __init__(
        self,
        message: str,
        region: Optional[tuple] = None,
        image_shape: Optional[tuple] = None,
        operation: Optional[str] = None
    ):
        """
        Initialize RegionError.
        
        Args:
            message: Error message.
            region: Region specification that caused the error.
            image_shape: Shape of the image.
            operation: Operation being performed.
        """
        self.region = region
        self.image_shape = image_shape
        super().__init__(message, operation=operation)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'"]
        if self.region:
            parts.append(f"region={self.region}")
        if self.image_shape:
            parts.append(f"image_shape={self.image_shape}")
        if self.operation:
            parts.append(f"operation='{self.operation}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# ============================================================================
# Header and Keyword Exceptions
# ============================================================================

class HeaderKeywordError(FitsToolsError):
    """
    Raised for errors related to FITS header keywords.
    
    This includes missing required keywords, invalid keyword values,
    type mismatches, etc.
    
    Attributes:
        keyword (str, optional): The problematic keyword name.
        filepath (str, optional): Path to the FITS file.
    """
    
    def __init__(
        self, 
        message: str, 
        keyword: Optional[str] = None, 
        filepath: Optional[str] = None
    ):
        """
        Initialize HeaderKeywordError.
        
        Args:
            message: Error message.
            keyword: The keyword that caused the error.
            filepath: Path to the FITS file.
        """
        self.keyword = keyword
        self.filepath = filepath
        super().__init__(message)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'"]
        if self.keyword:
            parts.append(f"keyword='{self.keyword}'")
        if self.filepath:
            parts.append(f"filepath='{self.filepath}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class SectionParsingError(FitsToolsError):
    """
    Raised when FITS section string parsing fails.
    
    FITS sections like '[100:200,50:150]' define image regions (BIASSEC,
    DATASEC, TRIMSEC, etc.). This error is raised when the format is invalid.
    
    Attributes:
        section_string (str): The section string that failed to parse.
        keyword (str, optional): The keyword containing the section.
    """
    
    def __init__(
        self,
        message: str,
        section_string: str,
        keyword: Optional[str] = None
    ):
        """
        Initialize SectionParsingError.
        
        Args:
            message: Error message.
            section_string: The unparseable section string.
            keyword: Header keyword (e.g., 'BIASSEC', 'DATASEC').
        """
        self.section_string = section_string
        self.keyword = keyword
        super().__init__(message)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'", f"section_string='{self.section_string}'"]
        if self.keyword:
            parts.append(f"keyword='{self.keyword}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# ============================================================================
# Selection and Validation Exceptions
# ============================================================================

class SelectionError(FitsToolsError):
    """
    Raised for errors during file selection operations.
    
    This includes pattern matching failures, no files found, ambiguous
    selections, etc.
    """
    pass


class ValidationError(FitsToolsError):
    """
    Raised when FITS file validation fails.
    
    This exception is used by the validation framework when a FITS file
    does not meet required specifications or quality criteria.
    
    Attributes:
        filepath (str, optional): Path to the file that failed validation.
        validation_type (str, optional): Type of validation that failed.
        errors (list, optional): List of specific validation errors.
    """
    
    def __init__(
        self, 
        message: str, 
        filepath: Optional[str] = None,
        validation_type: Optional[str] = None,
        errors: Optional[list] = None
    ):
        """
        Initialize ValidationError.
        
        Args:
            message: Error message.
            filepath: Path to the file that failed validation.
            validation_type: Type of validation (e.g., 'header', 'data').
            errors: List of specific validation error messages.
        """
        self.filepath = filepath
        self.validation_type = validation_type
        self.errors = errors or []
        super().__init__(message)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'"]
        if self.filepath:
            parts.append(f"filepath='{self.filepath}'")
        if self.validation_type:
            parts.append(f"validation_type='{self.validation_type}'")
        if self.errors:
            parts.append(f"errors={len(self.errors)} items")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# ============================================================================
# Parameter and Operation Exceptions
# ============================================================================

class InvalidParameterError(FitsToolsError):
    """
    Raised when function parameters are invalid or incompatible.
    
    This includes wrong types, out-of-range values, mutually exclusive
    parameters specified together, etc.
    
    Attributes:
        parameter (str, optional): Name of the invalid parameter.
        value: The invalid value provided.
        expected (str, optional): Description of expected value/type.
    """
    
    def __init__(
        self, 
        message: str, 
        parameter: Optional[str] = None,
        value: Any = None,
        expected: Optional[str] = None
    ):
        """
        Initialize InvalidParameterError.
        
        Args:
            message: Error message.
            parameter: Name of the parameter.
            value: The invalid value.
            expected: Description of what was expected.
        """
        self.parameter = parameter
        self.value = value
        self.expected = expected
        super().__init__(message)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'"]
        if self.parameter:
            parts.append(f"parameter='{self.parameter}'")
        if self.value is not None:
            parts.append(f"value={self.value!r}")
        if self.expected:
            parts.append(f"expected='{self.expected}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class UnsupportedOperationError(FitsToolsError):
    """
    Raised when an operation is not supported for the given data or context.
    
    For example, attempting operations that require specific data types,
    structures, or conditions that are not present.
    
    Attributes:
        operation (str): The unsupported operation.
        reason (str, optional): Why the operation is not supported.
    """
    
    def __init__(
        self,
        message: str,
        operation: str,
        reason: Optional[str] = None
    ):
        """
        Initialize UnsupportedOperationError.
        
        Args:
            message: Error message.
            operation: The operation that is not supported.
            reason: Why it's not supported.
        """
        self.operation = operation
        self.reason = reason
        super().__init__(message)
    
    def __repr__(self) -> str:
        parts = [f"'{self.args[0]}'", f"operation='{self.operation}'"]
        if self.reason:
            parts.append(f"reason='{self.reason}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


# ============================================================================
# Master Frame Generation Exceptions
# ============================================================================

class MasterFrameGenerationError(FitsToolsError):
    """
    Base exception for master frame generation failures.
    
    Master frames are combined calibration frames (bias, dark, flat) used
    in image calibration pipelines.
    
    Attributes:
        frame_type (str, optional): Type of master frame (BIAS, DARK, FLAT).
    """
    
    def __init__(self, message: str, frame_type: Optional[str] = None):
        """
        Initialize MasterFrameGenerationError.
        
        Args:
            message: Error message.
            frame_type: Type of frame being generated.
        """
        self.frame_type = frame_type
        super().__init__(message)
    
    def __repr__(self) -> str:
        if self.frame_type:
            return f"{self.__class__.__name__}('{self.args[0]}', frame_type='{self.frame_type}')"
        return f"{self.__class__.__name__}('{self.args[0]}')"


class NoInputFilesFoundError(MasterFrameGenerationError):
    """
    Raised when no suitable input files are found for master frame creation.
    
    Attributes:
        file_type (str): Type of files being searched for.
        search_param (str): Directory or pattern used for search.
    """
    
    def __init__(
        self, 
        file_type: str, 
        search_param: str, 
        message: Optional[str] = None
    ):
        """
        Initialize NoInputFilesFoundError.
        
        Args:
            file_type: Type of files (e.g., 'BIAS', 'DARK', 'FLAT').
            search_param: Search location or pattern.
            message: Custom error message.
        """
        self.file_type = file_type
        self.search_param = search_param
        
        if message is None:
            message = f"No suitable {file_type} files found in '{search_param}'."
        
        super().__init__(message, frame_type=file_type)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(file_type='{self.file_type}', "
            f"search_param='{self.search_param}')"
        )


class MasterBiasCreationError(MasterFrameGenerationError):
    """
    Raised when master bias creation fails.
    
    Master bias frames are created by combining multiple bias exposures
    to measure the readout bias level of the detector.
    """
    
    def __init__(self, message: Optional[str] = None):
        """
        Initialize MasterBiasCreationError.
        
        Args:
            message: Error message. Defaults to standard message.
        """
        if message is None:
            message = "Failed to create master bias image."
        super().__init__(message, frame_type='BIAS')


class MasterDarkCreationError(MasterFrameGenerationError):
    """
    Raised when master dark creation fails.
    
    Master dark frames are created by combining multiple dark exposures
    to measure the dark current of the detector.
    """
    
    def __init__(self, message: Optional[str] = None):
        """
        Initialize MasterDarkCreationError.
        
        Args:
            message: Error message. Defaults to standard message.
        """
        if message is None:
            message = "Failed to create master dark image."
        super().__init__(message, frame_type='DARK')


class MasterFlatCreationError(MasterFrameGenerationError):
    """
    Raised when master flat creation fails.
    
    Master flat frames are created by combining multiple flat field exposures
    to correct for pixel-to-pixel sensitivity variations.
    """
    
    def __init__(self, message: Optional[str] = None):
        """
        Initialize MasterFlatCreationError.
        
        Args:
            message: Error message. Defaults to standard message.
        """
        if message is None:
            message = "Failed to create master flat image."
        super().__init__(message, frame_type='FLAT')


class NormalizationError(MasterFlatCreationError):
    """
    Raised when master flat normalization fails.
    
    Flat fields must be normalized to a mean value (typically 1.0) before
    use. This fails if the mean signal is zero or non-positive.
    
    Attributes:
        mean_value (float, optional): The problematic mean value.
    """
    
    def __init__(self, message: Optional[str] = None, mean_value: Optional[float] = None):
        """
        Initialize NormalizationError.
        
        Args:
            message: Error message.
            mean_value: The mean value that caused the error.
        """
        self.mean_value = mean_value
        
        if message is None:
            if mean_value is not None:
                message = (
                    f"Cannot normalize master flat: mean signal is {mean_value} "
                    f"(must be positive)."
                )
            else:
                message = "Cannot normalize master flat: mean signal is zero or non-positive."
        
        super().__init__(message)
    
    def __repr__(self) -> str:
        if self.mean_value is not None:
            return f"{self.__class__.__name__}(mean_value={self.mean_value})"
        return f"{self.__class__.__name__}('{self.args[0]}')"


# ============================================================================
# Utility Functions
# ============================================================================

@contextmanager
def fits_operation_context(
    operation: str,
    filepath: Optional[str] = None,
    reraise_as: Type[FitsToolsError] = FitsToolsError
) -> Generator[None, None, None]:
    """Context manager for wrapping FITS operations with better error messages.

    This catches non-:class:`~fits_tools.exceptions.FitsToolsError` exceptions and
    re-raises them as *fits_tools* exceptions with additional context.

    The wrapper attempts to preserve context by passing optional keyword
    arguments to the target exception class when supported (e.g. ``filepath``,
    ``path``, ``operation``, or ``wcs_operation``).

    Args:
        operation: Description of the operation being performed.
        filepath: Path to the file being processed (optional).
        reraise_as: Exception class to wrap caught exceptions in.

    Raises:
        FitsToolsError: Wrapped or original exception.
    """
    try:
        yield
    except FitsToolsError:
        # Re-raise our own exceptions as-is
        raise
    except Exception as e:
        msg = f"Error during {operation}"
        if filepath:
            msg += f" on file '{filepath}'"
        msg += f": {e}"

        # Best-effort: pass context fields if the exception supports them.
        kwargs = {}
        try:
            sig = inspect.signature(reraise_as.__init__)
            params = sig.parameters

            if filepath is not None:
                if "filepath" in params:
                    kwargs["filepath"] = filepath
                elif "path" in params:
                    kwargs["path"] = filepath

            if "operation" in params:
                kwargs["operation"] = operation

            # Some exceptions use a more specific field name
            if "wcs_operation" in params:
                kwargs["wcs_operation"] = operation
        except Exception:
            kwargs = {}

        try:
            raise reraise_as(msg, **kwargs) from e
        except TypeError:
            # Fallback for unexpected signatures
            raise reraise_as(msg) from e


def get_exception_hierarchy() -> str:
    """
    Return a string representation of the exception hierarchy.
    
    Returns:
        Formatted exception hierarchy tree.
    
    Example:
        >>> print(get_exception_hierarchy())
        Exception Hierarchy for fits_tools:
        
        FitsToolsError (base)
        ├── FileHandlingError
        │   ├── InvalidHDUIndexError
        ...
    """
    hierarchy = """
Exception Hierarchy for fits_tools:

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
    return hierarchy.strip()


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Base exception
    'FitsToolsError',
    
    # File handling
    'FileHandlingError',
    'InvalidHDUIndexError',
    'FileOverwriteError',
    'HDUStructureError',
    
    # Processing and data
    'ImageProcessingError',
    'DataShapeError',
    'DimensionalityError',
    'WCSError',
    'RegionError',
    
    # Headers and keywords
    'HeaderKeywordError',
    'SectionParsingError',
    
    # Selection and validation
    'SelectionError',
    'ValidationError',
    
    # Parameters and operations
    'InvalidParameterError',
    'UnsupportedOperationError',
    
    # Master frame generation
    'MasterFrameGenerationError',
    'NoInputFilesFoundError',
    'MasterBiasCreationError',
    'MasterDarkCreationError',
    'MasterFlatCreationError',
    'NormalizationError',
    
    # Utilities
    'get_exception_hierarchy',
    'fits_operation_context',
]


# ============================================================================
# Demo / Self-Test
# ============================================================================

if __name__ == "__main__":
    # Demonstrate exception usage
    print(get_exception_hierarchy())
    print("\n" + "=" * 70)
    print("Exception Examples:")
    print("=" * 70)
    
    # Example 1: InvalidHDUIndexError
    print("\n1. InvalidHDUIndexError:")
    try:
        raise InvalidHDUIndexError('test.fits', 5, 3)
    except InvalidHDUIndexError as e:
        print(f"   Message: {e}")
        print(f"   File: {e.path}")
        print(f"   Requested HDU: {e.hdu_index}")
        print(f"   Max HDU: {e.max_hdu}")
        print(f"   repr: {repr(e)}")
    
    # Example 2: HeaderKeywordError
    print("\n2. HeaderKeywordError:")
    try:
        raise HeaderKeywordError(
            "Missing required keyword", 
            keyword='EXPTIME', 
            filepath='image.fits'
        )
    except HeaderKeywordError as e:
        print(f"   Message: {e}")
        print(f"   Keyword: {e.keyword}")
        print(f"   File: {e.filepath}")
        print(f"   repr: {repr(e)}")
    
    # Example 3: ValidationError
    print("\n3. ValidationError:")
    try:
        raise ValidationError(
            "File failed validation",
            filepath='bad_data.fits',
            validation_type='data_range',
            errors=['Values exceed maximum', 'Contains NaN values']
        )
    except ValidationError as e:
        print(f"   Message: {e}")
        print(f"   File: {e.filepath}")
        print(f"   Type: {e.validation_type}")
        print(f"   Errors: {e.errors}")
        print(f"   repr: {repr(e)}")
    
    # Example 4: NormalizationError
    print("\n4. NormalizationError:")
    try:
        raise NormalizationError(mean_value=0.0)
    except NormalizationError as e:
        print(f"   Message: {e}")
        print(f"   Mean: {e.mean_value}")
        print(f"   repr: {repr(e)}")
    
    # Example 5: InvalidParameterError
    print("\n5. InvalidParameterError:")
    try:
        raise InvalidParameterError(
            "Width must be positive",
            parameter='width',
            value=-100,
            expected='positive integer'
        )
    except InvalidParameterError as e:
        print(f"   Message: {e}")
        print(f"   Parameter: {e.parameter}")
        print(f"   Value: {e.value}")
        print(f"   Expected: {e.expected}")
        print(f"   repr: {repr(e)}")
    
    # Example 6: RegionError
    print("\n6. RegionError:")
    try:
        raise RegionError(
            "Crop region out of bounds",
            region=(100, 200, 512, 512),
            image_shape=(256, 256),
            operation='crop'
        )
    except RegionError as e:
        print(f"   Message: {e}")
        print(f"   Region: {e.region}")
        print(f"   Image shape: {e.image_shape}")
        print(f"   Operation: {e.operation}")
        print(f"   repr: {repr(e)}")
    
    # Example 7: DimensionalityError
    print("\n7. DimensionalityError:")
    try:
        raise DimensionalityError(
            "Cannot crop 1D array",
            ndim=1,
            required_ndim=2,
            operation='crop'
        )
    except DimensionalityError as e:
        print(f"   Message: {e}")
        print(f"   Actual dimensions: {e.ndim}")
        print(f"   Required dimensions: {e.required_ndim}")
        print(f"   repr: {repr(e)}")
    
    # Example 8: FileOverwriteError
    print("\n8. FileOverwriteError:")
    try:
        raise FileOverwriteError('output.fits')
    except FileOverwriteError as e:
        print(f"   Message: {e}")
        print(f"   File: {e.filepath}")
        print(f"   repr: {repr(e)}")
    
    # Example 9: HDUStructureError
    print("\n9. HDUStructureError:")
    try:
        raise HDUStructureError(
            "Cannot remove PRIMARY HDU",
            filepath='test.fits',
            issue='PRIMARY HDU required by FITS standard'
        )
    except HDUStructureError as e:
        print(f"   Message: {e}")
        print(f"   File: {e.filepath}")
        print(f"   Issue: {e.issue}")
        print(f"   repr: {repr(e)}")
    
    # Example 10: Context manager usage
    print("\n10. fits_operation_context usage:")
    try:
        with fits_operation_context('test operation', 'test.fits', ImageProcessingError):
            # Simulate an error
            raise ValueError("Something went wrong")
    except ImageProcessingError as e:
        print(f"   Caught and wrapped: {e}")
        print(f"   Original cause: {e.__cause__}")
    
    # Example 11: Catching all package exceptions
    print("\n11. Catching all FitsToolsError exceptions:")
    errors_to_test = [
        InvalidHDUIndexError('file.fits', 10, 5),
        HeaderKeywordError("Missing keyword", keyword='EXPTIME'),
        RegionError("Invalid region", region=(0, 0, -10, -10)),
    ]
    
    for err in errors_to_test:
        try:
            raise err
        except FitsToolsError as e:
            print(f"   Caught {type(e).__name__}: {e}")
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)

    
