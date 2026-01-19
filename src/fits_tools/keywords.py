#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for managing FITS header keywords.

This module provides utilities to check for, get, set, and delete FITS header
keywords in individual FITS files or in batches across multiple files.
It supports both primary HDUs and specified extensions.

Features:
    - Check for keyword existence
    - Get/set/delete individual keywords
    - Batch operations across multiple files
    - Support for filtering files by header criteria
    - Comprehensive error handling

Version History:
    0.1.0 - Initial release
    0.2.0 - Refactored to use fits_tools modules, improved type hints,
            added doctest examples, removed unused imports
"""

import argparse
import doctest
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

from astropy.io import fits

from fits_tools.constants import (
    FITS_KEY_IMGTYPE,
    FITS_KEY_EXPTIME,
    FITS_KEY_OBSERVER,
)

from fits_tools.exceptions import (
    FileHandlingError,
    HeaderKeywordError,
    InvalidHDUIndexError,
    SelectionError,
    FitsToolsError,
)

from fits_tools.utils import (
    is_fits_file,
)

# Module logger
logger = logging.getLogger(__name__)

def _is_image_like_hdu(hdu: fits.hdu.base.ExtensionHDU) -> bool:
    """Return True if the HDU looks like an image HDU without reading pixel data.

    This avoids triggering decompression or materializing large arrays when performing
    header-only operations (e.g., set/delete keywords).
    """
    if not isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)):
        return False
    try:
        naxis = int(hdu.header.get('NAXIS', 0) or 0)
        if naxis < 2:
            return False
        # Require nonzero shape in first two axes when available
        naxis1 = int(hdu.header.get('NAXIS1', 0) or 0)
        naxis2 = int(hdu.header.get('NAXIS2', 0) or 0)
        return (naxis1 > 0) and (naxis2 > 0)
    except Exception:
        # If header is malformed, be conservative
        return False


__version__ = "0.2.0"


# =============================================================================
# Single File Keyword Operations
# =============================================================================

def check_keyword(
    filepath: str, 
    keyword: str, 
    ext: int = 0, 
    verbose: bool = False
) -> bool:
    """
    Check if a FITS keyword exists in a specified HDU of a FITS file.

    Args:
        filepath: The path to the FITS file.
        keyword: The FITS keyword to check for (case-insensitive).
        ext: The HDU extension number to check. Defaults to 0 (primary HDU).
        verbose: If True, print diagnostic messages.

    Returns:
        True if the keyword exists in the specified HDU, False otherwise.

    Raises:
        FileHandlingError: If the filepath does not exist, or there is an error
            opening/reading the FITS file.
        InvalidHDUIndexError: If the specified extension ext is out of bounds.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> # Create a test FITS file
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['TESTKEY'] = 'VALUE'
        >>> hdu.writeto(test_file, overwrite=True)
        >>> check_keyword(test_file, 'TESTKEY')
        True
        >>> check_keyword(test_file, 'MISSING')
        False
        >>> os.unlink(test_file)
    """
    if not os.path.exists(filepath):
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))
            return keyword.upper() in hdul[ext].header
            
    except InvalidHDUIndexError:
        raise
    except Exception as e:
        if verbose:
            logger.error(
                f"Error checking keyword '{keyword}' in {filepath} (ext {ext}): {e}",
                exc_info=True
            )
        raise FileHandlingError(
            f"Error opening or reading FITS file {filepath}: {e}",
            filepath=filepath
        ) from e


def get_key(
    filepath: str, 
    keyword: str, 
    ext: int = 0, 
    verbose: bool = False
) -> Any:
    """
    Retrieve the value of a FITS keyword from a specified HDU.

    Args:
        filepath: The path to the FITS file.
        keyword: The FITS keyword whose value is to be retrieved (case-insensitive).
        ext: The HDU extension number. Defaults to 0 (primary HDU).
        verbose: If True, print diagnostic messages.

    Returns:
        The value of the specified keyword.

    Raises:
        FileHandlingError: If the filepath does not exist, or there is an error
            opening/reading the FITS file.
        InvalidHDUIndexError: If the specified extension ext is out of bounds.
        HeaderKeywordError: If the keyword does not exist in the specified HDU.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['EXPTIME'] = 10.0
        >>> hdu.header['OBSERVER'] = 'TEST'
        >>> hdu.writeto(test_file, overwrite=True)
        >>> get_key(test_file, 'EXPTIME')
        10.0
        >>> get_key(test_file, 'OBSERVER')
        'TEST'
        >>> os.unlink(test_file)
    """
    if not os.path.exists(filepath):
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)
    
    try:
        with fits.open(filepath, memmap=True) as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))
            
            header = hdul[ext].header
            keyword_upper = keyword.upper()
            
            if keyword_upper in header:
                return header[keyword_upper]
            else:
                raise HeaderKeywordError(
                    f"Keyword '{keyword}' not found in HDU {ext} of file '{os.path.basename(filepath)}'.",
                    keyword=keyword,
                    filepath=filepath
                )
                
    except (InvalidHDUIndexError, HeaderKeywordError):
        raise
    except Exception as e:
        if verbose:
            logger.error(
                f"Error getting key '{keyword}' from {filepath} (ext {ext}): {e}",
                exc_info=True
            )
        raise FileHandlingError(
            f"Error opening or reading FITS file {filepath}: {e}",
            filepath=filepath
        ) from e


def set_key(
    filepath: str, 
    keyword: str, 
    value: Any, 
    comment: Optional[str] = None, 
    ext: int = 0, 
    verbose: bool = False
) -> bool:
    """
    Set or update a FITS keyword with a new value and optional comment.

    Args:
        filepath: The path to the FITS file.
        keyword: The FITS keyword to set or update (case-insensitive).
        value: The new value for the keyword.
        comment: An optional comment for the keyword.
        ext: The HDU extension number to modify. Defaults to 0 (primary HDU).
        verbose: If True, print diagnostic messages.

    Returns:
        True if the keyword was successfully set/updated.

    Raises:
        FileHandlingError: If the filepath does not exist, or there is an error
            opening/writing to the FITS file.
        InvalidHDUIndexError: If the specified extension ext is out of bounds.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.writeto(test_file, overwrite=True)
        >>> set_key(test_file, 'NEWKEY', 42, comment='Answer')
        True
        >>> get_key(test_file, 'NEWKEY')
        42
        >>> os.unlink(test_file)
    """
    if not os.path.exists(filepath):
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)
    
    try:
        with fits.open(filepath, mode='update') as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))
            
            keyword_upper = keyword.upper()
            if comment:
                hdul[ext].header[keyword_upper] = (value, comment)
            else:
                hdul[ext].header[keyword_upper] = value
            
            hdul.flush()  # Write changes to disk
            
            if verbose:
                logger.info(
                    f"Successfully set keyword '{keyword}' to '{value}' "
                    f"in {filepath} (ext {ext})."
                )
            return True
            
    except InvalidHDUIndexError:
        raise
    except Exception as e:
        if verbose:
            logger.error(
                f"Error setting key '{keyword}' in {filepath} (ext {ext}): {e}",
                exc_info=True
            )
        raise FileHandlingError(
            f"Error opening or writing to FITS file {filepath}: {e}",
            filepath=filepath
        ) from e


def delete_key(
    filepath: str, 
    keyword: str, 
    ext: int = 0, 
    verbose: bool = False
) -> bool:
    """
    Delete a FITS keyword from a specified HDU.

    Args:
        filepath: The path to the FITS file.
        keyword: The FITS keyword to delete (case-insensitive).
        ext: The HDU extension number to modify. Defaults to 0 (primary HDU).
        verbose: If True, print diagnostic messages.

    Returns:
        True if the keyword was successfully deleted.

    Raises:
        FileHandlingError: If the filepath does not exist, or there is an error
            opening/writing to the FITS file.
        InvalidHDUIndexError: If the specified extension ext is out of bounds.
        HeaderKeywordError: If the keyword does not exist in the specified HDU.

    Examples:
        >>> import tempfile
        >>> import numpy as np
        >>> from astropy.io import fits
        >>> with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
        ...     test_file = tmp.name
        >>> hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
        >>> hdu.header['DELKEY'] = 'DELETEME'
        >>> hdu.writeto(test_file, overwrite=True)
        >>> check_keyword(test_file, 'DELKEY')
        True
        >>> delete_key(test_file, 'DELKEY')
        True
        >>> check_keyword(test_file, 'DELKEY')
        False
        >>> os.unlink(test_file)
    """
    if not os.path.exists(filepath):
        raise FileHandlingError(f"File not found: {filepath}", filepath=filepath)
    
    try:
        with fits.open(filepath, mode='update') as hdul:
            if ext < 0 or ext >= len(hdul):
                raise InvalidHDUIndexError(filepath, ext, len(hdul))
            
            header = hdul[ext].header
            keyword_upper = keyword.upper()
            
            if keyword_upper in header:
                del header[keyword_upper]
                hdul.flush()  # Write changes to disk
                
                if verbose:
                    logger.info(
                        f"Successfully deleted keyword '{keyword}' "
                        f"from {filepath} (ext {ext})."
                    )
                return True
            else:
                raise HeaderKeywordError(
                    f"Keyword '{keyword}' not found in HDU {ext} of file '{os.path.basename(filepath)}'.",
                    keyword=keyword,
                    filepath=filepath
                )
                
    except (InvalidHDUIndexError, HeaderKeywordError):
        raise
    except Exception as e:
        if verbose:
            logger.error(
                f"Error deleting key '{keyword}' from {filepath} (ext {ext}): {e}",
                exc_info=True
            )
        raise FileHandlingError(
            f"Error opening or writing to FITS file {filepath}: {e}",
            filepath=filepath
        ) from e


# =============================================================================
# Batch Keyword Operations
# =============================================================================

def batch_set_key(
    input_param: Union[str, List[str]],
    keyword: str,
    value: Any,
    comment: Optional[str] = None,
    ext: Optional[int] = None,
    keys: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
    verbose: bool = False
) -> int:
    """
    Set or update a FITS keyword in multiple FITS files.

    Args:
        input_param: Path to a directory, a glob pattern, or a list of FITS file paths.
        keyword: The FITS keyword to set or update (case-insensitive).
        value: The new value for the keyword.
        comment: An optional comment for the keyword.
        ext: The HDU extension number to modify. If None, the keyword is set
            in all image extensions found (including primary if it's an image HDU).
        keys: Dictionary of FITS header keyword/value pairs to filter files.
            These keys are passed to select_files.
        recursive: If True, search recursively through subdirectories.
        verbose: If True, print diagnostic messages.

    Returns:
        The number of FITS files where the keyword was successfully set/updated.

    Raises:
        SelectionError: If there's an error during file selection.

    Examples:
        >>> # Example would require setting up multiple test files
        >>> # See test suite for comprehensive examples
    """
    # Import here to avoid circular dependency
    from fits_tools.selection import select_files
    
    try:
        files_to_process = select_files(
            input_param, 
            keys=keys, 
            recursive=recursive, 
            verbose=verbose
        )
    except SelectionError as e:
        logger.error(f"Error selecting files for batch keyword setting: {e}")
        raise
    
    if not files_to_process:
        if verbose:
            logger.info("No FITS files found matching the criteria for batch setting keyword.")
        return 0

    updated_files_count = 0
    keyword_upper = keyword.upper()
    
    for file_path in files_to_process:
        try:
            with fits.open(file_path, mode='update') as hdul:
                exts_to_modify = []
                
                if ext is not None:
                    if ext < 0 or ext >= len(hdul):
                        if verbose:
                            logger.warning(
                                f"Skipping {os.path.basename(file_path)}: "
                                f"Extension {ext} is out of bounds. File has {len(hdul)} HDUs."
                            )
                        continue
                    exts_to_modify.append(ext)
                else:
                    # If ext is None, modify all image extensions (including primary if it has data)
                    exts_to_modify = [i for i, hdu in enumerate(hdul) if _is_image_like_hdu(hdu)]
                    if not exts_to_modify:
                        if verbose:
                            logger.warning(
                                f"Skipping {os.path.basename(file_path)}: "
                                f"No image extensions found to modify."
                            )
                        continue

                for current_ext in exts_to_modify:
                    if comment:
                        hdul[current_ext].header[keyword_upper] = (value, comment)
                    else:
                        hdul[current_ext].header[keyword_upper] = value
                    
                    if verbose:
                        logger.info(
                            f"  Set keyword '{keyword}' to '{value}' "
                            f"in {os.path.basename(file_path)} (ext {current_ext})."
                        )
                
                hdul.flush()  # Write changes to disk
                updated_files_count += 1

        except FileNotFoundError:
            if verbose:
                logger.warning(f"  File not found during batch processing: {file_path}. Skipping.")
        except OSError as e:
            if verbose:
                logger.warning(
                    f"  Error opening or writing FITS file {file_path}: {e}. "
                    f"Skipping this file."
                )
        except Exception as e:
            if verbose:
                logger.warning(
                    f"  An unexpected error occurred while processing {file_path}: {e}. "
                    f"Skipping this file.",
                    exc_info=True
                )

    if verbose:
        logger.info(
            f"Total FITS files where keyword '{keyword}' was set/updated: {updated_files_count}"
        )

    return updated_files_count


def batch_delete_key(
    input_param: Union[str, List[str]],
    keyword: str,
    ext: Optional[int] = None,
    keys: Optional[Dict[str, Any]] = None,
    recursive: bool = False,
    verbose: bool = False
) -> int:
    """
    Delete a FITS keyword from multiple FITS files.

    Args:
        input_param: Path to a directory, a glob pattern, or a list of FITS file paths.
        keyword: The FITS keyword to delete (case-insensitive).
        ext: The HDU extension number to modify. If None, the keyword is deleted
            from all image extensions found (including primary if it's an image HDU).
        keys: Dictionary of FITS header keyword/value pairs to filter files.
            These keys are passed to select_files.
        recursive: If True, search recursively through subdirectories.
        verbose: If True, print diagnostic messages.

    Returns:
        The number of FITS files where the keyword was successfully deleted.

    Raises:
        SelectionError: If there's an error during file selection.

    Examples:
        >>> # Example would require setting up multiple test files
        >>> # See test suite for comprehensive examples
    """
    # Import here to avoid circular dependency
    from fits_tools.selection import select_files
    
    try:
        files_to_process = select_files(
            input_param, 
            keys=keys, 
            recursive=recursive, 
            verbose=verbose
        )
    except SelectionError as e:
        logger.error(f"Error selecting files for batch keyword deletion: {e}")
        raise
    
    if not files_to_process:
        if verbose:
            logger.info("No FITS files found matching the criteria for batch deleting keyword.")
        return 0

    updated_files_count = 0
    keyword_upper = keyword.upper()
    
    for file_path in files_to_process:
        try:
            with fits.open(file_path, mode='update') as hdul:
                exts_to_modify = []
                
                if ext is not None:
                    if ext < 0 or ext >= len(hdul):
                        if verbose:
                            logger.warning(
                                f"Skipping {os.path.basename(file_path)}: "
                                f"Extension {ext} is out of bounds. File has {len(hdul)} HDUs."
                            )
                        continue
                    exts_to_modify.append(ext)
                else:
                    # If ext is None, modify all image extensions (including primary if it has data)
                    exts_to_modify = [i for i, hdu in enumerate(hdul) if _is_image_like_hdu(hdu)]
                    if not exts_to_modify:
                        if verbose:
                            logger.warning(
                                f"Skipping {os.path.basename(file_path)}: "
                                f"No image extensions found to modify."
                            )
                        continue

                changes_made_in_file = False
                for current_ext in exts_to_modify:
                    header = hdul[current_ext].header
                    if keyword_upper in header:
                        del header[keyword_upper]
                        if verbose:
                            logger.info(
                                f"  Deleted keyword '{keyword}' "
                                f"from {os.path.basename(file_path)} (ext {current_ext})."
                            )
                        changes_made_in_file = True
                    elif verbose:
                        logger.debug(
                            f"  Keyword '{keyword}' not found "
                            f"in {os.path.basename(file_path)} (ext {current_ext})."
                        )
                
                if changes_made_in_file:
                    hdul.flush()  # Write changes to disk only if something was deleted
                    updated_files_count += 1

        except FileNotFoundError:
            if verbose:
                logger.warning(f"  File not found during batch processing: {file_path}. Skipping.")
        except OSError as e:
            if verbose:
                logger.warning(
                    f"  Error opening or reading FITS file {file_path}: {e}. "
                    f"Skipping this file."
                )
        except Exception as e:
            if verbose:
                logger.warning(
                    f"  An unexpected error occurred while processing {file_path}: {e}. "
                    f"Skipping this file.",
                    exc_info=True
                )

    if verbose:
        logger.info(
            f"Total FITS files where keyword '{keyword}' was deleted: {updated_files_count}"
        )

    return updated_files_count


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Single file operations
    'check_keyword',
    'get_key',
    'set_key',
    'delete_key',
    
    # Batch operations
    'batch_set_key',
    'batch_delete_key',
]


# =============================================================================
# Main / Testing
# =============================================================================

def main():
    """
    Command-line interface for FITS keyword utilities and testing.
    """
    parser = argparse.ArgumentParser(
        description='FITS Keyword Management Utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run doctests
  %(prog)s --test
  
  # Check for a keyword
  %(prog)s --check file.fits EXPTIME
  
  # Get a keyword value
  %(prog)s --get file.fits OBSERVER
  
  # Set a keyword
  %(prog)s --set file.fits NEWKEY 42 --comment "Test value"
  
  # Delete a keyword
  %(prog)s --delete file.fits OLDKEY
        """
    )
    
    parser.add_argument('--test', action='store_true', help='Run doctests')
    parser.add_argument('--check', nargs=2, metavar=('FILE', 'KEYWORD'),
                       help='Check if keyword exists')
    parser.add_argument('--get', nargs=2, metavar=('FILE', 'KEYWORD'),
                       help='Get keyword value')
    parser.add_argument('--set', nargs=3, metavar=('FILE', 'KEYWORD', 'VALUE'),
                       help='Set keyword value')
    parser.add_argument('--delete', nargs=2, metavar=('FILE', 'KEYWORD'),
                       help='Delete keyword')
    parser.add_argument('--comment', help='Comment for set operation')
    parser.add_argument('--ext', type=int, default=0, help='HDU extension number')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run doctests
    if args.test:
        logger.info("Running doctests for keywords.py...")
        result = doctest.testmod(verbose=args.verbose)
        
        if result.failed == 0:
            logger.info(f"All {result.attempted} doctests passed!")
            return 0
        else:
            logger.error(f"{result.failed} of {result.attempted} doctests failed!")
            return 1
    
    # Check keyword
    if args.check:
        filepath, keyword = args.check
        exists = check_keyword(filepath, keyword, ext=args.ext, verbose=args.verbose)
        print(f"Keyword '{keyword}' {'exists' if exists else 'does not exist'} in {filepath}")
        return 0
    
    # Get keyword
    if args.get:
        filepath, keyword = args.get
        try:
            value = get_key(filepath, keyword, ext=args.ext, verbose=args.verbose)
            print(f"{keyword} = {value}")
            return 0
        except HeaderKeywordError as e:
            print(f"Error: {e}")
            return 1
    
    # Set keyword
    if args.set:
        filepath, keyword, value = args.set
        # Try to convert value to appropriate type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
        
        set_key(filepath, keyword, value, comment=args.comment, ext=args.ext, verbose=args.verbose)
        print(f"Set {keyword} = {value}")
        return 0
    
    # Delete keyword
    if args.delete:
        filepath, keyword = args.delete
        try:
            delete_key(filepath, keyword, ext=args.ext, verbose=args.verbose)
            print(f"Deleted keyword '{keyword}' from {filepath}")
            return 0
        except HeaderKeywordError as e:
            print(f"Error: {e}")
            return 1
    
    # No arguments
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
    
