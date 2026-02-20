#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bad Pixel Mask construction utilities and workflows.

This module provides high-level tools for building BPM arrays from
calibration data and defect detection algorithms. The BPMBuilder class
offers a fluent interface with automatic audit trails and quality validation.

The separation from bitmask.py allows this module to integrate with
detection algorithms without creating circular dependencies or bloating
the core bitmask schema definitions.

See Also
--------
fits_tools.bitmask : Core bit definitions and low-level utilities
fits_tools.defects.detection : Defect detection algorithms
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging
import warnings

# Import core bitmask definitions (no circular dependency risk)
from .bitmask import (
    # Bit constants
    BPM_BIT_PIX_BAD, BPM_BIT_PIX_DEAD, BPM_BIT_PIX_BRIGHT, BPM_BIT_PIX_HOT,
    BPM_BIT_PIX_PRNU_HIGH, BPM_BIT_PIX_PRNU_LOW, BPM_BIT_PIX_NONLIN,
    BPM_BIT_PIX_NOISE, BPM_BIT_PIX_RTS, BPM_BIT_PIX_TRAP, BPM_BIT_PIX_GLOW,
    BPM_BIT_PIX_EXCLUDED,
    BPM_BIT_COL_BAD, BPM_BIT_COL_BRIGHT, BPM_BIT_COL_DARK, BPM_BIT_COL_TRAP,
    BPM_BIT_COL_BLOCKED, BPM_BIT_COL_CTE_LOW, BPM_BIT_COL_BOUNDARY,
    BPM_BIT_COL_EXCLUDED,
    BPM_BIT_ROW_BAD, BPM_BIT_ROW_EXCLUDED,
    BPM_BIT_SATURATED, BPM_BIT_COSMIC, BPM_BIT_CROSSTALK,
    BPM_BIT_BLOOM, BPM_BIT_OUTLIER,
    
    # Masks
    BPM_MASK_PIX, BPM_MASK_PERMANENT, BPM_MASK_HOT, BPM_MASK_COLD,
    
    # Utilities
    create_bpm, merge_bpms, get_bpm_statistics, count_defects,
    check_bad_rows_cols, FLAG_NAMES, format_bit_comments,
)

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

class BPMBuilder:
    """
    Fluent interface for constructing Bad Pixel Mask arrays with audit trails.
    
    This builder pattern provides a declarative, chainable API for constructing
    complex BPM arrays from multiple detection algorithms and manual inputs.
    Each operation is logged, allowing full reproducibility and debugging.
    
    The builder maintains:
    - The BPM array under construction
    - An operation history (audit trail)
    - Validation state
    - Metadata for FITS header injection
    
    Parameters
    ----------
    shape : tuple of int
        Dimensions of the BPM array (typically (nrows, ncols)).
    dtype : np.dtype, optional
        Bit depth of the mask. Default uint32 supports all 32 flags.
    sensor_id : str, optional
        Identifier for the sensor (included in metadata/logs).
    
    Attributes
    ----------
    bpm : np.ndarray
        The BPM array being constructed (mutable).
    history : list of dict
        Audit trail of all operations applied.
    metadata : dict
        FITS header keywords and values to document the BPM.
    
    Examples
    --------
    Basic usage:
        >>> builder = BPMBuilder((2048, 2048), sensor_id='CCD-SN12345')
        >>> builder.mark_pixels(hot_coords, BPM_BIT_PIX_HOT)
        >>> builder.mark_column(512, BPM_BIT_COL_DARK)
        >>> bpm = builder.build()
    
    Chained fluent interface:
        >>> bpm = (BPMBuilder((2048, 2048))
        ...     .mark_pixels(dead_coords, BPM_BIT_PIX_DEAD, reason="<0.5% QE")
        ...     .mark_column(100, BPM_BIT_COL_BAD)
        ...     .exclude_edges(width=4)
        ...     .build())
    
    With validation:
        >>> bpm = (BPMBuilder((2048, 2048))
        ...     .from_dark_analysis(dark_stack)
        ...     .validate(max_defect_fraction=0.05)  # Enforce <5% defect rate
        ...     .build())
    
    Integration with detection pipeline:
        >>> from fits_tools.defects.detection import detect_hot_pixels
        >>> hot_pixels, metrics = detect_hot_pixels(dark_stack)
        >>> bpm = (BPMBuilder((2048, 2048))
        ...     .from_detection_result(hot_pixels, BPM_BIT_PIX_HOT, 
        ...                           metadata=metrics)
        ...     .build())
    
    See Also
    --------
    create_bpm : Low-level BPM array creation
    merge_bpms : Combine multiple completed BPM arrays
    """
    
    def __init__(
        self, 
        shape: Tuple[int, ...], 
        dtype: type = np.uint32,
        sensor_id: Optional[str] = None
    ):
        self.bpm = create_bpm(shape, dtype=dtype)
        self.shape = shape
        self.dtype = dtype
        self.sensor_id = sensor_id or "UNKNOWN"
        self.history: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            'SENSOR_ID': self.sensor_id,
            'BPM_SHAPE': str(shape),
            'BPM_DTYPE': str(dtype),
        }
        self._validated = False
        
        logger.debug(f"BPMBuilder initialized: shape={shape}, dtype={dtype}")
    
    def _log_operation(
        self, 
        operation: str, 
        flag: Optional[int] = None,
        count: Optional[int] = None,
        reason: Optional[str] = None,
        **kwargs
    ) -> None:
        """Internal: Record an operation in the history."""
        entry = {
            'operation': operation,
            'timestamp': np.datetime64('now'),
        }
        if flag is not None:
            entry['flag'] = FLAG_NAMES.get(flag, f"0x{flag:08X}")
        if count is not None:
            entry['affected_pixels'] = count
        if reason is not None:
            entry['reason'] = reason
        entry.update(kwargs)
        
        self.history.append(entry)
        logger.debug(f"BPMBuilder: {operation} - {count} pixels affected")
    
    # -------------------------------------------------------------------------
    # Core Marking Methods
    # -------------------------------------------------------------------------
    
    def mark_pixels(
        self, 
        coords: Union[np.ndarray, List[Tuple[int, int]]], 
        flag: int,
        reason: Optional[str] = None
    ) -> 'BPMBuilder':
        """
        Mark individual pixels with a specific defect flag.
        
        Parameters
        ----------
        coords : array-like of shape (N, 2) or list of (row, col) tuples
            Pixel coordinates to mark.
        flag : int
            Defect flag to apply (e.g., BPM_BIT_PIX_HOT).
        reason : str, optional
            Human-readable explanation for audit trail.
        
        Returns
        -------
        self : BPMBuilder
            For method chaining.
        
        Examples
        --------
        >>> coords = np.array([[100, 200], [150, 300]])
        >>> builder.mark_pixels(coords, BPM_BIT_PIX_HOT, reason="Dark current > 5 e-/s")
        """
        coords = np.atleast_2d(coords)
        if coords.shape[1] != 2:
            raise ValueError(f"coords must have shape (N, 2), got {coords.shape}")
        
        rows, cols = coords[:, 0], coords[:, 1]
        
        # Validate coordinates are in bounds
        if np.any(rows >= self.shape[0]) or np.any(cols >= self.shape[1]):
            raise ValueError("Coordinates out of bounds")
        
        self.bpm[rows, cols] |= flag
        
        self._log_operation(
            'mark_pixels',
            flag=flag,
            count=len(coords),
            reason=reason
        )
        
        return self
    
    def mark_column(
        self, 
        col_index: Union[int, List[int]], 
        flag: int = BPM_BIT_COL_BAD,
        reason: Optional[str] = None
    ) -> 'BPMBuilder':
        """
        Mark entire column(s) with a defect flag.
        
        Parameters
        ----------
        col_index : int or list of int
            Column index or indices to mark.
        flag : int
            Defect flag (typically a COL_* flag).
        reason : str, optional
            Explanation (e.g., "Dead readout channel").
        
        Returns
        -------
        self : BPMBuilder
        """
        if isinstance(col_index, int):
            col_index = [col_index]
        
        for col in col_index:
            if col < 0 or col >= self.shape[1]:
                raise ValueError(f"Column {col} out of bounds [0, {self.shape[1]})")
            self.bpm[:, col] |= flag
        
        affected = len(col_index) * self.shape[0]
        self._log_operation(
            'mark_column',
            flag=flag,
            count=affected,
            reason=reason,
            columns=col_index
        )
        
        return self
    
    def mark_row(
        self, 
        row_index: Union[int, List[int]], 
        flag: int = BPM_BIT_ROW_BAD,
        reason: Optional[str] = None
    ) -> 'BPMBuilder':
        """
        Mark entire row(s) with a defect flag.
        
        Parameters
        ----------
        row_index : int or list of int
            Row index or indices to mark.
        flag : int
            Defect flag (typically a ROW_* flag).
        reason : str, optional
            Explanation.
        
        Returns
        -------
        self : BPMBuilder
        """
        if isinstance(row_index, int):
            row_index = [row_index]
        
        for row in row_index:
            if row < 0 or row >= self.shape[0]:
                raise ValueError(f"Row {row} out of bounds [0, {self.shape[0]})")
            self.bpm[row, :] |= flag
        
        affected = len(row_index) * self.shape[1]
        self._log_operation(
            'mark_row',
            flag=flag,
            count=affected,
            reason=reason,
            rows=row_index
        )
        
        return self
    
    def mark_region(
        self, 
        row_slice: slice, 
        col_slice: slice,
        flag: int,
        reason: Optional[str] = None
    ) -> 'BPMBuilder':
        """
        Mark a rectangular region with a defect flag.
        
        Parameters
        ----------
        row_slice : slice
            Row selection (e.g., slice(100, 200)).
        col_slice : slice
            Column selection.
        flag : int
            Defect flag to apply.
        reason : str, optional
            Explanation (e.g., "Amplifier glow region").
        
        Returns
        -------
        self : BPMBuilder
        
        Examples
        --------
        >>> # Mark bottom-right corner as excluded
        >>> builder.mark_region(slice(1900, None), slice(1900, None),
        ...                     BPM_BIT_PIX_EXCLUDED, reason="Overscan")
        """
        self.bpm[row_slice, col_slice] |= flag
        
        affected = self.bpm[row_slice, col_slice].size
        self._log_operation(
            'mark_region',
            flag=flag,
            count=affected,
            reason=reason,
            row_slice=str(row_slice),
            col_slice=str(col_slice)
        )
        
        return self
    
    # -------------------------------------------------------------------------
    # Convenience Methods (Common Patterns)
    # -------------------------------------------------------------------------
    
    def exclude_edges(
        self, 
        width: int = 1,
        reason: Optional[str] = None
    ) -> 'BPMBuilder':
        """
        Exclude edge pixels (common for boundary effects).
        
        Parameters
        ----------
        width : int
            Number of pixels to exclude from each edge.
        reason : str, optional
            Defaults to "Edge exclusion".
        
        Returns
        -------
        self : BPMBuilder
        """
        if reason is None:
            reason = f"Edge exclusion (width={width})"
        
        # Top and bottom edges
        self.mark_region(slice(0, width), slice(None), 
                        BPM_BIT_PIX_EXCLUDED, reason=reason)
        self.mark_region(slice(-width, None), slice(None), 
                        BPM_BIT_PIX_EXCLUDED, reason=reason)
        
        # Left and right edges
        self.mark_region(slice(None), slice(0, width), 
                        BPM_BIT_PIX_EXCLUDED, reason=reason)
        self.mark_region(slice(None), slice(-width, None), 
                        BPM_BIT_PIX_EXCLUDED, reason=reason)
        
        return self
    
    def exclude_region(
        self,
        row_slice: slice,
        col_slice: slice,
        reason: str = "Manual exclusion"
    ) -> 'BPMBuilder':
        """Convenience wrapper for marking regions as excluded."""
        return self.mark_region(row_slice, col_slice, 
                               BPM_BIT_PIX_EXCLUDED, reason=reason)
    
    # -------------------------------------------------------------------------
    # Integration with Detection Algorithms
    # -------------------------------------------------------------------------
    
    def from_detection_result(
        self,
        coords: np.ndarray,
        flag: int,
        metadata: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None
    ) -> 'BPMBuilder':
        """
        Integrate results from a defect detection algorithm.
        
        This is the primary integration point with fits_tools.defects.detection.
        
        Parameters
        ----------
        coords : np.ndarray of shape (N, 2)
            Detected defect coordinates.
        flag : int
            Defect type flag.
        metadata : dict, optional
            Algorithm parameters and metrics (stored in builder metadata).
        reason : str, optional
            Detection method description.
        
        Returns
        -------
        self : BPMBuilder
        
        Examples
        --------
        >>> from fits_tools.defects.detection import detect_hot_pixels
        >>> coords, metrics = detect_hot_pixels(dark_stack, threshold=5.0)
        >>> builder.from_detection_result(
        ...     coords, BPM_BIT_PIX_HOT, 
        ...     metadata=metrics,
        ...     reason=f"Dark current > {metrics['threshold']} e-/s"
        ... )
        """
        if metadata:
            flag_name = FLAG_NAMES.get(flag, f"FLAG_{flag}")
            self.metadata[f"{flag_name}_DETECTION"] = metadata
        
        return self.mark_pixels(coords, flag, reason=reason)
    
    def from_dark_analysis(
        self,
        dark_stack: np.ndarray,
        hot_threshold: float = 5.0,
        dead_threshold: float = 0.1,
        method: str = 'sigma_clip'
    ) -> 'BPMBuilder':
        """
        Detect and mark hot/dead pixels from a dark frame stack.
        
        This is a high-level convenience method that calls your detection
        pipeline and automatically marks the results.
        
        Parameters
        ----------
        dark_stack : np.ndarray of shape (N, rows, cols)
            Stack of dark frames (bias-subtracted).
        hot_threshold : float
            Dark current threshold in e-/s for hot pixels.
        dead_threshold : float
            Response threshold for dead pixels (fraction of median).
        method : str
            Detection algorithm to use.
        
        Returns
        -------
        self : BPMBuilder
        
        Notes
        -----
        Requires fits_tools.defects.detection module.
        """
        try:
            from fits_tools.defects.detection import (
                detect_hot_pixels, detect_prnu_anomalies
            )
        except ImportError:
            raise ImportError(
                "from_dark_analysis requires fits_tools.defects.detection"
            )
        
        # Detect hot pixels
        hot_coords, hot_metrics = detect_hot_pixels(
            dark_stack, threshold=hot_threshold, method=method
        )
        if len(hot_coords) > 0:
            self.from_detection_result(
                hot_coords, BPM_BIT_PIX_HOT,
                metadata=hot_metrics,
                reason=f"Dark current > {hot_threshold} e-/s"
            )
        
        # Detect dead pixels
        dead_coords, dead_metrics = detect_dead_pixels(
            dark_stack, threshold=dead_threshold, method=method
        )
        if len(dead_coords) > 0:
            self.from_detection_result(
                dead_coords, BPM_BIT_PIX_DEAD,
                metadata=dead_metrics,
                reason=f"Response < {dead_threshold * 100}% of median"
            )
        
        return self
    
    def from_flat_analysis(
        self,
        flat_stack: np.ndarray,
        qe_low_threshold: float = 0.8,
        qe_high_threshold: float = 1.2,
        sigma: float = 5.0
    ) -> 'BPMBuilder':
        """
        Detect and mark QE anomalies from a flat field stack.
        
        Parameters
        ----------
        flat_stack : np.ndarray
            Stack of flat field images (bias/dark corrected).
        qe_low_threshold : float
            Relative QE below which pixels are flagged (0.8 = 80% of median).
        qe_high_threshold : float
            Relative QE above which pixels are flagged.
        sigma : float
            Statistical threshold for outlier detection.
        
        Returns
        -------
        self : BPMBuilder
        """
        try:
            from fits_tools.defects.detection import (
                detect_qe_anomalies
            )
        except ImportError:
            raise ImportError(
                "from_flat_analysis requires fits_tools.defects.detection"
            )
        
        low_coords, high_coords, metrics = detect_qe_anomalies(
            flat_stack,
            low_threshold=qe_low_threshold,
            high_threshold=qe_high_threshold,
            sigma=sigma
        )
        
        if len(low_coords) > 0:
            self.from_detection_result(
                low_coords, BPM_BIT_PIX_PRNU_LOW,
                metadata=metrics,
                reason=f"QE < {qe_low_threshold * 100}% of median"
            )
        
        if len(high_coords) > 0:
            self.from_detection_result(
                high_coords, BPM_BIT_PIX_PRNU_HIGH,
                metadata=metrics,
                reason=f"QE > {qe_high_threshold * 100}% of median"
            )
        
        return self
    
    # -------------------------------------------------------------------------
    # Structural Defect Propagation
    # -------------------------------------------------------------------------
    
    def propagate_column_flags(
        self,
        row_fraction: float = 0.1,
        source_mask: int = BPM_MASK_PIX,
        target_flag: int = BPM_BIT_COL_BAD
    ) -> 'BPMBuilder':
        """
        Automatically flag columns based on pixel defect density.
        
        Calls check_bad_rows_cols internally and applies COL_BAD flags
        to columns exceeding the threshold.
        
        Parameters
        ----------
        row_fraction : float
            Fraction of rows with defects needed to flag the column.
        source_mask : int
            Which pixel defects to count (default: all pixel-level defects).
        target_flag : int
            Column flag to apply (default: COL_BAD).
        
        Returns
        -------
        self : BPMBuilder
        
        See Also
        --------
        check_bad_rows_cols : Underlying detection function
        """
        _, bad_cols = check_bad_rows_cols(
            self.bpm,
            row_threshold=1.0,  # Ignore rows for this operation
            col_threshold=row_fraction,
            defect_mask=source_mask
        )
        
        bad_col_indices = np.where(bad_cols)[0]
        
        if len(bad_col_indices) > 0:
            self.mark_column(
                bad_col_indices.tolist(),
                flag=target_flag,
                reason=f"Propagated: >{row_fraction*100}% pixel defects"
            )
        
        return self
    
    def propagate_row_flags(
        self,
        col_fraction: float = 0.1,
        source_mask: int = BPM_MASK_PIX,
        target_flag: int = BPM_BIT_ROW_BAD
    ) -> 'BPMBuilder':
        """
        Automatically flag rows based on pixel defect density.
        
        Parameters
        ----------
        col_fraction : float
            Fraction of columns with defects needed to flag the row.
        source_mask : int
            Which pixel defects to count.
        target_flag : int
            Row flag to apply.
        
        Returns
        -------
        self : BPMBuilder
        """
        bad_rows, _ = check_bad_rows_cols(
            self.bpm,
            row_threshold=col_fraction,
            col_threshold=1.0,  # Ignore columns
            defect_mask=source_mask
        )
        
        bad_row_indices = np.where(bad_rows)[0]
        
        if len(bad_row_indices) > 0:
            self.mark_row(
                bad_row_indices.tolist(),
                flag=target_flag,
                reason=f"Propagated: >{col_fraction*100}% pixel defects"
            )
        
        return self
    
    # -------------------------------------------------------------------------
    # Merging and Importing
    # -------------------------------------------------------------------------
    
    def merge_from(
        self,
        other_bpm: np.ndarray,
        operation: str = 'or',
        reason: Optional[str] = None
    ) -> 'BPMBuilder':
        """
        Merge another BPM array into this builder.
        
        Parameters
        ----------
        other_bpm : np.ndarray
            External BPM to merge.
        operation : str
            Merge strategy: 'or', 'and', 'xor', 'diff'.
        reason : str, optional
            Explanation for the merge.
        
        Returns
        -------
        self : BPMBuilder
        
        Examples
        --------
        >>> # Merge a vendor-supplied master BPM
        >>> vendor_bpm = load_vendor_bpm()
        >>> builder.merge_from(vendor_bpm, operation='or', 
        ...                    reason="Merged vendor master BPM")
        """
        before_count = np.count_nonzero(self.bpm)
        
        self.bpm = merge_bpms(self.bpm, other_bpm, operation=operation)
        
        after_count = np.count_nonzero(self.bpm)
        delta = after_count - before_count
        
        self._log_operation(
            f'merge_{operation}',
            count=delta,
            reason=reason,
            total_after_merge=after_count
        )
        
        return self
    
    # -------------------------------------------------------------------------
    # Validation and Quality Control
    # -------------------------------------------------------------------------
    
    def validate(
        self,
        max_defect_fraction: float = 0.10,
        max_col_defect_count: int = 100,
        max_row_defect_count: int = 100,
        raise_on_fail: bool = True
    ) -> 'BPMBuilder':
        """
        Validate the BPM against quality thresholds.
        
        Parameters
        ----------
        max_defect_fraction : float
            Maximum allowable fraction of defective pixels (0-1).
        max_col_defect_count : int
            Maximum number of columns allowed to be flagged as bad.
        max_row_defect_count : int
            Maximum number of rows allowed to be flagged as bad.
        raise_on_fail : bool
            If True, raise ValueError on validation failure.
            If False, log a warning and continue.
        
        Returns
        -------
        self : BPMBuilder
        
        Raises
        ------
        ValueError
            If validation fails and raise_on_fail=True.
        
        Examples
        --------
        >>> # Enforce strict quality requirements
        >>> bpm = (BPMBuilder((2048, 2048))
        ...     .from_dark_analysis(dark_stack)
        ...     .validate(max_defect_fraction=0.05)  # <5% defects
        ...     .build())
        """
        stats = get_bpm_statistics(self.bpm)
        
        # Check overall defect fraction
        total_defect_fraction = np.count_nonzero(self.bpm != 0) / self.bpm.size
        
        # Check column/row counts
        col_bad_count = stats.get('COL_BAD', {}).get('count', 0)
        row_bad_count = stats.get('ROW_BAD', {}).get('count', 0)
        
        failures = []
        
        if total_defect_fraction > max_defect_fraction:
            failures.append(
                f"Defect fraction {total_defect_fraction:.2%} exceeds "
                f"limit {max_defect_fraction:.2%}"
            )
        
        if col_bad_count > max_col_defect_count:
            failures.append(
                f"Bad column count {col_bad_count} exceeds limit {max_col_defect_count}"
            )
        
        if row_bad_count > max_row_defect_count:
            failures.append(
                f"Bad row count {row_bad_count} exceeds limit {max_row_defect_count}"
            )
        
        if failures:
            msg = "BPM validation failed:\n  " + "\n  ".join(failures)
            if raise_on_fail:
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning)
                logger.warning(msg)
        else:
            logger.info(
                f"BPM validation passed: {total_defect_fraction:.2%} defect rate"
            )
            self._validated = True
        
        self._log_operation(
            'validate',
            passed=len(failures) == 0,
            defect_fraction=total_defect_fraction,
            validation_criteria={
                'max_defect_fraction': max_defect_fraction,
                'max_col_defect_count': max_col_defect_count,
                'max_row_defect_count': max_row_defect_count,
            }
        )
        
        return self
    
    # -------------------------------------------------------------------------
    # Output and Finalization
    # -------------------------------------------------------------------------
    
    def build(self) -> np.ndarray:
        """
        Finalize and return the constructed BPM array.
        
        Returns
        -------
        np.ndarray
            The completed BPM array.
        
        Notes
        -----
        This method returns a COPY of the internal BPM, allowing the
        builder to be reused if needed.
        """
        if not self._validated:
            logger.warning(
                "Building BPM without validation. "
                "Consider calling .validate() before .build()"
            )
        
        self._log_operation('build', count=np.count_nonzero(self.bpm))
        
        logger.info(
            f"BPM built: {np.count_nonzero(self.bpm)} defective pixels "
            f"({np.count_nonzero(self.bpm) / self.bpm.size * 100:.2f}%)"
        )
        
        return self.bpm.copy()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Return the operation history (audit trail).
        
        Returns
        -------
        list of dict
            Each entry contains operation metadata.
        """
        return self.history.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata suitable for FITS header injection.
        
        Returns
        -------
        dict
            Key-value pairs for FITS keywords.
        """
        # Add summary statistics
        stats = get_bpm_statistics(self.bpm)
        meta = self.metadata.copy()
        
        meta['BPM_NPIX'] = int(np.count_nonzero(self.bpm != 0))
        meta['BPM_FRAC'] = float(meta['BPM_NPIX'] / self.bpm.size)
        meta['BPM_NHOT'] = int(count_defects(self.bpm, BPM_MASK_HOT))
        meta['BPM_NDEAD'] = int(count_defects(self.bpm, BPM_MASK_COLD))
        meta['BPM_NOPS'] = len(self.history)
        
        return meta
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the BPM construction."""
        print(f"\n{'='*70}")
        print(f"BPM Builder Summary: {self.sensor_id}")
        print(f"{'='*70}")
        print(f"Shape:           {self.shape}")
        print(f"Operations:      {len(self.history)}")
        
        stats = get_bpm_statistics(self.bpm)
        total_defects = np.count_nonzero(self.bpm != 0)
        print(f"Total defects:   {total_defects:,} ({total_defects/self.bpm.size*100:.2f}%)")
        print(f"Validated:       {self._validated}")
        
        print(f"\nDefect Breakdown:")
        for name, info in stats.items():
            if info['count'] > 0 and info['type'] == 'Individual Bit':
                print(f"  {name:18s}: {info['count']:7,} ({info['fraction']*100:5.2f}%)")
        
        print(f"\nOperation History:")
        for i, op in enumerate(self.history, 1):
            op_str = f"  {i}. {op['operation']}"
            if 'flag' in op:
                op_str += f" [{op['flag']}]"
            if 'affected_pixels' in op:
                op_str += f" - {op['affected_pixels']:,} pixels"
            if 'reason' in op:
                op_str += f" ({op['reason']})"
            print(op_str)
        
        print(f"{'='*70}\n")
    
    def save(self, filename: str, include_history: bool = True) -> None:
        """
        Save the BPM to a FITS file with full metadata.
        
        Parameters
        ----------
        filename : str
            Output FITS filename.
        include_history : bool
            If True, write operation history as FITS HISTORY cards.
        
        Notes
        -----
        Requires astropy.io.fits or your FITS I/O module.
        
        Examples
        --------
        >>> builder.save('master_bpm_ccd123.fits')
        """
        try:
            from astropy.io import fits
        except ImportError:
            raise ImportError("save() requires astropy")
        
        # Create primary HDU
        hdu = fits.PrimaryHDU(self.bpm)
        
        # Add metadata as header keywords
        for key, value in self.get_metadata().items():
            try:
                hdu.header[key] = value
            except (ValueError, TypeError):
                # Skip values that can't be serialized to FITS
                logger.warning(f"Could not write metadata key {key} to FITS header")
        
        # Add bit flag documentation
        for comment in format_bit_comments():
            hdu.header['COMMENT'] = comment
        
        # Add operation history
        if include_history:
            for op in self.history:
                hist_str = f"{op['operation']}"
                if 'flag' in op:
                    hist_str += f" flag={op['flag']}"
                if 'reason' in op:
                    hist_str += f" | {op['reason']}"
                hdu.header['HISTORY'] = hist_str
        
        hdu.writeto(filename, overwrite=True)
        logger.info(f"BPM saved to {filename}")
    
    def clone(self) -> 'BPMBuilder':
        """
        Create a copy of this builder (useful for branching workflows).
        
        Returns
        -------
        BPMBuilder
            A new builder with the same state.
        
        Examples
        --------
        >>> # Create variants from a base BPM
        >>> base_builder = BPMBuilder((2048, 2048)).from_dark_analysis(dark_stack)
        >>> 
        >>> # Strict version
        >>> strict_bpm = base_builder.clone().propagate_column_flags(0.05).build()
        >>> 
        >>> # Lenient version
        >>> lenient_bpm = base_builder.clone().propagate_column_flags(0.20).build()
        """
        new_builder = BPMBuilder(self.shape, dtype=self.dtype, sensor_id=self.sensor_id)
        new_builder.bpm = self.bpm.copy()
        new_builder.history = self.history.copy()
        new_builder.metadata = self.metadata.copy()
        new_builder._validated = self._validated
        return new_builder
    
    def reset(self) -> 'BPMBuilder':
        """
        Clear all flags and history (restart from blank slate).
        
        Returns
        -------
        self : BPMBuilder
        
        Notes
        -----
        Metadata and sensor_id are preserved.
        """
        self.bpm = create_bpm(self.shape, dtype=self.dtype)
        self.history = []
        self._validated = False
        logger.info("BPMBuilder reset")
        return self
    
    # -------------------------------------------------------------------------
    # Context Manager Support (Optional but nice)
    # -------------------------------------------------------------------------
    
    def __enter__(self) -> 'BPMBuilder':
        """Enable use as context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup on context exit."""
        if exc_type is not None:
            logger.error(f"BPMBuilder context exited with error: {exc_val}")
        # Could add auto-save on success here if desired
        return False  # Don't suppress exceptions
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        total_defects = np.count_nonzero(self.bpm != 0)
        defect_pct = total_defects / self.bpm.size * 100
        return (
            f"BPMBuilder(shape={self.shape}, "
            f"defects={total_defects} ({defect_pct:.2f}%), "
            f"operations={len(self.history)}, "
            f"validated={self._validated})"
        )

# =============================================================================
# Convenience Functions (Module-Level)
# =============================================================================

def build_bpm_from_calibrations(
    shape: Tuple[int, int],
    dark_stack: Optional[np.ndarray] = None,
    flat_stack: Optional[np.ndarray] = None,
    vendor_bpm: Optional[np.ndarray] = None,
    hot_threshold: float = 5.0,
    dead_threshold: float = 0.1,
    qe_low: float = 0.8,
    qe_high: float = 1.2,
    validate: bool = True,
    max_defect_fraction: float = 0.10,
    sensor_id: Optional[str] = None
) -> Tuple[np.ndarray, BPMBuilder]:
    """
    One-shot function to build a BPM from standard calibration data.
    
    This is a convenience wrapper around BPMBuilder for the most common
    workflow: combine vendor BPM + dark analysis + flat analysis.
    
    Parameters
    ----------
    shape : tuple of int
        Sensor dimensions (rows, cols).
    dark_stack : np.ndarray, optional
        Stack of dark frames for hot/dead pixel detection.
    flat_stack : np.ndarray, optional
        Stack of flat frames for QE anomaly detection.
    vendor_bpm : np.ndarray, optional
        Pre-existing vendor/master BPM to merge.
    hot_threshold : float
        Dark current threshold (e-/s) for hot pixels.
    dead_threshold : float
        Response threshold for dead pixels.
    qe_low, qe_high : float
        QE thresholds for flagging anomalies.
    validate : bool
        Whether to validate before returning.
    max_defect_fraction : float
        Maximum allowable defect fraction for validation.
    sensor_id : str, optional
        Sensor identifier for metadata.
    
    Returns
    -------
    bpm : np.ndarray
        The constructed BPM array.
    builder : BPMBuilder
        The builder instance (for accessing history/metadata).
    
    Examples
    --------
    >>> bpm, builder = build_bpm_from_calibrations(
    ...     shape=(2048, 2048),
    ...     dark_stack=darks,
    ...     flat_stack=flats,
    ...     sensor_id='CCD-001'
    ... )
    >>> builder.print_summary()
    >>> builder.save('master_bpm.fits')
    """
    builder = BPMBuilder(shape, sensor_id=sensor_id)
    
    # Merge vendor BPM if provided
    if vendor_bpm is not None:
        builder.merge_from(vendor_bpm, operation='or', 
                          reason="Vendor master BPM")
    
    # Dark analysis
    if dark_stack is not None:
        builder.from_dark_analysis(
            dark_stack,
            hot_threshold=hot_threshold,
            dead_threshold=dead_threshold
        )
    
    # Flat analysis
    if flat_stack is not None:
        builder.from_flat_analysis(
            flat_stack,
            qe_low_threshold=qe_low,
            qe_high_threshold=qe_high
        )
    
    # Auto-propagate structural defects
    builder.propagate_column_flags(row_fraction=0.1)
    builder.propagate_row_flags(col_fraction=0.1)
    
    # Validate if requested
    if validate:
        builder.validate(
            max_defect_fraction=max_defect_fraction,
            raise_on_fail=True
        )
    
    return builder.build(), builder


__all__ = [
    'BPMBuilder',
    'build_bpm_from_calibrations',
]

# =============================================================================
# Add to __all__ at the end of bitmask.py
# =============================================================================

__all__ = [
    # ... existing exports ...
    'BPMBuilder',  # ← ADD THIS
]