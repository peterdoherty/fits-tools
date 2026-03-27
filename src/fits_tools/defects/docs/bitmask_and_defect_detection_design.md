# Bitmask and Defect Detection Design

This document describes the design of the bad-pixel / data-quality bitmask and
the associated defect-detection pipeline used in the `fits_tools` library.

It is intended as a reference for:
- users of the bad pixel mask (BPM) FITS files,
- developers extending the detection algorithms, and
- reviewers evaluating the scientific and engineering rationale.

---

## 1. Bitmask Overview

### 1.1 Goals

The BPM and data-quality (DQ) bitmask is designed to:

- Encode **hardware defects**, **structural defects**, and **transient artifacts**
  in a single array.
- Support both **permanent** (silicon / electronics) and **per-frame** (exposure)
  bit flags.
- Be **self-describing** on disk, so that downstream tools can interpret the
  mask without importing Python code.

The mask uses a 32-bit unsigned integer per pixel (`uint32` recommended).

### 1.2 Bit Layout

Bits are grouped into three logical regions:

- **Bits 0–15**: pixel-level, static hardware defects.
- **Bits 16–26**: structural defects at row/column level.
- **Bits 27–31**: transient per-frame data-quality flags.

#### Pixel-level (0–15)

These bits represent single-pixel hardware properties:

- `PIX_BAD` (bit 0): general unusable pixel (summary flag).
- `PIX_DEAD` (bit 1): dead/unresponsive pixel (near-zero signal).
- `PIX_BRIGHT` (bit 2): anomalously high bias offset.
- `PIX_HOT` (bit 3): excessive dark current.
- `PIX_QE_HIGH` (bit 4): high quantum efficiency.
- `PIX_QE_LOW` (bit 5): low quantum efficiency (cold/dead in flats).
- `PIX_NONLIN` (bit 6): non-linear response.
- `PIX_NOISE` (bit 7): excessive read noise.
- `PIX_RTS` (bit 8): Random Telegraph Signal (temporal instability).
- `PIX_TRAP` (bit 9): pixel-level charge trap.
- `PIX_GLOW` (bit 10): amplifier / circuitry glow region.
- `PIX_EXCLUDED` (bit 15): manually excluded from statistics.

Bits 11–14 are reserved for future pixel-level use.

#### Structural (16–26)

These bits represent row/column‑scale defects:

- `COL_BAD` (bit 16): unusable column (summary flag).
- `COL_BRIGHT` (17): bright column.
- `COL_DARK` (18): dark/blocked column.
- `COL_TRAP` (19): column-level charge traps.
- `COL_BLOCKED` (20): vertical transfer blocked by defect.
- `COL_CTE_LOW` (21): poor CTE (high CTI).
- `COL_BOUNDARY` (22): segment / amplifier boundary column.
- `COL_EXCLUDED` (23): column excluded from measurements.
- `ROW_BAD` (24): unusable row.
- `ROW_EXCLUDED` (26): row excluded from measurements.

Bit 25 is reserved for future row-level use.

#### Transient / DQ (27–31)

These bits represent per-exposure conditions:

- `SATURATED` (bit 27): pixel at or above saturation level.
- `PIX_COSMIC` (28): cosmic ray hit.
- `PIX_XTALK` (29): electronic crosstalk artifact.
- `PIX_BLOOM` (30): blooming from saturated neighbors.
- `PIX_OUTLIER` (31): pipeline-specific statistical outlier.

---

## 2. Group Masks and Semantics

To simplify operations, higher-level masks are defined as bitwise OR
combinations of single-bit flags:

- Physical categories:
  - `BPM_MASK_PIX`: all pixel-level bits.
  - `BPM_MASK_COL`: all column-level bits.
  - `BPM_MASK_ROW`: all row-level bits.

- Semantic categories:
  - `BPM_MASK_PERMANENT`: all permanent silicon-related defects (PIX_* and
    COL_*/ROW_* that represent real hardware issues).
  - `BPM_MASK_TRANSIENT`: transient per-frame artifacts
    (`SATURATED`, `PIX_COSMIC`, `PIX_XTALK`, `PIX_BLOOM`, `PIX_OUTLIER`).
  - `BPM_MASK_HOT`: high-signal anomalies (e.g. `PIX_HOT`, `PIX_BRIGHT`,
    `PIX_QE_HIGH`, `COL_BRIGHT`).
  - `BPM_MASK_COLD`: dead/cold behavior (`PIX_DEAD`, `PIX_QE_LOW`,
    `COL_DARK`, `COL_BLOCKED`).
  - `BPM_MASK_NOISE`: temporal/read noise (`PIX_NOISE`, `PIX_RTS`,
    `PIX_XTALK`).
  - `BPM_MASK_TRANSPORT`: charge-transport issues (`PIX_TRAP`, `COL_TRAP`,
    `COL_BLOCKED`, `COL_CTE_LOW`).
  - `BPM_MASK_BAD`: pixels/rows/cols explicitly marked “bad”
    (`PIX_BAD`, `ROW_BAD`, `COL_BAD`).
  - `BPM_MASK_TRAPS`: all trap-related bits.
  - `BPM_MASK_RESPONSE`: response anomalies relevant to photometry
    (`PIX_DEAD`, `SATURATED`, `PIX_BLOOM`).
  - `BPM_MASK_IGNORE`: excluded from statistics
    (`PIX_EXCLUDED`, `COL_EXCLUDED`, `ROW_EXCLUDED`).

**Design choice:** `BPM_MASK_PERMANENT` and `BPM_MASK_TRANSIENT` do *not*
include the “EXCLUDED” flags. Ignoring excluded pixels is an explicit choice,
using `BPM_MASK_IGNORE`, so that you can distinguish between “physically bad”
and “excluded for a specific analysis”.

---

## 3. Detection Pipeline Design

The detection pipeline is implemented in `fits_tools.defects.detection` and
organised into stages. Each stage:

1. Computes robust statistics from a “clean” region (optionally ROI + sigma clipping).
2. Applies thresholds to the **full frame**.
3. Sets the appropriate bit(s) on a stage-specific BPM.
4. Returns a BPM and a statistics dictionary.
5. Is run in order, with earlier stages excluded from later statistics.

High-level stages:

1. **Bias defects** (`detect_bias_defects`)
   - Input: master bias.
   - Goal: detect pixels with anomalous offset (high or low).
   - Statistics: median and scale from ROI or full frame.
   - Default: absolute offset threshold in ADU (with optional sigma mode).
   - Flags: `PIX_BRIGHT`.

2. **Dark hot pixels** (`detect_dark_hot_pixels`)
   - Input: bias-subtracted master dark.
   - Goal: detect hot pixels with elevated dark current.
   - Statistics: median and sigma from clean pixels (excluding bias defects).
   - Default: high-side threshold at Nσ above median.
   - Flags: `PIX_HOT`.
   - Rationale: cold/dead pixels are better captured in flats.

3. **Flat QE defects** (`detect_flat_defects`)
   - Input: bias+dark-corrected master flat.
   - Goal: detect abnormal quantum efficiency (hot and cold/dead).
   - Statistics: robust median and sigma, optionally normalized.
   - Default: ±20% deviation from median.
   - Flags:
     - `PIX_QE_HIGH` for “flat hot”.
     - `PIX_QE_LOW` for “flat cold/dead”.

4. **Structural rows/columns** (`detect_bad_rows_cols`)
   - Input: cumulative BPM from previous stages.
   - Goal: promote high defect-density rows/columns to structural flags.
   - Statistics: per-row and per-column defect fractions.
   - Default: >10% bad pixels in a row/column.
   - Flags: `ROW_BAD`, `COL_BAD`.

5. **Temporal RTS/telegraph noise** (`detect_temporal_defects`)
   - Input: stack of frames of the same type (e.g. darks).
   - Goal: detect pixels with excessive temporal variance.
   - Statistics:
     - per-pixel temporal std,
     - distribution of std over pixels (excluding known-bad).
   - Threshold: `median_std + threshold_sigma * robust_std`.
   - Flags: `PIX_RTS`.

6. **Saturation** (`detect_saturated_pixels`)
   - Input: single frame (e.g. science or flat).
   - Goal: detect pixels at or above saturation.
   - Threshold: fraction of full-well (via ADC bit depth).
   - Flags: `SATURATED`.
   - Note: typically treated as a per-frame DQ mask, not merged
     permanently into the master BPM.

### 3.1 Pipeline orchestration

The high-level entry points are:

- `create_calibration_bpm(bias_frame, dark_frame, flat_frame, config, stats_roi)`
- `create_calibration_bpm_batch(bias_frames, dark_frames, flat_frames, config, stats_roi, combine_method)`

They:

1. Combine raw calibration stacks into master bias/dark/flat (median/mean/sigma-clip).
2. Optionally run temporal detection on the stacks (`detect_temporal_defects`).
3. Run bias, dark, flat, and row/column detection in sequence, accumulating a
   `cumulative_bpm` via bitwise OR.
4. Merge temporal BPMs into the final mask.
5. Compute overall statistics:
   - `total_defects = np.count_nonzero(cumulative_bpm)`
   - `total_fraction = total_defects / cumulative_bpm.size`

The `config` structure produced by `get_default_detection_config()` defines all
thresholds, sigma clipping settings, and whether flats are normalized.

---

## 4. FITS File Format for BPMs

FITS I/O is handled exclusively by `fits_tools.defects.io`.

### 4.1 On-disk structure

A BPM FITS file written by `save_bpm_to_fits` contains:

1. **Primary HDU (`PRIMARY`)**

   - Data: the BPM array (typically `uint32`, 2-D).
   - Header:
     - Provenance: `DATE`, `ORIGIN`, `BPMVERS`, `BITDEPTH`, `BPMDTYPE`.
     - Overall defect counts: `NBADPIX`, `BADFRAC`.
     - Per-bit counts: `HIERARCH BPM.N.<NAME>` for each defined bit name.
     - Detection parameters (if available via `create_calibration_bpm` stats), e.g.:
       - `BIASTHR`, `BIASMED`, `BIASSTD`
       - `DARKSIG`, `DARKMED`, `DARKSTD`
       - `FLATHOT`, `FLATCOLD`, `FLATMED`
     - Bit definition metadata:
       - `HIERARCH BPM.DEF.SCHEMA`
       - `HIERARCH BPM.DEF.NBITS`
       - `HIERARCH BPM.DEF.HASH` — SHA-256 of the bit definitions payload.
       - `HIERARCH BPM.DEF.BIT.xx.{VALUE,NAME,DESC}` — per-bit entries.
     - Human-readable bit descriptions as `COMMENT` cards generated by
       `format_bit_comments()`.

2. **`DQ_DEF` binary table extension**

   - Name: `DQ_DEF`
   - Columns:
     - `BIT` (uint16): bit index (0–31).
     - `VALUE` (int64): bit value (1 << bit).
     - `NAME` (20-char string): symbolic name.
     - `DESC` (80-char string): brief description.
   - Header contains the same `BPM.DEF.*` keywords as the primary.

This makes each BPM file self-describing and verifiable against the Python
implementation.

### 4.2 Calibration set saving

`save_calibration_bpm` saves:

- `{prefix}_bpm.fits` — BPM with full metadata.
- `{prefix}_master_bias.fits`
- `{prefix}_master_dark.fits`
- `{prefix}_master_flat.fits`

Each master frame header includes a `BPMFILE` keyword pointing at the BPM file.

---

## 5. Command-line Interface

The CLI entry point (`fits-defects`) provides:

- `fits-defects create`
  - Inputs: master bias/dark/flat FITS files.
  - Options: thresholds, ROI, output path, optional PDF report.
  - Output: BPM FITS file (and optionally the master frames).
- `fits-defects analyze`
  - Input: BPM FITS file.
  - Output: text summary of total defect fraction and per-bit / per-mask statistics;
    optional JSON export.
  - (Report generation from `analyze` is reserved for future work.)
- `fits-defects compare`
  - Inputs: multiple BPM FITS files.
  - Output: side-by-side textual summary and interactive comparison plots.

The CLI is intentionally thin: it delegates all numerical work to the library,
so behavior matches the Python API exactly.

---

## 6. Summary

The `fits_tools` bitmask and detection pipeline are built around:

- A **clear bit layout** covering pixel-, column/row-, and transient-level defects.
- **Type-safe and self-describing** encoding via `uint32` + FITS metadata.
- A **staged detection pipeline** that:
  - uses robust statistics from clean regions,
  - excludes earlier-stage defects from later-stage statistics, and
  - accumulates defects via bitwise OR.
- A **FITS format** that captures both machine-readable definitions and
  human-readable documentation, suitable for long-term archival and
  cross-language use.

This design aims to be instrument-agnostic but scientifically explicit, making
it appropriate for CCD/CMOS metrology, survey-quality control, and general
astronomical image processing.