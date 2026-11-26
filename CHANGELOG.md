# Changelog

All notable changes to this project should be documented in this file.

The format is based on Keep a Changelog, and this project uses semantic
versioning for public releases.

## [Unreleased]

## [0.7.0] - 2026-11-25

`combine_images_robust()` is renamed `combine_images()` (the old name still
works, with a `FutureWarning`), the legacy `combine_images()` is removed,
`calibrate_batch()` can calibrate several frames at once, and
`preview_calibration()` matches `calibrate_image()` for bad-pixel masks and
regions. Read **Changed** before upgrading.

### Added

- `calibrate_batch(num_workers=N)` calibrates N frames at a time in threads,
  sharing the master frames; results keep the input order. On 24
  2048x2048 frames: ~2.5x faster for plain calibration, ~4x with
  `validate_inputs` and `compute_quality` (4-8 workers).
- `preview_calibration(master_bpm=..., bpm_hdu=...)`: flagged pixels are
  excluded from the flat normalization, as in `calibrate_image`, and the
  mask is returned as `preview["bpm"]`.

### Changed

- `combine_images_robust()` is renamed `combine_images()`, now that the old
  legacy function of that name is gone. The old name still works for now: it
  emits a `FutureWarning` (shown by default) and calls `combine_images()`.
  It will be removed in a future release. Note that an old-style legacy call
  such as `combine_images(files, "median")` now runs the new function and
  returns a `CombiningResult` rather than an array.

### Removed

- The deprecated legacy `combine_images()` (which returned `None` on
  errors); its name now belongs to the former `combine_images_robust()`.

### Fixed

- `preview_calibration(region=..., normalize_flat=True)` normalized the flat
  by the region's own median instead of the whole frame's, so its
  flat-corrected values did not match a full calibration of the same pixels.

## [0.6.0] - 2026-11-25

Statistics result fields use plain names (`mean`, `std`, `min`, ...) instead
of the `_val` suffix, and the code base is formatted with `ruff format`.
Read **Behavior changes** before upgrading.

### Behavior changes

- Statistics result fields now use plain names, matching `RobustStatsResult`,
  `QualityMetrics`, and every statistics dictionary in the package:
  `StatsResult.mean_val`, `median_val`, `std_val`, `variance_val`,
  `min_val`, `max_val`, and `range_val` are now `mean`, `median`, `std`,
  `variance`, `min`, `max`, and `range`, and `QualityMetrics.min_val` /
  `max_val` are now `min` / `max`. `StatsResult.to_dict()` keys change the
  same way. Code using the old names fails with `AttributeError`.

### Changed

- Source and tests are formatted with `ruff format`; CI and the pre-push
  hook now check formatting.

## [0.5.0] - 2026-11-25

Every median, quantile, and MAD in the streaming code is now exact by
default, with memory bounded by `max_memory_mb` rather than by the number of
frames. The approximate P² estimator is used only when requested. Read
**Behavior changes** before upgrading.

### Behavior changes

- `compute_stack_reference()` now computes the **exact** per-pixel median and
  MAD-based sigma by default, in one banded pass with memory bounded by the
  new `max_memory_mb` (default 1024), for FITS, `.npy`, and `.npz` frames.
  Previously stacks of more than 50 frames got the approximate P² estimate,
  and smaller stacks (or any call with `existing_bpm`) loaded every frame
  into memory at once. `method="p2"` keeps the approximate estimate;
  `"auto"`, `"chunked"`, and `"streaming"` are aliases for `"exact"`, and
  `chunk_size` is ignored.
- `compute_stack_reference(existing_bpm=...)` sets masked pixels to `NaN` in
  both outputs (no valid data), instead of `0`, which gave `sigma=0`.
- `streaming_statistics(compute_quantiles=True)` now computes **exact**
  quantiles by default (NumPy's linear interpolation), with mean, variance,
  min, max, and sum computed in the same banded pass; memory is bounded by
  the new `max_memory_mb`. `quantile_algorithm="p2"` keeps the approximate
  single-pass estimate. Without quantiles the single-pass path is unchanged.
  The P² estimator is now used only when explicitly requested.

## [0.4.0] - 2026-11-25

Calibration is now strict about master darks it cannot exposure-scale, and
the manual's examples are corrected and tested on every run. Read
**Behavior changes** before upgrading.

### Behavior changes

- Calibration is now **strict** by default: `apply_calibration()`,
  `calibrate_image()`, `calibrate_hdus()`, `calibrate_batch()`, and
  `preview_calibration()` raise `ImageProcessingError` when a master dark is
  supplied but cannot be exposure-scaled (raw or dark `EXPTIME` missing,
  dark `EXPTIME` <= 0, or an array dark without `master_dark_header`).
  Previously the dark was skipped (or equal exposure times assumed) with
  only a log warning. Pass `strict=False` for the old behavior; skipped darks
  are then recorded in the output header's HISTORY. In `calibrate_batch()`,
  frames that fail this check are recorded as failures in the batch result.

### Added

- `tests/test_docs_examples.py` runs every Python example in `README.md` and
  `docs/manual/*.html` against synthetic FITS data, and checks every
  signature on the API reference page against the code. Examples that need
  optional packages are marked `<!-- docs-test: skip -->`.

### Changed

- README install instructions now install from GitHub (the package is not
  on PyPI), pinned to a release tag, including the extras syntax.
- Manual: rewrote the examples that no longer matched the API (about 40
  across operations, transformations, cosmic rays, bad-pixel masks, defect
  detection, and quality control), fixed `write_simple_fits` argument order
  in Getting Started and I/O, and regenerated every signature on the API
  reference page from the code (about 90 were stale; seven listed functions
  no longer existed).

### Fixed

- `apply_bpm()` failed with its default `fill_value=NaN` on float32 images,
  which is how most FITS images load; the fill value is now checked by value
  (NaN or any in-range float fits a float array; integers must fit the
  integer type).
- `create_master_frame()` wrote the output file before adding `FRMTYPE` and
  its HISTORY line, so saved masters lacked them unless `normalize=True`. The
  file is now written once, after the header is final.
- `create_master_frame(normalize=True)` ignores `NaN` pixels (from
  `combine_images_robust` masking) when computing the normalization median;
  previously the median was `NaN`.
- `detect_defect_clusters()` returns the same keys when no defects are found.

## [0.3.0] - 2026-11-25

This release fixes several bugs that could silently produce wrong numbers in
calibration and image combining. Some fixes change results; read
**Behavior changes** before upgrading.

### Behavior changes

- `streaming_combine(method="median")` and `streaming_calibrated_combine()`
  now compute the **exact** per-pixel median by default (previously an
  approximate P² estimate). Use `median_algorithm="p2"` for the old
  behavior.
- `apply_calibration()` / `calibrate_image()` test `saturation_level`
  against **raw** ADU, before bias/dark subtraction. Previously a pixel at
  the raw ADC limit could be missed once the bias was subtracted.
- `combine_images_robust()` sets pixels with no valid data (masked or
  clipped in every frame) to `NaN` instead of `0.0`. Use the new
  `fill_value` argument to choose another value.
- `combine_images_robust()` raises `ValueError` when `weights` are passed
  with `median` or `sigma_clipped_median`. Previously they were silently
  ignored while the header still said `WEIGHTED=True`.
- `streaming_calibrated_combine()` raises when a frame has no `EXPTIME`
  while a dark is supplied (previously the dark was silently skipped), and
  applies `bad_pixel_mask` (masked pixels are `NaN`; previously the mask was
  ignored).
- `calibrate_image()` and `apply_calibration()` set any `NaN`/infinite
  result pixel to 0 and flag it `OUTLIER`, whichever step produced it
  (previously only after flat division).
- `streaming_statistics()`, `streaming_combine()`, and
  `compute_stack_reference()` default to `memmap=False`, so unsigned/scaled
  (BZERO) camera frames load by default.

### Added

- `median_algorithm` and `max_memory_mb` options for `streaming_combine()`
  and `streaming_calibrated_combine()`; output headers record `MEDALG`.
- `max_saturated_pixels` option for `streaming_calibrated_combine()`;
  rejected frames are logged by filename and counted in `NREJSAT`
  (with `SATLEVEL`) in the output header.
- `fill_value` option for `combine_images_robust()`.
- `py.typed` marker (PEP 561), so type checkers use the package's type
  hints.
- `fits_tools.simulation` is now a regular package with an `__init__.py`.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `CITATION.cff`, and `MANIFEST.in`.

### Changed

- `combine_images_robust()` applies `weights` for `sigma_clipped_mean`,
  honours `mask` for every method, and always returns a plain `ndarray`.
- `streaming_calibrated_combine()` checks master and mask shapes before
  reading frames, and makes saturation rejections in a first pass.
- Manual: documented masks, weights, `NaN` fill, raw-ADU saturation,
  NaN-safe calibration, the exact streaming median, and
  `streaming_calibrated_combine()`; corrected the `combine_images_robust()`
  description (its default is `sigma_clipped_median`).
- `combine_images()` deprecation notice now refers to 0.2.0 and "a future
  release" instead of nonexistent versions 2.0.0/3.0.0.
- Author and citation metadata aligned; `copyright` uses the current GNU
  license URL.

### Removed

- The unused `docs` extra (it installed Sphinx, which nothing used).
- Internal MEF development notes are no longer shipped in the source
  distribution.

### Fixed

- `P2QuantileEstimator` now follows Jain & Chlamtac (1985): markers can no
  longer collide, and the linear fallback uses the correct neighbour.
  Previously the worst-pixel median error stayed above 1σ even with 100
  frames. With fewer than five samples, `get_quantile()` returns the
  requested quantile rather than the median.
- A single `NaN` in a master flat no longer turns the whole image `NaN`
  when `normalize_flat=True` (`apply_calibration()` and
  `preview_calibration()`); a flat with no finite pixels raises
  `ImageProcessingError`.
- `combine_images_robust()` no longer counts pre-masked pixels in
  `NCLIPPED`, and accepts a single `Path` input.
- `combine_images_robust()` honoured masks inconsistently: `median`,
  `mean`, and `weighted_mean` returned a `MaskedArray` whose mask was lost
  when written to FITS.
- Streaming memory estimates no longer log spurious warnings for
  unsigned/scaled FITS files.
- README quick-start used nonexistent `StatsResult` fields (`mean`/`std`
  instead of `mean_val`/`std_val`).
- Removed shebangs and executable bits from library modules.

## [0.2.0] - 2026-07-22

### Added

- MEF-aware HDU selection, FITS I/O, calibration, image-combining, bad-pixel
  mask, artifact-detection, simulation, statistics, validation, and quality
  control utilities.

[Unreleased]: https://github.com/peterdoherty/fits-tools/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.7.0
[0.6.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.6.0
[0.5.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.5.0
[0.4.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.4.0
[0.3.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.3.0
[0.2.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.2.0
