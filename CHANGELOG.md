# Changelog

All notable changes to this project should be documented in this file.

The format is based on Keep a Changelog, and this project uses semantic
versioning for public releases.

## [Unreleased]

### Added

- Added publication metadata files: `CHANGELOG.md`, `CONTRIBUTING.md`,
  `CITATION.cff`, and `MANIFEST.in`.
- Added regression coverage for `combine_images_robust()` with single
  `Path` inputs.

### Changed

- **Behavior change:** `streaming_combine(method="median")` and
  `streaming_calibrated_combine(method="median")` now compute the exact
  per-pixel median by default, processing bands of rows within a
  `max_memory_mb` budget (default 1024) and reading only the needed rows of
  each file. The previous single-pass P² estimate remains available with
  `median_algorithm="p2"`. Output headers record `MEDALG`.
- `streaming_calibrated_combine()` checks master and mask shapes up front,
  and decides saturation rejections in a first pass before combining.
- Expanded the configured mypy check set to include
  `src/fits_tools/image_processing/combining.py`.
- Updated README wording for unimplemented detector APIs.
- Fixed the README quick-start (`StatsResult` fields are `mean_val` and
  `std_val`) and updated the manual's version to 0.2.0.
- Manual: corrected the `combine_images_robust` description (default is
  `sigma_clipped_median`), and documented masks, weights, `NaN` fill,
  raw-ADU saturation, NaN-safe calibration, the approximate streaming median,
  and `streaming_calibrated_combine`.
- `combine_images()` deprecation now refers to 0.2.0 and "a future release"
  instead of nonexistent versions 2.0.0/3.0.0.
- Packaging: added `py.typed` (PEP 561) and `fits_tools.simulation`
  `__init__.py`; stopped shipping internal MEF handoff notes in the sdist;
  removed the unused `docs` extra; removed shebangs and executable bits from
  library modules; aligned author/citation metadata; updated the FSF notice
  in `copyright` to the current URL form.

- **Behavior change:** `apply_calibration()` / `calibrate_image()` now test
  `saturation_level` against raw ADU, before bias/dark subtraction.
  Previously pixels at the raw ADC limit could be missed after bias
  subtraction.
- **Behavior change:** `combine_images_robust()` fills pixels with no valid
  data (masked or clipped in every frame) with `NaN` instead of `0.0`. Use
  the new `fill_value` argument to choose another value.
- **Behavior change:** `combine_images_robust()` raises `ValueError` when
  `weights` are passed with `median` or `sigma_clipped_median`; previously
  they were silently ignored while `WEIGHTED=True` was written.
- `streaming_statistics()` and `streaming_combine()` default to
  `memmap=False`; `compute_stack_reference()` and its helpers now do too, so
  unsigned/scaled (BZERO) camera frames load by default.
- `streaming_calibrated_combine()` now applies `bad_pixel_mask` (masked
  pixels are `NaN` in the result), raises when a frame lacks `EXPTIME` while
  a dark is supplied, logs each frame rejected for saturation, records
  `NREJSAT`/`SATLEVEL` in the output header, and accepts
  `max_saturated_pixels`.

### Fixed

- `combine_images_robust()` now accepts a single `Path` input consistently
  with other `StrOrPath` APIs.
- `P2QuantileEstimator` now follows Jain & Chlamtac: markers move only when
  the neighbouring marker is more than one position away, and the linear
  fallback uses the correct neighbour. Previously markers could collide and
  the worst-pixel median error stayed above 1σ even for 100 frames.
- `P2QuantileEstimator.get_quantile()` returns the requested quantile, not the
  median, when fewer than five samples have been added.
- `combine_images_robust()` honours `weights` for `sigma_clipped_mean`,
  honours `mask` for `median`/`mean`/`weighted_mean`, always returns a plain
  `ndarray`, and no longer counts pre-masked pixels in `NCLIPPED`.
- `_estimate_memory_usage()` reads only headers, so it no longer warns on
  unsigned/scaled FITS files.
- Fixed the ruff (UP035) and mypy failures in `image_processing/streaming.py`.
- `apply_calibration()` now flags non-finite pixels (set to 0, `OUTLIER` bit)
  after all corrections, not only after flat division, so NaN from a master
  bias or dark is flagged even without a flat.
- `apply_calibration()` and `preview_calibration()` compute flat statistics
  over finite pixels only; previously a single NaN in the master flat made
  `normalize_flat=True` turn the whole image NaN. A flat with no finite,
  unmasked pixels now raises `ImageProcessingError`.

## [0.2.0] - 2026-07-22

### Added

- MEF-aware HDU selection, FITS I/O, calibration, image-combining, bad-pixel
  mask, artifact-detection, simulation, statistics, validation, and quality
  control utilities.

[Unreleased]: https://github.com/peterdoherty/fits-tools/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.2.0
