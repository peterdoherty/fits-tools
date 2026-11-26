# Changelog

All notable changes to this project should be documented in this file.

The format is based on Keep a Changelog, and this project uses semantic
versioning for public releases.

## [Unreleased]

### Changed

- README install instructions now install from GitHub (the package is not
  on PyPI), pinned to a release tag, including the extras syntax.

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

[Unreleased]: https://github.com/peterdoherty/fits-tools/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.3.0
[0.2.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.2.0
