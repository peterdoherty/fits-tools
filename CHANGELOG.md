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

- Expanded the configured mypy check set to include
  `src/fits_tools/image_processing/combining.py`.
- Updated README wording for unimplemented detector APIs.

### Fixed

- `combine_images_robust()` now accepts a single `Path` input consistently
  with other `StrOrPath` APIs.

## [0.2.0] - 2026-07-22

### Added

- MEF-aware HDU selection, FITS I/O, calibration, image-combining, bad-pixel
  mask, artifact-detection, simulation, statistics, validation, and quality
  control utilities.

[Unreleased]: https://github.com/peterdoherty/fits-tools/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/peterdoherty/fits-tools/releases/tag/v0.2.0
