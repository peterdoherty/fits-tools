# Contributing

Thanks for helping improve `fits-tools`. This project handles scientific
FITS data, so contributions should favor correctness, explicit behavior, and
clear provenance over cleverness.

## Development Setup

```bash
git clone https://github.com/peterdoherty/fits-tools.git
cd fits-tools
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

Run these before opening a pull request:

```bash
ruff check src tests
mypy
pytest -q
python -m build --wheel
```

These are the same checks CI runs (`.github/workflows/ci.yml`).

The test configuration enforces strict pytest settings and a minimum coverage
threshold. Add focused regression tests for bug fixes and broader tests for
changes that affect shared FITS, HDU, calibration, or mask behavior.

## Code Guidelines

- Preserve existing public API behavior unless the change is intentional and
  documented.
- Keep MEF files and HDU identity handling explicit; avoid relying on
  extension order when an `EXTNAME`/`EXTVER` identity is available.
- Use structured FITS/Astropy APIs instead of ad hoc header or byte parsing.
- Keep scientific placeholder APIs clearly labeled. Do not present draft or
  unvalidated algorithms as production-ready detectors.
- Prefer small, reviewable changes with tests and documentation updates.

## Documentation

Update README or `docs/manual/` when a public behavior, option, return type,
or scientific caveat changes. For user-visible detector behavior, make the
warning or exception behavior explicit.

## Releases

Before publishing a release:

1. Update `CHANGELOG.md`.
2. Confirm `pyproject.toml` version and package metadata.
3. Run `ruff check src tests`, `mypy`, `pytest -q`, and `python -m build`.
4. Build and inspect the source distribution and wheel.
