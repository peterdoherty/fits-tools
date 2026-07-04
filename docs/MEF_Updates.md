# MEF Integration Handoff

## Purpose

This document records the current multi-extension FITS (MEF) architecture,
completed work, known limitations, and recommended implementation order. It is
intended to let a future maintainer or AI instance continue without repeating
the repository review or reintroducing superseded APIs.

`docs/MEF_Notes.txt` describes the original assessment. Parts of it are now
stale; this file is the current handoff.

## Current verified baseline

Verified on 2026-07-03 (macOS) after the hdu_ops selector migration, the
output-transaction `overwrite` cleanup, calibration provenance/preview fixes,
and the scikit-image compatibility fix:

- Ruff passed for `src` and `tests`.
- Pytest passed: **407 tests**.
- Total coverage was **85.01%** (required minimum: 80%).
- The configured `mypy` gate passed for all 22 listed files, now including
  `resolved_hdu.py`, `fits_output.py`, `io.py`, and
  `image_processing/streaming.py`. `python_version` matches the supported
  minimum (3.11).
- Full-source mypy (unlisted processing modules) still has known errors;
  extend the `files` list module by module as they are cleaned.

Always inspect the current worktree before editing.

Recommended verification commands (Windows PowerShell / POSIX):

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

```bash
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pytest -q
.venv/bin/python -m mypy
.venv/bin/python -m pip check
git diff --check
```

The committed `.venv` must contain the dev extras
(`pip install -e ".[dev,plotting,progress]"`); the dev extra now includes
scikit-image so the sub-pixel alignment path is exercised by tests.

## Architectural rules

Use these rules consistently:

```text
Public API selecting one HDU       -> HDUSelector
Public API selecting several HDUs  -> Iterable[HDUSelector] or HDUQuery
Resolved state inside an operation -> integer HDU index
Cross-file extension association   -> HDUIdentity, preferably (EXTNAME, EXTVER)
Physical atomic FITS writing       -> io.transactional_write_hdulist()
Output placement policy            -> fits_output.py
```

Do not add file access to array/header-only modules merely to support selectors.
Do not pair MEF extensions by numeric position unless the caller explicitly
selected a numeric position.

## HDU selection layer

`src/fits_tools/hdu.py` is the canonical selection module.

It provides:

- `HDUSelector`: index, `EXTNAME`, `(EXTNAME, EXTVER)`, or `"auto"`.
- `HDUIdentity`: index, normalized name/version, HDU type, and shape.
- `HDUQuery`: plural discovery criteria.
- `resolve_hdu_index()`: exact singular resolution with ambiguity rejection.
- `resolve_hdu_indices()`: ordered, deduplicated exact plural resolution.
- `find_hdu_indices()`: query-based discovery in file order.
- `get_hdu_identity()`.
- `is_image_like_hdu()`.

Important semantics:

- A bare duplicated `EXTNAME`, such as `"SCI"`, is an error.
- Use `("SCI", 2)` to disambiguate.
- Booleans are not accepted as integer selectors.
- Automatic image policies require actual image-like HDUs.
- Exact plural resolution and query-based discovery are intentionally separate.

## I/O layer

The following `src/fits_tools/io.py` readers accept `HDUSelector` and resolve it
inside each opened file:

- `get_image_shape()`
- `get_header()`
- `get_data_and_header()`
- `get_data()`

`transactional_write_hdulist()` owns the low-level safe-write mechanism:

1. create a sibling temporary file;
2. write and checksum the complete HDU list;
3. atomically replace the destination with `os.replace`;
4. remove the temporary file after failure.

Keep this function policy-neutral. Scientific decisions about destination
identity and structural compatibility belong in `fits_output.py`.

## Output disposition layer

`src/fits_tools/fits_output.py` centralizes result placement.

Public types:

- `OutputMode = "sef" | "preserve" | "inplace"`
- `StructurePolicy = "target" | "compatible" | "strict"`
- `ResolvedHDU`
- `HDUResult`

Current modes:

- `sef`: create a standalone primary-image result.
- `preserve`: update a matching HDU in an existing destination MEF, or initialize
  a missing destination from the source MEF.
- `inplace`: update the selected source file transactionally.

Current structural policies:

- `target`: require a usable target with matching result shape.
- `compatible`: additionally require source/destination extension identity and
  non-conflicting instrument/detector metadata.
- `strict`: require matching ordered HDU identities/types/shapes for the whole
  source and destination.

`commit_hdu_result()` is the stable public singular commit API used by
operations, filters, and calibration. `_validate_destination_structure()`
remains an internal policy helper.

## Modules already migrated

### `image_processing/operations.py`

- Singular APIs accept `HDUSelector`.
- File operands and masks have independent selectors.
- Supports all output modes, destination-HDU selection, and structure policies.
- `apply_operation_to_hdus()` accepts exact selectors or an `HDUQuery`.
- Binary file operands are associated by extension identity.

Plural processing computes all selected results and commits them through one
all-or-nothing `commit_hdu_results()` transaction.

### `image_processing/filters.py`

- `filter_hdu()` uses central singular resolution.
- `filter_image_hdus()` supports exact selectors and `HDUQuery`.
- Supports shared output disposition.
- Ambiguous names are rejected and `(EXTNAME, EXTVER)` is supported.
- Existing destination MEFs preserve unrelated extensions and tables.

Plural filter results use the same all-or-nothing transaction. A computation,
selection, validation, or physical-write failure leaves the destination
unchanged.

### Read-only and metadata consumers

These APIs now accept `HDUSelector` where they select an HDU:

- `statistics.py`
- `quality_control.py`
- `survey.py`
- `selection.py`

`selection.py` resolves the requested metadata HDU independently in each file.
The default remains primary HDU index 0. Missing, malformed, or ambiguous
selectors currently follow existing scan behavior: the candidate is skipped and
logged instead of aborting the scan.

### Simple processing wrappers

Singular selector annotations were propagated to:

- `image_processing/transformations.py`
- `image_processing/cosmic_rays.py`
- `simulation/noise.py`

These address named/versioned extensions and support shared SEF,
MEF-preserving, and in-place output. Shape-changing transformations explicitly
replace the selected HDU while preserving unrelated structure. Cosmic-ray mask
embedding remains policy-gated; `save_mask=True` is SEF-only until CR_MASK
association and append behavior are defined.

### Structural editing (`hdu_ops.py`)

Migrated to the selector layer. Each public function accepts exact
selectors alongside the legacy parameters:

- `remove_hdus_from_file(..., hdus=[...])` — exact removal with ambiguity
  rejection; `(EXTNAME, EXTVER)` removes exactly one of several same-named
  extensions.
- `add_hdus_to_file(..., source_hdus=[...])` — exact selection from the
  source file.
- `reorder_hdus_in_file(..., order=[...])` — complete ordering that must
  place every HDU exactly once with the primary HDU first; incomplete or
  duplicate orders are rejected instead of silently dropping extensions.

The legacy name-list parameters remain but their duplicate-name behavior
(remove/copy **all** matches; positional pairing in reorder) is documented
loudly in the module and per-parameter docstrings. Selector and legacy
selection are mutually exclusive per call.

### Modules intentionally unchanged

- `sections.py` operates on headers, arrays, shapes, and `Section`; selection
  belongs to its caller.
- `bitmask.py` operates on arrays and flags; SCI/DQ file association belongs in
  orchestration code.
- Internal parameters that already represent a resolved index or an actual HDU
  object should remain integers/objects rather than becoming `HDUSelector`.

## Highest-priority remaining work

### 1. Calibration

`image_processing/calibration.py` now supports singular MEF-aware calibration:

```python
hdu: HDUSelector
bias_hdu: HDUSelector | None = None
dark_hdu: HDUSelector | None = None
flat_hdu: HDUSelector | None = None
bpm_hdu: HDUSelector | None = None
```

Each calibration file resolves independently. Explicit auxiliary selectors
override inheritance, while omitted selectors follow the contract below.

The selector-inheritance contract is:

- an explicitly supplied auxiliary selector always wins;
- an explicit numeric science selector retains the same numeric position for
  backward compatibility;
- a science selector by name, `(EXTNAME, EXTVER)`, or `"auto"` is first
  resolved in the raw file and then converted to the most exact identity
  available for each calibration product;
- a corresponding DQ selector uses `("DQ", EXTVER)` when the resolved science
  HDU has `EXTVER`, or bare `"DQ"` otherwise;
- a derived selector is resolved independently in every file, so a missing or
  ambiguous match is an error rather than a positional fallback.

`src/fits_tools/resolved_hdu.py` implements this contract through
`derive_associated_selector()` and provides the shared `load_resolved_hdu()`
reader. The loader opens one file, resolves its selector, and returns detached
data, header, index, and `HDUIdentity`. Processing modules should use it rather
than creating module-local resolved-HDU loaders.

Singular output disposition is integrated through `commit_hdu_result()`:

- SEF output remains the default.
- Preserve/in-place modes replace only the selected science HDU.
- Existing unrelated tables and extensions survive.
- DQ output remains a sidecar for now; embedding DQ is deferred until science
  and DQ can be committed atomically with the plural result API.

`preview_calibration()` and `calibrate_batch()` now support independent
selectors and the same inheritance rules. Batch calibration resolves a
reference identity from the first readable raw file, loads each master once
as a `ResolvedHDU`, and verifies later non-numeric raw selections have the
same extension identity and shape before reusing the cached arrays.

Master products (`master_bias`, `master_dark`, `master_flat`, `master_bpm`)
accept a path, a bare array, or a `ResolvedHDU` (the `MasterInput` alias).
A `ResolvedHDU` master carries its data, header, and originating path, so
batch calibration records `BIASFILE`/`BIASSUM` (and dark/flat/BPM
equivalents) in every output while still loading each master only once.
Bare arrays carry no provenance and produce no `*FILE`/`*SUM` keywords.

`preview_calibration()` skips dark correction — and omits the
`'dark_subtracted'` stage — under exactly the same conditions as
`apply_calibration()`: missing raw `EXPTIME`, missing dark `EXPTIME`, or a
dark `EXPTIME` of zero. Preview output therefore matches the final
calibration.

`calibrate_hdus()` provides plural calibration with exact selectors or an
`HDUQuery`. It computes every science/master pairing before writing and commits
all calibrated science HDUs through one `commit_hdu_results()` transaction.
Returned `CalibrationResult` objects retain their DQ arrays in memory; DQ
extensions are not guessed or inserted implicitly.

### 2. Combining

Singular file combining now:

- accepts `HDUSelector`;
- resolves the selector independently in every input;
- verifies normalized extension name/version, HDU type, and shape;
- supports named/versioned selection across reordered MEFs;
- rejects missing, ambiguous, or mismatched selections for the complete stack
  rather than silently skipping files;
- retains standalone primary-image output.

`combine_image_hdus()` provides plural combining. It discovers identities from
the first input, resolves each identity independently in every input file,
constructs one stack per identity, and commits all combined products
atomically. Query result order is never used for cross-file association.

### 3. Streaming

`image_processing/streaming.py` now:

- accepts `HDUSelector` in public file-based APIs;
- resolves independently inside every open-file context;
- keeps resolved integer indices local to those contexts;
- performs header-based selection before accessing pixel data;
- preserves memory-mapping behavior;
- verifies extension name/version, HDU type, and shape before accumulation;
- supports named/versioned selection across reordered MEFs.

### 4. Other remaining integer-only file APIs

`defects/io.py` now accepts `HDUSelector` and resolves named/versioned BPM
extensions centrally. Continue auditing newly introduced wrappers with:

```powershell
rg -n "hdu\s*:\s*int|hdul\[hdu\]" src\fits_tools -g "*.py"
```

Do not change exception fields, resolved `hdu_index` parameters, or pure-array
functions just because this search finds an integer.

## Shared output transactions

`fits_output.py` provides the public multi-result transaction:

```python
commit_hdu_results(
    results=[
        HDUResult(source=..., destination_hdu=..., data=..., header=...),
        ...
    ],
    output_file=destination_file,
    output_mode="preserve",
    structure_policy="compatible",
)
```

It:

1. resolve and validate every destination target;
2. reject duplicate destination targets;
3. update one detached `HDUList`;
4. write once with `transactional_write_hdulist()`;
5. leave the original destination unchanged after any validation or write error.

There is deliberately no `overwrite` parameter on `commit_hdu_results()`:
preserve mode updates an existing destination MEF (or initializes a missing
one) by design, and in-place mode rewrites the source by definition. The
singular `commit_hdu_result()` keeps `overwrite`, which applies only to SEF
output. The plural public APIs (`calibrate_hdus`, `combine_image_hdus`,
`filter_image_hdus`, `apply_operation_to_hdus`) therefore no longer accept
an `overwrite` argument.

Plural filters and operations use this API. Multi-result SEF output is
intentionally unsupported because a standalone primary HDU cannot represent
several selected extension identities.

The resolved-HDU loader now lives in `resolved_hdu.py`; keep it there rather
than creating a dependency from `hdu.py` to I/O or output modules.

## SCI/DQ convention still needed

The repository does not yet define a complete MEF-native data-quality policy.
Adopt and document at least:

- SCI and DQ association by identical `EXTVER`.
- DQ extension name (`DQ`) and integer dtype expectations.
- Whether absent `EXTVER` may be inferred.
- Whether missing DQ is an error or may be explicitly appended.
- Bitwise merge behavior for existing DQ data.
- Provenance and HISTORY placement.
- Shape/WCS compatibility requirements.

Do not encode implicit SCI-to-DQ guesses until this convention is agreed.

## End-to-end MEF fixture

`tests/conftest.py` now provides a reusable integration fixture containing:

- an empty primary HDU;
- `SCI, EXTVER=1` and `SCI, EXTVER=2`;
- matching `DQ, EXTVER=1/2`;
- one compressed image extension;
- a binary table that must survive processing;
- per-extension WCS and detector metadata;
- different amplifier shapes;
- checksums.

Cross-module acceptance tests exercise structure, WCS, compressed-HDU, and
checksum preservation. Reordered variants and focused tests cover:

- exact and query selection;
- ambiguous bare `SCI`;
- missing partner;
- duplicate identity;
- incompatible shapes;
- existing destination MEF;
- strict-structure mismatch;
- rollback after a late plural failure;
- Windows in-place replacement with all read handles closed;
- checksum validity after commit.

## Compatibility and documentation cleanup

- DONE: `hdu : int` docstrings updated to `HDUSelector` across `io.py`,
  `statistics.py`, `survey.py`, `quality_control.py`, `operations.py`, and
  `transformations.py`. The `max_hdu : int` fields in `exceptions.py` are
  resolved indices and intentionally remain integers.
- DONE: `apply_operation_to_hdus()` is exported from
  `image_processing/__init__.py`.
- DONE: `align_frames_cross_correlation()` no longer passes `return_error`
  to `skimage.registration.phase_cross_correlation` (removed in
  scikit-image 0.22); scikit-image is declared in the `alignment` and `dev`
  extras.
- Update the user manual with selector examples and output modes.
- Replace stale recommendations for `preserve_mef=True` with
  `output_mode="preserve"`.
- Explain that `filter_hdu()` defaults to preserve-style output, while operation
  APIs retain SEF defaults for backward compatibility.

## Working practices for the next implementation

1. Inspect `git status` and preserve unrelated user changes.
2. Make one coherent module migration at a time.
3. Add duplicate-name and reordered-extension tests with every migration.
4. Test on Windows-compatible atomic replacement paths; close FITS handles
   before `os.replace`.
5. Run focused tests first, then the complete validation commands listed above.
6. Update this handoff after each completed stage.

## Suggested next task

The end-to-end MEF fixture and cross-module acceptance coverage exist. The
highest-value remaining work, in order:

1. Define and document the SCI/DQ convention (see the section above); it
   blocks embedded-DQ output for calibration.
2. Update the user manual with selector examples and output modes.
3. Continue extending the mypy `files` gate: `calibration.py`,
   `combining.py`, `filters.py`, and `operations.py` each have roughly
   15 errors remaining.
4. Consider an `output_mode` parameter for `calibrate_batch()`, which is
   currently SEF-only.
