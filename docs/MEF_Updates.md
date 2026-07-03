# MEF Integration Handoff

## Purpose

This document records the current multi-extension FITS (MEF) architecture,
completed work, known limitations, and recommended implementation order. It is
intended to let a future maintainer or AI instance continue without repeating
the repository review or reintroducing superseded APIs.

`docs/MEF_Notes.txt` describes the original assessment. Parts of it are now
stale; this file is the current handoff.

## Current verified baseline

Verified on 2026-07-03 after introducing the shared resolved-HDU loader and
selector-inheritance contract:

- Ruff passed for `src` and `tests`.
- Pytest passed: **374 tests**.
- Total coverage was **84.55%** (required minimum: 80%).
- Targeted mypy passed for `resolved_hdu.py` and `fits_output.py`.
- Full-source mypy is not currently a green gate: it reports 209 existing
  errors across 15 files.

Always inspect the current worktree before editing.

Recommended verification commands:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

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

Known limitation: plural processing commits one extension at a time. It is not
yet a single all-or-nothing transaction across the whole selected set.

### `image_processing/filters.py`

- `filter_hdu()` uses central singular resolution.
- `filter_image_hdus()` supports exact selectors and `HDUQuery`.
- Supports shared output disposition.
- Ambiguous names are rejected and `(EXTNAME, EXTVER)` is supported.
- Existing destination MEFs preserve unrelated extensions and tables.

Known limitation: plural filter results are also committed sequentially. A
failure after an earlier extension commit can leave a partial plural update.
The future shared output API should accept several result payloads, validate all
targets first, update one in-memory destination, and commit once.

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

These can address named/versioned extensions through the upgraded I/O readers,
but they do **not** yet have complete MEF-preserving output integration.

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

`preview_calibration()` and `calibrate_batch()` still need independent selector
propagation. Plural calibration remains later work and must validate every
SCI/master pair before committing anything.

### 2. Combining

`image_processing/combining.py` still uses numeric HDU positions.

Required migration:

- Accept `HDUSelector`.
- Resolve the selector independently in every input.
- Verify `HDUIdentity` agreement across files.
- Reject reordered/missing/mismatched extensions clearly.
- Use the first input only as the structural template when preserve output is
  requested.

For plural combining, construct identity-keyed stacks and pair by
`(EXTNAME, EXTVER)`, never by query result order.

### 3. Streaming

`image_processing/streaming.py` still directly indexes `HDUList` objects using
integer `hdu` parameters.

Required migration:

- Accept `HDUSelector` in public APIs.
- Resolve independently per input file.
- Keep the resolved integer local to each open-file context.
- Preserve memory-mapping behavior and avoid materializing data merely for
  selection.
- Verify identities and shapes before accumulation.

### 4. Other remaining integer-only file APIs

Review and migrate as appropriate:

- `defects/io.py`
- any newly introduced file wrappers found with:

```powershell
rg -n "hdu\s*:\s*int|hdul\[hdu\]" src\fits_tools -g "*.py"
```

Do not change exception fields, resolved `hdu_index` parameters, or pure-array
functions just because this search finds an integer.

## Shared output improvements

Before expanding plural MEF workflows, improve `fits_output.py` with a public
multi-result transaction, conceptually:

```python
commit_hdu_results(
    source_file,
    destination_file,
    results=[
        HDUResult(source_identity=..., destination_selector=..., data=..., header=...),
        ...
    ],
    output_mode="preserve",
    structure_policy="compatible",
)
```

It should:

1. resolve and validate every destination target;
2. reject duplicate destination targets;
3. update one detached `HDUList`;
4. write once with `transactional_write_hdulist()`;
5. leave the original destination unchanged after any validation or write error.

Use this to replace sequential plural commits in filters and operations.

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

Create one reusable integration fixture containing:

- an empty primary HDU;
- `SCI, EXTVER=1` and `SCI, EXTVER=2`;
- matching `DQ, EXTVER=1/2`;
- one compressed image extension;
- a binary table that must survive processing;
- per-extension WCS and detector metadata;
- different amplifier shapes;
- checksums.

Use reordered versions of this fixture for cross-file tests. Required scenarios:

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

- Several docstrings still say `hdu : int` after their signatures migrated to
  `HDUSelector`; update them.
- Update the user manual with selector examples and output modes.
- Replace stale recommendations for `preserve_mef=True` with
  `output_mode="preserve"`.
- Explain that `filter_hdu()` defaults to preserve-style output, while operation
  APIs retain SEF defaults for backward compatibility.
- Add `apply_operation_to_hdus()` to module exports if it is intended as public.

## Working practices for the next implementation

1. Inspect `git status` and preserve unrelated user changes.
2. Make one coherent module migration at a time.
3. Add duplicate-name and reordered-extension tests with every migration.
4. Test on Windows-compatible atomic replacement paths; close FITS handles
   before `os.replace`.
5. Run focused tests first, then the complete validation commands listed above.
6. Update this handoff after each completed stage.

## Suggested next task

Propagate the singular selector contract through `preview_calibration()` and
`calibrate_batch()`:

1. add independent raw/bias/dark/flat/BPM selectors where applicable;
2. resolve file-based products independently using `load_resolved_hdu()`;
3. retain array-input and batch caching behavior;
4. add reordered-extension and duplicate-name tests;
5. keep batch output SEF-compatible before adding plural calibration.
