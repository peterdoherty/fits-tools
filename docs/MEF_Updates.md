# MEF Integration Handoff

## Purpose

This document records the current multi-extension FITS (MEF) architecture,
completed work, known limitations, and recommended implementation order. It is
intended to let a future maintainer or AI instance continue without repeating
the repository review or reintroducing superseded APIs.

`docs/MEF_Notes.txt` describes the original assessment. Parts of it are now
stale; this file is the current handoff.

## Current verified baseline

At the end of the latest change:

- Ruff passed for `src` and `tests`.
- Pytest passed: **363 tests**.
- Total coverage was **84.53%** (required minimum: 80%).
- The only then-uncommitted files reported by `git status --short` were
  `src/fits_tools/selection.py` and `tests/test_selection.py`. Always inspect
  the current worktree before editing because this may have changed.

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

The implementation currently exposes `_write_operation_result()` and
`_validate_destination_structure()` as private functions, although processing
modules import `_write_operation_result()`. A future cleanup should promote a
stable public commit API rather than spreading imports of private names.

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

`image_processing/calibration.py` still exposes integer-only selectors and uses
one selector for raw science data and every master product.

Migrate singular calibration first:

```python
hdu: HDUSelector
bias_hdu: HDUSelector | None = None
dark_hdu: HDUSelector | None = None
flat_hdu: HDUSelector | None = None
bpm_hdu: HDUSelector | None = None
```

When an auxiliary selector is omitted, derive it from the resolved science
identity, not its numeric index. Each calibration file must resolve independently.

Then add output disposition through `fits_output.py`. Required behavior:

- SEF output remains available.
- Preserve/in-place modes replace only selected SCI data.
- Existing unrelated tables and extensions survive.
- DQ output can be placed in the corresponding `("DQ", EXTVER)` extension.
- Missing DQ behavior must be explicit: error or explicit append, never silently
  append by default.

Plural calibration should be a later step and should validate every SCI/master
pair before committing anything.

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

Also consider moving the resolved-HDU loader out of `operations.py` into a
shared read helper. Avoid creating a dependency from `hdu.py` to I/O or output
modules.

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

Implement singular calibration selection correctly:

1. introduce independent raw/bias/dark/flat/BPM selectors;
2. resolve every file independently;
3. verify extension identities and shapes;
4. retain current SEF behavior;
5. add preserve/in-place output only after the singular association tests pass.

That task provides the largest immediate correctness improvement and creates the
foundation for plural per-amplifier calibration.
