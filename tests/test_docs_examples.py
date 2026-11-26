#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"

"""Execute every Python example in README.md and docs/manual/*.html.

Each page's code blocks run in order in one namespace (later examples may use
names defined by earlier ones), inside a temporary directory populated with
synthetic FITS files for the file names the page mentions. A block preceded
by ``<!-- docs-test: skip -->`` is not executed; use that only for examples
that need optional packages or external data.

Run one page while editing docs with, for example::

    pytest tests/test_docs_examples.py -k operations
"""

from __future__ import annotations

import ast
import contextlib
import html
import importlib
import inspect
import io
import logging
import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO = Path(__file__).resolve().parents[1]
MANUAL = REPO / "docs" / "manual"
SKIP_MARKER = "<!-- docs-test: skip -->"

SHAPE = (1024, 1024)  # large enough for the regions the examples use

_HTML_BLOCK = re.compile(
    r"(?P<skip><!-- docs-test: skip -->\s*)?<pre><code[^>]*>(?P<code>.*?)</code></pre>",
    re.S,
)
_MD_BLOCK = re.compile(
    r"(?P<skip><!-- docs-test: skip -->\s*)?```python\n(?P<code>.*?)```", re.S
)


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------


def _is_python(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _blocks(path: Path) -> list[tuple[int, str, bool]]:
    """Return (index, code, skipped) for each Python block on a page."""
    text = path.read_text()
    pattern = _MD_BLOCK if path.suffix == ".md" else _HTML_BLOCK
    out = []
    for i, match in enumerate(pattern.finditer(text)):
        code = match.group("code")
        if path.suffix != ".md":
            code = html.unescape(code)
        if _is_python(code):
            out.append((i, code, match.group("skip") is not None))
    return out


PAGES = [REPO / "README.md", *sorted(MANUAL.glob("*.html"))]


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(20261125)


def _frame(level: float, noise: float = 5.0) -> np.ndarray:
    return (level + _rng.normal(0.0, noise, SHAPE)).astype(np.float32)


def _header(**cards: object) -> fits.Header:
    header = fits.Header()
    header["DATASEC"] = f"[1:{SHAPE[1]},1:{SHAPE[0]}]"
    header["BIASSEC"] = f"[{SHAPE[1] - 31}:{SHAPE[1]},1:{SHAPE[0]}]"
    header["GAIN"] = 1.5
    header["RDNOISE"] = 5.0
    for key, value in cards.items():
        header[key] = value
    return header


_KINDS: dict[str, tuple[float, dict[str, object]]] = {
    "bias": (100.0, {"IMAGETYP": "BIAS", "EXPTIME": 0.0}),
    "dark": (5.0, {"IMAGETYP": "DARK", "EXPTIME": 10.0}),
    "flat": (20000.0, {"IMAGETYP": "FLAT", "EXPTIME": 1.0}),
    "light": (1000.0, {"IMAGETYP": "LIGHT", "EXPTIME": 10.0, "OBJECT": "M31"}),
}


def _kind_for(name: str) -> str:
    lower = name.lower()
    for kind in ("bias", "dark", "flat"):
        if kind in lower:
            return kind
    return "light"


def _write_simple(path: Path, kind: str) -> None:
    level, cards = _KINDS[kind]
    data = _frame(level)
    if "master_flat" in path.name:
        data = (data / level).astype(np.float32)  # normalized master flat
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data, header=_header(**cards)).writeto(path, overwrite=True)


def _write_mef(path: Path, kind: str, n_sci: int = 2) -> None:
    level, cards = _KINDS[kind]
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
        fits.PrimaryHDU(header=fits.Header(cards))
    ]
    for ver in range(1, n_sci + 1):
        data = _frame(level)
        if "master_flat" in path.name:
            data = (data / level).astype(np.float32)
        hdus.append(
            fits.ImageHDU(data=data, header=_header(**cards), name="SCI", ver=ver)
        )
    hdus.append(fits.ImageHDU(data=np.zeros(SHAPE, dtype=np.uint32), name="DQ", ver=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _write_bpm(path: Path) -> None:
    bpm = np.zeros(SHAPE, dtype=np.uint32)
    bpm[10, 10] = 1
    bpm[20, :] = 2
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=bpm).writeto(path, overwrite=True)


def _make_file(path: Path) -> None:
    name = path.name.lower()
    if "bpm" in name or "mask" in name:
        _write_bpm(path)
    elif "mef" in name or "multi" in name:
        _write_mef(path, _kind_for(name))
    else:
        _write_simple(path, _kind_for(name))


# Names that examples create as outputs; never pre-create these.
_FITS_NAME = re.compile(r"""["']([\w./-]+\.fits)["']""")
_DIR_NAME = re.compile(r"""["']([\w-]+)/["']""")
# Directories the examples read frames from: name -> [(file prefix, kind, MEF?)]
_FRAME_DIRS: dict[str, list[tuple[str, str, bool]]] = {
    "bias_frames": [("bias", "bias", False)],
    "darks": [("dark", "dark", False)],
    "dark_frames": [("dark", "dark", False)],
    "flats": [("flat", "flat", False)],
    "flat_frames": [("flat", "flat", False)],
    "data": [("science", "light", False)],
    "science": [("science", "light", False)],
    "calibrated": [("science", "light", False)],
    "raw": [("science", "light", True)],
    # A mixed night of frames, for the selection helpers
    "raw_data": [
        ("bias", "bias", False),
        ("dark", "dark", False),
        ("flat", "flat", False),
        ("light", "light", False),
    ],
}


def _populate(page: Path, workdir: Path, outputs: set[str]) -> None:
    text = html.unescape(page.read_text())
    for name in sorted(set(_FITS_NAME.findall(text))):
        if name in outputs or "*" in name:
            continue
        target = workdir / name
        if not target.exists():
            _make_file(target)
    for dirname in sorted(set(_DIR_NAME.findall(text))):
        for prefix, kind, mef in _FRAME_DIRS.get(dirname, []):
            for i in range(1, 4):
                target = workdir / dirname / f"{prefix}_{i:03d}.fits"
                if mef:
                    _write_mef(target, kind)
                else:
                    _write_simple(target, kind)


def _is_output_param(name: str, func_name: str = "") -> bool:
    if "output" in name or name in {"filename", "filebase", "out_path"}:
        return True
    # write_simple_fits(filepath, ...), save_bpm_to_fits(bpm, filename, ...)
    writes = func_name.startswith(("write", "save", "export"))
    return writes and name in {"filepath", "path", "file"}


def _imported_callables(tree: ast.AST) -> dict[str, object]:
    """Map local names to fits_tools objects imported in a block."""
    found: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "fits_tools"
        ):
            module = importlib.import_module(node.module or "")
            for alias in node.names:
                obj = getattr(module, alias.name, None)
                if callable(obj):
                    found[alias.asname or alias.name] = obj
    return found


def _output_names(blocks: list[tuple[int, str, bool]]) -> set[str]:
    """File names passed to an output parameter anywhere on the page.

    Arguments are bound to the real function signature, so positional output
    paths are recognised too. These files are never pre-created.
    """
    outputs: set[str] = set()
    callables: dict[str, object] = {}
    for _, code, _ in blocks:
        tree = ast.parse(code)
        callables.update(_imported_callables(tree))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = node.func.id if isinstance(node.func, ast.Name) else ""
            # An output written with overwrite=True may already exist (e.g. an
            # earlier example on the page reads it), so it is pre-created.
            if any(
                kw.arg == "overwrite"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            ):
                continue
            for kw in node.keywords:
                if kw.arg and _is_output_param(kw.arg, func_name):
                    if isinstance(kw.value, ast.Constant) and isinstance(
                        kw.value.value, str
                    ):
                        outputs.add(kw.value.value)
            func = (
                callables.get(node.func.id) if isinstance(node.func, ast.Name) else None
            )
            if func is None:
                continue
            try:
                params = list(inspect.signature(func).parameters)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            for name, arg in zip(params, node.args, strict=False):
                if (
                    _is_output_param(name, node.func.id)
                    and isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                ):
                    outputs.add(arg.value)
    return outputs


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    with (
        contextlib.redirect_stdout(io.StringIO()),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous)


def _run_page(page: Path, workdir: Path) -> list[str]:
    blocks = _blocks(page)
    _populate(page, workdir, _output_names(blocks))
    namespace: dict[str, object] = {"__name__": "__docs__"}
    failures = []
    for index, code, skipped in blocks:
        if skipped:
            continue
        label = f"{page.name} block {index}"
        try:
            with _quiet():
                exec(compile(code, label, "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - report every failing block
            first_line = code.strip().splitlines()[0][:70]
            failures.append(f"{label} ({first_line!r}): {type(exc).__name__}: {exc}")
    return failures


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_documentation_examples_run(
    page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    failures = _run_page(page, tmp_path)
    assert not failures, "Broken documentation examples:\n" + "\n".join(failures)


def test_every_page_has_examples() -> None:
    # Guards against the extraction regex silently matching nothing.
    total = sum(len(_blocks(page)) for page in PAGES)
    assert total > 100


# ---------------------------------------------------------------------------
# API reference page: listed signatures must match the code
# ---------------------------------------------------------------------------

_SECTION = re.compile(r"<h2>(?:<a [^>]*>)?(fits_tools[\w.]*)")
_ENTRY = re.compile(r"<tr><td><code>(\w+)\((.*?)\)</code></td>")


def _split_top_level(args: str) -> list[str]:
    """Split on commas that are not inside brackets, e.g. ``shape=(512, 512)``."""
    parts, depth, current = [], 0, []
    for char in args:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def _reference_entries() -> list[tuple[str, str, list[str]]]:
    text = html.unescape((MANUAL / "reference.html").read_text())
    entries = []
    for section in re.split(r"(?=<h2>)", text):
        match = _SECTION.match(section)
        if not match:
            continue
        for name, args in _ENTRY.findall(section):
            params = []
            for arg in _split_top_level(args):
                arg = arg.strip()
                if not arg or arg in {"...", "*"} or arg.startswith("*"):
                    continue
                params.append(arg.split("=", 1)[0].strip())
            entries.append((match.group(1), name, params))
    return entries


def test_reference_page_signatures_match_code() -> None:
    entries = _reference_entries()
    assert len(entries) > 100
    problems = []
    for module_name, name, params in entries:
        module = importlib.import_module(module_name)
        obj = getattr(module, name, None)
        if obj is None:
            problems.append(f"{module_name}.{name}: not found")
            continue
        real = inspect.signature(obj).parameters
        accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in real.values())
        unknown = [p for p in params if p not in real and not accepts_kwargs]
        if unknown:
            problems.append(f"{module_name}.{name}: unknown parameters {unknown}")
    assert not problems, "Stale API reference entries:\n" + "\n".join(problems)
