"""
Artifact reporting and visualization.

This module provides report-generation helpers for artifact detection outputs
(e.g., from `detect_all_artifacts`). It intentionally contains matplotlib
dependencies and should remain separate from core detection logic so that
headless pipelines can import detection code without pulling in plotting stacks.

Public API
----------
- create_artifact_report: Generate a multi-panel visualization and text summary.
- create_artifact_stats_dict: Build a report-compatible stats dictionary when
  detectors are run individually.

Notes
-----
- Matplotlib is imported lazily inside `create_artifact_report`.
- Frame visualization uses a "safe" normalization: log scaling is used only
  when the data are strictly positive; otherwise a percentile-based linear
  normalization is used.
"""

from __future__ import annotations

import logging
import numpy as np

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from ..bitmask import (
    BPM_BIT_BLOOM,
    BPM_BIT_COSMIC,
    BPM_BIT_OUTLIER,
    BPM_BIT_SATURATED,
)

logger = logging.getLogger(__name__)

__all__ = [
    "create_artifact_report",
    "create_artifact_stats_dict",
]

from .types import StrOrPath


def _validate_artifact_stats(stats: Mapping[str, Any]) -> None:
    """
    Validate that `stats` has the basic structure expected by create_artifact_report().

    This validates presence of required keys and rough type/shape expectations.
    It is intentionally *not* a full schema validation, but catches common errors
    (typos, missing keys, wrong nesting).

    Parameters
    ----------
    stats
        Stats dictionary, typically produced by `detect_all_artifacts()`.

    Raises
    ------
    ValueError
        If required keys are missing or structures are incompatible.
    """
    required_top_level = [
        "saturation",
        "cosmic_rays",
        "bloom",
        "persistence",
        "artifacts_only",
        "artifacts_fraction",
        "total_flagged",
        "total_fraction",
    ]
    for key in required_top_level:
        if key not in stats:
            raise ValueError(
                f"Stats dictionary missing required key {key!r}. "
                f"Required keys: {required_top_level}"
            )

    if not isinstance(stats["saturation"], Mapping):
        raise ValueError("stats['saturation'] must be a mapping/dict.")
    if not isinstance(stats["cosmic_rays"], Mapping):
        raise ValueError("stats['cosmic_rays'] must be a mapping/dict.")
    if not isinstance(stats["bloom"], Mapping):
        raise ValueError("stats['bloom'] must be a mapping/dict.")
    if not isinstance(stats["persistence"], Mapping):
        raise ValueError("stats['persistence'] must be a mapping/dict.")

    # Detector sub-dict key checks
    for k in ("saturation_level", "num_saturated", "fraction_saturated"):
        if k not in stats["saturation"]:
            raise ValueError(f"stats['saturation'] missing required key {k!r}.")

    for k in ("method", "num_cosmic_rays", "fraction"):
        if k not in stats["cosmic_rays"]:
            raise ValueError(f"stats['cosmic_rays'] missing required key {k!r}.")

    for k in (
        "bleed_direction",
        "num_saturated_clusters",
        "num_saturated_total",
        "num_saturated_excluded",
        "num_trails",
        "num_bloom_pixels",
        "fraction",
    ):
        if k not in stats["bloom"]:
            raise ValueError(f"stats['bloom'] missing required key {k!r}.")

    # Scalar type checks (loose but useful)
    if not isinstance(stats["artifacts_only"], (int, np.integer)):
        raise ValueError("stats['artifacts_only'] must be an integer.")
    if not isinstance(stats["total_flagged"], (int, np.integer)):
        raise ValueError("stats['total_flagged'] must be an integer.")
    if not isinstance(stats["artifacts_fraction"], (float, np.floating)):
        raise ValueError("stats['artifacts_fraction'] must be a float.")
    if not isinstance(stats["total_fraction"], (float, np.floating)):
        raise ValueError("stats['total_fraction'] must be a float.")


def create_artifact_stats_dict(
    sat_stats: Mapping[str, Any],
    cr_stats: Mapping[str, Any],
    bloom_stats: Mapping[str, Any],
    pers_stats: Optional[Mapping[str, Any]] = None,
    dq: Optional[np.ndarray] = None,
    frame_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Construct a stats dictionary compatible with `create_artifact_report()`.

    This is a convenience helper for workflows where individual detectors are run
    separately and the caller wants a report-compatible `stats` structure.

    Parameters
    ----------
    sat_stats
        Statistics from saturated-pixel detection.
    cr_stats
        Statistics from cosmic-ray detection.
    bloom_stats
        Statistics from bloom detection.
    pers_stats
        Statistics from persistence detection. If None, persistence is marked as skipped.
    dq
        Combined DQ array. If provided, summary counts/fractions are computed from it.
    frame_size
        Total number of pixels in the frame, used only if `dq` is not provided.
        Note: without `dq`, totals cannot be computed accurately due to overlap
        between detectors, so summary totals are set to 0.

    Returns
    -------
    dict
        Report-compatible stats dictionary.

    Raises
    ------
    ValueError
        If neither `dq` nor `frame_size` is provided.
    """
    out: Dict[str, Any] = {
        "saturation": dict(sat_stats),
        "cosmic_rays": dict(cr_stats),
        "bloom": dict(bloom_stats),
        "persistence": dict(pers_stats) if pers_stats is not None else {
            "skipped": True,
            "reason": "Not run (manual construction)",
        },
    }

    if dq is not None:
        flagged = int(np.sum(dq > 0))
        out["artifacts_only"] = flagged
        out["artifacts_fraction"] = float(flagged) / dq.size
        out["total_flagged"] = flagged
        out["total_fraction"] = float(flagged) / dq.size
    elif frame_size is not None:
        # Without dq, we cannot reliably compute totals due to overlap.
        out["artifacts_only"] = 0
        out["artifacts_fraction"] = 0.0
        out["total_flagged"] = 0
        out["total_fraction"] = 0.0
        out["note"] = (
            "Summary totals set to 0 because dq was not provided; "
            "overlap between detector masks cannot be resolved."
        )
    else:
        raise ValueError("Must provide either 'dq' or 'frame_size'.")

    return out


def _format_summary_text(stats: Mapping[str, Any]) -> str:
    """Format a human-readable text summary for the report."""
    pers = stats.get("persistence", {})
    pers_skipped = bool(pers.get("skipped", False))

    lines = []
    lines.append("ARTIFACT DETECTION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("SATURATION")
    lines.append(f"  Threshold: {stats['saturation']['saturation_level']:.1f} ADU")
    lines.append(
        f"  Flagged:   {stats['saturation']['num_saturated']} pixels "
        f"({stats['saturation']['fraction_saturated']*100:.3f}%)"
    )
    lines.append("")
    lines.append("COSMIC RAYS")
    lines.append(f"  Method:    {stats['cosmic_rays']['method']}")
    lines.append(
        f"  Flagged:   {stats['cosmic_rays']['num_cosmic_rays']} pixels "
        f"({stats['cosmic_rays']['fraction']*100:.3f}%)"
    )
    lines.append("")
    lines.append("BLOOM TRAILS")
    lines.append(f"  Direction:       {stats['bloom']['bleed_direction']}")
    lines.append(f"  Saturated seeds: {stats['bloom']['num_saturated_clusters']} clusters")
    lines.append(f"  Total saturated: {stats['bloom']['num_saturated_total']} pixels")
    lines.append(f"  Excluded (BPM):  {stats['bloom']['num_saturated_excluded']} pixels")
    lines.append(f"  Trails detected: {stats['bloom']['num_trails']}")
    lines.append(
        f"  Flagged:         {stats['bloom']['num_bloom_pixels']} pixels "
        f"({stats['bloom']['fraction']*100:.3f}%)"
    )
    lines.append("")
    lines.append("PERSISTENCE")
    if pers_skipped:
        lines.append("  Status:    Skipped (no reference frame)")
    else:
        lines.append("  Status:    Detected")
        lines.append(f"  Regions:   {pers.get('num_regions', 0)}")
        lines.append(f"  Pixels:    {pers.get('num_pixels', 0)}")
        if "mean_persistence" in pers:
            lines.append(f"  Mean flux: {pers.get('mean_persistence', 0):.1f} ADU")
        if "max_persistence" in pers:
            lines.append(f"  Max flux:  {pers.get('max_persistence', 0):.1f} ADU")
    lines.append("")
    lines.append("OVERALL")
    lines.append(
        f"  Total artifacts: {stats['artifacts_only']} pixels "
        f"({stats['artifacts_fraction']*100:.3f}%)"
    )
    lines.append(
        f"  Total flagged:   {stats['total_flagged']} pixels "
        f"({stats['total_fraction']*100:.3f}%)"
    )

    return "\n".join(lines)


def create_artifact_report(
    frame: np.ndarray,
    dq: np.ndarray,
    stats: Mapping[str, Any],
    output_path: Optional[StrOrPath] = None,
    show_plots: bool = True,
    validate_stats: bool = True,
) -> None:
    """
    Generate a multi-panel artifact detection report with a text summary.

    Parameters
    ----------
    frame
        Original science frame.
    dq
        DQ array from artifact detection.
    stats
        Stats dictionary, typically from `detect_all_artifacts()`.
    output_path
        If provided, save the report figure to this path (PNG/PDF/etc).
    show_plots
        If True, display the figure interactively via matplotlib.
    validate_stats
        If True, validate the `stats` structure before generating the report.

    Notes
    -----
    This function imports matplotlib lazily and is intended to live outside
    core detection logic.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    if validate_stats:
        _validate_artifact_stats(stats)

    if frame.shape != dq.shape:
        raise ValueError(f"frame shape {frame.shape} does not match dq shape {dq.shape}")

    # Safe normalization for display
    finite = np.isfinite(frame)
    if np.any(finite):
        vmin, vmax = np.nanpercentile(frame[finite], [1, 99])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(frame[finite])), float(np.nanmax(frame[finite]))
            if vmax <= vmin:
                vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    # Use log only if data are strictly positive in the displayed range
    if vmin > 0:
        norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Original frame
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(frame, origin="lower", cmap="gray", norm=norm)
    ax1.set_title("Original Frame")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, label="ADU", fraction=0.046)

    # Panel 2: Artifact overlay
    ax2 = fig.add_subplot(gs[0, 1])
    artifact_mask = dq > 0
    overlay = np.ma.masked_where(~artifact_mask, dq)
    ax2.imshow(frame, origin="lower", cmap="gray", norm=norm, alpha=0.7)
    ax2.imshow(overlay, origin="lower", cmap="hot", alpha=0.5)
    ax2.set_title(f"Artifacts Overlay\n({stats['artifacts_fraction']*100:.2f}% flagged)")
    ax2.axis("off")

    # Panel 3: Saturation
    ax3 = fig.add_subplot(gs[0, 2])
    sat_mask = (dq & BPM_BIT_SATURATED) != 0
    ax3.imshow(sat_mask, origin="lower", cmap="Reds", vmin=0, vmax=1)
    ax3.set_title(f"Saturated\n({stats['saturation']['num_saturated']} pixels)")
    ax3.axis("off")

    # Panel 4: Cosmic rays
    ax4 = fig.add_subplot(gs[1, 0])
    cr_mask = (dq & BPM_BIT_COSMIC) != 0
    ax4.imshow(cr_mask, origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax4.set_title(f"Cosmic Rays\n({stats['cosmic_rays']['num_cosmic_rays']} pixels)")
    ax4.axis("off")

    # Panel 5: Bloom
    ax5 = fig.add_subplot(gs[1, 1])
    bloom_mask = (dq & BPM_BIT_BLOOM) != 0
    ax5.imshow(bloom_mask, origin="lower", cmap="Purples", vmin=0, vmax=1)
    num_bloom = stats["bloom"]["num_bloom_pixels"]
    num_trails = stats["bloom"]["num_trails"]
    ax5.set_title(f"Bloom Trails\n({num_bloom} pixels, {num_trails} trails)")
    ax5.axis("off")

    # Panel 6: Persistence (if available)
    ax6 = fig.add_subplot(gs[1, 2])
    pers = stats.get("persistence", {})
    if isinstance(pers, Mapping) and not pers.get("skipped", False):
        pers_regions = pers.get("num_regions", 0)
        pers_pixels = pers.get("num_pixels", 0)
        ax6.set_title(f"Persistence\n({pers_pixels} pixels, {pers_regions} regions)")
        # NOTE: This assumes persistence is encoded in BPM_BIT_OUTLIER.
        pers_mask = (dq & BPM_BIT_OUTLIER) != 0
        ax6.imshow(pers_mask, origin="lower", cmap="Oranges", vmin=0, vmax=1)
    else:
        ax6.text(
            0.5, 0.5, "Skipped", ha="center", va="center",
            transform=ax6.transAxes, fontsize=12
        )
        ax6.set_title("Persistence\n(not run)")
    ax6.axis("off")

    # Panel 7: Statistics text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    summary_text = _format_summary_text(stats)
    ax7.text(
        0.05, 0.95, summary_text, transform=ax7.transAxes,
        fontfamily="monospace", fontsize=10, va="top"
    )

    plt.suptitle("Artifact Detection Report", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        out = Path(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Artifact report saved to %s", out)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

        