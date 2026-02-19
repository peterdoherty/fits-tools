import numpy as np
import pytest

from fits_tools.bitmask import (
    __version__,
    create_bpm,
    get_defect_types,
    get_bpm_statistics,
    has_defect,
    apply_bpm,
    merge_bpms,
    DefectType,
    # bits / masks used in the old __main__ demo:
    BPM_BIT_PIX_BRIGHT,
    BPM_BIT_PIX_HOT,
    BPM_BIT_PIX_NONLIN,
    BPM_BIT_COL_BAD,
    BPM_BIT_COL_BLOCKED,
    BPM_BIT_ROW_EXCLUDED,
    BPM_BIT_PIX_COSMIC,
    BPM_BIT_SATURATED,
    BPM_MASK_PERMANENT,
)


def test_version_string():
    assert isinstance(__version__, str)
    assert __version__  # non-empty


def test_basic_flagging_and_get_defect_types():
    bpm = create_bpm((10, 10))

    bpm[2, 3] = BPM_BIT_PIX_BRIGHT
    bpm[5, 6] = BPM_BIT_PIX_HOT
    bpm[7, 8] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN
    bpm[:, 1] = BPM_BIT_COL_BAD | BPM_BIT_COL_BLOCKED
    bpm[9, :] |= BPM_BIT_ROW_EXCLUDED

    assert get_defect_types(bpm[2, 3]) == ["PIX_BRIGHT"]
    assert get_defect_types(bpm[5, 6]) == ["PIX_HOT"]

    mixed = get_defect_types(bpm[7, 8])
    assert set(mixed) == {"PIX_HOT", "PIX_NONLIN"}

    col_defects = get_defect_types(bpm[0, 1])
    assert "COL_BAD" in col_defects
    assert "COL_BLOCKED" in col_defects

    row_defects = get_defect_types(bpm[9, 0])
    assert "ROW_EXCLUDED" in row_defects


def test_get_bpm_statistics_counts_match_manual():
    bpm = create_bpm((10, 10))
    bpm[2, 3] = BPM_BIT_PIX_BRIGHT
    bpm[5, 6] = BPM_BIT_PIX_HOT

    stats = get_bpm_statistics(bpm)

    assert stats["PIX_BRIGHT"]["count"] == 1
    assert stats["PIX_HOT"]["count"] == 1
    assert stats["PIX_BRIGHT"]["fraction"] == pytest.approx(1 / bpm.size)
    assert stats["PIX_HOT"]["fraction"] == pytest.approx(1 / bpm.size)

    # Sanity check: all entries have a defined count
    for name, entry in stats.items():
        assert isinstance(entry["count"], int)
        assert entry["count"] >= 0


def test_has_defect_with_group_masks():
    bpm = create_bpm((10, 10))
    bpm[7, 8] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN
    bpm[:, 1] = BPM_BIT_COL_BAD | BPM_BIT_COL_BLOCKED

    from fits_tools.bitmask import BPM_MASK_HOT, BPM_MASK_COL

    assert has_defect(bpm[7, 8], BPM_MASK_HOT) is True
    assert has_defect(bpm[7, 8], BPM_MASK_COL) is False

    assert has_defect(bpm[0, 1], BPM_MASK_COL) is True
    assert has_defect(bpm[0, 1], BPM_MASK_HOT) is False


def test_apply_bpm_masks_permanent_defects():
    rng = np.random.default_rng(123)
    data = rng.normal(100.0, 5.0, (10, 10))

    bpm = create_bpm((10, 10))
    bpm[:, 1] = BPM_BIT_COL_BAD | BPM_BIT_COL_BLOCKED
    bpm[9, :] |= BPM_BIT_ROW_EXCLUDED

    masked = apply_bpm(data, bpm, defect_mask=BPM_MASK_PERMANENT, fill_value=np.nan)

    # Column 1 is permanently bad => should be NaN
    assert np.isnan(masked[0, 1])

    # ROW_EXCLUDED is not part of BPM_MASK_PERMANENT; those pixels remain unchanged
    assert not np.isnan(masked[9, 0])
    assert masked[9, 0] == pytest.approx(data[9, 0])

    # A known-good pixel (no flags) should be unchanged and finite
    assert np.isfinite(masked[0, 0])
    assert masked[0, 0] == pytest.approx(data[0, 0])


def test_merge_bpms_or_and_dynamic_dq():
    bpm = create_bpm((10, 10))
    bpm[7, 8] = BPM_BIT_PIX_HOT

    dq_transient = create_bpm((10, 10))
    dq_transient[7, 8] = BPM_BIT_PIX_COSMIC
    dq_transient[3, 4] = BPM_BIT_SATURATED

    final_dq = merge_bpms(bpm, dq_transient, operation="or")

    # Master BPM at (7,8) has HOT only
    assert set(get_defect_types(bpm[7, 8])) == {"PIX_HOT"}

    # Transient DQ at (7,8) has COSMIC only
    assert set(get_defect_types(dq_transient[7, 8])) == {"PIX_COSMIC"}

    # Combined has both
    assert set(get_defect_types(final_dq[7, 8])) == {"PIX_HOT", "PIX_COSMIC"}

    # (3,4) should only have SATURATED in the final mask
    assert set(get_defect_types(final_dq[3, 4])) == {"SATURATED"}


def test_defecttype_enum_combination():
    bpm = create_bpm((2, 2))
    bpm[0, 0] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN

    status = DefectType(bpm[0, 0])

    assert DefectType.PIX_HOT in status
    assert DefectType.PIX_NONLIN in status

    # The string form should include the names of the bits that are set
    s = str(status)
    assert "PIX_HOT" in s
    assert "PIX_NONLIN" in s


    