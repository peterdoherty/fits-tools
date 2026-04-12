import numpy as np
import pytest

from fits_tools.bitmask import (
    create_bpm,
    create_master_bpm,
    merge_bpms,
    apply_bpm,
    get_defect_types,
    get_bpm_statistics,
    has_defect,
    bit_index,
    combine_flags,
    to_boolean_mask,
    build_good_pixel_mask,
    validate_bpm,
    DefectType,
    BPM_BIT_PIX_BRIGHT,
    BPM_BIT_PIX_HOT,
    BPM_BIT_PIX_NONLIN,
    BPM_BIT_PIX_DEAD,
    BPM_BIT_COL_BAD,
    BPM_BIT_COL_BLOCKED,
    BPM_BIT_ROW_EXCLUDED,
    BPM_BIT_COSMIC,
    BPM_BIT_SATURATED,
    BPM_MASK_PERMANENT,
    BPM_MASK_HOT,
    BPM_MASK_COL,
    BPM_MASK_IGNORE,
)


def test_create_bpm_defaults_to_uint32_and_zero_initialized():
    bpm = create_bpm((4, 5))
    assert bpm.shape == (4, 5)
    assert bpm.dtype == np.uint32
    assert np.count_nonzero(bpm) == 0


def test_create_master_bpm_marks_requested_defects():
    dead = np.array([[0, 1], [2, 3]])
    hot = np.array([[1, 1]])
    bad_columns = np.array([4])

    bpm = create_master_bpm((4, 5), dead_pixels=dead, hot_pixels=hot, bad_columns=bad_columns)

    assert "PIX_DEAD" in get_defect_types(int(bpm[0, 1]))
    assert "PIX_DEAD" in get_defect_types(int(bpm[2, 3]))
    assert "PIX_HOT" in get_defect_types(int(bpm[1, 1]))
    assert "COL_BAD" in get_defect_types(int(bpm[0, 4]))


def test_basic_flagging_and_get_defect_types():
    bpm = create_bpm((10, 10))

    bpm[2, 3] = BPM_BIT_PIX_BRIGHT
    bpm[5, 6] = BPM_BIT_PIX_HOT
    bpm[7, 8] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN
    bpm[:, 1] = BPM_BIT_COL_BAD | BPM_BIT_COL_BLOCKED
    bpm[9, :] |= BPM_BIT_ROW_EXCLUDED

    assert get_defect_types(int(bpm[2, 3])) == ["PIX_BRIGHT"]
    assert get_defect_types(int(bpm[5, 6])) == ["PIX_HOT"]

    mixed = get_defect_types(int(bpm[7, 8]))
    assert set(mixed) == {"PIX_HOT", "PIX_NONLIN"}

    col_defects = get_defect_types(int(bpm[0, 1]))
    assert "COL_BAD" in col_defects
    assert "COL_BLOCKED" in col_defects

    row_defects = get_defect_types(int(bpm[9, 0]))
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

    for name, entry in stats.items():
        assert isinstance(entry["count"], int)
        assert entry["count"] >= 0
        assert 0.0 <= entry["fraction"] <= 1.0


def test_has_defect_with_group_masks():
    bpm = create_bpm((10, 10))
    bpm[7, 8] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN
    bpm[:, 1] = BPM_BIT_COL_BAD | BPM_BIT_COL_BLOCKED

    assert has_defect(int(bpm[7, 8]), BPM_MASK_HOT) is True
    assert has_defect(int(bpm[7, 8]), BPM_MASK_COL) is False

    assert has_defect(int(bpm[0, 1]), BPM_MASK_COL) is True
    assert has_defect(int(bpm[0, 1]), BPM_MASK_HOT) is False


def test_apply_bpm_masks_permanent_defects_but_not_excluded_flags():
    rng = np.random.default_rng(123)
    data = rng.normal(100.0, 5.0, (10, 10))

    bpm = create_bpm((10, 10))
    bpm[:, 1] = BPM_BIT_COL_BAD | BPM_BIT_COL_BLOCKED
    bpm[9, :] |= BPM_BIT_ROW_EXCLUDED

    masked = apply_bpm(data, bpm, defect_mask=BPM_MASK_PERMANENT, fill_value=np.nan)

    assert np.isnan(masked[0, 1])
    assert not np.isnan(masked[9, 0])
    assert masked[9, 0] == pytest.approx(data[9, 0])
    assert np.isfinite(masked[0, 0])
    assert masked[0, 0] == pytest.approx(data[0, 0])


def test_merge_bpms_or_combines_static_and_transient_flags():
    bpm = create_bpm((10, 10))
    bpm[7, 8] = BPM_BIT_PIX_HOT

    dq_transient = create_bpm((10, 10))
    dq_transient[7, 8] = BPM_BIT_COSMIC
    dq_transient[3, 4] = BPM_BIT_SATURATED

    final_dq = merge_bpms(bpm, dq_transient, operation="or")

    assert set(get_defect_types(int(bpm[7, 8]))) == {"PIX_HOT"}
    assert set(get_defect_types(int(dq_transient[7, 8]))) == {"COSMIC"}
    assert set(get_defect_types(int(final_dq[7, 8]))) == {"PIX_HOT", "COSMIC"}
    assert set(get_defect_types(int(final_dq[3, 4]))) == {"SATURATED"}


def test_merge_bpms_rejects_shape_mismatch():
    bpm1 = create_bpm((5, 5))
    bpm2 = create_bpm((4, 5))

    with pytest.raises(ValueError, match="BPM shapes must match"):
        merge_bpms(bpm1, bpm2, operation="or")


def test_bit_index_and_combine_flags():
    assert bit_index(BPM_BIT_PIX_BRIGHT) == 2
    assert bit_index(BPM_BIT_COSMIC) == 28

    combined = combine_flags([BPM_BIT_PIX_HOT, BPM_BIT_PIX_NONLIN, BPM_BIT_COSMIC])
    assert combined == (BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN | BPM_BIT_COSMIC)


def test_bit_index_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="single-bit power of two"):
        bit_index(3)

    with pytest.raises(ValueError, match="single-bit power of two"):
        bit_index(0)


def test_to_boolean_mask_and_invert():
    bpm = create_bpm((3, 3))
    bpm[0, 0] = BPM_BIT_PIX_HOT
    bpm[1, 1] = BPM_BIT_ROW_EXCLUDED

    bad_perm = to_boolean_mask(bpm, defect_mask=BPM_MASK_PERMANENT, invert=False)
    good_perm = to_boolean_mask(bpm, defect_mask=BPM_MASK_PERMANENT, invert=True)

    assert bad_perm.dtype == bool
    assert good_perm.dtype == bool

    assert bad_perm[0, 0]
    assert not bad_perm[1, 1]
    assert bool(good_perm[0, 0]) is False
    assert bool(good_perm[1, 1]) is True


def test_build_good_pixel_mask_combines_bpm_and_data_filters():
    data = np.array(
        [
            [1.0, 2.0, np.nan],
            [0.0, -1.0, 5.0],
            [7.0, 8.0, 9.0],
        ]
    )
    bpm = create_bpm(data.shape)
    bpm[2, 2] = BPM_BIT_PIX_DEAD
    bpm[2, 1] = BPM_BIT_ROW_EXCLUDED

    good = build_good_pixel_mask(
        data,
        bad_pixel_mask=bpm,
        exclude_flags=BPM_MASK_PERMANENT | BPM_MASK_IGNORE,
        exclude_nan=True,
        exclude_negative=True,
        exclude_zero=True,
    )

    expected = np.array(
        [
            [True, True, False],
            [False, False, True],
            [True, False, False],
        ],
        dtype=bool,
    )
    assert np.array_equal(good, expected)


def test_build_good_pixel_mask_rejects_shape_mismatch():
    data = np.ones((3, 3))
    bpm = create_bpm((2, 3))

    with pytest.raises(ValueError, match="does not match data shape"):
        build_good_pixel_mask(data, bad_pixel_mask=bpm)


def test_validate_bpm_passes_for_small_clean_mask():
    bpm = create_bpm((5, 5))
    bpm[0, 0] = BPM_BIT_PIX_HOT

    is_valid, messages = validate_bpm(
        bpm,
        max_defect_fraction=0.5,
        max_bad_rows=10,
        max_bad_cols=10,
    )

    assert is_valid is True
    assert messages == []


def test_validate_bpm_fails_when_permanent_defect_fraction_too_high():
    bpm = create_bpm((4, 4))
    bpm[:, :] = BPM_BIT_PIX_HOT

    is_valid, messages = validate_bpm(
        bpm,
        max_defect_fraction=0.1,
        max_bad_rows=10,
        max_bad_cols=10,
    )

    assert is_valid is False
    assert any("Permanent defect fraction" in msg for msg in messages)


def test_get_bpm_statistics_rejects_non_2d_input():
    bpm = create_bpm((2, 2, 2))
    with pytest.raises(ValueError, match="must be 2-D"):
        get_bpm_statistics(bpm)


def test_defecttype_enum_combination():
    bpm = create_bpm((2, 2))
    bpm[0, 0] = BPM_BIT_PIX_HOT | BPM_BIT_PIX_NONLIN

    status = DefectType(int(bpm[0, 0]))

    assert DefectType.PIX_HOT in status
    assert DefectType.PIX_NONLIN in status

    s = str(status)
    assert "PIX_HOT" in s
    assert "PIX_NONLIN" in s
