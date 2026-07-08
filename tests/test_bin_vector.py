"""Tests for dept_similarity.bin_vector — ppm binning encoder."""

import numpy as np
import pytest

from dept_similarity.bin_vector import (
    TOTAL_BINS,
    generate_ppm_bins,
    generate_ppm_bins_typed,
    ppm_bins_similarity,
    ppm_bins_typed_similarity,
)


def _f32(*values):
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# generate_ppm_bins
# ---------------------------------------------------------------------------


def test_empty_spectrum_returns_zeros():
    vec = generate_ppm_bins(np.array([], dtype=np.float32))
    assert vec.shape == (TOTAL_BINS,)
    assert vec.dtype == np.float32
    assert vec.sum() == 0.0


def test_output_shape_and_dtype():
    vec = generate_ppm_bins(_f32(30.0, 60.0, 90.0))
    assert vec.shape == (TOTAL_BINS,)
    assert vec.dtype == np.float32


def test_peak_count_equals_number_of_peaks():
    peaks = _f32(20.0, 50.0, 100.0)
    vec = generate_ppm_bins(peaks)
    assert vec.sum() == pytest.approx(len(peaks), abs=1e-6)


def test_sign_is_ignored_same_bin():
    # Negative ppm (CH2) and positive ppm (CH/CH3) at the same absolute shift
    # should land in the same bin.
    pos = generate_ppm_bins(_f32(30.0))
    neg = generate_ppm_bins(_f32(-30.0))
    np.testing.assert_array_equal(pos, neg)


def test_two_peaks_same_bin_count_two():
    # Both shifts land inside a single 3.7-ppm bin around 30 ppm.
    vec = generate_ppm_bins(_f32(30.0, 30.5))
    assert vec.max() == 2.0


def test_out_of_range_low_clamped():
    vec_clamped = generate_ppm_bins(_f32(0.0))
    vec_in_range = generate_ppm_bins(_f32(1.0))
    # Both should be non-zero in some bin (clamp keeps them in the vector).
    assert vec_clamped.sum() == 1.0
    assert vec_in_range.sum() == 1.0


def test_out_of_range_high_clamped():
    # Peaks well beyond MAX_CPPM should be clamped to the last bin.
    vec = generate_ppm_bins(_f32(9999.0))
    assert vec.sum() == 1.0
    assert vec[TOTAL_BINS - 1] == 1.0


# ---------------------------------------------------------------------------
# ppm_bins_similarity
# ---------------------------------------------------------------------------


def test_cosine_identical_spectra():
    peaks = _f32(20.0, 50.0, 100.0)
    score = ppm_bins_similarity(peaks, peaks, metric="cosine")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_cosine_disjoint_spectra():
    a = _f32(20.0)
    b = _f32(200.0)
    score = ppm_bins_similarity(a, b, metric="cosine")
    assert score == pytest.approx(0.0, abs=1e-6)


def test_euclidean_identical_spectra():
    peaks = _f32(30.0, 60.0)
    score = ppm_bins_similarity(peaks, peaks, metric="euclidean")
    assert score == pytest.approx(0.0, abs=1e-5)


def test_euclidean_different_spectra_positive():
    a = _f32(20.0)
    b = _f32(100.0)
    score = ppm_bins_similarity(a, b, metric="euclidean")
    assert score > 0.0


def test_manhattan_identical_spectra():
    peaks = _f32(30.0, 60.0)
    score = ppm_bins_similarity(peaks, peaks, metric="manhattan")
    assert score == pytest.approx(0.0, abs=1e-5)


def test_unknown_metric_raises():
    peaks = _f32(30.0)
    with pytest.raises(ValueError):
        ppm_bins_similarity(peaks, peaks, metric="not_a_metric")


# ---------------------------------------------------------------------------
# generate_ppm_bins_typed
# ---------------------------------------------------------------------------


def test_typed_empty_spectrum_returns_zeros():
    vec = generate_ppm_bins_typed(np.array([], dtype=np.float32))
    assert vec.shape == (2 * TOTAL_BINS,)
    assert vec.dtype == np.float32
    assert vec.sum() == 0.0


def test_typed_output_shape():
    vec = generate_ppm_bins_typed(_f32(30.0, -45.0, 80.0))
    assert vec.shape == (2 * TOTAL_BINS,)


def test_typed_ch2_peak_lands_in_first_channel():
    # CH2 peak at -30 ppm → only first TOTAL_BINS slots should be non-zero.
    vec = generate_ppm_bins_typed(_f32(-30.0))
    assert vec[:TOTAL_BINS].sum() == 1.0
    assert vec[TOTAL_BINS:].sum() == 0.0


def test_typed_nonch2_peak_lands_in_second_channel():
    # CH/CH3 peak at +30 ppm → only second TOTAL_BINS slots should be non-zero.
    vec = generate_ppm_bins_typed(_f32(30.0))
    assert vec[:TOTAL_BINS].sum() == 0.0
    assert vec[TOTAL_BINS:].sum() == 1.0


def test_typed_mixed_peaks_split_correctly():
    # 1 CH2 + 1 CH/CH3 → one count in each channel.
    vec = generate_ppm_bins_typed(_f32(-30.0, 80.0))
    assert vec[:TOTAL_BINS].sum() == 1.0
    assert vec[TOTAL_BINS:].sum() == 1.0


def test_typed_different_from_untyped_for_opposing_types():
    # Single-channel bins treat +30 and -30 identically.
    # Two-channel bins must distinguish them.
    ch2_vec = generate_ppm_bins_typed(_f32(-30.0))
    nonch2_vec = generate_ppm_bins_typed(_f32(30.0))
    assert not np.array_equal(ch2_vec, nonch2_vec)


# ---------------------------------------------------------------------------
# ppm_bins_typed_similarity
# ---------------------------------------------------------------------------


def test_typed_cosine_identical():
    peaks = _f32(20.0, -45.0, 80.0)
    assert ppm_bins_typed_similarity(peaks, peaks) == pytest.approx(1.0, abs=1e-5)


def test_typed_cosine_same_shift_opposite_type_low_score():
    # Single-channel cosine = 1.0 for +30 vs -30 (same bin).
    # Two-channel cosine should be 0.0 (orthogonal channels).
    a = _f32(30.0)
    b = _f32(-30.0)
    untyped = ppm_bins_similarity(a, b, metric="cosine")
    typed = ppm_bins_typed_similarity(a, b, metric="cosine")
    assert untyped == pytest.approx(1.0, abs=1e-5)
    assert typed == pytest.approx(0.0, abs=1e-5)


def test_typed_cosine_disjoint():
    a = _f32(20.0)
    b = _f32(150.0)
    assert ppm_bins_typed_similarity(a, b) == pytest.approx(0.0, abs=1e-6)
