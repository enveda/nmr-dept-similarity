"""Tests for binary vs normalized intensity weighting across encoders and metrics.

Two invariants underpin all of these:
  * ``binary`` mode is bit-for-bit identical to the previous count-based code.
  * ``normalized`` mode with all-equal intensities equals ``binary`` (every peak
    normalises to weight 1.0), and diverges only when intensities differ.
"""

import numpy as np
import pytest

from dept_similarity.bin_vector import generate_ppm_bins, generate_ppm_bins_typed
from dept_similarity.gaussian_vector import generate_gaussian_vector
from dept_similarity.spectral_matching import (
    AlignmentResult,
    align_dept_peaks,
    align_dept_peaks_hungarian,
    cosine_from_alignment,
    jaccard_from_alignment,
    recall_from_alignment,
    resolve_intensity_weights,
)


def _f32(*values):
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# resolve_intensity_weights
# ---------------------------------------------------------------------------


def test_binary_mode_returns_unit_weights():
    peaks = _f32(20.0, -45.0, 80.0)
    w = resolve_intensity_weights(peaks, intensities=_f32(9.0, 3.0, 1.0), intensity_mode="binary")
    np.testing.assert_array_equal(w, np.ones(3, dtype=np.float32))


def test_binary_mode_ignores_missing_intensities():
    w = resolve_intensity_weights(_f32(20.0, 30.0), intensities=None, intensity_mode="binary")
    np.testing.assert_array_equal(w, np.ones(2, dtype=np.float32))


def test_normalized_divides_by_max():
    w = resolve_intensity_weights(
        _f32(20.0, 30.0, 40.0), intensities=_f32(10.0, 5.0, 2.0), intensity_mode="normalized"
    )
    np.testing.assert_allclose(w, [1.0, 0.5, 0.2], atol=1e-6)


def test_normalized_takes_absolute_value():
    # Multiplicity lives in the sign of the *peak*, not the intensity.
    w = resolve_intensity_weights(
        _f32(20.0, -30.0), intensities=_f32(-10.0, 5.0), intensity_mode="normalized"
    )
    np.testing.assert_allclose(w, [1.0, 0.5], atol=1e-6)


def test_normalized_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        resolve_intensity_weights(_f32(20.0, 30.0), intensities=_f32(1.0), intensity_mode="normalized")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown intensity_mode"):
        resolve_intensity_weights(_f32(20.0), intensities=_f32(1.0), intensity_mode="log")


def test_empty_peaks_binary_and_normalized():
    assert resolve_intensity_weights(_f32(), None, "binary").size == 0
    assert resolve_intensity_weights(_f32(), _f32(), "normalized").size == 0


# ---------------------------------------------------------------------------
# AlignmentResult binary defaults
# ---------------------------------------------------------------------------


def test_alignment_result_binary_defaults_backfill():
    r = AlignmentResult(n_matched=2, n_peaks_a=4, n_peaks_b=4)
    assert r.matched_dot == 2.0
    assert r.matched_min == 2.0
    assert r.matched_wa == 2.0
    assert r.norm_a_sq == 4.0
    assert r.norm_b_sq == 4.0
    assert r.sum_wa == 4.0
    assert r.sum_wb == 4.0


# ---------------------------------------------------------------------------
# Spectral matching: binary vs normalized
# ---------------------------------------------------------------------------


def test_equal_intensities_equal_binary():
    a = _f32(20.0, -45.0, 80.0)
    b = _f32(20.5, -45.0, 150.0)
    r_bin = align_dept_peaks(a, b)
    r_norm = align_dept_peaks(
        a, b, intensities_a=_f32(7, 7, 7), intensities_b=_f32(3, 3, 3), intensity_mode="normalized"
    )
    for fn in (cosine_from_alignment, jaccard_from_alignment, recall_from_alignment):
        assert fn(r_norm) == pytest.approx(fn(r_bin), abs=1e-6)


def test_weighted_cosine_known_value():
    # Both spectra have peaks {20 (CH/CH3), -45 (CH2)} that both align.
    # a intensities [10, 5] -> wa [1, 0.5]; b [5, 10] -> wb [0.5, 1].
    # dot = 1*0.5 + 0.5*1 = 1.0 ; norm_a_sq = norm_b_sq = 1.25 ; cosine = 1/1.25 = 0.8
    a = _f32(20.0, -45.0)
    b = _f32(20.0, -45.0)
    r = align_dept_peaks(
        a, b, intensities_a=_f32(10, 5), intensities_b=_f32(5, 10), intensity_mode="normalized"
    )
    assert cosine_from_alignment(r) == pytest.approx(0.8, abs=1e-6)


def test_weighted_jaccard_known_value():
    # matched_min = min(1,0.5)+min(0.5,1) = 1.0 ; sums = 1.5 each
    # jaccard = 1.0 / (1.5 + 1.5 - 1.0) = 0.5
    a = _f32(20.0, -45.0)
    b = _f32(20.0, -45.0)
    r = align_dept_peaks(
        a, b, intensities_a=_f32(10, 5), intensities_b=_f32(5, 10), intensity_mode="normalized"
    )
    assert jaccard_from_alignment(r) == pytest.approx(0.5, abs=1e-6)


def test_weighted_identical_spectra_is_one():
    a = _f32(20.0, -45.0, 80.0)
    r = align_dept_peaks(
        a, a, intensities_a=_f32(9, 3, 1), intensities_b=_f32(9, 3, 1), intensity_mode="normalized"
    )
    assert cosine_from_alignment(r) == pytest.approx(1.0, abs=1e-6)
    assert jaccard_from_alignment(r) == pytest.approx(1.0, abs=1e-6)
    assert recall_from_alignment(r) == pytest.approx(1.0, abs=1e-6)


def test_hungarian_weighted_matches_twopointer_on_simple_case():
    a = _f32(20.0, -45.0)
    b = _f32(20.0, -45.0)
    kw = dict(intensities_a=_f32(10, 5), intensities_b=_f32(5, 10), intensity_mode="normalized")
    r_h = align_dept_peaks_hungarian(a, b, **kw)
    r_tp = align_dept_peaks(a, b, **kw)
    assert cosine_from_alignment(r_h) == pytest.approx(cosine_from_alignment(r_tp), abs=1e-6)
    assert cosine_from_alignment(r_h) == pytest.approx(0.8, abs=1e-6)


def test_hungarian_precomputed_weights_path():
    # Passing weights_a/weights_b directly must equal resolving from intensities.
    a = _f32(20.0, -45.0)
    b = _f32(20.0, -45.0)
    r_intens = align_dept_peaks_hungarian(
        a, b, intensities_a=_f32(10, 5), intensities_b=_f32(5, 10), intensity_mode="normalized"
    )
    r_weights = align_dept_peaks_hungarian(
        a, b, weights_a=_f32(1.0, 0.5), weights_b=_f32(0.5, 1.0)
    )
    assert cosine_from_alignment(r_weights) == pytest.approx(cosine_from_alignment(r_intens), abs=1e-6)


# ---------------------------------------------------------------------------
# Bin vectors: binary vs normalized
# ---------------------------------------------------------------------------


def test_bins_equal_intensities_equal_binary():
    peaks = _f32(20.0, 60.0, 100.0)
    binary = generate_ppm_bins(peaks)
    norm = generate_ppm_bins(peaks, intensities=_f32(4, 4, 4), intensity_mode="normalized")
    np.testing.assert_allclose(binary, norm, atol=1e-6)


def test_bins_normalized_weights_sum():
    peaks = _f32(20.0, 60.0)
    # intensities [10, 5] -> weights [1.0, 0.5] -> total mass 1.5 (vs 2.0 binary)
    binary = generate_ppm_bins(peaks)
    norm = generate_ppm_bins(peaks, intensities=_f32(10, 5), intensity_mode="normalized")
    assert binary.sum() == pytest.approx(2.0, abs=1e-6)
    assert norm.sum() == pytest.approx(1.5, abs=1e-6)


def test_bins_typed_normalized_weights_sum():
    peaks = _f32(-20.0, 60.0)  # one CH2, one CH/CH3
    norm = generate_ppm_bins_typed(peaks, intensities=_f32(10, 5), intensity_mode="normalized")
    assert norm.sum() == pytest.approx(1.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Gaussian vector: binary vs normalized
# ---------------------------------------------------------------------------


def test_gaussian_equal_intensities_equal_binary():
    peaks = _f32(30.0, -60.0)
    binary = generate_gaussian_vector(peaks)
    norm = generate_gaussian_vector(peaks, intensities=_f32(2, 2), intensity_mode="normalized")
    np.testing.assert_allclose(binary, norm, atol=1e-6)


def test_gaussian_normalized_scales_amplitude():
    # peaks at 30 and 60 ppm, intensities 10 and 5 -> weights 1.0 and 0.5.
    peaks = _f32(30.0, 60.0)
    vec = generate_gaussian_vector(peaks, intensities=_f32(10, 5), intensity_mode="normalized")
    # Grid is 0..210 at 0.1 ppm -> index = ppm * 10.
    peak_30 = vec[300]
    peak_60 = vec[600]
    assert peak_30 == pytest.approx(1.0, abs=1e-3)
    assert peak_60 == pytest.approx(0.5, abs=1e-3)
