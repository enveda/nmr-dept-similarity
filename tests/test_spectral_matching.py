"""Tests for dept_similarity.spectral_matching — peak alignment and scoring."""

import numpy as np
import pytest

from dept_similarity.spectral_matching import (
    AlignmentResult,
    align_dept_peaks,
    align_dept_peaks_hungarian,
    cosine_from_alignment,
    dept_spectral_matching,
    jaccard_from_alignment,
    recall_from_alignment,
)


def _f32(*values):
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# align_dept_peaks
# ---------------------------------------------------------------------------


def test_identical_spectra_all_matched():
    peaks = _f32(20.0, -45.0, 80.0)
    result = align_dept_peaks(peaks, peaks)
    assert result.n_matched == 3
    assert result.n_peaks_a == 3
    assert result.n_peaks_b == 3


def test_no_overlap_no_matches():
    a = _f32(10.0)
    b = _f32(100.0)
    result = align_dept_peaks(a, b, ppm_tolerance=4.0)
    assert result.n_matched == 0


def test_partial_overlap():
    a = _f32(20.0, 80.0)  # two peaks
    b = _f32(20.5, 150.0)  # one within 4 ppm, one far away
    result = align_dept_peaks(a, b, ppm_tolerance=4.0)
    assert result.n_matched == 1
    assert result.n_peaks_a == 2
    assert result.n_peaks_b == 2


def test_match_multiplicity_type_gating():
    # Peak at +30 ppm (CH/CH3) vs peak at -30 ppm (CH2): same absolute shift,
    # different multiplicity → should NOT match when match_multiplicity=True.
    a = _f32(30.0)   # CH/CH3
    b = _f32(-30.0)  # CH2
    result_typed = align_dept_peaks(a, b, ppm_tolerance=4.0, match_multiplicity=True)
    result_positional = align_dept_peaks(a, b, ppm_tolerance=4.0, match_multiplicity=False)
    assert result_typed.n_matched == 0
    assert result_positional.n_matched == 1


def test_empty_spectrum_a():
    result = align_dept_peaks(np.array([], dtype=np.float32), _f32(30.0))
    assert result.n_matched == 0
    assert result.n_peaks_a == 0
    assert result.n_peaks_b == 1


def test_empty_spectrum_b():
    result = align_dept_peaks(_f32(30.0), np.array([], dtype=np.float32))
    assert result.n_matched == 0
    assert result.n_peaks_a == 1
    assert result.n_peaks_b == 0


def test_tolerance_boundary():
    # Peaks exactly ppm_tolerance apart should still match.
    a = _f32(30.0)
    b = _f32(34.0)  # exactly 4.0 ppm away
    result = align_dept_peaks(a, b, ppm_tolerance=4.0)
    assert result.n_matched == 1

    # One ppm beyond tolerance should not match.
    c = _f32(35.0)
    result2 = align_dept_peaks(a, c, ppm_tolerance=4.0)
    assert result2.n_matched == 0


# ---------------------------------------------------------------------------
# cosine_from_alignment
# ---------------------------------------------------------------------------


def test_cosine_perfect_match():
    result = AlignmentResult(n_matched=4, n_peaks_a=4, n_peaks_b=4)
    assert cosine_from_alignment(result) == pytest.approx(1.0, abs=1e-6)


def test_cosine_known_value():
    # 2 matched, 4 peaks each → 2 / sqrt(4*4) = 2/4 = 0.5
    result = AlignmentResult(n_matched=2, n_peaks_a=4, n_peaks_b=4)
    assert cosine_from_alignment(result) == pytest.approx(0.5, abs=1e-6)


def test_cosine_below_min_matched_peaks():
    result = AlignmentResult(n_matched=0, n_peaks_a=3, n_peaks_b=3)
    assert cosine_from_alignment(result, min_matched_peaks=1) == 0.0


def test_cosine_zero_peaks():
    result = AlignmentResult(n_matched=0, n_peaks_a=0, n_peaks_b=0)
    assert cosine_from_alignment(result) == 0.0


# ---------------------------------------------------------------------------
# jaccard_from_alignment
# ---------------------------------------------------------------------------


def test_jaccard_perfect_match():
    result = AlignmentResult(n_matched=3, n_peaks_a=3, n_peaks_b=3)
    assert jaccard_from_alignment(result) == pytest.approx(1.0, abs=1e-6)


def test_jaccard_known_value():
    # 2 matched, 3 + 3 - 2 = 4 union → 2/4 = 0.5
    result = AlignmentResult(n_matched=2, n_peaks_a=3, n_peaks_b=3)
    assert jaccard_from_alignment(result) == pytest.approx(0.5, abs=1e-6)


def test_jaccard_below_min_matched_peaks():
    result = AlignmentResult(n_matched=0, n_peaks_a=3, n_peaks_b=3)
    assert jaccard_from_alignment(result, min_matched_peaks=1) == 0.0


def test_jaccard_asymmetric_penalty():
    # 1 match: a has 1 peak, b has 5 peaks → 1/(1+5-1) = 1/5 = 0.2
    result = AlignmentResult(n_matched=1, n_peaks_a=1, n_peaks_b=5)
    assert jaccard_from_alignment(result) == pytest.approx(0.2, abs=1e-6)


# ---------------------------------------------------------------------------
# recall_from_alignment
# ---------------------------------------------------------------------------


def test_recall_all_query_peaks_matched():
    result = AlignmentResult(n_matched=3, n_peaks_a=3, n_peaks_b=10)
    assert recall_from_alignment(result) == pytest.approx(1.0, abs=1e-6)


def test_recall_partial():
    result = AlignmentResult(n_matched=1, n_peaks_a=4, n_peaks_b=4)
    assert recall_from_alignment(result) == pytest.approx(0.25, abs=1e-6)


def test_recall_zero_query_peaks():
    result = AlignmentResult(n_matched=0, n_peaks_a=0, n_peaks_b=5)
    assert recall_from_alignment(result) == 0.0


def test_recall_ignores_extra_library_peaks():
    # Query has 2 peaks, library has 100 but both query peaks matched.
    r_small = AlignmentResult(n_matched=2, n_peaks_a=2, n_peaks_b=5)
    r_large = AlignmentResult(n_matched=2, n_peaks_a=2, n_peaks_b=100)
    assert recall_from_alignment(r_small) == pytest.approx(recall_from_alignment(r_large), abs=1e-6)


# ---------------------------------------------------------------------------
# dept_spectral_matching (convenience wrapper)
# ---------------------------------------------------------------------------


def test_wrapper_cosine_identical():
    peaks = _f32(20.0, -45.0, 80.0)
    score = dept_spectral_matching(peaks, peaks, metric="cosine")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_wrapper_jaccard_identical():
    peaks = _f32(20.0, -45.0)
    score = dept_spectral_matching(peaks, peaks, metric="jaccard")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_wrapper_recall_identical():
    peaks = _f32(20.0, -45.0)
    score = dept_spectral_matching(peaks, peaks, metric="recall")
    assert score == pytest.approx(1.0, abs=1e-5)


def test_wrapper_unknown_metric_raises():
    peaks = _f32(20.0)
    with pytest.raises(ValueError, match="Unknown metric"):
        dept_spectral_matching(peaks, peaks, metric="unknown")


def test_wrapper_min_matched_peaks_threshold():
    # Only 1 peak matches but we require 2 → score should be 0.
    a = _f32(20.0, 80.0)
    b = _f32(20.5, 150.0)
    score = dept_spectral_matching(a, b, min_matched_peaks=2, metric="cosine")
    assert score == 0.0


# ---------------------------------------------------------------------------
# align_dept_peaks_hungarian
# ---------------------------------------------------------------------------


def test_hungarian_identical_spectra():
    peaks = _f32(20.0, -45.0, 80.0)
    result = align_dept_peaks_hungarian(peaks, peaks)
    assert result.n_matched == 3
    assert result.n_peaks_a == 3
    assert result.n_peaks_b == 3


def test_hungarian_no_overlap():
    a = _f32(10.0)
    b = _f32(100.0)
    result = align_dept_peaks_hungarian(a, b, ppm_tolerance=4.0)
    assert result.n_matched == 0


def test_hungarian_partial_overlap():
    a = _f32(20.0, 80.0)
    b = _f32(20.5, 150.0)
    result = align_dept_peaks_hungarian(a, b, ppm_tolerance=4.0)
    assert result.n_matched == 1


def test_hungarian_type_gating():
    # Same absolute shift, opposite types → no match when match_multiplicity=True.
    a = _f32(30.0)   # CH/CH3
    b = _f32(-30.0)  # CH2
    result_typed = align_dept_peaks_hungarian(a, b, ppm_tolerance=4.0, match_multiplicity=True)
    result_positional = align_dept_peaks_hungarian(a, b, ppm_tolerance=4.0, match_multiplicity=False)
    assert result_typed.n_matched == 0
    assert result_positional.n_matched == 1


def test_hungarian_empty_inputs():
    result = align_dept_peaks_hungarian(np.array([], dtype=np.float32), _f32(30.0))
    assert result.n_matched == 0
    assert result.n_peaks_a == 0
    assert result.n_peaks_b == 1


def test_hungarian_optimal_over_greedy():
    # A: CH2@30, CH/CH3@31   B: CH/CH3@30, CH2@31
    # Valid pairings with multiplicity: A[CH2@30]↔B[CH2@31] (Δ=1 ppm) and
    # A[CH/CH3@31]↔B[CH/CH3@30] (Δ=1 ppm) → 2 matches.
    # The two-pointer groups both peaks into one window and counts by type bin
    # (ch2_a=1, nch2_a=1, ch2_b=1, nch2_b=1 → min matches per type = 1+1 = 2)
    # but its greedy window anchor can produce 0 when the closest same-type
    # partners sit on opposite sides of the window boundary.
    # Hungarian solves the global assignment and always recovers both matches.
    a = np.array([-30.0, 31.0], dtype=np.float32)  # CH2@30, CH/CH3@31
    b = np.array([30.0, -31.0], dtype=np.float32)  # CH/CH3@30, CH2@31
    result_h = align_dept_peaks_hungarian(a, b, ppm_tolerance=4.0)
    result_tp = align_dept_peaks(a, b, ppm_tolerance=4.0)
    assert result_h.n_matched == 2
    # Two-pointer misses both matches in this interleaved-type configuration.
    assert result_tp.n_matched < result_h.n_matched


def test_hungarian_agrees_with_twopointer_on_simple_cases():
    # For well-separated non-overlapping peaks, both methods must agree.
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(2, 8)
        # Space peaks 20 ppm apart so no tolerance window overlap.
        shifts = np.arange(n, dtype=np.float32) * 20.0 + 10.0
        signs = rng.choice([-1.0, 1.0], n).astype(np.float32)
        peaks = shifts * signs
        result_h = align_dept_peaks_hungarian(peaks, peaks, ppm_tolerance=4.0)
        result_tp = align_dept_peaks(peaks, peaks, ppm_tolerance=4.0)
        assert result_h.n_matched == result_tp.n_matched
