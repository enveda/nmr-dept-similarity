"""Tests for dept_similarity.gaussian_vector — Gaussian spectral encoder."""

import numpy as np
import pytest

from dept_similarity.gaussian_vector import (
    _GRID_LEN,
    gaussian_cosine_similarity,
    generate_gaussian_vector,
)


def _f32(*values):
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# generate_gaussian_vector
# ---------------------------------------------------------------------------


def test_empty_spectrum_returns_zeros():
    vec = generate_gaussian_vector(np.array([], dtype=np.float32))
    assert vec.shape == (_GRID_LEN,)
    assert vec.dtype == np.float32
    assert vec.sum() == 0.0


def test_output_shape_and_dtype():
    vec = generate_gaussian_vector(_f32(30.0, 60.0))
    assert vec.shape == (_GRID_LEN,)
    assert vec.dtype == np.float32


def test_positive_peak_gives_positive_region():
    # CH/CH3 peak at +30 ppm → Gaussian centred at 30 ppm with positive amplitude.
    vec = generate_gaussian_vector(_f32(30.0))
    # The grid point nearest 30 ppm must be positive.
    assert vec.max() > 0.0
    assert vec.min() >= 0.0


def test_negative_peak_gives_negative_region():
    # CH2 peak at -30 ppm (encoded as negative) → negative Gaussian at |−30| = 30 ppm.
    vec = generate_gaussian_vector(_f32(-30.0))
    assert vec.min() < 0.0
    assert vec.max() <= 0.0


def test_opposite_types_cancel_at_same_shift():
    # CH/CH3 at +30 and CH2 at -30 → equal-magnitude but opposite-sign Gaussians,
    # which should cancel very nearly to zero at the grid point for 30 ppm.
    vec = generate_gaussian_vector(_f32(30.0, -30.0))
    assert abs(vec).max() < 1e-4


def test_symmetric_encoding():
    # Absolute ppm position should be the same regardless of sign.
    pos_vec = generate_gaussian_vector(_f32(50.0))
    neg_vec = generate_gaussian_vector(_f32(-50.0))
    np.testing.assert_allclose(np.abs(pos_vec), np.abs(neg_vec), atol=1e-6)


def test_gaussian_peak_is_centred_at_correct_shift():
    shift = 60.0
    vec = generate_gaussian_vector(_f32(shift))
    # Find the grid index of the peak maximum.
    peak_idx = int(np.argmax(np.abs(vec)))
    # The grid runs 0–210 ppm at 0.1 ppm resolution, so index = shift / 0.1.
    expected_idx = round(shift / 0.1)
    assert abs(peak_idx - expected_idx) <= 2  # allow ±2 grid steps


# ---------------------------------------------------------------------------
# gaussian_cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_identical_spectra():
    peaks = _f32(20.0, 50.0, -30.0)
    score = gaussian_cosine_similarity(peaks, peaks)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_cosine_same_shifts_opposite_types():
    # One spectrum has a CH/CH3 peak at 40 ppm; the other has a CH2 at −40 ppm.
    # Their Gaussian vectors are sign-flipped → cosine ≈ −1.
    a = _f32(40.0)
    b = _f32(-40.0)
    score = gaussian_cosine_similarity(a, b)
    assert score < -0.9


def test_cosine_disjoint_shifts():
    # Peaks far apart on the grid → near-zero cosine.
    a = _f32(10.0)
    b = _f32(200.0)
    score = gaussian_cosine_similarity(a, b)
    assert abs(score) < 0.01


def test_cosine_nearby_shifts_high_similarity():
    # Peaks close together (within 1 sigma) → high cosine.
    a = _f32(50.0)
    b = _f32(51.0)
    score = gaussian_cosine_similarity(a, b)
    assert score > 0.9


def test_cosine_empty_vs_nonempty():
    empty = np.array([], dtype=np.float32)
    nonempty = _f32(30.0)
    score = gaussian_cosine_similarity(empty, nonempty)
    assert score == pytest.approx(0.0, abs=1e-6)
