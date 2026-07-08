"""Tests for dept_similarity.score — numba scalar similarity functions."""

import numpy as np
import pytest

from dept_similarity.score import (
    compute_metric_numba,
    cosine_numba,
    euclidean_numba,
    jaccard_numba,
    manhattan_numba,
)


def _f32(*values):
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# cosine_numba
# ---------------------------------------------------------------------------


def test_cosine_identical():
    a = _f32(1.0, 2.0, 3.0)
    assert cosine_numba(a, a) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal():
    a = _f32(1.0, 0.0)
    b = _f32(0.0, 1.0)
    assert cosine_numba(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_anti_parallel():
    a = _f32(1.0, 2.0)
    b = _f32(-1.0, -2.0)
    assert cosine_numba(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_zero_vector():
    a = _f32(0.0, 0.0)
    b = _f32(1.0, 2.0)
    assert cosine_numba(a, b) == 0.0
    assert cosine_numba(b, a) == 0.0


def test_cosine_known_value():
    # [1, 0] vs [1, 1] → cos(45°) = 1/√2
    a = _f32(1.0, 0.0)
    b = _f32(1.0, 1.0)
    assert cosine_numba(a, b) == pytest.approx(1.0 / 2.0**0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# euclidean_numba
# ---------------------------------------------------------------------------


def test_euclidean_identical():
    a = _f32(3.0, 4.0)
    assert euclidean_numba(a, a) == pytest.approx(0.0, abs=1e-6)


def test_euclidean_pythagorean():
    a = _f32(0.0, 0.0)
    b = _f32(3.0, 4.0)
    assert euclidean_numba(a, b) == pytest.approx(5.0, abs=1e-5)


def test_euclidean_commutative():
    a = _f32(1.0, 2.0, 3.0)
    b = _f32(4.0, 5.0, 6.0)
    assert euclidean_numba(a, b) == pytest.approx(euclidean_numba(b, a), abs=1e-6)


# ---------------------------------------------------------------------------
# manhattan_numba
# ---------------------------------------------------------------------------


def test_manhattan_identical():
    a = _f32(1.0, 2.0, 3.0)
    assert manhattan_numba(a, a) == pytest.approx(0.0, abs=1e-6)


def test_manhattan_known_value():
    a = _f32(0.0, 0.0)
    b = _f32(3.0, 4.0)
    assert manhattan_numba(a, b) == pytest.approx(7.0, abs=1e-5)


def test_manhattan_commutative():
    a = _f32(1.0, 5.0)
    b = _f32(4.0, 2.0)
    assert manhattan_numba(a, b) == pytest.approx(manhattan_numba(b, a), abs=1e-6)


# ---------------------------------------------------------------------------
# jaccard_numba
# ---------------------------------------------------------------------------


def test_jaccard_identical_binary():
    a = _f32(1.0, 0.0, 1.0)
    assert jaccard_numba(a, a) == pytest.approx(1.0, abs=1e-6)


def test_jaccard_no_overlap():
    a = _f32(1.0, 0.0)
    b = _f32(0.0, 1.0)
    assert jaccard_numba(a, b) == pytest.approx(0.0, abs=1e-6)


def test_jaccard_partial_overlap():
    # intersection = 1 (position 0), union = 2 → 0.5
    a = _f32(1.0, 0.0)
    b = _f32(1.0, 1.0)
    assert jaccard_numba(a, b) == pytest.approx(0.5, abs=1e-6)


def test_jaccard_all_zeros():
    a = _f32(0.0, 0.0)
    b = _f32(0.0, 0.0)
    assert jaccard_numba(a, b) == 0.0


# ---------------------------------------------------------------------------
# compute_metric_numba dispatcher
# ---------------------------------------------------------------------------


def test_compute_metric_cosine():
    a = _f32(1.0, 0.0)
    b = _f32(1.0, 0.0)
    assert compute_metric_numba("cosine", a, b) == pytest.approx(1.0, abs=1e-6)


def test_compute_metric_euclidean():
    a = _f32(0.0, 0.0)
    b = _f32(3.0, 4.0)
    assert compute_metric_numba("euclidean", a, b) == pytest.approx(5.0, abs=1e-5)


def test_compute_metric_manhattan():
    a = _f32(0.0, 0.0)
    b = _f32(3.0, 4.0)
    assert compute_metric_numba("manhattan", a, b) == pytest.approx(7.0, abs=1e-5)


def test_compute_metric_jaccard():
    a = _f32(1.0, 0.0)
    b = _f32(1.0, 1.0)
    assert compute_metric_numba("jaccard", a, b) == pytest.approx(0.5, abs=1e-6)


def test_compute_metric_unknown_raises():
    a = _f32(1.0, 0.0)
    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        compute_metric_numba("unknown", a, a)
