"""Gaussian spectral representation for DEPT similarity.

Each peak is encoded as a Gaussian function placed on a fine ppm grid.
Sign convention — consistent with the signed ppm pipeline and the manuscript:
    CH/CH₃ (positive ppm values) → positive Gaussian  (+1 amplitude)
    CH₂    (negative ppm values) → negative Gaussian  (-1 amplitude)

The resulting signed continuous vector is compared via cosine similarity.
Peaks that are closely aligned between two spectra of the same type contribute
more to the score than partially overlapping ones; peaks of opposite type
reduce the score.  σ acts as a continuous analog of a hard ppm tolerance window.

Grid
----
0–210 ppm at 0.1 ppm resolution (2 101 points), from ``BASELINE_PARAMS["gaussian_grid"]``.
The grid is computed once at module import and shared across all calls.

Performance
-----------
The inner loop is compiled with numba ``@njit``.  A warm-up call at import
time triggers JIT compilation so the first real invocation is not penalised.

When ``sigma == DEFAULT_SIGMA`` the kernel uses a pre-computed lookup table
(LUT) of exp(−(k·Δx)²/2σ²) values indexed by integer grid-point offset k.
This eliminates all transcendental calls in the hot path for the common case.
For non-default sigma the kernel falls back to direct exp() computation.
"""

import math
from typing import Optional

import numpy as np
from numba import njit

from dept_similarity.constants import BASELINE_PARAMS
from dept_similarity.score import compute_metric_numba
from dept_similarity.spectral_matching import resolve_intensity_weights

# ---------------------------------------------------------------------------
# Grid — computed once at import
# ---------------------------------------------------------------------------

_GRID_MIN: float
_GRID_MAX: float
_GRID_MIN, _GRID_MAX = BASELINE_PARAMS["gaussian_grid"]
_GRID_STEP: float = 0.1  # ppm resolution
_GRID: np.ndarray = np.arange(_GRID_MIN, _GRID_MAX + _GRID_STEP, _GRID_STEP, dtype=np.float32)
_GRID_LEN: int = len(_GRID)

DEFAULT_SIGMA: float = BASELINE_PARAMS["gaussian_sigma"]

# ---------------------------------------------------------------------------
# Lookup table for default sigma — eliminates exp() calls in the hot path
# ---------------------------------------------------------------------------
# LUT[k] = exp(-(k * _GRID_STEP)^2 / (2 * DEFAULT_SIGMA^2))
# Covers offsets 0 … ceil(4σ / Δx) inclusive.  Values beyond this are < e⁻⁸.
_DEFAULT_TWO_SIGMA_SQ: float = float(2.0 * DEFAULT_SIGMA**2)
_DEFAULT_CUTOFF: float = float(4.0 * DEFAULT_SIGMA)
_LUT_HALF: int = int(math.ceil(_DEFAULT_CUTOFF / _GRID_STEP)) + 1
_LUT: np.ndarray = np.array(
    [math.exp(-((k * _GRID_STEP) ** 2) / _DEFAULT_TWO_SIGMA_SQ) for k in range(_LUT_HALF + 1)],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _gaussian_kernel(
    peaks_arr: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
    two_sigma_sq: float,
    cutoff: float,
    grid_min: float,
    grid_step: float,
    grid_len: int,
) -> np.ndarray:
    """Place signed Gaussians for each peak onto a pre-allocated grid vector.

    Parameters are all plain scalars or pre-typed arrays so numba can compile
    this to tight native code with no Python objects in the hot path.

    Window bounds use ``math.floor`` (not ``int()``) so that negative
    floating-point values truncate correctly toward −∞ rather than toward zero.
    A +1 buffer on ``idx_hi`` converts the floor to an inclusive upper bound:
    ``floor(x)`` is the largest integer ≤ x, so ``floor(x) + 1`` is the first
    integer strictly greater than x, making the range ``[idx_lo, idx_hi)``
    cover every grid point within the 4σ cutoff.
    """
    vec = np.zeros(grid_len, dtype=np.float32)

    for k in range(peaks_arr.shape[0]):
        signed_ppm = peaks_arr[k]

        # Absolute chemical shift and sign-based, intensity-scaled amplitude
        ppm = -signed_ppm if signed_ppm < 0.0 else signed_ppm
        amplitude = np.float32(-weights[k] if signed_ppm < 0.0 else weights[k])

        # Floor-based window bounds — avoids int() truncation-toward-zero bug
        idx_lo = int(math.floor((ppm - cutoff - grid_min) / grid_step))
        idx_hi = int(math.floor((ppm + cutoff - grid_min) / grid_step)) + 1

        # Clamp to valid grid range
        if idx_lo < 0:
            idx_lo = 0
        if idx_hi > grid_len:
            idx_hi = grid_len

        # Skip peaks whose Gaussian falls entirely outside the grid
        if idx_lo >= grid_len or idx_hi <= 0:
            continue

        for i in range(idx_lo, idx_hi):
            diff = grid[i] - np.float32(ppm)
            vec[i] += amplitude * np.exp(-diff * diff / np.float32(two_sigma_sq))

    return vec


@njit(cache=True, fastmath=True)
def _gaussian_kernel_lut(
    peaks_arr: np.ndarray,
    weights: np.ndarray,
    grid_min: float,
    grid_step: float,
    grid_len: int,
    cutoff: float,
    lut: np.ndarray,
    lut_half: int,
) -> np.ndarray:
    """LUT-backed Gaussian kernel for the default sigma.

    Replaces every ``exp()`` call with an integer-indexed array lookup.
    The LUT stores exp(−(k·Δx)²/2σ²) for k = 0, 1, …, lut_half, so the
    per-point cost is one subtraction, one int cast, and one array read.
    """
    vec = np.zeros(grid_len, dtype=np.float32)

    for k in range(peaks_arr.shape[0]):
        signed_ppm = peaks_arr[k]
        ppm = -signed_ppm if signed_ppm < 0.0 else signed_ppm
        amplitude = np.float32(-weights[k] if signed_ppm < 0.0 else weights[k])

        idx_lo = int(math.floor((ppm - cutoff - grid_min) / grid_step))
        idx_hi = int(math.floor((ppm + cutoff - grid_min) / grid_step)) + 1

        if idx_lo < 0:
            idx_lo = 0
        if idx_hi > grid_len:
            idx_hi = grid_len
        if idx_lo >= grid_len or idx_hi <= 0:
            continue

        for i in range(idx_lo, idx_hi):
            # Integer offset from peak centre; clamp to LUT bounds
            offset = int(math.floor(abs(grid_min + i * grid_step - ppm) / grid_step + 0.5))
            if offset > lut_half:
                offset = lut_half
            vec[i] += amplitude * lut[offset]

    return vec


# Warm-up: compile both kernels at import time.
_DUMMY_PEAKS = np.array([30.0, -45.0], dtype=np.float32)
_DUMMY_WEIGHTS = np.array([1.0, 1.0], dtype=np.float32)
_gaussian_kernel(
    _DUMMY_PEAKS,
    _DUMMY_WEIGHTS,
    _GRID,
    _DEFAULT_TWO_SIGMA_SQ,
    _DEFAULT_CUTOFF,
    _GRID_MIN,
    _GRID_STEP,
    _GRID_LEN,
)
_gaussian_kernel_lut(
    _DUMMY_PEAKS,
    _DUMMY_WEIGHTS,
    _GRID_MIN,
    _GRID_STEP,
    _GRID_LEN,
    _DEFAULT_CUTOFF,
    _LUT,
    _LUT_HALF,
)
del _DUMMY_PEAKS, _DUMMY_WEIGHTS


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------


def generate_gaussian_vector(
    peaks: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
    intensities: Optional[np.ndarray] = None,
    intensity_mode: str = "binary",
) -> np.ndarray:
    """Build a continuous signed Gaussian spectrum from a signed ppm array.

    Parameters
    ----------
    peaks:
        1D array of signed cppm values.
        Convention: negative = CH₂, positive = CH/CH₃.
    sigma:
        Gaussian width in ppm.  Default from ``BASELINE_PARAMS["gaussian_sigma"]``
        (3.0 ppm, validated on NMRshiftDB2).
    intensities:
        Optional peak intensities aligned 1-to-1 with ``peaks``; used only when
        ``intensity_mode="normalized"``.
    intensity_mode:
        ``"binary"`` (default): every peak gets unit amplitude (±1).
        ``"normalized"``: each peak's Gaussian is scaled by its max-normalised
        intensity, so stronger peaks contribute taller Gaussians.

    Returns
    -------
    np.ndarray
        Float32 vector of length ``_GRID_LEN`` (2 101 points, 0–210 ppm at
        0.1 ppm resolution).  Values can be negative where CH₂ peaks dominate.

    Notes
    -----
    A 4σ cutoff clips each Gaussian at ±4σ from its centre; contributions
    beyond that point are < e⁻⁸ ≈ 3 × 10⁻⁴ and safely ignored.
    """
    peaks_arr = np.asarray(peaks, dtype=np.float32)
    if peaks_arr.size == 0:
        return np.zeros(_GRID_LEN, dtype=np.float32)

    weights = resolve_intensity_weights(peaks_arr, intensities, intensity_mode)

    # Fast path: use the pre-computed LUT for the default sigma (no exp() calls)
    if sigma == DEFAULT_SIGMA:
        return _gaussian_kernel_lut(
            peaks_arr,
            weights,
            _GRID_MIN,
            _GRID_STEP,
            _GRID_LEN,
            _DEFAULT_CUTOFF,
            _LUT,
            _LUT_HALF,
        )

    # Fallback: compute exp() directly for non-default sigma
    return _gaussian_kernel(
        peaks_arr,
        weights,
        _GRID,
        float(2.0 * sigma * sigma),
        float(4.0 * sigma),
        _GRID_MIN,
        _GRID_STEP,
        _GRID_LEN,
    )


# ---------------------------------------------------------------------------
# Convenience pairwise wrapper
# ---------------------------------------------------------------------------


def gaussian_cosine_similarity(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
) -> float:
    """Encode two spectra as Gaussian vectors and return cosine similarity.

    Thin wrapper for ad-hoc pairwise use.  In batch pipelines, call
    ``generate_gaussian_vector`` directly to precompute vectors once per spectrum.

    Parameters
    ----------
    peaks_a, peaks_b:
        1D float32 arrays of signed cppm values.
    sigma:
        Gaussian width in ppm.  Default 3.0 ppm.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].  Negative values indicate spectra with
        opposing CH₂/CH₃ patterns at the same chemical shifts.
    """
    return float(
        compute_metric_numba(
            "cosine",
            generate_gaussian_vector(peaks_a, sigma),
            generate_gaussian_vector(peaks_b, sigma),
        )
    )
