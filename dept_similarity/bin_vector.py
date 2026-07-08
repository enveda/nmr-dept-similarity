"""Code for aligning spectra to a common m/z grid and computing similarity metrics."""

from typing import Optional

import numpy as np

from dept_similarity.constants import BASELINE_PARAMS
from dept_similarity.score import compute_metric_numba
from dept_similarity.spectral_matching import resolve_intensity_weights

# ---------------------------------------------------------------------------
# Bin vector encoder
# ---------------------------------------------------------------------------

MIN_CPPM, MAX_CPPM = BASELINE_PARAMS["ppm_range"]
BIN_WIDTH = BASELINE_PARAMS["bin_width"]
BIN_EDGES = np.arange(MIN_CPPM, MAX_CPPM + BIN_WIDTH, BIN_WIDTH)
TOTAL_BINS = BIN_EDGES.size


def _bin_peaks(peaks_arr: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Bin an array of absolute ppm values into TOTAL_BINS; clamped at edges.

    When ``weights`` is provided each peak contributes its weight to its bin
    (intensity-weighted histogram); otherwise every peak contributes 1.0.
    """
    if peaks_arr.size == 0:
        return np.zeros(TOTAL_BINS, dtype=np.float32)
    idx = np.searchsorted(BIN_EDGES, peaks_arr, side="right")
    np.clip(idx, 0, TOTAL_BINS - 1, out=idx)
    return np.bincount(idx, weights=weights, minlength=TOTAL_BINS).astype(
        np.float32, copy=False
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def generate_ppm_bins(
    peaks: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    intensity_mode: str = "binary",
) -> np.ndarray:
    """Single-channel binned vector, shape (TOTAL_BINS,).

    Sign is ignored — CH2 and CH/CH3 peaks at the same absolute shift land in
    the same bin.  Out-of-range peaks are clamped to the first/last bin.

    In ``"binary"`` mode (default) each bin counts the peaks that fall in it.
    In ``"normalized"`` mode each peak instead adds its max-normalised intensity,
    yielding an intensity-weighted histogram.
    """
    peaks_arr = np.asarray(peaks, dtype=np.float32)
    weights = _resolve_bin_weights(peaks_arr, intensities, intensity_mode)
    return _bin_peaks(np.abs(peaks_arr), weights)


def generate_ppm_bins_typed(
    peaks: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    intensity_mode: str = "binary",
) -> np.ndarray:
    """Two-channel binned vector, shape (2 * TOTAL_BINS,).

    The first TOTAL_BINS elements hold CH2 peaks (negative ppm values) per bin;
    the second TOTAL_BINS elements hold CH/CH3 peaks (positive ppm values).
    Concatenating the two channels lets cosine similarity distinguish spectra
    that differ only in multiplicity at the same chemical shift — information
    that ``generate_ppm_bins`` discards entirely.

    In ``"binary"`` mode each bin counts peaks; in ``"normalized"`` mode each
    peak adds its max-normalised intensity.
    """
    peaks_arr = np.asarray(peaks, dtype=np.float32)
    if peaks_arr.size == 0:
        return np.zeros(2 * TOTAL_BINS, dtype=np.float32)
    weights = _resolve_bin_weights(peaks_arr, intensities, intensity_mode)
    ch2_mask = peaks_arr < 0
    ch2_abs = np.abs(peaks_arr[ch2_mask])
    nonch2 = peaks_arr[~ch2_mask]
    w_ch2 = weights[ch2_mask] if weights is not None else None
    w_non = weights[~ch2_mask] if weights is not None else None
    return np.concatenate([_bin_peaks(ch2_abs, w_ch2), _bin_peaks(nonch2, w_non)])


def _resolve_bin_weights(
    peaks_arr: np.ndarray,
    intensities: Optional[np.ndarray],
    intensity_mode: str,
) -> Optional[np.ndarray]:
    """Return per-peak weights for binning, or None for plain (binary) counts.

    ``None`` is returned in binary mode so that ``np.bincount`` uses its fast
    unit-weight path and the output is bit-for-bit identical to the previous
    count-based encoder.
    """
    if intensity_mode == "binary" or intensities is None:
        return None
    return resolve_intensity_weights(peaks_arr, intensities, intensity_mode)


# ---------------------------------------------------------------------------
# Convenience pairwise wrapper
# ---------------------------------------------------------------------------


def ppm_bins_similarity(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Bin two DEPT spectra (single channel) and return a similarity score.

    Thin wrapper for ad-hoc use.  In batch pipelines call ``generate_ppm_bins``
    directly to precompute vectors once per spectrum.

    Parameters
    ----------
    peaks_a, peaks_b:
        1D float32 arrays of signed cppm values.
    metric:
        ``"cosine"`` (default), ``"euclidean"``, or ``"manhattan"``.
        Euclidean and manhattan are distance metrics — lower = more similar.
    """
    return float(
        compute_metric_numba(metric, generate_ppm_bins(peaks_a), generate_ppm_bins(peaks_b))
    )


def ppm_bins_typed_similarity(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    metric: str = "cosine",
) -> float:
    """Bin two DEPT spectra (two-channel typed) and return a similarity score.

    Uses ``generate_ppm_bins_typed`` so that CH2 and CH/CH3 peaks are compared
    in separate channels.

    Parameters
    ----------
    peaks_a, peaks_b:
        1D float32 arrays of signed cppm values.
    metric:
        ``"cosine"`` (default), ``"euclidean"``, or ``"manhattan"``.
    """
    return float(
        compute_metric_numba(
            metric, generate_ppm_bins_typed(peaks_a), generate_ppm_bins_typed(peaks_b)
        )
    )
