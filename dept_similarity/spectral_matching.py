"""Alignment and scoring for DEPT spectral similarity.

The core two-pointer sweep is JIT-compiled with numba so that the per-pair
alignment cost is dominated by the O(n+m) algorithm itself rather than Python
interpreter overhead.  A warm-up call at import time ensures the compiled
kernel is ready before the first real invocation.

Intensity weighting
--------------------
Every peak carries a scalar weight.  In the default ``"binary"`` mode all
weights are 1.0, so each present peak contributes equally and the metrics
reduce to peak-presence set similarity.  In ``"normalized"`` mode a paired
intensity array is min-normalised per spectrum (divide by the maximum
intensity, so the strongest peak weighs 1.0 and the rest scale proportionally)
and the metrics become intensity-weighted:

* cosine  = Σ_matched wa·wb / sqrt(Σ wa² · Σ wb²)
* jaccard = Σ_matched min(wa, wb) / (Σ wa + Σ wb − Σ_matched min(wa, wb))  (Ruzicka)
* recall  = Σ_matched wa / Σ wa

With all weights equal to 1.0 these collapse to the classic count-based
cosine / Jaccard / recall, so binary mode is bit-for-bit backward compatible.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment

from dept_similarity.constants import BASELINE_PARAMS

# ---------------------------------------------------------------------------
# Numba kernel — compiled once, reused for every pair
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _align_kernel(
    abs_a: np.ndarray,
    abs_b: np.ndarray,
    is_ch2_a: np.ndarray,
    is_ch2_b: np.ndarray,
    wa: np.ndarray,
    wb: np.ndarray,
    ppm_tolerance: float,
    match_multiplicity: bool,
) -> tuple:
    """Strict 1-to-1 two-pointer alignment kernel with intensity weighting.

    Each peak is used at most once.  On a positional match within tolerance,
    both pointers advance.  On a type mismatch (when ``match_multiplicity``
    is True), only the pointer pointing at the smaller absolute ppm advances
    — discarding the leftmost unmatched peak rather than consuming both,
    which would skip a valid same-type pair nearby.

    Alongside the match count it accumulates the three weighted sums needed to
    derive any peak-count / intensity-weighted metric downstream:

    - ``matched_dot`` = Σ wa·wb over matched pairs (cosine numerator),
    - ``matched_min`` = Σ min(wa, wb) over matched pairs (Ruzicka intersection),
    - ``matched_wa``  = Σ wa over matched pairs (recall numerator).

    Algorithm
    ---------
    - ``diff < -tolerance``: a[i] is strictly left of b[j]'s window → advance i.
    - ``diff > +tolerance``: b[j] is strictly left of a[i]'s window → advance j.
    - Within tolerance, types agree (or unchecked) → count match, advance both.
    - Within tolerance, types disagree → advance whichever pointer is smaller,
      leaving the other available for the next candidate in the opposite spectrum.
    """
    n_a = abs_a.shape[0]
    n_b = abs_b.shape[0]
    matched = 0
    matched_dot = 0.0
    matched_min = 0.0
    matched_wa = 0.0
    i = 0
    j = 0

    while i < n_a and j < n_b:
        diff = abs_a[i] - abs_b[j]

        if diff < -ppm_tolerance:
            i += 1
        elif diff > ppm_tolerance:
            j += 1
        else:
            # Peaks are within tolerance
            if (not match_multiplicity) or (is_ch2_a[i] == is_ch2_b[j]):
                # Types agree (or not checked): confirmed match, consume both
                wai = wa[i]
                wbj = wb[j]
                matched += 1
                matched_dot += wai * wbj
                matched_min += wai if wai < wbj else wbj
                matched_wa += wai
                i += 1
                j += 1
            else:
                # Type mismatch: discard the leftmost peak only so the other
                # remains available for the next peak in the opposite spectrum
                if abs_a[i] < abs_b[j]:
                    i += 1
                else:
                    j += 1

    return matched, matched_dot, matched_min, matched_wa


# Warm-up: compile both match_multiplicity variants at import time so neither
# pays the JIT cost on the first real invocation.
_DUMMY = np.array([1.0], dtype=np.float32)
_DUMMY_BOOL = np.array([False], dtype=np.bool_)
_align_kernel(_DUMMY, _DUMMY, _DUMMY_BOOL, _DUMMY_BOOL, _DUMMY, _DUMMY, 0.5, True)
_align_kernel(_DUMMY, _DUMMY, _DUMMY_BOOL, _DUMMY_BOOL, _DUMMY, _DUMMY, 0.5, False)
del _DUMMY, _DUMMY_BOOL


# ---------------------------------------------------------------------------
# Intensity weights
# ---------------------------------------------------------------------------


def resolve_intensity_weights(
    signed_peaks: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    intensity_mode: str = "binary",
) -> np.ndarray:
    """Resolve per-peak scalar weights for a spectrum.

    Parameters
    ----------
    signed_peaks:
        1D array of signed ppm values.  Only its length is used here; the sign
        (multiplicity) is handled by the alignment kernels.
    intensities:
        Optional 1D array of peak intensities, aligned 1-to-1 with
        ``signed_peaks``.  Required (and must match length) when
        ``intensity_mode="normalized"``.
    intensity_mode:
        - ``"binary"`` (default): every present peak weighs 1.0.  ``intensities``
          is ignored.  Reproduces classic peak-presence set similarity.
        - ``"normalized"``: weights are ``|intensities| / max(|intensities|)``
          so the strongest peak weighs 1.0 and the rest scale proportionally.
          The absolute value is taken because multiplicity is already encoded in
          the sign of ``signed_peaks``, not the intensity.

    Returns
    -------
    np.ndarray
        Float32 weights, same length as ``signed_peaks``.
    """
    n = int(np.asarray(signed_peaks).size)

    if intensity_mode == "binary" or intensities is None:
        return np.ones(n, dtype=np.float32)

    if intensity_mode == "normalized":
        w = np.abs(np.asarray(intensities, dtype=np.float32))
        if w.size != n:
            raise ValueError(
                f"intensities length ({w.size}) does not match peaks length ({n})."
            )
        if w.size == 0:
            return w
        mx = float(w.max())
        if mx > 0.0:
            w = w / mx
        return w.astype(np.float32, copy=False)

    raise ValueError(
        f"Unknown intensity_mode '{intensity_mode}'. Choose 'binary' or 'normalized'."
    )


# ---------------------------------------------------------------------------
# AlignmentResult
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """Raw alignment sums returned by ``align_dept_peaks``.

    Contains everything needed to compute any peak-count-based or
    intensity-weighted similarity metric downstream, without committing to a
    particular formula.

    Attributes
    ----------
    n_matched:
        Number of peak pairs that were within ``ppm_tolerance`` of each other
        in absolute chemical shift and, when ``match_multiplicity=True``, also
        share the same multiplicity type (CH2–CH2 or non-CH2–non-CH2).
    n_peaks_a, n_peaks_b:
        Total number of peaks in each input spectrum.
    matched_dot:
        Σ (wa · wb) over matched pairs — cosine numerator.
    matched_min:
        Σ min(wa, wb) over matched pairs — Ruzicka (weighted Jaccard) intersection.
    matched_wa:
        Σ wa over matched pairs — recall numerator.
    norm_a_sq, norm_b_sq:
        Σ wa² and Σ wb² — cosine denominator terms.
    sum_wa, sum_wb:
        Σ wa and Σ wb — Jaccard union / recall denominator terms.

    The five weighted fields default to their binary equivalents (weights all
    1.0) when left unset, so ``AlignmentResult(n_matched, n_peaks_a, n_peaks_b)``
    behaves exactly like classic count-based scoring.
    """

    n_matched: int
    n_peaks_a: int
    n_peaks_b: int
    matched_dot: Optional[float] = None
    matched_min: Optional[float] = None
    matched_wa: Optional[float] = None
    norm_a_sq: Optional[float] = None
    norm_b_sq: Optional[float] = None
    sum_wa: Optional[float] = None
    sum_wb: Optional[float] = None

    def __post_init__(self) -> None:
        # Fill unset weighted sums with the binary (all-weights-1.0) equivalents
        # so that count-only construction reproduces classic set similarity.
        m = float(self.n_matched)
        if self.matched_dot is None:
            self.matched_dot = m
        if self.matched_min is None:
            self.matched_min = m
        if self.matched_wa is None:
            self.matched_wa = m
        if self.norm_a_sq is None:
            self.norm_a_sq = float(self.n_peaks_a)
        if self.norm_b_sq is None:
            self.norm_b_sq = float(self.n_peaks_b)
        if self.sum_wa is None:
            self.sum_wa = float(self.n_peaks_a)
        if self.sum_wb is None:
            self.sum_wb = float(self.n_peaks_b)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def align_dept_peaks(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    ppm_tolerance: float = BASELINE_PARAMS["spectral_ppm_tol"],
    match_multiplicity: bool = True,
    intensities_a: Optional[np.ndarray] = None,
    intensities_b: Optional[np.ndarray] = None,
    intensity_mode: str = "binary",
) -> AlignmentResult:
    """Align two DEPT spectra using a two-pointer sweep.

    Peaks are 1D arrays of signed ppm values (convention: negative = CH2,
    positive = CH/CH3).  Alignment is on absolute chemical shift; multiplicity
    is optionally checked within each tolerance window.

    Parameters
    ----------
    peaks_a, peaks_b:
        1D float32 arrays of signed ppm values.
    ppm_tolerance:
        Maximum absolute ppm difference for two peaks to be considered aligned.
        Default is taken from ``BASELINE_PARAMS["spectral_ppm_tol"]`` (3.0 ppm).
    match_multiplicity:
        If True (default), peaks must share the same type (CH2–CH2 or
        non-CH2–non-CH2) to count as a match.  Set to False for
        positional-only matching, which is useful for ablation studies
        isolating whether CH2 assignment quality is limiting retrieval.
    intensities_a, intensities_b:
        Optional peak intensities aligned 1-to-1 with ``peaks_a`` / ``peaks_b``.
        Used only when ``intensity_mode="normalized"``.
    intensity_mode:
        ``"binary"`` (default, peak-presence) or ``"normalized"`` (max-normalised
        intensity weighting).  See ``resolve_intensity_weights``.

    Returns
    -------
    AlignmentResult
        Raw match sums; pass to a scoring function for a similarity score.
    """
    prep_a = preprocess_dept_peaks(peaks_a, intensities_a, intensity_mode)
    prep_b = preprocess_dept_peaks(peaks_b, intensities_b, intensity_mode)
    return align_dept_preprocessed(
        prep_a,
        prep_b,
        ppm_tolerance=ppm_tolerance,
        match_multiplicity=match_multiplicity,
    )


def preprocess_dept_peaks(
    peaks: np.ndarray,
    intensities: Optional[np.ndarray] = None,
    intensity_mode: str = "binary",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Precompute sorted absolute shifts, type mask and weights for repeated matching.

    Returns a compact representation ``(abs_sorted, is_ch2_sorted, weights_sorted,
    n_peaks)`` that can be reused across many pairwise comparisons to avoid
    repeated sorting, mask construction and weight resolution overhead.

    Parameters
    ----------
    peaks:
        1D array of signed ppm values.
    intensities:
        Optional peak intensities aligned 1-to-1 with ``peaks``; used only when
        ``intensity_mode="normalized"``.
    intensity_mode:
        ``"binary"`` (default) or ``"normalized"``.
    """
    arr = np.asarray(peaks, dtype=np.float32)
    if arr.size == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.bool_),
            np.empty(0, dtype=np.float32),
            0,
        )

    weights = resolve_intensity_weights(arr, intensities, intensity_mode)

    order = np.argsort(np.abs(arr))
    arr_sorted = arr[order]
    weights_sorted = weights[order]
    return np.abs(arr_sorted), (arr_sorted < 0), weights_sorted, int(arr.size)


def align_dept_preprocessed(
    prep_a: tuple[np.ndarray, np.ndarray, np.ndarray, int],
    prep_b: tuple[np.ndarray, np.ndarray, np.ndarray, int],
    ppm_tolerance: float = BASELINE_PARAMS["spectral_ppm_tol"],
    match_multiplicity: bool = True,
) -> AlignmentResult:
    """Align two preprocessed DEPT spectra using the Numba kernel directly.

    This variant avoids per-pair preprocessing and is intended for high-volume
    query-vs-library scoring where each spectrum is compared many times.
    """
    abs_a, is_ch2_a, wa, n_peaks_a = prep_a
    abs_b, is_ch2_b, wb, n_peaks_b = prep_b

    norm_a_sq = float(np.dot(wa, wa)) if n_peaks_a else 0.0
    norm_b_sq = float(np.dot(wb, wb)) if n_peaks_b else 0.0
    sum_wa = float(wa.sum()) if n_peaks_a else 0.0
    sum_wb = float(wb.sum()) if n_peaks_b else 0.0

    if n_peaks_a == 0 or n_peaks_b == 0:
        return AlignmentResult(
            n_matched=0,
            n_peaks_a=n_peaks_a,
            n_peaks_b=n_peaks_b,
            matched_dot=0.0,
            matched_min=0.0,
            matched_wa=0.0,
            norm_a_sq=norm_a_sq,
            norm_b_sq=norm_b_sq,
            sum_wa=sum_wa,
            sum_wb=sum_wb,
        )

    matched, matched_dot, matched_min, matched_wa = _align_kernel(
        abs_a,
        abs_b,
        is_ch2_a,
        is_ch2_b,
        wa,
        wb,
        float(ppm_tolerance),
        match_multiplicity,
    )
    return AlignmentResult(
        n_matched=int(matched),
        n_peaks_a=n_peaks_a,
        n_peaks_b=n_peaks_b,
        matched_dot=float(matched_dot),
        matched_min=float(matched_min),
        matched_wa=float(matched_wa),
        norm_a_sq=norm_a_sq,
        norm_b_sq=norm_b_sq,
        sum_wa=sum_wa,
        sum_wb=sum_wb,
    )


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def cosine_from_alignment(result: AlignmentResult, min_matched_peaks: int = 1) -> float:
    """Cosine similarity from alignment sums.

    Cosine = Σ_matched (wa·wb) / sqrt(Σ wa² · Σ wb²).  In binary mode every
    weight is 1.0, so this reduces to n_matched / sqrt(n_peaks_a · n_peaks_b).

    Returns 0.0 if fewer than ``min_matched_peaks`` were aligned.
    """
    if result.n_matched < min_matched_peaks:
        return 0.0
    norm = (float(result.norm_a_sq) * float(result.norm_b_sq)) ** 0.5
    if norm == 0.0:
        return 0.0
    return float(result.matched_dot) / norm


def jaccard_from_alignment(result: AlignmentResult, min_matched_peaks: int = 1) -> float:
    """Jaccard (Ruzicka) similarity from alignment sums.

    Jaccard = Σ_matched min(wa, wb) / (Σ wa + Σ wb − Σ_matched min(wa, wb)).
    In binary mode this reduces to n_matched / (n_peaks_a + n_peaks_b − n_matched),
    the classic intersection-over-union on peak sets.  Penalises unmatched
    peaks symmetrically on both sides, making it appropriate when both spectra
    are expected to be complete observations.

    Returns 0.0 if fewer than ``min_matched_peaks`` were aligned.
    """
    if result.n_matched < min_matched_peaks:
        return 0.0
    denom = float(result.sum_wa) + float(result.sum_wb) - float(result.matched_min)
    if denom == 0.0:
        return 0.0
    return float(result.matched_min) / denom


def recall_from_alignment(result: AlignmentResult, min_matched_peaks: int = 1) -> float:
    """Query recall from alignment sums.

    Recall = Σ_matched wa / Σ wa.  In binary mode this reduces to
    n_matched / n_peaks_a.

    Measures the fraction of query (intensity) that was found in the library
    compound, irrespective of how many additional peaks the library compound
    has.  This is the most appropriate metric for DEPT structure elucidation:
    a library compound that contains all observed carbon signals scores 1.0
    even if it has many more carbons than the query.

    Fixes the size-asymmetry bias of cosine similarity, where a small query
    matched against a large library compound is penalised by the geometric-mean
    denominator sqrt(n_a × n_b).

    Returns 0.0 if fewer than ``min_matched_peaks`` were aligned.
    """
    if result.n_matched < min_matched_peaks:
        return 0.0
    if float(result.sum_wa) == 0.0:
        return 0.0
    return float(result.matched_wa) / float(result.sum_wa)


# ---------------------------------------------------------------------------
# Convenience wrapper for ad-hoc pairwise use
# ---------------------------------------------------------------------------


def dept_spectral_matching(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    ppm_tolerance: float = BASELINE_PARAMS["spectral_ppm_tol"],
    min_matched_peaks: int = BASELINE_PARAMS["spectral_min_matched_peaks"],
    metric: str = "cosine",
    match_multiplicity: bool = True,
    intensities_a: Optional[np.ndarray] = None,
    intensities_b: Optional[np.ndarray] = None,
    intensity_mode: str = "binary",
) -> float:
    """Align two DEPT spectra and return a similarity score.

    Thin wrapper around ``align_dept_peaks`` + a scoring function.
    Use ``align_dept_peaks`` directly when you need the raw sums or want
    to compute multiple metrics from a single alignment pass.

    Parameters
    ----------
    peaks_a, peaks_b:
        1D float32 arrays of signed ppm values.
    ppm_tolerance:
        Passed to ``align_dept_peaks``.  Default from BASELINE_PARAMS (3.0 ppm).
    min_matched_peaks:
        Minimum aligned peaks required for a non-zero score.  Default 1.
    metric:
        ``"cosine"`` (default), ``"jaccard"``, or ``"recall"``.
    match_multiplicity:
        Passed to ``align_dept_peaks``.  Default True.
    intensities_a, intensities_b, intensity_mode:
        Intensity weighting controls; see ``align_dept_peaks``.
    """
    _SCORING = {
        "cosine": cosine_from_alignment,
        "jaccard": jaccard_from_alignment,
        "recall": recall_from_alignment,
    }
    if metric not in _SCORING:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {sorted(_SCORING)}")
    result = align_dept_peaks(
        peaks_a,
        peaks_b,
        ppm_tolerance,
        match_multiplicity,
        intensities_a=intensities_a,
        intensities_b=intensities_b,
        intensity_mode=intensity_mode,
    )
    return _SCORING[metric](result, min_matched_peaks)


# ---------------------------------------------------------------------------
# Hungarian (optimal assignment) alignment
# ---------------------------------------------------------------------------


def align_dept_peaks_hungarian(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    ppm_tolerance: float = BASELINE_PARAMS["spectral_ppm_tol"],
    match_multiplicity: bool = True,
    intensities_a: Optional[np.ndarray] = None,
    intensities_b: Optional[np.ndarray] = None,
    intensity_mode: str = "binary",
    weights_a: Optional[np.ndarray] = None,
    weights_b: Optional[np.ndarray] = None,
) -> AlignmentResult:
    """Align two DEPT spectra via optimal 1-to-1 peak assignment (Hungarian algorithm).

    Builds a cost matrix where entry (i, j) equals the absolute ppm difference
    between peak i from spectrum A and peak j from spectrum B when:

    * ``|shift_a[i] - shift_b[j]| ≤ ppm_tolerance``, AND
    * either ``match_multiplicity=False`` or both peaks share the same type
      (CH2–CH2 or non-CH2–non-CH2).

    Entries that violate either condition are set to a large sentinel (``1e9``),
    making them prohibitively expensive for the solver.
    ``scipy.optimize.linear_sum_assignment`` then finds the globally optimal
    1-to-1 assignment that minimises total ppm distance among valid pairs.

    Compared to the greedy two-pointer sweep in ``align_dept_peaks``, this
    approach is guaranteed to find the maximum-cardinality matching and, among
    all matchings of that cardinality, the one with the smallest total ppm
    displacement.  The difference is only observable when multiple peaks cluster
    within the tolerance window and the greedy ordering leads to a suboptimal
    choice — a rare but non-zero occurrence in real DEPT spectra.

    Parameters
    ----------
    peaks_a, peaks_b:
        1D float32 arrays of signed ppm values (negative = CH2, positive = CH/CH3).
    ppm_tolerance:
        Maximum absolute ppm difference for a valid match.  Default 3.0 ppm.
    match_multiplicity:
        If True (default), only CH2–CH2 and non-CH2–non-CH2 pairs are valid.
    intensities_a, intensities_b, intensity_mode:
        Intensity weighting controls; see ``align_dept_peaks``.  The optimal
        assignment itself is always chosen to minimise ppm displacement; the
        weights only enter the accumulated match sums used for scoring.
    weights_a, weights_b:
        Optional precomputed per-peak weight arrays (aligned 1-to-1 with
        ``peaks_a`` / ``peaks_b``).  When given they are used directly and
        ``intensities_*`` / ``intensity_mode`` are ignored — this lets callers
        resolve weights once and reuse them across many pairwise comparisons.

    Returns
    -------
    AlignmentResult
        Raw match sums; pass to a scoring function for a similarity score.
        Complexity: O(n³) via the Jonker–Volgenant solver in scipy, where n is
        max(n_peaks_a, n_peaks_b).  For typical DEPT peak counts (5–25) this
        is negligible compared to I/O overhead.
    """
    a = np.asarray(peaks_a, dtype=np.float32)
    b = np.asarray(peaks_b, dtype=np.float32)

    wa = (
        np.asarray(weights_a, dtype=np.float32)
        if weights_a is not None
        else resolve_intensity_weights(a, intensities_a, intensity_mode)
    )
    wb = (
        np.asarray(weights_b, dtype=np.float32)
        if weights_b is not None
        else resolve_intensity_weights(b, intensities_b, intensity_mode)
    )

    norm_a_sq = float(np.dot(wa, wa)) if a.size else 0.0
    norm_b_sq = float(np.dot(wb, wb)) if b.size else 0.0
    sum_wa = float(wa.sum()) if a.size else 0.0
    sum_wb = float(wb.sum()) if b.size else 0.0

    if a.size == 0 or b.size == 0:
        return AlignmentResult(
            n_matched=0,
            n_peaks_a=int(a.size),
            n_peaks_b=int(b.size),
            matched_dot=0.0,
            matched_min=0.0,
            matched_wa=0.0,
            norm_a_sq=norm_a_sq,
            norm_b_sq=norm_b_sq,
            sum_wa=sum_wa,
            sum_wb=sum_wb,
        )

    abs_a = np.abs(a)
    abs_b = np.abs(b)
    is_ch2_a = a < 0
    is_ch2_b = b < 0

    # Build cost matrix via broadcasting — no Python loop needed.
    ppm_diff = np.abs(abs_a[:, None] - abs_b[None, :]).astype(np.float64)
    within_tol = ppm_diff <= ppm_tolerance

    if match_multiplicity:
        same_type = is_ch2_a[:, None] == is_ch2_b[None, :]
        valid = within_tol & same_type
    else:
        valid = within_tol

    _INF = 1e9
    cost = np.where(valid, ppm_diff, _INF)

    row_ind, col_ind = linear_sum_assignment(cost)
    keep = cost[row_ind, col_ind] < _INF
    matched_rows = row_ind[keep]
    matched_cols = col_ind[keep]

    matched = int(keep.sum())
    wa_m = wa[matched_rows]
    wb_m = wb[matched_cols]
    matched_dot = float(np.dot(wa_m, wb_m))
    matched_min = float(np.minimum(wa_m, wb_m).sum())
    matched_wa = float(wa_m.sum())

    return AlignmentResult(
        n_matched=matched,
        n_peaks_a=int(a.size),
        n_peaks_b=int(b.size),
        matched_dot=matched_dot,
        matched_min=matched_min,
        matched_wa=matched_wa,
        norm_a_sq=norm_a_sq,
        norm_b_sq=norm_b_sq,
        sum_wa=sum_wa,
        sum_wb=sum_wb,
    )
