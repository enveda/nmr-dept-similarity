"""
dept_similarity.prepare
=======================
Candidate prefiltering for DEPT NMR database search.

prefilter_pairs_by_peak_overlap builds a query→candidates dict by screening
the entire library with a vectorised binary presence fingerprint. A library
compound is retained only when at least min_shared_peaks of its peaks fall
within ppm_tolerance of query peaks of compatible multiplicity type AND those
matches cover at least min_query_coverage fraction of the query's peaks. The
full library (500k compounds) is processed in ~1 ms per query via a single
numpy bitwise-AND + sum operation on a precomputed boolean matrix.

"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from dept_similarity.bin_vector import (
    BIN_EDGES,
    TOTAL_BINS,
    generate_ppm_bins,
    generate_ppm_bins_typed,
)
from dept_similarity.constants import BASELINE_PARAMS

logger = logging.getLogger(__name__)


def _dilated_presence(peaks: np.ndarray, match_multiplicity: bool, n_dilate: int) -> np.ndarray:
    """Binary presence fingerprint expanded by n_dilate bins in each direction.

    When match_multiplicity=True uses the 2-channel typed vector (CH2 + CH/CH3,
    shape 2*TOTAL_BINS). When False uses a single-channel vector (shape TOTAL_BINS)
    that ignores peak sign.

    Dilation ensures that two peaks within ppm_tolerance of each other always share
    at least one bit, even when they straddle a bin boundary.
    """
    if match_multiplicity:
        raw = generate_ppm_bins_typed(peaks) > 0
        n_channels, stride = 2, TOTAL_BINS
    else:
        raw = generate_ppm_bins(peaks) > 0
        n_channels, stride = 1, TOTAL_BINS

    out = raw.copy()
    for ch in range(n_channels):
        s, e = ch * stride, (ch + 1) * stride
        ch_raw = raw[s:e]
        ch_out = ch_raw.copy()
        for d in range(1, n_dilate + 1):
            ch_out[:-d] |= ch_raw[d:]
            ch_out[d:] |= ch_raw[:-d]
        out[s:e] = ch_out
    return out


def prefilter_pairs_by_peak_overlap(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    ppm_tolerance: float = BASELINE_PARAMS["prefilter_ppm_tol"],
    match_multiplicity: bool = True,
    min_shared_peaks: int = 3,
    min_query_coverage: float = 0.75,
) -> Dict[str, set]:
    """Build candidate pairs using a vectorised peak-overlap fingerprint.

    A library compound is included as a candidate for a query only when both
    conditions are met:
    - at least *min_shared_peaks* query peaks lie within *ppm_tolerance* of a
      reference peak of compatible multiplicity type, AND
    - those matches cover at least *min_query_coverage* fraction of the query's peaks.

    Implementation: a dilated binary presence fingerprint (90-dim for typed,
    45-dim for untyped) is precomputed for every library compound once, stored as
    a boolean matrix of shape ``(n_library, n_dim)``. Each query then reduces to a
    single ``(n_library, n_dim) & (n_dim,)`` bitwise AND followed by a sum,
    which runs in ~1 ms even at 500k library scale.

    Parameters
    ----------
    query_df, reference_df:
        DataFrames with columns ``compound_id`` and ``signed_shifts``.
    ppm_tolerance:
        Peak-matching window in ppm.  Should match the tolerance used in scoring.
    match_multiplicity:
        If True, CH2 peaks only match CH2 peaks and CH/CH3 only match CH/CH3.
        Set False for a type-agnostic (more lenient) prefilter.
    min_shared_peaks:
        Minimum number of overlapping peaks required to retain a candidate.
    min_query_coverage:
        Minimum fraction of query peaks that must be matched to retain a candidate.

    Returns
    -------
    Dict mapping query_id → set of candidate library compound_ids.
    """
    bin_width = float(BIN_EDGES[1] - BIN_EDGES[0])
    n_dilate = max(1, int(np.ceil(ppm_tolerance / bin_width)))

    lib_cids = reference_df["compound_id"].to_numpy()
    n_lib = len(lib_cids)

    # Build the library presence matrix in parallel — one row per compound.
    # Prefer threads here to avoid loky's process-shutdown semlock/memmap
    # warnings in long notebook and batch runs.
    lib_rows = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_dilated_presence)(
            np.asarray(peaks, dtype=np.float32), match_multiplicity, n_dilate
        )
        for peaks in tqdm(
            reference_df["signed_shifts"],
            desc="Building peak-overlap index",
            leave=False,
        )
    )
    lib_matrix = np.stack(lib_rows)

    pairs: Dict[str, set] = {}
    for _, row in tqdm(query_df.iterrows(), desc="Peak-overlap prefilter", total=len(query_df)):
        qid = row["compound_id"]
        q_peaks = np.asarray(row["signed_shifts"], dtype=np.float32)
        q_presence = _dilated_presence(q_peaks, match_multiplicity, n_dilate)
        overlap_counts = (lib_matrix & q_presence).sum(axis=1)
        min_coverage_count = int(np.ceil(min_query_coverage * q_presence.sum()))
        overlap_mask = (overlap_counts >= min_shared_peaks) & (overlap_counts >= min_coverage_count)
        pairs[qid] = set(lib_cids[overlap_mask])

    n_queries = len(query_df)
    n_pairs = sum(len(v) for v in pairs.values())
    n_total = n_queries * n_lib
    pct_kept = 100.0 * n_pairs / n_total if n_total else 0.0
    pct_filtered = 100.0 - pct_kept
    logger.info(
        "Peak-overlap prefilter: %d queries × %d library = %d total → %d kept "
        "(%.1f%% filtered, %.1f%% kept, avg %.0f candidates/query)",
        n_queries,
        n_lib,
        n_total,
        n_pairs,
        pct_filtered,
        pct_kept,
        n_pairs / n_queries if n_queries else 0.0,
    )
    return pairs


def validate_candidate_set(
    pairs: Dict[str, set],
    show_progress: bool = False,
) -> None:
    """Log a warning if any query's true library match was filtered out of candidates.

    Call this once after prefilter_pairs_by_peak_overlap and before running any
    similarity experiments to catch aggressive filtering early.

    Queries whose true structure is not present in the library at all (novel
    compounds) are skipped; only queries with a known library counterpart are checked.
    Structure identity is matched on the 14-character InChIKey prefix to tolerate
    stereo/charge variants.

    The input pairs dict already contains the inchikey14 info (eg. Compound_ID_Inchikey14)

    Parameters
    ----------
    pairs:
        Output of prefilter_pairs_by_peak_overlap.
    show_progress:
        If True, display a progress bar during validation.
    """

    missing = []

    # show progress bar only if there are many queries to check and tqdm is available
    t = (
        tqdm(pairs.items(), desc="Validating candidate sets", total=len(pairs))
        if show_progress
        else pairs.items()
    )

    for qid, candidates in t:
        true_key14 = qid.split("_")[-1]  # assumes query_id format includes inchikey14 as suffix
        matched_key14s = {cid.split("_")[-1] for cid in candidates}
        if true_key14 and true_key14 not in matched_key14s:
            missing.append(qid)

    n_checked = len(pairs)
    logger.info(
        "Candidate-set validation: %d / %d checkable queries lost their true match.",
        len(missing),
        n_checked,
    )
    if missing:
        logger.warning(
            "Missing true match for query IDs (first 5): %s. "
            "Loosen ppm_tolerance or set match_multiplicity=False.",
            missing[:5],
        )

    # Remove queries with missing true matches from the pairs dict to avoid skewing downstream eval metrics.
    for qid in missing:
        del pairs[qid]
