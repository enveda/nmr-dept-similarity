"""Shared retrieval helpers for the robustness / analog / per-class experiments.

Used by notebook 07 (query-noise robustness) and the ``scripts/`` CLI drivers
(``exp_noise_run.py``, ``exp_analog_run.py``). Reuses the exact scoring paths from
notebook 04 (Gaussian / shift-binned cosine as a single library matmul; DEPT-Match via
Hungarian alignment) but lets the caller pass in *perturbed* query peaks so we can measure
robustness to chemical-shift noise.

The candidate set is fixed once from the clean queries (``build_experiment_inputs``) so that
every noise level is scored against an identical library pool — degradation then reflects
ranking robustness alone, not a shifting prefilter recall ceiling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dept_similarity.bin_vector import generate_ppm_bins_typed
from dept_similarity.constants import PROCESSED_DATA_DIR
from dept_similarity.gaussian_vector import generate_gaussian_vector
from dept_similarity.prepare import prefilter_pairs_by_peak_overlap
from dept_similarity.spectral_matching import (
    align_dept_peaks_hungarian,
    cosine_from_alignment,
    jaccard_from_alignment,
)

TOP_K = 10


# ---------------------------------------------------------------------------
# Data loading: fixed candidate pool + shared lookups
# ---------------------------------------------------------------------------
def build_experiment_inputs():
    """Load library + query spectra and build the clean-query candidate pool.

    Returns a dict with the fixed candidate ``pairs`` (query_id -> sorted candidate ids),
    per-spectrum peak lookups and library structure lookups — everything the noise sweep
    needs. The candidate pool is built once from the *clean* queries so that all noise
    levels share an identical library pool.
    """
    def _arr(v):
        return v.astype(np.float32, copy=False) if isinstance(v, np.ndarray) else np.asarray(v, np.float32)

    lib = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "library_data.parquet")
    q = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "experimental_validation_set.parquet")
    lib["signed_shifts"] = lib["signed_shifts"].map(_arr)
    q["signed_shifts"] = q["signed_shifts"].map(_arr)
    if not (q["compound_id"].str.split("_").str[-1] == q["inchikey14"]).all():
        q["compound_id"] = q["compound_id"] + "_" + q["inchikey14"]

    pairs = {k: sorted(v) for k, v in prefilter_pairs_by_peak_overlap(q, lib).items()}
    return {
        "query_df": q,
        "pairs": pairs,
        "query_peaks": q.set_index("compound_id")["signed_shifts"].to_dict(),
        "lib_peaks": lib.set_index("compound_id")["signed_shifts"].to_dict(),
        "lib_smiles": lib.set_index("compound_id")["smiles"].to_dict(),
        "lib_inchikey": lib.set_index("compound_id")["inchikey"].to_dict(),
        "n_total_queries": q["compound_id"].nunique(),
    }


# ---------------------------------------------------------------------------
# Chemical-shift noise model
# ---------------------------------------------------------------------------
def perturb_peaks(peaks: np.ndarray, sigma: float, rng: np.random.Generator, mode: str) -> np.ndarray:
    """Add chemical-shift noise to a signed-shift spectrum.

    ``peaks`` encode multiplicity in the sign and shift in the magnitude. Noise is added
    to the *magnitude* only (calibration / solvent effects move a peak's ppm but do not
    change whether a carbon is CH/CH3 vs CH2), so the sign is preserved.

    mode = "jitter"  : each peak shifted independently ~ N(0, sigma)  (peak-picking / resolution)
    mode = "offset"  : whole spectrum shifted by one draw ~ N(0, sigma) (referencing / calibration)
    """
    if sigma <= 0:
        return peaks
    signs = np.sign(peaks)
    signs[signs == 0] = 1.0
    mag = np.abs(peaks).astype(np.float64)
    if mode == "jitter":
        mag = mag + rng.normal(0.0, sigma, size=mag.shape)
    elif mode == "offset":
        mag = mag + rng.normal(0.0, sigma)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    mag = np.clip(mag, 0.01, None)  # a shift magnitude must stay positive
    return (signs * mag).astype(np.float32)


# ---------------------------------------------------------------------------
# Vector methods (Gaussian, shift-binned): one library encode, reused per level
# ---------------------------------------------------------------------------
def _row_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def encode_library_vectors(cand_order, lib_peaks, method, gauss_sigma, n_jobs=-1):
    """Encode every unique candidate once (row-normalised). Reused across all noise levels."""
    if method == "gauss_kernel":
        jobs = (delayed(generate_gaussian_vector)(lib_peaks[c], sigma=gauss_sigma) for c in cand_order)
    elif method == "ppm_bins_typed":
        jobs = (delayed(generate_ppm_bins_typed)(lib_peaks[c]) for c in cand_order)
    else:
        raise ValueError(method)
    mat = np.stack(Parallel(n_jobs=n_jobs, backend="multiprocessing")(jobs))
    return _row_normalise(mat)


def rank_vector_method(query_ids, query_peaks_lookup, pairs, cand_index, lib_norm,
                       method, gauss_sigma, lib_smiles, lib_inchikey):
    """Rank candidates for every query with a fixed (already-encoded) library matrix.

    ``query_peaks_lookup`` maps query_id -> (possibly perturbed) peak array.
    """
    if method == "gauss_kernel":
        def enc(p):
            return generate_gaussian_vector(p, sigma=gauss_sigma)
    else:
        enc = generate_ppm_bins_typed

    # Encode every query, then score all candidates in one batched GEMM:
    #   scores_full[cand, query] = lib_norm(n_cand, D) @ Q(D, n_query)
    # BLAS runs this multithreaded in seconds; far cheaper than 648 separate
    # memory passes over the multi-GB library matrix.
    q_mat = _row_normalise(np.stack([enc(query_peaks_lookup[q]) for q in query_ids]))  # (nq, D)
    scores_full = lib_norm @ q_mat.T  # (n_cand, nq), float32

    rows = []
    for j, q in enumerate(query_ids):
        cand = pairs[q]
        if not cand:
            continue
        idx = np.fromiter((cand_index[c] for c in cand), dtype=np.int64, count=len(cand))
        scores = scores_full[idx, j]
        order = np.argsort(-scores, kind="stable")[:TOP_K]
        sel = [cand[i] for i in order]
        rows.append({
            "query_id": q, "metric_name": "cosine",
            "candidate_id_list": sel,
            "score_list": scores[order].astype(float).tolist(),
            "candidate_smiles_list": [lib_smiles.get(c) for c in sel],
            "candidate_inchikey_list": [lib_inchikey.get(c) for c in sel],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DEPT-Match (Hungarian) : re-aligns every pair; parallel over queries
# ---------------------------------------------------------------------------
# The library-peaks dict (452k arrays) and structure lookups are shared read-only via
# fork inheritance (copy-on-write) rather than passed as task arguments — otherwise joblib
# would re-pickle the multi-GB dict once per query task. Set these module globals with
# ``set_hungarian_globals`` before calling ``rank_hungarian``.
_LIB_PEAKS = None
_LIB_SMILES = None
_LIB_INCHIKEY = None


def set_hungarian_globals(lib_peaks, lib_smiles, lib_inchikey):
    global _LIB_PEAKS, _LIB_SMILES, _LIB_INCHIKEY
    _LIB_PEAKS, _LIB_SMILES, _LIB_INCHIKEY = lib_peaks, lib_smiles, lib_inchikey


def _rank_query_hungarian(q, qpeaks, cand, ppm_tol):
    if not cand:
        return []
    lib_peaks = _LIB_PEAKS
    cos = np.empty(len(cand))
    jac = np.empty(len(cand))
    for i, c in enumerate(cand):
        res = align_dept_peaks_hungarian(qpeaks, lib_peaks[c], ppm_tolerance=ppm_tol,
                                         match_multiplicity=True)
        cos[i] = cosine_from_alignment(res, 1)
        jac[i] = jaccard_from_alignment(res, 1)
    out = []
    for metric, sc in (("cosine", cos), ("jaccard", jac)):
        order = np.argsort(-sc, kind="stable")[:TOP_K]
        sel = [cand[i] for i in order]
        out.append({
            "query_id": q, "metric_name": metric,
            "candidate_id_list": sel,
            "score_list": sc[order].astype(float).tolist(),
            "candidate_smiles_list": [_LIB_SMILES.get(c) for c in sel],
            "candidate_inchikey_list": [_LIB_INCHIKEY.get(c) for c in sel],
        })
    return out


def rank_hungarian(query_ids, query_peaks_lookup, pairs, ppm_tol, n_jobs=-1):
    """Rank via Hungarian alignment. Requires ``set_hungarian_globals`` to have been called."""
    nested = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(_rank_query_hungarian)(q, query_peaks_lookup[q], pairs[q], ppm_tol)
        for q in query_ids
    )
    flat = [r for chunk in nested for r in chunk]
    return {
        "cosine": pd.DataFrame([r for r in flat if r["metric_name"] == "cosine"]),
        "jaccard": pd.DataFrame([r for r in flat if r["metric_name"] == "jaccard"]),
    }
