"""Code for matching spectra based on similarity metrics and alignment of choice."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from dept_similarity.bin_vector import generate_ppm_bins, generate_ppm_bins_typed
from dept_similarity.constants import BASELINE_PARAMS
from dept_similarity.gaussian_vector import generate_gaussian_vector
from dept_similarity.score import compute_metric_numba
from dept_similarity.spectral_matching import (
    align_dept_peaks,
    align_dept_peaks_hungarian,
    align_dept_preprocessed,
    cosine_from_alignment,
    jaccard_from_alignment,
    preprocess_dept_peaks,
    recall_from_alignment,
    resolve_intensity_weights,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """Describes how a similarity method encodes spectra and scores pairs.

    Every method exposes two steps with a uniform interface so that
    ``rank_library`` never needs to know whether a method is
    vector-based or pairwise:

    encode_fn(peaks) -> representation
        For vector-based methods (binning, Gaussian): returns a fixed-length
        numpy array precomputed once per spectrum and reused across all pairs.
        For pairwise methods (spectral matching): returns a method-specific
        representation optimised for repeated pairwise scoring.

    score_fn(repr_a, repr_b, metrics) -> dict[metric_name, score]
        Computes all requested metrics from a single pair of representations
        in one call.  For vector methods this avoids re-reading the vectors per
        metric; for pairwise methods it avoids re-running alignment per metric.

    supported_metrics : frozenset[str]
        Metric names accepted by this method's score_fn.

    distance_metrics : frozenset[str]
        Subset of supported_metrics where lower scores indicate more similarity
        (used to invert ranking direction in top-k selection).
    """

    encode_fn: Callable[[np.ndarray], Any]
    score_fn: Callable[..., dict[str, float]]
    supported_metrics: frozenset[str]
    distance_metrics: frozenset[str] = field(default_factory=frozenset)
    batch_score_fn: Callable[..., np.ndarray] | None = None


def _vector_score_fn(
    vec_a: np.ndarray, vec_b: np.ndarray, metrics: tuple[str, ...], **_kwargs: Any
) -> dict[str, float]:
    """Score two precomputed vectors across all requested metrics.

    Extra keyword arguments are silently ignored so that a uniform
    ``score_fn(repr_a, repr_b, metrics, **method_kwargs)`` call site works
    for both vector-based and pairwise methods.
    """
    return {m: float(compute_metric_numba(m, vec_a, vec_b)) for m in metrics}


def _vector_batch_score_fn(
    vec_a: np.ndarray, cand_matrix: np.ndarray, metrics: tuple[str, ...]
) -> np.ndarray:
    """Score a query vector against all candidate vectors in one numpy call.

    Parameters
    ----------
    vec_a:
        Query vector, shape (D,).
    cand_matrix:
        Candidate vectors stacked row-wise, shape (N, D).
    metrics:
        Metric names to compute.

    Returns
    -------
    np.ndarray of shape (N, len(metrics)), dtype float32.
    """
    n_cand = cand_matrix.shape[0]
    n_metrics = len(metrics)
    result = np.empty((n_cand, n_metrics), dtype=np.float32)

    for m_idx, metric in enumerate(metrics):
        if metric == "cosine":
            dots = cand_matrix @ vec_a
            norm_a = float(np.linalg.norm(vec_a))
            norms_b = np.linalg.norm(cand_matrix, axis=1)
            denom = norms_b * norm_a
            result[:, m_idx] = np.where(denom > 0, dots / denom, 0.0)
        elif metric == "euclidean":
            result[:, m_idx] = np.linalg.norm(cand_matrix - vec_a, axis=1)
        elif metric == "manhattan":
            result[:, m_idx] = np.abs(cand_matrix - vec_a).sum(axis=1)
        else:
            raise ValueError(f"Unsupported batch metric: {metric}")

    return result


def _pairwise_score_fn(
    peaks_a: Any,
    peaks_b: Any,
    metrics: tuple[str, ...],
    ppm_tolerance: float = BASELINE_PARAMS["spectral_ppm_tol"],
    min_matched_peaks: int = BASELINE_PARAMS["spectral_min_matched_peaks"],
    match_multiplicity: bool = True,
    **_kwargs: Any,
) -> dict[str, float]:
    """Align two spectra once and derive all requested metrics from the result.

    Parameters
    ----------
    ppm_tolerance:
        Maximum absolute ppm difference for alignment.
        Default from ``BASELINE_PARAMS`` (4.0 ppm).
    min_matched_peaks:
        Minimum aligned peaks required for a non-zero score.  Default 1.
    match_multiplicity:
        Whether to require CH2/non-CH2 type agreement within each alignment
        window.  Set to False for positional-only matching (ablation study).
    """
    if isinstance(peaks_a, tuple) and isinstance(peaks_b, tuple):
        result = align_dept_preprocessed(
            peaks_a,
            peaks_b,
            ppm_tolerance=ppm_tolerance,
            match_multiplicity=match_multiplicity,
        )
    else:
        # Backward compatible path for callers passing raw arrays.
        result = align_dept_peaks(
            peaks_a,
            peaks_b,
            ppm_tolerance=ppm_tolerance,
            match_multiplicity=match_multiplicity,
        )
    _scoring = {
        "cosine": cosine_from_alignment,
        "jaccard": jaccard_from_alignment,
        "recall": recall_from_alignment,
    }
    return {m: _scoring[m](result, min_matched_peaks) for m in metrics}


def _hungarian_score_fn(
    peaks_a: Any,
    peaks_b: Any,
    metrics: tuple[str, ...],
    ppm_tolerance: float = BASELINE_PARAMS["spectral_ppm_tol"],
    min_matched_peaks: int = BASELINE_PARAMS["spectral_min_matched_peaks"],
    match_multiplicity: bool = True,
    **_kwargs: Any,
) -> dict[str, float]:
    """Optimally assign peaks via the Hungarian algorithm and score all requested metrics.

    Drops in as a replacement for ``_pairwise_score_fn`` when you want the
    globally optimal 1-to-1 peak assignment instead of the greedy two-pointer
    sweep.  Accepts the same ``method_kwargs`` keys so the two methods are
    interchangeable in ``rank_library``.
    """
    # ``_encode_hungarian`` bakes intensity weights into a (peaks, weights)
    # tuple so scoring never needs to re-resolve intensities.  Raw-array callers
    # (binary, no weighting) are still supported via the else branch.
    if isinstance(peaks_a, tuple) and isinstance(peaks_b, tuple):
        arr_a, wa = peaks_a
        arr_b, wb = peaks_b
        result = align_dept_peaks_hungarian(
            arr_a,
            arr_b,
            ppm_tolerance=ppm_tolerance,
            match_multiplicity=match_multiplicity,
            weights_a=wa,
            weights_b=wb,
        )
    else:
        result = align_dept_peaks_hungarian(
            peaks_a,
            peaks_b,
            ppm_tolerance=ppm_tolerance,
            match_multiplicity=match_multiplicity,
        )
    _scoring = {
        "cosine": cosine_from_alignment,
        "jaccard": jaccard_from_alignment,
        "recall": recall_from_alignment,
    }
    return {m: _scoring[m](result, min_matched_peaks) for m in metrics}


def _encode_hungarian(
    peaks: np.ndarray,
    intensities: Any = None,
    intensity_mode: str = "binary",
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a spectrum for Hungarian matching as ``(peaks, weights)``.

    Weights are resolved once here so per-pair scoring never re-resolves
    intensities.  In binary mode the weights are all 1.0.
    """
    arr = np.asarray(peaks, dtype=np.float32)
    weights = resolve_intensity_weights(arr, intensities, intensity_mode)
    return arr, weights


METHOD_REGISTRY: dict[str, MethodSpec] = {
    "ppm_bins": MethodSpec(
        encode_fn=generate_ppm_bins,
        score_fn=_vector_score_fn,
        supported_metrics=frozenset({"cosine", "euclidean", "manhattan"}),
        distance_metrics=frozenset({"euclidean", "manhattan"}),
        batch_score_fn=_vector_batch_score_fn,
    ),
    "ppm_bins_typed": MethodSpec(
        encode_fn=generate_ppm_bins_typed,
        score_fn=_vector_score_fn,
        supported_metrics=frozenset({"cosine", "euclidean", "manhattan"}),
        distance_metrics=frozenset({"euclidean", "manhattan"}),
        batch_score_fn=_vector_batch_score_fn,
    ),
    "spectral_match": MethodSpec(
        encode_fn=preprocess_dept_peaks,
        score_fn=_pairwise_score_fn,
        supported_metrics=frozenset({"cosine", "jaccard", "recall"}),
        distance_metrics=frozenset(),
    ),
    "hungarian_match": MethodSpec(
        encode_fn=_encode_hungarian,
        score_fn=_hungarian_score_fn,
        supported_metrics=frozenset({"cosine", "jaccard", "recall"}),
        distance_metrics=frozenset(),
    ),
    "gauss_kernel": MethodSpec(
        encode_fn=generate_gaussian_vector,
        score_fn=_vector_score_fn,
        # Euclidean/Manhattan are not meaningful on signed continuous vectors;
        # cosine is the only well-defined scalar similarity here.
        supported_metrics=frozenset({"cosine"}),
        distance_metrics=frozenset(),
        batch_score_fn=_vector_batch_score_fn,
    ),
}
"""Registry of available similarity methods.

To add a new method, register a ``MethodSpec`` here.  The rest of the pipeline
— encoding, scoring loop, top-k selection — requires no changes.
"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def encode_spectra(
    method: str,
    compound_ids: list[str],
    spectra_df: pd.DataFrame,
    intensity_mode: str = "binary",
    intensity_col: str | None = None,
) -> dict[str, Any]:
    """Pre-encode spectra for a given method, returning a ``{compound_id: repr}`` dict.

    Pass the returned dict to ``rank_library`` via
    ``precomputed_query_reprs`` / ``precomputed_candidate_reprs`` to avoid
    re-encoding the same spectra when running multiple methods in sequence.

    Parameters
    ----------
    method:
        Key into ``METHOD_REGISTRY``.
    compound_ids:
        Compound IDs to encode.  Only these rows are read from ``spectra_df``.
    spectra_df:
        DataFrame with columns ``compound_id`` and ``signed_shifts``.
    intensity_mode:
        ``"binary"`` (default) or ``"normalized"``; must match the mode later
        passed to ``rank_library`` so cached reprs stay consistent.
    intensity_col:
        Column holding a per-peak intensity array; used only when
        ``intensity_mode="normalized"``.
    """
    spec = METHOD_REGISTRY.get(method)
    if spec is None:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(METHOD_REGISTRY)}")
    df = spectra_df.set_index("compound_id")
    use_col = intensity_col is not None and intensity_col in df.columns

    def _encode(cid: str) -> Any:
        intens = df.loc[cid, intensity_col] if use_col else None
        return spec.encode_fn(
            df.loc[cid, "signed_shifts"],
            intensities=intens,
            intensity_mode=intensity_mode,
        )

    return {cid: _encode(cid) for cid in compound_ids}


def rank_library(
    matched_pairs,
    query_spectra_df,
    library_spectra_df,
    method: str = "ppm_bins",
    sim_metrics=("cosine",),
    method_kwargs: dict | None = None,
    precomputed_query_reprs: dict | None = None,
    precomputed_candidate_reprs: dict | None = None,
    n_jobs: int | None = None,
    intensity_mode: str = "binary",
    intensity_col: str | None = None,
    top_k: int | None = None,
    include_structure: bool = True,
    smiles_col: str = "smiles",
    inchikey_col: str = "inchikey",
) -> pd.DataFrame:
    """Run pairwise similarity metrics for matched query-candidate pairs.

    Returns long-format rows with columns:
    ``query_id``, ``metric_name``, ``candidate_id_list`` and ``score_list``.
    When the library DataFrame carries structure columns and
    ``include_structure=True`` (default), two extra ranked columns are added:
    ``candidate_smiles_list`` and ``candidate_inchikey_list`` (full InChIKey).
    These let downstream evaluation compute Tanimoto similarity (needs SMILES)
    and hit@k on the InChIKey without a second join.

    For each query and each metric, rows correspond to that query's candidate list sorted by score
    (best score for similarity metrics, lowest score for distance metrics).

    Parameters
    ----------
    matched_pairs:
        Dict mapping query_id -> iterable of candidate_ids.
    query_spectra_df, library_spectra_df:
        DataFrames with columns ``compound_id`` and ``signed_shifts``.
        Ignored for any compound whose repr is supplied via
        ``precomputed_query_reprs`` / ``precomputed_candidate_reprs``.
    method:
        Key into ``METHOD_REGISTRY``.  Default ``"ppm_bins"``.
    sim_metrics:
        Metric names to compute.  Must be a subset of the chosen method's
        ``supported_metrics``.
    method_kwargs:
        Extra keyword arguments forwarded to the method's ``score_fn``.
        For ``"spectral_match"`` / ``"hungarian_match"`` the supported keys are:

        - ``match_multiplicity`` (bool, default True)
        - ``ppm_tolerance`` (float)
        - ``min_matched_peaks`` (int)

        Ignored by vector-based methods (``"ppm_bins"``, ``"gauss_kernel"``).
    precomputed_query_reprs:
        Optional ``{query_id: repr}`` dict from a previous ``encode_spectra``
        call.  Any query ID present here skips re-encoding.
    precomputed_candidate_reprs:
        Optional ``{candidate_id: repr}`` dict.  Any candidate ID present here
        skips re-encoding.
    n_jobs:
        Number of threads for pairwise methods (``spectral_match``,
        ``hungarian_match``) that have no vectorised batch path.
        ``None`` defaults to ``1``; set ``>1`` to enable threading.
        Ignored for vector-based methods which use numpy batching instead.
    intensity_mode:
        ``"binary"`` (default): every peak weighs 1.0 (peak-presence similarity,
        bit-for-bit identical to the previous behaviour).  ``"normalized"``:
        peaks are weighted by their max-normalised intensity (read from
        ``intensity_col``).
    intensity_col:
        Column in the spectra DataFrames holding a per-peak intensity array
        aligned 1-to-1 with ``signed_shifts``.  Required when
        ``intensity_mode="normalized"``; ignored in binary mode.
    top_k:
        If given, truncate every ranked list (candidate ids, scores, smiles,
        inchikeys) to the top ``top_k`` candidates.  ``None`` (default) keeps
        the full ranked list.
    include_structure:
        When True (default), attach ``candidate_smiles_list`` and
        ``candidate_inchikey_list`` columns whenever the library DataFrame
        contains ``smiles_col`` / ``inchikey_col``.
    smiles_col, inchikey_col:
        Names of the SMILES and full-InChIKey columns in the library
        DataFrame.  Defaults ``"smiles"`` and ``"inchikey"``.
    """
    if intensity_mode not in ("binary", "normalized"):
        raise ValueError(
            f"Unknown intensity_mode '{intensity_mode}'. Choose 'binary' or 'normalized'."
        )
    # --- validate method and metrics ---
    spec = METHOD_REGISTRY.get(method)
    if spec is None:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(METHOD_REGISTRY)}")

    sim_metrics = tuple(sim_metrics)
    unsupported = set(sim_metrics) - spec.supported_metrics
    if unsupported:
        raise ValueError(
            f"Method '{method}' does not support metric(s): {sorted(unsupported)}. "
            f"Supported: {sorted(spec.supported_metrics)}"
        )

    metric_is_distance = {m: m in spec.distance_metrics for m in sim_metrics}
    method_kwargs = method_kwargs or {}

    columns = ["query_id", "metric_name", "candidate_id_list", "score_list"]

    # --- encode spectra (skip any IDs already in precomputed dicts) ---
    query_spectra_df = query_spectra_df.set_index("compound_id")
    library_spectra_df = library_spectra_df.set_index("compound_id")

    query_reprs = precomputed_query_reprs or {}
    candidate_reprs = precomputed_candidate_reprs or {}

    def _encode(df: pd.DataFrame, cid: str) -> Any:
        peaks = df.loc[cid, "signed_shifts"]
        intens = (
            df.loc[cid, intensity_col]
            if (intensity_col is not None and intensity_col in df.columns)
            else None
        )
        return spec.encode_fn(peaks, intensities=intens, intensity_mode=intensity_mode)

    missing_query_ids = [qid for qid in matched_pairs if qid not in query_reprs]
    for qid in missing_query_ids:
        query_reprs[qid] = _encode(query_spectra_df, qid)

    all_candidate_ids = {cid for cids in matched_pairs.values() for cid in cids}
    missing_cand_ids = [cid for cid in all_candidate_ids if cid not in candidate_reprs]
    for cid in missing_cand_ids:
        candidate_reprs[cid] = _encode(library_spectra_df, cid)

    # Structure lookups for ranked-candidate export (candidates are library rows)
    cand_smiles = (
        library_spectra_df[smiles_col].to_dict()
        if (include_structure and smiles_col in library_spectra_df.columns)
        else None
    )
    cand_inchikey = (
        library_spectra_df[inchikey_col].to_dict()
        if (include_structure and inchikey_col in library_spectra_df.columns)
        else None
    )

    use_batch = spec.batch_score_fn is not None
    # Pairwise spectral matching is sub-10us per pair once encoded, so Python
    # thread scheduling usually dominates unless explicitly tuned by caller.
    _n_jobs = 1 if use_batch else (n_jobs if n_jobs is not None else 1)

    # --- score all pairs ---
    def _score_query(
        query_id: str,
        candidate_ids: list[str],
        pool: ThreadPoolExecutor | None = None,
    ) -> list[tuple[str, str, list[str], list[float]]]:
        """Score all candidates for one query; one aggregated row per metric."""
        repr_a = query_reprs[query_id]
        n_candidates = len(candidate_ids)
        n_metrics = len(sim_metrics)
        metric_matrix = np.empty((n_candidates, n_metrics), dtype=np.float32)

        if use_batch:
            cand_matrix = np.stack([candidate_reprs[cid] for cid in candidate_ids])
            metric_matrix[:] = spec.batch_score_fn(repr_a, cand_matrix, sim_metrics)
        else:

            def _score_one(args):
                idx, cid = args
                return idx, spec.score_fn(
                    repr_a, candidate_reprs[cid], sim_metrics, **method_kwargs
                )

            if _n_jobs == 1:
                for idx, cid in enumerate(candidate_ids):
                    _, scores = _score_one((idx, cid))
                    for m_idx, m in enumerate(sim_metrics):
                        metric_matrix[idx, m_idx] = scores[m]
            else:
                if pool is None:
                    raise RuntimeError("Thread pool must be provided when _n_jobs > 1")
                for idx, scores in pool.map(_score_one, enumerate(candidate_ids)):
                    for m_idx, m in enumerate(sim_metrics):
                        metric_matrix[idx, m_idx] = scores[m]

        local_rows: list[dict[str, Any]] = []
        for metric_idx, metric_name in enumerate(sim_metrics):
            scores_col = metric_matrix[:, metric_idx]

            # Sort the list of scores, keeping track of original indices for tie summary
            if metric_is_distance[metric_name]:
                metric_order = np.argsort(scores_col)  # Ascending for distance metrics
            else:
                metric_order = np.argsort(-scores_col)  # Descending for similarity metrics

            if top_k is not None:
                metric_order = metric_order[:top_k]

            sorted_scores = scores_col[metric_order]
            sorted_candidate_ids = [candidate_ids[i] for i in metric_order]

            # Save one row per query+metric with ranked candidates and scores.
            row: dict[str, Any] = {
                "query_id": query_id,
                "metric_name": metric_name,
                "candidate_id_list": sorted_candidate_ids,
                "score_list": sorted_scores.astype(float).tolist(),
            }
            if cand_smiles is not None:
                row["candidate_smiles_list"] = [
                    cand_smiles.get(cid) for cid in sorted_candidate_ids
                ]
            if cand_inchikey is not None:
                row["candidate_inchikey_list"] = [
                    cand_inchikey.get(cid) for cid in sorted_candidate_ids
                ]
            local_rows.append(row)

        return local_rows

    if cand_smiles is not None:
        columns.append("candidate_smiles_list")
    if cand_inchikey is not None:
        columns.append("candidate_inchikey_list")

    pool: ThreadPoolExecutor | None = None
    all_rows: list[dict[str, Any]] = []
    try:
        if _n_jobs > 1:
            pool = ThreadPoolExecutor(max_workers=_n_jobs)

        for query_id, candidate_ids in tqdm(matched_pairs.items(), desc="Calculating similarities"):
            candidate_ids = list(candidate_ids)
            if not candidate_ids:
                continue
            local_rows = _score_query(query_id, candidate_ids, pool=pool)
            all_rows.extend(local_rows)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if not all_rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(all_rows)[columns]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def get_metrics(
    results_df: pd.DataFrame,
) -> tuple[int, int, float]:
    """Compute hit@k accuracy from a results DataFrame produced by ``rank_library``.

    A query counts as a true hit if at least one candidate in its ranked list
    shares the same InChIKey as the query.  The ranked list is taken as-is —
    apply top-k filtering upstream via the ``top_k`` parameter of
    ``rank_library`` before calling this function.

    Parameters
    ----------
    results_df:
        Output of ``rank_library`` with columns
        ``query_id``, ``candidate_id``, ``metric_name``, ``score``.
    query_spectra_df, library_spectra_df:
        DataFrames with columns ``compound_id`` and ``inchikey``/``inchikey14``.
    with_inchikey14:
        Use the first 14 characters of the InChIKey for matching (ignores
        stereochemistry).  Default True — appropriate for DEPT NMR, which
        cannot distinguish stereoisomers (identical carbon multiplicities).
        Set False to require an exact full-InChIKey match.
    n_total_queries:
        Total number of queries being evaluated.  When provided this is used
        as the denominator so that queries with no candidates (filtered out
        upstream) are counted as misses rather than excluded.  When None,
        falls back to ``results_df["query_id"].nunique()``.

    Returns
    -------
    tuple[int, int, float]
        ``(true_hits, total_queries, accuracy_pct)``
    """
    hits = results_df.copy()

    # Add inchikey14 columns to spectra DataFrames for matching
    hits["query_inchikey14"] = hits["query_id"].apply(lambda qid: qid.split("_")[-1])
    hits["candidate_inchikey14"] = hits["candidate_id"].apply(lambda cid: cid.split("_")[-1])

    hits["_hit"] = hits["query_inchikey14"] == hits["candidate_inchikey14"]
    true_hits = int(hits.groupby("query_id")["_hit"].any().sum())
    total_queries = len(hits.groupby("query_id"))

    return true_hits, total_queries, (true_hits / total_queries) * 100.0


def get_metric(
    results_df: pd.DataFrame,
    query_spectra_df: pd.DataFrame,
    library_spectra_df: pd.DataFrame,
    with_inchikey14: bool = True,
    n_total_queries: int | None = None,
) -> tuple[int, int, float]:
    """Compute hit@k accuracy from a results DataFrame produced by ``rank_library``.

    A query counts as a true hit if at least one candidate in its ranked list
    shares the same InChIKey as the query.  The ranked list is taken as-is —
    apply top-k filtering upstream via the ``top_k`` parameter of
    ``rank_library`` before calling this function.

    Parameters
    ----------
    results_df:
        Output of ``rank_library`` with columns
        ``query_id``, ``candidate_id``, ``metric_name``, ``score``.
    query_spectra_df, library_spectra_df:
        DataFrames with columns ``compound_id`` and ``inchikey``/``inchikey14``.
    with_inchikey14:
        Use the first 14 characters of the InChIKey for matching (ignores
        stereochemistry).  Default True — appropriate for DEPT NMR, which
        cannot distinguish stereoisomers (identical carbon multiplicities).
        Set False to require an exact full-InChIKey match.
    n_total_queries:
        Total number of queries being evaluated.  When provided this is used
        as the denominator so that queries with no candidates (filtered out
        upstream) are counted as misses rather than excluded.  When None,
        falls back to ``results_df["query_id"].nunique()``.

    Returns
    -------
    tuple[int, int, float]
        ``(true_hits, total_queries, accuracy_pct)``
    """
    query_spectra_df = query_spectra_df.set_index("compound_id")
    library_spectra_df = library_spectra_df.set_index("compound_id")

    inchikey_col = "inchikey14" if with_inchikey14 else "inchikey"

    n_in_results = results_df["query_id"].nunique()
    total_queries = n_total_queries if n_total_queries is not None else n_in_results
    if total_queries == 0:
        return 0, 0, 0.0

    # Vectorised: join query true keys and candidate keys, then compare
    q_keys = query_spectra_df[[inchikey_col]].rename(columns={inchikey_col: "_true_key"})
    c_keys = library_spectra_df[[inchikey_col]].rename(columns={inchikey_col: "_cand_key"})

    df = (
        results_df[["query_id", "candidate_id"]]
        .merge(q_keys, left_on="query_id", right_index=True, how="left")
        .merge(c_keys, left_on="candidate_id", right_index=True, how="left")
    )
    df["_hit"] = df["_true_key"] == df["_cand_key"]
    true_hits = int(df.groupby("query_id")["_hit"].any().sum())

    return true_hits, total_queries, (true_hits / total_queries) * 100.0


# ---------------------------------------------------------------------------
# Tanimoto helper
# ---------------------------------------------------------------------------


def _make_fingerprint_cache(fp_radius: int = 2, fp_bits: int = 2048):
    """Return a ``smiles -> Morgan fingerprint`` memoised lookup and a Tanimoto fn.

    RDKit is imported lazily so that importing this module never requires it;
    it is only needed when ``get_metrics_at_k(..., compute_tanimoto=True)``.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Computing Tanimoto similarity requires RDKit. Install it "
            "(`pip install rdkit`) or call get_metrics_at_k(compute_tanimoto=False)."
        ) from exc

    cache: dict[str, Any] = {}

    def fp_of(smiles: str | None):
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
            return None
        if smiles in cache:
            return cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        fp = (
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_bits)
            if mol is not None
            else None
        )
        cache[smiles] = fp
        return fp

    def tanimoto(smiles_a: str | None, smiles_b: str | None) -> float:
        fa, fb = fp_of(smiles_a), fp_of(smiles_b)
        if fa is None or fb is None:
            return float("nan")
        return float(DataStructs.TanimotoSimilarity(fa, fb))

    return fp_of, tanimoto


# ---------------------------------------------------------------------------
# hits@k + Tanimoto@k evaluation
# ---------------------------------------------------------------------------


def get_metrics_at_k(
    results_df: pd.DataFrame,
    query_spectra_df: pd.DataFrame,
    library_spectra_df: pd.DataFrame | None = None,
    k_values: tuple[int, ...] = (1, 3, 10),
    with_inchikey14: bool = True,
    compute_tanimoto: bool = True,
    fp_radius: int = 2,
    fp_bits: int = 2048,
    n_total_queries: int | None = None,
    smiles_col: str = "smiles",
    inchikey_col: str = "inchikey",
    inchikey14_col: str = "inchikey14",
) -> pd.DataFrame:
    """Compute hits@k and mean Tanimoto@k from ranked-list similarity results.

    Consumes the wide output of ``rank_library`` (one row per
    ``query_id`` × ``metric_name`` with ranked ``candidate_id_list`` and,
    optionally, ``candidate_smiles_list`` / ``candidate_inchikey_list``).

    For each query and each ``k``:

    * **hit@k** — 1 if any of the top-``k`` candidates shares the query's
      InChIKey (14-char prefix when ``with_inchikey14=True``, else full).
    * **Tanimoto@k** — the mean Morgan-fingerprint Tanimoto similarity between
      the query and each of its top-``k`` candidates.  Candidates whose SMILES
      is missing or unparseable are skipped; a query with no usable candidate
      SMILES contributes NaN and is excluded from the aggregate mean.

    Candidate SMILES/InChIKey are read from the ``candidate_smiles_list`` /
    ``candidate_inchikey_list`` columns when present (recommended — this is why
    ``rank_library`` exports them, so full-InChIKey stereochemistry
    is preserved for Tanimoto).  Otherwise they are looked up from
    ``library_spectra_df`` by ``candidate_id``.

    Parameters
    ----------
    results_df:
        Output of ``rank_library``.
    query_spectra_df:
        DataFrame with ``compound_id`` plus SMILES / InChIKey columns, used to
        resolve each query's structure and true key.
    library_spectra_df:
        Fallback structure source for candidates when the ranked SMILES/InChIKey
        columns are absent from ``results_df``.
    k_values:
        The ``k`` cut-offs to evaluate.  Default ``(1, 3, 10)``.
    with_inchikey14:
        Match on the 14-char InChIKey prefix (ignores stereochemistry).
    compute_tanimoto:
        Compute Tanimoto@k (requires RDKit).  Set False to get hits@k only.
    fp_radius, fp_bits:
        Morgan fingerprint parameters (default radius 2, 2048 bits — matching
        notebook 05).
    n_total_queries:
        Denominator override so filtered-out queries count as misses.  When
        None, uses the number of distinct queries present in ``results_df``.
    smiles_col, inchikey_col, inchikey14_col:
        Column names in the spectra DataFrames.

    Returns
    -------
    pd.DataFrame
        One row per ``(metric_name, k)`` with columns ``metric_name``, ``k``,
        ``true_positives``, ``total_queries``, ``accuracy`` (percent) and, when
        ``compute_tanimoto=True``, ``mean_tanimoto``.
    """
    if results_df.empty:
        cols = ["metric_name", "k", "true_positives", "total_queries", "accuracy"]
        if compute_tanimoto:
            cols.append("mean_tanimoto")
        return pd.DataFrame(columns=cols)

    q_idx = query_spectra_df.set_index("compound_id")

    def _key14(full: str | None) -> str | None:
        return full[:14] if isinstance(full, str) else None

    # Query true key + SMILES lookups
    if with_inchikey14 and inchikey14_col in q_idx.columns:
        q_true_key = q_idx[inchikey14_col].to_dict()
    else:
        q_full = q_idx[inchikey_col].to_dict()
        q_true_key = {k: (_key14(v) if with_inchikey14 else v) for k, v in q_full.items()}
    q_smiles = q_idx[smiles_col].to_dict() if smiles_col in q_idx.columns else {}

    # Candidate lookups (only needed when results_df lacks the ranked columns)
    has_cand_ik = "candidate_inchikey_list" in results_df.columns
    has_cand_smiles = "candidate_smiles_list" in results_df.columns
    lib_idx = (
        library_spectra_df.set_index("compound_id") if library_spectra_df is not None else None
    )
    lib_ik = (
        lib_idx[inchikey_col].to_dict()
        if (lib_idx is not None and inchikey_col in lib_idx.columns)
        else {}
    )
    lib_smiles = (
        lib_idx[smiles_col].to_dict()
        if (lib_idx is not None and smiles_col in lib_idx.columns)
        else {}
    )

    _, tanimoto = _make_fingerprint_cache(fp_radius, fp_bits) if compute_tanimoto else (None, None)

    total_queries = (
        n_total_queries if n_total_queries is not None else results_df["query_id"].nunique()
    )

    # Accumulators keyed by (metric_name, k)
    hit_counts: dict[tuple[str, int], int] = {}
    tan_sums: dict[tuple[str, int], float] = {}
    tan_counts: dict[tuple[str, int], int] = {}
    metric_names: list[str] = []

    for _, row in results_df.iterrows():
        metric_name = row["metric_name"]
        if metric_name not in metric_names:
            metric_names.append(metric_name)
        qid = row["query_id"]
        cand_ids = list(row["candidate_id_list"])

        # Resolve ranked candidate keys and SMILES for this query.
        if has_cand_ik:
            cand_full_ik = list(row["candidate_inchikey_list"])
        else:
            cand_full_ik = [lib_ik.get(cid) for cid in cand_ids]
        cand_keys = [(_key14(ik) if with_inchikey14 else ik) for ik in cand_full_ik]

        if compute_tanimoto:
            if has_cand_smiles:
                cand_smiles = list(row["candidate_smiles_list"])
            else:
                cand_smiles = [lib_smiles.get(cid) for cid in cand_ids]

        true_key = q_true_key.get(qid)
        query_smiles = q_smiles.get(qid)

        for k in k_values:
            key = (metric_name, k)
            top_keys = cand_keys[:k]
            if true_key is not None and true_key in top_keys:
                hit_counts[key] = hit_counts.get(key, 0) + 1

            if compute_tanimoto:
                sims = [
                    tanimoto(query_smiles, cs)
                    for cs in cand_smiles[:k]
                ]
                sims = [s for s in sims if not np.isnan(s)]
                if sims:
                    tan_sums[key] = tan_sums.get(key, 0.0) + float(np.mean(sims))
                    tan_counts[key] = tan_counts.get(key, 0) + 1

    rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        for k in k_values:
            key = (metric_name, k)
            tp = hit_counts.get(key, 0)
            rec: dict[str, Any] = {
                "metric_name": metric_name,
                "k": k,
                "true_positives": tp,
                "total_queries": total_queries,
                "accuracy": (tp / total_queries * 100.0) if total_queries else 0.0,
            }
            if compute_tanimoto:
                n = tan_counts.get(key, 0)
                rec["mean_tanimoto"] = (tan_sums.get(key, 0.0) / n) if n else float("nan")
            rows.append(rec)

    return pd.DataFrame(rows)
