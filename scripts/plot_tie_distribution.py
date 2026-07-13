"""Fig — distribution of the number of ties at the top retrieval score across methods.

For every query and every method we compute the *full* candidate score vector (not just the
top-10 stored in the ranked lists) and count how many candidates share the maximum score —
the "ties at top score". The per-method distribution of that count shows how often a method
leaves the identity of the #1 hit ambiguous.

Reuses the exact scoring paths from notebook 04 / ``experiments.py`` (Gaussian and
shift-binned cosine as a single library matmul; DEPT-Match via Hungarian alignment).
DEPT-Match re-aligns every prefiltered pair, so this is opt-in slow (minutes on a laptop).

Before plotting, the recomputed top-10 rankings are checked against the committed
``ranked_top10.parquet`` files to confirm the on-disk data still matches the cached results.

Writes ``data/figures/tie_distribution_comparison.png``.
"""
import os

# The Hungarian workers rely on fork inheritance of the shared library-peaks dict (see
# ``experiments.py``); macOS defaults to "spawn", which loses that global. Force fork and
# pin BLAS/OMP to one thread so forking after the vector-method GEMM can't deadlock.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import multiprocessing as _mp
try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass

import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dept_similarity.bin_vector import generate_ppm_bins_typed
from dept_similarity.constants import (
    FIGURE_DIR,
    METHOD_NAME_MAPPER,
    PALETTE_METHODS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from dept_similarity.gaussian_vector import generate_gaussian_vector
from dept_similarity.prepare import prefilter_pairs_by_peak_overlap
from dept_similarity.spectral_matching import (
    align_dept_peaks_hungarian,
    cosine_from_alignment,
    jaccard_from_alignment,
)

MAX_TIES = 20          # x-axis extent (ties >= this fold into the last bin)
TIE_ATOL = 1e-9        # two scores counted as tied if within this absolute tolerance
CACHE = Path("../data/cache/tie_inputs.pkl")
SCORES_CACHE = Path("../data/cache/tie_scores.pkl")  # full per-query score vectors per method

# Legend order and colors reproduce the original figure (green / amber / blue / red).
# DEPT-Match (Jaccard) tracks DEPT-Match (Cosine) almost exactly, so it is drawn as a dashed
# line on top (high zorder) to stay legible where the two curves overlap.
#           method            metric     color               linestyle marker zorder
METHOD_ORDER = [
    ("hungarian_match", "jaccard", PALETTE_METHODS[0], "--", "s", 6),  # DEPT-Match (Jaccard) — green
    ("gauss_kernel", "cosine", PALETTE_METHODS[1], "-", "o", 3),       # Gaussian (Cosine)    — amber
    ("hungarian_match", "cosine", PALETTE_METHODS[2], "-", "o", 4),    # DEPT-Match (Cosine)  — blue
    ("ppm_bins_typed", "cosine", PALETTE_METHODS[3], "-", "o", 3),     # Shift binned (Cosine)— red
]


def _arr(v):
    return v.astype(np.float32, copy=False) if isinstance(v, np.ndarray) else np.asarray(v, np.float32)


def build_inputs():
    """Load library + query spectra and the clean-query candidate pool.

    The on-disk library uses ``intensity_signed`` / ``inchikey_14``; the pipeline expects
    ``signed_shifts`` / ``inchikey14`` (as in ``experiments.build_experiment_inputs``), so we
    rename on load. Cached to disk since the prefilter takes ~30 s.
    """
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)

    lib = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "library_data.parquet").rename(
        columns={"intensity_signed": "signed_shifts", "inchikey_14": "inchikey14"})
    q = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "experimental_validation_set.parquet")
    lib["signed_shifts"] = lib["signed_shifts"].map(_arr)
    q["signed_shifts"] = q["signed_shifts"].map(_arr)
    if not (q["compound_id"].str.split("_").str[-1] == q["inchikey14"]).all():
        q["compound_id"] = q["compound_id"] + "_" + q["inchikey14"]

    pairs = {k: sorted(v) for k, v in prefilter_pairs_by_peak_overlap(q, lib).items()}
    out = {
        "pairs": pairs,
        "query_peaks": q.set_index("compound_id")["signed_shifts"].to_dict(),
        "lib_peaks": lib.set_index("compound_id")["signed_shifts"].to_dict(),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def _row_normalise(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _count_ties(scores):
    """Number of candidates sharing the maximum score."""
    if len(scores) == 0:
        return 0
    return int(np.sum(np.isclose(scores, scores.max(), rtol=0, atol=TIE_ATOL)))


# ---- vector methods: full score vector per query via one batched GEMM ----
def vector_full_scores(query_ids, query_peaks, pairs, lib_peaks, method, gauss_sigma):
    cand_order = sorted({c for v in pairs.values() for c in v})
    cand_index = {c: i for i, c in enumerate(cand_order)}
    enc = ((lambda p: generate_gaussian_vector(p, sigma=gauss_sigma))
           if method == "gauss_kernel" else generate_ppm_bins_typed)

    lib_norm = _row_normalise(np.stack(Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(generate_gaussian_vector)(lib_peaks[c], sigma=gauss_sigma) if method == "gauss_kernel"
        else delayed(generate_ppm_bins_typed)(lib_peaks[c]) for c in cand_order)))
    q_mat = _row_normalise(np.stack([enc(query_peaks[q]) for q in query_ids]))
    scores_full = lib_norm @ q_mat.T  # (n_cand, n_query)

    per_query = {}
    for j, q in enumerate(query_ids):
        cand = pairs[q]
        idx = np.fromiter((cand_index[c] for c in cand), dtype=np.int64, count=len(cand))
        per_query[q] = scores_full[idx, j]
    return per_query


# ---- Hungarian: full re-alignment per pair (shared lib via fork) ----
_LIB = None


def _set_lib(lib_peaks):
    global _LIB
    _LIB = lib_peaks


def _hungarian_query(q, qpeaks, cand, ppm_tol):
    cos = np.empty(len(cand)); jac = np.empty(len(cand))
    for i, c in enumerate(cand):
        res = align_dept_peaks_hungarian(qpeaks, _LIB[c], ppm_tolerance=ppm_tol, match_multiplicity=True)
        cos[i] = cosine_from_alignment(res, 1)
        jac[i] = jaccard_from_alignment(res, 1)
    return q, cos, jac


def hungarian_full_scores(query_ids, query_peaks, pairs, lib_peaks, ppm_tol):
    _set_lib(lib_peaks)
    results = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(_hungarian_query)(q, query_peaks[q], pairs[q], ppm_tol) for q in query_ids)
    cos = {q: c for q, c, _ in results}
    jac = {q: j for q, _, j in results}
    return {"cosine": cos, "jaccard": jac}


def _check_topk(per_query_scores, pairs, method, metric):
    """Sanity-check recomputed rankings against the committed ranked_top10.parquet."""
    path = Path(RESULTS_DIR) / f"{method}_{metric}" / "ranked_top10.parquet"
    old = pd.read_parquet(path).set_index("query_id")
    ok = tot = 0
    for q, scores in per_query_scores.items():
        if q not in old.index:
            continue
        cand = pairs[q]
        order = np.argsort(-scores, kind="stable")[:10]
        sel = [cand[i] for i in order]
        tot += 1
        ok += (sel == list(old.loc[q, "candidate_id_list"]))
    frac = ok / tot if tot else float("nan")
    print(f"  check {method}_{metric}: {ok}/{tot} top-10 identical ({frac:.1%})", flush=True)
    return frac


def main():
    t0 = time.time()
    d = build_inputs()
    pairs, qp, lp = d["pairs"], d["query_peaks"], d["lib_peaks"]
    query_ids = list(pairs.keys())
    print(f"inputs: {len(query_ids)} queries, "
          f"{sum(len(v) for v in pairs.values()):,} pairs ({time.time() - t0:.0f}s)", flush=True)

    OPT = Path(RESULTS_DIR) / "optimization"
    gauss_sigma = float(pd.read_parquet(OPT / "gaussian" / "summary_hits_at_k.parquet")
                        .sort_values("Hits@10 (%)").iloc[-1]["sigma"])
    hung_tol = float(pd.read_parquet(OPT / "hungarian" / "summary_hits_at_k.parquet")
                     .sort_values("Hits@10 cosine (%)").iloc[-1]["ppm_tolerance"])
    print(f"gauss_sigma={gauss_sigma}  hung_tol={hung_tol}", flush=True)

    # Full score vectors per method (cached so tie-tolerance can be tuned without recompute).
    if SCORES_CACHE.exists():
        with open(SCORES_CACHE, "rb") as fh:
            scores = pickle.load(fh)
        print(f"loaded cached scores ({time.time() - t0:.0f}s)", flush=True)
    else:
        scores = {}
        t = time.time()
        scores["gauss_kernel", "cosine"] = vector_full_scores(query_ids, qp, pairs, lp, "gauss_kernel", gauss_sigma)
        print(f"gauss_kernel done ({time.time() - t:.0f}s)", flush=True)
        t = time.time()
        scores["ppm_bins_typed", "cosine"] = vector_full_scores(query_ids, qp, pairs, lp, "ppm_bins_typed", gauss_sigma)
        print(f"ppm_bins_typed done ({time.time() - t:.0f}s)", flush=True)
        t = time.time()
        hung = hungarian_full_scores(query_ids, qp, pairs, lp, hung_tol)
        scores["hungarian_match", "cosine"] = hung["cosine"]
        scores["hungarian_match", "jaccard"] = hung["jaccard"]
        print(f"hungarian done ({time.time() - t:.0f}s)", flush=True)
        with open(SCORES_CACHE, "wb") as fh:
            pickle.dump(scores, fh)

    # Consistency check vs committed ranked lists.
    print("consistency check vs committed ranked_top10:", flush=True)
    for (method, metric), pq in scores.items():
        _check_topk(pq, pairs, method, metric)

    # Tie-count distributions (probability mass over 1..MAX_TIES, tail folded into MAX_TIES).
    x = np.arange(1, MAX_TIES + 1)
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for method, metric, color, ls, marker, zorder in METHOD_ORDER:
        counts = np.array([_count_ties(scores[method, metric][q]) for q in query_ids])
        counts = np.clip(counts, 1, MAX_TIES)
        density = np.array([(counts == n).sum() for n in x], dtype=float) / len(counts)
        label = METHOD_NAME_MAPPER.get(f"{method}_{metric}", f"{method} ({metric})")
        ax.plot(x, density, ls=ls, marker=marker, color=color, lw=2, ms=6, label=label, zorder=zorder)

    ax.set_title("Distribution of the number of ties\namong the top hit(s) across methods", fontsize=18)
    ax.set_xlabel("Number of ties at top score", fontsize=17)
    ax.set_ylabel("Density", fontsize=17)
    ax.set_xlim(0.5, MAX_TIES)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(FIGURE_DIR) / "tie_distribution_comparison.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    print(f"saved {out}  (total {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
