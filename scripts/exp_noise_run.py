"""Experiment 1 — query-side chemical-shift noise robustness.

Perturbs the experimental query spectra with synthetic chemical-shift noise and measures
how Hits@k degrades for each retrieval method. The candidate pool is fixed from the clean
queries so degradation isolates ranking robustness.

Noise modes:
  jitter  — each peak shifted independently ~ N(0, sigma)   (peak-picking / digital resolution)
  offset  — whole spectrum shifted by one draw ~ N(0, sigma) (referencing / calibration error)

Writes data/results/noise_robustness/summary.parquet with one row per
(method, mode, sigma, seed, k).
"""
import time
from pathlib import Path
import numpy as np, pandas as pd

from dept_similarity.constants import RESULTS_DIR
from dept_similarity.match import get_metrics_at_k
import dept_similarity.experiments as R

SIGMAS = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0]
MODES = ["jitter", "offset"]
SEEDS = list(range(5))
K_VALUES = (1, 3, 10)
RUN_HUNGARIAN = True  # DEPT-Match re-aligns 64M pairs per config (~1 min each on 96 cores)

d = R.build_experiment_inputs()
pairs, qp, lp = d["pairs"], d["query_peaks"], d["lib_peaks"]
lib_smiles, lib_inchikey = d["lib_smiles"], d["lib_inchikey"]
N_TOTAL = d["n_total_queries"]
q = d["query_df"]
query_ids = list(pairs.keys())

OPT = Path(RESULTS_DIR) / "optimization"
GAUSS_SIGMA = float(pd.read_parquet(OPT/"gaussian"/"summary_hits_at_k.parquet").sort_values("Hits@10 (%)").iloc[-1]["sigma"])
HUNG_TOL = float(pd.read_parquet(OPT/"hungarian"/"summary_hits_at_k.parquet").sort_values("Hits@10 cosine (%)").iloc[-1]["ppm_tolerance"])

cand_order = sorted({c for v in pairs.values() for c in v})
cand_index = {c: i for i, c in enumerate(cand_order)}


def hits(df):
    r = get_metrics_at_k(df, query_spectra_df=q, k_values=K_VALUES,
                         with_inchikey14=True, compute_tanimoto=False, n_total_queries=N_TOTAL)
    return {int(k): float(a) for k, a in zip(r["k"], r["accuracy"])}


def perturbed_lookup(sigma, mode, seed):
    rng = np.random.default_rng(seed)
    return {qid: R.perturb_peaks(qp[qid], sigma, rng, mode) for qid in query_ids}


rows = []


def record(method, mode, sigma, seed, h):
    for k, acc in h.items():
        rows.append({"method": method, "mode": mode, "sigma": sigma, "seed": seed, "k": k, "accuracy": acc})


# ---- vector methods: encode library once, reuse across all configs ----
for method in ["gauss_kernel", "ppm_bins_typed"]:
    t = time.time()
    lib_norm = R.encode_library_vectors(cand_order, lp, method, GAUSS_SIGMA)
    print(f"[{method}] library encoded in {time.time()-t:.0f}s", flush=True)

    # baseline (sigma=0): identical for every mode/seed
    base = hits(R.rank_vector_method(query_ids, qp, pairs, cand_index, lib_norm, method, GAUSS_SIGMA, lib_smiles, lib_inchikey))
    record(method, "baseline", 0.0, -1, base)
    print(f"[{method}] baseline {base}", flush=True)

    for mode in MODES:
        for sigma in SIGMAS:
            for seed in SEEDS:
                ql = perturbed_lookup(sigma, mode, seed)
                h = hits(R.rank_vector_method(query_ids, ql, pairs, cand_index, lib_norm, method, GAUSS_SIGMA, lib_smiles, lib_inchikey))
                record(method, mode, sigma, seed, h)
            sub = [r for r in rows if r["method"] == method and r["mode"] == mode and r["sigma"] == sigma and r["k"] == 10]
            print(f"[{method}] {mode} sigma={sigma} H@10 mean={np.mean([r['accuracy'] for r in sub]):.1f}", flush=True)
    del lib_norm

# ---- DEPT-Match (Hungarian): full re-alignment per config (opt-in; slow) ----
if RUN_HUNGARIAN:
    method = "hungarian_match"
    R.set_hungarian_globals(lp, lib_smiles, lib_inchikey)
    base = hits(R.rank_hungarian(query_ids, qp, pairs, HUNG_TOL)["cosine"])
    record(method, "baseline", 0.0, -1, base)
    print(f"[{method}] baseline {base}", flush=True)
    for mode in MODES:
        for sigma in SIGMAS:
            for seed in SEEDS:
                t = time.time()
                ql = perturbed_lookup(sigma, mode, seed)
                h = hits(R.rank_hungarian(query_ids, ql, pairs, HUNG_TOL)["cosine"])
                record(method, mode, sigma, seed, h)
                print(f"[{method}] {mode} sigma={sigma} seed={seed} H@10={h[10]:.1f} ({time.time()-t:.0f}s)", flush=True)

out_dir = Path(RESULTS_DIR) / "noise_robustness"
out_dir.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame(rows)
summary.to_parquet(out_dir / "summary.parquet", index=False)
print("SAVED", out_dir / "summary.parquet", summary.shape, flush=True)
