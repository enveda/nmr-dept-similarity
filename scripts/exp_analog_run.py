"""Analog-retrieval, phase 2: rank out-of-library queries and score analog recovery.

Uses the phase-1 nearest-analog table (data/results/analog/query_tmax.parquet), keeps the
0.70<=Tmax<0.85 operationally-meaningful band, builds the same peak-overlap candidate pool and
ranks with the same methods/parameters as the exact-match benchmark. Because the query is not
in the library, a "hit" is analog recovery, scored three complementary ways:

  nearest_analog_hit@k : the single argmax-Tanimoto library compound appears in the top-k
  near_optimal@k       : some top-k hit reaches Tanimoto >= Tmax - 0.05 (an ~equally good analog)
  analog@k             : some top-k hit reaches Tanimoto >= 0.6 (a genuine structural analog)
  mean_best_tanimoto@k : mean over queries of the best query->top-k Tanimoto (vs Tmax ceiling)

Writes data/results/analog/{metrics.parquet, per_query.parquet}.
"""
import os
# macOS defaults to the 'spawn' start method, which re-imports this (unguarded)
# module in every joblib worker and crashes. Force 'fork' + disable the ObjC
# fork-safety check so the backend="multiprocessing" pools work at module scope.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
import multiprocessing as mp
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

import time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from dept_similarity.constants import PROCESSED_DATA_DIR, RESULTS_DIR
from dept_similarity.prepare import prefilter_pairs_by_peak_overlap
from dept_similarity.utils import passed_cleanup
import dept_similarity.experiments as R

RDLogger.DisableLog("rdApp.*")
OUT = Path(RESULTS_DIR) / "analog"
K_VALUES = (1, 3, 10)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
RUN_HUNGARIAN = True  # analog set is small; DEPT-Match is affordable here


def _arr(v):
    return v.astype(np.float32, copy=False) if isinstance(v, np.ndarray) else np.asarray(v, np.float32)


def fp(smiles):
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return GEN.GetFingerprint(m) if m is not None else None


# ---- analog query band, attach spectra ----
# Floor (0.70) > analog metric threshold (0.60): every query has a genuine analog in the
# library with headroom above the hit line, so analog@k is not circular. Ceiling (0.85)
# excludes near-duplicate scaffolds that blur into exact-match retrieval.
BAND_LO, BAND_HI = 0.70, 0.85
tmax = pd.read_parquet(OUT / "query_tmax.parquet")
band = tmax[(tmax["tmax"] >= BAND_LO) & (tmax["tmax"] < BAND_HI)].copy()
# Section 2.1 quality + elemental filters (defensive; also applied upstream in
# exp_analog_tanimoto.py): RDKit-parseable, elements in {C,H,O,N,S,P,Cl,Br,I},
# neutral formal charge, 120 <= MW <= 1200 Da.
band = band[band["smiles"].apply(passed_cleanup)].reset_index(drop=True)

full = pd.read_parquet(f"{PROCESSED_DATA_DIR}/full_nmrshiftdb_dept135_like_spectra.parquet")
# Tolerate either column naming: intensity_signed (older exports) or signed_shifts.
full = full.rename(columns={"intensity_signed": "signed_shifts", "inchikey_14": "inchikey14"})
full = full.sort_values("score", ascending=False).drop_duplicates("inchikey14")
spec = full.set_index("compound_id")["signed_shifts"]
band["signed_shifts"] = band["compound_id"].map(spec).map(_arr)
band = band[band["signed_shifts"].map(len) > 0].reset_index(drop=True)
print(f"analog queries ({BAND_LO}<=Tmax<{BAND_HI}, with spectra): {len(band)}", flush=True)

lib = pd.read_parquet(f"{PROCESSED_DATA_DIR}/library_data.parquet")
lib = lib.rename(columns={"intensity_signed": "signed_shifts", "inchikey_14": "inchikey14"})
lib["signed_shifts"] = lib["signed_shifts"].map(_arr)

qdf = band[["compound_id", "signed_shifts"]].copy()
pairs = {k: sorted(v) for k, v in prefilter_pairs_by_peak_overlap(qdf, lib).items()}
avg = np.mean([len(v) for v in pairs.values()])
print(f"candidate pool built: avg {avg:.0f} candidates/query", flush=True)

qp = qdf.set_index("compound_id")["signed_shifts"].to_dict()
cand_ids = {c for v in pairs.values() for c in v}
lib_sub = lib[lib["compound_id"].isin(cand_ids)]
lp = lib_sub.set_index("compound_id")["signed_shifts"].to_dict()
lib_smiles = lib.set_index("compound_id")["smiles"].to_dict()
lib_inchikey = lib.set_index("compound_id")["inchikey"].to_dict()

query_ids = list(pairs.keys())
nearest = dict(zip(band["compound_id"], band["nearest_analog_id"]))
qtmax = dict(zip(band["compound_id"], band["tmax"]))
qsmiles = dict(zip(band["compound_id"], band["smiles"]))

# coverage: does the true nearest analog survive the prefilter?
covered = {q: (nearest[q] in set(pairs[q])) for q in query_ids}
print(f"nearest-analog prefilter coverage: {100*np.mean(list(covered.values())):.1f}%", flush=True)

OPT = Path(RESULTS_DIR) / "optimization"
GAUSS_SIGMA = float(pd.read_parquet(OPT/"gaussian"/"summary_hits_at_k.parquet").sort_values("Hits@10 (%)").iloc[-1]["sigma"])
HUNG_TOL = float(pd.read_parquet(OPT/"hungarian"/"summary_hits_at_k.parquet").sort_values("Hits@10 cosine (%)").iloc[-1]["ppm_tolerance"])
cand_order = sorted(cand_ids)
cand_index = {c: i for i, c in enumerate(cand_order)}


def tanimoto_to(qsmi, cand_id_list):
    """query -> each retrieved candidate Tanimoto (Morgan r=2)."""
    qf = fp(qsmi)
    if qf is None:
        return [0.0] * len(cand_id_list)
    cfs = [fp(lib_smiles.get(c)) for c in cand_id_list]
    return [DataStructs.TanimotoSimilarity(qf, cf) if cf is not None else 0.0 for cf in cfs]


def score(ranked_df, method):
    ranked = ranked_df.set_index("query_id")["candidate_id_list"].to_dict()
    rows = []
    tani = {q: tanimoto_to(qsmiles[q], ranked.get(q, [])) for q in query_ids}
    for k in K_VALUES:
        na_hit, near_opt, analog, best_t = [], [], [], []
        for q in query_ids:
            top = ranked.get(q, [])[:k]
            t = tani[q][:k]
            bt = max(t) if t else 0.0
            na_hit.append(nearest[q] in top)
            near_opt.append(bt >= qtmax[q] - 0.05)
            analog.append(bt >= 0.6)
            best_t.append(bt)
            # persist per-query outcomes for CIs / paired tests (McNemar)
            per_query_hits.append({"method": method, "k": k, "query_id": q,
                                   "analog_hit": bool(bt >= 0.6),
                                   "near_optimal": bool(bt >= qtmax[q] - 0.05),
                                   "nearest_analog_hit": bool(nearest[q] in top),
                                   "best_tanimoto": float(bt)})
        rows.append({"method": method, "k": k,
                     "nearest_analog_hit_pct": 100*np.mean(na_hit),
                     "near_optimal_pct": 100*np.mean(near_opt),
                     "analog_pct": 100*np.mean(analog),
                     "mean_best_tanimoto": float(np.mean(best_t)),
                     "n": len(query_ids)})
    return rows


all_rows = []
per_query_hits = []

for method in ["gauss_kernel", "ppm_bins_typed"]:
    t = time.time()
    lib_norm = R.encode_library_vectors(cand_order, lp, method, GAUSS_SIGMA)
    df = R.rank_vector_method(query_ids, qp, pairs, cand_index, lib_norm, method, GAUSS_SIGMA, lib_smiles, lib_inchikey)
    all_rows += score(df, {"gauss_kernel": "Gaussian", "ppm_bins_typed": "Shift binned"}[method])
    print(f"[{method}] ranked+scored in {time.time()-t:.0f}s", flush=True)
    del lib_norm

if RUN_HUNGARIAN:
    t = time.time()
    R.set_hungarian_globals(lp, lib_smiles, lib_inchikey)
    df = R.rank_hungarian(query_ids, qp, pairs, HUNG_TOL)["cosine"]
    all_rows += score(df, "DEPT-Match")
    print(f"[hungarian] ranked+scored in {time.time()-t:.0f}s", flush=True)

metrics = pd.DataFrame(all_rows)
metrics.to_parquet(OUT / "metrics.parquet", index=False)
pd.DataFrame({"query_id": query_ids,
             "tmax": [qtmax[q] for q in query_ids],
             "covered": [covered[q] for q in query_ids]}).to_parquet(OUT / "per_query.parquet", index=False)
pd.DataFrame(per_query_hits).to_parquet(OUT / "per_query_hits.parquet", index=False)
print("\n=== ANALOG RETRIEVAL METRICS ===", flush=True)
print(metrics.round(1).to_string(index=False), flush=True)
print("SAVED", OUT / "metrics.parquet", flush=True)
