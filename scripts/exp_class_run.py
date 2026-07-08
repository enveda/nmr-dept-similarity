"""Experiment 2 — retrieval success & error analysis by natural-product class.

Uses the cached baseline ranked lists (data/results/<method>_<metric>/ranked_top10.parquet)
and the NPClassifier annotations (query set) to answer: for which structural classes does
DEPT retrieval succeed or fail?

Outputs (data/results/class_analysis/):
  per_class_hits.parquet   — per (method, class_level, class, k): n_queries, hit_rate
  confusion.parquet        — for rank-1 misses, whether the top-1 wrong hit shares the
                             query's superclass (within- vs cross-class error), where the
                             top-1 candidate's class is known from the NPClassifier cache.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from pathlib import Path
import numpy as np, pandas as pd, requests
from tqdm import tqdm

from dept_similarity.constants import RESULTS_DIR, PROCESSED_DATA_DIR, CACHE_DIR

K_VALUES = (1, 3, 10)

q = pd.read_parquet(f"{PROCESSED_DATA_DIR}/experimental_validation_set.parquet")
q["compound_id"] = q["compound_id"] + "_" + q["inchikey14"]
q_true14 = dict(zip(q["compound_id"], q["inchikey14"]))

NPC_PATH = Path(CACHE_DIR) / "npclassifier_cache.json"
npc = json.load(open(NPC_PATH))

METHOD_DIRS = ["gauss_kernel_cosine", "hungarian_match_cosine", "ppm_bins_typed_cosine"]


def _fetch(smiles, retries=3):
    url = f"https://npclassifier.gnps2.org/classify?smiles={quote(smiles)}"
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
    return None


# ---- enrich cache with the top-1 wrong candidate of every rank-1 miss ----
to_classify = {}  # inchikey14 -> smiles
for md in METHOD_DIRS:
    df = pd.read_parquet(Path(RESULTS_DIR) / md / "ranked_top10.parquet")
    for _, row in df.iterrows():
        true14 = q_true14.get(row["query_id"])
        cand_ik = list(row["candidate_inchikey_list"])
        cand_sm = list(row["candidate_smiles_list"])
        if not cand_ik:
            continue
        top1_14 = cand_ik[0][:14] if isinstance(cand_ik[0], str) else None
        if top1_14 and top1_14 != true14 and top1_14 not in npc and isinstance(cand_sm[0], str):
            to_classify[top1_14] = cand_sm[0]

print(f"top-1 wrong candidates needing classification: {len(to_classify)}", flush=True)
if to_classify:
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch, sm): ik for ik, sm in to_classify.items()}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="NPClassifier"):
            res = fut.result()
            if res is not None:
                npc[futs[fut]] = res
    json.dump(npc, open(NPC_PATH, "w"))
    print(f"cache now holds {len(npc)} entries", flush=True)


def classes_of(ik14, level):
    """level in {'pathway','superclass','class'} -> list of labels (['Unclassified'] if none)."""
    r = npc.get(ik14)
    if r is None:
        return ["Unknown"]
    vals = r.get(f"{level}_results") or []
    return vals if vals else ["Unclassified"]


METHODS = {
    "gauss_kernel_cosine": "Gaussian (Cosine)",
    "hungarian_match_cosine": "DEPT-Match (Cosine)",
    "ppm_bins_typed_cosine": "Shift binned (Cosine)",
}

hit_rows = []
conf_rows = []

for method_dir, label in METHODS.items():
    df = pd.read_parquet(Path(RESULTS_DIR) / method_dir / "ranked_top10.parquet")
    for _, row in df.iterrows():
        qid = row["query_id"]
        true14 = q_true14.get(qid)
        if true14 is None:
            continue
        cand_ik = list(row["candidate_inchikey_list"])
        cand14 = [ik[:14] if isinstance(ik, str) else None for ik in cand_ik]

        hit_at = {k: (true14 in cand14[:k]) for k in K_VALUES}

        # stratify this query by each class level (a query may map to several labels)
        for level in ("pathway", "superclass", "class"):
            for cls in classes_of(true14, level):
                for k in K_VALUES:
                    hit_rows.append({"method": label, "level": level, "class": cls,
                                     "k": k, "query_id": qid, "hit": int(hit_at[k])})

        # rank-1 error analysis: when the top-1 is wrong, is it in the same superclass?
        if not hit_at[1] and cand14:
            top1_14 = cand14[0]
            q_sc = set(classes_of(true14, "superclass"))
            top1_sc = set(classes_of(top1_14, "superclass")) if top1_14 in npc else None
            conf_rows.append({
                "method": label, "query_id": qid,
                "query_superclass": ";".join(sorted(q_sc)),
                "top1_superclass": (";".join(sorted(top1_sc)) if top1_sc else None),
                "top1_in_cache": top1_14 in npc,
                "same_superclass": (bool(q_sc & top1_sc) if top1_sc else None),
            })

hit_df = pd.DataFrame(hit_rows)
# aggregate to per-class hit rate
per_class = (hit_df.groupby(["method", "level", "class", "k"])
             .agg(n_queries=("query_id", "nunique"), hits=("hit", "sum"))
             .reset_index())
per_class["hit_rate"] = 100.0 * per_class["hits"] / per_class["n_queries"]

conf_df = pd.DataFrame(conf_rows)

out = Path(RESULTS_DIR) / "class_analysis"
out.mkdir(parents=True, exist_ok=True)
per_class.to_parquet(out / "per_class_hits.parquet", index=False)
conf_df.to_parquet(out / "confusion.parquet", index=False)

# quick console summary
print("=== per-class Hits@10 by pathway (Gaussian) ===")
g = per_class[(per_class.method == "Gaussian (Cosine)") & (per_class.level == "pathway") & (per_class.k == 10)]
print(g.sort_values("hit_rate")[["class", "n_queries", "hit_rate"]].to_string(index=False))
print(f"\nconfusion rows (rank-1 misses): {len(conf_df)}")
if len(conf_df):
    known = conf_df[conf_df.same_superclass.notna()]
    print(f"top-1 wrong hit class known for {len(known)}/{len(conf_df)} misses")
    if len(known):
        rate = known.assign(same=known["same_superclass"].astype(float)).groupby("method")["same"].mean()
        print("same-superclass error rate:", rate.round(3).to_dict())
print("SAVED", out)
