"""Analog-retrieval, phase 1: query->library nearest-analog Tanimoto (vectorised).

For every NMRShiftDB DEPT-like compound NOT present in the COCONUT library, find its
maximum Morgan (r=2, 2048-bit) Tanimoto to the library and the identity of that nearest
analog. Tanimoto is computed as a chunked BLAS matmul (intersection = bit-matrix product),
which is ~100x faster than per-query BulkTanimoto and does not hog cores.
Checkpointed to data/results/analog/query_tmax.parquet.
"""
import os
# be a good neighbour: the noise sweep may be using most cores
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import time
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from dept_similarity.constants import PROCESSED_DATA_DIR, RESULTS_DIR
from dept_similarity.utils import passed_cleanup

RDLogger.DisableLog("rdApp.*")
OUT = Path(RESULTS_DIR) / "analog"
OUT.mkdir(parents=True, exist_ok=True)
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
NBITS = 2048


def bit_matrix(smiles_list):
    """(N, 2048) uint8 fingerprint matrix; rows of invalid SMILES are all-zero, flagged False."""
    mat = np.zeros((len(smiles_list), NBITS), dtype=np.uint8)
    ok = np.zeros(len(smiles_list), dtype=bool)
    buf = np.zeros(NBITS, dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(m), buf)
        mat[i] = buf
        ok[i] = True
    return mat, ok


# ---- library bit matrix ----
lib = pd.read_parquet(f"{PROCESSED_DATA_DIR}/library_data.parquet")
# Tolerate either column naming: inchikey_14/intensity_signed (older exports) or inchikey14/signed_shifts.
lib = lib.rename(columns={"inchikey_14": "inchikey14", "intensity_signed": "signed_shifts"})
lib = lib[["compound_id", "smiles", "inchikey14"]]
t = time.time()
L, lok = bit_matrix(lib["smiles"].tolist())
lib = lib[lok].reset_index(drop=True)
L = L[lok].astype(np.float32)
lib_ids = lib["compound_id"].to_numpy()
lsum = L.sum(1)  # popcount per library compound
print(f"library bit matrix {L.shape} in {time.time()-t:.0f}s", flush=True)

# ---- query pool: full NMRShiftDB, best spectrum per compound, NOT in library ----
full = pd.read_parquet(f"{PROCESSED_DATA_DIR}/full_nmrshiftdb_dept135_like_spectra.parquet")
full = full.sort_values("score", ascending=False).drop_duplicates("inchikey14")
full = full[~full["inchikey14"].isin(set(lib["inchikey14"]))].reset_index(drop=True)
# Section 2.1 quality + elemental filters: RDKit-parseable, elements in
# {C,H,O,N,S,P,Cl,Br,I}, neutral formal charge, 120 <= MW <= 1200 Da.
n_pre = len(full)
full = full[full["smiles"].apply(passed_cleanup)].reset_index(drop=True)
print(f"quality/elemental filter: {n_pre} -> {len(full)} out-of-library structures", flush=True)
Q, qok = bit_matrix(full["smiles"].tolist())
Q = Q.astype(np.float32)
qsum = Q.sum(1)
print(f"out-of-library query pool: {len(full)} ({int(qok.sum())} valid)", flush=True)

# ---- chunked BLAS Tanimoto: best (max) similarity + argmax library id per query ----
t = time.time()
best = np.zeros(len(full), dtype=np.float32)
best_j = np.full(len(full), -1, dtype=np.int64)
QC, LB = 3000, 60000
for qs in range(0, len(full), QC):
    qe = min(qs + QC, len(full))
    Qc, qsc = Q[qs:qe], qsum[qs:qe]
    for ls in range(0, len(lib), LB):
        le = min(ls + LB, len(lib))
        inter = Qc @ L[ls:le].T                       # (qc, lb) intersection counts
        union = qsc[:, None] + lsum[None, ls:le] - inter
        tan = np.where(union > 0, inter / union, 0.0)
        j = tan.argmax(1)
        v = tan[np.arange(tan.shape[0]), j]
        upd = v > best[qs:qe]
        best[qs:qe] = np.where(upd, v, best[qs:qe])
        best_j[qs:qe] = np.where(upd, ls + j, best_j[qs:qe])
    print(f"  queries {qe}/{len(full)}  ({time.time()-t:.0f}s)", flush=True)

full["tmax"] = np.where(qok, best, np.nan)
full["nearest_analog_id"] = [lib_ids[j] if j >= 0 else None for j in best_j]
full.loc[~qok, "nearest_analog_id"] = None

full[["compound_id", "inchikey14", "smiles", "tmax", "nearest_analog_id"]].to_parquet(
    OUT / "query_tmax.parquet", index=False)
band = full[(full["tmax"] >= 0.70) & (full["tmax"] < 0.85)]
print(f"SAVED {OUT/'query_tmax.parquet'}  |  0.70<=Tmax<0.85: n={len(band)}", flush=True)
print("Tmax distribution:", full["tmax"].describe().round(3).to_dict(), flush=True)
