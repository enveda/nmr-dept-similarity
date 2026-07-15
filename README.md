# Automated natural product dereplication via DEPT-135 spectral matching

`dept-similarity` implements and benchmarks several methods for ranking library
compounds by how closely their DEPT NMR spectra match an experimental query
spectrum. It ships the similarity methods themselves, an end-to-end evaluation
pipeline on the [NMRShiftDB2](https://nmrshiftdb.nmr.uni-koeln.de/) benchmark,
and the notebooks that reproduce every figure in the accompanying study.

The data directory with the preloaded results can be found on [Zenodo](http://doi.org/10.5281/zenodo.21373988)

## Citation

If you use this code or the accompanying benchmark, please cite:

> Gadiya Y, Singh M, Kind T, Macherla V, Allen A, Misra BB, Domingo-Fernández D.
> **Automated natural product dereplication via DEPT-135 spectral matching.**

```bibtex
@article{gadiya_dept135_dereplication,
  title   = {Automated natural product dereplication via DEPT-135 spectral matching},
  author  = {Gadiya, Yojana and Singh, Manvendra and Kind, Tobias and
             Macherla, Venkat and Allen, August and Misra, Biswapriya B. and
             Domingo-Fern{\'a}ndez, Daniel},
  year    = {2026},
}
```

## Methods

Spectra are encoded as sets of signed chemical shifts — the *magnitude* is the
shift in ppm, the *sign* encodes multiplicity (negative = CH₂, positive =
CH/CH₃). Four similarity methods are implemented and benchmarked:

| Method | Key | Idea |
| --- | --- | --- |
| **DEPT-Match** | `hungarian_match` | Hungarian optimal peak-to-peak assignment within a ppm tolerance, scored by cosine or Jaccard over matched peaks. *Best single method.* |
| **Gaussian** | `gauss_kernel` | Each peak becomes a Gaussian on a fixed ppm grid; cosine over the resulting dense vectors. |
| **Shift-binned** | `ppm_bins_typed` | Peaks binned into fixed-width multiplicity-typed ppm bins; cosine over bin counts. |
| **Ensemble** | — | Rank fusion of DEPT-Match + Gaussian — beats every single method (notebook 04). |

Optimal parameters were selected on the full validation set: DEPT-Match
`ppm_tolerance = 3.0`, Gaussian `σ = 2.5`. Peaks are weighted by presence
(**binary**; every peak weighs 1.0) throughout.

## Requirements

- Python ≥ 3.10, < 3.13
- [uv](https://docs.astral.sh/uv/) for environment management

## Getting Started

```bash
# First-time setup (locks deps, syncs, installs pre-commit hooks)
make project-init

# Subsequent setups
make project-setup
# or simply
uv sync --all-groups
```

## Usage

The similarity methods share one entry point, `rank_library`,
selected by method key. A spectrum is encoded in the `signed_shifts` column
which holds **signed chemical shifts** (ppm). The *magnitude* is the shift and the *sign* is the multiplicity
(negative = CH₂, positive = CH/CH₃). So a CH at 68.7 ppm is `68.7` and a CH₂ at
58.7 ppm is `-58.7`. (Peak amplitude, if you have it, lives in a separate
`dept_intensity` column and is not used by the default binary matching.)

```python
import numpy as np
import pandas as pd

from dept_similarity.match import rank_library, METHOD_REGISTRY
from dept_similarity.prepare import prefilter_pairs_by_peak_overlap

print(sorted(METHOD_REGISTRY))
# ['gauss_kernel', 'hungarian_match', 'ppm_bins', 'ppm_bins_typed', 'spectral_match']

# --- Reference library: one row per compound (id, structure, signed shifts) ---
# DEPT-135 shows only protonated carbons; the sign encodes multiplicity
# (CH/CH₃ positive, CH₂ negative). These are real 5-peak NMRShiftDB2 entries.
library_df = pd.DataFrame(
    [
        {"compound_id": "dept_1000113", "smiles": "OCC[C@]1(O)CC[C@H](O)CC1",
         "inchikey": "TWORTZAXDSRCIT-ZKCHVHJHSA-N",       # cyclohexane-1,4-diol deriv.
         "signed_shifts": np.array([-31.1, -35.3, -43.0, -58.7, 68.7], dtype=np.float32)},
        {"compound_id": "dept_1000503", "smiles": "OCCCCCCCCCO",
         "inchikey": "ALVZNPYWJMLXKV-UHFFFAOYSA-N",       # nonane-1,9-diol
         "signed_shifts": np.array([-26.1, -29.4, -29.5, -32.8, -62.8], dtype=np.float32)},
        {"compound_id": "dept_1000750", "smiles": "C1=CC=C(C2=CN=CS2)C=C1",
         "inchikey": "ZLLOWHFKKIOINR-UHFFFAOYSA-N",       # 4-phenylthiazole
         "signed_shifts": np.array([127.2, 128.6, 128.7, 139.0, 152.1], dtype=np.float32)},
        # ... add the rest of your reference spectra ...
    ]
)

# --- Query spectrum, straight from Python: a list of signed shifts (ppm) ---
# (here, a measured spectrum of the cyclohexane-diol above)
query_df = pd.DataFrame(
    [
        {
            "compound_id": "my_query",
            "signed_shifts": np.array([-31.0, -35.4, -43.2, -58.5, 68.9], dtype=np.float32),
        }
    ]
)

# 1) Screen candidates by peak overlap -> {query_id: {candidate_ids}}
#    (defaults min_shared_peaks=3, min_query_coverage=0.75 suit full spectra;
#    relaxed here because the toy spectra have only a couple of peaks)
pairs = prefilter_pairs_by_peak_overlap(
    query_df, library_df, min_shared_peaks=1, min_query_coverage=0.0,
)

# 2a) Gaussian (cosine) — each peak becomes a Gaussian on a fixed ppm grid
gaussian = rank_library(
    pairs, query_df, library_df,
    method="gauss_kernel", sim_metrics=("cosine",),
    method_kwargs={"sigma": 2.5},          # 03b-optimal σ
)

# 2b) DEPT-Match (cosine) — Hungarian optimal peak-to-peak assignment
dept_match = rank_library(
    pairs, query_df, library_df,
    method="hungarian_match", sim_metrics=("cosine",),
    method_kwargs={"ppm_tolerance": 3.0},  # 03a-optimal tolerance
)

top = dept_match.iloc[0]
print(top["candidate_id_list"])   # library ids, best match first
print(top["score_list"])          # matching cosine scores, descending
```

Each call returns a DataFrame with one row per `(query, metric)` carrying the
ranked `candidate_id_list` / `score_list` (plus `candidate_smiles_list` /
`candidate_inchikey_list` when the library has structure columns). When your
queries have a known true structure, turn those ranked lists into retrieval
metrics with `get_metrics_at_k` — **Hits@k** (14-char InChIKey match) and
**mean Tanimoto@k** (Morgan fingerprints).

## Repository structure

```text
dept_similarity/   # the Python package: encoders, matchers, scoring, metrics
├── match.py           # method registry + rank_library / get_metrics_at_k
├── spectral_matching.py  # Hungarian peak alignment (DEPT-Match)
├── gaussian_vector.py    # Gaussian-kernel encoding
├── bin_vector.py         # shift-binned encoding
├── score.py              # numba-accelerated distance/similarity kernels
├── experiments.py        # noise model & candidate loaders for notebooks 07–08
└── constants.py          # baseline parameters, paths, palettes
notebooks/         # end-to-end workflow (run in order) — see notebooks/README.md
scripts/           # standalone CLI drivers for the heavy cached experiments
data/              # raw / processed / results / figures (not git-tracked)
tests/             # pytest suite
```

## Reproducing the study

The [`notebooks/`](notebooks/) directory contains the full pipeline — from raw
NMRShiftDB2 parsing through parameter optimization to the final figures. Run
them **in numeric order**; later notebooks consume artifacts produced by earlier
ones. See [notebooks/README.md](notebooks/README.md) for the complete
notebook-by-notebook description.

| # | Notebook | Purpose |
| --- | --- | --- |
| 01a/01b | data preparation | Build the reference library and the 648-query experimental validation set from NMRShiftDB2. |
| 02 | exploratory analysis | Peak/shift distributions, NPClassifier coverage, t-SNE, library↔query overlap. |
| 03a–03b | parameter optimization | Sweep DEPT-Match tolerance & Gaussian σ. |
| 04 | evaluation results | Headline benchmark: Hits@k per method, true-positive Venn, ensemble, molecular-formula effect. |
| 05 | structural vs spectral | Relate spectral similarity to Morgan/Tanimoto structural similarity. |
| 06 | supplementary figures | Consolidated robustness figures. |
| 07 | noise robustness | Degradation under synthetic chemical-shift noise (jitter + global offset). |
| 08 | retrieval by class | Per-NPClassifier-pathway Hits@k and rank-1 error analysis. |

## Extending

- **New method:** register a `MethodSpec` in
  [`METHOD_REGISTRY`](dept_similarity/match.py) — the rest of the pipeline
  (ranking, metrics, notebooks) picks it up automatically.
- **New metric:** add a numba kernel in [`score.py`](dept_similarity/score.py).

## Development

```bash
make test      # uv run pytest -v
make lint      # uv run ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution workflow. Code
style is enforced with [ruff](https://docs.astral.sh/ruff/) and
[black](https://black.readthedocs.io/) (100-char lines) via pre-commit.

## License

Released under the [MIT License](LICENSE). © 2026 Enveda.
