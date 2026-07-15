# Data Directory Guide

This document summarizes what each directory contains for the
`nmr-dept-similarity` dataset and how artifacts are organized.

Zenodo dump - http://doi.org/10.5281/zenodo.21373988 

## Repository context (where data artifacts come from)

- `dept_similarity/`: Python package implementing DEPT spectrum encoders,
	matchers, ranking, and evaluation metrics.
- `notebooks/`: End-to-end analysis workflow that produces most data artifacts
	in this folder (`data/processed`, `data/results`, `data/figures`).
- `scripts/`: Standalone runners for heavier experiments (for example,
	noise-robustness, class-stratified analysis, analog-focused evaluation).
- `tests/`: Unit/integration tests for the package and experiment utilities.

## `data/` directory structure

### `raw/`

Original input data downloaded from NMRShiftDB2.

- `nmrshiftdb2withsignals.sd`: raw source SDF file used to build the library
	and validation subsets.
- `README`: source download pointer and retrieval note.

### `processed/`

Cleaned and analysis-ready tabular datasets in Parquet format.

- `library_data.parquet`: searchable reference library built from raw
	NMRShiftDB2 spectra.
- `experimental_validation_set.parquet`: curated query/validation set used for
	benchmarking retrieval methods.
- `nmrshiftdb_dept135_like_spectra.parquet` and
	`full_nmrshiftdb_dept135_like_spectra.parquet`: intermediate/full processed
	DEPT-like spectra tables used during preparation and analysis.
- `multiple_spectra_overlap_set.parquet`: overlap/consistency subset used for
	QC and overlap-focused analyses.

### `cache/`

Cached auxiliary metadata used to avoid recomputation or repeated API calls.

- `npclassifier_cache.json`: cached NPClassifier annotations consumed by
	exploratory and class-based evaluation workflows.

### `results/`

Method outputs, optimization summaries, and experiment-specific result tables.

- `hungarian_match_cosine/`, `hungarian_match_jaccard/`,
	`gauss_kernel_cosine/`, `ppm_bins_typed_cosine/`: baseline method result
	directories containing `ranked_top10.parquet` ranked candidate lists.
- `optimization/`: parameter sweep outputs.
	- `hungarian/`: tolerance sweep summaries and per-threshold Hits@K tables.
	- `gaussian/`: sigma sweep summaries, plots, and per-sigma Hits@K tables.
	- `peak_count/`: retrieval-versus-peak-count performance curve.
- `noise_robustness/`: summary table for synthetic chemical-shift perturbation
	experiments.
- `analog/`: nearest-analog focused evaluation tables, including per-query and
	aggregate metrics (`query_tmax.parquet`, `metrics.parquet`, etc.).

### `figures/`

Publication and supplementary figure outputs (PNG), including headline and
diagnostic visualizations.

- Main-study figures (for example, `figure2.png`, `figure3.png`,
	`figure4.png`, `figure5_ensemble.png`, `figure6_analog_retrieval.png`).
- Supplementary/diagnostic figures (for example, `tolerance_sweep.png`,
	`noise_robustness.png`, `peak_count.png`, `class_analysis.png`,
	`tsne_distribution.png`).

## File formats and conventions

- `*.parquet`: columnar tabular data for processed datasets and result tables.
- `*.json`: cached metadata.
- `*.png`: rendered analysis figures.
- `*.sd`/`*.sdf`: raw chemical/spectral source records.

## Reproducibility mapping

Typical production flow for archived artifacts:

1. Place raw source file in `data/raw/`.
2. Run preparation notebooks/scripts to populate `data/processed/`.
3. Run optimization/evaluation notebooks/scripts to populate `data/results/`.
4. Generate publication/supporting visuals in `data/figures/`.

This layout is designed so Zenodo users can distinguish source data,
derived/processed tables, cached metadata, final benchmark outputs, and
presentation-ready figures.
