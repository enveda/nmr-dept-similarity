"""Constant values used across the dept similarity analysis."""

# Default tolerances validated on the NMRshiftDB2 benchmark
from pathlib import Path

# Standardized figure and cache directories
DATA_DIR = Path("../data")
FIGURE_DIR = Path("../data/figures")
CACHE_DIR = Path("../data/cache")
RESULTS_DIR = Path("../data/results")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# Make dirs if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_ELEMENTS = {"C", "H", "O", "N", "S", "P", "Cl", "Br", "I"}

LIBRARY_PALETTE = {
    "Library": "#9CA3AF",  # grey
    "Query": "#0EA5A4",  # teal
    "Other": "#F4A261",  # orange
}

HITS_K_PALETTE = {
    "hits@1": "#F0C98A",
    "hits@5": "#C4714A",
    "hits@10": "#6B3520",
}

PALETTE_METHODS = ["#6bb48b", "#e9af7b", "#7fb5d6", "#e97b7b", "#7be9e9", "#e9e97b"]
METHOD_NAME_MAPPER = {
    "hungarian_match_jaccard": "DEPT-Match (Jaccard)",
    "gauss_kernel_cosine": "Gaussian (Cosine)",
    "hungarian_match_cosine": "DEPT-Match (Cosine)",
    "ppm_bins_typed_cosine": "Shift binned (Cosine)",
}

# Constants for the similarity metrics
BASELINE_PARAMS = {
    "ppm_range": (10.0, 170.0),  # Consider peaks between 10 and 170 ppm
    "bin_width": 3.7,  # Target bin width of 3.7 ppm (≈ 45 bins total)
    "prefilter_ppm_tol": 4.0,  # Broad peak-overlap tolerance for candidate-set screening (prefilter)
    "spectral_ppm_tol": 3.0,  # DEPT-Match matching tolerance — 03a full-N Hits@10 optimum
    "spectral_min_matched_peaks": 1,  # Minimum aligned peaks required for a non-zero score
    "gaussian_grid": (0, 210),  # Grid parameters: (min, max) in ppm
    "gaussian_sigma": 2.5,  # Gaussian σ (ppm) — 03b full-N Hits@10 optimum; also the LUT fast-path anchor
}
