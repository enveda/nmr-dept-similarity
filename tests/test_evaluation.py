"""Tests for structure export in rank_library and get_metrics_at_k."""

import numpy as np
import pandas as pd
import pytest

from dept_similarity.match import get_metrics_at_k, rank_library


def _spectra_df(records, with_structure=False):
    rows = []
    for cid, peaks, inchikey, smiles in records:
        row = {
            "compound_id": cid,
            "signed_shifts": np.asarray(peaks, dtype=np.float32),
            "dept_intensity": np.asarray([abs(p) for p in peaks], dtype=np.float32),
        }
        if with_structure:
            row["inchikey"] = inchikey
            row["inchikey14"] = inchikey[:14]
            row["smiles"] = smiles
        rows.append(row)
    return pd.DataFrame(rows)


# Ethanol / benzene / methane — distinct real SMILES so Tanimoto is meaningful.
QUERY = _spectra_df(
    [("Q1", [20.0, -45.0, 80.0], "AAAAAAAAAAAAAA-AAAAAAAAAA-N", "CCO")],
    with_structure=True,
)
LIBRARY = _spectra_df(
    [
        ("L1", [20.0, -45.0, 80.0], "AAAAAAAAAAAAAA-BBBBBBBBBB-N", "CCO"),  # same key14 as Q1
        ("L2", [30.0, -60.0], "CCCCCCCCCCCCCC-CCCCCCCCCC-N", "c1ccccc1"),
        ("L3", [100.0, 150.0], "DDDDDDDDDDDDDD-DDDDDDDDDD-N", "C"),
    ],
    with_structure=True,
)
PAIRS = {"Q1": ["L1", "L2", "L3"]}


# ---------------------------------------------------------------------------
# rank_library structure export
# ---------------------------------------------------------------------------


def test_structure_columns_added_when_library_has_them():
    out = rank_library(
        PAIRS, QUERY.copy(), LIBRARY.copy(), method="spectral_match", sim_metrics=("cosine",)
    )
    assert "candidate_smiles_list" in out.columns
    assert "candidate_inchikey_list" in out.columns
    row = out.iloc[0]
    # ranked lists stay aligned across id / smiles / inchikey
    assert len(row["candidate_smiles_list"]) == len(row["candidate_id_list"])
    top_id = row["candidate_id_list"][0]
    top_smiles = row["candidate_smiles_list"][0]
    assert top_smiles == LIBRARY.set_index("compound_id").loc[top_id, "smiles"]


def test_no_structure_columns_when_library_lacks_them():
    lib_no_struct = LIBRARY[["compound_id", "signed_shifts"]].copy()
    q_no_struct = QUERY[["compound_id", "signed_shifts"]].copy()
    out = rank_library(
        PAIRS, q_no_struct, lib_no_struct, method="spectral_match", sim_metrics=("cosine",)
    )
    assert set(out.columns) == {"query_id", "metric_name", "candidate_id_list", "score_list"}


def test_include_structure_false_suppresses_columns():
    out = rank_library(
        PAIRS,
        QUERY.copy(),
        LIBRARY.copy(),
        method="spectral_match",
        sim_metrics=("cosine",),
        include_structure=False,
    )
    assert "candidate_smiles_list" not in out.columns


def test_top_k_truncates_all_ranked_lists():
    out = rank_library(
        PAIRS, QUERY.copy(), LIBRARY.copy(), method="spectral_match", sim_metrics=("cosine",), top_k=2
    )
    row = out.iloc[0]
    assert len(row["candidate_id_list"]) == 2
    assert len(row["score_list"]) == 2
    assert len(row["candidate_smiles_list"]) == 2
    assert len(row["candidate_inchikey_list"]) == 2


def test_invalid_intensity_mode_raises():
    with pytest.raises(ValueError, match="Unknown intensity_mode"):
        rank_library(
            PAIRS, QUERY.copy(), LIBRARY.copy(), method="ppm_bins", intensity_mode="bogus"
        )


def test_normalized_mode_runs_with_intensity_column():
    out = rank_library(
        PAIRS,
        QUERY.copy(),
        LIBRARY.copy(),
        method="spectral_match",
        sim_metrics=("cosine",),
        intensity_mode="normalized",
        intensity_col="dept_intensity",
    )
    assert len(out) == 1
    assert all(0.0 <= s <= 1.0 for s in out.iloc[0]["score_list"])


# ---------------------------------------------------------------------------
# get_metrics_at_k
# ---------------------------------------------------------------------------


def test_hits_at_k_uses_inchikey14():
    out = rank_library(
        PAIRS, QUERY.copy(), LIBRARY.copy(), method="spectral_match", sim_metrics=("cosine",)
    )
    res = get_metrics_at_k(out, QUERY.copy(), k_values=(1, 3), with_inchikey14=True, compute_tanimoto=False)
    # L1 shares Q1's key14 and is the top cosine hit -> hit@1 and hit@3.
    at1 = res[res["k"] == 1].iloc[0]
    assert at1["true_positives"] == 1
    assert at1["accuracy"] == pytest.approx(100.0)
    assert "mean_tanimoto" not in res.columns


def test_tanimoto_at_1_is_self_similarity():
    out = rank_library(
        PAIRS, QUERY.copy(), LIBRARY.copy(), method="spectral_match", sim_metrics=("cosine",)
    )
    res = get_metrics_at_k(out, QUERY.copy(), k_values=(1,), with_inchikey14=True, compute_tanimoto=True)
    # Top-1 candidate is L1 (SMILES 'CCO') == query 'CCO' -> Tanimoto 1.0.
    assert res.iloc[0]["mean_tanimoto"] == pytest.approx(1.0, abs=1e-6)


def test_tanimoto_at_k_decreases_with_more_candidates():
    out = rank_library(
        PAIRS, QUERY.copy(), LIBRARY.copy(), method="spectral_match", sim_metrics=("cosine",)
    )
    res = get_metrics_at_k(out, QUERY.copy(), k_values=(1, 3), compute_tanimoto=True)
    t1 = res[res["k"] == 1].iloc[0]["mean_tanimoto"]
    t3 = res[res["k"] == 3].iloc[0]["mean_tanimoto"]
    # Adding dissimilar benzene/methane lowers the mean Tanimoto below 1.0.
    assert t1 > t3


def test_metrics_at_k_falls_back_to_library_df():
    # Drop the exported structure columns; get_metrics_at_k must map from library.
    out = rank_library(
        PAIRS,
        QUERY.copy(),
        LIBRARY.copy(),
        method="spectral_match",
        sim_metrics=("cosine",),
        include_structure=False,
    )
    assert "candidate_inchikey_list" not in out.columns
    res = get_metrics_at_k(
        out, QUERY.copy(), library_spectra_df=LIBRARY.copy(), k_values=(1,), compute_tanimoto=True
    )
    assert res.iloc[0]["true_positives"] == 1
    assert res.iloc[0]["mean_tanimoto"] == pytest.approx(1.0, abs=1e-6)


def test_n_total_queries_denominator():
    out = rank_library(
        PAIRS, QUERY.copy(), LIBRARY.copy(), method="spectral_match", sim_metrics=("cosine",)
    )
    res = get_metrics_at_k(
        out, QUERY.copy(), k_values=(1,), n_total_queries=4, compute_tanimoto=False
    )
    at1 = res.iloc[0]
    assert at1["total_queries"] == 4
    assert at1["accuracy"] == pytest.approx(25.0)  # 1 hit / 4


def test_empty_results_returns_empty_frame():
    empty = pd.DataFrame(
        columns=["query_id", "metric_name", "candidate_id_list", "score_list"]
    )
    res = get_metrics_at_k(empty, QUERY.copy(), compute_tanimoto=False)
    assert res.empty
