"""Tests for dept_similarity.prepare — peak-overlap prefilter and candidate validation."""

import numpy as np
import pandas as pd

from dept_similarity.prepare import (
    prefilter_pairs_by_peak_overlap,
    validate_candidate_set,
)


def _make_signed_df(rows):
    """Build a DataFrame with compound_id and signed_shifts columns."""
    return pd.DataFrame(
        [
            {"compound_id": cid, "signed_shifts": np.asarray(peaks, dtype=np.float32)}
            for cid, peaks in rows
        ]
    )


def test_prefilter_keeps_overlapping_pair():
    # Query peak at +30, library peak at +31 (1 ppm apart, within default 4 ppm tolerance).
    query_df = _make_signed_df([("Q1", [30.0])])
    ref_df = _make_signed_df([("R1", [31.0])])
    pairs = prefilter_pairs_by_peak_overlap(query_df, ref_df)
    assert "R1" in pairs["Q1"]


def test_prefilter_removes_nonoverlapping_pair():
    # Query peak at +30, library peak at +100 — far outside any tolerance.
    query_df = _make_signed_df([("Q1", [30.0])])
    ref_df = _make_signed_df([("R1", [100.0])])
    pairs = prefilter_pairs_by_peak_overlap(query_df, ref_df)
    assert "R1" not in pairs["Q1"]


def test_prefilter_type_gating_removes_opposite_type():
    # Query CH/CH3 peak at +30 ppm, library CH2 peak at -30 ppm (same magnitude, different type).
    # With match_multiplicity=True (default) → no overlap → excluded.
    query_df = _make_signed_df([("Q1", [30.0])])
    ref_df = _make_signed_df([("R1", [-30.0])])
    pairs = prefilter_pairs_by_peak_overlap(query_df, ref_df, match_multiplicity=True)
    assert "R1" not in pairs["Q1"]


def test_prefilter_type_agnostic_keeps_opposite_type():
    # Same setup but match_multiplicity=False → type ignored → overlap detected.
    query_df = _make_signed_df([("Q1", [30.0])])
    ref_df = _make_signed_df([("R1", [-30.0])])
    pairs = prefilter_pairs_by_peak_overlap(query_df, ref_df, match_multiplicity=False)
    assert "R1" in pairs["Q1"]


def test_prefilter_empty_query_peaks_no_candidates():
    query_df = _make_signed_df([("Q1", [])])
    ref_df = _make_signed_df([("R1", [30.0])])
    pairs = prefilter_pairs_by_peak_overlap(query_df, ref_df)
    assert pairs["Q1"] == set()


def test_prefilter_empty_library_peaks_excluded():
    # Library compound with no peaks cannot match any query peak.
    query_df = _make_signed_df([("Q1", [30.0])])
    ref_df = _make_signed_df([("R1", []), ("R2", [30.5])])
    pairs = prefilter_pairs_by_peak_overlap(query_df, ref_df)
    assert "R1" not in pairs["Q1"]
    assert "R2" in pairs["Q1"]


def test_prefilter_all_queries_present_in_output():
    query_df = _make_signed_df([("Q1", [30.0]), ("Q2", [80.0])])
    ref_df = _make_signed_df([("R1", [30.5])])
    pairs = prefilter_pairs_by_peak_overlap(query_df, ref_df)
    assert "Q1" in pairs
    assert "Q2" in pairs


def test_prefilter_multiple_queries_independent():
    query_df = _make_signed_df([("Q1", [30.0]), ("Q2", [80.0])])
    ref_df = _make_signed_df([("R1", [30.5]), ("R2", [80.5])])
    pairs = prefilter_pairs_by_peak_overlap(query_df, ref_df)
    assert "R1" in pairs["Q1"]
    assert "R2" not in pairs["Q1"]
    assert "R2" in pairs["Q2"]
    assert "R1" not in pairs["Q2"]


# ---------------------------------------------------------------------------
# validate_candidate_set
# ---------------------------------------------------------------------------


def test_validate_passes_when_true_positive_present():
    validate_candidate_set({"Q1_IK1": {"L1_IK1", "L2_IK2"}})


def test_validate_warns_when_true_positive_filtered(caplog):
    pairs = {"Q1_IK1": {"L2_IK2"}}  # L1_IK1 missing
    with caplog.at_level("WARNING"):
        validate_candidate_set(pairs)
    assert "Missing true match for query IDs" in caplog.text
    assert "Q1_IK1" not in pairs


def test_validate_skips_query_absent_from_library_suffix_mode(caplog):
    # Novel query key not present in candidate suffixes should emit a warning.
    pairs = {"Q1_NOVEL": {"L1_IK1"}}
    with caplog.at_level("WARNING"):
        validate_candidate_set(pairs)
    assert "Missing true match for query IDs" in caplog.text
    assert "Q1_NOVEL" not in pairs


def test_validate_accepts_show_progress_flag():
    validate_candidate_set({"Q1_IK1": {"L1_IK1"}}, show_progress=True)
