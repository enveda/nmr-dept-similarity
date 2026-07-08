"""Contract tests for rank_library output shape and ordering."""

import numpy as np
import pandas as pd

from dept_similarity.match import rank_library


def _make_df(records):
    return pd.DataFrame(
        [
            {
                "compound_id": cid,
                "signed_shifts": np.asarray(peaks, dtype=np.float32),
            }
            for cid, peaks in records
        ]
    )


QUERY_DF = _make_df(
    [
        ("Q1", [20.0, -45.0, 80.0]),
        ("Q2", [30.0, -60.0]),
    ]
)

LIBRARY_DF = _make_df(
    [
        ("L1", [20.0, -45.0, 80.0]),
        ("L2", [30.0, -60.0]),
        ("L3", [100.0, 150.0]),
    ]
)

PAIRS = {
    "Q1": ["L1", "L2", "L3"],
    "Q2": ["L1", "L2", "L3"],
}


def _row(df: pd.DataFrame, query_id: str, metric_name: str) -> pd.Series:
    rows = df[(df["query_id"] == query_id) & (df["metric_name"] == metric_name)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_returns_one_row_per_query_metric_with_list_columns():
    out = rank_library(
        PAIRS,
        QUERY_DF.copy(),
        LIBRARY_DF.copy(),
        method="ppm_bins",
        sim_metrics=("cosine", "euclidean"),
    )

    assert set(out.columns) == {
        "query_id",
        "metric_name",
        "candidate_id_list",
        "score_list",
    }
    assert len(out) == len(PAIRS) * 2

    assert out["candidate_id_list"].map(type).eq(list).all()
    assert out["score_list"].map(type).eq(list).all()

    for _, row in out.iterrows():
        qid = row["query_id"]
        assert len(row["candidate_id_list"]) == len(PAIRS[qid])
        assert len(row["score_list"]) == len(PAIRS[qid])


def test_similarity_metric_is_sorted_descending():
    out = rank_library(
        PAIRS,
        QUERY_DF.copy(),
        LIBRARY_DF.copy(),
        method="ppm_bins",
        sim_metrics=("cosine",),
    )

    q1 = _row(out, "Q1", "cosine")
    assert q1["candidate_id_list"][0] == "L1"
    scores = q1["score_list"]
    assert scores == sorted(scores, reverse=True)


def test_distance_metric_is_sorted_ascending():
    out = rank_library(
        PAIRS,
        QUERY_DF.copy(),
        LIBRARY_DF.copy(),
        method="ppm_bins",
        sim_metrics=("euclidean",),
    )

    q1 = _row(out, "Q1", "euclidean")
    assert q1["candidate_id_list"][0] == "L1"
    scores = q1["score_list"]
    assert scores == sorted(scores)


def test_empty_candidate_query_is_skipped():
    out = rank_library(
        {"Q1": [], "Q2": ["L1", "L2"]},
        QUERY_DF.copy(),
        LIBRARY_DF.copy(),
        method="ppm_bins",
        sim_metrics=("cosine",),
    )

    assert set(out["query_id"]) == {"Q2"}
