"""Tie-aware (optimistic) Hits@k vs the standard stable-tie-break Hits@k — Figure 2 variant.

The ranked lists are built with a *stable* argsort, so within a block of candidates tied at the
same score the one inserted first (arbitrary) wins. If the true compound is tied at the top but
ordered behind a wrong candidate, the standard Hits@k misses it.

This script recomputes the full candidate score vector for the three cosine methods and reports,
alongside the standard metric, an **optimistic tie-aware** Hits@k: a query counts as a hit@k iff
fewer than k candidates *strictly* outscore its best true-label candidate — i.e. some tie-break
consistent with the scores would place the true compound within the top k. Two scores within
``TIE_ATOL`` are treated as tied (same definition as the tie-distribution figure).

Writes ``data/figures/figure2.png`` (bars = standard Hits@k = committed values, with a whisker
bracketing the pessimistic..optimistic tie-break range) and prints the full table.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from dept_similarity.constants import FIGURE_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
# Reuse the exact data loading + scoring paths (module import does not run its main()).
from plot_tie_distribution import build_inputs, vector_full_scores, hungarian_full_scores, TIE_ATOL

K_VALUES = (1, 3, 10)
# The three cosine methods shown in Figure 2, in the original display order.
METHODS = [
    ("hungarian_match", "DEPT-Match", "#6B3520"),
    ("gauss_kernel", "Gaussian", "#6B3520"),
    ("ppm_bins_typed", "Shift binned", "#6B3520"),
]
K_COLOR = {1: "#F0C98A", 3: "#D89168", 10: "#6B3520"}


def load_labels():
    """Query true InChIKey14 (keyed by the suffixed compound_id used everywhere else) and the
    candidate InChIKey14 lookup, matching get_metrics_at_k (full InChIKey truncated to 14)."""
    q = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "experimental_validation_set.parquet")
    if not (q["compound_id"].str.split("_").str[-1] == q["inchikey14"]).all():
        q["compound_id"] = q["compound_id"] + "_" + q["inchikey14"]
    q_true_key = q.set_index("compound_id")["inchikey14"].to_dict()
    n_total = q["compound_id"].nunique()

    lib = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "library_data.parquet", columns=["compound_id", "inchikey"])
    cand_key = {c: (ik[:14] if isinstance(ik, str) else None)
                for c, ik in zip(lib["compound_id"], lib["inchikey"])}
    return q_true_key, cand_key, n_total


def method_hits(per_query_scores, pairs, cand_key, q_true_key, n_total):
    """Hit counts per k under three tie-breaking regimes.

    standard   — stable descending sort keeps candidate insertion order for ties (as in the
                 committed ranked lists).
    optimistic — best case: the true compound is ordered first within its tie block. Hit@k iff
                 fewer than k candidates *strictly* outscore the best true-label match (g < k).
    pessimistic— worst case: every other candidate tied at the true score is ordered ahead of it,
                 so the earliest true match sits at rank g + (non-true ties) + 1. Hit@k iff that
                 rank <= k.
    """
    std = {k: 0 for k in K_VALUES}
    opt = {k: 0 for k in K_VALUES}
    pess = {k: 0 for k in K_VALUES}
    for q, sc in per_query_scores.items():
        cand = pairs[q]
        tk = q_true_key.get(q)
        if tk is None or len(cand) == 0:
            continue
        keys = np.array([cand_key.get(c) for c in cand], dtype=object)
        sc = np.asarray(sc, dtype=float)

        order = np.argsort(-sc, kind="stable")
        for k in K_VALUES:
            if tk in set(keys[order[:k]]):
                std[k] += 1

        is_true = keys == tk
        if is_true.any():
            best_true = sc[is_true].max()
            g = int(np.sum(sc > best_true + TIE_ATOL))                       # strictly better
            tied = np.abs(sc - best_true) <= TIE_ATOL                        # tied at true score
            non_true_tied = int(np.sum(tied & ~is_true))
            worst_rank = g + non_true_tied + 1                               # earliest true, worst case
            for k in K_VALUES:
                if g < k:
                    opt[k] += 1
                if worst_rank <= k:
                    pess[k] += 1
    to_pct = lambda d: {k: 100.0 * v / n_total for k, v in d.items()}
    return to_pct(pess), to_pct(std), to_pct(opt)


def validate_standard(std_pct, method_dir, n_total):
    """Cross-check the recomputed standard Hits@k against the committed ranked lists."""
    from dept_similarity.match import get_metrics_at_k
    q = pd.read_parquet(Path(PROCESSED_DATA_DIR) / "experimental_validation_set.parquet")
    if not (q["compound_id"].str.split("_").str[-1] == q["inchikey14"]).all():
        q["compound_id"] = q["compound_id"] + "_" + q["inchikey14"]
    ranked = pd.read_parquet(Path(RESULTS_DIR) / f"{method_dir}_cosine" / "ranked_top10.parquet")
    m = get_metrics_at_k(ranked, query_spectra_df=q, k_values=K_VALUES,
                         with_inchikey14=True, compute_tanimoto=False, n_total_queries=n_total)
    committed = {int(k): float(a) for k, a in zip(m["k"], m["accuracy"])}
    diffs = {k: round(std_pct[k] - committed[k], 2) for k in K_VALUES}
    print(f"    committed {committed}  recompute-Δ {diffs}")


RESULTS_CACHE = Path("../data/cache/tie_aware_results.pkl")


def compute_results():
    d = build_inputs()
    pairs, qp, lp = d["pairs"], d["query_peaks"], d["lib_peaks"]
    query_ids = list(pairs.keys())
    q_true_key, cand_key, n_total = load_labels()
    print(f"{len(query_ids)} queries, n_total={n_total}", flush=True)

    OPT = Path(RESULTS_DIR) / "optimization"
    gauss_sigma = float(pd.read_parquet(OPT / "gaussian" / "summary_hits_at_k.parquet")
                        .sort_values("Hits@10 (%)").iloc[-1]["sigma"])
    hung_tol = float(pd.read_parquet(OPT / "hungarian" / "summary_hits_at_k.parquet")
                     .sort_values("Hits@10 cosine (%)").iloc[-1]["ppm_tolerance"])

    scores = {}
    scores["gauss_kernel"] = vector_full_scores(query_ids, qp, pairs, lp, "gauss_kernel", gauss_sigma)
    print("gauss_kernel scored", flush=True)
    scores["ppm_bins_typed"] = vector_full_scores(query_ids, qp, pairs, lp, "ppm_bins_typed", gauss_sigma)
    print("ppm_bins_typed scored", flush=True)
    scores["hungarian_match"] = hungarian_full_scores(query_ids, qp, pairs, lp, hung_tol)["cosine"]
    print("hungarian_match scored", flush=True)

    results = {}  # method_dir -> (pess_pct, std_pct, opt_pct)
    print("\nHits@k (%):  method            k   pessim.  standard  optimistic   [range]")
    for method_dir, disp, _ in METHODS:
        pess, std, opt = method_hits(scores[method_dir], pairs, cand_key, q_true_key, n_total)
        results[method_dir] = (pess, std, opt)
        for k in K_VALUES:
            print(f"             {disp:<14} {k:>4}  {pess[k]:7.1f}  {std[k]:8.1f}  {opt[k]:10.1f}   "
                  f"[{pess[k]:.1f}, {opt[k]:.1f}]  (std{std[k]-pess[k]:+.1f}/{opt[k]-std[k]:+.1f})")
        validate_standard(std, method_dir, n_total)
    with open(RESULTS_CACHE, "wb") as fh:
        pickle.dump(results, fh)
    return results


def main():
    if RESULTS_CACHE.exists():
        with open(RESULTS_CACHE, "rb") as fh:
            results = pickle.load(fh)
        print("loaded cached results (delete data/cache/tie_aware_results.pkl to recompute)")
    else:
        results = compute_results()

    # Export the numbers so notebook 04 can render Figure 2 without the heavy recompute.
    res_rows = [{"method": md, "display": disp, "k": k,
                 "pessimistic": results[md][0][k], "standard": results[md][1][k],
                 "optimistic": results[md][2][k]}
                for md, disp, _ in METHODS for k in K_VALUES]
    out_parquet = Path(RESULTS_DIR) / "tie_aware_hits.parquet"
    pd.DataFrame(res_rows).to_parquet(out_parquet, index=False)
    print(f"wrote {out_parquet}")

    # ---- Figure 2: standard bars with a [pessimistic, optimistic] tie-breaking whisker ----
    # Bar height = standard (stable tie-break, = committed Figure 2). The whisker brackets the
    # range the score allows under any tie-break: lower cap = pessimistic, upper cap = optimistic.
    # Gaussian has no ties, so its range collapses to the bar (no whisker) — that is the point.
    labels = [disp for _, disp, _ in METHODS]
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for i, k in enumerate(K_VALUES):
        pess_vals = np.array([results[m][0][k] for m, _, _ in METHODS])
        std_vals = np.array([results[m][1][k] for m, _, _ in METHODS])
        opt_vals = np.array([results[m][2][k] for m, _, _ in METHODS])
        xpos = x + i * width
        # Solid = guaranteed hits under any tie-break (pessimistic bound).
        ax.bar(xpos, pess_vals, width, color=K_COLOR[k], label=f"k = {k}", zorder=2)
        # Hatched cap = tie-dependent zone up to the optimistic bound.
        ax.bar(xpos, opt_vals - pess_vals, width, bottom=pess_vals, color=K_COLOR[k],
               alpha=0.38, hatch="///", edgecolor="white", linewidth=0, zorder=2)
        for xp, pv, sv, ov in zip(xpos, pess_vals, std_vals, opt_vals):
            # Dark tick marks the standard (stable tie-break) value = committed Hits@k.
            ax.plot([xp - width / 2, xp + width / 2], [sv, sv], color="#1a1a1a", lw=2.0, zorder=4)
            ax.text(xp, sv, f"{sv:.1f}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#1a1a1a", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
            if ov - pv >= 0.3:  # ties create a range → annotate [pess–opt] above the cap
                ax.text(xp, ov + 0.6, f"[{pv:.1f}–{ov:.1f}]", ha="center", va="bottom",
                        fontsize=8.5, color="#444", zorder=4)

    all_opt = [results[m][2][k] for m, _, _ in METHODS for k in K_VALUES]
    all_pess = [results[m][0][k] for m, _, _ in METHODS for k in K_VALUES]
    ax.set_ylim(np.floor((min(all_pess) - 4) / 5) * 5, np.ceil((max(all_opt) + 5) / 5) * 5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylabel("Hits@k", fontsize=17)
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.spines[["top", "right"]].set_visible(False)
    leg_k = ax.legend(title="top-k", frameon=False, fontsize=13, title_fontsize=13, loc="upper left")
    ax.add_artist(leg_k)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([0], [0], color="#1a1a1a", lw=2.0, label="standard (stable tie-break)"),
                       Patch(facecolor="#8a8a8a", alpha=0.38, hatch="///", edgecolor="white",
                             label="tie-dependent zone\n(solid = guaranteed, top = optimistic)")],
              frameon=False, fontsize=10.5, loc="upper right")
    fig.tight_layout()
    out = Path(FIGURE_DIR) / "figure2.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
