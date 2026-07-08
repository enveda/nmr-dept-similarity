"""Figure 6 — analog retrieval for out-of-library queries."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from dept_similarity.constants import RESULTS_DIR, PROCESSED_DATA_DIR

FIG_DIR = Path(PROCESSED_DATA_DIR).parent / "figures"
DPI = 400
SLATE = "#8499a4"   # DEPT-Match (matches Figs 3/4/5)
TAN   = "#debe8f"   # Gaussian   (matches Figs 3/4/5)
SAGE  = "#7d9471"   # Shift binned
BROWN = "#6B3520"
GREY  = "#8A8A8A"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})

TMAX_MEAN = 0.750         # analog-band ceiling (mean Tmax, 0.70<=Tmax<0.85)

m = pd.read_parquet(Path(RESULTS_DIR) / "analog" / "metrics.parquet")
METHOD_STYLE = [("DEPT-Match", SLATE), ("Gaussian", TAN), ("Shift binned", SAGE)]
KS = (1, 3, 10)


def wilson_err(pct, n, z=1.96):
    """Asymmetric Wilson 95% CI half-widths (in percentage points) for a proportion."""
    p = pct / 100.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (p - (c - hw)), 100 * ((c + hw) - p)

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

# --- Panel a: analog@k (a top-k hit reaches Tanimoto >= 0.6) ---
ax = axes[0]
x = np.arange(len(KS)); w = 0.26
for i, (label, c) in enumerate(METHOD_STYLE):
    rows = [m[(m.method == label) & (m.k == k)].iloc[0] for k in KS]
    vals = [r["analog_pct"] for r in rows]
    errs = np.array([wilson_err(r["analog_pct"], r["n"]) for r in rows]).T  # (2, len(KS))
    b = ax.bar(x + (i - 1) * w, vals, w, color=c, label=label,
               yerr=errs, capsize=2.5, error_kw={"lw": 1, "ecolor": "#555555"})
    ax.bar_label(b, fmt="%.1f%%", fontsize=9, padding=8)
ax.set_xticks(x); ax.set_xticklabels([f"k = {k}" for k in KS])
ax.set_ylabel("Analog recovery\n(top-k hit with Tanimoto ≥ 0.6)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
ax.set_ylim(0, 36)
ax.grid(axis="y", ls=":", color=GREY, alpha=0.4)
ax.set_axisbelow(True)
ax.set_title("Recovery of a genuine structural analog", fontsize=15, loc="center", pad=18)
ax.text(-0.13, 1.10, "a)", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")
ax.legend(frameon=False, fontsize=10, loc="upper left")

# --- Panel b: mean best query->top-k Tanimoto vs the best-available-analog ceiling ---
ax = axes[1]
for label, c in METHOD_STYLE:
    vals = [m[(m.method == label) & (m.k == k)]["mean_best_tanimoto"].iloc[0] for k in KS]
    ax.plot(KS, vals, "-o", color=c, lw=2, ms=7, label=label)
ax.axhline(TMAX_MEAN, color=BROWN, ls=":", lw=1.5)
ax.text(10, TMAX_MEAN + 0.012, f"best available analog (mean Tmax = {TMAX_MEAN:.2f})",
        ha="right", color=BROWN, fontsize=9.5)
ax.set_xticks(KS); ax.set_xticklabels([f"k = {k}" for k in KS])
ax.set_ylabel("Mean best query→top-k\nTanimoto")
ax.set_ylim(0, 0.8)
ax.locator_params(axis="y", nbins=5)
ax.grid(axis="y", ls=":", color=GREY, alpha=0.4)
ax.set_axisbelow(True)
ax.set_title("Structural quality of retrieved hits", fontsize=15, loc="center", pad=18)
ax.text(-0.13, 1.10, "b)", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")
ax.legend(frameon=False, fontsize=10, loc="lower right")

fig.tight_layout()
fig.savefig(FIG_DIR / "figure6_analog_retrieval.png", dpi=DPI, bbox_inches="tight")
print("saved", FIG_DIR / "figure6_analog_retrieval.png")
