"""domain_shift_analysis.py
Quantify distributional shift between Clínic, VitalDB, and MIMIC cohorts.

Metrics per feature (40 features × 3 cohort pairs):
  - KS statistic + p-value (Kolmogorov–Smirnov, two-sample)
  - Wasserstein distance (earth mover's distance, normalised by pooled SD)

Outputs:
  results/act4/domain_shift_ks.csv           (KS + Wasserstein per feature × pair)
  results/figures/fig_domain_shift_ks.pdf/png  (heatmap KS statistic)
  results/figures/fig_domain_shift_density.pdf/png  (density overlay for parsimonious features)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
CACHE   = ROOT / "results" / "cache"
OUT_DIR = ROOT / "results" / "act4"
FIG_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Parsimonious features reported in the manuscript
PARS_HYPO = ["std_pa_mean", "cv_pa_std", "rsa_max", "arv_std",
             "brs_min", "arv_mean", "std_pa_max", "rsa_mean"]
PARS_HYPER = ["std_pa_std", "std_pa_max", "cv_pa_std", "std_pa_slope",
              "arv_std", "cv_pa_mean", "brs_min", "sdnn_mean"]
PARS_ALL = sorted(set(PARS_HYPO + PARS_HYPER))

META_COLS = {"patient_id", "event_type", "label",
             "window_start_s", "window_end_s", "event_onset_s"}

PAIR_LABELS = {
    "clinic_vs_vitaldb": ("Clínic", "VitalDB"),
    "clinic_vs_mimic":   ("Clínic", "MIMIC"),
    "vitaldb_vs_mimic":  ("VitalDB", "MIMIC"),
}


def load_control_windows(cohort: str) -> pd.DataFrame:
    """Load windows, keep only control rows (label=0) for fair comparison."""
    df = pd.read_parquet(CACHE / f"{cohort}_windows.parquet")
    # Use hypotension frame as canonical, control rows only — one row per original sample
    sub = df[df["event_type"] == "hypotension"].copy()
    # Deduplicate by patient_id × window_start to avoid counting same window twice
    sub = sub.drop_duplicates(subset=["patient_id", "window_start_s"])
    return sub


def compute_shift(a: np.ndarray, b: np.ndarray) -> dict:
    """KS test + normalised Wasserstein for two 1-D samples."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return {"ks_stat": np.nan, "ks_pval": np.nan, "wasserstein_norm": np.nan}

    ks_stat, ks_pval = stats.ks_2samp(a, b)

    # Normalise Wasserstein by pooled SD (≈ Cohen's d scale)
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    wass = wasserstein_distance(a, b)
    wass_norm = wass / pooled_sd if pooled_sd > 0 else np.nan

    return {"ks_stat": round(ks_stat, 4),
            "ks_pval": round(ks_pval, 6),
            "wasserstein_norm": round(wass_norm, 4)}


def run_shift_analysis(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute shift metrics for all features × cohort pairs."""
    feat_cols = [c for c in dfs["clinic"].columns if c not in META_COLS]
    rows = []
    for pair_key, (name_a, name_b) in PAIR_LABELS.items():
        key_a = name_a.lower().replace("í", "i")   # clinic
        key_b = name_b.lower()                       # vitaldb / mimic
        df_a = dfs[key_a]
        df_b = dfs[key_b]
        for feat in feat_cols:
            if feat not in df_a.columns or feat not in df_b.columns:
                continue
            metrics = compute_shift(df_a[feat].values, df_b[feat].values)
            rows.append({"pair": pair_key, "feature": feat, **metrics,
                         "pars": feat in PARS_ALL})
    return pd.DataFrame(rows)


def plot_heatmap(shift_df: pd.DataFrame) -> None:
    """KS statistic heatmap: features × cohort pairs."""
    pivot = shift_df.pivot(index="feature", columns="pair", values="ks_stat")
    # Sort by mean KS (most shifted first)
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")

    pars_mask = [f in PARS_ALL for f in pivot.index]

    fig, ax = plt.subplots(figsize=(7, 11))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=0.6)
    plt.colorbar(im, ax=ax, label="KS statistic")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [c.replace("_vs_", "\nvs\n").replace("clinic", "Clínic")
           .replace("vitaldb","VitalDB").replace("mimic","MIMIC")
         for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    yticklabels = []
    for feat, is_pars in zip(pivot.index, pars_mask):
        yticklabels.append(f"{'★ ' if is_pars else '  '}{feat}")
    ax.set_yticklabels(yticklabels, fontsize=7.5,
                       fontweight="bold" if False else "normal")
    # Bold parsimonious
    for tick, is_pars in zip(ax.get_yticklabels(), pars_mask):
        if is_pars:
            tick.set_fontweight("bold")

    ax.set_title("Feature Distribution Shift (KS statistic)\n★ = parsimonious features",
                 fontsize=10)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"fig_domain_shift_ks.{ext}"
        fig.savefig(p, dpi=180 if ext == "png" else None, bbox_inches="tight")
        logger.info("Saved: %s", p)
    plt.close(fig)


def plot_density(dfs: dict[str, pd.DataFrame]) -> None:
    """Density overlay for parsimonious features across three cohorts."""
    features = PARS_ALL
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))

    colors = {"clinic": "#1f77b4", "vitaldb": "#ff7f0e", "mimic": "#2ca02c"}
    labels = {"clinic": "Clínic", "vitaldb": "VitalDB", "mimic": "MIMIC"}

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.5))
    axes = axes.flatten()

    for i, feat in enumerate(sorted(features)):
        ax = axes[i]
        in_pars_hypo  = feat in PARS_HYPO
        in_pars_hyper = feat in PARS_HYPER
        title_tag = ""
        if in_pars_hypo and in_pars_hyper:
            title_tag = " [H+T]"
        elif in_pars_hypo:
            title_tag = " [H]"
        else:
            title_tag = " [T]"

        for cohort, df in dfs.items():
            if feat not in df.columns:
                continue
            vals = df[feat].dropna().values
            if len(vals) < 5:
                continue
            # Clip at 1/99 percentile for plotting
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            vals_clip = vals[(vals >= lo) & (vals <= hi)]
            ax.hist(vals_clip, bins=40, density=True, alpha=0.45,
                    color=colors[cohort], label=labels[cohort])

        ax.set_title(f"{feat}{title_tag}", fontsize=7.5, fontweight="bold")
        ax.set_yticks([])
        ax.tick_params(labelsize=6.5)

    # Legend on first axis
    axes[0].legend(fontsize=7, loc="upper right")

    # Hide unused subplots
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    # Add [H]=hypotension [T]=hypertension note
    fig.text(0.5, 0.01,
             "[H] = parsimonious hypotension feature  |  [T] = parsimonious hypertension feature  "
             "|  [H+T] = both",
             ha="center", fontsize=8)

    plt.suptitle("Feature Distributions: Clínic vs VitalDB vs MIMIC (control windows)",
                 fontsize=10, y=1.01)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        p = FIG_DIR / f"fig_domain_shift_density.{ext}"
        fig.savefig(p, dpi=150 if ext == "png" else None, bbox_inches="tight")
        logger.info("Saved: %s", p)
    plt.close(fig)


def main() -> None:
    logger.info("=== Domain Shift Analysis ===")

    dfs = {
        "clinic":  load_control_windows("clinic"),
        "vitaldb": load_control_windows("vitaldb"),
        "mimic":   load_control_windows("mimic"),
    }

    for name, df in dfs.items():
        logger.info("  %s: %d windows, %d patients", name, len(df),
                    df["patient_id"].nunique())

    shift_df = run_shift_analysis(dfs)

    out_csv = OUT_DIR / "domain_shift_ks.csv"
    shift_df.to_csv(out_csv, index=False)
    logger.info("Saved: %s", out_csv)

    # Summary: parsimonious features shift
    logger.info("\n--- KS summary (parsimonious features, Clínic vs VitalDB) ---")
    pars_cv = shift_df[(shift_df["pars"]) & (shift_df["pair"] == "clinic_vs_vitaldb")]
    for _, r in pars_cv.sort_values("ks_stat", ascending=False).iterrows():
        sig = "***" if r["ks_pval"] < 0.001 else ("**" if r["ks_pval"] < 0.01 else "*" if r["ks_pval"] < 0.05 else "")
        logger.info("  %-18s  KS=%.3f  W_norm=%.3f  %s",
                    r["feature"], r["ks_stat"], r["wasserstein_norm"], sig)

    logger.info("\n--- KS summary (parsimonious features, Clínic vs MIMIC) ---")
    pars_cm = shift_df[(shift_df["pars"]) & (shift_df["pair"] == "clinic_vs_mimic")]
    for _, r in pars_cm.sort_values("ks_stat", ascending=False).iterrows():
        sig = "***" if r["ks_pval"] < 0.001 else ("**" if r["ks_pval"] < 0.01 else "*" if r["ks_pval"] < 0.05 else "")
        logger.info("  %-18s  KS=%.3f  W_norm=%.3f  %s",
                    r["feature"], r["ks_stat"], r["wasserstein_norm"], sig)

    plot_heatmap(shift_df)
    plot_density(dfs)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
