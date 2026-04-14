"""Fig 7 — Calibration plots: observed vs predicted probability.

Three cohorts:
  - Clínic (act1) — training set apparent calibration  
  - MIMIC (act2)  — direct transfer without refit
  - [VitalDB: no hold-out calibration CSV available from act3]

For hypotension and hypertension (1 row × 2 columns).
Annotates: calibration slope, E/O ratio, n events.

Outputs
-------
  results/figures/fig7_calibration.{pdf,png}

Run
---
.venv/bin/python3 experiments/fig_calibration.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR  = Path(__file__).parent.parent / "results" / "figures"
ACT1_DIR = Path(__file__).parent.parent / "results" / "act1"
ACT2_DIR = Path(__file__).parent.parent / "results" / "act2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ETYPES  = ["hypotension", "hypertension"]
TITLES  = {"hypotension": "Hipotensión", "hypertension": "Hipertensión"}

# Cohort specs: (label, dir, linestyle, color, marker)
COHORTS = [
    ("Clínic (desarrollo)",  ACT1_DIR, "-",  "#2c7bb6", "o"),
    ("MIMIC (transferencia directa)", ACT2_DIR, "--", "#d7191c", "s"),
]


def _hosmer_lemeshow(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Approximate HL chi² and p-value from binned calibration data."""
    # obs and pred are already bin means
    mask = pred > 0
    chi2 = float(np.sum((obs[mask] - pred[mask]) ** 2 / (pred[mask] * (1 - pred[mask] + 1e-9))))
    df   = max(len(pred) - 2, 1)
    pval = float(1 - stats.chi2.cdf(chi2, df))
    return chi2, pval


def _draw_calibration_panel(ax: plt.Axes, etype: str,
                             cal_dfs: list[tuple[str, pd.DataFrame, str, str, str]],
                             slopes: dict, citls: dict, n_events: dict) -> None:
    """Draw one calibration panel."""
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Calibración perfecta", zorder=1)
    ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.04, color="black")

    for label, df, ls, color, marker in cal_dfs:
        x = df["mean_predicted"].values
        y = df["observed_rate"].values
        n = df["count"].values

        ax.plot(x, y, ls=ls, color=color, lw=1.8,
                marker=marker, ms=5, zorder=3, label=label)

        # Confidence interval per bin (Wilson)
        for xi, yi, ni in zip(x, y, n):
            z = 1.96
            ci = z * np.sqrt(yi * (1 - yi) / max(ni, 1))
            ax.errorbar(xi, yi, yerr=ci, fmt="none", color=color,
                        alpha=0.4, elinewidth=0.8, capsize=2, zorder=2)

    # Annotation box
    lines = []
    for label, df, *_ in cal_dfs:
        cohort_key = "clinic" if "Clínic" in label else "mimic"
        slope = slopes.get(cohort_key, None)
        citl  = citls.get(cohort_key, None)
        nev   = n_events.get(cohort_key, "?")
        s_str = f"{slope:.2f}" if slope is not None else "N/A"
        c_str = f"{citl:+.3f}" if citl is not None else "N/A"
        lines.append(f"{label.split(' (')[0]}: slope={s_str}, CITL={c_str}, n={nev}")

    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Probabilidad predicha (media por decil)", fontsize=9)
    ax.set_ylabel("Tasa observada (media por decil)", fontsize=9)
    ax.set_title(TITLES[etype], fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)


def run_fig7() -> None:
    logger.info("=== Generating Fig 7: Calibration plots ===")

    # Load act2 calibration metrics (slope, CITL)
    with open(ACT2_DIR.parent / "act2" / "act2_results.json") as fh:
        act2 = json.load(fh)
    with open(ACT1_DIR / "act1_results.json") as fh:
        act1 = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Fig. 7 — Calibración del modelo GLMM: probabilidad predicha vs observada\n"
        "Hipotensión e hipertensión intraoperatoria",
        fontsize=11, fontweight="bold", y=1.01
    )

    for ax, etype in zip(axes, ETYPES):
        cal_dfs = []
        for label, src_dir, ls, color, marker in COHORTS:
            csv_path = src_dir / f"calibration_glmm_{etype}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                cal_dfs.append((label, df, ls, color, marker))
                logger.info("  Loaded %s %s calibration: %d bins", label, etype, len(df))
            else:
                logger.warning("  Missing: %s", csv_path)

        # Collect slopes and CITL
        slopes  = {}
        citls   = {}
        n_events = {}

        # Act1 (Clínic): apparent calibration; slope not saved — compute from CSV
        calib_path = ACT1_DIR / f"calibration_glmm_{etype}.csv"
        if calib_path.exists():
            df1 = pd.read_csv(calib_path)
            # Approximate slope: regress observed_rate on mean_predicted
            from sklearn.linear_model import LinearRegression
            if len(df1) >= 3:
                lr = LinearRegression().fit(
                    df1["mean_predicted"].values.reshape(-1,1),
                    df1["observed_rate"].values
                )
                slopes["clinic"] = float(lr.coef_[0])
                citls["clinic"]  = float(df1["mean_predicted"].mean() - df1["observed_rate"].mean())
        if etype in act1:
            n_events["clinic"] = act1[etype].get("n_events", "?")

        # Act2 (MIMIC): slope and CITL from act2_results.json
        if etype in act2:
            slopes["mimic"]  = act2[etype].get("glmm_cal_slope", None)
            citls["mimic"]   = act2[etype].get("glmm_citl", None)
            n_events["mimic"] = act2[etype].get("n_events", "?")

        _draw_calibration_panel(ax, etype, cal_dfs, slopes, citls, n_events)

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out = FIG_DIR / f"fig7_calibration.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", out)

    plt.close(fig)
    logger.info("=== Fig 7 DONE ===")


if __name__ == "__main__":
    run_fig7()
