"""fig_dca.py
Decision Curve Analysis figure from pre-computed CSV files.

Layout: 1×2 panels (hypotension + hypertension).
Each panel shows two curves: Clínic development (act1) and MIMIC validation (act2).
Lines: net_benefit_model (solid), net benefit treat-all (dashed), treat-none (dotted zero).
Shaded region: range where model > treat-all AND model > 0.

Inputs  : results/act1/dca_glmm_{etype}.csv
          results/act2/dca_glmm_{etype}.csv
          (columns: threshold, net_benefit_model, net_benefit_all, net_benefit_none)
Outputs : results/figures/fig_dca.pdf
          results/figures/fig_dca.png
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
ACT1    = ROOT / "results" / "act1"
ACT2    = ROOT / "results" / "act2"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ETYPES = {
    "hypotension":  "Hypotension",
    "hypertension": "Hypertension",
}

# Colour palette
CLR_ACT1 = "#1f77b4"   # blue  — Clínic development
CLR_ACT2 = "#d62728"   # red   — MIMIC external validation


def load_dca(act_dir: Path, etype: str) -> pd.DataFrame | None:
    path = act_dir / f"dca_glmm_{etype}.csv"
    if not path.exists():
        logger.warning("Missing: %s", path)
        return None
    return pd.read_csv(path)


def plot_panel(ax: plt.Axes, df1: pd.DataFrame | None, df2: pd.DataFrame | None,
               title: str, threshold_range: tuple[float, float] = (0.0, 0.5)) -> None:
    """Draw one DCA panel."""
    lo, hi = threshold_range

    def shade_benefit(df: pd.DataFrame, color: str) -> None:
        """Shade threshold range where model > treat-all and > 0."""
        if df is None:
            return
        mask = (df["threshold"] >= lo) & (df["threshold"] <= hi)
        sub = df[mask].copy()
        good = (sub["net_benefit_model"] > sub["net_benefit_all"]) & \
               (sub["net_benefit_model"] > 0)
        if good.sum() == 0:
            return
        first = sub.loc[good.idxmax(), "threshold"]
        ax.axvspan(first, sub.loc[good][ "threshold"].max(),
                   alpha=0.08, color=color, linewidth=0)

    # Plot Clínic development (act1)
    if df1 is not None:
        mask1 = (df1["threshold"] >= lo) & (df1["threshold"] <= hi)
        sub1 = df1[mask1]
        shade_benefit(df1, CLR_ACT1)
        ax.plot(sub1["threshold"], sub1["net_benefit_model"],
                color=CLR_ACT1, lw=2, label="GLMM model – Clínic (development)")
        ax.plot(sub1["threshold"], sub1["net_benefit_all"],
                color=CLR_ACT1, lw=1.2, ls="--",
                label="Treat all – Clínic")

    # Plot MIMIC external validation (act2)
    if df2 is not None:
        mask2 = (df2["threshold"] >= lo) & (df2["threshold"] <= hi)
        sub2 = df2[mask2]
        shade_benefit(df2, CLR_ACT2)
        ax.plot(sub2["threshold"], sub2["net_benefit_model"],
                color=CLR_ACT2, lw=2, label="GLMM model – MIMIC (external)")
        ax.plot(sub2["threshold"], sub2["net_benefit_all"],
                color=CLR_ACT2, lw=1.2, ls="--",
                label="Treat all – MIMIC")

    # Treat none = 0
    ax.axhline(0, color="k", lw=0.8, ls=":", label="Treat none (NB = 0)")

    # Aesthetics
    ax.set_xlim(lo, hi)
    ymin = -0.03
    all_nb = []
    for df in (df1, df2):
        if df is not None:
            mask = (df["threshold"] >= lo) & (df["threshold"] <= hi)
            all_nb.extend(df.loc[mask, "net_benefit_model"].tolist())
            all_nb.extend(df.loc[mask, "net_benefit_all"].tolist())
    ymax = max(0.12, max(all_nb) * 1.1) if all_nb else 0.12
    ax.set_ylim(ymin, ymax)

    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("Threshold probability", fontsize=9)
    ax.set_ylabel("Net benefit", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    handles, labels_ = ax.get_legend_handles_labels()
    # Remove duplicate 'treat none' entries
    seen = {}
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=7.5, loc="upper right")


def main() -> None:
    logger.info("=== DCA Figure ===")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.35)

    for ax, (etype, label) in zip(axes, ETYPES.items()):
        df1 = load_dca(ACT1, etype)
        df2 = load_dca(ACT2, etype)
        if df1 is not None:
            logger.info("[%s] act1: %d rows, threshold %.3f–%.3f",
                        etype, len(df1), df1["threshold"].min(), df1["threshold"].max())
        if df2 is not None:
            logger.info("[%s] act2: %d rows, threshold %.3f–%.3f",
                        etype, len(df2), df2["threshold"].min(), df2["threshold"].max())

        plot_panel(ax, df1, df2, label, threshold_range=(0.05, 0.45))

    fig.suptitle("Decision Curve Analysis — GLMM haemodynamic lability prediction",
                 fontsize=10, y=1.01)

    for ext in ("pdf", "png"):
        p = FIG_DIR / f"fig_dca.{ext}"
        fig.savefig(p, dpi=180 if ext == "png" else None, bbox_inches="tight")
        logger.info("Saved: %s", p)
    plt.close(fig)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
