"""Fig 6 — Forest plot of GLMM standardized coefficients.

Refits a SINGLE GLMM (no CV) per event type to extract fe_mean + fe_sd
from the variational Bayes approximation (BinomialBayesMixedGLM.fit_vb).
These are posterior mean and posterior SD of the fixed effects.

Both full (40-feature) and parsimonious (8-feature) models are shown.

Outputs
-------
  results/figures/fig6_forest_plot.{pdf,png}

Run
---
.venv/bin/python3 experiments/fig_forest_plot.py
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from beatlabile.config import CFG, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIG_DIR   = RESULTS_DIR / "figures"
ACT1_DIR  = RESULTS_DIR / "act1"
CACHE_DIR = RESULTS_DIR / "cache"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Event types to show (not variability — physiological interpretation unclear)
PANEL_ETYPES = ["hypotension", "hypertension"]
PANEL_TITLES = {
    "hypotension": "Hipotensión intraoperatoria",
    "hypertension": "Hipertensión intraoperatoria",
}

N_TOP = 15        # top N features from full model by |coef|
ALPHA = 0.05      # CI level (1.96 σ)


# ---------------------------------------------------------------------------
# Load or refit GLMM to extract posterior mean + SD
# ---------------------------------------------------------------------------

def _load_or_refit_glmm_with_se(etype: str, feature_cols: list[str],
                                  X: pd.DataFrame, y: np.ndarray,
                                  patient_ids: np.ndarray,
                                  pars: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (fe_mean, fe_sd, feature_names) from GLMM variational Bayes fit.
    
    Loads the existing pkl and extracts posterior moments from the stored
    _statsmodels_result object.
    """
    label = "parsimonious" if pars else "full"
    pkl_path = ACT1_DIR / (f"glmm_parsimonious_{etype}.pkl" if pars else f"glmm_{etype}.pkl")

    try:
        with open(pkl_path, "rb") as fh:
            glmm = pickle.load(fh)

        result = glmm._statsmodels_result
        param_names = result.model.fep_names
        fe_mean = result.fe_mean
        fe_sd   = result.fe_sd

        # Filter to feature_cols only (exclude Intercept)
        pairs = [(n, m, s) for n, m, s in zip(param_names, fe_mean, fe_sd)
                 if n != "Intercept" and n in feature_cols]
        if not pairs:
            raise ValueError("No matching features found in pkl result.")

        names  = [p[0] for p in pairs]
        means  = np.array([p[1] for p in pairs])
        sds    = np.array([p[2] for p in pairs])
        logger.info("  Loaded %s %s GLMM from pkl: %d features", etype, label, len(names))
        return means, sds, names

    except Exception as e:
        logger.warning("  Could not load pkl (%s), falling back to CSV (no SE): %s", pkl_path.name, e)
        # Fallback: use CSV (no SE available)
        csv_path = ACT1_DIR / (f"glmm_pars_coef_{etype}.csv" if pars else f"glmm_coef_{etype}.csv")
        df = pd.read_csv(csv_path)
        names = df["feature"].tolist()
        means = df["coef_raw"].values
        sds   = np.zeros(len(means))  # no SE available
        return means, sds, names


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _draw_forest_panel(ax: plt.Axes, fe_mean: np.ndarray, fe_sd: np.ndarray,
                        feature_names: list[str], title: str,
                        n_top: int, has_se: bool) -> None:
    """Draw a horizontal bar forest plot on ax."""
    # Sort by absolute magnitude, take top n_top
    order   = np.argsort(np.abs(fe_mean))[::-1][:n_top]
    means   = fe_mean[order]
    sds     = fe_sd[order]
    names   = [feature_names[i] for i in order]

    # Reverse so largest is top
    means   = means[::-1]
    sds     = sds[::-1]
    names   = names[::-1]

    n = len(names)
    ys = np.arange(n)

    z = 1.96
    lo  = means - z * sds
    hi  = means + z * sds

    colors = ["#c0392b" if m > 0 else "#2980b9" for m in means]

    if has_se:
        ax.barh(ys, means, xerr=np.vstack([means - lo, hi - means]),
                color=colors, alpha=0.78, edgecolor="none",
                error_kw={"lw": 1.4, "capsize": 3, "ecolor": "#555"})
    else:
        ax.barh(ys, means, color=colors, alpha=0.78, edgecolor="none")

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Coeficiente posterior (escala estandarizada)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)

    ci_note = "(media ± 1.96·SD posterior)" if has_se else "(media posterior — SD no disponible)"
    ax.text(0.98, 0.02, ci_note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#555",
            style="italic")


def run_fig6() -> None:
    from beatlabile.config import CFG as _CFG
    from experiments.pipeline import get_feature_cols, EVENT_TYPES
    from beatlabile.models.mixed_logistic import MixedLogisticModel

    logger.info("=== Generating Fig 6: Forest plot ===")

    # Load cached windows to have feature_cols and patient_ids
    cache_file = CACHE_DIR / "clinic_windows.parquet"
    if not cache_file.exists():
        logger.error("Cache not found. Run act1_clinic.py first.")
        return
    windows_df = pd.read_parquet(cache_file)
    logger.info("Loaded cache: %d windows", len(windows_df))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Fig. 6 — Coeficientes GLMM: métricas autonómicas de PA beat-by-beat\n"
        "Modelo parsimonioso (8 features) — Cohorte Clínic UCIQ",
        fontsize=12, fontweight="bold", y=1.02
    )

    for ax, etype in zip(axes, PANEL_ETYPES):
        sub = windows_df[windows_df["event_type"] == etype].copy()
        feat_cols   = get_feature_cols(sub)
        X           = sub[feat_cols]
        y           = sub["label"].values
        patient_ids = sub["patient_id"].values

        # --- Parsimonious model (primary display) ---
        pars_features_all = {
            "hypotension": [
                "std_pa_mean", "cv_pa_std", "rsa_max", "arv_std",
                "brs_min", "arv_mean", "std_pa_max", "rsa_mean",
            ],
            "hypertension": [
                "std_pa_std", "std_pa_max", "cv_pa_std", "std_pa_slope",
                "arv_std", "cv_pa_mean", "brs_min", "sdnn_mean",
            ],
        }
        pars_feat_cols = [c for c in pars_features_all.get(etype, []) if c in feat_cols]

        fe_mean, fe_sd, feat_names = _load_or_refit_glmm_with_se(
            etype, pars_feat_cols, X, y, patient_ids, pars=True
        )
        has_se = np.any(fe_sd > 0)

        # Show all parsimonious features
        _draw_forest_panel(
            ax, fe_mean, fe_sd, feat_names,
            title=PANEL_TITLES[etype],
            n_top=len(feat_names),
            has_se=has_se,
        )

    # Legend
    red_patch  = mpatches.Patch(color="#c0392b", alpha=0.78, label="Coef > 0: ↑ riesgo")
    blue_patch = mpatches.Patch(color="#2980b9", alpha=0.78, label="Coef < 0: ↓ riesgo")
    fig.legend(handles=[red_patch, blue_patch],
               loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out = FIG_DIR / f"fig6_forest_plot.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", out)

    plt.close(fig)
    logger.info("=== Fig 6 DONE ===")


if __name__ == "__main__":
    run_fig6()
