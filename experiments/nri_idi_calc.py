"""NRI/IDI calculation: GLMM vs conventional model (PAM+FC).

Computes:
  - Referencia: logistic regression on [map_mean, hr_mean]  
  - Incremento: GLMM full model
  - Incremento parsimonioso: GLMM parsimonious (8 features)

Reports continuous NRI and IDI for hypotension (primary).

Outputs
-------
  results/act1/nri_idi_results.json

Run
---
.venv/bin/python3 experiments/nri_idi_calc.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from beatlabile.config import RESULTS_DIR
from beatlabile.models.mixed_logistic import compute_nri_idi
from experiments.pipeline import get_feature_cols, EVENT_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR   = RESULTS_DIR / "act1"
CACHE_DIR = RESULTS_DIR / "cache"

# Reference: simple PA variability metrics available on any standard monitor
# (mean SD of PA and mean CV of PA over the window).
# These represent the "minimum interpretable baseline" from a pressure waveform.
# The GLMM adds BRS, RSA, ARV — autonomic coupling metrics beyond simple variability.
CONVENTIONAL_COLS = ["std_pa_mean", "cv_pa_mean"]  # PA variability only

# Event types to compute (primary: hypotension; secondary: hypertension)
TARGET_ETYPES = ["hypotension", "hypertension"]


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _fit_conventional_lr(X: pd.DataFrame, y: np.ndarray,
                          conv_cols: list[str]) -> np.ndarray:
    """Fit simple logistic regression on conventional hemodynamic features."""
    X_conv = X[conv_cols].fillna(X[conv_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_conv)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    p_old = lr.predict_proba(X_scaled)[:, 1]
    logger.info("  Conventional LR AUC (train): computing...")
    from sklearn.metrics import roc_auc_score
    logger.info("  Conv LR train AUC: %.3f", roc_auc_score(y, p_old))
    return p_old


def run_nri_idi() -> None:
    # ------------------------------------------------------------------ #
    # Load cache
    # ------------------------------------------------------------------ #
    cache_file = CACHE_DIR / "clinic_windows.parquet"
    if not cache_file.exists():
        logger.error("Cache not found. Run act1_clinic.py first.")
        return
    windows_df = pd.read_parquet(cache_file)
    logger.info("Loaded cache: %d windows", len(windows_df))

    nri_idi_results = {}

    for etype in TARGET_ETYPES:
        logger.info("=== %s ===", etype)
        sub = windows_df[windows_df["event_type"] == etype].copy()
        feat_cols  = get_feature_cols(sub)
        X          = sub[feat_cols]
        y          = sub["label"].values

        n_events   = int(np.sum(y))
        n_ctrl     = int(np.sum(y == 0))
        logger.info("  n=%d  events=%d  controls=%d", len(y), n_events, n_ctrl)

        conv_cols = [c for c in CONVENTIONAL_COLS if c in feat_cols]
        if not conv_cols:
            logger.warning("  Conventional cols not found in features, skipping.")
            continue

        # Reference: conventional hemodynamic model
        p_old = _fit_conventional_lr(X, y, conv_cols)

        # Load full GLMM predictions
        try:
            with open(OUT_DIR / f"glmm_{etype}.pkl", "rb") as fh:
                glmm_full = pickle.load(fh)
            p_full = glmm_full.predict_proba(X)
        except Exception as e:
            logger.error("  Could not load full GLMM pkl: %s", e)
            p_full = None

        # Load parsimonious GLMM predictions
        try:
            with open(OUT_DIR / f"glmm_parsimonious_{etype}.pkl", "rb") as fh:
                glmm_pars = pickle.load(fh)
            p_pars = glmm_pars.predict_proba(X)
        except Exception as e:
            logger.error("  Could not load parsimonious GLMM pkl: %s", e)
            p_pars = None

        etype_results = {
            "n_events": n_events,
            "n_controls": n_ctrl,
            "conventional_cols": conv_cols,
        }

        if p_full is not None:
            nri_full = compute_nri_idi(p_old, p_full, y)
            etype_results["glmm_full_vs_conventional"] = nri_full
            logger.info("  NRI (full GLMM vs conv): %.3f  IDI: %.3f",
                        nri_full["nri_continuous"], nri_full["idi"])

        if p_pars is not None:
            nri_pars = compute_nri_idi(p_old, p_pars, y)
            etype_results["glmm_pars_vs_conventional"] = nri_pars
            logger.info("  NRI (pars GLMM vs conv): %.3f  IDI: %.3f",
                        nri_pars["nri_continuous"], nri_pars["idi"])

        nri_idi_results[etype] = etype_results

    # Save
    out_path = OUT_DIR / "nri_idi_results.json"
    with open(out_path, "w") as fh:
        json.dump(nri_idi_results, fh, indent=2, default=_json_default)
    logger.info("Saved: %s", out_path)
    logger.info("=== NRI/IDI DONE ===")


if __name__ == "__main__":
    run_nri_idi()
