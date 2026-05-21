"""Act 2 — Blind External Validation on MIMIC-IV Waveform (Section 4.2 / 9.2).

Applies Act 1 trained models (fixed coefficients) to 150 MIMIC patients
without retraining.  Random intercepts are set to 0 (marginal prediction).

Steps
-----
1. Process MIMIC records (same pipeline as Act 1).
2. Load Act 1 models (GLMM, MILP tree) for each event type.
3. Apply models → compute AUC, calibration, decision curves.
4. Recalibration sensitivity analyses (CITL, slope updates).
5. Save results to results/act2/

Run
---
python experiments/act2_mimic.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from beatlabile.config import CFG, DATA_MIMIC, RESULTS_DIR
from beatlabile.io.loader_mimic import iter_mimic_files
from beatlabile.models.mixed_logistic import MixedLogisticModel, ValidationResult
from beatlabile.models.milp_tree import MILPTree
from experiments.pipeline import process_cohort, get_feature_cols, EVENT_TYPES
from beatlabile.stats import bootstrap_auc_ci, calibration_data, net_benefit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ACT1_DIR = RESULTS_DIR / "act1"
OUT_DIR = RESULTS_DIR / "act2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cache"


def run_act2() -> dict:
    """Main entry point for Act 2. Returns summary results dict."""
    logger.info("=== ACT 2: MIMIC-IV Blind Validation ===")

    # ------------------------------------------------------------------ #
    # 1. Process MIMIC cohort
    # ------------------------------------------------------------------ #
    windows_df, events_df = process_cohort(
        iter_fn=lambda: iter_mimic_files(DATA_MIMIC),
        cfg=CFG,
        cohort_name="mimic",
        cache_dir=CACHE_DIR,
    )

    if windows_df.empty:
        logger.error("No windows extracted from MIMIC cohort.")
        return {}

    events_df.to_csv(OUT_DIR / "mimic_events.csv", index=False)
    results: dict = {}

    for etype in EVENT_TYPES:
        logger.info("--- Event type: %s ---", etype)
        sub = windows_df[windows_df["event_type"] == etype].copy()
        if len(sub) == 0:
            logger.warning("No windows for %s in MIMIC.", etype)
            continue

        feat_cols = get_feature_cols(sub)
        X = sub[feat_cols]
        y = sub["label"].values

        # ------------------------------------------------------------ #
        # A. Apply GLMM (Act 1 coefficients)
        # ------------------------------------------------------------ #
        glmm_path = ACT1_DIR / f"glmm_{etype}.pkl"
        if not glmm_path.exists():
            logger.warning("Act 1 GLMM not found at %s. Run act1_clinic.py first.", glmm_path)
            continue

        with open(glmm_path, "rb") as fh:
            glmm: MixedLogisticModel = pickle.load(fh)

        # Align feature columns — add missing cols as NaN
        for col in glmm.feature_cols:
            if col not in X.columns:
                X[col] = np.nan

        glmm_val = glmm.validate(X, y)
        logger.info(
            "  GLMM AUC=%.3f | CITL=%.3f | Slope=%.3f",
            glmm_val.auc, glmm_val.calibration_in_the_large, glmm_val.calibration_slope,
        )

        # Bootstrap CI + calibration curve + DCA
        _proba_glmm = glmm.predict_proba(X)
        _auc_pt, auc_lo, auc_hi = bootstrap_auc_ci(y, _proba_glmm)
        logger.info("  GLMM validation AUC 95%% CI: [%.3f, %.3f]", auc_lo, auc_hi)
        calibration_data(y, _proba_glmm).to_csv(
            OUT_DIR / f"calibration_glmm_{etype}.csv", index=False
        )
        net_benefit(y, _proba_glmm).to_csv(
            OUT_DIR / f"dca_glmm_{etype}.csv", index=False
        )

        # Recalibration (intercept update — sensitivity)
        from beatlabile.models.mixed_logistic import _sigmoid
        proba_raw = glmm.predict_proba(X)
        citl = float(np.mean(y) - np.mean(proba_raw))  # correction direction
        log_odds_recal = np.log(
            np.clip(proba_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(proba_raw, 1e-6, 1 - 1e-6))
        ) + citl
        proba_recal = _sigmoid(log_odds_recal)
        auc_recal = float(roc_auc_score(y, proba_recal)) if len(np.unique(y)) == 2 else np.nan

        # ------------------------------------------------------------ #
        # B. Apply MILP tree
        # ------------------------------------------------------------ #
        milp_path = ACT1_DIR / f"milp_{etype}.pkl"
        milp_auc = np.nan
        if milp_path.exists():
            with open(milp_path, "rb") as fh:
                milp: MILPTree = pickle.load(fh)
            for col in milp.feature_cols:
                if col not in X.columns:
                    X[col] = np.nan
            milp_preds = milp.predict_proba(X)
            milp_auc = float(roc_auc_score(y, milp_preds)) if len(np.unique(y)) == 2 else np.nan
            logger.info("  MILP tree AUC=%.3f", milp_auc)

        results[etype] = {
            "glmm_auc": glmm_val.auc,
            "glmm_auc_ci95": [auc_lo, auc_hi],
            "glmm_citl": glmm_val.calibration_in_the_large,
            "glmm_cal_slope": glmm_val.calibration_slope,
            "glmm_auc_recalibrated": auc_recal,
            "milp_auc": milp_auc,
            "n_events": int(np.sum(y)),
            "n_controls": int(np.sum(y == 0)),
        }

    with open(OUT_DIR / "act2_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)

    logger.info("Act 2 complete. Results in %s", OUT_DIR)
    return results


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    run_act2()
