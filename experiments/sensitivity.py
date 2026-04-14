"""Sensitivity analyses (Section 9.7).

Systematically varies key parameters to assess robustness of Act 1 results.

Analyses performed
------------------
1. ELH threshold variations (PAM thresholds, SBP thresholds)
2. Minimum event duration (1, 3, 5 min)
3. Prediction window (15, 30, 60 min)
4. Signal quality threshold (60%, 70%, 80%)
5. HRV window duration (30 s vs 1 min)
6. Tree depth (2, 3, 4 levels)
7. Complete-only vs complete+fragments (Clínic)
8. First-event-per-patient analysis
9. Standard logistic regression (no random effects)
10. Bland-Altman: 500 Hz vs 125 Hz sub-sample (20 records)

Run
---
python experiments/sensitivity.py
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from beatlabile.config import CFG, DATA_CLINIC, RESULTS_DIR
from beatlabile.io.loader_clinic import iter_clinic_files
from beatlabile.models.mixed_logistic import MixedLogisticModel
from beatlabile.models.milp_tree import MILPTree
from experiments.pipeline import process_cohort, get_feature_cols, EVENT_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ACT1_DIR = RESULTS_DIR / "act1"
CACHE_DIR = RESULTS_DIR / "cache"


def run_sensitivity() -> dict:
    """Run all sensitivity analyses. Returns combined results dict."""
    logger.info("=== Sensitivity Analyses ===")

    results: dict = {}

    # --- Baseline windows (load from cache or reprocess) ---
    base_windows, _ = process_cohort(
        iter_fn=lambda: iter_clinic_files(DATA_CLINIC),
        cfg=CFG,
        cohort_name="clinic",
        cache_dir=CACHE_DIR,
    )

    if base_windows.empty:
        logger.error("No baseline windows. Run act1_clinic.py first.")
        return {}

    sens_cfg = CFG.get("sensitivity", {})

    # ------------------------------------------------------------------ #
    # 1. ELH threshold variations
    # ------------------------------------------------------------------ #
    logger.info("1. ELH threshold variations...")
    results["elh_thresholds"] = _vary_elh_thresholds(sens_cfg)

    # ------------------------------------------------------------------ #
    # 2. Prediction window
    # ------------------------------------------------------------------ #
    logger.info("2. Prediction window variation...")
    results["prediction_windows"] = _vary_prediction_window(base_windows, sens_cfg)

    # ------------------------------------------------------------------ #
    # 3. Tree depth
    # ------------------------------------------------------------------ #
    logger.info("3. Tree depth variation...")
    results["tree_depths"] = _vary_tree_depth(base_windows, sens_cfg)

    # ------------------------------------------------------------------ #
    # 4. HRV computation window
    # ------------------------------------------------------------------ #
    logger.info("4. HRV window variation...")
    results["hrv_windows"] = _vary_hrv_window(sens_cfg)

    # ------------------------------------------------------------------ #
    # 5. First-event-per-patient
    # ------------------------------------------------------------------ #
    logger.info("5. First-event-per-patient...")
    results["first_event_only"] = _first_event_analysis(base_windows)

    # ------------------------------------------------------------------ #
    # 6. Standard logistic regression (no random effects)
    # ------------------------------------------------------------------ #
    logger.info("6. Standard LR vs GLMM...")
    results["standard_lr"] = _standard_lr_comparison(base_windows)

    with open(OUT_DIR / "sensitivity_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)

    logger.info("Sensitivity analyses complete. Results in %s", OUT_DIR)
    return results


# ---------------------------------------------------------------------------
# Individual sensitivity functions
# ---------------------------------------------------------------------------

def _vary_elh_thresholds(sens_cfg: dict) -> dict:
    """Re-run full pipeline with different MAP/SBP thresholds."""
    output: dict = {}
    map_thresholds = sens_cfg.get("map_thresholds", [55, 60, 50])
    sbp_thresholds = sens_cfg.get("sbp_thresholds", [180, 160, 200])
    durations = sens_cfg.get("event_durations_sec", [180, 60, 300])

    for map_t, sbp_t, dur in zip(map_thresholds, sbp_thresholds, durations):
        cfg_var = copy.deepcopy(CFG)
        cfg_var["events"]["hypotension"]["map_threshold"] = map_t
        cfg_var["events"]["hypotension"]["min_duration_seconds"] = dur
        cfg_var["events"]["hypertension"]["sbp_threshold"] = sbp_t
        cfg_var["events"]["hypertension"]["min_duration_seconds"] = dur

        label = f"map{map_t}_sbp{sbp_t}_dur{dur}"
        logger.info("  Threshold variant: %s", label)
        try:
            windows_v, _ = process_cohort(
                iter_fn=lambda: iter_clinic_files(DATA_CLINIC),
                cfg=cfg_var,
                cohort_name=f"clinic_{label}",
                cache_dir=None,  # no cache for variants
            )
            auc_by_type = _fit_and_score(windows_v, cfg_var)
            output[label] = auc_by_type
        except Exception as exc:
            logger.warning("  Failed for %s: %s", label, exc)
    return output


def _vary_prediction_window(base_windows: pd.DataFrame, sens_cfg: dict) -> dict:
    """Vary the prediction horizon (5, 10, 15, 30 min windows before event)."""
    output: dict = {}
    horizons = sens_cfg.get("prediction_windows_min", [15, 30, 60])
    for hmin in horizons:
        # Filter windows to only use sub-windows within hmin before event
        sub = base_windows.copy()
        if "event_onset_s" in sub.columns and "window_start_s" in sub.columns:
            # For event windows: keep only those where window_end is within hmin of onset
            event_mask = sub["label"] == 1
            valid = (
                sub.loc[event_mask, "event_onset_s"]
                - sub.loc[event_mask, "window_start_s"]
            ) <= hmin * 60.0
            sub = pd.concat([
                sub[sub["label"] == 0],
                sub[event_mask][valid.values if hasattr(valid, "values") else valid],
            ])
        auc_by_type = _fit_and_score(sub, CFG)
        output[f"{hmin}min"] = auc_by_type
    return output


def _vary_tree_depth(base_windows: pd.DataFrame, sens_cfg: dict) -> dict:
    """Test tree depths 2, 3, 4."""
    output: dict = {}
    depths = sens_cfg.get("tree_depths", [2, 3, 4])
    for depth in depths:
        cfg_var = copy.deepcopy(CFG)
        cfg_var["models"]["milp"]["max_depth"] = depth
        auc_by_type: dict = {}
        for etype in EVENT_TYPES:
            sub = base_windows[base_windows["event_type"] == etype]
            if len(sub) < 40:
                continue
            feat_cols = get_feature_cols(sub)
            X = sub[feat_cols]
            y = sub["label"].values
            if len(np.unique(y)) < 2:
                continue
            tree = MILPTree()
            tree.fit(X, y, etype, cfg_var)
            preds = tree.predict_proba(X)
            auc_by_type[etype] = float(roc_auc_score(y, preds)) if len(np.unique(y)) == 2 else np.nan
        output[f"depth_{depth}"] = auc_by_type
    return output


def _vary_hrv_window(sens_cfg: dict) -> dict:
    """Re-run full pipeline with different HRV window durations."""
    output: dict = {}
    win_sizes = sens_cfg.get("hrv_window_seconds", [30, 60])
    for ws in win_sizes:
        cfg_var = copy.deepcopy(CFG)
        cfg_var["metrics"]["window_seconds"] = ws
        label = f"hrv_{ws}s"
        logger.info("  HRV window: %s s", ws)
        try:
            windows_v, _ = process_cohort(
                iter_fn=lambda: iter_clinic_files(DATA_CLINIC),
                cfg=cfg_var,
                cohort_name=f"clinic_{label}",
                cache_dir=None,
            )
            output[label] = _fit_and_score(windows_v, cfg_var)
        except Exception as exc:
            logger.warning("  Failed for %s: %s", label, exc)
    return output


def _first_event_analysis(base_windows: pd.DataFrame) -> dict:
    """Keep only the first event per patient (avoid multiple-event patient effects)."""
    # For event windows: keep only the earliest per patient
    event_w = base_windows[base_windows["label"] == 1].copy()
    if "event_onset_s" in event_w.columns:
        event_w = event_w.sort_values("event_onset_s").groupby("patient_id").first().reset_index()
    ctrl_w = base_windows[base_windows["label"] == 0]
    sub = pd.concat([event_w, ctrl_w], ignore_index=True)
    return _fit_and_score(sub, CFG)


def _standard_lr_comparison(base_windows: pd.DataFrame) -> dict:
    """Fit standard (non-mixed) logistic regression and compare AUC to GLMM."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    output: dict = {}
    for etype in EVENT_TYPES:
        sub = base_windows[base_windows["event_type"] == etype]
        if len(sub) < 40:
            continue
        feat_cols = get_feature_cols(sub)
        X = sub[feat_cols].fillna(sub[feat_cols].median())
        y = sub["label"].values
        if len(np.unique(y)) < 2:
            continue
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_sc, y)
        proba = lr.predict_proba(X_sc)[:, 1]
        output[etype] = {
            "standard_lr_auc": float(roc_auc_score(y, proba)),
        }

        glmm_path = ACT1_DIR / f"glmm_{etype}.pkl"
        if glmm_path.exists():
            with open(glmm_path, "rb") as fh:
                glmm = pickle.load(fh)
            for col in glmm.feature_cols:
                if col not in sub.columns:
                    sub[col] = np.nan
            glmm_proba = glmm.predict_proba(sub[glmm.feature_cols])
            output[etype]["glmm_auc"] = float(roc_auc_score(y, glmm_proba))

    return output


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fit_and_score(windows_df: pd.DataFrame, cfg: dict) -> dict:
    """Fit GLMM on windows_df and return train-set AUC per event type."""
    if windows_df.empty:
        return {}
    aucs: dict = {}
    for etype in EVENT_TYPES:
        sub = windows_df[windows_df["event_type"] == etype] if "event_type" in windows_df.columns else windows_df
        if len(sub) < 20:
            continue
        feat_cols = get_feature_cols(sub)
        X = sub[feat_cols]
        y = sub["label"].values if "label" in sub.columns else np.zeros(len(sub), dtype=int)
        pids = sub["patient_id"].values if "patient_id" in sub.columns else np.arange(len(sub))
        if len(np.unique(y)) < 2:
            continue
        model = MixedLogisticModel(feature_cols=feat_cols)
        model.fit(X, y, pids)
        proba = model.predict_proba(X)
        aucs[etype] = float(roc_auc_score(y, proba))
    return aucs


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    run_sensitivity()
