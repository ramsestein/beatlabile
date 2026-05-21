"""Act 1 — Model Development on Clínic UCIQ cohort (Section 4.1 / 9).

Steps
-----
1. Process all Clínic records (QC → features → events → windows).
2. For each event type × model approach:
   A. GLMM: fit + 10×50 patient-level CV
   B. MILP tree: fit + bootstrap stability
   C. Benchmark: RF + XGBoost CV
3. Save trained models and results to results/act1/

Run
---
python experiments/act1_clinic.py
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from beatlabile.config import CFG, DATA_CLINIC, RESULTS_DIR
from beatlabile.io.loader_clinic import iter_clinic_files
from beatlabile.models.mixed_logistic import MixedLogisticModel, compute_nri_idi
from beatlabile.models.milp_tree import MILPTree
from beatlabile.models.benchmark_ml import RFModel, XGBModel, compare_benchmarks
from experiments.pipeline import process_cohort, get_feature_cols, EVENT_TYPES
from beatlabile.stats import ci_from_folds, calibration_data, net_benefit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "act1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cache"

# Conventional predictor columns (used for NRI/IDI baseline model)
CONVENTIONAL_COLS = ["map_mean", "hr_mean"]  # PAM media, FC media from feature set

# Pre-specified parsimonious feature sets: top ~8 features per event type
# selected a priori from Act 4 domain-robustness ranking (min AUC across cohorts).
# These are the PRIMARY model features (EPV-adequate).
# The 40-feature GLMM is reported as sensitivity analysis.
PARSIMONIOUS_FEATURES: dict[str, list[str]] = {
    "hypotension": [
        "std_pa_mean", "cv_pa_std", "rsa_max", "arv_std",
        "brs_min", "arv_mean", "std_pa_max", "rsa_mean",
    ],
    "hypertension": [
        "std_pa_std", "std_pa_max", "cv_pa_std", "std_pa_slope",
        "arv_std", "cv_pa_mean", "brs_min", "sdnn_mean",
    ],
    "variability": [
        "sdnn_std", "cv_pa_mean", "rmssd_std", "rmssd_mean",
        "std_pa_std", "std_pa_max", "pnn50_mean", "sdnn_mean",
    ],
}


def run_act1() -> dict:
    """Main entry point for Act 1. Returns summary results dict."""
    logger.info("=== ACT 1: Clínic UCIQ Development ===")

    # ------------------------------------------------------------------ #
    # 1. Load & process cohort
    # ------------------------------------------------------------------ #
    windows_df, events_df = process_cohort(
        iter_fn=lambda: iter_clinic_files(DATA_CLINIC),
        cfg=CFG,
        cohort_name="clinic",
        cache_dir=CACHE_DIR,
    )

    if windows_df.empty:
        logger.error("No windows extracted from Clínic cohort. Check data paths.")
        return {}

    events_df.to_csv(OUT_DIR / "clinic_events.csv", index=False)
    _print_cohort_summary(windows_df, events_df, "Clínic")
    _save_table1(windows_df, OUT_DIR / "table1_clinic.csv", "Clínic")

    results: dict = {}

    for etype in EVENT_TYPES:
        logger.info("--- Event type: %s ---", etype)
        sub = windows_df[windows_df["event_type"] == etype].copy()
        if len(sub) == 0:
            logger.warning("No windows for event type %s.", etype)
            continue

        feat_cols = get_feature_cols(sub)
        X = sub[feat_cols]
        y = sub["label"].values
        patient_ids = sub["patient_id"].values

        n_events = int(np.sum(y))
        n_ctrl = int(np.sum(y == 0))
        logger.info("  Windows: %d events, %d controls", n_events, n_ctrl)

        if n_events < 20 or n_ctrl < 20:
            logger.warning("  Too few samples for reliable modelling. Skipping.")
            continue

        # ------------------------------------------------------------ #
        # A. GLMM
        # ------------------------------------------------------------ #
        logger.info("  Fitting GLMM...")
        glmm = MixedLogisticModel(feature_cols=feat_cols)
        glmm.fit(X, y, patient_ids)
        glmm_cv = glmm.cross_validate(X, y, patient_ids, CFG)

        logger.info(
            "  GLMM CV AUC: %.3f ± %.3f", glmm_cv.auc_mean, glmm_cv.auc_std
        )
        glmm_ci = ci_from_folds(glmm_cv.auc_folds)
        logger.info("  GLMM CV AUC 95%% CI: [%.3f, %.3f]", *glmm_ci)

        # Calibration curve + DCA on full training set (apparent performance)
        _proba_glmm = glmm.predict_proba(X)
        calibration_data(y, _proba_glmm).to_csv(
            OUT_DIR / f"calibration_glmm_{etype}.csv", index=False
        )
        net_benefit(y, _proba_glmm).to_csv(
            OUT_DIR / f"dca_glmm_{etype}.csv", index=False
        )

        # Incremental value over conventional predictors
        conv_cols = [c for c in CONVENTIONAL_COLS if c in feat_cols]
        if len(conv_cols) >= 1:
            glmm_base = MixedLogisticModel(feature_cols=conv_cols)
            glmm_base.fit(X, y, patient_ids)
            nri_idi = compute_nri_idi(
                glmm_base.predict_proba(X),
                glmm.predict_proba(X),
                y,
            )
        else:
            nri_idi = {}

        # Save GLMM
        with open(OUT_DIR / f"glmm_{etype}.pkl", "wb") as fh:
            pickle.dump(glmm, fh)

        # Standardized GLMM coefficients (feature importance proxy)
        _save_glmm_coefficients(glmm, OUT_DIR / f"glmm_coef_{etype}.csv")

        # ------------------------------------------------------------ #
        # A2. Parsimonious GLMM (primary model — EPV-justified)
        # ------------------------------------------------------------ #
        pars_feat_cols = [
            c for c in PARSIMONIOUS_FEATURES.get(etype, []) if c in feat_cols
        ]
        if len(pars_feat_cols) >= 3:
            logger.info(
                "  Fitting parsimonious GLMM (%d features, EPV=%.1f)...",
                len(pars_feat_cols), n_events / len(pars_feat_cols),
            )
            glmm_pars = MixedLogisticModel(feature_cols=pars_feat_cols)
            glmm_pars.fit(X, y, patient_ids)
            glmm_pars_cv = glmm_pars.cross_validate(X, y, patient_ids, CFG)
            glmm_pars_ci = ci_from_folds(glmm_pars_cv.auc_folds)
            logger.info(
                "  Parsimonious GLMM CV AUC: %.3f ± %.3f  [%.3f, %.3f]",
                glmm_pars_cv.auc_mean, glmm_pars_cv.auc_std, *glmm_pars_ci,
            )
            with open(OUT_DIR / f"glmm_parsimonious_{etype}.pkl", "wb") as fh:
                pickle.dump(glmm_pars, fh)
            _save_glmm_coefficients(glmm_pars, OUT_DIR / f"glmm_pars_coef_{etype}.csv")
            pars_results = {
                "glmm_pars_cv_auc_mean": glmm_pars_cv.auc_mean,
                "glmm_pars_cv_auc_std": glmm_pars_cv.auc_std,
                "glmm_pars_cv_auc_ci95": list(glmm_pars_ci),
                "glmm_pars_n_features": len(pars_feat_cols),
                "glmm_pars_epv": round(n_events / len(pars_feat_cols), 2),
                "glmm_pars_features": pars_feat_cols,
            }
        else:
            logger.warning("  Parsimonious GLMM: insufficient features, skipping.")
            pars_results = {}

        # ------------------------------------------------------------ #
        # B. MILP tree
        # ------------------------------------------------------------ #
        logger.info("  Fitting MILP tree...")
        milp = MILPTree()
        milp.fit(X, y, etype, CFG)
        milp_preds = milp.predict_proba(X)
        from sklearn.metrics import roc_auc_score
        milp_auc = float(roc_auc_score(y, milp_preds)) if len(np.unique(y)) == 2 else np.nan
        logger.info("  MILP tree (train) AUC: %.3f", milp_auc)

        logger.info("  Bootstrap stability analysis (N=%d)...", CFG["models"]["milp"]["bootstrap_reps"])
        stability = milp.bootstrap_stability(X, y, etype, CFG)
        logger.info(
            "  Top features by selection freq: %s",
            sorted(stability.feature_freq.items(), key=lambda x: -x[1])[:5],
        )

        with open(OUT_DIR / f"milp_{etype}.pkl", "wb") as fh:
            pickle.dump(milp, fh)

        # Save stability
        stab_df = pd.DataFrame({
            "feature": list(stability.feature_freq.keys()),
            "selection_freq": list(stability.feature_freq.values()),
        }).sort_values("selection_freq", ascending=False)
        stab_df.to_csv(OUT_DIR / f"milp_stability_{etype}.csv", index=False)

        # ------------------------------------------------------------ #
        # C. Benchmark ML
        # ------------------------------------------------------------ #
        logger.info("  Fitting RF benchmark...")
        rf = RFModel()
        rf.fit(X, y, CFG)
        rf_cv = rf.cross_validate(X, y, patient_ids, CFG)

        logger.info("  Fitting XGBoost benchmark...")
        xgb = XGBModel()
        xgb.fit(X, y, CFG)
        xgb_cv = xgb.cross_validate(X, y, patient_ids, CFG)

        rf_ci = ci_from_folds(rf_cv.auc_folds)
        xgb_ci = ci_from_folds(xgb_cv.auc_folds)
        logger.info(
            "  RF 95%% CI: [%.3f, %.3f] | XGB 95%% CI: [%.3f, %.3f]",
            *rf_ci, *xgb_ci,
        )

        comparison = compare_benchmarks(
            glmm_cv.auc_mean, milp_auc, rf_cv.auc_mean, xgb_cv.auc_mean
        )
        logger.info(
            "  Benchmark comparison: GLMM=%.3f, MILP=%.3f, RF=%.3f, XGB=%.3f | ΔMax=%.3f",
            glmm_cv.auc_mean, milp_auc, rf_cv.auc_mean, xgb_cv.auc_mean,
            max(abs(comparison["delta_auc_vs_rf"]), abs(comparison["delta_auc_vs_xgb"])),
        )

        results[etype] = {
            "glmm_cv_auc_mean": glmm_cv.auc_mean,
            "glmm_cv_auc_std": glmm_cv.auc_std,
            "glmm_cv_auc_ci95": list(glmm_ci),
            "rf_cv_auc_ci95": list(rf_ci),
            "xgb_cv_auc_ci95": list(xgb_ci),
            "milp_train_auc": milp_auc,
            "milp_bootstrap_auc_mean": float(np.mean(stability.bootstrap_auc)) if len(stability.bootstrap_auc) > 0 else np.nan,
            "rf_cv_auc": rf_cv.auc_mean,
            "xgb_cv_auc": xgb_cv.auc_mean,
            "nri_idi": nri_idi,
            "benchmark": comparison,
            "n_events": n_events,
            "n_controls": n_ctrl,
            **pars_results,
        }

    # Save summary
    with open(OUT_DIR / "act1_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)

    logger.info("Act 1 complete. Results in %s", OUT_DIR)
    return results


def _save_glmm_coefficients(glmm, out_path: Path) -> None:
    """Save standardized GLMM coefficients as CSV (feature importance proxy)."""
    if not hasattr(glmm, "coef_") or glmm.coef_ is None:
        return
    coef_df = pd.DataFrame({
        "feature": glmm.feature_cols,
        "coef_raw": glmm.coef_,
        "coef_abs": np.abs(glmm.coef_),
    }).sort_values("coef_abs", ascending=False)
    coef_df.to_csv(out_path, index=False)
    logger.info("GLMM coefficients saved: %s", out_path)


def _save_table1(
    windows_df: pd.DataFrame,
    out_path: Path,
    cohort_name: str,
) -> None:
    """Save a Table 1 with demographics and feature statistics."""
    feat_cols = get_feature_cols(windows_df)
    rows = []
    for etype in EVENT_TYPES:
        sub = windows_df[windows_df["event_type"] == etype]
        if len(sub) == 0:
            continue
        n_patients = sub["patient_id"].nunique() if "patient_id" in sub.columns else None
        n_events = int((sub["label"] == 1).sum())
        n_controls = int((sub["label"] == 0).sum())
        base_row = {
            "cohort": cohort_name,
            "event_type": etype,
            "n_windows": len(sub),
            "n_events": n_events,
            "n_controls": n_controls,
            "prevalence": round(n_events / max(len(sub), 1), 3),
            "n_patients": n_patients,
        }
        for feat in feat_cols:
            vals = sub[feat].dropna()
            base_row[f"{feat}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
            base_row[f"{feat}_sd"] = round(float(vals.std()), 4) if len(vals) else None
        rows.append(base_row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Table 1 saved: %s", out_path)


def _print_cohort_summary(
    windows_df: pd.DataFrame, events_df: pd.DataFrame, name: str
) -> None:
    n_patients = windows_df["patient_id"].nunique() if "patient_id" in windows_df.columns else 0
    logger.info("Cohort %s: %d patients, %d windows total", name, n_patients, len(windows_df))
    if not events_df.empty and "event_type" in events_df.columns:
        for etype in EVENT_TYPES:
            n = len(events_df[events_df["event_type"] == etype])
            logger.info("  %s events: %d", etype, n)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == "__main__":
    run_act1()
