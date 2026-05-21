"""Act 1 finish — targeted completion of missing artifacts.

Reutiliza PKLs existentes, hardcodea CV GLMM ya calculados (hypo/hyper),
y solo corre lo que falta con parámetros reducidos para terminar rápido.

Lo que hace:
  1. Carga cohort desde cache (no re-procesa).
  2. Hypo + Hyper: usa CV GLMM hardcodeados del run completado.
     Extrae glmm_coef_variability.csv del pkl existente.
  3. Variability: re-corre full GLMM CV (cv_repeats=5) + fit parsimonious (cv_repeats=5).
  4. MILP bootstrap para los 3 (n_boot=50).
  5. RF/XGB CV para los 3 (cv_repeats=5).
  6. Escribe act1_results.json.

Run
---
python experiments/act1_finish.py
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

from beatlabile.config import CFG, RESULTS_DIR
from beatlabile.models.mixed_logistic import MixedLogisticModel, compute_nri_idi
from beatlabile.models.milp_tree import MILPTree
from beatlabile.models.benchmark_ml import RFModel, XGBModel, compare_benchmarks
from beatlabile.stats import ci_from_folds
from experiments.pipeline import get_feature_cols, EVENT_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "act1"
CACHE_DIR = RESULTS_DIR / "cache"

# ------------------------------------------------------------------ #
# Resultados GLMM ya calculados (run completado 2026-04-09 14:43-15:00)
# ------------------------------------------------------------------ #
KNOWN_GLMM = {
    "hypotension": {
        "glmm_cv_auc_mean": 0.750,
        "glmm_cv_auc_std": 0.182,
        "glmm_cv_auc_ci95": [0.258, 0.988],
        "glmm_pars_cv_auc_mean": 0.656,
        "glmm_pars_cv_auc_std": 0.192,
        "glmm_pars_cv_auc_ci95": [0.238, 1.000],
    },
    "hypertension": {
        "glmm_cv_auc_mean": 0.882,
        "glmm_cv_auc_std": 0.162,
        "glmm_cv_auc_ci95": [0.391, 1.000],
        "glmm_pars_cv_auc_mean": 0.834,
        "glmm_pars_cv_auc_std": 0.167,
        "glmm_pars_cv_auc_ci95": [0.412, 1.000],
    },
}

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

CONVENTIONAL_COLS = ["map_mean", "hr_mean"]

# Configuración rápida: menos repeticiones para terminar en ~20 min
FAST_CFG = copy.deepcopy(CFG)
FAST_CFG["models"]["cv_repeats"] = 5
FAST_CFG["models"]["milp"]["bootstrap_reps"] = 50


def _save_glmm_coefficients(glmm, out_path: Path) -> None:
    if not hasattr(glmm, "coef_") or glmm.coef_ is None:
        return
    coef_df = pd.DataFrame({
        "feature": glmm.feature_cols,
        "coef_raw": glmm.coef_,
        "coef_abs": np.abs(glmm.coef_),
    }).sort_values("coef_abs", ascending=False)
    coef_df.to_csv(out_path, index=False)
    logger.info("Coef saved: %s", out_path)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_finish() -> dict:
    logger.info("=== ACT 1 FINISH (targeted) ===")

    # ------------------------------------------------------------------ #
    # 1. Cargar cohort desde cache
    # ------------------------------------------------------------------ #
    cache_file = CACHE_DIR / "clinic_windows.parquet"
    if not cache_file.exists():
        # Fallback: re-process (slow, but shouldn't be needed)
        logger.warning("Cache not found, re-processing...")
        from beatlabile.io.loader_clinic import iter_clinic_files
        from beatlabile.config import DATA_CLINIC
        from experiments.pipeline import process_cohort
        windows_df, _ = process_cohort(
            iter_fn=lambda: iter_clinic_files(DATA_CLINIC),
            cfg=CFG,
            cohort_name="clinic",
            cache_dir=CACHE_DIR,
        )
    else:
        windows_df = pd.read_parquet(cache_file)
        logger.info("Loaded cache: %d windows", len(windows_df))

    results: dict = {}

    for etype in EVENT_TYPES:
        logger.info("--- %s ---", etype)
        sub = windows_df[windows_df["event_type"] == etype].copy()
        feat_cols = get_feature_cols(sub)
        X = sub[feat_cols]
        y = sub["label"].values
        patient_ids = sub["patient_id"].values

        n_events = int(np.sum(y))
        n_ctrl = int(np.sum(y == 0))
        logger.info("  n_events=%d  n_controls=%d", n_events, n_ctrl)

        pars_feat_cols = [c for c in PARSIMONIOUS_FEATURES.get(etype, []) if c in feat_cols]

        # ------------------------------------------------------------ #
        # A. GLMM — cargar pkl existente, extraer coef, CV si necesario
        # ------------------------------------------------------------ #
        glmm_pkl = OUT_DIR / f"glmm_{etype}.pkl"
        with open(glmm_pkl, "rb") as fh:
            glmm = pickle.load(fh)
        logger.info("  Loaded GLMM pkl: %s", glmm_pkl.name)

        coef_csv = OUT_DIR / f"glmm_coef_{etype}.csv"
        if not coef_csv.exists():
            _save_glmm_coefficients(glmm, coef_csv)

        if etype in KNOWN_GLMM:
            known = KNOWN_GLMM[etype]
            glmm_auc_mean = known["glmm_cv_auc_mean"]
            glmm_auc_std  = known["glmm_cv_auc_std"]
            glmm_ci       = known["glmm_cv_auc_ci95"]
            logger.info("  GLMM CV AUC (known): %.3f ± %.3f [%.3f, %.3f]",
                        glmm_auc_mean, glmm_auc_std, *glmm_ci)
        else:
            logger.info("  Running GLMM CV (cv_repeats=5)...")
            glmm_cv = glmm.cross_validate(X, y, patient_ids, FAST_CFG)
            glmm_auc_mean = glmm_cv.auc_mean
            glmm_auc_std  = glmm_cv.auc_std
            glmm_ci       = list(ci_from_folds(glmm_cv.auc_folds))
            logger.info("  GLMM CV AUC: %.3f ± %.3f [%.3f, %.3f]",
                        glmm_auc_mean, glmm_auc_std, *glmm_ci)

        # Incremental vs conventional
        conv_cols = [c for c in CONVENTIONAL_COLS if c in feat_cols]
        if conv_cols:
            glmm_base = MixedLogisticModel(feature_cols=conv_cols)
            glmm_base.fit(X, y, patient_ids)
            nri_idi = compute_nri_idi(
                glmm_base.predict_proba(X),
                glmm.predict_proba(X),
                y,
            )
        else:
            nri_idi = {}

        # ------------------------------------------------------------ #
        # A2. Parsimonious GLMM
        # ------------------------------------------------------------ #
        pars_pkl = OUT_DIR / f"glmm_parsimonious_{etype}.pkl"
        pars_coef_csv = OUT_DIR / f"glmm_pars_coef_{etype}.csv"

        if pars_pkl.exists() and etype in KNOWN_GLMM:
            with open(pars_pkl, "rb") as fh:
                glmm_pars = pickle.load(fh)
            known = KNOWN_GLMM[etype]
            pars_auc_mean = known["glmm_pars_cv_auc_mean"]
            pars_auc_std  = known["glmm_pars_cv_auc_std"]
            pars_ci       = known["glmm_pars_cv_auc_ci95"]
            logger.info("  Parsimonious GLMM CV (known): %.3f ± %.3f [%.3f, %.3f]",
                        pars_auc_mean, pars_auc_std, *pars_ci)
            if not pars_coef_csv.exists():
                _save_glmm_coefficients(glmm_pars, pars_coef_csv)
        else:
            logger.info("  Fitting parsimonious GLMM (%d features, cv_repeats=5)...",
                        len(pars_feat_cols))
            glmm_pars = MixedLogisticModel(feature_cols=pars_feat_cols)
            glmm_pars.fit(X, y, patient_ids)
            pars_cv = glmm_pars.cross_validate(X, y, patient_ids, FAST_CFG)
            pars_ci = list(ci_from_folds(pars_cv.auc_folds))
            pars_auc_mean = pars_cv.auc_mean
            pars_auc_std  = pars_cv.auc_std
            logger.info("  Parsimonious GLMM CV AUC: %.3f ± %.3f [%.3f, %.3f]",
                        pars_auc_mean, pars_auc_std, *pars_ci)
            with open(pars_pkl, "wb") as fh:
                pickle.dump(glmm_pars, fh)
            _save_glmm_coefficients(glmm_pars, pars_coef_csv)

        pars_results = {
            "glmm_pars_cv_auc_mean": pars_auc_mean,
            "glmm_pars_cv_auc_std":  pars_auc_std,
            "glmm_pars_cv_auc_ci95": pars_ci,
            "glmm_pars_n_features":  len(pars_feat_cols),
            "glmm_pars_epv":         round(n_events / max(len(pars_feat_cols), 1), 2),
            "glmm_pars_features":    pars_feat_cols,
        }

        # ------------------------------------------------------------ #
        # B. MILP — cargar pkl + re-predict + bootstrap (n_boot=50)
        # ------------------------------------------------------------ #
        milp_pkl = OUT_DIR / f"milp_{etype}.pkl"
        with open(milp_pkl, "rb") as fh:
            milp = pickle.load(fh)
        milp_preds = milp.predict_proba(X)
        milp_auc = float(roc_auc_score(y, milp_preds)) if len(np.unique(y)) == 2 else np.nan
        logger.info("  MILP train AUC: %.3f", milp_auc)

        logger.info("  MILP bootstrap (n_boot=50)...")
        stability = milp.bootstrap_stability(X, y, etype, FAST_CFG)
        milp_boot_mean = float(np.mean(stability.bootstrap_auc)) if len(stability.bootstrap_auc) > 0 else np.nan
        logger.info("  MILP bootstrap AUC mean: %.3f", milp_boot_mean)

        # ------------------------------------------------------------ #
        # C. RF + XGB (cv_repeats=5)
        # ------------------------------------------------------------ #
        logger.info("  Fitting RF (cv_repeats=5)...")
        rf = RFModel()
        rf.fit(X, y, FAST_CFG)
        rf_cv = rf.cross_validate(X, y, patient_ids, FAST_CFG)
        rf_ci = list(ci_from_folds(rf_cv.auc_folds))

        logger.info("  Fitting XGBoost (cv_repeats=5)...")
        xgb = XGBModel()
        xgb.fit(X, y, FAST_CFG)
        xgb_cv = xgb.cross_validate(X, y, patient_ids, FAST_CFG)
        xgb_ci = list(ci_from_folds(xgb_cv.auc_folds))

        logger.info("  RF AUC: %.3f [%.3f, %.3f] | XGB AUC: %.3f [%.3f, %.3f]",
                    rf_cv.auc_mean, *rf_ci, xgb_cv.auc_mean, *xgb_ci)

        comparison = compare_benchmarks(
            glmm_auc_mean, milp_auc, rf_cv.auc_mean, xgb_cv.auc_mean
        )

        results[etype] = {
            "glmm_cv_auc_mean":          glmm_auc_mean,
            "glmm_cv_auc_std":           glmm_auc_std,
            "glmm_cv_auc_ci95":          glmm_ci,
            "rf_cv_auc_ci95":            rf_ci,
            "xgb_cv_auc_ci95":           xgb_ci,
            "milp_train_auc":            milp_auc,
            "milp_bootstrap_auc_mean":   milp_boot_mean,
            "rf_cv_auc":                 rf_cv.auc_mean,
            "xgb_cv_auc":                xgb_cv.auc_mean,
            "nri_idi":                   nri_idi,
            "benchmark":                 comparison,
            "n_events":                  n_events,
            "n_controls":                n_ctrl,
            **pars_results,
        }

    with open(OUT_DIR / "act1_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)

    logger.info("=== ACT 1 FINISH complete. Results: %s ===", OUT_DIR / "act1_results.json")
    return results


if __name__ == "__main__":
    run_finish()
