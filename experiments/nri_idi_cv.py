"""nri_idi_cv.py
Cross-validated NRI/IDI for GLMM parsimonious vs PA-only conventional model.

Uses patient-level 5-fold CV to generate out-of-fold predictions, then
computes NRI/IDI + bootstrap CI on the pooled OOF predictions.

Training-apparent NRI/IDI (from nri_idi_calc.py) are inflated by overfitting.
This script provides the publishable cross-validated version.

Outputs:
  results/act1/nri_idi_cv_results.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from beatlabile.models.mixed_logistic import MixedLogisticModel, compute_nri_idi
from beatlabile.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = RESULTS_DIR / "cache"
OUT_DIR   = RESULTS_DIR / "act1"

CONVENTIONAL_COLS = ["std_pa_mean", "cv_pa_mean"]

PARSIMONIOUS_FEATURES: dict[str, list[str]] = {
    "hypotension": [
        "std_pa_mean", "cv_pa_std", "rsa_max", "arv_std",
        "brs_min", "arv_mean", "std_pa_max", "rsa_mean",
    ],
    "hypertension": [
        "std_pa_std", "std_pa_max", "cv_pa_std", "std_pa_slope",
        "arv_std", "cv_pa_mean", "brs_min", "sdnn_mean",
    ],
}

TARGET_ETYPES = ["hypotension", "hypertension"]
N_FOLDS = 5
N_BOOT  = 500


def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def bootstrap_nri_idi_ci(p_old: np.ndarray, p_new: np.ndarray,
                          y: np.ndarray, groups: np.ndarray,
                          n_boot: int = 500, seed: int = 42) -> dict:
    """Patient-cluster bootstrap CI for NRI and IDI."""
    rng = np.random.default_rng(seed)
    unique_g = np.unique(groups)
    nri_vals, idi_vals = [], []

    for _ in range(n_boot):
        sampled = rng.choice(unique_g, size=len(unique_g), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        yt, po, pn = y[idx], p_old[idx], p_new[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            res = compute_nri_idi(po, pn, yt)
            nri_vals.append(res["nri_continuous"])
            idi_vals.append(res["idi"])
        except Exception:
            continue

    def _pct(vals):
        if len(vals) < 10:
            return float("nan"), float("nan")
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    return {
        "nri_ci95": _pct(nri_vals),
        "idi_ci95": _pct(idi_vals),
    }


def patient_kfold_oof(windows: pd.DataFrame, etype: str,
                       n_folds: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions for conventional + parsimonious GLMM.

    Returns (y_true, p_conventional_oof, p_glmm_oof) pooled across folds.
    CV is done at patient level (each patient's windows stay together).
    """
    feat_cols = PARSIMONIOUS_FEATURES[etype]
    conv_cols = CONVENTIONAL_COLS

    sub = windows[windows["event_type"] == etype].copy().reset_index(drop=True)
    unique_pts = np.unique(sub["patient_id"].values)

    rng = np.random.default_rng(seed)
    shuffled_pts = rng.permutation(unique_pts)
    folds = np.array_split(shuffled_pts, n_folds)

    y_all, p_conv_all, p_glmm_all, groups_all = [], [], [], []

    for fold_i, test_pts in enumerate(folds):
        test_mask = sub["patient_id"].isin(test_pts).values
        tr = sub.iloc[~test_mask]
        te = sub.iloc[test_mask]

        if tr["label"].nunique() < 2 or te["label"].nunique() < 2:
            logger.debug("Fold %d skipped (homogeneous label)", fold_i)
            continue

        # Conventional model
        avail_conv = [c for c in conv_cols if c in tr.columns]
        scaler = StandardScaler()
        X_tr_conv = scaler.fit_transform(tr[avail_conv].fillna(0))
        X_te_conv = scaler.transform(te[avail_conv].fillna(0))
        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        lr.fit(X_tr_conv, tr["label"].values)
        p_conv_te = lr.predict_proba(X_te_conv)[:, 1]

        # GLMM parsimonious
        avail_feat = [c for c in feat_cols if c in tr.columns]
        try:
            m = MixedLogisticModel(feature_cols=avail_feat)
            m.fit(tr[avail_feat], tr["label"].values, tr["patient_id"].values)
            p_glmm_te = m.predict_proba(te[avail_feat])
        except Exception as e:
            logger.warning("Fold %d GLMM fit failed: %s", fold_i, e)
            continue

        y_all.extend(te["label"].values)
        p_conv_all.extend(p_conv_te)
        p_glmm_all.extend(p_glmm_te)
        groups_all.extend(te["patient_id"].values)

        logger.info("  Fold %d/%d: n_test=%d  events=%d",
                    fold_i + 1, n_folds, len(te), int(te["label"].sum()))

    return (np.asarray(y_all), np.asarray(p_conv_all),
            np.asarray(p_glmm_all), np.asarray(groups_all))


def run_nri_idi_cv() -> None:
    logger.info("=== NRI/IDI Cross-Validated ===")

    windows = pd.read_parquet(CACHE_DIR / "clinic_windows.parquet")
    logger.info("Loaded: %d windows", len(windows))

    results = {}

    for etype in TARGET_ETYPES:
        logger.info("--- %s ---", etype)
        sub = windows[windows["event_type"] == etype]
        logger.info("  n_events=%d  n_controls=%d",
                    int(sub["label"].sum()), int((sub["label"] == 0).sum()))

        y, p_conv, p_glmm, groups = patient_kfold_oof(windows, etype, n_folds=N_FOLDS)

        if len(np.unique(y)) < 2 or y.sum() < 10:
            logger.warning("  Insufficient events in OOF set — skipping")
            continue

        # Point estimates on pooled OOF
        nri_res = compute_nri_idi(p_conv, p_glmm, y)
        logger.info("  CV NRI=%.3f  IDI=%.3f", nri_res["nri_continuous"], nri_res["idi"])

        # Bootstrap CI
        ci_res = bootstrap_nri_idi_ci(p_conv, p_glmm, y, groups, n_boot=N_BOOT)
        logger.info("  NRI CI: [%.3f–%.3f]  IDI CI: [%.3f–%.3f]",
                    ci_res["nri_ci95"][0], ci_res["nri_ci95"][1],
                    ci_res["idi_ci95"][0], ci_res["idi_ci95"][1])

        from sklearn.metrics import roc_auc_score
        auc_conv = roc_auc_score(y, p_conv)
        auc_glmm = roc_auc_score(y, p_glmm)
        logger.info("  OOF AUC — conventional: %.3f  GLMM pars: %.3f", auc_conv, auc_glmm)

        results[etype] = {
            "n_events_oof": int(y.sum()),
            "n_controls_oof": int((y == 0).sum()),
            "auc_conventional_cv": round(auc_conv, 4),
            "auc_glmm_pars_cv":   round(auc_glmm, 4),
            "nri_continuous_cv":  round(nri_res["nri_continuous"], 4),
            "idi_cv":             round(nri_res["idi"], 4),
            "nri_ci95":  [round(ci_res["nri_ci95"][0], 3),
                          round(ci_res["nri_ci95"][1], 3)],
            "idi_ci95":  [round(ci_res["idi_ci95"][0], 3),
                          round(ci_res["idi_ci95"][1], 3)],
            "note": (
                "NRI/IDI computed on out-of-fold (OOF) predictions from "
                f"patient-level {N_FOLDS}-fold CV. "
                "CI via patient-cluster bootstrap (B=500)."
            ),
        }

    out_path = OUT_DIR / "nri_idi_cv_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    logger.info("Saved: %s", out_path)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    run_nri_idi_cv()
