"""composite_outcome.py
Compute and report the composite haemodynamic lability outcome.

The composite outcome ("any haemodynamic lability event") is defined as:
  label=1 if the window precedes a hypotension OR hypertension event.

This was referenced in the Methods section but never reported quantitatively.

Outputs:
  results/act1/composite_outcome_results.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from beatlabile.models.mixed_logistic import MixedLogisticModel
from beatlabile.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = RESULTS_DIR / "cache"
OUT_DIR   = RESULTS_DIR / "act1"

PARSIMONIOUS_FEATURES_HYPO: list[str] = [
    "std_pa_mean", "cv_pa_std", "rsa_max", "arv_std",
    "brs_min", "arv_mean", "std_pa_max", "rsa_mean",
]

N_FOLDS = 5


def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                      groups: np.ndarray, n_boot: int = 500,
                      seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_g = np.unique(groups)
    aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_g, size=len(unique_g), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if len(aucs) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def build_composite_dataset(clinic: pd.DataFrame) -> pd.DataFrame:
    """Build composite outcome dataset.

    Uses hypotension windows as the base (consistent sampling frame).
    Composite label=1 if patient had a hypotension OR hypertension event.
    Controls (label=0 in both) remain as controls.
    """
    hypo_df = clinic[clinic["event_type"] == "hypotension"].copy()
    hyper_df = clinic[clinic["event_type"] == "hypertension"]

    pts_hypo  = set(hypo_df[hypo_df["label"] == 1]["patient_id"])
    pts_hyper = set(hyper_df[hyper_df["label"] == 1]["patient_id"])
    pts_any   = pts_hypo | pts_hyper

    # Mark composite positive
    hypo_df["composite"] = hypo_df["label"].copy()
    hypo_df.loc[
        (hypo_df["label"] == 0) & (hypo_df["patient_id"].isin(pts_hyper)),
        "composite"
    ] = 1

    return hypo_df, pts_any


def cv_auc_composite(df: pd.DataFrame, feat_cols: list[str],
                      label_col: str = "composite",
                      n_folds: int = 5, seed: int = 42) -> dict:
    """Patient-level k-fold CV AUC for composite outcome."""
    pids = df["patient_id"].values
    unique_pts = np.unique(pids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_pts)
    folds = np.array_split(shuffled, n_folds)

    y_all, p_all, g_all = [], [], []

    for fold_i, test_pts in enumerate(folds):
        test_mask = df["patient_id"].isin(test_pts).values
        tr = df.iloc[~test_mask]
        te = df.iloc[test_mask]

        if tr[label_col].nunique() < 2 or te[label_col].nunique() < 2:
            logger.debug("Fold %d skipped", fold_i)
            continue

        avail = [c for c in feat_cols if c in tr.columns]
        try:
            m = MixedLogisticModel(feature_cols=avail)
            m.fit(tr[avail], tr[label_col].values, tr["patient_id"].values)
            preds = m.predict_proba(te[avail])
        except Exception as e:
            logger.warning("Fold %d fit failed: %s", fold_i, e)
            continue

        y_all.extend(te[label_col].values)
        p_all.extend(preds)
        g_all.extend(te["patient_id"].values)
        logger.info("  Fold %d/%d: n_test=%d  events=%d",
                    fold_i + 1, n_folds, len(te), int(te[label_col].sum()))

    y_all = np.asarray(y_all)
    p_all = np.asarray(p_all)
    g_all = np.asarray(g_all)

    if len(np.unique(y_all)) < 2 or y_all.sum() < 5:
        return {"cv_auc": float("nan")}

    auc_cv = roc_auc_score(y_all, p_all)
    ci_lo, ci_hi = bootstrap_auc_ci(y_all, p_all, g_all)
    return {
        "cv_auc": round(auc_cv, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "n_oof": int(len(y_all)),
        "n_oof_events": int(y_all.sum()),
    }


def main() -> None:
    logger.info("=== Composite Outcome Analysis ===")

    clinic = pd.read_parquet(CACHE_DIR / "clinic_windows.parquet")
    logger.info("Loaded: %d windows, %d patients",
                len(clinic), clinic["patient_id"].nunique())

    comp_df, pts_any = build_composite_dataset(clinic)

    # --- Summary statistics ---
    n_pts_total = clinic["patient_id"].nunique()
    n_pts_any   = len(pts_any)
    n_hypo_only = len(
        set(clinic[(clinic["event_type"]=="hypotension") & (clinic["label"]==1)]["patient_id"])
        - set(clinic[(clinic["event_type"]=="hypertension") & (clinic["label"]==1)]["patient_id"])
    )
    n_hyper_only = len(
        set(clinic[(clinic["event_type"]=="hypertension") & (clinic["label"]==1)]["patient_id"])
        - set(clinic[(clinic["event_type"]=="hypotension") & (clinic["label"]==1)]["patient_id"])
    )
    n_both = len(
        set(clinic[(clinic["event_type"]=="hypotension") & (clinic["label"]==1)]["patient_id"])
        & set(clinic[(clinic["event_type"]=="hypertension") & (clinic["label"]==1)]["patient_id"])
    )

    n_comp_events = int(comp_df["composite"].sum())
    n_comp_ctrl   = int((comp_df["composite"] == 0).sum())
    epv_comp      = round(n_comp_events / 8, 1)

    logger.info("Composite events: %d  Controls: %d  EPV(8f): %.1f",
                n_comp_events, n_comp_ctrl, epv_comp)
    logger.info("Patients with hypo only: %d  hyper only: %d  both: %d  any: %d / %d",
                n_hypo_only, n_hyper_only, n_both, n_pts_any, n_pts_total)

    # --- Training-apparent AUC ---
    # Refit on full data (for training-apparent)
    feat_cols = [c for c in PARSIMONIOUS_FEATURES_HYPO if c in comp_df.columns]
    m_all = MixedLogisticModel(feature_cols=feat_cols)
    m_all.fit(comp_df[feat_cols], comp_df["composite"].values, comp_df["patient_id"].values)
    p_all = m_all.predict_proba(comp_df[feat_cols])
    auc_apparent = roc_auc_score(comp_df["composite"], p_all)
    logger.info("Training-apparent AUC: %.4f", auc_apparent)

    # --- 5-fold CV AUC ---
    logger.info("Running %d-fold CV...", N_FOLDS)
    cv_res = cv_auc_composite(comp_df, feat_cols, n_folds=N_FOLDS)
    logger.info("CV AUC: %.4f [%.3f–%.3f]", cv_res["cv_auc"], cv_res["ci_lo"], cv_res["ci_hi"])

    results = {
        "composite_definition": "hypotension OR hypertension event in 30-min horizon",
        "n_patients_total": n_pts_total,
        "n_patients_any_event": n_pts_any,
        "n_patients_hypo_only": n_hypo_only,
        "n_patients_hyper_only": n_hyper_only,
        "n_patients_both": n_both,
        "n_events": n_comp_events,
        "n_controls": n_comp_ctrl,
        "epv_8features": epv_comp,
        "auc_training_apparent": round(auc_apparent, 4),
        "auc_cv_5fold": cv_res,
        "model": "GLMM parsimonious hypotension features (8 features)",
        "note": (
            "Composite uses hypotension feature set as proxy; "
            "a dedicated joint model would be needed for a proper composite predictor. "
            "Results reported here are exploratory."
        ),
    }

    out_path = OUT_DIR / "composite_outcome_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    logger.info("Saved: %s", out_path)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
