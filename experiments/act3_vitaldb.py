"""Act 3 — Revalidation + Sufficiency Test on VitalDB (Section 4.3 / 9.5).

Two sub-analysis:
  3a. Signal-only validation (same as Act 2)
  3b. Sufficiency test: models M1–M6 progressively adding clinical context

Models M1–M6 (Table in Section 9.5):
  M1: Signal-only (Act 1 coefficients, no refit)
  M2: Signal refit on VitalDB
  M3: M2 + demographics (age, sex, BMI)
  M4: M3 + clinical (ASA, optype, HTA, DM, emop)
  M5: M4 + lab (Hb, Cr, glucose, K)
  M6: Clinical-only (demographics + clinical + lab, no signal)

Subgroups: AUC of M1 by age tercile, sex, ASA, optype, BMI group.

Run
---
python experiments/act3_vitaldb.py
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from beatlabile.config import CFG, DATA_VITALDB, RESULTS_DIR
from beatlabile.io.loader_vitaldb import iter_vitaldb_files, load_clinical_info, load_labs
from beatlabile.models.mixed_logistic import MixedLogisticModel, compute_nri_idi
from beatlabile.models.milp_tree import MILPTree
from experiments.pipeline import process_cohort, get_feature_cols, EVENT_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ACT1_DIR = RESULTS_DIR / "act1"
OUT_DIR = RESULTS_DIR / "act3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cache"

# Clinical columns available in VitalDB
DEMO_COLS = ["age", "bmi"]
CLINICAL_COLS = ["asa", "preop_htn", "preop_dm", "emop"]
# Actual lab column names in vitaldb labs.csv (long format 'name' values)
LAB_COLS = ["hb", "cr", "gluc", "k"]


def _fit_model_worker(
    args: tuple[str, list[str], np.ndarray, np.ndarray, np.ndarray],
) -> tuple[str, dict]:
    """Top-level picklable worker: fits one GLMM and returns (name, auc_dict)."""
    import warnings
    warnings.filterwarnings("ignore")
    name, feat_cols, X_arr, y, patient_ids = args
    X_df = pd.DataFrame(X_arr, columns=feat_cols)
    m = MixedLogisticModel(feature_cols=feat_cols)
    result = _quick_cv_auc(m, X_df, y, patient_ids)
    return name, result


def run_act3() -> dict:
    """Main entry point for Act 3. Returns summary dict."""
    logger.info("=== ACT 3: VitalDB Revalidation + Sufficiency Test ===")

    # ------------------------------------------------------------------ #
    # 1. Process VitalDB waveforms
    # ------------------------------------------------------------------ #
    windows_df, events_df = process_cohort(
        iter_fn=lambda: iter_vitaldb_files(DATA_VITALDB),
        cfg=CFG,
        cohort_name="vitaldb",
        cache_dir=CACHE_DIR,
    )

    if windows_df.empty:
        logger.error("No windows extracted from VitalDB cohort.")
        return {}

    events_df.to_csv(OUT_DIR / "vitaldb_events.csv", index=False)

    # ------------------------------------------------------------------ #
    # 2. Load clinical metadata
    # ------------------------------------------------------------------ #
    clinical_df = load_clinical_info(DATA_VITALDB)
    labs_df = load_labs(DATA_VITALDB)

    if clinical_df is not None:
        # Normalise merge key: VitalDB files use zero-padded strings ("0013")
        # while cases.csv uses integer caseid → align both to zero-padded str
        if "caseid" in clinical_df.columns and "patient_id" not in clinical_df.columns:
            clinical_df = clinical_df.rename(columns={"caseid": "patient_id"})
        if "patient_id" in clinical_df.columns:
            clinical_df["patient_id"] = (
                clinical_df["patient_id"].astype(str).str.zfill(4)
            )
        # Compute BMI if not present
        if "bmi" not in clinical_df.columns and "weight" in clinical_df.columns and "height" in clinical_df.columns:
            h_m = clinical_df["height"] / 100.0
            clinical_df["bmi"] = clinical_df["weight"] / (h_m**2)
        windows_df = windows_df.merge(
            clinical_df, on="patient_id", how="left", suffixes=("", "_clin")
        )

    if labs_df is not None:
        if "caseid" in labs_df.columns and "patient_id" not in labs_df.columns:
            labs_df = labs_df.rename(columns={"caseid": "patient_id"})
        if "patient_id" in labs_df.columns:
            labs_df["patient_id"] = labs_df["patient_id"].astype(str).str.zfill(4)
        # Long → wide: median per patient per test name to avoid row explosion
        if {"name", "result"}.issubset(labs_df.columns):
            labs_wide = (
                labs_df.groupby(["patient_id", "name"])["result"]
                .median()
                .unstack("name")
                .reset_index()
            )
            windows_df = windows_df.merge(labs_wide, on="patient_id", how="left")
        else:
            windows_df = windows_df.merge(
                labs_df.drop_duplicates("patient_id"),
                on="patient_id", how="left", suffixes=("", "_lab")
            )

    results: dict = {}

    for etype in EVENT_TYPES:
        logger.info("--- Event type: %s ---", etype)
        sub = windows_df[windows_df["event_type"] == etype].copy().reset_index(drop=True)
        if len(sub) == 0:
            continue

        feat_cols = get_feature_cols(sub)
        X_signal = sub[feat_cols]
        y = sub["label"].values
        patient_ids = sub["patient_id"].values

        if len(np.unique(y)) < 2:
            logger.warning("  Single class in %s. Skipping.", etype)
            continue

        # ------------------------------------------------------------ #
        # 3a. Signal-only (M1) — apply Act 1 model without refit
        # ------------------------------------------------------------ #
        glmm_path = ACT1_DIR / f"glmm_{etype}.pkl"
        m1_auc = np.nan
        if glmm_path.exists():
            with open(glmm_path, "rb") as fh:
                glmm_m1: MixedLogisticModel = pickle.load(fh)
            for col in glmm_m1.feature_cols:
                if col not in X_signal.columns:
                    X_signal[col] = np.nan
            m1_auc = float(roc_auc_score(y, glmm_m1.predict_proba(X_signal)))
            logger.info("  M1 (signal-only, no refit) AUC=%.3f", m1_auc)

        # ------------------------------------------------------------ #
        # 3b. Sufficiency test: M2–M6 in parallel
        # ------------------------------------------------------------ #
        m_aucs = {"M1": m1_auc}

        num_cols = set(sub.select_dtypes(include="number").columns)
        demo_avail = [c for c in DEMO_COLS if c in sub.columns and c in num_cols]
        clin_avail = [c for c in CLINICAL_COLS if c in sub.columns and c in num_cols]
        lab_avail  = [c for c in LAB_COLS if c in sub.columns and c in num_cols]

        # Build (model_name, feature_cols, X_df) for each model
        model_specs: list[tuple[str, list[str], pd.DataFrame]] = [
            ("M2", feat_cols, X_signal),
        ]
        if demo_avail:
            X_m3 = pd.concat([X_signal, sub[demo_avail]], axis=1)
            model_specs.append(("M3", list(X_m3.columns), X_m3))
        if demo_avail + clin_avail:
            X_m4 = pd.concat([X_signal, sub[demo_avail + clin_avail]], axis=1)
            model_specs.append(("M4", list(X_m4.columns), X_m4))
        if demo_avail + clin_avail + lab_avail:
            X_m5 = pd.concat([X_signal, sub[demo_avail + clin_avail + lab_avail]], axis=1)
            model_specs.append(("M5", list(X_m5.columns), X_m5))
        if demo_avail + clin_avail + lab_avail:
            all_clin = demo_avail + clin_avail + lab_avail
            model_specs.append(("M6", all_clin, sub[all_clin]))

        args_list = [
            (name, fcols, X_df.values, y, patient_ids)
            for name, fcols, X_df in model_specs
        ]
        n_workers = min(len(args_list), mp.cpu_count())
        logger.info("  Fitting M2–M6 in parallel (%d workers)...", n_workers)
        with mp.Pool(processes=n_workers) as pool:
            fit_results = pool.map(_fit_model_worker, args_list)

        for name, auc_dict in fit_results:
            m_aucs[name] = auc_dict
            logger.info("  %s AUC=%.3f [%.3f\u2013%.3f]",
                        name, auc_dict["auc"], auc_dict["ci_lo"], auc_dict["ci_hi"])

        # ------------------------------------------------------------ #
        # Subgroup analysis (M1 signal-only)
        # ------------------------------------------------------------ #
        subgroup_aucs = {}
        if glmm_path.exists() and m1_auc is not np.nan:
            subgroup_aucs = _subgroup_analysis(sub, X_signal, y, glmm_m1)
            logger.info("  Subgroup AUCs: %s", subgroup_aucs)

        results[etype] = {
            "sufficiency_models": m_aucs,
            "subgroup_auc": subgroup_aucs,
            "n_events": int(np.sum(y)),
            "n_controls": int(np.sum(y == 0)),
        }

    with open(OUT_DIR / "act3_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)

    logger.info("Act 3 complete. Results in %s", OUT_DIR)
    return results


def _quick_cv_auc(
    model: MixedLogisticModel,
    X: pd.DataFrame,
    y: np.ndarray,
    patient_ids: np.ndarray,
    n_folds: int = 5,  # kept for signature compatibility; not used
    n_boot: int = 1000,
) -> dict:
    """Single 70/30 patient-level hold-out AUC using GLMM + cluster bootstrap CI.

    Returns dict with keys: auc, ci_lo, ci_hi.
    Cluster bootstrap (B=n_boot) resamples at patient level to give valid CIs
    that account for within-patient window correlation.
    """
    rng = np.random.default_rng(42)
    unique_pts = np.unique(patient_ids)
    rng.shuffle(unique_pts)
    split = int(0.7 * len(unique_pts))
    train_pts = set(unique_pts[:split])

    tr = np.array([pid in train_pts for pid in patient_ids])
    te = ~tr

    if not np.any(te) or len(np.unique(y[te])) < 2:
        return {"auc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    m = MixedLogisticModel(feature_cols=model.feature_cols)
    m.fit(X.iloc[tr], y[tr], patient_ids[tr])
    proba_te = m.predict_proba(X.iloc[te])
    auc_point = float(roc_auc_score(y[te], proba_te))

    # Cluster bootstrap on test set only (resample patients)
    test_pts = np.array([pid for pid in unique_pts[split:]])
    boot_aucs: list[float] = []
    rng2 = np.random.default_rng(0)
    for _ in range(n_boot):
        sampled_pts = set(rng2.choice(test_pts, size=len(test_pts), replace=True))
        mask = np.array([pid in sampled_pts for pid in patient_ids[te]])
        yb = y[te][mask]
        pb = proba_te[mask]
        if len(np.unique(yb)) < 2:
            continue
        try:
            boot_aucs.append(float(roc_auc_score(yb, pb)))
        except Exception:
            continue

    if len(boot_aucs) >= 20:
        ci_lo = float(np.percentile(boot_aucs, 2.5))
        ci_hi = float(np.percentile(boot_aucs, 97.5))
    else:
        ci_lo = ci_hi = float("nan")

    return {"auc": auc_point, "ci_lo": ci_lo, "ci_hi": ci_hi}


def _subgroup_analysis(
    sub: pd.DataFrame,
    X_signal: pd.DataFrame,
    y: np.ndarray,
    model: MixedLogisticModel,
) -> dict[str, float]:
    """Compute M1 AUC within pre-defined subgroups."""
    result = {}
    proba = model.predict_proba(X_signal)

    # Age terciles
    if "age" in sub.columns and sub["age"].notna().sum() > 10:
        t33, t66 = np.nanpercentile(sub["age"], [33, 66])
        for label, mask in [
            ("age_low", sub["age"] <= t33),
            ("age_mid", (sub["age"] > t33) & (sub["age"] <= t66)),
            ("age_high", sub["age"] > t66),
        ]:
            _record_subgroup_auc(result, label, proba, y, mask.values)

    # Sex
    if "sex" in sub.columns:
        for val in sub["sex"].dropna().unique():
            mask = (sub["sex"] == val).values
            _record_subgroup_auc(result, f"sex_{val}", proba, y, mask)

    # ASA
    if "asa" in sub.columns:
        for grp_label, mask_expr in [
            ("asa_1_2", sub["asa"] <= 2),
            ("asa_3plus", sub["asa"] >= 3),
        ]:
            mask = mask_expr.values if hasattr(mask_expr, "values") else mask_expr
            _record_subgroup_auc(result, grp_label, proba, y, mask)

    # BMI
    if "bmi" in sub.columns and sub["bmi"].notna().sum() > 10:
        for bmi_label, lo, hi in [("bmi_normal", 0, 25), ("bmi_overweight", 25, 30), ("bmi_obese", 30, 999)]:
            mask = ((sub["bmi"] >= lo) & (sub["bmi"] < hi)).values
            _record_subgroup_auc(result, bmi_label, proba, y, mask)

    return result


def _record_subgroup_auc(
    result: dict, label: str, proba: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> None:
    if np.sum(mask) < 10 or len(np.unique(y[mask])) < 2:
        return
    result[label] = float(roc_auc_score(y[mask], proba[mask]))


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    run_act3()
