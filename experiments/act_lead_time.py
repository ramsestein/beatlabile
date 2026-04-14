"""Lead time analysis — AUC of M1 (GLMM signal-only) vs prediction horizon.

Methodology
-----------
Clinic prediction windows end at event_onset_s (lead time = 0).  Each window
has 30-minute aggregate features including *_slope terms (rate of change per
second).  To approximate the physiological state L minutes BEFORE event onset:

  For mean/min/max features (level variables):
      f̂(–L) ≈ f(0) – slope · L · 60  [linear back-extrapolation]

  For std and slope features:
      f̂(–L) = f(0)  [std = within-window variability; assumed constant]

This is an approximation valid for slowly varying signals over 30-minute
prediction-window durations, and is labeled accordingly in all outputs.

Lead times evaluated: 0, 5, 10, 15, 20, 30 minutes.

For each lead time:
  · Event windows  (label=1) → extrapolated features applied
  · Control windows (label=0) → unmodified (no event anchor to extrapolate from)
  · AUC + 95% CI computed via cluster-bootstrap (B=500, resample patients)

Outputs
-------
results/lead_time/lead_time_auc.csv  — lead_min × event_type → AUC + CI
results/lead_time/fig_lead_time.pdf  — AUC vs lead time line plot

Run
---
python experiments/act_lead_time.py
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Monkey-patch patsy so statsmodels pickling works outside interactive sessions
import patsy.eval as _pe
_orig_capture = _pe.EvalEnvironment.capture.__func__ if hasattr(_pe.EvalEnvironment.capture, '__func__') else None

def _safe_capture(cls, eval_env=0, reference=0):  # type: ignore[override]
    try:
        if _orig_capture is not None:
            return _orig_capture(cls, eval_env, reference=reference)
        return _pe.EvalEnvironment.capture.__wrapped__(eval_env, reference=reference)
    except (AttributeError, TypeError):
        return _pe.EvalEnvironment([{}])

_pe.EvalEnvironment.capture = classmethod(_safe_capture)

from beatlabile.config import RESULTS_DIR
from beatlabile.models.mixed_logistic import MixedLogisticModel
from experiments.pipeline import get_feature_cols, EVENT_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = RESULTS_DIR / "cache"
ACT1_DIR = RESULTS_DIR / "act1"
OUT_DIR = RESULTS_DIR / "lead_time"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAD_TIMES_MIN = [0, 5, 10, 15, 20, 30]
B_BOOTSTRAP = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Feature extrapolation
# ---------------------------------------------------------------------------

def _extrapolate_features(
    df: pd.DataFrame,
    feat_cols: list[str],
    lead_s: float,
) -> pd.DataFrame:
    """Back-extrapolate *_mean/_min/_max features by lead_s seconds.

    For each HRV metric `m`, if columns `m_mean`, `m_slope` exist:
        m_mean_new  = m_mean  – m_slope · lead_s
        m_min_new   = m_min   – m_slope · lead_s
        m_max_new   = m_max   – m_slope · lead_s
    std and slope features are left unchanged.

    Only event windows (label=1) are modified; controls are returned as-is.
    """
    if lead_s == 0.0:
        return df.copy()

    result = df.copy()

    # Identify base metrics (columns whose name ends in _mean)
    mean_cols = [c for c in feat_cols if c.endswith("_mean")]
    base_metrics = [c.replace("_mean", "") for c in mean_cols]

    event_mask = result["label"] == 1

    for m in base_metrics:
        slope_col = f"{m}_slope"
        if slope_col not in result.columns:
            continue
        slope = result.loc[event_mask, slope_col]
        delta = slope * lead_s
        for suffix in ("_mean", "_min", "_max"):
            col = f"{m}{suffix}"
            if col in result.columns:
                result.loc[event_mask, col] -= delta

    return result


# ---------------------------------------------------------------------------
# Bootstrap AUC with patient-level resampling
# ---------------------------------------------------------------------------

def _bootstrap_auc(
    preds: np.ndarray,
    labels: np.ndarray,
    patients: np.ndarray,
    B: int = B_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Return (auc, ci_lo, ci_hi) using cluster-bootstrap over patients."""
    rng = np.random.default_rng(seed)
    unique_pts = np.unique(patients)
    n_pts = len(unique_pts)

    point_auc = roc_auc_score(labels, preds)
    boot_aucs: list[float] = []

    for _ in range(B):
        boot_pts = rng.choice(unique_pts, size=n_pts, replace=True)
        idx = np.concatenate([np.where(patients == p)[0] for p in boot_pts])
        y_b = labels[idx]
        p_b = preds[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, p_b))

    if len(boot_aucs) < 10:
        return point_auc, float("nan"), float("nan")

    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))
    return point_auc, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Out-of-sample CV predictions (patient-level, same scheme as Act 1)
# ---------------------------------------------------------------------------

def _oos_predictions(
    df: pd.DataFrame,
    feat_cols: list[str],
    lead_s: float,
    n_folds: int = 10,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return out-of-sample (preds, labels, patient_ids) via patient-level k-fold CV.

    For each test fold:
      1. Extrapolate features of EVENT windows by lead_s seconds (linear trend).
      2. Train a fresh GLMM on all train-fold windows (unshifted — represents
         the model that would have been built without knowing the lead time).
      3. Predict on shifted test-fold windows.

    This ensures AUC at lead=0 matches Act 1 CV AUC, with no in-sample leakage.
    """
    rng = np.random.default_rng(seed)
    unique_pts = np.array(sorted(df["patient_id"].unique()))
    perm = rng.permutation(len(unique_pts))
    unique_pts = unique_pts[perm]
    fold_map = {p: i % n_folds for i, p in enumerate(unique_pts)}
    df = df.copy()
    df["_fold"] = df["patient_id"].map(fold_map)

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_patients: list[np.ndarray] = []

    for fold in range(n_folds):
        train_mask = df["_fold"] != fold
        test_mask  = df["_fold"] == fold
        train_df = df[train_mask]
        test_df  = df[test_mask]

        if len(np.unique(train_df["label"])) < 2:
            continue

        # Train on unshifted train windows
        model = MixedLogisticModel(feature_cols=feat_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                train_df[feat_cols],
                train_df["label"].values,
                train_df["patient_id"].values,
            )

        # Extrapolate test windows
        test_shifted = _extrapolate_features(test_df, feat_cols, lead_s)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = model.predict_proba(test_shifted[feat_cols])

        all_preds.append(preds)
        all_labels.append(test_shifted["label"].values)
        all_patients.append(test_shifted["patient_id"].values)

    if not all_preds:
        return np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_patients),
    )


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_lead_time_analysis() -> pd.DataFrame:
    """Run lead time analysis for all event types.  Returns results DataFrame.

    Methodology
    -----------
    Uses 10-fold patient-level CV (same scheme as Act 1) to produce
    out-of-sample predictions at each lead time.  AUC at lead=0 is
    therefore directly comparable to Act 1 CV AUC (no in-sample leakage).

    For lead > 0: event-window features are back-extrapolated using slope
    coefficients before scoring.  Control windows are unmodified.
    """

    rows: list[dict] = []

    for etype in EVENT_TYPES:
        # Load clinic windows for this event type
        win_path = CACHE_DIR / "clinic_windows.parquet"
        if not win_path.exists():
            logger.error("Clinic windows cache not found: %s", win_path)
            break

        df = pd.read_parquet(win_path)
        df = df[df["event_type"] == etype].copy()

        if df.empty:
            logger.warning("No windows for event type %s.", etype)
            continue

        if "patient_id" not in df.columns:
            logger.warning("No patient_id in windows for %s.", etype)
            continue

        feat_cols = get_feature_cols(df)
        n_events_total = int(df["label"].sum())
        n_pts = int(df["patient_id"].nunique())
        logger.info(
            "%s: %d windows (%d events, %d patients, %d features)",
            etype, len(df), n_events_total, n_pts, len(feat_cols),
        )

        for lead_min in LEAD_TIMES_MIN:
            lead_s = lead_min * 60.0
            logger.info("  %s lead %d min — running 10-fold OOS CV...", etype, lead_min)

            preds, y, patients = _oos_predictions(df, feat_cols, lead_s)

            if len(preds) == 0 or len(np.unique(y)) < 2:
                logger.warning("  Lead %d min — insufficient OOS data.", lead_min)
                continue

            auc, ci_lo, ci_hi = _bootstrap_auc(preds, y, patients)
            n_evt = int(y.sum())
            n_win = len(preds)

            logger.info(
                "  %s lead %2d min → AUC=%.3f [%.3f–%.3f]  (n_events=%d)",
                etype, lead_min, auc, ci_lo, ci_hi, n_evt,
            )
            rows.append({
                "event_type": etype,
                "lead_min": lead_min,
                "auc": round(auc, 4),
                "ci_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
                "ci_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
                "n_events": n_evt,
                "n_windows": n_win,
                "method": "oos_cv_linear_extrap" if lead_min > 0 else "oos_cv",
            })

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        csv_out = OUT_DIR / "lead_time_auc.csv"
        result_df.to_csv(csv_out, index=False)
        logger.info("Saved: %s", csv_out)
    return result_df


def plot_lead_time(result_df: pd.DataFrame) -> None:
    """Line plot of AUC vs lead time, one line per event type."""
    if result_df.empty:
        logger.warning("No results to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"hypotension": "#d62728", "hypertension": "#1f77b4", "variability": "#2ca02c"}
    labels_map = {"hypotension": "Hipotensión", "hypertension": "Hipertensión", "variability": "Variabilidad"}

    for etype in EVENT_TYPES:
        sub = result_df[result_df["event_type"] == etype].sort_values("lead_min")
        if sub.empty:
            continue
        x = sub["lead_min"].values
        y = sub["auc"].values
        lo = sub["ci_lo"].values.astype(float)
        hi = sub["ci_hi"].values.astype(float)
        col = colors.get(etype, "gray")
        ax.plot(x, y, "o-", color=col, label=labels_map.get(etype, etype), linewidth=2, markersize=6)
        valid_ci = ~(np.isnan(lo) | np.isnan(hi))
        if np.any(valid_ci):
            ax.fill_between(x[valid_ci], lo[valid_ci], hi[valid_ci], alpha=0.15, color=col)

    ax.axhline(0.65, color="gray", linestyle="--", linewidth=1, label="AUC=0.65 (umbral clínico)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Tiempo de antelación (minutos)", fontsize=12)
    ax.set_ylabel("AUC (M1 – GLMM señal)", fontsize=12)
    ax.set_title("AUC vs horizonte temporal de predicción\n(extrapolación lineal de tendencia)", fontsize=12)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xticks(LEAD_TIMES_MIN)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig_out = OUT_DIR / "fig_lead_time.pdf"
    fig.tight_layout()
    fig.savefig(fig_out, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", fig_out)


if __name__ == "__main__":
    result_df = run_lead_time_analysis()
    if not result_df.empty:
        plot_lead_time(result_df)
        print("\nLead time analysis complete:")
        print(result_df.to_string(index=False))
    else:
        print("No results generated — check that Act 1 models are trained.")
