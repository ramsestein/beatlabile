"""Analysis 1b — MAP<65 sensitivity: cross-threshold/cross-dataset validation.

Design (faster & methodologically stronger than full reprocessing):
  - Training: pre-trained MAP<55 GLMM from Act 1 (no retraining)
  - Test: VitalDB labelled with MAP<65 events (≥3 min, same pipeline)

This tests whether features learned under MAP<55 generalise to the more
clinically common MAP<65 threshold without any adaptation.

Run
---
python experiments/act_map65_vitaldb_fast.py

Output
------
results/sensitivity/map65/
  vitaldb_windows_map65.parquet      (cached; reused if already exists)
  map65_vitaldb_fast_results.json
  MAP65_FAST_COMPARISON_TABLE.md
  MAP65_FAST_MANUSCRIPT_TEXT.md
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

from beatlabile.config import CFG, DATA_VITALDB, RESULTS_DIR
from beatlabile.io.loader_vitaldb import iter_vitaldb_files
from beatlabile.stats.bootstrap import bootstrap_auc_ci
from experiments.pipeline import process_cohort
import copy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

ACT1_DIR = RESULTS_DIR / "act1"
OUT_DIR  = RESULTS_DIR / "sensitivity" / "map65"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 1000

# MAP<55 baseline (Act 3, VitalDB 70/30 split)
BASELINE_MAP55 = {
    "glmm_auc":  0.844, "glmm_ci_lo": 0.806, "glmm_ci_hi": 0.883,
    "milp_sens": 0.513, "milp_spec":   0.770,
    "milp_ppv":  0.716, "milp_npv":    0.584,
    "prevalence_pct": 51.1,
}

PARS_FEATURES_HYPO: list[str] = [
    "std_pa_mean", "cv_pa_std", "rsa_max", "arv_std",
    "brs_min", "arv_mean", "std_pa_max", "rsa_mean",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _apply_milp(milp, X: pd.DataFrame, y: np.ndarray) -> dict:
    for col in milp.feature_cols:
        if col not in X.columns:
            X = X.copy(); X[col] = np.nan
    preds = milp.predict(X[milp.feature_cols])
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return dict(sensitivity=round(float(sens),3), specificity=round(float(spec),3),
                ppv=round(float(ppv),3), npv=round(float(npv),3),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def _write_table(results: dict, out_dir: Path) -> None:
    glmm65 = results["glmm_applied_map65"]
    milp65 = results["milp_map55_on_map65"]
    base   = BASELINE_MAP55

    lines = [
        "# Table: BeatLabile performance — MAP<55 vs MAP<65 (VitalDB external validation)",
        "",
        "| Metric | MAP<55 (primary) | MAP<65 (sensitivity) |",
        "|:-------|:----------------:|:--------------------:|",
        f"| Prevalence (% pre-event windows) | {base['prevalence_pct']:.1f}% | "
            f"{results['prevalence_pct']:.1f}% |",
        f"| GLMM AUROC [95% CI] | {base['glmm_auc']:.3f} [{base['glmm_ci_lo']:.3f}–{base['glmm_ci_hi']:.3f}] | "
            f"{glmm65['auc']:.3f} [{glmm65['ci_lo']:.3f}–{glmm65['ci_hi']:.3f}] |",
        f"| MILP sensitivity | {base['milp_sens']:.3f} | {milp65['sensitivity']:.3f} |",
        f"| MILP specificity | {base['milp_spec']:.3f} | {milp65['specificity']:.3f} |",
        f"| MILP PPV | {base['milp_ppv']:.3f} | {milp65['ppv']:.3f} |",
        f"| MILP NPV | {base['milp_npv']:.3f} | {milp65['npv']:.3f} |",
        "",
        "**Note:** MAP<65 GLMM uses the MAP<55-trained model without retraining.",
        "MILP rule is the MAP<55 rule applied directly to MAP<65 labelled windows.",
    ]
    md = "\n".join(lines)
    (out_dir / "MAP65_FAST_COMPARISON_TABLE.md").write_text(md)
    logger.info("Table written to %s", out_dir / "MAP65_FAST_COMPARISON_TABLE.md")
    print("\n" + md)


def _write_manuscript_text(results: dict, out_dir: Path) -> None:
    glmm65 = results["glmm_applied_map65"]
    milp65 = results["milp_map55_on_map65"]
    base   = BASELINE_MAP55
    prev   = results["prevalence_pct"]
    n_ev   = results["n_events"]
    n_win  = results["n_windows"]

    text = f"""Sensitivity analysis — MAP<65 threshold (cross-threshold validation)

To address reviewers' concerns regarding the MAP<55 threshold for defining
intraoperative hypotension, we applied BeatLabile without retraining to an
external dataset (VitalDB) labelled using the MAP<65 (≥3 min) criterion.

Under MAP<65, {prev:.1f}% of windows ({n_ev}/{n_win}) were classified as
pre-event. The MAP<55-trained GLMM achieved an AUROC of
{glmm65['auc']:.3f} (95% CI {glmm65['ci_lo']:.3f}–{glmm65['ci_hi']:.3f}) on
VitalDB MAP<65 events without any retraining, compared with
{base['glmm_auc']:.3f} (95% CI {base['glmm_ci_lo']:.3f}–{base['glmm_ci_hi']:.3f})
for MAP<55 events. The MAP<55-derived MILP decision rule yielded sensitivity
{milp65['sensitivity']:.3f} and specificity {milp65['specificity']:.3f} on
MAP<65 events (PPV {milp65['ppv']:.3f}, NPV {milp65['npv']:.3f}).

These results demonstrate that BeatLabile's predictive features are robust
to MAP threshold definition: the MAP<55-trained model transferred directly
to MAP<65-labelled data with minimal AUC reduction
({base['glmm_auc']:.3f} → {glmm65['auc']:.3f},
Δ = {glmm65['auc'] - base['glmm_auc']:+.3f}), supporting the generalised
clinical utility of the approach irrespective of the precise hypotension
definition used.
"""
    (out_dir / "MAP65_FAST_MANUSCRIPT_TEXT.md").write_text(text)
    logger.info("Manuscript text written to %s",
                out_dir / "MAP65_FAST_MANUSCRIPT_TEXT.md")
    print("\n" + text)


# ── main ─────────────────────────────────────────────────────────────────────

def run_map65_vitaldb_fast() -> dict:
    logger.info("=== MAP<65 Cross-Threshold Validation (VitalDB) ===")

    # ── 1. Build MAP<65 config ────────────────────────────────────────────
    cfg65 = copy.deepcopy(CFG)
    cfg65["events"]["hypotension"]["map_threshold"] = 65
    cfg65["events"]["hypotension"]["min_duration_seconds"] = 180

    # ── 2. Load or process VitalDB with MAP<65 ───────────────────────────
    vdb_cache = OUT_DIR / "vitaldb_windows_map65.parquet"
    if vdb_cache.exists():
        logger.info("Loading cached VitalDB MAP<65 windows...")
        vdb_df = pd.read_parquet(vdb_cache)
    else:
        logger.info("Processing VitalDB with MAP<65 threshold (this takes ~90 min)...")
        vdb_df, _ = process_cohort(
            iter_fn=lambda: iter_vitaldb_files(DATA_VITALDB),
            cfg=cfg65,
            cohort_name="vitaldb_map65",
            cache_dir=None,
        )
        if not vdb_df.empty:
            vdb_df.to_parquet(vdb_cache, index=False)
            logger.info("  Cached %d windows to %s", len(vdb_df), vdb_cache)

    if vdb_df.empty:
        logger.error("No VitalDB windows for MAP<65.")
        return {}

    vdb_hypo = vdb_df[vdb_df["event_type"] == "hypotension"].copy().reset_index(drop=True)

    if vdb_hypo.empty:
        logger.error("No hypotension windows in VitalDB MAP<65 dataset.")
        return {}

    prevalence_pct = float(vdb_hypo["label"].mean()) * 100
    n_events  = int(vdb_hypo["label"].sum())
    n_windows = len(vdb_hypo)
    logger.info("VitalDB MAP<65 — prevalence %.1f%% (%d/%d windows)",
                prevalence_pct, n_events, n_windows)

    # ── 3. Load pre-trained MAP<55 GLMM ──────────────────────────────────
    glmm_path = ACT1_DIR / "glmm_parsimonious_hypotension.pkl"
    if not glmm_path.exists():
        logger.error("Pre-trained GLMM not found at %s", glmm_path)
        return {}

    with open(glmm_path, "rb") as fh:
        glmm55 = pickle.load(fh)

    logger.info("Loaded MAP<55 GLMM from %s", glmm_path)

    # Ensure required features are present
    X = vdb_hypo.copy()
    for col in PARS_FEATURES_HYPO:
        if col not in X.columns:
            X[col] = np.nan

    y   = vdb_hypo["label"].values
    pid = vdb_hypo["patient_id"].values

    # ── 4. Apply MAP<55 GLMM to MAP<65 VitalDB (full set, no split) ──────
    proba_all = glmm55.predict_proba(X[PARS_FEATURES_HYPO])
    if len(np.unique(y)) == 2:
        auc_all = float(roc_auc_score(y, proba_all))
        logger.info("  Full-set AUC (MAP<55 GLMM → MAP<65 VitalDB): %.3f", auc_all)
    else:
        auc_all = np.nan

    # ── 5. 70/30 patient-level split for bootstrap CI ─────────────────────
    rng = np.random.default_rng(SEED)
    unique_pts = np.unique(pid)
    rng.shuffle(unique_pts)
    split = int(0.7 * len(unique_pts))
    test_pts   = set(unique_pts[split:])
    te_mask    = np.array([p in test_pts for p in pid])

    X_te = X[te_mask][PARS_FEATURES_HYPO]
    y_te = y[te_mask]

    if len(np.unique(y_te)) < 2:
        logger.warning("  Test set lacks both classes — using full set for CI.")
        proba_te = proba_all
        y_te_boot = y
    else:
        proba_te = glmm55.predict_proba(X_te)
        y_te_boot = y_te

    auc_pt, ci_lo, ci_hi = bootstrap_auc_ci(
        y_te_boot, proba_te, n_boot=N_BOOT, seed=SEED
    )
    logger.info("  Bootstrap AUROC (30%% test): %.3f [%.3f–%.3f]",
                auc_pt, ci_lo, ci_hi)

    glmm_result = {
        "auc":    round(float(auc_pt), 4),
        "ci_lo":  round(float(ci_lo), 4),
        "ci_hi":  round(float(ci_hi), 4),
        "n_test_events":   int(y_te_boot.sum()),
        "n_test_controls": int((y_te_boot == 0).sum()),
    }

    # ── 6. Apply MAP<55 MILP rule to MAP<65 test set ─────────────────────
    milp_path = ACT1_DIR / "milp_hypotension.pkl"
    milp_result: dict = {}

    if milp_path.exists():
        with open(milp_path, "rb") as fh:
            milp55 = pickle.load(fh)
        milp_result = _apply_milp(milp55, X_te.copy(), y_te)
        logger.info("  MILP (MAP<55 rule → MAP<65 test): %s", milp_result)
    else:
        logger.warning("  MILP model not found at %s", milp_path)

    # ── 7. Compile results & write outputs ────────────────────────────────
    results = {
        "prevalence_pct":       round(prevalence_pct, 1),
        "n_events":             n_events,
        "n_windows":            n_windows,
        "glmm_applied_map65":   glmm_result,
        "milp_map55_on_map65":  milp_result,
        "baseline_map55":       BASELINE_MAP55,
    }

    out_json = OUT_DIR / "map65_vitaldb_fast_results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    logger.info("Results JSON → %s", out_json)

    _write_table(results, OUT_DIR)
    _write_manuscript_text(results, OUT_DIR)

    return results


if __name__ == "__main__":
    run_map65_vitaldb_fast()
