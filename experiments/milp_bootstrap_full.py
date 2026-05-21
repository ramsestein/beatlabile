"""Re-run MILP bootstrap stability with B=500 (publication standard).

The MILP tree itself is unchanged — only the bootstrap evaluation is re-run.
Updates:
  - results/act1/milp_stability_{etype}.csv  (500-rep frequencies)
  - results/act1/act1_results.json            (milp_bootstrap_auc_mean)

Run
---
nohup .venv/bin/python3 experiments/milp_bootstrap_full.py > milp_boot500.log 2>&1 &
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from beatlabile.config import CFG, RESULTS_DIR
from beatlabile.models.milp_tree import MILPTree
from experiments.pipeline import get_feature_cols, EVENT_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "act1"
CACHE_DIR = RESULTS_DIR / "cache"
N_BOOT = 500


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def run_milp_bootstrap_full() -> None:
    # ------------------------------------------------------------------ #
    # Load cached Clínic windows
    # ------------------------------------------------------------------ #
    cache_file = CACHE_DIR / "clinic_windows.parquet"
    if not cache_file.exists():
        logger.error("Cache not found at %s. Run act1_clinic.py first.", cache_file)
        return
    windows_df = pd.read_parquet(cache_file)
    logger.info("Loaded cache: %d windows", len(windows_df))

    # ------------------------------------------------------------------ #
    # Build CFG with B=500
    # ------------------------------------------------------------------ #
    cfg500 = copy.deepcopy(CFG)
    cfg500["models"]["milp"]["bootstrap_reps"] = N_BOOT
    logger.info("Bootstrap B = %d", N_BOOT)

    # ------------------------------------------------------------------ #
    # Load act1_results.json to update in place
    # ------------------------------------------------------------------ #
    results_json_path = OUT_DIR / "act1_results.json"
    with open(results_json_path) as fh:
        results = json.load(fh)

    # ------------------------------------------------------------------ #
    # Per event type
    # ------------------------------------------------------------------ #
    for etype in EVENT_TYPES:
        logger.info("=== %s ===", etype)
        sub = windows_df[windows_df["event_type"] == etype].copy()
        if len(sub) == 0:
            logger.warning("No windows for %s, skipping.", etype)
            continue

        feat_cols = get_feature_cols(sub)
        X = sub[feat_cols]
        y = sub["label"].values

        n_events = int(np.sum(y))
        logger.info("  n=%d  events=%d  controls=%d", len(y), n_events, int(np.sum(y == 0)))

        # Load existing fitted MILP tree
        milp_pkl = OUT_DIR / f"milp_{etype}.pkl"
        with open(milp_pkl, "rb") as fh:
            milp = pickle.load(fh)
        logger.info("  Loaded MILP pkl: %s", milp_pkl.name)

        # Re-run bootstrap with B=500
        logger.info("  Running bootstrap_stability (B=%d)...", N_BOOT)
        stability = milp.bootstrap_stability(X, y, etype, cfg500)
        boot_auc_mean = float(np.mean(stability.bootstrap_auc)) if len(stability.bootstrap_auc) > 0 else float("nan")
        boot_auc_std  = float(np.std(stability.bootstrap_auc))  if len(stability.bootstrap_auc) > 0 else float("nan")
        logger.info("  Bootstrap AUC: %.3f ± %.3f  (n_valid_boots=%d)",
                    boot_auc_mean, boot_auc_std, len(stability.bootstrap_auc))

        # Save updated stability CSV
        stab_rows = []
        for feat, freq in stability.feature_freq.items():
            if freq > 0:
                thresh_arr = stability.threshold_distributions.get(feat, np.array([]))
                stab_rows.append({
                    "feature": feat,
                    "freq": freq,
                    "thresh_mean": float(np.mean(thresh_arr)) if len(thresh_arr) > 0 else np.nan,
                    "thresh_std":  float(np.std(thresh_arr))  if len(thresh_arr) > 0 else np.nan,
                })
        stab_df = pd.DataFrame(stab_rows).sort_values("freq", ascending=False)
        stab_csv = OUT_DIR / f"milp_stability_{etype}.csv"
        stab_df.to_csv(stab_csv, index=False)
        logger.info("  Saved: %s", stab_csv)

        # Update results dict
        if etype not in results:
            results[etype] = {}
        results[etype]["milp_bootstrap_auc_mean"] = boot_auc_mean
        results[etype]["milp_bootstrap_auc_std"]  = boot_auc_std
        results[etype]["milp_bootstrap_n_boot"]   = N_BOOT
        results[etype]["milp_bootstrap_n_valid"]  = int(len(stability.bootstrap_auc))

    # ------------------------------------------------------------------ #
    # Save updated results JSON
    # ------------------------------------------------------------------ #
    with open(results_json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    logger.info("Updated %s", results_json_path)
    logger.info("=== DONE — MILP bootstrap B=%d complete ===", N_BOOT)


if __name__ == "__main__":
    run_milp_bootstrap_full()
