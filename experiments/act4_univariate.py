"""Act 4 — Univariate predictor validity & domain-shift robustness.

For each feature and each event type, computes the individual AUC as a
predictor across all available cohorts (Clínic, MIMIC, VitalDB).  The
cross-cohort AUC drop quantifies how well each variable retains its
predictive signal under domain shift.

Output (results/act4/)
----------------------
univariate_auc.csv       — long-format: feature × cohort × event_type → AUC
univariate_summary.csv   — wide-format: one row per feature, AUC in each cohort
domain_shift_rank.csv    — features ranked by AUC stability (min AUC across cohorts)

Run
---
python run_all.py --act 4
python experiments/act4_univariate.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from beatlabile.config import RESULTS_DIR
from experiments.pipeline import get_feature_cols, EVENT_TYPES
from beatlabile.stats.unsupervised import population_pca, cross_cohort_correlations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = RESULTS_DIR / "cache"
OUT_DIR = RESULTS_DIR / "act4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cohort → cache file name mapping
COHORT_CACHE: dict[str, str] = {
    "clinic": "clinic_windows.parquet",
    "mimic": "mimic_windows.parquet",
    "vitaldb": "vitaldb_windows.parquet",
}


def _univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """AUC for a single feature.  Returns max(auc, 1-auc) so direction-agnostic."""
    mask = ~np.isnan(x)
    if mask.sum() < 10 or len(np.unique(y[mask])) < 2:
        return np.nan
    auc = float(roc_auc_score(y[mask], x[mask]))
    return max(auc, 1 - auc)


def _compute_for_cohort(
    windows_df: pd.DataFrame,
    cohort: str,
) -> pd.DataFrame:
    """Return long-format DataFrame with univariate AUC per feature × event_type."""
    feat_cols = get_feature_cols(windows_df)
    records = []
    for etype in EVENT_TYPES:
        sub = windows_df[windows_df["event_type"] == etype]
        if len(sub) == 0:
            continue
        y = sub["label"].values
        if len(np.unique(y)) < 2:
            continue
        for feat in feat_cols:
            auc = _univariate_auc(sub[feat].values, y)
            records.append({
                "cohort": cohort,
                "event_type": etype,
                "feature": feat,
                "auc": auc,
                "n_events": int(np.sum(y)),
                "n_controls": int(np.sum(y == 0)),
            })
    return pd.DataFrame(records)


def run_act4() -> dict:
    """Main entry point for Act 4."""
    logger.info("=== ACT 4: Univariate Predictor Validity & Domain-Shift Robustness ===")

    all_parts: list[pd.DataFrame] = []
    cohort_windows_dfs: dict[str, pd.DataFrame] = {}

    for cohort, fname in COHORT_CACHE.items():
        cache_path = CACHE_DIR / fname
        if not cache_path.exists():
            logger.info("  Cohort %s: cache not found (%s), skipping.", cohort, cache_path)
            continue

        logger.info("  Loading %s windows from cache...", cohort)
        windows_df = pd.read_parquet(cache_path)
        logger.info("    %d windows, %d patients", len(windows_df),
                    windows_df["patient_id"].nunique() if "patient_id" in windows_df.columns else -1)

        df = _compute_for_cohort(windows_df, cohort)
        all_parts.append(df)
        cohort_windows_dfs[cohort] = windows_df
        logger.info("    Computed AUC for %d feature×event combinations.", len(df))

    if not all_parts:
        logger.error("No cohort data available. Run Act 1/2/3 first.")
        return {}

    long_df = pd.concat(all_parts, ignore_index=True)
    long_df.to_csv(OUT_DIR / "univariate_auc.csv", index=False)
    logger.info("Saved: univariate_auc.csv  (%d rows)", len(long_df))

    # ------------------------------------------------------------------ #
    # Wide format: one row per feature × event_type, one col per cohort
    # ------------------------------------------------------------------ #
    wide_df = long_df.pivot_table(
        index=["feature", "event_type"],
        columns="cohort",
        values="auc",
    ).reset_index()
    wide_df.columns.name = None
    wide_df.to_csv(OUT_DIR / "univariate_summary.csv", index=False)
    logger.info("Saved: univariate_summary.csv")

    # ------------------------------------------------------------------ #
    # Domain-shift ranking: for each feature × event_type,
    # rank by min AUC across available cohorts (most robust = highest min)
    # and by AUC drop from clinic (best-developed) to external cohorts
    # ------------------------------------------------------------------ #
    cohort_cols = [c for c in ["clinic", "mimic", "vitaldb"] if c in wide_df.columns]
    if len(cohort_cols) >= 2:
        wide_df["min_auc_across_cohorts"] = wide_df[cohort_cols].min(axis=1)
        if "clinic" in wide_df.columns:
            external_cols = [c for c in cohort_cols if c != "clinic"]
            wide_df["max_auc_drop"] = wide_df["clinic"] - wide_df[external_cols].max(axis=1)

        rank_df = wide_df.sort_values(
            ["event_type", "min_auc_across_cohorts"], ascending=[True, False]
        )
        rank_df.to_csv(OUT_DIR / "domain_shift_rank.csv", index=False)
        logger.info("Saved: domain_shift_rank.csv")

        # Log top 5 most robust features per event type
        for etype in EVENT_TYPES:
            sub = rank_df[rank_df["event_type"] == etype].head(5)
            if len(sub) == 0:
                continue
            top = [
                f"{row['feature']} (min={row['min_auc_across_cohorts']:.3f})"
                for _, row in sub.iterrows()
                if not np.isnan(row["min_auc_across_cohorts"])
            ]
            logger.info("  [%s] Top robust features: %s", etype, ", ".join(top))

    # ------------------------------------------------------------------ #
    # Population PCA: compare cohort distributions in feature space
    # ------------------------------------------------------------------ #
    if len(cohort_windows_dfs) >= 2:
        logger.info("Running population PCA across cohorts...")
        _first_df = next(iter(cohort_windows_dfs.values()))
        _feat_cols = get_feature_cols(_first_df)

        pca_result = population_pca(cohort_windows_dfs, _feat_cols)
        pca_result["scores"].to_csv(OUT_DIR / "pca_scores.csv", index=False)
        pca_result["loadings"].to_csv(OUT_DIR / "pca_loadings.csv")
        _pca_meta = {
            "explained_variance_pc1": pca_result["explained"][0],
            "explained_variance_pc2": pca_result["explained"][1],
            "silhouette_cohort_separation": pca_result["silhouette"],
        }
        with open(OUT_DIR / "pca_meta.json", "w") as fh:
            json.dump(_pca_meta, fh, indent=2)
        logger.info(
            "  PCA: PC1=%.1f%%, PC2=%.1f%%, cohort silhouette=%.3f",
            pca_result["explained"][0] * 100,
            pca_result["explained"][1] * 100,
            pca_result["silhouette"],
        )

        # Cross-cohort Spearman feature correlations
        logger.info("Computing cross-cohort Spearman feature correlations...")
        corr_result = cross_cohort_correlations(cohort_windows_dfs, _feat_cols)
        corr_result["direction_consistency"].to_csv(
            OUT_DIR / "feature_correlation_consistency.csv", index=False
        )
        for cname, mat in corr_result["corr_matrices"].items():
            mat.to_csv(OUT_DIR / f"spearman_r_{cname}.csv")
        _top_consistent = (
            corr_result["direction_consistency"]
            .query("pct_sign_consistent == 1.0")
            .nlargest(5, "mean_abs_r")
        )
        logger.info(
            "  Top fully-consistent feature pairs (all cohorts same sign): %s",
            list(zip(_top_consistent["feature_a"], _top_consistent["feature_b"])),
        )

    results = {
        "n_cohorts": len(all_parts),
        "cohorts": [c for c in COHORT_CACHE if (CACHE_DIR / COHORT_CACHE[c]).exists()],
        "n_features": long_df["feature"].nunique(),
        "event_types": EVENT_TYPES,
    }

    with open(OUT_DIR / "act4_results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("Act 4 complete. Results in %s", OUT_DIR)
    return results


if __name__ == "__main__":
    run_act4()
