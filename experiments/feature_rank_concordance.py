"""Feature ranking concordance across cohorts — evidence of physiological invariance.

For each event type, computes the Spearman rank correlation of univariate AUC
vectors (40 features) between all pairs of cohorts (Clínic, MIMIC, VitalDB).

High ρ means the same features are predictive in the same order regardless of
hospital, country, or clinical context — a model-free, EPV-free argument for
physiological invariance.

Reads from: results/act4/univariate_auc.csv  (produced by act4_univariate.py)

Writes to:
  results/act4/rank_concordance.csv    — ρ, p-value per cohort-pair × event-type
  results/act4/rank_concordance.json   — same, machine-readable

Run
---
python experiments/feature_rank_concordance.py
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from beatlabile.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IN_FILE = RESULTS_DIR / "act4" / "univariate_auc.csv"
OUT_DIR = RESULTS_DIR / "act4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_concordance() -> dict:
    if not IN_FILE.exists():
        raise FileNotFoundError(
            f"{IN_FILE} not found. Run 'python experiments/act4_univariate.py' first."
        )

    df = pd.read_csv(IN_FILE)
    cohorts = sorted(df["cohort"].unique())
    event_types = sorted(df["event_type"].unique())

    records: list[dict] = []

    for etype in event_types:
        sub = df[df["event_type"] == etype]

        # AUC vectors per cohort, aligned on the same feature set
        auc_by_cohort: dict[str, pd.Series] = {}
        for cohort in cohorts:
            s = sub[sub["cohort"] == cohort].set_index("feature")["auc"]
            auc_by_cohort[cohort] = s

        # Keep only features present in ALL cohorts and not fully NaN
        common_feats = set.intersection(*[set(s.index) for s in auc_by_cohort.values()])
        common_feats = sorted(common_feats)

        if len(common_feats) < 5:
            logger.warning("  %s: only %d common features, skipping.", etype, len(common_feats))
            continue

        logger.info("  %s: %d common features across %d cohorts", etype, len(common_feats), len(cohorts))

        for c1, c2 in combinations(cohorts, 2):
            v1 = auc_by_cohort[c1].reindex(common_feats).values
            v2 = auc_by_cohort[c2].reindex(common_feats).values

            # Drop pairs where either value is NaN
            valid = ~(np.isnan(v1) | np.isnan(v2))
            if valid.sum() < 5:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(v1[valid], v2[valid])

            records.append({
                "event_type": etype,
                "cohort_1": c1,
                "cohort_2": c2,
                "pair": f"{c1} vs {c2}",
                "n_features": int(valid.sum()),
                "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else None,
                "p_value": float(f"{pval:.4g}") if not np.isnan(pval) else None,
            })

            star = "***" if (pval is not None and pval < 0.001) else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            logger.info("    %-14s vs %-8s  ρ=%.3f  p=%.4f  n=%d  %s",
                        c1, c2, rho, pval, valid.sum(), star)

    result_df = pd.DataFrame(records)
    result_df.to_csv(OUT_DIR / "rank_concordance.csv", index=False)

    # Pretty console summary
    print("\n" + "=" * 70)
    print("FEATURE RANKING CONCORDANCE — Spearman ρ of 40 univariate AUCs")
    print("=" * 70)
    for etype in event_types:
        sub_r = result_df[result_df["event_type"] == etype]
        if sub_r.empty:
            continue
        print(f"\n  {etype.upper()}")
        for _, row in sub_r.iterrows():
            rho_str = f"{row['spearman_rho']:.3f}" if row["spearman_rho"] is not None else "N/A"
            p_str = f"{row['p_value']:.4f}" if row["p_value"] is not None else "N/A"
            print(f"    {row['cohort_1']:8s} vs {row['cohort_2']:8s}  ρ={rho_str}  p={p_str}  (n={row['n_features']})")
    print()

    # Also save JSON for downstream consumption
    result_dict: dict = {}
    for etype in event_types:
        result_dict[etype] = {}
        for _, row in result_df[result_df["event_type"] == etype].iterrows():
            result_dict[etype][row["pair"]] = {
                "rho": row["spearman_rho"],
                "p_value": row["p_value"],
                "n_features": row["n_features"],
            }
    with open(OUT_DIR / "rank_concordance.json", "w") as fh:
        json.dump(result_dict, fh, indent=2)

    return {r["pair"]: r for r in records}


if __name__ == "__main__":
    run_concordance()
