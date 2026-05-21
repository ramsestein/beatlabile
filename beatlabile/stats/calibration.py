"""Calibration curves and Decision Curve Analysis (DCA)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_data(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Mean predicted probability vs observed event rate per bin.

    Returns DataFrame with columns:
        bin_center, mean_predicted, observed_rate, count
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bin_edges[1:-1])  # 0..n_bins-1

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_center": float(bin_edges[b] + (bin_edges[b + 1] - bin_edges[b]) / 2),
                "mean_predicted": float(y_score[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "count": count,
            }
        )

    return pd.DataFrame(rows)


def net_benefit(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Net benefit for Decision Curve Analysis.

    Returns DataFrame with columns:
        threshold, net_benefit_model, net_benefit_all, net_benefit_none
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    prevalence = y_true.mean()

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    for t in thresholds:
        predicted_pos = y_score >= t
        tp = float((predicted_pos & (y_true == 1)).sum())
        fp = float((predicted_pos & (y_true == 0)).sum())

        nb_model = tp / n - fp / n * (t / (1.0 - t))
        nb_all = prevalence - (1.0 - prevalence) * (t / (1.0 - t))

        rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": 0.0,
            }
        )

    return pd.DataFrame(rows)
