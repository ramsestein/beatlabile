"""Bootstrap confidence intervals for AUC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap AUC with percentile CI.

    Returns (auc_point, lower, upper).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    auc_point = float(roc_auc_score(y_true, y_score))

    boot_aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(float(roc_auc_score(yt, ys)))

    if len(boot_aucs) < 20:
        return auc_point, float("nan"), float("nan")

    lower = float(np.percentile(boot_aucs, 100 * alpha / 2))
    upper = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
    return auc_point, lower, upper


def ci_from_folds(auc_folds: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Percentile 95 CI from cross-validation fold AUCs.

    Returns (lower, upper).
    """
    arr = np.asarray(auc_folds)
    lower = float(np.percentile(arr, 100 * alpha / 2))
    upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return lower, upper
