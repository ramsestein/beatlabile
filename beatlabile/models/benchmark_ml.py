"""Approach C — Benchmark ML (Random Forest + XGBoost, Section 9.4).

These are reference comparators to demonstrate that interpretable models
(GLMM + MILP tree) achieve comparable performance.

Benchmark models are trained only on Act 1 (Clínic). They are NOT applied
during blind validation to keep the interpretable pipeline pure.

Public API
----------
RFModel.fit(X, y, cfg) -> RFModel
RFModel.predict_proba(X) -> np.ndarray
RFModel.cross_validate(X, y, patient_ids, cfg) -> CVResult
XGBModel (same interface)
compare_benchmarks(glmm_auc, milp_auc, rf_auc, xgb_auc) -> dict
"""

from __future__ import annotations

import subprocess
import warnings

# Suppress sklearn/joblib parallel context warning (fires in multiprocessing workers)
warnings.filterwarnings(
    "ignore",
    message=".*sklearn.utils.parallel.delayed.*",
    category=UserWarning,
)

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False
    warnings.warn("XGBoost not installed; XGBModel will raise.")

from beatlabile.models.mixed_logistic import CVResult


def _cuda_available() -> bool:
    """Return True if an NVIDIA GPU with CUDA is accessible."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


_USE_GPU: bool = _cuda_available()


class RFModel:
    """Random Forest with grid-search hyperparameter tuning."""

    def __init__(self) -> None:
        self.model: RandomForestClassifier | None = None
        self.feature_cols: list[str] = []
        self._medians: pd.Series | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, cfg: dict) -> "RFModel":
        self.feature_cols = list(X.columns)
        self._medians = X.median()
        X_clean = X.fillna(self._medians)
        m_cfg = cfg["models"]["rf"]

        param_grid = {
            "n_estimators": [100, 300],
            "max_features": ["sqrt", "log2"],
        }
        base = RandomForestClassifier(
            random_state=cfg["models"]["random_state"],
            n_jobs=1,
        )
        gs = GridSearchCV(
            base, param_grid, cv=m_cfg["cv_folds"], scoring="roc_auc", n_jobs=1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_clean, y)

        self.model = gs.best_estimator_
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.feature_cols].fillna(self._medians)
        return self.model.predict_proba(X_clean)[:, 1]

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        patient_ids: np.ndarray,
        cfg: dict,
    ) -> CVResult:
        return _patient_cv(self, X, y, patient_ids, cfg)


class XGBModel:
    """XGBoost with early stopping and grid search."""

    def __init__(self) -> None:
        self.model = None
        self.feature_cols: list[str] = []
        self._medians: pd.Series | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, cfg: dict) -> "XGBModel":
        if not _HAS_XGB:
            raise ImportError("xgboost is required: pip install xgboost")
        self.feature_cols = list(X.columns)
        self._medians = X.median()
        X_clean = X.fillna(self._medians)
        m_cfg = cfg["models"]["xgboost"]

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }
        base = XGBClassifier(
            eval_metric="logloss",
            random_state=cfg["models"]["random_state"],
            verbosity=0,
            device="cuda" if _USE_GPU else "cpu",
        )
        gs = GridSearchCV(
            base, param_grid, cv=m_cfg["cv_folds"], scoring="roc_auc", n_jobs=1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_clean, y)

        self.model = gs.best_estimator_
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X[self.feature_cols].fillna(self._medians)
        return self.model.predict_proba(X_clean)[:, 1]

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        patient_ids: np.ndarray,
        cfg: dict,
    ) -> CVResult:
        return _patient_cv(self, X, y, patient_ids, cfg)


def compare_benchmarks(
    glmm_auc: float,
    milp_auc: float,
    rf_auc: float,
    xgb_auc: float,
) -> dict[str, float]:
    """Compute delta-AUC between interpretable models and ML benchmarks.

    If |ΔAUC| < 0.05, interpretability has no meaningful performance cost.
    """
    delta_rf = rf_auc - max(glmm_auc, milp_auc)
    delta_xgb = xgb_auc - max(glmm_auc, milp_auc)
    return {
        "best_interpretable_auc": max(glmm_auc, milp_auc),
        "rf_auc": rf_auc,
        "xgb_auc": xgb_auc,
        "delta_auc_vs_rf": delta_rf,
        "delta_auc_vs_xgb": delta_xgb,
        "interpretability_cost_acceptable": abs(delta_rf) < 0.05 and abs(delta_xgb) < 0.05,
    }


# ---------------------------------------------------------------------------
# Shared patient-level cross-validation
# ---------------------------------------------------------------------------

def _patient_cv(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    patient_ids: np.ndarray,
    cfg: dict,
) -> CVResult:
    """Patient-level cross-validation reusing the best params from model.fit()."""
    n_folds = cfg["models"]["cv_folds"]
    rng = np.random.default_rng(cfg["models"]["random_state"])

    # Reuse best hyperparams from the already-fitted model (avoid per-fold grid search)
    best_params = model.model.get_params()

    auc_list = []
    brier_list = []

    unique_patients = np.unique(patient_ids)
    perm = rng.permutation(len(unique_patients))
    shuffled_patients = unique_patients[perm]
    fold_map = {p: i % n_folds for i, p in enumerate(shuffled_patients)}
    sample_fold = np.array([fold_map[pid] for pid in patient_ids])

    for fold in range(n_folds):
        train_mask = sample_fold != fold
        test_mask = sample_fold == fold
        if not np.any(test_mask) or not np.any(train_mask):
            continue

        X_tr = X.iloc[train_mask]
        y_tr = y[train_mask]
        X_te = X.iloc[test_mask]
        y_te = y[test_mask]

        # Clone model with best params, fit directly (no grid search)
        fold_model = model.model.__class__(**best_params)
        medians = X_tr.median()
        X_tr_clean = X_tr.fillna(medians)
        X_te_clean = X_te[X_tr.columns].fillna(medians)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold_model.fit(X_tr_clean, y_tr)

        proba = fold_model.predict_proba(X_te_clean)[:, 1]
        if len(np.unique(y_te)) == 2:
            auc_list.append(float(roc_auc_score(y_te, proba)))
            brier_list.append(float(brier_score_loss(y_te, proba)))

    return CVResult(
        auc_mean=float(np.mean(auc_list)) if auc_list else np.nan,
        auc_std=float(np.std(auc_list)) if auc_list else np.nan,
        auc_folds=auc_list,
        brier_mean=float(np.mean(brier_list)) if brier_list else np.nan,
        brier_std=float(np.std(brier_list)) if brier_list else np.nan,
        n_folds=n_folds,
        n_repeats=1,
    )
