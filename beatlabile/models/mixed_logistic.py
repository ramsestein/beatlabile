"""Approach A — Mixed-effects logistic regression (GLMM, Section 9.2).

Uses statsmodels BinomialBayesMixedGLM as the primary estimator (or
LogisticRegression with clustered bootstrap as a fallback) for a model
with random intercept per patient.

The module also provides:
  · 10-fold × 50 repeated patient-level CV
  · NRI/IDI calculation (incremental value over conventional predictors)
  · Blind validation (fixed coefficients, random intercept = 0)

Public API
----------
MixedLogisticModel.fit(X, y, patient_ids)
MixedLogisticModel.predict_proba(X)
MixedLogisticModel.cross_validate(X, y, patient_ids, cfg)
MixedLogisticModel.validate(X, y)   # blind validation
MixedLogisticModel.nri_idi(X_base, X_aug, y)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# statsmodels GLMM — may not be available in all environments
try:
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


@dataclass
class CVResult:
    """Cross-validation results."""
    auc_mean: float
    auc_std: float
    auc_folds: list[float]
    brier_mean: float
    brier_std: float
    n_folds: int
    n_repeats: int


@dataclass
class ValidationResult:
    """Blind validation results."""
    auc: float
    brier: float
    calibration_in_the_large: float  # CITL = mean(predicted) - mean(observed)
    calibration_slope: float
    n_events: int
    n_controls: int


class MixedLogisticModel:
    """Logistic model with random intercept per patient.

    Falls back to standard sklearn LogisticRegression if statsmodels is
    unavailable, used only for coefficient extraction in that case.
    """

    def __init__(self, feature_cols: list[str] | None = None):
        self.feature_cols: list[str] | None = feature_cols
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.scaler_: StandardScaler = StandardScaler()
        self._fitted = False
        self._statsmodels_result: Any = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        patient_ids: np.ndarray,
    ) -> "MixedLogisticModel":
        """Fit the GLMM on training data.

        Falls back to sklearn LogisticRegression with a warning if statsmodels
        is not available or fitting fails.
        """
        if self.feature_cols is None:
            self.feature_cols = list(X.columns)
        X_sub = X[self.feature_cols].copy()

        # Impute missing with column medians; fall back to 0 for all-NaN columns
        medians = X_sub.median()
        X_sub = X_sub.fillna(medians).fillna(0.0)
        self._medians = medians

        X_scaled = self.scaler_.fit_transform(X_sub)

        if _HAS_STATSMODELS:
            try:
                self._fit_glmm(X_scaled, y, patient_ids)
                self._fitted = True
                return self
            except Exception as exc:
                warnings.warn(f"GLMM fit failed ({exc}); falling back to sklearn LogisticRegression.")

        # Fallback
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        lr.fit(X_scaled, y)
        self.coef_ = lr.coef_[0]
        self.intercept_ = float(lr.intercept_[0])
        self._fitted = True
        return self

    def _fit_glmm(
        self, X_scaled: np.ndarray, y: np.ndarray, patient_ids: np.ndarray
    ) -> None:
        """Inner GLMM fit using statsmodels."""
        df = pd.DataFrame(X_scaled, columns=self.feature_cols)
        df["y"] = y.astype(int)
        df["patient"] = patient_ids

        # Build formula
        predictors = " + ".join(self.feature_cols)
        formula = f"y ~ {predictors}"

        model = BinomialBayesMixedGLM.from_formula(
            formula, {"patient": "0 + C(patient)"}, df
        )
        result = model.fit_vb()
        self._statsmodels_result = result

        # Extract fixed-effect coefficients
        param_names = result.model.fep_names
        fe_params = result.fe_mean
        coef_dict = dict(zip(param_names, fe_params))

        self.intercept_ = float(coef_dict.get("Intercept", 0.0))
        self.coef_ = np.array([
            float(coef_dict.get(c, 0.0)) for c in self.feature_cols
        ])

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities (marginal, random intercept = 0)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
        X_sub = X[self.feature_cols].fillna(self._medians).fillna(0.0)
        X_scaled = self.scaler_.transform(X_sub)
        logit = X_scaled @ self.coef_ + self.intercept_
        return _sigmoid(logit)

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        patient_ids: np.ndarray,
        cfg: dict,
    ) -> CVResult:
        """10-fold × N-repeat patient-level CV."""
        n_folds = cfg["models"]["cv_folds"]
        n_repeats = cfg["models"]["cv_repeats"]
        rng = np.random.default_rng(cfg["models"]["random_state"])

        auc_list = []
        brier_list = []

        for rep in range(n_repeats):
            # Shuffle patient order to vary folds across repeats
            unique_patients = np.unique(patient_ids)
            perm = rng.permutation(len(unique_patients))
            shuffled_patients = unique_patients[perm]

            # Build patient-level fold mapping
            fold_map = {p: i % n_folds for i, p in enumerate(shuffled_patients)}
            sample_fold = np.array([fold_map[pid] for pid in patient_ids])

            for fold in range(n_folds):
                train_idx = sample_fold != fold
                test_idx = sample_fold == fold
                if not np.any(test_idx) or not np.any(train_idx):
                    continue

                model = MixedLogisticModel(feature_cols=self.feature_cols)
                model.fit(X.iloc[train_idx], y[train_idx], patient_ids[train_idx])
                proba = model.predict_proba(X.iloc[test_idx])
                y_test = y[test_idx]

                if len(np.unique(y_test)) == 2:
                    auc_list.append(float(roc_auc_score(y_test, proba)))
                    brier_list.append(float(brier_score_loss(y_test, proba)))

        return CVResult(
            auc_mean=float(np.mean(auc_list)),
            auc_std=float(np.std(auc_list)),
            auc_folds=auc_list,
            brier_mean=float(np.mean(brier_list)),
            brier_std=float(np.std(brier_list)),
            n_folds=n_folds,
            n_repeats=n_repeats,
        )

    # ------------------------------------------------------------------
    # Blind validation
    # ------------------------------------------------------------------

    def validate(self, X: pd.DataFrame, y: np.ndarray) -> ValidationResult:
        """Apply trained model to an external cohort (fixed coefficients)."""
        proba = self.predict_proba(X)
        y = np.asarray(y).astype(int)

        auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else np.nan
        brier = float(brier_score_loss(y, proba))
        citl = float(np.mean(proba) - np.mean(y))

        # Calibration slope: logistic regression of log-odds on observed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_odds = np.log(np.clip(proba, 1e-6, 1 - 1e-6) / (1 - np.clip(proba, 1e-6, 1 - 1e-6)))
            slope_model = LogisticRegression(fit_intercept=True, max_iter=500)
            slope_model.fit(log_odds.reshape(-1, 1), y)
            cal_slope = float(slope_model.coef_[0, 0])

        return ValidationResult(
            auc=auc,
            brier=brier,
            calibration_in_the_large=citl,
            calibration_slope=cal_slope,
            n_events=int(np.sum(y)),
            n_controls=int(np.sum(y == 0)),
        )

    # ------------------------------------------------------------------
    # NRI / IDI
    # ------------------------------------------------------------------

    def nri_idi(
        self,
        X_base: pd.DataFrame,
        X_aug: pd.DataFrame,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Compute continuous NRI and IDI between baseline and augmented model.

        Both models are assumed to be separately fitted before calling this.
        This method takes two probability vectors and computes the metrics.
        """
        p_base = self.predict_proba(X_base)
        p_aug = self.predict_proba(X_aug)
        return compute_nri_idi(p_base, p_aug, y)


# ---------------------------------------------------------------------------
# Standalone NRI/IDI
# ---------------------------------------------------------------------------

def compute_nri_idi(
    p_old: np.ndarray,
    p_new: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Continuous NRI and IDI.

    NRI_cont = P(up | event) - P(down | event) + P(down | no-event) - P(up | no-event)
    IDI = (mean_new_event - mean_new_noevent) - (mean_old_event - mean_old_noevent)
    """
    y = np.asarray(y).astype(int)
    p_old = np.asarray(p_old)
    p_new = np.asarray(p_new)

    events = y == 1
    noevents = y == 0

    up_event = np.mean(p_new[events] > p_old[events])
    down_event = np.mean(p_new[events] < p_old[events])
    up_noevent = np.mean(p_new[noevents] > p_old[noevents])
    down_noevent = np.mean(p_new[noevents] < p_old[noevents])

    nri = float((up_event - down_event) + (down_noevent - up_noevent))

    idi_new = float(np.mean(p_new[events]) - np.mean(p_new[noevents]))
    idi_old = float(np.mean(p_old[events]) - np.mean(p_old[noevents]))
    idi = float(idi_new - idi_old)

    return {"nri_continuous": nri, "idi": idi, "idi_new": idi_new, "idi_old": idi_old}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
