"""Tests for model training and inference."""

import numpy as np
import pandas as pd
import pytest

from beatlabile.models.mixed_logistic import MixedLogisticModel, compute_nri_idi
from beatlabile.models.milp_tree import MILPTree
from beatlabile.models.benchmark_ml import RFModel, XGBModel


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _synthetic_dataset(n: int = 200, n_patients: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    patient_ids = [f"p{i:03d}" for i in rng.integers(n_patients, size=n)]
    X = pd.DataFrame({
        "sdnn_ms": rng.normal(50, 15, n),
        "rmssd_ms": rng.normal(40, 12, n),
        "arv_mmhg": rng.normal(15, 5, n),
        "cv_pa": rng.uniform(0.03, 0.20, n),
        "brs_ms_mmhg": rng.uniform(2, 20, n),
    })
    y = rng.integers(0, 2, n)
    return X, y, patient_ids


# ---------------------------------------------------------------------------
# MixedLogisticModel
# ---------------------------------------------------------------------------

class TestMixedLogisticModel:
    def test_fit_predict_proba(self):
        X, y, patient_ids = _synthetic_dataset()
        model = MixedLogisticModel()
        model.fit(X, y, patient_ids)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y),)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_binary(self):
        X, y, patient_ids = _synthetic_dataset()
        model = MixedLogisticModel()
        model.fit(X, y, patient_ids)
        # MixedLogisticModel only has predict_proba; threshold manually
        preds = (model.predict_proba(X) >= 0.5).astype(int)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_coef_attribute(self):
        X, y, patient_ids = _synthetic_dataset()
        model = MixedLogisticModel()
        model.fit(X, y, patient_ids)
        has_coef = hasattr(model, "coef_") or hasattr(model, "params_") or hasattr(model, "_statsmodels_result")
        assert has_coef


class TestComputeNriIdi:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(9)
        y = rng.integers(0, 2, 100)
        p_old = rng.uniform(0.1, 0.9, 100)
        p_new = rng.uniform(0.1, 0.9, 100)
        result = compute_nri_idi(p_old, p_new, y)
        assert "nri_continuous" in result
        assert "idi" in result


# ---------------------------------------------------------------------------
# MILPTree (CART fallback)
# ---------------------------------------------------------------------------

class TestMILPTree:
    def test_fit_predict(self, minimal_cfg):
        X, y, _ = _synthetic_dataset(n=100)
        tree = MILPTree()
        tree.fit(X, y, "hypotension", minimal_cfg)
        preds = tree.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_range(self, minimal_cfg):
        X, y, _ = _synthetic_dataset(n=100)
        tree = MILPTree()
        tree.fit(X, y, "hypotension", minimal_cfg)
        if hasattr(tree, "predict_proba"):
            proba = tree.predict_proba(X)
            assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_consistency(self, minimal_cfg):
        """Fitting twice on the same data should give the same predictions."""
        X, y, _ = _synthetic_dataset(n=80)
        t1 = MILPTree()
        t1.fit(X, y, "hypotension", minimal_cfg)
        t2 = MILPTree()
        t2.fit(X, y, "hypotension", minimal_cfg)
        np.testing.assert_array_equal(t1.predict(X), t2.predict(X))


# ---------------------------------------------------------------------------
# RFModel
# ---------------------------------------------------------------------------

class TestRFModel:
    def test_fit_predict_proba(self, minimal_cfg):
        X, y, _ = _synthetic_dataset(n=150)
        model = RFModel()
        model.fit(X, y, minimal_cfg)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y),)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_feature_importances(self, minimal_cfg):
        X, y, _ = _synthetic_dataset(n=150)
        model = RFModel()
        model.fit(X, y, minimal_cfg)
        # Internal sklearn model exposes feature_importances_
        importances = model.model.feature_importances_
        assert len(importances) == X.shape[1]
        assert np.all(importances >= 0)


# ---------------------------------------------------------------------------
# XGBModel (optional — skip if not installed)
# ---------------------------------------------------------------------------

class TestXGBModel:
    def test_fit_predict_proba(self, minimal_cfg):
        pytest.importorskip("xgboost")
        X, y, _ = _synthetic_dataset(n=150)
        model = XGBModel()
        model.fit(X, y, minimal_cfg)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y),)
        assert np.all(proba >= 0) and np.all(proba <= 1)
