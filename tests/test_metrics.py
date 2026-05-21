"""Tests for HRV and hemodynamic metrics."""

import numpy as np
import pandas as pd
import pytest

from beatlabile.signal.metrics import compute_all_metrics, aggregate_window_features


# ---------------------------------------------------------------------------
# Helpers to build a minimal SyncResult-like object
# ---------------------------------------------------------------------------

class _FakeSyncResult:
    """Matches the real SyncResult dataclass from beatlabile.signal.sync."""
    def __init__(self, rr_ms, r_times, sbp, dbp, map_beat, pp_beat, beat_times):
        n = len(rr_ms)
        self.rr_ms = np.array(rr_ms, dtype=float)           # real attr name
        self.r_time_s = np.array(r_times, dtype=float)       # real attr name
        self.sbp = np.array(sbp, dtype=float)
        self.sbp_time_s = np.array(beat_times, dtype=float)
        self.delay_ms = np.zeros(n, dtype=float)
        self.is_ectopic = np.zeros(n, dtype=bool)
        self.matched = np.ones(n, dtype=bool)                # required by metrics


def _fake_sync(n_beats=300, rr_mean=857.0, sbp_mean=120.0, dbp_mean=80.0):
    rng = np.random.default_rng(42)
    rr = rng.normal(rr_mean, 30, n_beats).clip(300, 2000)
    times = np.cumsum(rr / 1000.0)
    sbp = rng.normal(sbp_mean, 5, n_beats).clip(60, 250)
    dbp = rng.normal(dbp_mean, 3, n_beats).clip(40, 150)
    dbp = np.minimum(dbp, sbp - 5)
    map_b = dbp + (sbp - dbp) / 3.0
    pp = sbp - dbp
    return _FakeSyncResult(rr, times, sbp, dbp, map_b, pp, times)


class _FakeART:
    """Minimal ARTResult-like object consumed by compute_all_metrics."""
    def __init__(self, n_beats=300, sbp_mean=120.0, dbp_mean=80.0):
        rng = np.random.default_rng(43)
        rr = rng.normal(857, 30, n_beats).clip(300, 2000)
        self.times_s = np.cumsum(rr / 1000.0)
        sbp = rng.normal(sbp_mean, 5, n_beats).clip(60, 250)
        dbp = rng.normal(dbp_mean, 3, n_beats).clip(40, 150)
        dbp = np.minimum(dbp, sbp - 5)
        self.sbp = sbp
        self.dbp = dbp
        self.map_beat = dbp + (sbp - dbp) / 3.0


class TestComputeAllMetrics:
    def test_returns_dataframe(self, minimal_cfg):
        sync = _fake_sync(n_beats=300)
        art = _FakeART(n_beats=300)
        df = compute_all_metrics(sync, art, minimal_cfg)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_expected_columns_present(self, minimal_cfg):
        sync = _fake_sync(n_beats=300)
        art = _FakeART(n_beats=300)
        df = compute_all_metrics(sync, art, minimal_cfg)
        # Actual column names from the implementation
        for col in ["sdnn", "rmssd", "arv", "cv_pa"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_sdnn_positive(self, minimal_cfg):
        sync = _fake_sync(n_beats=300)
        art = _FakeART(n_beats=300)
        df = compute_all_metrics(sync, art, minimal_cfg)
        if "sdnn" in df.columns:
            assert df["sdnn"].dropna().gt(0).all()

    def test_rmssd_positive(self, minimal_cfg):
        sync = _fake_sync(n_beats=300)
        art = _FakeART(n_beats=300)
        df = compute_all_metrics(sync, art, minimal_cfg)
        if "rmssd" in df.columns:
            assert df["rmssd"].dropna().ge(0).all()

    def test_pnn50_bounded(self, minimal_cfg):
        sync = _fake_sync(n_beats=300)
        art = _FakeART(n_beats=300)
        df = compute_all_metrics(sync, art, minimal_cfg)
        if "pnn50" in df.columns:
            valid = df["pnn50"].dropna()
            # pnn50 is stored as a percentage (0–100), not a fraction
            assert (valid >= 0).all() and (valid <= 100).all()


class TestAggregateWindowFeatures:
    def test_aggregate_returns_dataframe(self, minimal_cfg):
        sync = _fake_sync(n_beats=300)
        art = _FakeART(n_beats=300)
        metrics_df = compute_all_metrics(sync, art, minimal_cfg)
        if len(metrics_df) == 0:
            pytest.skip("No metric windows computed on synthetic data")
        agg = aggregate_window_features(metrics_df, window_s=30)
        assert isinstance(agg, pd.DataFrame)

    def test_aggregate_has_numeric_cols(self, minimal_cfg):
        sync = _fake_sync(n_beats=300)
        art = _FakeART(n_beats=300)
        metrics_df = compute_all_metrics(sync, art, minimal_cfg)
        if len(metrics_df) == 0:
            pytest.skip("No metric windows computed on synthetic data")
        agg = aggregate_window_features(metrics_df, window_s=30)
        numeric_cols = agg.select_dtypes(include=np.number).columns.tolist()
        assert len(numeric_cols) > 0
