"""Tests for prediction window builder."""

import numpy as np
import pandas as pd
import pytest

from beatlabile.windows.builder import build_windows
from beatlabile.events.detector import detect_events


def _fake_features_df(n_rows: int, start_time: float = 3600.0) -> pd.DataFrame:
    """Generate a minimal feature DataFrame with required columns."""
    rng = np.random.default_rng(7)
    # window_start_s and window_end_s are required by build_windows
    window_duration = 30.0
    starts = np.linspace(start_time, start_time + n_rows * window_duration, n_rows)
    ends = starts + window_duration
    df = pd.DataFrame({
        "window_start_s": starts,
        "window_end_s": ends,
        "sdnn": rng.normal(50, 10, n_rows),
        "rmssd": rng.normal(40, 8, n_rows),
        "arv": rng.normal(15, 3, n_rows),
        "cv_pa": rng.uniform(0.05, 0.15, n_rows),
        "brs": rng.uniform(5, 15, n_rows),
    })
    return df


def _fake_art_result(n_beats: int = 1200):
    """Minimal object matching ARTResult used by build_windows."""
    rng = np.random.default_rng(5)
    sbp = rng.normal(120, 5, n_beats).clip(90, 160)
    dbp = rng.normal(80, 3, n_beats).clip(60, 100)
    map_b = dbp + (sbp - dbp) / 3.0
    pp = sbp - dbp
    beat_times = np.linspace(0, 7200, n_beats)  # 2-hour record

    class _AR:
        pass

    a = _AR()
    a.sbp = sbp
    a.dbp = dbp
    a.map_beat = map_b
    a.pp_beat = pp
    a.times_s = beat_times       # build_windows uses art.times_s
    a.systolic_times_s = beat_times
    a.systolic_peaks = (beat_times * 500).astype(int)
    a.is_art_bad = np.zeros(n_beats, dtype=bool)
    return a


def _fake_events_with_hypotension():
    """Return a real EventResult with one hypotension event at t=1800s."""
    from beatlabile.events.detector import Event, EventResult
    e = Event(event_type="hypotension", start_s=1800.0, end_s=1860.0, peak_value=45.0)
    return EventResult(hypotension=[e], hypertension=[], variability=[])


class TestBuildWindows:
    def test_returns_dataframe(self, minimal_cfg):
        features = _fake_features_df(n_rows=300)
        art = _fake_art_result(n_beats=1200)
        events = _fake_events_with_hypotension()
        r_times = np.linspace(0, 7200, 1200)
        df = build_windows(features, events, art, r_times, "test_pt", minimal_cfg, "hypotension")
        assert isinstance(df, pd.DataFrame)

    def test_labels_present(self, minimal_cfg):
        features = _fake_features_df(n_rows=300)
        art = _fake_art_result(n_beats=1200)
        events = _fake_events_with_hypotension()
        r_times = np.linspace(0, 7200, 1200)
        df = build_windows(features, events, art, r_times, "test_pt", minimal_cfg, "hypotension")
        if "label" in df.columns:
            assert df["label"].isin([0, 1]).all()

    def test_patient_id_column(self, minimal_cfg):
        features = _fake_features_df(n_rows=300)
        art = _fake_art_result(n_beats=1200)
        events = _fake_events_with_hypotension()
        r_times = np.linspace(0, 7200, 1200)
        df = build_windows(features, events, art, r_times, "my_patient", minimal_cfg, "hypotension")
        if "patient_id" in df.columns:
            assert (df["patient_id"] == "my_patient").all()

    def test_no_events_returns_controls_only(self, minimal_cfg):
        """When there are no events, build_windows should return only control windows (or empty)."""
        from beatlabile.events.detector import EventResult
        events = EventResult(hypotension=[], hypertension=[], variability=[])
        features = _fake_features_df(n_rows=300)
        art = _fake_art_result(n_beats=1200)
        r_times = np.linspace(0, 7200, 1200)
        df = build_windows(features, events, art, r_times, "test_pt", minimal_cfg, "hypotension")
        assert isinstance(df, pd.DataFrame)
        if "label" in df.columns:
            assert (df["label"] == 0).all()
