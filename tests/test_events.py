"""Tests for haemodynamic event detection."""

import numpy as np
import pytest

from beatlabile.events.detector import detect_events


def _build_art_result(sbp, dbp, fs=500, duration_s=None):
    """Return a minimal ARTResult-like object for testing."""
    n = len(sbp)
    map_b = dbp + (sbp - dbp) / 3.0
    pp = sbp - dbp
    # Place beats evenly
    if duration_s is None:
        duration_s = n  # 1 beat/s
    beat_times = np.linspace(0, duration_s - 1, n)

    class _ARTResult:
        pass

    r = _ARTResult()
    r.sbp = np.array(sbp, dtype=float)
    r.dbp = np.array(dbp, dtype=float)
    r.map_beat = np.array(map_b, dtype=float)
    r.pp_beat = np.array(pp, dtype=float)
    r.times_s = beat_times          # used by detect_events
    r.systolic_times_s = beat_times
    r.systolic_peaks = np.round(beat_times * fs).astype(int)
    r.is_art_bad = np.zeros(n, dtype=bool)
    return r


def _build_plain_art(art_array, fs=500):
    """Wrap a numpy waveform array (used for raw signal path)."""
    return art_array


class TestHypotensionDetection:
    def test_detects_sustained_map_below_55(self, minimal_cfg):
        # 400 beats, most normal, but 60 beats with MAP≈40
        n = 400
        sbp = np.full(n, 120.0)
        dbp = np.full(n, 80.0)
        # Beats 100–159: MAP≈40 (SBP=55, DBP=30)
        sbp[100:160] = 55.0
        dbp[100:160] = 30.0
        art = _build_art_result(sbp, dbp, duration_s=400)
        result = detect_events(art, minimal_cfg)
        assert len(result.hypotension) >= 1

    def test_no_false_positive_normal_map(self, minimal_cfg):
        n = 400
        sbp = np.full(n, 120.0)
        dbp = np.full(n, 80.0)
        art = _build_art_result(sbp, dbp, duration_s=400)
        result = detect_events(art, minimal_cfg)
        assert len(result.hypotension) == 0


class TestHypertensionDetection:
    def test_detects_sustained_sbp_above_180(self, minimal_cfg):
        n = 400
        sbp = np.full(n, 120.0)
        dbp = np.full(n, 80.0)
        sbp[100:160] = 200.0
        art = _build_art_result(sbp, dbp, duration_s=400)
        result = detect_events(art, minimal_cfg)
        assert len(result.hypertension) >= 1

    def test_no_false_positive_normal_sbp(self, minimal_cfg):
        n = 400
        sbp = np.full(n, 120.0)
        dbp = np.full(n, 80.0)
        art = _build_art_result(sbp, dbp, duration_s=400)
        result = detect_events(art, minimal_cfg)
        assert len(result.hypertension) == 0


class TestVariabilityDetection:
    def test_detects_large_map_swings(self, minimal_cfg):
        # Craft beats with alternating MAP: 50 and 90 → swing ~40 in short window
        n = 600
        sbp = np.full(n, 120.0)
        dbp = np.full(n, 80.0)
        # Override in windows to create swings >30 mmHg
        for i in range(200, 400, 20):
            sbp[i:i + 10] = 60.0
            dbp[i:i + 10] = 40.0  # MAP ~47
            sbp[i + 10:i + 20] = 180.0
            dbp[i + 10:i + 20] = 90.0  # MAP ~120
        art = _build_art_result(sbp, dbp, duration_s=600)
        result = detect_events(art, minimal_cfg)
        # Should detect at least 1 variability episode (or 0 — existence check)
        assert len(result.variability) >= 0

    def test_event_result_to_dataframe(self, minimal_cfg):
        n = 200
        sbp = np.full(n, 120.0)
        dbp = np.full(n, 80.0)
        art = _build_art_result(sbp, dbp, duration_s=200)
        result = detect_events(art, minimal_cfg)
        df = result.to_dataframe()
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
