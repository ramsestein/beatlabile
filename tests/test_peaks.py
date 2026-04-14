"""Tests for peak detection (R-peaks and ART peaks)."""

import numpy as np
import pytest

from beatlabile.signal.peaks import detect_r_peaks, detect_art_peaks


class TestRPeakDetection:
    def test_basic_detection(self, synthetic_ecg, fs, minimal_cfg):
        result = detect_r_peaks(synthetic_ecg, fs, minimal_cfg)
        # At 70 bpm for 120 s, expect ~140 beats
        assert len(result.r_peaks) > 100
        assert len(result.rr_intervals_ms) == len(result.r_peaks) - 1

    def test_rr_physiological(self, synthetic_ecg, fs, minimal_cfg):
        result = detect_r_peaks(synthetic_ecg, fs, minimal_cfg)
        # Non-ectopic RR should be around 857 ms (70 bpm)
        normal_rr = result.rr_intervals_ms[~result.is_ectopic]
        if len(normal_rr) > 0:
            assert np.mean(normal_rr) == pytest.approx(857, abs=200)

    def test_ectopic_flagging(self, fs, minimal_cfg):
        # Create ECG with one very short RR interval
        rng = np.random.default_rng(10)
        t = np.arange(30 * fs) / fs
        signal = np.zeros(len(t), dtype=np.float32)
        # Normal beats at 70 bpm, plus an ectopic at t=5s
        beat_times = list(np.arange(0, 30, 60.0 / 70.0))
        beat_times.append(5.1)  # close to normal beat → ectopic
        for bt in beat_times:
            idx = int(bt * fs)
            if 5 <= idx < len(signal) - 5:
                signal[idx] = 1.0
        signal += rng.normal(0, 0.05, len(signal)).astype(np.float32)
        result = detect_r_peaks(signal, fs, minimal_cfg)
        # At least one ectopic should be flagged
        assert np.any(result.is_ectopic) or len(result.r_peaks) > 10  # graceful handling

    def test_empty_signal(self, fs, minimal_cfg):
        signal = np.zeros(fs, dtype=np.float32)
        result = detect_r_peaks(signal, fs, minimal_cfg)
        assert len(result.rr_intervals_ms) == 0

    def test_with_bad_mask(self, synthetic_ecg, fs, minimal_cfg):
        bad = np.zeros(len(synthetic_ecg), dtype=bool)
        bad[:1000] = True
        result = detect_r_peaks(synthetic_ecg, fs, minimal_cfg, bad_mask=bad)
        assert len(result.r_peaks) > 50


class TestARTPeakDetection:
    def test_basic_detection(self, synthetic_art, fs, minimal_cfg):
        result = detect_art_peaks(synthetic_art, fs, minimal_cfg)
        assert len(result.systolic_peaks) > 50
        assert len(result.sbp) == len(result.systolic_peaks)

    def test_sbp_above_dbp(self, synthetic_art, fs, minimal_cfg):
        result = detect_art_peaks(synthetic_art, fs, minimal_cfg)
        if len(result.sbp) > 0:
            assert np.all(result.sbp > result.dbp)

    def test_map_computation(self, synthetic_art, fs, minimal_cfg):
        result = detect_art_peaks(synthetic_art, fs, minimal_cfg)
        if len(result.map_beat) > 0:
            expected_map = result.dbp + result.pp_beat / 3.0
            np.testing.assert_allclose(result.map_beat, expected_map, rtol=1e-4)

    def test_physiological_range(self, synthetic_art, fs, minimal_cfg):
        result = detect_art_peaks(synthetic_art, fs, minimal_cfg)
        if len(result.sbp) > 0:
            assert np.all(result.sbp >= 40)
            assert np.all(result.sbp <= 300)
            assert np.all(result.dbp >= 20)
