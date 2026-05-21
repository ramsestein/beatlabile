"""Pytest configuration and shared fixtures for BeatLabile tests."""

from __future__ import annotations

import numpy as np
import pytest


FS = 500  # Default sampling frequency
N_SECONDS = 120  # 2 minutes of synthetic signal


@pytest.fixture(scope="session")
def fs():
    return FS


@pytest.fixture(scope="session")
def minimal_cfg():
    """Minimal config sufficient to run all modules."""
    return {
        "qc": {
            "min_duration_seconds": 60,
            "complete_duration_seconds": 300,
            "max_artifact_fraction": 0.30,
            "max_dampened_fraction": 0.20,
            "max_af_fraction": 0.50,
            "flatline_variance_threshold": 0.01,
            "flatline_duration_seconds": 2.0,
            "art_sbp_max": 300,
            "art_sbp_min": 40,
            "art_dbp_max": 200,
            "art_dbp_min": 20,
            "art_pp_min": 10,
            "noise_ratio_threshold": 0.99,  # relaxed: synthetic QRS causes high HF content
            "noise_window_seconds": 5.0,
            "dampened_pp_threshold": 15,
            "disconnect_seconds": 2.0,  # > RR interval at 70 bpm so no window is QRS-free
        },
        "peaks": {
            "r_peak": {
                "rr_min_ms": 300,
                "rr_max_ms": 2000,
                "rr_diff_max_ms": 500,
            },
            "art": {
                "min_height_mmhg": 40,
                "min_distance_seconds": 0.25,
            },
        },
        "metrics": {
            "window_seconds": 30,
            "window_step": 1,
            "brs": {
                "min_sequence_length": 3,
                "min_correlation": 0.6,
                "search_window_ms": [100, 400],
                "delay_change_threshold_ms": 50,
            },
            "rsa": {
                "resp_freq_min": 0.12,
                "resp_freq_max": 0.40,
            },
        },
        "events": {
            "hypotension": {
                "map_threshold": 55,
                "min_duration_seconds": 30,  # shorter for tests
            },
            "hypertension": {
                "sbp_threshold": 180,
                "min_duration_seconds": 30,
            },
            "extreme_variability": {
                "map_swing_threshold": 30,
                "swing_window_seconds": 120,
                "min_episodes": 2,
                "episode_window_seconds": 300,
            },
        },
        "windows": {
            "pre_event_minutes": 5,   # shorter for tests
            "control_exclusion_minutes": 10,
            "max_controls_per_patient": 2,
            "event_buffer_minutes": 5,
        },
        "models": {
            "cv_folds": 3,
            "cv_repeats": 2,
            "random_state": 42,
            "milp": {
                "max_depth": 2,
                "max_features": 2,
                "bootstrap_reps": 10,
                "stability_freq_threshold": 0.80,
                "threshold_variability_fraction": 0.15,
            },
            "rf": {"n_estimators": 20, "cv_folds": 3},
            "xgboost": {"early_stopping_rounds": 5, "cv_folds": 3},
        },
    }


@pytest.fixture(scope="session")
def synthetic_ecg(fs):
    """Regular ECG-like signal at 70 bpm with no noise."""
    rng = np.random.default_rng(42)
    t = np.arange(N_SECONDS * fs) / fs
    # Simulate R-peaks at 70 bpm
    rr_s = 60.0 / 70.0
    signal = np.zeros(len(t), dtype=np.float32)
    _qrs = np.array([-0.2, -0.4, 0.6, 2.0, 0.4, -0.3, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    for r_time in np.arange(0, N_SECONDS, rr_s):
        idx = int(r_time * fs)
        i_start = max(0, idx - 5)
        i_end = min(len(signal), idx + 5)
        waveform = _qrs[i_start - (idx - 5) : i_end - (idx - 5)]
        if len(waveform) == i_end - i_start:
            signal[i_start:i_end] += waveform
    signal += rng.normal(0, 0.12, len(signal)).astype(np.float32)
    return signal


@pytest.fixture(scope="session")
def synthetic_art(fs):
    """Synthetic arterial pressure waveform at 70 bpm, 120/80 mmHg."""
    rng = np.random.default_rng(43)
    t = np.arange(N_SECONDS * fs) / fs
    rr_s = 60.0 / 70.0
    signal = np.full(len(t), 80.0, dtype=np.float32)
    for r_time in np.arange(0, N_SECONDS, rr_s):
        beat_t = t - r_time - 0.2  # 200 ms delay
        mask = (beat_t >= 0) & (beat_t < rr_s)
        pulse = 40.0 * np.clip(
            np.exp(-10 * beat_t[mask]) * np.sin(np.pi * beat_t[mask] / 0.3), 0, None
        )
        signal[mask] += pulse.astype(np.float32)
    signal += rng.normal(0, 1, len(signal)).astype(np.float32)
    return signal


@pytest.fixture(scope="session")
def synthetic_record(synthetic_ecg, synthetic_art, fs):
    return {
        "ecg": synthetic_ecg,
        "art": synthetic_art,
        "fs_ecg": fs,
        "fs_art": fs,
        "patient": "test_patient",
        "filepath": "synthetic",
        "source": "test",
    }
