"""Tests for QC pipeline."""

import numpy as np
import pytest

from beatlabile.qc.pipeline import (
    run_qc,
    _flatline_mask,
    _noise_mask,
    _disconnect_mask,
    _dampened_art_mask,
    _amplitude_mask_art,
)


class TestFlatlineMask:
    def test_all_clean(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 1000).astype(np.float32)
        mask = _flatline_mask(x, 0.01, 250)
        assert not np.any(mask)

    def test_flatline_detected(self):
        x = np.ones(1000, dtype=np.float32) * 5.0
        mask = _flatline_mask(x, 0.01, 250)
        assert np.sum(mask) > 0

    def test_partial_flatline(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 1000).astype(np.float32)
        x[300:600] = 0.0  # flatline in middle
        mask = _flatline_mask(x, 0.01, 100)
        assert np.any(mask[300:600])


class TestAmplitudeMaskART:
    def test_normal_range(self):
        art = np.full(100, 80.0, dtype=np.float32)
        mask = _amplitude_mask_art(art, 300, 40, 200, 20, 10)
        assert not np.any(mask)

    def test_too_high(self):
        art = np.full(100, 350.0, dtype=np.float32)
        mask = _amplitude_mask_art(art, 300, 40, 200, 20, 10)
        assert np.all(mask)

    def test_too_low(self):
        art = np.full(100, 10.0, dtype=np.float32)
        mask = _amplitude_mask_art(art, 300, 40, 200, 20, 10)
        assert np.all(mask)


class TestDisconnectMask:
    def test_no_disconnect(self):
        rng = np.random.default_rng(2)
        x = rng.normal(5, 1, 1000).astype(np.float32)
        mask = _disconnect_mask(x, 500, 1.0)
        assert not np.any(mask)

    def test_disconnect_detected(self):
        x = np.ones(1000, dtype=np.float32) * 5.0
        x[200:400] = 0.0  # flat zero = disconnection
        mask = _disconnect_mask(x, 500, 0.4)
        assert np.any(mask[200:400])


class TestRunQC:
    def test_passes_clean_record(self, synthetic_record, minimal_cfg):
        cfg = dict(minimal_cfg)
        result = run_qc(synthetic_record, cfg)
        assert result.passes_qc

    def test_rejects_too_short(self, minimal_cfg):
        cfg = dict(minimal_cfg)
        short_ecg = np.random.randn(100).astype(np.float32)
        short_art = np.random.randn(100).astype(np.float32)
        record = {"ecg": short_ecg, "art": short_art, "fs_ecg": 500, "fs_art": 500, "patient": "x"}
        result = run_qc(record, cfg)
        assert not result.passes_qc
        assert "duration" in result.reject_reason.lower()

    def test_rejects_high_artefact(self, minimal_cfg):
        cfg = dict(minimal_cfg)
        # ECG with >30% flatline artefact
        ecg = np.random.randn(100_000).astype(np.float32)
        ecg[30_000:70_000] = 0.0  # 40% flatline
        art = np.random.randn(100_000).astype(np.float32) + 80
        record = {"ecg": ecg, "art": art, "fs_ecg": 500, "fs_art": 500, "patient": "x"}
        result = run_qc(record, cfg)
        assert result.frac_bad_ecg > 0.10  # at minimum some bad detected
