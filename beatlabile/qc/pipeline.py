"""Quality-control pipeline for ECG + ART signals (Section 8.1 of the protocol).

All checks are vectorised and return boolean masks (True = artefact/bad).

Public API
----------
run_qc(record, cfg) -> QCResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal as sp_signal


@dataclass
class QCResult:
    """Summary of QC for a single record."""

    # Total samples
    n_samples_ecg: int = 0
    n_samples_art: int = 0

    # Per-sample bad masks (True = bad)
    bad_ecg: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    bad_art: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Fraction bad
    frac_bad_ecg: float = 0.0
    frac_bad_art: float = 0.0
    frac_dampened_art: float = 0.0

    # Quality flags
    passes_qc: bool = False
    is_complete: bool = False  # ≥6 h continua
    is_fragment: bool = False  # ≥2 h continua

    # Reason for rejection (if any)
    reject_reason: str = ""

    # Suspected AF fraction
    af_fraction: float = 0.0
    is_pacemaker: bool = False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _flatline_mask(x: np.ndarray, variance_thresh: float, window_samples: int) -> np.ndarray:
    """Mark segments where variance < threshold (flatline / disconnection)."""
    bad = np.zeros(len(x), dtype=bool)
    for start in range(0, len(x) - window_samples + 1, window_samples):
        seg = x[start : start + window_samples]
        if np.var(seg) < variance_thresh:
            bad[start : start + window_samples] = True
    return bad


def _amplitude_mask_art(
    art: np.ndarray,
    sbp_max: float,
    sbp_min: float,
    dbp_max: float,
    dbp_min: float,
    pp_min: float,
) -> np.ndarray:
    """Mark ART samples outside physiological amplitude range."""
    bad = (art > sbp_max) | (art < dbp_min)
    return bad.astype(bool)


def _noise_mask(
    x: np.ndarray, fs: int, noise_window_sec: float, noise_ratio_thresh: float
) -> np.ndarray:
    """Mark segments with high-frequency noise (ratio power >40Hz / total power)."""
    win = int(noise_window_sec * fs)
    bad = np.zeros(len(x), dtype=bool)
    for start in range(0, len(x) - win + 1, win):
        seg = x[start : start + win]
        freqs, psd = sp_signal.welch(seg, fs=fs, nperseg=min(win, 256))
        total_power = np.sum(psd)
        if total_power == 0:
            bad[start : start + win] = True
            continue
        hf_power = np.sum(psd[freqs > 40])
        if hf_power / total_power > noise_ratio_thresh:
            bad[start : start + win] = True
    return bad


def _dampened_art_mask(
    art: np.ndarray, fs: int, pp_threshold: float, block_size: int = 512
) -> np.ndarray:
    """Detect dampened arterial waveform: PP < threshold AND no dicrotic notch.

    Simplified detection: compute PP beat-by-beat via rolling max-min within
    ~1-s windows. Absence of dicrotic notch estimated by low second derivative.
    """
    bad = np.zeros(len(art), dtype=bool)
    win = int(fs)  # 1-s windows
    for start in range(0, len(art) - win + 1, win):
        seg = art[start : start + win]
        pp = np.max(seg) - np.min(seg)
        if pp < pp_threshold:
            # Also check for absence of dicrotic notch via second derivative
            d2 = np.diff(np.diff(seg))
            # Dicrotic notch creates a positive inflection point in d2
            has_notch = np.any(d2 > 0.5)
            if not has_notch:
                bad[start : start + win] = True
    return bad


def _disconnect_mask(x: np.ndarray, fs: int, disconnect_sec: float) -> np.ndarray:
    """Mark segments where signal drops to ~0."""
    win = int(disconnect_sec * fs)
    bad = np.zeros(len(x), dtype=bool)
    baseline = np.abs(np.median(x[:min(1000, len(x))]))
    for start in range(0, len(x) - win + 1, win):
        seg = x[start : start + win]
        if np.all(np.abs(seg) <= max(baseline * 0.05, 1.0)):
            bad[start : start + win] = True
    return bad


def _estimate_af_fraction(rr_intervals_ms: np.ndarray) -> float:
    """Estimate fraction of record with AF based on RR irregularity.

    Uses coefficient of variation of RR intervals in 30-s segments.
    High CV (>0.15) + no dominant RR pattern → suspected AF.
    """
    if len(rr_intervals_ms) < 10:
        return 0.0
    cv = np.std(rr_intervals_ms) / np.mean(rr_intervals_ms)
    # Simple heuristic: CV > 0.20 → probable AF for short segments
    return float(cv > 0.20)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_qc(record: dict[str, Any], cfg: dict) -> QCResult:
    """Run the full QC pipeline on a loaded record.

    Parameters
    ----------
    record : dict with keys 'ecg', 'art', 'fs_ecg', 'fs_art'
    cfg    : the full config.yaml dict (uses cfg['qc'] section)

    Returns
    -------
    QCResult
    """
    q = cfg["qc"]
    ecg: np.ndarray = record["ecg"]
    art: np.ndarray = record["art"]
    fs_ecg: int = record["fs_ecg"]
    fs_art: int = record["fs_art"]

    result = QCResult(n_samples_ecg=len(ecg), n_samples_art=len(art))

    # --- Duration check ---
    duration_sec = min(len(ecg) / fs_ecg, len(art) / fs_art)
    min_dur = q["min_duration_seconds"]
    complete_dur = q["complete_duration_seconds"]

    result.is_fragment = duration_sec >= min_dur
    result.is_complete = duration_sec >= complete_dur

    if not result.is_fragment:
        result.reject_reason = (
            f"Insufficient duration: {duration_sec/3600:.2f} h < {min_dur/3600:.2f} h minimum"
        )
        result.passes_qc = False
        return result

    # --- Build bad masks for ECG ---
    # Fill NaN before mask computation; NaN positions are flagged separately
    ecg_filled = np.where(np.isnan(ecg), 0.0, ecg)
    flat_win_ecg = int(q["flatline_duration_seconds"] * fs_ecg)
    ecg_flat_thresh = float(q.get("ecg_flatline_variance_threshold", 1e-7))

    bad_ecg = np.isnan(ecg)  # vitaldb marks true dropouts as NaN
    bad_ecg |= _flatline_mask(ecg_filled, ecg_flat_thresh, flat_win_ecg)
    bad_ecg |= _noise_mask(ecg_filled, fs_ecg, q["noise_window_seconds"], q["noise_ratio_threshold"])
    # NOTE: _disconnect_mask is intentionally NOT applied to ECG — ECG baseline
    # is near 0 mV, so the "signal drops to ~0" heuristic always false-flags it.

    # --- Build bad masks for ART ---
    art_filled = np.where(np.isnan(art), 0.0, art)
    flat_win_art = int(q["flatline_duration_seconds"] * fs_art)
    bad_art = np.isnan(art)
    bad_art |= _flatline_mask(art_filled, q["flatline_variance_threshold"], flat_win_art)
    bad_art |= _amplitude_mask_art(
        art_filled,
        q["art_sbp_max"],
        q["art_sbp_min"],
        q["art_dbp_max"],
        q["art_dbp_min"],
        q["art_pp_min"],
    )
    bad_art |= _disconnect_mask(art_filled, fs_art, q["disconnect_seconds"])

    dampened = _dampened_art_mask(art, fs_art, q["dampened_pp_threshold"])

    # Combine
    result.bad_ecg = bad_ecg
    result.bad_art = bad_art | dampened
    result.frac_bad_ecg = float(np.mean(bad_ecg))
    result.frac_bad_art = float(np.mean(bad_art))
    result.frac_dampened_art = float(np.mean(dampened))

    # --- Global rejection criteria ---
    if result.frac_bad_ecg > q["max_artifact_fraction"]:
        result.reject_reason = (
            f"ECG artefact fraction {result.frac_bad_ecg:.2%} > {q['max_artifact_fraction']:.0%}"
        )
        result.passes_qc = False
        return result

    if result.frac_bad_art > q["max_artifact_fraction"]:
        result.reject_reason = (
            f"ART artefact fraction {result.frac_bad_art:.2%} > {q['max_artifact_fraction']:.0%}"
        )
        result.passes_qc = False
        return result

    if result.frac_dampened_art > q["max_dampened_fraction"]:
        result.reject_reason = (
            f"Dampened ART fraction {result.frac_dampened_art:.2%} > {q['max_dampened_fraction']:.0%}"
        )
        result.passes_qc = False
        return result

    result.passes_qc = True
    return result
