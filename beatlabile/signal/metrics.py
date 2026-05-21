"""Compute physiological metrics from beat-to-beat data (Section 7 of protocol).

Metrics
-------
BRS   — Baroreflex sensitivity (sequence method, ≥3 monotonic beats, r>0.6)
SDNN  — SD of NN intervals
RMSSD — Root mean square of successive RR differences
pNN50 — % successive differences > 50 ms
ARV   — Average Real Variability of MAP beat-to-beat
CV_PA — Coefficient of variation of arterial pressure
STD_PA — Standard deviation of arterial pressure beat-to-beat
RSA   — Respiratory sinus arrhythmia (amplitude of respiratory modulation of RR)

All sliding-window metrics computed with:
  window = 30 s (≈30–80 beats at normal HR) — configurable
  step   = 1 beat

Public API
----------
compute_all_metrics(sync, art, cfg, bad_mask_art) -> pd.DataFrame
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from beatlabile.signal.sync import SyncResult
from beatlabile.signal.peaks import ARTResult


# ---------------------------------------------------------------------------
# BRS: sequence method
# ---------------------------------------------------------------------------

def _brs_sequence_method(
    rr_ms: np.ndarray,
    sbp_mmhg: np.ndarray,
    min_len: int = 3,
    min_r: float = 0.6,
) -> float:
    """Compute BRS from paired (SBP, RR) arrays using the sequence method.

    Finds monotonically increasing and decreasing subsequences of ≥ min_len
    consecutive beats where ΔSBP and ΔRR covary, computes the slope of
    SBP→RR regression for each, and returns the mean slope (ms/mmHg).

    Returns NaN if no valid sequences found.
    """
    valid = np.isfinite(rr_ms) & np.isfinite(sbp_mmhg)
    rr = rr_ms[valid]
    sbp = sbp_mmhg[valid]

    if len(rr) < min_len:
        return np.nan

    slopes = []
    n = len(rr)

    for direction in (+1, -1):  # up-sequences and down-sequences
        i = 0
        while i < n - 1:
            # Start a sequence
            seq_rr = [rr[i]]
            seq_sbp = [sbp[i]]
            j = i + 1
            while j < n:
                drr = rr[j] - rr[j - 1]
                dsbp = sbp[j] - sbp[j - 1]
                if direction * drr > 0 and direction * dsbp > 0:
                    seq_rr.append(rr[j])
                    seq_sbp.append(sbp[j])
                    j += 1
                else:
                    break
            if len(seq_rr) >= min_len:
                arr_rr = np.array(seq_rr)
                arr_sbp = np.array(seq_sbp)
                # Linear regression
                r_mat = np.corrcoef(arr_sbp, arr_rr)
                r_val = r_mat[0, 1]
                if abs(r_val) >= min_r:
                    slope = np.polyfit(arr_sbp, arr_rr, 1)[0]
                    slopes.append(slope)
            i = j if j > i + 1 else i + 1

    return float(np.mean(slopes)) if slopes else np.nan


# ---------------------------------------------------------------------------
# Sliding-window metric computation
# ---------------------------------------------------------------------------

def _window_indices(
    times_s: np.ndarray,
    window_s: float,
) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for sliding windows of *window_s* seconds.

    Step = 1 beat.
    """
    windows = []
    n = len(times_s)
    for i in range(n):
        t_start = times_s[i]
        t_end = t_start + window_s
        j = np.searchsorted(times_s, t_end, side="right")
        if j > i + 1:  # at least 2 beats
            windows.append((i, j))
    return windows


def _sdnn(rr: np.ndarray) -> float:
    return float(np.std(rr, ddof=1)) if len(rr) > 1 else np.nan


def _rmssd(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return np.nan
    diffs = np.diff(rr)
    return float(np.sqrt(np.mean(diffs**2)))


def _pnn50(rr: np.ndarray) -> float:
    if len(rr) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(rr)) > 50.0) * 100.0)


def _arv(map_b: np.ndarray) -> float:
    if len(map_b) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(map_b))))


def _cv_pa(pa: np.ndarray) -> float:
    m = np.mean(pa)
    if m == 0 or len(pa) < 2:
        return np.nan
    return float(np.std(pa, ddof=1) / m)


def _std_pa(pa: np.ndarray) -> float:
    return float(np.std(pa, ddof=1)) if len(pa) > 1 else np.nan


def _rsa_amplitude(rr: np.ndarray, times_s: np.ndarray, cfg_rsa: dict) -> float:
    """Estimate RSA as amplitude of respiratory modulation of RR.

    Method: power in the respiratory frequency band (0.12–0.40 Hz) of the
    RR tachogram interpolated to uniform sampling.
    """
    if len(rr) < 8 or len(times_s) < 8:
        return np.nan
    try:
        fs_uni = 4.0  # 4 Hz resample
        t_uni = np.arange(times_s[0], times_s[-1], 1.0 / fs_uni)
        rr_uni = np.interp(t_uni, times_s, rr)
        freqs, psd = sp_signal.welch(rr_uni, fs=fs_uni, nperseg=min(len(rr_uni), 64))
        lo = cfg_rsa["resp_freq_min"]
        hi = cfg_rsa["resp_freq_max"]
        band = (freqs >= lo) & (freqs <= hi)
        return float(np.sqrt(np.sum(psd[band]) * (freqs[1] - freqs[0]))) if np.any(band) else np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_all_metrics(
    sync: SyncResult,
    art: ARTResult,
    cfg: dict,
) -> pd.DataFrame:
    """Compute all sliding-window metrics.

    Returns a DataFrame with one row per window, indexed by window start time.
    Columns: brs, sdnn, rmssd, pnn50, arv, cv_pa, std_pa, rsa, time_s, n_beats
    """
    m_cfg = cfg["metrics"]
    window_s: float = float(m_cfg["window_seconds"])
    brs_cfg = m_cfg["brs"]
    rsa_cfg = m_cfg["rsa"]

    # Use only matched, non-ectopic beats
    valid = sync.matched & np.isfinite(sync.rr_ms) & np.isfinite(sync.sbp)
    # For HRV we also exclude ectopics
    non_ectopic = valid & ~sync.is_ectopic

    rr_all = sync.rr_ms[valid]
    sbp_all = sync.sbp[valid]
    t_all = sync.r_time_s[valid]

    rr_hrv = sync.rr_ms[non_ectopic]
    t_hrv = sync.r_time_s[non_ectopic]

    # ART beat-to-beat series (aligned to art result times)
    map_b_all = art.map_beat
    map_t_all = art.times_s
    sbp_b_all = art.sbp
    map_b_aligned: np.ndarray

    records = []
    if len(t_hrv) == 0:
        return pd.DataFrame(columns=[
            "time_s", "brs", "sdnn", "rmssd", "pnn50", "arv", "cv_pa", "std_pa", "rsa", "n_beats"
        ])

    windows_hrv = _window_indices(t_hrv, window_s)

    for w_start, w_end in windows_hrv:
        rr_w = rr_hrv[w_start:w_end]
        t_w = t_hrv[w_start:w_end]
        t0 = t_w[0]

        # BRS: use matched beats including ectopic-adjacent (per protocol they are excluded from HRV but included for BRS count)
        brs_valid = valid & (sync.r_time_s >= t0) & (sync.r_time_s < t0 + window_s)
        brs_val = _brs_sequence_method(
            sync.rr_ms[brs_valid],
            sync.sbp[brs_valid],
            min_len=brs_cfg["min_sequence_length"],
            min_r=brs_cfg["min_correlation"],
        )

        # ART metrics in same time window
        art_in = (map_t_all >= t0) & (map_t_all < t0 + window_s)
        map_w = map_b_all[art_in]
        sbp_w = sbp_b_all[art_in]

        records.append({
            "time_s": t0,
            "brs": brs_val,
            "sdnn": _sdnn(rr_w),
            "rmssd": _rmssd(rr_w),
            "pnn50": _pnn50(rr_w),
            "arv": _arv(map_w) if len(map_w) > 1 else np.nan,
            "cv_pa": _cv_pa(map_w) if len(map_w) > 1 else np.nan,
            "std_pa": _std_pa(map_w) if len(map_w) > 1 else np.nan,
            "rsa": _rsa_amplitude(rr_w, t_w, rsa_cfg),
            "n_beats": len(rr_w),
        })

    return pd.DataFrame(records)


def aggregate_window_features(metrics_df: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """Aggregate per-beat metrics into prediction-window features.

    For each prediction window of *window_s* seconds, compute:
      mean, std, slope (linear trend), min, max, cv
    for each metric.

    Returns a wide DataFrame with columns like brs_mean, brs_std, ...
    """
    feature_cols = ["brs", "sdnn", "rmssd", "pnn50", "arv", "cv_pa", "std_pa", "rsa"]
    rows = []

    if metrics_df.empty:
        return pd.DataFrame()

    times = metrics_df["time_s"].values
    t_min = times[0]
    t_max = times[-1]

    for t_start in np.arange(t_min, t_max - window_s + 1, window_s / 2):
        in_win = (times >= t_start) & (times < t_start + window_s)
        sub = metrics_df.loc[in_win]
        if len(sub) < 3:
            continue
        row: dict = {"window_start_s": t_start, "window_end_s": t_start + window_s}
        for col in feature_cols:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                for agg in ("mean", "std", "slope", "min", "max", "cv"):
                    row[f"{col}_{agg}"] = np.nan
                continue
            row[f"{col}_mean"] = np.mean(vals)
            row[f"{col}_std"] = np.std(vals, ddof=1) if len(vals) > 1 else np.nan
            row[f"{col}_min"] = np.min(vals)
            row[f"{col}_max"] = np.max(vals)
            row[f"{col}_cv"] = (np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else np.nan
            if len(vals) >= 2:
                x = np.arange(len(vals), dtype=float)
                row[f"{col}_slope"] = float(np.polyfit(x, vals, 1)[0])
            else:
                row[f"{col}_slope"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)
