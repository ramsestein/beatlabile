"""R-peak and arterial peak detection (Section 8.2 of the protocol).

R-peak detection
----------------
Pan-Tompkins adaptive algorithm followed by ectopic-beat post-processing:
  · RR < 300 ms or > 2000 ms → ectopic candidate
  · |RR_n - RR_{n-1}| > 500 ms → ectopic candidate
  Ectopics are flagged (not removed) so they can be excluded selectively.

ART peak detection
------------------
scipy.signal.find_peaks with minimum height and distance constraints,
followed by physiological validation (each systolic peak has a preceding
and subsequent diastolic valley).

Public API
----------
detect_r_peaks(ecg, fs, cfg)  -> RRResult
detect_art_peaks(art, fs, cfg) -> ARTResult
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RRResult:
    """Output of R-peak detector."""
    r_peaks: np.ndarray              # sample indices
    rr_intervals_ms: np.ndarray      # inter-beat intervals in ms
    rr_times_s: np.ndarray           # timestamps of each RR interval (end beat)
    is_ectopic: np.ndarray           # bool, same length as rr_intervals_ms
    fs: int


@dataclass
class ARTResult:
    """Output of ART peak/valley detector."""
    systolic_peaks: np.ndarray       # sample indices of SBP peaks
    diastolic_valleys: np.ndarray    # sample indices of DBP valleys
    sbp: np.ndarray                  # mmHg at systolic peaks
    dbp: np.ndarray                  # mmHg at diastolic valleys
    map_beat: np.ndarray             # MAP per beat: DBP + (SBP-DBP)/3
    pp_beat: np.ndarray              # pulse pressure per beat
    times_s: np.ndarray              # timestamps of systolic peaks (seconds)
    fs: int


# ---------------------------------------------------------------------------
# Pan-Tompkins ECG R-peak detector
# ---------------------------------------------------------------------------

def _bandpass_filter(ecg: np.ndarray, fs: int) -> np.ndarray:
    """5–15 Hz bandpass — Pan-Tompkins step 1."""
    low, high = 5.0, 15.0
    sos = sp_signal.butter(2, [low / (fs / 2), high / (fs / 2)], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, ecg)


def _derivative_filter(x: np.ndarray, fs: int) -> np.ndarray:
    """First-order derivative emphasising slope."""
    h = np.array([-1, -2, 0, 2, 1], dtype=float) * (1.0 / 8.0) * fs
    return np.convolve(x, h, mode="same")


def _moving_window_integration(x: np.ndarray, fs: int, window_ms: int = 150) -> np.ndarray:
    """Moving-window integration — Pan-Tompkins step 4."""
    win = max(1, int(window_ms / 1000 * fs))
    kernel = np.ones(win) / win
    return np.convolve(x**2, kernel, mode="same")


def _adaptive_threshold_peaks(
    integrated: np.ndarray,
    fs: int,
    min_distance_ms: int = 300,
) -> np.ndarray:
    """Adaptive threshold detection on the integrated signal."""
    min_dist = max(1, int(min_distance_ms / 1000 * fs))
    # Initial threshold = 50% of max in first 2 s
    init_win = min(int(2 * fs), len(integrated))
    threshold = 0.5 * np.max(integrated[:init_win])

    peaks, _ = sp_signal.find_peaks(integrated, height=threshold, distance=min_dist)
    # Adaptive update: SPKI (signal peak) and NPKI (noise peak)
    # Simplified: rerun with threshold = mean of detected peaks * 0.5
    if len(peaks) > 0:
        threshold = 0.5 * np.mean(integrated[peaks])
        peaks, _ = sp_signal.find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks


def detect_r_peaks(
    ecg: np.ndarray,
    fs: int,
    cfg: dict,
    bad_mask: np.ndarray | None = None,
) -> RRResult:
    """Run Pan-Tompkins with ectopic post-processing.

    Parameters
    ----------
    ecg      : Raw ECG signal
    fs       : Sampling frequency
    cfg      : Full config dict (uses cfg['peaks']['r_peak'])
    bad_mask : Optional bool array marking bad samples (True = bad)
    """
    p = cfg["peaks"]["r_peak"]
    rr_min = p["rr_min_ms"]
    rr_max = p["rr_max_ms"]
    rr_diff = p["rr_diff_max_ms"]

    # Replace bad samples with linear interpolation before filtering
    ecg_clean = ecg.copy()
    if bad_mask is not None and np.any(bad_mask):
        indices = np.arange(len(ecg_clean))
        good = ~bad_mask
        ecg_clean = np.interp(indices, indices[good], ecg_clean[good])

    # Pan-Tompkins pipeline
    filtered = _bandpass_filter(ecg_clean, fs)
    deriv = _derivative_filter(filtered, fs)
    integrated = _moving_window_integration(deriv, fs)
    r_peaks = _adaptive_threshold_peaks(integrated, fs, min_distance_ms=rr_min)

    if len(r_peaks) < 2:
        return RRResult(
            r_peaks=r_peaks,
            rr_intervals_ms=np.array([]),
            rr_times_s=np.array([]),
            is_ectopic=np.array([], dtype=bool),
            fs=fs,
        )

    # RR intervals in ms
    rr_ms = np.diff(r_peaks) / fs * 1000.0
    rr_times = r_peaks[1:] / fs  # seconds

    # Ectopic flagging
    is_ectopic = np.zeros(len(rr_ms), dtype=bool)
    is_ectopic |= rr_ms < rr_min
    is_ectopic |= rr_ms > rr_max
    rr_diff_arr = np.abs(np.diff(np.concatenate([[rr_ms[0]], rr_ms])))
    is_ectopic |= rr_diff_arr > rr_diff

    return RRResult(
        r_peaks=r_peaks,
        rr_intervals_ms=rr_ms,
        rr_times_s=rr_times,
        is_ectopic=is_ectopic,
        fs=fs,
    )


# ---------------------------------------------------------------------------
# ART peak / valley detection
# ---------------------------------------------------------------------------

def detect_art_peaks(
    art: np.ndarray,
    fs: int,
    cfg: dict,
    bad_mask: np.ndarray | None = None,
) -> ARTResult:
    """Detect systolic peaks and diastolic valleys in ART signal.

    Parameters
    ----------
    art      : Arterial pressure signal (mmHg)
    fs       : Sampling frequency
    cfg      : Full config dict (uses cfg['peaks']['art'])
    bad_mask : Optional bool array marking bad samples (True = bad)
    """
    p = cfg["peaks"]["art"]
    min_dist = max(1, int(p["min_distance_seconds"] * fs))
    min_h = p["min_height_mmhg"]

    art_work = art.copy()
    if bad_mask is not None and np.any(bad_mask):
        indices = np.arange(len(art_work))
        good = ~bad_mask
        if np.sum(good) > 2:
            art_work = np.interp(indices, indices[good], art_work[good])

    # Systolic peaks
    sys_peaks, _ = sp_signal.find_peaks(art_work, height=min_h, distance=min_dist)

    if len(sys_peaks) < 2:
        empty = np.array([], dtype=np.float32)
        return ARTResult(
            systolic_peaks=sys_peaks,
            diastolic_valleys=np.array([], dtype=int),
            sbp=empty,
            dbp=empty,
            map_beat=empty,
            pp_beat=empty,
            times_s=empty,
            fs=fs,
        )

    # Diastolic valleys: find minimum between consecutive systolic peaks
    dia_valleys = np.array(
        [
            sys_peaks[i] + np.argmin(art_work[sys_peaks[i] : sys_peaks[i + 1]])
            for i in range(len(sys_peaks) - 1)
        ],
        dtype=int,
    )

    # Align arrays: each systolic peak (index 1 onwards) paired with preceding valley
    # Use sys_peaks[1:] and dia_valleys
    n = min(len(sys_peaks) - 1, len(dia_valleys))
    sys_idx = sys_peaks[1 : n + 1]
    dia_idx = dia_valleys[:n]

    sbp = art_work[sys_idx].astype(np.float32)
    dbp = art_work[dia_idx].astype(np.float32)
    pp = sbp - dbp
    map_b = (dbp + pp / 3.0).astype(np.float32)
    times = sys_idx / fs

    # Physiological validation: remove beats where SBP <= DBP
    valid = sbp > dbp
    sys_idx = sys_idx[valid]
    dia_idx = dia_idx[valid]
    sbp = sbp[valid]
    dbp = dbp[valid]
    pp = pp[valid]
    map_b = map_b[valid]
    times = times[valid]

    return ARTResult(
        systolic_peaks=sys_idx,
        diastolic_valleys=dia_idx,
        sbp=sbp,
        dbp=dbp,
        map_beat=map_b,
        pp_beat=pp,
        times_s=times,
        fs=fs,
    )
