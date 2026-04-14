"""ECG–ART synchronisation for BRS calculation (Section 8.3 of the protocol).

For each R-peak, find the corresponding systolic peak within a search window
of 100–400 ms post-R.  Verify and recalibrate the delay if it drifts > 50 ms
between segments.

Public API
----------
sync_ecg_art(rr_result, art_result, cfg) -> SyncResult
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beatlabile.signal.peaks import RRResult, ARTResult


@dataclass
class SyncResult:
    """Paired beat-level ECG + ART data for BRS calculation."""
    # For each matched beat:
    rr_ms: np.ndarray          # RR interval (ms) preceding the beat
    sbp: np.ndarray            # SBP (mmHg) of paired systolic peak
    r_time_s: np.ndarray       # timestamp of R-peak (s)
    sbp_time_s: np.ndarray     # timestamp of systolic peak (s)
    delay_ms: np.ndarray       # R → SBP delay per beat (ms)
    is_ectopic: np.ndarray     # ectopic flag from RR detector
    matched: np.ndarray        # bool: this beat was successfully matched


def sync_ecg_art(
    rr: RRResult,
    art: ARTResult,
    cfg: dict,
    segment_duration_s: float = 300.0,
) -> SyncResult:
    """Match each R-peak to the closest systolic peak within ± search window.

    Parameters
    ----------
    rr               : RRResult from detect_r_peaks
    art              : ARTResult from detect_art_peaks
    cfg              : Full config dict (uses cfg['metrics']['brs'])
    segment_duration_s : Window over which to check delay stability
    """
    brs_cfg = cfg["metrics"]["brs"]
    win_lo_ms, win_hi_ms = brs_cfg["search_window_ms"]
    delay_drift_thresh = brs_cfg["delay_change_threshold_ms"]

    r_times = rr.r_peaks / rr.fs  # R-peak timestamps in seconds
    art_times = art.times_s        # systolic peak timestamps in seconds

    n_beats = len(rr.rr_intervals_ms)
    rr_ms_out = np.full(n_beats, np.nan)
    sbp_out = np.full(n_beats, np.nan)
    r_time_out = np.full(n_beats, np.nan)
    sbp_time_out = np.full(n_beats, np.nan)
    delay_out = np.full(n_beats, np.nan)
    matched = np.zeros(n_beats, dtype=bool)

    # Initial search window in seconds
    lo_s = win_lo_ms / 1000.0
    hi_s = win_hi_ms / 1000.0

    for i in range(n_beats):
        r_t = r_times[i + 1] if i + 1 < len(r_times) else r_times[-1]
        # Find systolic peaks in [r_t + lo_s, r_t + hi_s]
        candidates = np.where(
            (art_times >= r_t + lo_s) & (art_times <= r_t + hi_s)
        )[0]
        if len(candidates) == 0:
            continue
        # Pick closest
        best_idx = candidates[np.argmin(np.abs(art_times[candidates] - (r_t + (lo_s + hi_s) / 2)))]
        delay_ms = (art_times[best_idx] - r_t) * 1000.0

        rr_ms_out[i] = rr.rr_intervals_ms[i]
        sbp_out[i] = art.sbp[best_idx]
        r_time_out[i] = r_t
        sbp_time_out[i] = art_times[best_idx]
        delay_out[i] = delay_ms
        matched[i] = True

    # --- Delay stability check: recalibrate per segment ---
    if np.any(matched):
        valid_delays = delay_out[matched]
        valid_times = r_time_out[matched]
        # Segment into chunks
        if len(valid_times) > 0:
            t_min, t_max = valid_times.min(), valid_times.max()
            segs = np.arange(t_min, t_max, segment_duration_s)
            prev_mean = None
            for seg_start in segs:
                in_seg = (valid_times >= seg_start) & (valid_times < seg_start + segment_duration_s)
                if not np.any(in_seg):
                    continue
                seg_mean = np.mean(valid_delays[in_seg])
                if prev_mean is not None and abs(seg_mean - prev_mean) > delay_drift_thresh:
                    # Recalibrate: update search window for remaining beats
                    shift = seg_mean - (win_lo_ms + win_hi_ms) / 2.0
                    lo_s = (win_lo_ms + shift * 0.5) / 1000.0
                    hi_s = (win_hi_ms + shift * 0.5) / 1000.0
                prev_mean = seg_mean

    return SyncResult(
        rr_ms=rr_ms_out,
        sbp=sbp_out,
        r_time_s=r_time_out,
        sbp_time_s=sbp_time_out,
        delay_ms=delay_out,
        is_ectopic=rr.is_ectopic,
        matched=matched,
    )
