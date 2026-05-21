"""Prediction window construction with anti-leakage protection (Section 8.5).

For each ELH event:
  · Extract a prediction window of 30 min ending at event onset
  · Label = 1 (event window)

For each patient:
  · Extract up to 3 control windows (PAM 60–100, HR 50–110) that are
    ≥ 1 h from any event.  Selected by maximising distance to nearest event
    and spread across the recording.
  · Label = 0 (control window)

Anti-leakage:
  · Patient-level CV splits (no window from the same patient spans train/test).
  · Minimum 15 min gap between end of window N and start of window N+1
    for consecutive events of the same patient.

Public API
----------
build_windows(features_df, events, art_result, patient_id, cfg) -> pd.DataFrame
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from beatlabile.events.detector import Event, EventResult
from beatlabile.signal.peaks import ARTResult


_HR_MIN = 50.0
_HR_MAX = 110.0
_MAP_CTRL_MIN = 60.0
_MAP_CTRL_MAX = 100.0


def _compute_instantaneous_hr(
    r_times_s: np.ndarray, at_time_s: float, window_s: float = 30.0
) -> float:
    """Estimate instantaneous HR in bpm from R-peaks near *at_time_s*."""
    mask = (r_times_s >= at_time_s - window_s) & (r_times_s <= at_time_s)
    r_in = r_times_s[mask]
    if len(r_in) < 2:
        return np.nan
    rr_mean_s = np.mean(np.diff(r_in))
    return 60.0 / rr_mean_s if rr_mean_s > 0 else np.nan


def _features_in_window(
    features_df: pd.DataFrame, t_start: float, t_end: float
) -> pd.DataFrame | None:
    """Extract metric rows whose window_start_s falls within [t_start, t_end).

    features_df contains 30-second sliding-window rows.  For a 30-minute
    prediction window there will be ~60 matching rows, which are then
    aggregated by _aggregate_prediction_window.
    """
    sub = features_df[
        (features_df["window_start_s"] >= t_start)
        & (features_df["window_start_s"] < t_end)
    ]
    return sub if len(sub) > 0 else None


def build_windows(
    features_df: pd.DataFrame,
    event_result: EventResult,
    art: ARTResult,
    r_times_s: np.ndarray,
    patient_id: str,
    cfg: dict,
    event_type: str = "hypotension",
) -> pd.DataFrame:
    """Build labelled prediction windows for one patient and one event type.

    Parameters
    ----------
    features_df  : Wide feature DataFrame from aggregate_window_features()
    event_result : EventResult from detect_events()
    art          : ARTResult (for MAP/HR validation of control windows)
    r_times_s    : Array of R-peak timestamps (s), for HR estimation
    patient_id   : String patient identifier
    cfg          : Full config dict
    event_type   : 'hypotension' | 'hypertension' | 'variability'

    Returns
    -------
    pd.DataFrame with columns: patient_id, event_type, label, window_start_s,
        window_end_s, event_onset_s, [feature columns...]
    """
    w_cfg = cfg["windows"]
    pre_event_s = w_cfg["pre_event_minutes"] * 60.0
    ctrl_excl_s = w_cfg["control_exclusion_minutes"] * 60.0
    max_ctrl = w_cfg["max_controls_per_patient"]
    event_buf_s = w_cfg["event_buffer_minutes"] * 60.0

    events: list[Event] = getattr(event_result, event_type)
    all_events: list[Event] = event_result.all_events()
    all_event_times = [(e.start_s, e.end_s) for e in all_events]

    rows: list[dict] = []

    # ---- Event windows --------------------------------------------------
    # Sort events chronologically and enforce inter-window gap
    events_sorted = sorted(events, key=lambda e: e.start_s)
    prev_win_end: float | None = None

    for ev in events_sorted:
        w_start = ev.start_s - pre_event_s
        w_end = ev.start_s

        if w_start < 0:
            continue

        # Anti-leakage: gap from previous event window
        if prev_win_end is not None and w_start < prev_win_end + event_buf_s:
            continue

        feats = _features_in_window(features_df, w_start, w_end)
        if feats is None:
            continue

        # Aggregate features across the prediction window
        feat_row = _aggregate_prediction_window(feats)
        feat_row.update({
            "patient_id": patient_id,
            "event_type": event_type,
            "label": 1,
            "window_start_s": w_start,
            "window_end_s": w_end,
            "event_onset_s": ev.start_s,
        })
        rows.append(feat_row)
        prev_win_end = w_end

    # ---- Control windows ------------------------------------------------
    # Candidate times = every 60-min mark through the record
    if len(features_df) == 0:
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    t_record_start = features_df["window_start_s"].min()
    t_record_end = features_df["window_end_s"].max()
    control_candidates: list[float] = []

    for t in np.arange(t_record_start, t_record_end - pre_event_s, pre_event_s / 2):
        t_end_c = t + pre_event_s
        # Must be ≥ ctrl_excl_s from any event start or end
        too_close = any(
            abs(t - ev_s) < ctrl_excl_s or abs(t_end_c - ev_e) < ctrl_excl_s
            for ev_s, ev_e in all_event_times
        )
        if too_close:
            continue

        # Physiology check: MAP 60–100 and HR 50–110
        in_win = (art.times_s >= t) & (art.times_s < t_end_c)
        if not np.any(in_win):
            continue
        map_mean = float(np.mean(art.map_beat[in_win]))
        if not (_MAP_CTRL_MIN <= map_mean <= _MAP_CTRL_MAX):
            continue
        hr = _compute_instantaneous_hr(r_times_s, t + pre_event_s / 2)
        if not np.isnan(hr) and not (_HR_MIN <= hr <= _HR_MAX):
            continue

        control_candidates.append(t)

    # Select up to max_ctrl candidates, prioritising spread and distance
    control_candidates = _select_controls(
        control_candidates, all_event_times, max_ctrl, pre_event_s
    )

    for t in control_candidates:
        feats = _features_in_window(features_df, t, t + pre_event_s)
        if feats is None:
            continue
        feat_row = _aggregate_prediction_window(feats)
        feat_row.update({
            "patient_id": patient_id,
            "event_type": event_type,
            "label": 0,
            "window_start_s": t,
            "window_end_s": t + pre_event_s,
            "event_onset_s": np.nan,
        })
        rows.append(feat_row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _aggregate_prediction_window(feats: pd.DataFrame) -> dict:
    """Aggregate 30-second metric rows into prediction-window features.

    For each metric column computes mean, std, slope (linear trend over
    time), min and max across the ~60 rows in the 30-minute window.
    """
    _SKIP = {"window_start_s", "window_end_s", "n_beats"}
    agg = {}
    for col in feats.columns:
        if col in _SKIP:
            continue
        try:
            vals = feats[col].dropna().values.astype(float)
        except (TypeError, ValueError):
            continue
        if len(vals) == 0:
            for sfx in ("mean", "std", "slope", "min", "max"):
                agg[f"{col}_{sfx}"] = np.nan
            continue
        agg[f"{col}_mean"]  = float(np.mean(vals))
        agg[f"{col}_std"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        agg[f"{col}_min"]   = float(np.min(vals))
        agg[f"{col}_max"]   = float(np.max(vals))
        if len(vals) >= 3:
            x = np.arange(len(vals), dtype=float)
            agg[f"{col}_slope"] = float(np.polyfit(x, vals, 1)[0])
        else:
            agg[f"{col}_slope"] = np.nan
    return agg


def _select_controls(
    candidates: list[float],
    event_times: list[tuple[float, float]],
    max_ctrl: int,
    spread_period: float,
) -> list[float]:
    """Select up to *max_ctrl* control windows maximising spread and distance from events."""
    if not candidates:
        return []
    if len(candidates) <= max_ctrl:
        return candidates

    # Score each candidate: distance to nearest event (higher = better)
    def score(t: float) -> float:
        if not event_times:
            return t
        min_dist = min(abs(t - es) for es, ee in event_times)
        return min_dist

    scored = sorted(candidates, key=score, reverse=True)
    # Greedy selection ensuring spread ≥ spread_period between selected
    selected: list[float] = []
    for t in scored:
        if all(abs(t - s) >= spread_period for s in selected):
            selected.append(t)
        if len(selected) >= max_ctrl:
            break
    return selected
