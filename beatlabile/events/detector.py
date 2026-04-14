"""Automatic detection of haemodynamic lability events (Section 6 of protocol).

Three independent outcome types:
  1. Hypotension   — MAP < 55 mmHg for ≥ 3 consecutive minutes
  2. Hypertension  — SBP > 180 mmHg for ≥ 3 consecutive minutes
  3. Extreme variability — MAP swing > 30 mmHg peak-to-trough in ≤ 10 min,
                           ≥ 2 episodes within any 30-min window

Public API
----------
detect_events(art_result, cfg) -> EventResult
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from beatlabile.signal.peaks import ARTResult


@dataclass
class Event:
    """A single haemodynamic lability event."""
    event_type: str       # 'hypotension' | 'hypertension' | 'variability'
    start_s: float        # onset timestamp (s)
    end_s: float          # offset timestamp (s)
    peak_value: float     # worst MAP (hypo) / SBP (hyper) / swing (var)


@dataclass
class EventResult:
    hypotension: list[Event] = field(default_factory=list)
    hypertension: list[Event] = field(default_factory=list)
    variability: list[Event] = field(default_factory=list)

    def all_events(self) -> list[Event]:
        return self.hypotension + self.hypertension + self.variability

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "event_type": e.event_type,
                "start_s": e.start_s,
                "end_s": e.end_s,
                "peak_value": e.peak_value,
            }
            for e in self.all_events()
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["event_type", "start_s", "end_s", "peak_value"]
        )


def _detect_threshold_events(
    values: np.ndarray,
    times_s: np.ndarray,
    threshold: float,
    above: bool,
    min_duration_s: float,
) -> list[Event]:
    """Generic detector: contiguous run of values above/below threshold lasting ≥ min_duration.

    Parameters
    ----------
    values       : beat-level signal (MAP or SBP)
    times_s      : timestamps of each beat
    threshold    : crossing threshold
    above        : True = detect above, False = detect below
    min_duration_s : minimum run duration in seconds
    """
    if len(values) == 0:
        return []

    condition = values > threshold if above else values < threshold
    events: list[Event] = []
    in_event = False
    start_idx = 0

    for i, flag in enumerate(condition):
        if flag and not in_event:
            in_event = True
            start_idx = i
        elif not flag and in_event:
            duration = times_s[i - 1] - times_s[start_idx]
            if duration >= min_duration_s:
                extremum = (np.max(values[start_idx:i]) if above
                            else np.min(values[start_idx:i]))
                events.append(Event(
                    event_type="hypertension" if above else "hypotension",
                    start_s=float(times_s[start_idx]),
                    end_s=float(times_s[i - 1]),
                    peak_value=float(extremum),
                ))
            in_event = False

    # Handle event reaching end of record
    if in_event:
        duration = times_s[-1] - times_s[start_idx]
        if duration >= min_duration_s:
            extremum = (np.max(values[start_idx:]) if above
                        else np.min(values[start_idx:]))
            events.append(Event(
                event_type="hypertension" if above else "hypotension",
                start_s=float(times_s[start_idx]),
                end_s=float(times_s[-1]),
                peak_value=float(extremum),
            ))
    return events


def _detect_variability_events(
    map_b: np.ndarray,
    times_s: np.ndarray,
    swing_threshold: float,
    swing_window_s: float,
    min_episodes: int,
    episode_window_s: float,
) -> list[Event]:
    """Detect extreme haemodynamic variability episodes.

    An episode is defined as: MAP peak-to-trough swing > swing_threshold mmHg
    within swing_window_s seconds.

    An ELH event occurs when ≥ min_episodes episodes fall within episode_window_s.
    """
    if len(map_b) < 4:
        return []

    # Step 1: detect individual swing episodes
    episode_times: list[float] = []
    n = len(map_b)
    for i in range(n):
        t0 = times_s[i]
        in_win = (times_s >= t0) & (times_s <= t0 + swing_window_s)
        seg = map_b[in_win]
        if len(seg) < 3:
            continue
        swing = float(np.max(seg) - np.min(seg))
        if swing > swing_threshold:
            episode_times.append(t0)

    if len(episode_times) < min_episodes:
        return []

    # Deduplicate overlapping episodes (merge within 30 s)
    ep_arr = np.array(sorted(set(episode_times)))
    merged: list[float] = [ep_arr[0]]
    for t in ep_arr[1:]:
        if t - merged[-1] > 30.0:
            merged.append(t)
    ep_arr = np.array(merged)

    if len(ep_arr) < min_episodes:
        return []

    # Step 2: find windows of ≥ min_episodes within episode_window_s
    events: list[Event] = []
    i = 0
    while i <= len(ep_arr) - min_episodes:
        t_start = ep_arr[i]
        t_end = t_start + episode_window_s
        count = int(np.sum(ep_arr <= t_end))
        if count >= min_episodes:
            # Find worst swing in the full episode window
            in_win = (times_s >= t_start) & (times_s <= t_end)
            seg = map_b[in_win]
            peak_swing = float(np.max(seg) - np.min(seg)) if len(seg) > 1 else 0.0
            # Find end of last contributing episode
            last_ep = ep_arr[ep_arr <= t_end][-1]
            events.append(Event(
                event_type="variability",
                start_s=float(t_start),
                end_s=float(last_ep + swing_window_s),
                peak_value=peak_swing,
            ))
            # Skip forward past this event
            i += count
        else:
            i += 1

    return events


def detect_events(art: ARTResult, cfg: dict) -> EventResult:
    """Detect all three ELH types from beat-level ART data.

    Parameters
    ----------
    art : ARTResult from detect_art_peaks
    cfg : Full config dict (uses cfg['events'])
    """
    ev_cfg = cfg["events"]

    # 1. Hypotension
    hypo = _detect_threshold_events(
        values=art.map_beat,
        times_s=art.times_s,
        threshold=ev_cfg["hypotension"]["map_threshold"],
        above=False,
        min_duration_s=ev_cfg["hypotension"]["min_duration_seconds"],
    )

    # 2. Hypertension
    hyper = _detect_threshold_events(
        values=art.sbp,
        times_s=art.times_s,
        threshold=ev_cfg["hypertension"]["sbp_threshold"],
        above=True,
        min_duration_s=ev_cfg["hypertension"]["min_duration_seconds"],
    )

    # 3. Extreme variability
    var_cfg = ev_cfg["extreme_variability"]
    variability = _detect_variability_events(
        map_b=art.map_beat,
        times_s=art.times_s,
        swing_threshold=var_cfg["map_swing_threshold"],
        swing_window_s=var_cfg["swing_window_seconds"],
        min_episodes=var_cfg["min_episodes"],
        episode_window_s=var_cfg["episode_window_seconds"],
    )

    return EventResult(
        hypotension=hypo,
        hypertension=hyper,
        variability=variability,
    )
