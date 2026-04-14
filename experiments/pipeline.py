"""Shared processing pipeline applied to every cohort.

process_record(record, cfg) -> dict | None
   Full pipeline: QC → peak detection → sync → metrics → events → features

process_cohort(iter_fn, cfg, cohort_name) -> tuple[pd.DataFrame, pd.DataFrame]
   Runs process_record over all records from iter_fn.
   Returns (windows_df, events_df) for all three event types combined.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Callable, Generator, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from beatlabile.config import CFG, RESULTS_DIR
from beatlabile.qc.pipeline import run_qc
from beatlabile.signal.peaks import detect_r_peaks, detect_art_peaks
from beatlabile.signal.sync import sync_ecg_art
from beatlabile.signal.metrics import compute_all_metrics, aggregate_window_features
from beatlabile.events.detector import detect_events
from beatlabile.windows.builder import build_windows

logger = logging.getLogger(__name__)

EVENT_TYPES = ["hypotension", "hypertension", "variability"]

# Default workers: leave 2 cores free for the OS and main process
_DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 2)


def _process_record_worker(args: tuple) -> dict | None:
    """Top-level picklable worker: unpack (record, cfg) and call process_record."""
    record, cfg = args
    try:
        return process_record(record, cfg)
    except Exception:  # keep pool alive even if one record crashes
        return None


def process_record(record: dict, cfg: dict) -> dict | None:
    """Run the full pipeline on a single record dict.

    Returns a dict with keys:
      windows_df : pd.DataFrame of labelled windows (all event types)
      events_df  : pd.DataFrame of detected events
      qc_result  : QCResult
      patient_id : str
    Returns None if the record fails QC.
    """
    # 1. QC
    qc = run_qc(record, cfg)
    if not qc.passes_qc:
        logger.debug(
            "Record %s failed QC: %s", record.get("filepath", "?"), qc.reject_reason
        )
        return None

    ecg = record["ecg"]
    art = record["art"]
    fs_ecg = record["fs_ecg"]
    fs_art = record["fs_art"]
    patient_id = record.get("patient", record.get("caseid", "unknown"))

    # 2. Peak detection
    rr = detect_r_peaks(ecg, fs_ecg, cfg, bad_mask=qc.bad_ecg)
    art_peaks = detect_art_peaks(art, fs_art, cfg, bad_mask=qc.bad_art)

    if len(rr.rr_intervals_ms) < 10 or len(art_peaks.sbp) < 10:
        logger.debug("Record %s: insufficient beats after peak detection.", patient_id)
        return None

    # 3. ECG–ART synchronisation
    sync = sync_ecg_art(rr, art_peaks, cfg)

    # 4. Beat-level metrics
    metrics_df = compute_all_metrics(sync, art_peaks, cfg)
    if metrics_df.empty:
        return None

    # 5. Prepare feature DataFrame for build_windows.
    # Use the 30-second sliding-window metrics directly — build_windows
    # will aggregate over the exact prediction window for each event,
    # which avoids the alignment problem that arises from pre-aggregating
    # into 30-min windows with a fixed step.
    metrics_win_s = float(cfg["metrics"]["window_seconds"])
    features_df = metrics_df.copy()
    features_df["window_start_s"] = features_df["time_s"]
    features_df["window_end_s"]   = features_df["time_s"] + metrics_win_s
    features_df = features_df.drop(columns=["time_s"])
    if features_df.empty:
        return None

    # 6. Event detection
    event_result = detect_events(art_peaks, cfg)

    # 7. Build labelled windows for each event type
    all_windows: list[pd.DataFrame] = []
    for etype in EVENT_TYPES:
        win_df = build_windows(
            features_df=features_df,
            event_result=event_result,
            art=art_peaks,
            r_times_s=rr.r_peaks / rr.fs,
            patient_id=patient_id,
            cfg=cfg,
            event_type=etype,
        )
        if not win_df.empty:
            all_windows.append(win_df)

    windows_df = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()
    events_df = event_result.to_dataframe()
    if not events_df.empty:
        events_df["patient_id"] = patient_id

    return {
        "windows_df": windows_df,
        "events_df": events_df,
        "qc_result": qc,
        "patient_id": patient_id,
    }


def process_cohort(
    iter_fn: Callable[[], Iterator[dict]],
    cfg: dict,
    cohort_name: str,
    cache_dir: Path | None = None,
    n_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process every record from *iter_fn* and aggregate results.

    Parameters
    ----------
    iter_fn     : Zero-argument callable that yields record dicts
    cfg         : Full config dict
    cohort_name : Label for logging and cache filenames
    cache_dir   : If provided and cache files exist, load from cache
    n_workers   : Parallel workers (default: cpu_count - 2)

    Returns
    -------
    (windows_df, events_df) — combined across all patients
    """
    if cache_dir is not None:
        win_cache = cache_dir / f"{cohort_name}_windows.parquet"
        ev_cache = cache_dir / f"{cohort_name}_events.parquet"
        if win_cache.exists() and ev_cache.exists():
            logger.info("Loading %s from cache.", cohort_name)
            return pd.read_parquet(win_cache), pd.read_parquet(ev_cache)

    if n_workers is None:
        n_workers = _DEFAULT_WORKERS

    windows_all: list[pd.DataFrame] = []
    events_all: list[pd.DataFrame] = []
    n_total = 0
    n_passed = 0

    # Flush partial results to disk every N records so a crash mid-run
    # does not discard all work.  The checkpoint is NOT used as a resume
    # point (full cache above handles that), but it is a safety net.
    _CHECKPOINT_EVERY = 500
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    win_cache_tmp = (cache_dir / f"{cohort_name}_windows_tmp.parquet") if cache_dir else None
    ev_cache_tmp  = (cache_dir / f"{cohort_name}_events_tmp.parquet")  if cache_dir else None

    def _flush_checkpoint():
        if cache_dir is None or not windows_all:
            return
        _w = pd.concat(windows_all, ignore_index=True)
        _e = pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame()
        _w.to_parquet(win_cache_tmp, index=False)
        if not _e.empty:
            _e.to_parquet(ev_cache_tmp, index=False)

    ctx = mp.get_context("fork")  # fastest on Linux; avoids re-import overhead
    with ctx.Pool(processes=n_workers) as pool:
        tasks = ((r, cfg) for r in iter_fn())
        for result in tqdm(
            pool.imap_unordered(_process_record_worker, tasks, chunksize=1),
            desc=f"Processing {cohort_name} [{n_workers} workers]",
        ):
            n_total += 1
            if result is None:
                continue
            n_passed += 1
            if not result["windows_df"].empty:
                windows_all.append(result["windows_df"])
            if not result["events_df"].empty:
                events_all.append(result["events_df"])
            if n_total % _CHECKPOINT_EVERY == 0:
                _flush_checkpoint()

    logger.info(
        "%s: %d/%d records passed QC.", cohort_name, n_passed, n_total
    )

    windows_df = pd.concat(windows_all, ignore_index=True) if windows_all else pd.DataFrame()
    events_df = pd.concat(events_all, ignore_index=True) if events_all else pd.DataFrame()

    if cache_dir is not None:
        if not windows_df.empty:
            windows_df.to_parquet(win_cache, index=False)
        if not events_df.empty:
            events_df.to_parquet(ev_cache, index=False)
        # Remove temp checkpoint files
        for tmp in [win_cache_tmp, ev_cache_tmp]:
            if tmp and tmp.exists():
                tmp.unlink(missing_ok=True)

    return windows_df, events_df


def get_feature_cols(windows_df: pd.DataFrame) -> list[str]:
    """Return the list of numeric feature columns (exclude metadata columns)."""
    non_feature = {
        "patient_id", "event_type", "label",
        "window_start_s", "window_end_s", "event_onset_s",
    }
    return [
        c for c in windows_df.select_dtypes(include="number").columns
        if c not in non_feature
    ]
