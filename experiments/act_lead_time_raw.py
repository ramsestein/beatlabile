"""Lead time confirmatory analysis — re-extract windows at t-5, -15, -30 min.

For patients with confirmed hypotension events, re-processes raw .vital files
and extracts prediction windows ending at event_onset - lead_offset.

This converts the existing lead time exploration (linear extrapolation over
existing windows) to a TRUE confirmatory analysis: the GLMM features are
recomputed from raw signal at each lead time position.

Analysis:
  - Lead = 0  min: window ending AT event onset (reference = current analysis)
  - Lead = 5  min: window ending 5 min before event
  - Lead = 15 min: window ending 15 min before event
  - Lead = 30 min: window ending 30 min before event

Outputs
-------
  results/lead_time/lead_time_raw_auc.csv   — AUC per outcome × lead time
  results/lead_time/lead_time_raw_curves.{pdf,png}  — line plot

Run (takes ~15-30 min for all patients)
---
nohup .venv/bin/python3 experiments/act_lead_time_raw.py > lead_time_raw.log 2>&1 &
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beatlabile.config import CFG, DATA_CLINIC, RESULTS_DIR
from beatlabile.io.loader_clinic import load_vital_file, iter_clinic_files
from beatlabile.qc.pipeline import run_qc
from beatlabile.signal.peaks import detect_r_peaks, detect_art_peaks
from beatlabile.signal.sync import sync_ecg_art
from beatlabile.signal.metrics import compute_all_metrics, aggregate_window_features
from beatlabile.events.detector import detect_events
from beatlabile.windows.builder import build_windows, _features_in_window, _aggregate_prediction_window
from experiments.pipeline import get_feature_cols, EVENT_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR   = RESULTS_DIR / "lead_time"
ACT1_DIR  = RESULTS_DIR / "act1"
CACHE_DIR = RESULTS_DIR / "cache"
FIG_DIR   = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lead times to evaluate (minutes before event onset)
LEAD_OFFSETS_MIN = [0, 5, 15, 30]

# Event types for lead time analysis
TARGET_ETYPES = ["hypotension", "hypertension"]


def _find_vital_files_for_patient(patient_id: str, clinic_root: Path) -> list[Path]:
    """Find .vital files for a patient_id (= date directory name).
    
    Deduplicates by filename to avoid re-processing identical files that appear
    in multiple directory trees (e.g. backup/mirror subdirectories).
    """
    all_files = sorted(clinic_root.rglob(f"{patient_id}/*.vital"))
    seen: set[str] = set()
    unique: list[Path] = []
    for f in all_files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
    return unique


def _extract_features_at_lead(
    features_df: pd.DataFrame,
    event_onset_s: float,
    lead_offset_s: float,
    pre_event_s: float,
) -> pd.Series | None:
    """Extract window features ending at event_onset_s - lead_offset_s.
    
    Uses _aggregate_prediction_window to produce the same column names
    (brs_mean, brs_min, cv_pa_std, ...) that the GLMM was trained on.
    Window: [event_onset_s - lead_offset_s - pre_event_s, event_onset_s - lead_offset_s]
    """
    window_end = event_onset_s - lead_offset_s
    window_start = window_end - pre_event_s

    if window_start < 0:
        return None

    sub = _features_in_window(features_df, window_start, window_end)
    if sub is None or len(sub) < 3:
        return None

    agg = _aggregate_prediction_window(sub)
    agg["window_start_s"] = window_start
    agg["window_end_s"]   = window_end
    agg["event_onset_s"]  = event_onset_s
    return pd.Series(agg)


def run_lead_time_analysis() -> None:
    logger.info("=== Lead Time Confirmatory Analysis ===")

    # ------------------------------------------------------------------ #
    # 1. Identify patients with events from cached windows
    # ------------------------------------------------------------------ #
    windows_df = pd.read_parquet(CACHE_DIR / "clinic_windows.parquet")
    logger.info("Loaded cache: %d windows, %d unique patients",
                len(windows_df), windows_df["patient_id"].nunique())

    # Collect (patient_id, event_type, event_onset_s) for all events
    event_rows = windows_df[windows_df["label"] == 1][
        ["patient_id", "event_type", "event_onset_s"]
    ].drop_duplicates().copy()

    target_events = event_rows[event_rows["event_type"].isin(TARGET_ETYPES)]
    logger.info("Target events: %d (patients: %d)",
                len(target_events), target_events["patient_id"].nunique())

    target_patients = sorted(target_events["patient_id"].unique())

    # ------------------------------------------------------------------ #
    # 2. Load GLMM models (parsimonious — primary)
    # ------------------------------------------------------------------ #
    glmm_models = {}
    for etype in TARGET_ETYPES:
        pkl_path = ACT1_DIR / f"glmm_parsimonious_{etype}.pkl"
        try:
            with open(pkl_path, "rb") as fh:
                glmm_models[etype] = pickle.load(fh)
            logger.info("Loaded GLMM parsimonious: %s", etype)
        except Exception as e:
            logger.warning("Could not load GLMM for %s: %s", etype, e)

    if not glmm_models:
        logger.error("No GLMM models loaded. Aborting.")
        return

    # ------------------------------------------------------------------ #
    # 3. Process each target patient from raw vital files
    # ------------------------------------------------------------------ #
    cfg500 = copy.deepcopy(CFG)
    pre_event_s = float(CFG["windows"]["pre_event_minutes"]) * 60.0
    metrics_win_s = float(CFG["metrics"]["window_seconds"])

    # Results separated: events (label=1) and Clínic controls (label=0, same patients)
    # event_preds: {etype: {lead_min: [y_pred_float, ...]}}
    # ctrl_preds:  {etype: [y_pred_float, ...]}  — same for all lead offsets
    event_preds: dict[str, dict[int, list[float]]] = {
        etype: {lead: [] for lead in LEAD_OFFSETS_MIN}
        for etype in TARGET_ETYPES
    }
    ctrl_preds: dict[str, list[float]] = {etype: [] for etype in TARGET_ETYPES}

    for patient_id in target_patients:
        logger.info("Processing patient: %s", patient_id)

        # Find vital files for this patient
        vital_files = _find_vital_files_for_patient(patient_id, DATA_CLINIC)
        if not vital_files:
            logger.warning("  No vital files found for patient %s", patient_id)
            continue

        # Get events for this patient
        pt_events = target_events[target_events["patient_id"] == patient_id]

        for vital_path in vital_files:
            rec = load_vital_file(vital_path)
            if rec is None:
                continue

            # QC
            qc = run_qc(rec, CFG)
            if not qc.passes_qc:
                continue

            # Signal processing
            try:
                rr        = detect_r_peaks(rec["ecg"], rec["fs_ecg"], CFG, bad_mask=qc.bad_ecg)
                art_peaks = detect_art_peaks(rec["art"], rec["fs_art"], CFG, bad_mask=qc.bad_art)

                if len(rr.rr_intervals_ms) < 10 or len(art_peaks.sbp) < 10:
                    continue

                sync = sync_ecg_art(rr, art_peaks, CFG)
                metrics_df = compute_all_metrics(sync, art_peaks, CFG)
                if metrics_df.empty:
                    continue

                features_df = metrics_df.copy()
                features_df["window_start_s"] = features_df["time_s"]
                features_df["window_end_s"]   = features_df["time_s"] + metrics_win_s
                features_df = features_df.drop(columns=["time_s"])

            except Exception as e:
                logger.debug("  Signal processing failed for %s: %s", vital_path.name, e)
                continue

            # Duration of this file
            max_t = float(features_df["window_end_s"].max())

            # For each event of this patient, check if this file covers it
            for _, ev_row in pt_events.iterrows():
                etype       = ev_row["event_type"]
                event_onset = float(ev_row["event_onset_s"])

                # This vital file should contain the event (event_onset_s in [0, max_t])
                if event_onset > max_t + 300:  # allow 5min tolerance
                    continue
                if event_onset < 0:
                    continue

                if etype not in glmm_models:
                    continue

                glmm = glmm_models[etype]

                # For each lead offset, extract features and predict
                for lead_min in LEAD_OFFSETS_MIN:
                    lead_s = lead_min * 60.0
                    feat_row = _extract_features_at_lead(
                        features_df, event_onset, lead_s, pre_event_s
                    )
                    if feat_row is None:
                        continue

                    # Build X dataframe with correct feature cols
                    try:
                        feat_df = pd.DataFrame([feat_row])
                        p = glmm.predict_proba(feat_df)
                        if len(p) > 0:
                            event_preds[etype][lead_min].append(float(p[0]))
                    except Exception as e:
                        logger.info("  Predict failed for %s lead=%d: %s", etype, lead_min, e)

        # Collect Clínic training controls (label=0, same patients — training-apparent)
        for etype in TARGET_ETYPES:
            ctrl_windows = windows_df[
                (windows_df["patient_id"] == patient_id) &
                (windows_df["event_type"] == etype) &
                (windows_df["label"] == 0)
            ]
            if len(ctrl_windows) == 0:
                continue

            glmm = glmm_models.get(etype)
            if glmm is None:
                continue

            feat_cols_used = glmm.feature_cols
            available_cols = [c for c in feat_cols_used if c in ctrl_windows.columns]
            if len(available_cols) < len(feat_cols_used) // 2:
                continue

            try:
                p_ctrl = glmm.predict_proba(ctrl_windows)
                ctrl_preds[etype].extend([float(v) for v in p_ctrl])
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 3b. Save intermediate event predictions (avoids rerun for OOS check)
    # ------------------------------------------------------------------ #
    event_preds_path = OUT_DIR / "lead_time_raw_event_preds.pkl"
    import pickle as _pkl
    with open(event_preds_path, "wb") as fh:
        _pkl.dump(event_preds, fh)
    logger.info("Saved event predictions: %s", event_preds_path)

    # ------------------------------------------------------------------ #
    # 3c. Collect OOS controls from VitalDB (external cohort)
    # ------------------------------------------------------------------ #
    oos_ctrl_preds: dict[str, list[float]] = {etype: [] for etype in TARGET_ETYPES}
    try:
        vdb_path = CACHE_DIR / "vitaldb_windows.parquet"
        vdb_df = pd.read_parquet(vdb_path)
        for etype in TARGET_ETYPES:
            vdb_ctrl = vdb_df[(vdb_df["label"] == 0) & (vdb_df["event_type"] == etype)]
            glmm = glmm_models.get(etype)
            if glmm is None or len(vdb_ctrl) == 0:
                continue
            try:
                p_oos = glmm.predict_proba(vdb_ctrl)
                oos_ctrl_preds[etype].extend([float(v) for v in p_oos])
                logger.info("  VitalDB OOS controls for %s: n=%d", etype, len(vdb_ctrl))
            except Exception as e:
                logger.warning("  OOS controls predict failed for %s: %s", etype, e)
    except FileNotFoundError:
        logger.warning("VitalDB cache not found — OOS controls skipped")

    # ------------------------------------------------------------------ #
    # 4. Compute AUC per lead × outcome — two control regimes
    # ------------------------------------------------------------------ #
    def _build_auc_df(ep: dict, cp: dict, label: str) -> pd.DataFrame:
        rows_out = []
        for etype in TARGET_ETYPES:
            ctrl_p = np.array(cp[etype], dtype=float)
            n_ctrl = len(ctrl_p)
            for lead_min in LEAD_OFFSETS_MIN:
                ev_p = np.array(ep[etype][lead_min], dtype=float)
                n_ev = len(ev_p)
                if n_ev == 0 or n_ctrl == 0:
                    logger.warning("[%s] No data for %s lead=%d", label, etype, lead_min)
                    rows_out.append({"ctrl_set": label, "event_type": etype,
                                     "lead_min": lead_min, "n_events": n_ev,
                                     "n_controls": n_ctrl, "auc": np.nan})
                    continue
                ys    = np.concatenate([np.ones(n_ev), np.zeros(n_ctrl)])
                preds = np.concatenate([ev_p, ctrl_p])
                auc   = float(roc_auc_score(ys, preds)) if len(np.unique(ys)) > 1 else np.nan
                logger.info("  [%s] %s lead=%d min: AUC=%.3f (n_ev=%d, n_ctrl=%d)",
                            label, etype, lead_min, auc if not np.isnan(auc) else -1,
                            n_ev, n_ctrl)
                rows_out.append({"ctrl_set": label, "event_type": etype,
                                 "lead_min": lead_min, "n_events": n_ev,
                                 "n_controls": n_ctrl, "auc": auc})
        return pd.DataFrame(rows_out)

    logger.info("--- AUC with Clínic training controls (training-apparent) ---")
    df_clinic  = _build_auc_df(event_preds, ctrl_preds, "clinic_training")

    logger.info("--- AUC with VitalDB OOS controls ---")
    df_vitaldb = _build_auc_df(event_preds, oos_ctrl_preds, "vitaldb_oos")

    auc_combined = pd.concat([df_clinic, df_vitaldb], ignore_index=True)
    auc_combined_csv = OUT_DIR / "lead_time_raw_auc_combined.csv"
    auc_combined.to_csv(auc_combined_csv, index=False)
    logger.info("Saved: %s", auc_combined_csv)

    # Backward-compatible per-cohort CSVs
    auc_df = df_clinic.drop(columns=["ctrl_set"])
    auc_csv = OUT_DIR / "lead_time_raw_auc.csv"
    auc_df.to_csv(auc_csv, index=False)
    logger.info("Saved: %s", auc_csv)

    auc_oos_df = df_vitaldb.drop(columns=["ctrl_set"])
    auc_oos_csv = OUT_DIR / "lead_time_raw_oos_auc.csv"
    auc_oos_df.to_csv(auc_oos_csv, index=False)
    logger.info("Saved: %s", auc_oos_csv)

    # ------------------------------------------------------------------ #
    # 5. Plot — training-apparent vs OOS controls
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = {
        ("hypotension",  "clinic_training"): ("#2c7bb6", "-",  "Hipotensión (ctrl: Clínic, training)"),
        ("hypertension", "clinic_training"): ("#d7191c", "-",  "Hipertensión (ctrl: Clínic, training)"),
        ("hypotension",  "vitaldb_oos"):     ("#2c7bb6", "--", "Hipotensión (ctrl: VitalDB, OOS)"),
        ("hypertension", "vitaldb_oos"):     ("#d7191c", "--", "Hipertensión (ctrl: VitalDB, OOS)"),
    }
    for (etype, ctrl_set), (color, ls, label) in palette.items():
        sub = auc_combined[
            (auc_combined["event_type"] == etype) &
            (auc_combined["ctrl_set"] == ctrl_set) &
            auc_combined["auc"].notna()
        ]
        if sub.empty:
            continue
        ax.plot(sub["lead_min"], sub["auc"],
                marker="o", linestyle=ls, color=color, ms=6, lw=1.8, label=label)

    ax.axhline(0.5, color="gray", linestyle=":", lw=0.8, label="Azar (AUC=0.5)")
    ax.set_xlabel("Antelación al evento (minutos)", fontsize=10)
    ax.set_ylabel("AUC discriminación", fontsize=10)
    ax.set_title("Horizonte temporal confirmatorio\n"
                 "─ Clínic training controls  · · VitalDB OOS controls",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(-2, 35)
    ax.set_ylim(0.4, 1.0)
    ax.invert_xaxis()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for fmt in ("pdf", "png"):
        p = FIG_DIR / f"lead_time_raw_curves.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", p)
    plt.close(fig)

    logger.info("=== Lead time raw analysis DONE ===")


if __name__ == "__main__":
    run_lead_time_analysis()
