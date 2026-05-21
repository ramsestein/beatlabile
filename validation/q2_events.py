"""
Q2 Events  (PASO 1 + PASO 2)
─────────────────────────────
PASO 1 – build_vasopressor_events():
    Load medicacion_bolus annotations, apply 5 exclusion criteria, return
    cleaned event list as DataFrame → saved to VASOPRESSOR_EVENTS.

PASO 2 – build_control_windows():
    For each clean event, scan the patient's recording for quiescent 5-min
    windows that satisfy the same isolation criteria, then sample one window
    randomly (SEED=42).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_config import (
    ANNOTATIONS, FEATURES_LONG, DRUG_TIMESERIES, EVENT_WINDOWS,
    VASOPRESSOR_EVENTS,
    PRE_START_S, PRE_END_S, POST_START_S, POST_END_S, WINDOW_S,
    BOLUS_ISOLATION_S, INFUSION_BUFFER_S, STIMULUS_BUFFER_S,
    DELTA_DRUG_THRESH, MIN_PPG_VALID, MIN_VALID_FRAC,
    MIN_PRE_WINDOWS, PRIMARY, SEED,
)

VASOPRESSOR_DRUGS = ["efedrina", "fenilefrina"]


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_all():
    """Load all data files, normalise patient_id to str, return dict."""
    ann  = pd.read_csv(ANNOTATIONS)
    ann["patient_id"] = ann["patient_id"].astype(str)

    feat = pd.read_parquet(FEATURES_LONG)
    feat["patient_id"] = feat["patient_id"].astype(str)

    drug = pd.read_parquet(DRUG_TIMESERIES)
    drug["patient_id"] = drug["patient_id"].astype(str)

    ev = pd.read_csv(EVENT_WINDOWS)
    ev["patient_id"] = ev["patient_id"].astype(str)

    return ann, feat, drug, ev


def get_ag_windows(ann: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-patient AG window (t_ag_start, t_ag_end).
    For patients with multiple AG start annotations, take the minimum.
    For patients with no fin_AG, t_ag_end = NaN (handled later with
    max feature time).
    """
    fase = ann[ann["category"] == "fase"].copy()

    starts = (
        fase[fase["subcategory"] == "AG"]
        .groupby("patient_id")["t_seconds"]
        .min()
        .rename("t_ag_start")
    )
    ends = (
        fase[fase["subcategory"] == "fin_AG"]
        .groupby("patient_id")["t_seconds"]
        .min()
        .rename("t_ag_end")
    )

    ag = pd.concat([starts, ends], axis=1).reset_index()
    return ag


def get_group_map(ev: pd.DataFrame) -> pd.DataFrame:
    return (
        ev[["patient_id", "group"]]
        .drop_duplicates()
        .set_index("patient_id")["group"]
    )


def build_infusion_change_table(ann: pd.DataFrame) -> pd.DataFrame:
    """
    From infusion_change annotations, compute |Δ| in propofol and remi targets.
    Forward-fill within patient to handle remi_only / propo_only rows.
    Returns rows where |Δpropo| > DELTA_DRUG_THRESH OR |Δremi| > DELTA_DRUG_THRESH.
    """
    inf = ann[ann["category"] == "infusion_change"].copy()
    inf = inf.sort_values(["patient_id", "t_seconds"]).reset_index(drop=True)

    # Fill NaN targets via forward-fill within patient, then fill remaining with 0
    for col in ["propo_target", "remi_target"]:
        inf[col] = pd.to_numeric(inf[col], errors="coerce")
        inf[col] = inf.groupby("patient_id")[col].ffill().fillna(0)

    inf["prev_propo"] = inf.groupby("patient_id")["propo_target"].shift(1).fillna(0)
    inf["prev_remi"]  = inf.groupby("patient_id")["remi_target"].shift(1).fillna(0)
    inf["d_propo"] = (inf["propo_target"] - inf["prev_propo"]).abs()
    inf["d_remi"]  = (inf["remi_target"]  - inf["prev_remi"]).abs()

    large = inf[(inf["d_propo"] > DELTA_DRUG_THRESH) | (inf["d_remi"] > DELTA_DRUG_THRESH)]
    return large[["patient_id", "t_seconds", "d_propo", "d_remi"]].copy()


def get_cumulative_dose(drug: pd.DataFrame, patient_id: str, t: float) -> dict:
    """Return cumulative efedrina (mg) and fenilefrina (µg) up to time t."""
    pat = drug[drug["patient_id"] == patient_id]
    row = pat[pat["t_s"] <= t]
    if row.empty:
        return {"cum_efedrina_mg": 0.0, "cum_fenilefrina_mcg": 0.0}
    row = row.iloc[-1]
    return {
        "cum_efedrina_mg":      float(row.get("cumulative_efedrina_mg",   0) or 0),
        "cum_fenilefrina_mcg":  float(row.get("cumulative_fenilefrina_mcg", 0) or 0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Exclusion criterion checkers
# ═══════════════════════════════════════════════════════════════════════════════

def _check_other_bolus(t: float, pid: str, ann: pd.DataFrame) -> bool:
    """True → no other vasopresor in ±10 min. (passes criterion)"""
    other = ann[
        (ann["patient_id"] == pid)
        & (ann["category"] == "medicacion_bolus")
        & (ann["subcategory"].isin(VASOPRESSOR_DRUGS))
        & (ann["t_seconds"] != t)
        & (ann["t_seconds"].between(t - BOLUS_ISOLATION_S, t + BOLUS_ISOLATION_S))
    ]
    return len(other) == 0


def _check_infusion(t: float, pid: str, inf_changes: pd.DataFrame) -> bool:
    """True → no large infusion change within ±5 min."""
    near = inf_changes[
        (inf_changes["patient_id"] == pid)
        & (inf_changes["t_seconds"].between(t - INFUSION_BUFFER_S, t + INFUSION_BUFFER_S))
    ]
    return len(near) == 0


def _check_stimulus(t: float, pid: str, ann: pd.DataFrame) -> bool:
    """True → no pain stimulus in last STIMULUS_BUFFER_S before bolus."""
    stim = ann[
        (ann["patient_id"] == pid)
        & (ann["category"] == "estimulo")
        & (ann["t_seconds"].between(t - STIMULUS_BUFFER_S, t - 1))
    ]
    return len(stim) == 0


def _check_feature_windows(
    t: float, pid: str, feat: pd.DataFrame, window_type: str = "pre"
) -> tuple[bool, str]:
    """
    True → ≥ MIN_PRE_WINDOWS windows in the requested time range, AND each
    primary feature has ≥ MIN_VALID_FRAC non-NaN values.
    window_type: "pre" or "post".
    """
    pat = feat[feat["patient_id"] == pid]

    if window_type == "pre":
        t_lo = t + PRE_START_S
        t_hi = t + PRE_END_S - WINDOW_S   # last window starts ≤ t-30
    else:
        t_lo = t + POST_START_S
        t_hi = t + POST_END_S - WINDOW_S

    w = pat[(pat["t_window_start_s"] >= t_lo) & (pat["t_window_start_s"] <= t_hi)]

    if len(w) < MIN_PRE_WINDOWS:
        return False, f"n_windows_{window_type}={len(w)}"

    for fname, _, _ in PRIMARY:
        if fname not in w.columns:
            continue
        vf = w[fname].notna().mean()
        if vf < MIN_VALID_FRAC:
            return False, f"{fname}_valid_{window_type}={vf:.2f}"

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 1 – Vasopressor event list
# ═══════════════════════════════════════════════════════════════════════════════

def build_vasopressor_events(
    ann: pd.DataFrame,
    feat: pd.DataFrame,
    drug: pd.DataFrame,
    ev: pd.DataFrame,
) -> pd.DataFrame:
    """
    PASO 1. Apply 5 exclusion criteria to all medicacion_bolus annotations of
    vasopresor drugs. Return a DataFrame of clean events.
    """
    ag_windows  = get_ag_windows(ann)
    inf_changes = build_infusion_change_table(ann)
    group_map   = get_group_map(ev)

    # Max feature time per patient (for patients without fin_AG)
    max_feat_t = feat.groupby("patient_id")["t_window_start_s"].max() + WINDOW_S

    # Raw vasopresor events
    raw = ann[
        (ann["category"] == "medicacion_bolus")
        & (ann["subcategory"].isin(VASOPRESSOR_DRUGS))
    ].copy()

    print(f"\n[PASO 1] Raw vasopresor events: {len(raw)}")

    records = []
    for _, row in raw.iterrows():
        pid    = str(row["patient_id"])
        t      = float(row["t_seconds"])
        drug_  = row["subcategory"]

        # Lookup AG window for this patient
        ag_row = ag_windows[ag_windows["patient_id"] == pid]
        if ag_row.empty:
            status = "excluded"; reason = "no_AG_annotation"
            records.append(_make_row(pid, t, drug_, None, None, None, status, reason, group_map, drug))
            continue

        t_ag_start = float(ag_row["t_ag_start"].iloc[0])
        t_ag_end_raw = ag_row["t_ag_end"].iloc[0]
        t_ag_end = (
            float(max_feat_t.get(pid, t_ag_start + 10000))
            if pd.isna(t_ag_end_raw)
            else float(t_ag_end_raw)
        )

        # C1: within AG window
        if t <= t_ag_start or t >= t_ag_end - POST_END_S:
            status = "excluded"; reason = f"outside_AG_window (t={t:.0f}, AG=[{t_ag_start:.0f},{t_ag_end:.0f}])"
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None, status, reason, group_map, drug))
            continue

        # C2: no other vasopresor in ±10 min
        if not _check_other_bolus(t, pid, ann):
            status = "excluded"; reason = "other_bolus_pm10min"
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None, status, reason, group_map, drug))
            continue

        # C3: no large infusion change in ±5 min
        if not _check_infusion(t, pid, inf_changes):
            status = "excluded"; reason = "infusion_change_pm5min"
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None, status, reason, group_map, drug))
            continue

        # C4: no pain stimulus in [-5, 0] min before bolus
        if not _check_stimulus(t, pid, ann):
            status = "excluded"; reason = "pain_stimulus_in_pre5min"
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None, status, reason, group_map, drug))
            continue

        # C5: valid feature windows (pre + post)
        ok_pre,  msg_pre  = _check_feature_windows(t, pid, feat, "pre")
        ok_post, msg_post = _check_feature_windows(t, pid, feat, "post")
        if not ok_pre:
            status = "excluded"; reason = f"invalid_pre_features ({msg_pre})"
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None, status, reason, group_map, drug))
            continue
        if not ok_post:
            status = "excluded"; reason = f"invalid_post_features ({msg_post})"
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None, status, reason, group_map, drug))
            continue

        status = "clean"; reason = ""
        records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end,
                                 max_feat_t.get(pid), status, reason, group_map, drug))

    df = pd.DataFrame(records)
    df = df.sort_values(["patient_id", "t_bolus"]).reset_index(drop=True)
    df.insert(0, "event_id", range(len(df)))

    n_clean = (df["status"] == "clean").sum()
    print(f"[PASO 1] Clean events after exclusions: {n_clean} / {len(df)}")

    excl = df[df["status"] == "excluded"]
    if len(excl) > 0:
        print("[PASO 1] Excluded events:")
        for _, r in excl.iterrows():
            print(f"  pid={r['patient_id']}  t={r['t_bolus']:.0f}s  drug={r['drug']}  reason={r['exclusion_reason']}")

    return df


def _make_row(pid, t, drug_, t_ag_start, t_ag_end, max_t, status, reason, group_map, drug_df):
    t_from_ag = float(t - t_ag_start) if t_ag_start is not None else np.nan
    doses = get_cumulative_dose(drug_df, pid, t)
    grp   = group_map.get(pid, "unknown")
    return {
        "patient_id":        pid,
        "t_bolus":           t,
        "drug":              drug_,
        "group":             grp,
        "t_ag_start":        t_ag_start,
        "t_ag_end":          t_ag_end,
        "t_from_ag_start_s": t_from_ag,
        "cum_efedrina_mg":   doses["cum_efedrina_mg"],
        "cum_fenilefrina_mcg": doses["cum_fenilefrina_mcg"],
        "status":            status,
        "exclusion_reason":  reason,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2 – Quiescent control windows
# ═══════════════════════════════════════════════════════════════════════════════

def build_control_windows(
    events: pd.DataFrame,
    ann: pd.DataFrame,
    feat: pd.DataFrame,
    drug: pd.DataFrame,
) -> pd.DataFrame:
    """
    PASO 2. For each clean event, scan the patient's AG window for quiescent
    5-min control windows meeting isolation criteria, then randomly sample one.

    Returns a DataFrame with columns: event_id, patient_id, t_bolus,
    t_control_start, n_candidates, excluded.
    """
    np.random.seed(SEED)

    ag_windows  = get_ag_windows(ann)
    inf_changes = build_infusion_change_table(ann)
    max_feat_t  = feat.groupby("patient_id")["t_window_start_s"].max() + WINDOW_S

    clean = events[events["status"] == "clean"].copy()
    print(f"\n[PASO 2] Finding control windows for {len(clean)} clean events …")

    results = []
    for _, ev in clean.iterrows():
        pid     = str(ev["patient_id"])
        t_bolus = float(ev["t_bolus"])
        t_ag_start = float(ev["t_ag_start"]) if not pd.isna(ev["t_ag_start"]) else 0.0
        t_ag_end_raw = ev["t_ag_end"]
        t_ag_end = (
            float(max_feat_t.get(pid, t_ag_start + 10000))
            if pd.isna(t_ag_end_raw)
            else float(t_ag_end_raw)
        )

        # Candidate scan range: need full 5-min window within AG
        t_scan_start = t_ag_start + POST_END_S   # at least 5 min after AG start
        t_scan_end   = t_ag_end   - POST_END_S   # at least 5 min before AG end

        if t_scan_end <= t_scan_start:
            print(f"  [WARN] {pid}: AG window too short for control")
            results.append(_ctrl_row(ev, np.nan, 0, "AG_window_too_short"))
            continue

        # Scan at 30 s steps
        candidates_t = np.arange(t_scan_start, t_scan_end, 30.0)
        valid_cands  = []

        pat_ann  = ann[ann["patient_id"] == pid]
        pat_inf  = inf_changes[inf_changes["patient_id"] == pid]
        pat_feat = feat[feat["patient_id"] == pid]

        for t_c in candidates_t:
            # Exclude windows too close to the bolus
            if abs(t_c + POST_END_S / 2 - t_bolus) < BOLUS_ISOLATION_S:
                continue

            # Check zone: [t_c - INFUSION_BUFFER_S, t_c + POST_END_S + INFUSION_BUFFER_S]
            lo = t_c - INFUSION_BUFFER_S
            hi = t_c + POST_END_S + INFUSION_BUFFER_S

            # C2 – no other vasopresor bolus in zone
            if len(pat_ann[
                (pat_ann["category"] == "medicacion_bolus")
                & (pat_ann["subcategory"].isin(VASOPRESSOR_DRUGS))
                & (pat_ann["t_seconds"].between(lo, hi))
            ]) > 0:
                continue

            # C3 – no large infusion change in zone
            if len(pat_inf[pat_inf["t_seconds"].between(lo, hi)]) > 0:
                continue

            # C4 – no pain stimulus in zone
            if len(pat_ann[
                (pat_ann["category"] == "estimulo")
                & (pat_ann["t_seconds"].between(lo, hi))
            ]) > 0:
                continue

            # C5 – valid features in control window [t_c, t_c+270]
            w = pat_feat[
                (pat_feat["t_window_start_s"] >= t_c)
                & (pat_feat["t_window_start_s"] <= t_c + POST_END_S - WINDOW_S)
            ]
            if len(w) < MIN_PRE_WINDOWS:
                continue

            if "ppg_valid_pct" in w.columns:
                if w["ppg_valid_pct"].mean() < MIN_PPG_VALID:
                    continue

            ok_feat = True
            for fname, _, _ in PRIMARY:
                if fname in w.columns and w[fname].notna().mean() < MIN_VALID_FRAC:
                    ok_feat = False
                    break
            if not ok_feat:
                continue

            valid_cands.append(t_c)

        if len(valid_cands) == 0:
            print(f"  [WARN] {pid} t={t_bolus:.0f}: no valid control window found")
            results.append(_ctrl_row(ev, np.nan, 0, "no_valid_control_window"))
        else:
            chosen = float(np.random.choice(valid_cands))
            print(f"  [OK]   {pid} t={t_bolus:.0f}: {len(valid_cands)} candidates → t_ctrl={chosen:.0f}")
            results.append(_ctrl_row(ev, chosen, len(valid_cands), None))

    return pd.DataFrame(results)


def _ctrl_row(ev, t_ctrl, n_cand, exc_reason):
    return {
        "event_id":       int(ev["event_id"]),
        "patient_id":     str(ev["patient_id"]),
        "t_bolus":        float(ev["t_bolus"]),
        "drug":           str(ev["drug"]),
        "group":          str(ev["group"]),
        "t_control_start": t_ctrl,
        "n_candidates":   n_cand,
        "ctrl_excluded":  exc_reason,
    }
