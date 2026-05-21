"""
Q2 v2 Events  (PASO 2 + PASO 3)
──────────────────────────────────
PASO 2 – build_vasopressor_events_v2():
    Load medicacion_bolus annotations.  Apply exclusion criteria v2:
    • No other vasopresor bolus ±5 min
    • No pain stimulus in [-2, 0] min
    • Within AG window
    • Valid features in pre/post windows
    • PPG valid ≥70% in pre-window
    KEY CHANGE: infusion-change criterion REMOVED.
    Covariables delta_propofol_pre, delta_remi_pre ADDED.

PASO 3 – build_control_windows_v2():
    Same as v1 but:
    • Control window duration = 3 min (CONTROL_DURATION_S=180)
    • Infusion isolation KEPT for controls (quiescent baseline required)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2v2_config import (
    ANNOTATIONS, FEATURES_BRS_SEQ, DRUG_TIMESERIES, EVENT_WINDOWS,
    VASOPRESSOR_EVENTS, Q2V2_RES,
    PRE_START_S, PRE_END_S, POST_START_S, POST_END_S, WINDOW_S,
    BOLUS_ISOLATION_S, STIMULUS_BUFFER_S,
    DELTA_DRUG_THRESH, MIN_PPG_VALID_PRE, MIN_VALID_FRAC,
    MIN_PRE_WINDOWS, PRIMARY, SEED,
    CONTROL_DURATION_S, CONTROL_INFUSION_BUFFER_S, CONTROL_BOLUS_BUFFER_S,
)

VASOPRESSOR_DRUGS = ["efedrina", "fenilefrina"]

# ── FIX 1: validity check uses ONLY these 3 PTT-based features, NOT brs_seq ──
# brs_seq has NaN in 30s windows without detectable sequences (normal behaviour);
# including it in the validity criterion incorrectly excludes valid events.
VALIDITY_FEATURES = ["ptt_cv", "ptt_std", "pai_mean"]


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_all():
    """Load data files, normalise patient_id to str.  Returns (ann, feat, drug, ev)."""
    ann  = pd.read_csv(ANNOTATIONS)
    ann["patient_id"] = ann["patient_id"].astype(str)

    feat = pd.read_parquet(FEATURES_BRS_SEQ)
    feat["patient_id"] = feat["patient_id"].astype(str)

    drug = pd.read_parquet(DRUG_TIMESERIES)
    drug["patient_id"] = drug["patient_id"].astype(str)

    ev = pd.read_csv(EVENT_WINDOWS)
    ev["patient_id"] = ev["patient_id"].astype(str)

    return ann, feat, drug, ev


def get_ag_windows(ann: pd.DataFrame) -> pd.DataFrame:
    """Return per-patient AG window (t_ag_start, t_ag_end)."""
    fase = ann[ann["category"] == "fase"].copy()
    starts = (
        fase[fase["subcategory"] == "AG"]
        .groupby("patient_id")["t_seconds"].min().rename("t_ag_start")
    )
    ends = (
        fase[fase["subcategory"] == "fin_AG"]
        .groupby("patient_id")["t_seconds"].min().rename("t_ag_end")
    )
    return pd.concat([starts, ends], axis=1).reset_index()


def get_group_map(ev: pd.DataFrame) -> pd.Series:
    return (
        ev[["patient_id", "group"]].drop_duplicates()
        .set_index("patient_id")["group"]
    )


def build_infusion_change_table(ann: pd.DataFrame) -> pd.DataFrame:
    """Compute |Δ| in propofol and remi targets. Returns rows where |Δ| > threshold."""
    inf = ann[ann["category"] == "infusion_change"].copy()
    inf = inf.sort_values(["patient_id", "t_seconds"]).reset_index(drop=True)
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
        "cum_efedrina_mg":     float(row.get("cumulative_efedrina_mg",   0) or 0),
        "cum_fenilefrina_mcg": float(row.get("cumulative_fenilefrina_mcg", 0) or 0),
    }


def get_max_infusion_delta(
    inf_changes: pd.DataFrame, patient_id: str,
    t_lo: float, t_hi: float,
) -> tuple[float, float]:
    """Return max |Δpropo| and max |Δremi| in time range [t_lo, t_hi]."""
    near = inf_changes[
        (inf_changes["patient_id"] == patient_id)
        & (inf_changes["t_seconds"].between(t_lo, t_hi))
    ]
    if near.empty:
        return 0.0, 0.0
    return float(near["d_propo"].max()), float(near["d_remi"].max())


# ═══════════════════════════════════════════════════════════════════════════════
# Exclusion criterion checkers
# ═══════════════════════════════════════════════════════════════════════════════

def _check_other_bolus(t: float, pid: str, ann: pd.DataFrame) -> bool:
    """True → no other vasopresor in ±BOLUS_ISOLATION_S."""
    other = ann[
        (ann["patient_id"] == pid)
        & (ann["category"] == "medicacion_bolus")
        & (ann["subcategory"].isin(VASOPRESSOR_DRUGS))
        & (ann["t_seconds"] != t)
        & (ann["t_seconds"].between(t - BOLUS_ISOLATION_S, t + BOLUS_ISOLATION_S))
    ]
    return len(other) == 0


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
    True → ≥MIN_PRE_WINDOWS windows in the requested time range,
    AND each VALIDITY_FEATURE has ≥MIN_VALID_FRAC non-NaN values,
    AND PPG valid ≥MIN_PPG_VALID_PRE (pre-window only).

    FIX 1: brs_seq is NOT included in the validity check — it legitimately
    has NaN in 30s windows without detectable sequences.  Its NaN rate is
    checked separately for logging purposes only; it never drives exclusion.
    """
    pat = feat[feat["patient_id"] == pid]

    if window_type == "pre":
        t_lo = t + PRE_START_S
        t_hi = t + PRE_END_S - WINDOW_S
    else:
        t_lo = t + POST_START_S
        t_hi = t + POST_END_S - WINDOW_S

    w = pat[(pat["t_window_start_s"] >= t_lo) & (pat["t_window_start_s"] <= t_hi)]

    if len(w) < MIN_PRE_WINDOWS:
        return False, f"n_windows_{window_type}={len(w)}"

    # PPG validity check (pre window only)
    if window_type == "pre" and "ppg_valid_pct" in w.columns:
        ppg_ok = float(w["ppg_valid_pct"].mean())
        if ppg_ok < MIN_PPG_VALID_PRE:
            return False, f"ppg_valid_{window_type}={ppg_ok:.2f}"

    # Check VALIDITY_FEATURES only — brs_seq excluded intentionally
    for fname in VALIDITY_FEATURES:
        if fname not in w.columns:
            continue
        vf = w[fname].notna().mean()
        if vf < MIN_VALID_FRAC:
            return False, f"{fname}_valid_{window_type}={vf:.2f}"

    return True, ""


def _pre_window_validity_fracs(
    t: float, pid: str, feat: pd.DataFrame
) -> dict:
    """Return validity fractions for each feature in the pre-window (for audit log)."""
    pat = feat[feat["patient_id"] == pid]
    t_lo = t + PRE_START_S
    t_hi = t + PRE_END_S - WINDOW_S
    w = pat[(pat["t_window_start_s"] >= t_lo) & (pat["t_window_start_s"] <= t_hi)]
    if len(w) == 0:
        return {}
    result = {}
    for fname in VALIDITY_FEATURES + ["brs_seq", "ppg_valid_pct"]:
        if fname in w.columns:
            result[fname] = float(w[fname].notna().mean())
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2 – Vasopressor event list (v2: no infusion exclusion)
# ═══════════════════════════════════════════════════════════════════════════════

def build_vasopressor_events_v2(
    ann: pd.DataFrame,
    feat: pd.DataFrame,
    drug: pd.DataFrame,
    ev: pd.DataFrame,
) -> pd.DataFrame:
    """
    PASO 2.  Apply v2 exclusion criteria (infusion-change criterion REMOVED).
    Adds covariables: delta_propofol_pre, delta_remi_pre.
    Returns DataFrame of clean + excluded events.
    """
    ag_windows  = get_ag_windows(ann)
    inf_changes = build_infusion_change_table(ann)
    group_map   = get_group_map(ev)
    max_feat_t  = feat.groupby("patient_id")["t_window_start_s"].max() + WINDOW_S

    raw = ann[
        (ann["category"] == "medicacion_bolus")
        & (ann["subcategory"].isin(VASOPRESSOR_DRUGS))
    ].copy()

    print(f"\n[PASO 2] Raw vasopresor events: {len(raw)}")

    records = []
    for _, row in raw.iterrows():
        pid   = str(row["patient_id"])
        t     = float(row["t_seconds"])
        drug_ = row["subcategory"]

        ag_row = ag_windows[ag_windows["patient_id"] == pid]
        if ag_row.empty:
            records.append(_make_row(pid, t, drug_, None, None, None,
                                     "excluded", "no_AG_annotation",
                                     group_map, drug, inf_changes))
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
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None,
                                     "excluded",
                                     f"outside_AG_window (t={t:.0f}, AG=[{t_ag_start:.0f},{t_ag_end:.0f}])",
                                     group_map, drug, inf_changes))
            continue

        # C2: no other vasopresor ±5 min
        if not _check_other_bolus(t, pid, ann):
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None,
                                     "excluded", "other_bolus_pm5min",
                                     group_map, drug, inf_changes))
            continue

        # C3: no pain stimulus in [-2, 0] min
        if not _check_stimulus(t, pid, ann):
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None,
                                     "excluded", "pain_stimulus_in_pre2min",
                                     group_map, drug, inf_changes))
            continue

        # C4: valid feature windows (pre + post) — validity based on VALIDITY_FEATURES only
        ok_pre,  msg_pre  = _check_feature_windows(t, pid, feat, "pre")
        ok_post, msg_post = _check_feature_windows(t, pid, feat, "post")
        # Log brs_seq validity separately (informational, not exclusion criterion)
        vfracs = _pre_window_validity_fracs(t, pid, feat)
        brs_seq_valid_pre = vfracs.get("brs_seq", np.nan)
        if not ok_pre:
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None,
                                     "excluded", f"invalid_pre_features ({msg_pre})",
                                     group_map, drug, inf_changes,
                                     brs_seq_valid_pre=brs_seq_valid_pre))
            continue
        if not ok_post:
            records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end, None,
                                     "excluded", f"invalid_post_features ({msg_post})",
                                     group_map, drug, inf_changes,
                                     brs_seq_valid_pre=brs_seq_valid_pre))
            continue

        # EVENT IS CLEAN — note: NO infusion criterion
        records.append(_make_row(pid, t, drug_, t_ag_start, t_ag_end,
                                  max_feat_t.get(pid), "clean", "",
                                  group_map, drug, inf_changes,
                                  brs_seq_valid_pre=brs_seq_valid_pre))

    df = pd.DataFrame(records)
    df = df.sort_values(["patient_id", "t_bolus"]).reset_index(drop=True)
    df.insert(0, "event_id", range(len(df)))

    n_raw   = len(df)
    n_clean = int((df["status"] == "clean").sum())
    print(f"[PASO 2] Clean events after v2 exclusions: {n_clean} / {n_raw}")

    excl = df[df["status"] == "excluded"]
    if len(excl) > 0:
        print("[PASO 2] Excluded events:")
        for _, r in excl.iterrows():
            brs_note = (f"  [brs_seq_pre={r.get('brs_seq_valid_pre', float('nan')):.2f}]"
                        if not pd.isna(r.get('brs_seq_valid_pre', float('nan'))) else "")
            print(f"  pid={r['patient_id']}  t={r['t_bolus']:.0f}s"
                  f"  drug={r['drug']}  reason={r['exclusion_reason']}{brs_note}")

    # ── Filter-level summary ──
    reasons = df["exclusion_reason"].fillna("")
    n1 = int((df["status"] == "clean").sum() + (~reasons.str.contains("other_bolus|outside_AG")).sum() - len(df))
    n_after_bolus = n_raw - int(reasons.str.startswith("other_bolus").sum())
    n_after_pain  = n_after_bolus - int(reasons.str.startswith("pain_stimulus").sum())
    n_after_ag    = n_after_pain  - int(reasons.str.startswith("outside_AG").sum())
    n_after_ag2   = n_after_ag   - int(reasons.str.startswith("no_AG").sum())
    n_after_feat  = n_after_ag2  - int(reasons.str.startswith("invalid_pre").sum())
    n_after_post  = n_after_feat - int(reasons.str.startswith("invalid_post").sum())
    print(f"\n[PASO 2] Filter summary:")
    print(f"  Eventos crudos            : {n_raw}")
    print(f"  Tras filtro AG/anotacion  : {n_after_ag2}")
    print(f"  Tras filtro otro bolus    : {n_after_bolus}")
    print(f"  Tras filtro estimulo dolor: {n_after_pain}")
    print(f"  Tras filtro features pre  : {n_after_feat}")
    print(f"  Tras filtro features post : {n_after_post}  <- limpios = {n_clean}")

    print(f"\n[PASO 2] NOTE: infusion-change criterion REMOVED (v2 change).")
    print(f"[PASO 2] Infusion \u0394s are now covariables (delta_propofol_pre, delta_remi_pre).")

    # Save audit CSV (Fix 2)
    _save_audit_csv(df, ann, feat)

    return df


def _save_audit_csv(df: pd.DataFrame, ann: pd.DataFrame, feat: pd.DataFrame):
    """Generate event_filtering_audit.csv with per-event filter trace (Fix 2)."""
    audit_path = Q2V2_RES / "event_filtering_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        pid = str(r["patient_id"])
        t   = float(r["t_bolus"])
        reason = str(r.get("exclusion_reason", ""))
        status = str(r["status"])

        # Reconstruct per-filter pass flags from the exclusion reason
        passed_bolus    = not reason.startswith("other_bolus")
        passed_pain     = not reason.startswith("pain_stimulus")
        passed_ag       = not reason.startswith("outside_AG") and not reason.startswith("no_AG")
        passed_feat_pre = not reason.startswith("invalid_pre")
        passed_ppg      = not reason.startswith("invalid_pre (ppg") and passed_feat_pre
        # control window info
        has_ctrl = np.nan  # filled after PASO 3; placeholder here

        if status == "clean":
            final_status = "clean"
        else:
            final_status = f"excluded_{reason[:40]}"

        rows.append({
            "patient_id":                      pid,
            "drug":                            r["drug"],
            "t_bolus_s":                       t,
            "passes_no_other_vasopressor_pm5min": passed_bolus,
            "passes_no_pain_stim_pre2min":     passed_pain,
            "passes_within_AG_window":         passed_ag,
            "passes_valid_pre_features":       passed_feat_pre,
            "passes_ppg_valid_70pct_pre":      passed_ppg,
            "brs_seq_valid_pct_pre":           r.get("brs_seq_valid_pre", np.nan),
            "has_valid_control_window":        has_ctrl,
            "final_status":                    final_status,
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(audit_path, index=False)
    print(f"[AUDIT] {audit_path.name}  ({len(audit)} rows)")


def _make_row(pid, t, drug_, t_ag_start, t_ag_end, max_t,
              status, reason, group_map, drug_df, inf_changes,
              brs_seq_valid_pre=np.nan):
    t_from_ag = float(t - t_ag_start) if t_ag_start is not None else np.nan
    doses = get_cumulative_dose(drug_df, pid, t)
    grp   = group_map.get(pid, "unknown")

    # Compute infusion deltas in pre-window as covariables
    if t_ag_start is not None:
        d_pro, d_rem = get_max_infusion_delta(
            inf_changes, pid,
            t + PRE_START_S, t + PRE_END_S
        )
    else:
        d_pro, d_rem = np.nan, np.nan

    return {
        "patient_id":          pid,
        "t_bolus":             t,
        "drug":                drug_,
        "group":               grp,
        "t_ag_start":          t_ag_start,
        "t_ag_end":            t_ag_end,
        "t_from_ag_start_s":   t_from_ag,
        "cum_efedrina_mg":     doses["cum_efedrina_mg"],
        "cum_fenilefrina_mcg": doses["cum_fenilefrina_mcg"],
        "delta_propofol_pre":  d_pro,   # covariable (not exclusion criterion)
        "delta_remi_pre":      d_rem,   # covariable (not exclusion criterion)
        "brs_seq_valid_pre":   brs_seq_valid_pre,  # informational (not exclusion)
        "status":              status,
        "exclusion_reason":    reason,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 3 – Quiescent control windows (v2: 3 min, keep infusion check)
# ═══════════════════════════════════════════════════════════════════════════════

def build_control_windows_v2(
    events: pd.DataFrame,
    ann: pd.DataFrame,
    feat: pd.DataFrame,
    drug: pd.DataFrame,
) -> pd.DataFrame:
    """
    PASO 3.  For each clean event, find a quiescent 3-min control window
    from the SAME patient within the AG window.

    Control criteria:
      - No vasopresor bolus within ±5 min of control start
      - No pain stimulus within ±5 min of control start
      - No large infusion change within ±3 min (KEPT — controls must be quiescent)
      - Occurs within AG window
      - PPG valid ≥80% in control window
      - Feature data valid in control window

    Returns DataFrame with event_id, t_control_start, ctrl_excluded columns.
    """
    np.random.seed(SEED)

    ag_windows  = get_ag_windows(ann)
    inf_changes = build_infusion_change_table(ann)
    max_feat_t  = feat.groupby("patient_id")["t_window_start_s"].max() + WINDOW_S

    clean = events[events["status"] == "clean"].copy()
    print(f"\n[PASO 3] Finding 3-min control windows for {len(clean)} clean events …")

    results = []
    for _, ev in clean.iterrows():
        pid        = str(ev["patient_id"])
        t_bolus    = float(ev["t_bolus"])
        t_ag_start = float(ev["t_ag_start"]) if not pd.isna(ev["t_ag_start"]) else 0.0
        t_ag_end_raw = ev["t_ag_end"]
        t_ag_end = (
            float(max_feat_t.get(pid, t_ag_start + 10000))
            if pd.isna(t_ag_end_raw)
            else float(t_ag_end_raw)
        )

        # Scan range: inside AG window, buffer from edges
        EDGE_BUFFER = 60.0
        scan_lo = t_ag_start + EDGE_BUFFER
        scan_hi = t_ag_end   - EDGE_BUFFER - CONTROL_DURATION_S

        candidates = []
        t_scan = scan_lo
        STEP = 30.0
        while t_scan <= scan_hi:
            t_ctrl = t_scan
            t_ctrl_end = t_ctrl + CONTROL_DURATION_S

            # Exclude vicinity of bolus itself
            if abs(t_ctrl - t_bolus) < CONTROL_DURATION_S + 60.0:
                t_scan += STEP; continue

            # No bolus within ±CONTROL_BOLUS_BUFFER_S
            other_bolus = ann[
                (ann["patient_id"] == pid)
                & (ann["category"] == "medicacion_bolus")
                & (ann["subcategory"].isin(VASOPRESSOR_DRUGS))
                & (ann["t_seconds"].between(t_ctrl - CONTROL_BOLUS_BUFFER_S,
                                            t_ctrl + CONTROL_DURATION_S + CONTROL_BOLUS_BUFFER_S))
            ]
            if len(other_bolus) > 0:
                t_scan += STEP; continue

            # No pain stimulus within ±5 min
            stim = ann[
                (ann["patient_id"] == pid)
                & (ann["category"] == "estimulo")
                & (ann["t_seconds"].between(t_ctrl - 300, t_ctrl + CONTROL_DURATION_S + 300))
            ]
            if len(stim) > 0:
                t_scan += STEP; continue

            # No large infusion change within ±CONTROL_INFUSION_BUFFER_S
            inf_near = inf_changes[
                (inf_changes["patient_id"] == pid)
                & (inf_changes["t_seconds"].between(
                    t_ctrl - CONTROL_INFUSION_BUFFER_S,
                    t_ctrl + CONTROL_DURATION_S + CONTROL_INFUSION_BUFFER_S
                ))
            ]
            if len(inf_near) > 0:
                t_scan += STEP; continue

            # Feature validity in control window
            pat = feat[feat["patient_id"] == pid]
            w   = pat[
                (pat["t_window_start_s"] >= t_ctrl)
                & (pat["t_window_start_s"] <= t_ctrl + CONTROL_DURATION_S - WINDOW_S)
            ]
            if len(w) < 2:
                t_scan += STEP; continue

            # PPG validity
            if "ppg_valid_pct" in w.columns and float(w["ppg_valid_pct"].mean()) < 0.80:
                t_scan += STEP; continue

            # Primary feature validity
            feat_ok = True
            for fname, _, _ in PRIMARY:
                if fname not in w.columns:
                    continue
                if w[fname].notna().mean() < MIN_VALID_FRAC:
                    feat_ok = False; break
            if not feat_ok:
                t_scan += STEP; continue

            candidates.append(t_ctrl)
            t_scan += STEP

        if candidates:
            chosen = float(np.random.choice(candidates))
            results.append({
                "event_id":        int(ev["event_id"]),
                "patient_id":      pid,
                "t_bolus":         t_bolus,
                "t_control_start": chosen,
                "n_candidates":    len(candidates),
                "ctrl_excluded":   np.nan,
            })
            print(f"  [OK]   {pid} t={t_bolus:.0f}: "
                  f"{len(candidates)} candidates → t_ctrl={chosen:.0f}")
        else:
            results.append({
                "event_id":        int(ev["event_id"]),
                "patient_id":      pid,
                "t_bolus":         t_bolus,
                "t_control_start": np.nan,
                "n_candidates":    0,
                "ctrl_excluded":   "no_valid_control",
            })
            print(f"  [WARN] {pid} t={t_bolus:.0f}: no valid control window found")

    return pd.DataFrame(results)
