"""
Q2 v2 Statistics  (PASO 4 – 5 + PASO 8)
─────────────────────────────────────────
PASO 4  – build_paired_Q2a_v2()   → paired (pre, control)
          run_Q2a_v2_tests()       → GLMM one-sided Bonferroni (confirmatory)
PASO 5  – build_paired_Q2b_v2()   → paired (pre, post)
          run_Q2b_v2_tests()       → GLMM two-sided descriptive
PASO 8  – run_gradient_v2()       → group × feature interaction
          run_sensitivity_v2()     → SA-A..SA-D
PASO 7  – compute_brs_seq_q1()    → BRS_seq for Q1 events, save brs_seq_per_event.csv
"""
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2v2_config import (
    FEATURES_BRS_SEQ, FEATURES_LONG, EVENT_WINDOWS,
    Q1_RESULTS_V2, Q1_PAIRED_DATA, BRS_SEQ_Q1_EVENTS,
    PRE_START_S, PRE_END_S, POST_START_S, POST_END_S, WINDOW_S,
    CONTROL_DURATION_S,
    PRIMARY, EXPLORATORY, ALPHA_BONF, SEED, PARTIAL_PATIENT,
)

try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("[WARN] statsmodels not available; will use OLS fallback")


# ═══════════════════════════════════════════════════════════════════════════════
# Core statistics helpers (same as Q2 v1)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_agg(series: pd.Series, t_starts: pd.Series, agg: str) -> float:
    vals = series.dropna()
    if len(vals) < 1:
        return np.nan
    if agg == "mean":
        return float(vals.mean())
    elif agg == "std":
        return float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
    elif agg == "max":
        return float(vals.max())
    elif agg == "min":
        return float(vals.min())
    elif agg == "slope":
        if len(vals) < 2:
            return np.nan
        t = t_starts.loc[vals.index].values.astype(float)
        v = vals.values
        t_c = t - t.mean()
        denom = (t_c ** 2).sum()
        if denom < 1e-12:
            return np.nan
        return float((t_c * v).sum() / denom)
    return np.nan


def cohen_dz(delta: pd.Series) -> float:
    d = delta.dropna()
    if len(d) < 2:
        return np.nan
    return float(d.mean() / d.std(ddof=1))


def one_sided_p(p2: float, direction: int, beta: float) -> float:
    if np.isnan(p2):
        return np.nan
    p1 = p2 / 2
    if (direction < 0 and beta > 0) or (direction > 0 and beta < 0):
        p1 = 1 - p1
    return float(p1)


def run_mixed_model(df_model: pd.DataFrame, formula: str) -> dict:
    """Fit GLMM (random intercept per patient). Optimizer ladder: bfgs→cg→nm→OLS."""
    if not HAS_SM or len(df_model) < 3:
        return _ols_fallback(df_model, formula)
    for opt in ["bfgs", "cg", "nm"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md  = smf.mixedlm(formula, df_model, groups=df_model["patient_id"])
                res = md.fit(method=opt, reml=True, maxiter=800,
                             warn_convergence=False)
            if np.isfinite(res.params.get("delta", np.nan)):
                ci = res.conf_int()
                return {
                    "beta":     float(res.params["delta"]),
                    "ic_lo":    float(ci.loc["delta", 0]),
                    "ic_hi":    float(ci.loc["delta", 1]),
                    "p_2sided": float(res.pvalues["delta"]),
                    "method":   f"glmm_{opt}",
                }
        except Exception:
            continue
    return _ols_fallback(df_model, formula)


def run_mixed_model_covar(df_model: pd.DataFrame, covars: list[str]) -> dict:
    """GLMM with additional covariates: delta ~ covar1 + covar2 + (1 | patient_id)."""
    if not HAS_SM or len(df_model) < 4:
        return _ols_fallback(df_model, "delta ~ 1")
    cov_str = " + ".join(covars)
    formula = f"delta ~ {cov_str}"
    for opt in ["bfgs", "cg", "nm"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md  = smf.mixedlm(formula, df_model, groups=df_model["patient_id"])
                res = md.fit(method=opt, reml=True, maxiter=800,
                             warn_convergence=False)
            intercept_key = "Intercept"
            if intercept_key not in res.params:
                continue
            if np.isfinite(res.params[intercept_key]):
                ci = res.conf_int()
                return {
                    "beta":     float(res.params[intercept_key]),
                    "ic_lo":    float(ci.loc[intercept_key, 0]),
                    "ic_hi":    float(ci.loc[intercept_key, 1]),
                    "p_2sided": float(res.pvalues[intercept_key]),
                    "method":   f"glmm_{opt}_covar",
                }
        except Exception:
            continue
    return _ols_fallback(df_model, "delta ~ 1")


def _ols_fallback(df_model: pd.DataFrame, formula: str) -> dict:
    deltas = df_model["delta"].dropna()
    if len(deltas) < 2:
        return {"beta": np.nan, "ic_lo": np.nan, "ic_hi": np.nan,
                "p_2sided": np.nan, "method": "ols_fallback(n<2)"}
    res = stats.ttest_1samp(deltas, 0)
    se  = deltas.std(ddof=1) / np.sqrt(len(deltas))
    t_c = stats.t.ppf(0.975, df=len(deltas) - 1)
    return {
        "beta":     float(deltas.mean()),
        "ic_lo":    float(deltas.mean() - t_c * se),
        "ic_hi":    float(deltas.mean() + t_c * se),
        "p_2sided": float(res.pvalue),
        "method":   "ols_fallback",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_window_value(
    feat: pd.DataFrame, patient_id: str,
    t_ref: float, t_offset_lo: float, t_offset_hi: float,
    feature_name: str, agg: str,
    duration_override: float = None,
) -> float:
    """
    Compute aggregate of feature_name in window [t_ref + t_offset_lo, t_ref + t_offset_hi].
    duration_override: if set, use this as the window duration (for 3-min control windows).
    """
    pat  = feat[feat["patient_id"] == patient_id]
    t_lo = t_ref + t_offset_lo
    dur  = duration_override if duration_override else (t_offset_hi - t_offset_lo)
    t_hi = t_lo + dur - WINDOW_S
    w    = pat[(pat["t_window_start_s"] >= t_lo) & (pat["t_window_start_s"] <= t_hi)]
    if feature_name not in w.columns:
        return np.nan
    return compute_agg(w[feature_name], w["t_window_start_s"], agg)


def _empty_result(key, fname, agg, n, is_primary):
    return {
        "feature": key, "feature_name": fname, "agg": agg,
        "analysis": "primary" if is_primary else "exploratory",
        "direction": -1, "n_pairs": n, "n_patients": 0,
        "beta": np.nan, "ic_lo": np.nan, "ic_hi": np.nan, "sd_delta": np.nan,
        "p_2sided": np.nan, "p_1sided": np.nan, "p_bonferroni": np.nan,
        "cohen_dz": np.nan, "verdict": "NO_DATA", "model_note": "n<2",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 4 – Q2a v2 paired (pre vs control)
# ═══════════════════════════════════════════════════════════════════════════════

def build_paired_Q2a_v2(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each clean event with valid control, compute agg(pre) and agg(control).
    delta = value_pre − value_control.
    Control windows are CONTROL_DURATION_S (3 min) in v2.
    """
    ctrl_valid = controls[controls["ctrl_excluded"].isna()].copy()
    ev_ctrl = events[events["status"] == "clean"].merge(
        ctrl_valid[["event_id", "t_control_start"]],
        on="event_id", how="inner"
    )
    print(f"\n[PASO 4] Events with valid control: {len(ev_ctrl)}")

    all_features = PRIMARY + EXPLORATORY
    rows = []
    for _, ev in ev_ctrl.iterrows():
        pid     = str(ev["patient_id"])
        t_bolus = float(ev["t_bolus"])
        t_ctrl  = float(ev["t_control_start"])

        for fname, agg, direction in all_features:
            key   = f"{fname}__{agg}"
            v_pre = extract_window_value(feat, pid, t_bolus, PRE_START_S, PRE_END_S,
                                         fname, agg)
            v_ctrl = extract_window_value(feat, pid, t_ctrl, 0.0, CONTROL_DURATION_S,
                                          fname, agg,
                                          duration_override=CONTROL_DURATION_S)

            delta = float(v_pre - v_ctrl) if (not pd.isna(v_pre) and not pd.isna(v_ctrl)) else np.nan

            rows.append({
                "event_id":    int(ev["event_id"]),
                "patient_id":  pid,
                "drug":        str(ev["drug"]),
                "group":       str(ev["group"]),
                "feature_agg": key,
                "feature":     fname,
                "agg":         agg,
                "direction":   direction,
                "value_pre":   v_pre,
                "value_ctrl":  v_ctrl,
                "delta":       delta,
                "analysis":    "primary" if (fname, agg, direction) in PRIMARY else "exploratory",
                "delta_propofol_pre": float(ev.get("delta_propofol_pre", np.nan)),
                "delta_remi_pre":     float(ev.get("delta_remi_pre", np.nan)),
            })

    df = pd.DataFrame(rows)
    print(f"[PASO 4] Paired rows: {len(df)}  valid deltas: {df['delta'].notna().sum()} "
          f"({100*df['delta'].notna().mean():.1f}%)")
    return df


def run_Q2a_v2_tests(paired: pd.DataFrame) -> pd.DataFrame:
    """Q2a v2 confirmatory one-sided GLMM. H1: delta < 0."""
    all_features = PRIMARY + [(f, a, d) for f, a, d in EXPLORATORY]
    n_primary    = len(PRIMARY)
    results      = []

    print("\n[PASO 4] Running Q2a v2 tests …")
    for i, (fname, agg, direction) in enumerate(all_features):
        key        = f"{fname}__{agg}"
        subset     = paired[paired["feature_agg"] == key].dropna(subset=["delta"])
        is_primary = i < n_primary

        if len(subset) < 2:
            results.append(_empty_result(key, fname, agg, 0, is_primary))
            continue

        mm = run_mixed_model(subset[["delta", "patient_id"]].copy(), "delta ~ 1")
        p1     = one_sided_p(mm["p_2sided"], direction, mm["beta"])
        p_bonf = min(p1 * n_primary, 1.0) if is_primary else np.nan
        dz     = cohen_dz(subset["delta"])
        sd_d   = float(subset["delta"].std(ddof=1))

        verdict = ("VALIDADO"       if is_primary and not np.isnan(p_bonf) and p_bonf < 0.050
                   else "exploratory_sig" if not is_primary and mm["p_2sided"] < 0.050
                   else "NS")

        results.append({
            "feature": key, "feature_name": fname, "agg": agg,
            "analysis": "primary" if is_primary else "exploratory",
            "direction": direction,
            "n_pairs":   len(subset),
            "n_patients": subset["patient_id"].nunique(),
            "beta": mm["beta"], "ic_lo": mm["ic_lo"], "ic_hi": mm["ic_hi"],
            "sd_delta": sd_d,
            "p_2sided": mm["p_2sided"], "p_1sided": p1, "p_bonferroni": p_bonf,
            "cohen_dz": dz, "verdict": verdict, "model_note": mm["method"],
        })

        sym = "✓" if verdict == "VALIDADO" else "·"
        if is_primary:
            print(f"  {sym} {key:25s}  β={mm['beta']:+.4f} "
                  f"[{mm['ic_lo']:+.4f},{mm['ic_hi']:+.4f}]  p_bonf={p_bonf:.3f}")
        else:
            print(f"  {sym} {key:25s}  β={mm['beta']:+.4f}  "
                  f"p={mm['p_2sided']:.3f} [expl]")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 5 – Q2b v2 paired (pre vs post)
# ═══════════════════════════════════════════════════════════════════════════════

def build_paired_Q2b_v2(events: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """
    For each clean event compute agg(pre) and agg(post).
    delta_post = value_post − value_pre.
    """
    clean = events[events["status"] == "clean"].copy()
    print(f"\n[PASO 5] Building pre-post pairs for {len(clean)} events …")

    all_features = PRIMARY + EXPLORATORY
    rows = []
    for _, ev in clean.iterrows():
        pid     = str(ev["patient_id"])
        t_bolus = float(ev["t_bolus"])

        for fname, agg, direction in all_features:
            key   = f"{fname}__{agg}"
            v_pre = extract_window_value(feat, pid, t_bolus, PRE_START_S, PRE_END_S, fname, agg)
            v_post= extract_window_value(feat, pid, t_bolus, POST_START_S, POST_END_S, fname, agg)
            delta_post = float(v_post - v_pre) if (not pd.isna(v_pre) and not pd.isna(v_post)) else np.nan

            rows.append({
                "event_id":   int(ev["event_id"]),
                "patient_id": pid,
                "drug":       str(ev["drug"]),
                "group":      str(ev["group"]),
                "feature_agg": key,
                "feature":    fname,
                "agg":        agg,
                "value_pre":  v_pre,
                "value_post": v_post,
                "delta_post": delta_post,
                "analysis":   "primary" if (fname, agg, direction) in PRIMARY else "exploratory",
            })

    df = pd.DataFrame(rows)
    print(f"[PASO 5] Pre-post rows: {len(df)}  valid: {df['delta_post'].notna().sum()}")
    return df


def run_Q2b_v2_tests(paired_post: pd.DataFrame) -> pd.DataFrame:
    """Descriptive two-sided GLMM for pre→post changes."""
    all_features = PRIMARY + EXPLORATORY
    results = []
    print("\n[PASO 5] Q2b v2 pre→post tests:")

    for fname, agg, direction in all_features:
        key    = f"{fname}__{agg}"
        subset = paired_post[paired_post["feature_agg"] == key].dropna(subset=["delta_post"])
        is_primary = (fname, agg, direction) in PRIMARY

        if len(subset) < 2:
            results.append(_empty_result(key, fname, agg, 0, is_primary))
            continue

        df_m = subset[["delta_post", "patient_id"]].rename(columns={"delta_post": "delta"})
        mm   = run_mixed_model(df_m, "delta ~ 1")
        dz   = cohen_dz(subset["delta_post"])
        sd_d = float(subset["delta_post"].std(ddof=1))

        results.append({
            "feature": key, "feature_name": fname, "agg": agg,
            "analysis": "primary" if is_primary else "exploratory",
            "n_pairs":  len(subset), "n_patients": subset["patient_id"].nunique(),
            "beta": mm["beta"], "ic_lo": mm["ic_lo"], "ic_hi": mm["ic_hi"],
            "sd_delta": sd_d, "p_2sided": mm["p_2sided"],
            "cohen_dz": dz, "model_note": mm["method"],
        })
        print(f"  · {key:25s}  Δpost={mm['beta']:+.4f} "
              f"[{mm['ic_lo']:+.4f},{mm['ic_hi']:+.4f}]  p={mm['p_2sided']:.3f}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 8 – Gradient and sensitivity analyses
# ═══════════════════════════════════════════════════════════════════════════════

def run_gradient_v2(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Group × delta interaction.
    For each validated or all primary feature: delta ~ group + (1 | patient_id).
    H1 one-sided: |delta_interesc| > |delta_supra_ax|.
    """
    if not HAS_SM:
        print("[PASO 8a] statsmodels not available — skipping gradient")
        return pd.DataFrame()

    print("\n[PASO 8a] Gradient analysis (group effect on pre-ctrl delta):")
    results = []

    for fname, agg, direction in PRIMARY:
        key    = f"{fname}__{agg}"
        subset = paired[paired["feature_agg"] == key].dropna(subset=["delta"])

        if len(subset) < 4 or subset["group"].nunique() < 2:
            print(f"  {key}: insufficient data for gradient (n={len(subset)})")
            continue

        df_m = subset[["delta", "patient_id", "group"]].copy()
        df_m["group_bin"] = (df_m["group"] == "interescalenico").astype(float)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md  = smf.mixedlm("delta ~ group_bin", df_m, groups=df_m["patient_id"])
                res = md.fit(method="bfgs", reml=True, maxiter=800,
                             warn_convergence=False)
            beta_g = float(res.params.get("group_bin", np.nan))
            p_g    = float(res.pvalues.get("group_bin", np.nan))
        except Exception:
            beta_g, p_g = np.nan, np.nan

        print(f"  {key:25s}  β_interesk={beta_g:+.4f}  p={p_g:.3f}")
        results.append({
            "feature": key, "feature_name": fname, "agg": agg,
            "beta_interescalenico": beta_g, "p_gradient": p_g,
        })

    return pd.DataFrame(results)


def run_sensitivity_v2(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    feat: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Sensitivity analyses:
      SA-A: efedrina only
      SA-B: exclude partial patient (70767707)
      SA-C: adjust by delta_propofol_pre + delta_remi_pre (diagnostic for confounding)
      SA-D: adjust by demographics (exploratory flag)
    Returns dict {label: results_df}.
    """
    print("\n[PASO 8b] Sensitivity analyses …")

    def _run_sa(ev_subset, ctrl_subset, feat_, label, covar_cols=None):
        p_ctrl = build_paired_Q2a_v2(ev_subset, ctrl_subset, feat_)
        res    = run_Q2a_v2_tests(p_ctrl)
        n_events = (ev_subset["status"] == "clean").sum()
        print(f"  {label}: {n_events} events")
        if covar_cols:
            # Run additional GLMM with covariates for primary features
            covar_results = []
            for fname, agg, direction in PRIMARY:
                key    = f"{fname}__{agg}"
                subset = p_ctrl[p_ctrl["feature_agg"] == key].dropna(subset=["delta"])
                # Drop rows missing covariates
                cov_ok = subset.dropna(subset=covar_cols)
                if len(cov_ok) < 3:
                    covar_results.append({
                        "feature": key, "n_with_covar": len(cov_ok),
                        "beta_adj": np.nan, "p_adj": np.nan
                    })
                    continue
                mm = run_mixed_model_covar(cov_ok[["delta", "patient_id"] + covar_cols], covar_cols)
                covar_results.append({
                    "feature": key, "n_with_covar": len(cov_ok),
                    "beta_adj": mm["beta"], "p_adj": mm["p_2sided"],
                    "model_note": mm["method"],
                })
            res = res.copy()
            covar_df = pd.DataFrame(covar_results)
            if not covar_df.empty:
                res = res.merge(covar_df, on="feature", how="left", suffixes=("", "_covadjusted"))
        return res

    sa_results = {}

    # SA-A: efedrina only
    ev_a  = events[events["drug"] != "fenilefrina"].copy()
    sa_results["SA_A_efedrina_only"] = _run_sa(ev_a, controls, feat, "SA-A (efedrina only)")

    # SA-B: exclude partial patient
    ev_b   = events[events["patient_id"] != PARTIAL_PATIENT].copy()
    ctrl_b = controls[controls["patient_id"] != PARTIAL_PATIENT].copy() \
             if "patient_id" in controls.columns else controls.copy()
    sa_results["SA_B_no_partial"] = _run_sa(ev_b, ctrl_b, feat, f"SA-B (excl. {PARTIAL_PATIENT})")

    # SA-C: adjust by delta_propofol_pre + delta_remi_pre (CRITICAL diagnostic test)
    print("  SA-C: Adjusting for delta_propofol_pre + delta_remi_pre …")
    print("  (If rigidification loses significance: co-driven by infusion change)")
    paired_c = build_paired_Q2a_v2(events, controls, feat)
    sa_c_results = []
    for fname, agg, direction in PRIMARY:
        key    = f"{fname}__{agg}"
        subset = paired_c[paired_c["feature_agg"] == key].dropna(
            subset=["delta", "delta_propofol_pre", "delta_remi_pre"]
        )
        if len(subset) < 3:
            sa_c_results.append({
                "feature": key, "feature_name": fname, "agg": agg,
                "n_adj": len(subset), "beta_adj": np.nan, "p_adj": np.nan,
                "verdict_adj": "NO_DATA",
            })
            continue
        mm = run_mixed_model_covar(
            subset[["delta", "patient_id", "delta_propofol_pre", "delta_remi_pre"]],
            ["delta_propofol_pre", "delta_remi_pre"],
        )
        p1 = one_sided_p(mm["p_2sided"], direction, mm["beta"])
        sa_c_results.append({
            "feature": key, "feature_name": fname, "agg": agg,
            "n_adj": len(subset),
            "beta_adj": mm["beta"], "ic_lo_adj": mm["ic_lo"], "ic_hi_adj": mm["ic_hi"],
            "p_2sided_adj": mm["p_2sided"], "p_1sided_adj": p1,
            "model_note": mm["method"],
            "verdict_adj": "ROBUST" if p1 < ALPHA_BONF else "NS_after_adj",
        })
        print(f"    {key:25s}  β_adj={mm['beta']:+.4f}  p_adj={mm['p_2sided']:.3f}")
    sa_results["SA_C_infusion_adjusted"] = pd.DataFrame(sa_c_results)

    # SA-D: adjust by demographics (if available) — EXPLORATORY, flagged
    print("  SA-D: demographic adjustment (exploratory, n is small) …")
    # We don't have age/BMI in the current data model; skip gracefully
    sa_results["SA_D_demographics"] = pd.DataFrame([{
        "note": "Demographic variables not available in current dataset. "
                "SA-D not computable. Flag: EXPLORATORY."
    }])

    return sa_results


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 7 – Compute BRS_seq for Q1 events  (save to Q1_RES)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_brs_seq_q1(feat: pd.DataFrame) -> pd.DataFrame:
    """
    For each Q1 stimulus event, compute brs_seq_pre (nanmin in [-5,-2] min) and
    brs_seq_post (nanmin in [0,+2.5] min), then delta = post_min - pre_min.

    FIX 3: correct pairing approach.
    - The event_windows.csv "event" rows cover the 5-min PRE window:
        t_start_s = t_stimulus - 300 s
        t_end_s   = t_stimulus          ← stimulus timestamp
    - Controls have event_subcategory="none" and cannot be matched by subcategory.
    - The Q1 paired design is PRE vs POST the same stimulus (not event vs control window).
    - We compute brs_seq directly from features_brs_seq.parquet using the
      stimulus timestamp recovered from t_end_s of each "event" window.

    Pre window  : [t_stim - 300, t_stim - 120]  = [-5, -2] min
    Post window : [t_stim,       t_stim + 150]   = [ 0, +2.5] min
    Aggregation : nanmin over 30s sub-windows (consistent with Q1 agg="min")
    Direction   : H1 one-sided: β < 0 (BRS_seq decreases = deterioro, pre→post)
    """
    print("\n[PASO 7] Computing BRS_seq for Q1 events (Fix 3: pre/post within event) …")
    ev_win = pd.read_csv(EVENT_WINDOWS)
    ev_win["patient_id"] = ev_win["patient_id"].astype(str)

    if "brs_seq" not in feat.columns:
        print("[WARN] brs_seq not in features — Q1 BRS_seq unavailable")
        return pd.DataFrame()

    # Only process "event" rows; t_end_s = t_stimulus
    stim_rows = ev_win[ev_win["window_type"] == "event"].copy()
    print(f"[PASO 7] Q1 stimulus events: {len(stim_rows)}")

    rows = []
    for _, row in stim_rows.iterrows():
        pid        = str(row["patient_id"])
        t_stimulus = float(row["t_end_s"])   # FIX: t_end = stimulus timestamp
        subcat     = row["event_subcategory"]
        group      = row["group"]

        pat = feat[feat["patient_id"] == pid]

        # Pre window: [-5, -2] min = [-300, -120] s from stimulus
        t_pre_lo  = t_stimulus - 300
        t_pre_hi  = t_stimulus - 120
        pre_w = pat[(pat["t_window_start_s"] >= t_pre_lo)
                    & (pat["t_window_start_s"] <= t_pre_hi - WINDOW_S)]
        pre_brs = pre_w["brs_seq"].dropna()

        # Post window: [0, +2.5] min = [0, +150] s from stimulus
        t_post_lo = t_stimulus
        t_post_hi = t_stimulus + 150
        post_w = pat[(pat["t_window_start_s"] >= t_post_lo)
                     & (pat["t_window_start_s"] <= t_post_hi - WINDOW_S)]
        post_brs = post_w["brs_seq"].dropna()

        pre_min  = float(np.nanmin(pre_brs))  if len(pre_brs) > 0 else np.nan
        post_min = float(np.nanmin(post_brs)) if len(post_brs) > 0 else np.nan
        delta    = post_min - pre_min if (not np.isnan(pre_min) and not np.isnan(post_min)) else np.nan

        rows.append({
            "patient_id":        pid,
            "group":             group,
            "event_subcategory": subcat,
            "t_stimulus_s":      t_stimulus,
            "n_brs_pre":         len(pre_brs),
            "n_brs_post":        len(post_brs),
            "brs_seq_pre_min":   pre_min,
            "brs_seq_post_min":  post_min,
            "delta_brs_seq_min": delta,
        })

    df_pairs = pd.DataFrame(rows)

    n_valid = int(df_pairs["delta_brs_seq_min"].notna().sum())
    n_pre_ok  = int(df_pairs["brs_seq_pre_min"].notna().sum())
    n_post_ok = int(df_pairs["brs_seq_post_min"].notna().sum())
    print(f"[PASO 7] Events with valid brs_seq_pre : {n_pre_ok} / {len(df_pairs)}")
    print(f"[PASO 7] Events with valid brs_seq_post: {n_post_ok} / {len(df_pairs)}")
    print(f"[PASO 7] Events with complete delta     : {n_valid} / {len(df_pairs)}")

    n_patients = 0
    if n_valid >= 2:
        df_m = df_pairs[["delta_brs_seq_min", "patient_id"]].rename(
            columns={"delta_brs_seq_min": "delta"}
        ).dropna()
        n_patients = df_m["patient_id"].nunique()

        # Two-level random effects: patient_id (+ event_subcategory via strat.)
        # With only patient_id available as grouping factor, use single random effect
        mm = run_mixed_model(df_m, "delta ~ 1")
        # H1: β < 0 (BRS_seq decreases pre→post stimulus = deterioro)
        p1 = one_sided_p(mm["p_2sided"], direction=-1, beta=mm["beta"])
        dz = cohen_dz(df_m["delta"])

        print(f"[PASO 7] Q1 BRS_seq GLMM: β={mm['beta']:+.4f} "
              f"[{mm.get('ic_lo', float('nan')):+.4f},{mm.get('ic_hi', float('nan')):+.4f}]  "
              f"p_1sided={p1:.3f}  d_z={dz:.3f}")
    else:
        mm = {"beta": np.nan, "ic_lo": np.nan, "ic_hi": np.nan,
              "p_2sided": np.nan, "method": "n<2"}
        p1, dz = np.nan, np.nan
        print("[PASO 7] Insufficient Q1 brs_seq data for GLMM")

    # Save to Q1_RES (new file, do not modify other Q1 files)
    BRS_SEQ_Q1_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(BRS_SEQ_Q1_EVENTS, index=False)
    print(f"[SAVE] {BRS_SEQ_Q1_EVENTS.name}  ({len(df_pairs)} rows in Q1_RES)")

    # Return summary row for test_results_v2.csv
    summary = pd.DataFrame([{
        "feature":      "brs_seq__min",
        "feature_name": "brs_seq",
        "agg":          "min",
        "analysis":     "primary_brs_seq",  # flag: added in v2
        "direction":    -1,
        "n_pairs":      n_valid,
        "n_patients":   n_patients,
        "beta":         mm["beta"],
        "ic_lo":        mm.get("ic_lo", np.nan),
        "ic_hi":        mm.get("ic_hi", np.nan),
        "p_2sided":     mm.get("p_2sided", np.nan),
        "p_1sided":     p1,
        "cohen_dz":     dz,
        "verdict":      "Q1_SIG" if (not np.isnan(p1) and p1 < 0.050) else "Q1_NS",
        "model_note":   mm.get("method", ""),
    }])

    return summary
