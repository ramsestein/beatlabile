"""
Q2 Statistics  (PASO 3 – 5)
────────────────────────────
PASO 3  – build_paired_Q2a()      → paired (pre, control) per feature
         run_Q2a_tests()           → GLMM one-sided Bonferroni  (confirmatory)
PASO 4  – build_paired_Q2b()      → paired (pre, post)
         run_Q2b_tests()           → GLMM two-sided descriptive
PASO 5  – run_gradient_Q2()       → group × time interaction
         run_sensitivity_Q2()      → 3 sensitivity analyses
"""
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_config import (
    FEATURES_LONG, PAIRED_PRE_CONTROL, PAIRED_PRE_POST,
    TEST_RESULTS_Q2A, TEST_RESULTS_Q2B,
    PRE_START_S, PRE_END_S, POST_START_S, POST_END_S, WINDOW_S,
    PRIMARY, EXPLORATORY, ALPHA_BONF, SEED, PARTIAL_PATIENT,
)

try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("[WARN] statsmodels not available; will use OLS fallback")


# ═══════════════════════════════════════════════════════════════════════════════
# Core statistics helpers (same approach as Q1 v2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_agg(series: pd.Series, t_starts: pd.Series, agg: str) -> float:
    """
    Aggregate a series of feature values using the requested method.
    agg: "mean" | "std" | "max" | "min" | "slope"
    """
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
    """
    Convert two-sided p to one-sided in the pre-specified direction.
    direction: -1 (expect beta < 0), +1 (expect beta > 0).
    If the observed sign matches, p1 = p2/2; else p1 = 1 - p2/2.
    """
    if np.isnan(p2):
        return np.nan
    p1 = p2 / 2
    if (direction < 0 and beta > 0) or (direction > 0 and beta < 0):
        p1 = 1 - p1
    return float(p1)


def run_mixed_model(df_model: pd.DataFrame, formula: str) -> dict:
    """
    Fit GLMM (random intercept per patient) using statsmodels MixedLM.
    Optimizer ladder: bfgs → cg → nm → OLS fallback.
    Returns dict with beta, ic_lo, ic_hi, p_2sided, method.
    """
    if not HAS_SM or len(df_model) < 3:
        return _ols_fallback(df_model, formula)

    OPTIMIZERS = ["bfgs", "cg", "nm"]
    for opt in OPTIMIZERS:
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


def _ols_fallback(df_model: pd.DataFrame, formula: str) -> dict:
    """Simple one-sample t-test on deltas as OLS fallback."""
    deltas = df_model["delta"].dropna()
    if len(deltas) < 2:
        return {"beta": np.nan, "ic_lo": np.nan, "ic_hi": np.nan,
                "p_2sided": np.nan, "method": "ols_fallback(n<2)"}
    res = stats.ttest_1samp(deltas, 0)
    se  = deltas.std(ddof=1) / np.sqrt(len(deltas))
    t_c = stats.t.ppf(0.975, df=len(deltas)-1)
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
) -> float:
    """
    Compute aggregate of `feature_name` in the window
    [t_ref + t_offset_lo, t_ref + t_offset_hi - WINDOW_S].
    """
    pat = feat[feat["patient_id"] == patient_id]
    t_lo = t_ref + t_offset_lo
    t_hi = t_ref + t_offset_hi - WINDOW_S
    w    = pat[(pat["t_window_start_s"] >= t_lo) & (pat["t_window_start_s"] <= t_hi)]
    if feature_name not in w.columns:
        return np.nan
    return compute_agg(w[feature_name], w["t_window_start_s"], agg)


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 3 – Q2a paired (pre vs control)
# ═══════════════════════════════════════════════════════════════════════════════

def build_paired_Q2a(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each clean event that has a valid control window, compute
    agg(feature, pre-window) and agg(feature, control-window).
    delta = value_pre − value_control.
    """
    # Merge events with control windows
    ctrl_valid = controls[controls["ctrl_excluded"].isna()].copy()
    ev_ctrl = events[events["status"] == "clean"].merge(
        ctrl_valid[["event_id", "t_control_start"]],
        on="event_id", how="inner"
    )
    print(f"\n[PASO 3] Events with valid control: {len(ev_ctrl)}")

    all_features = PRIMARY + EXPLORATORY
    rows = []
    for _, ev in ev_ctrl.iterrows():
        pid     = str(ev["patient_id"])
        t_bolus = float(ev["t_bolus"])
        t_ctrl  = float(ev["t_control_start"])

        for fname, agg, direction in all_features:
            key = f"{fname}__{agg}"
            v_pre  = extract_window_value(feat, pid, t_bolus, PRE_START_S, PRE_END_S, fname, agg)
            v_ctrl = extract_window_value(feat, pid, t_ctrl,  0.0, POST_END_S, fname, agg)

            if not pd.isna(v_pre) and not pd.isna(v_ctrl):
                delta = v_pre - v_ctrl
            else:
                delta = np.nan

            rows.append({
                "event_id":   int(ev["event_id"]),
                "patient_id": pid,
                "drug":       str(ev["drug"]),
                "group":      str(ev["group"]),
                "feature_agg": key,
                "feature":    fname,
                "agg":        agg,
                "direction":  direction,
                "value_pre":  v_pre,
                "value_ctrl": v_ctrl,
                "delta":      delta,
                "analysis":   "primary" if (fname, agg, direction) in PRIMARY else "exploratory",
            })

    df = pd.DataFrame(rows)
    print(f"[PASO 3] Paired rows: {len(df)}  valid deltas: {df['delta'].notna().sum()} "
          f"({100*df['delta'].notna().mean():.1f}%)")
    return df


def run_Q2a_tests(paired: pd.DataFrame) -> pd.DataFrame:
    """
    PASO 3. Confirmatory one-sided GLMM for each primary feature.
    Also runs exploratory features (two-sided, flagged).
    H1: delta < 0  (pre-bolus feature < quiescent control).
    """
    all_features = PRIMARY + [(f, a, d) for f, a, d in EXPLORATORY]
    n_primary    = len(PRIMARY)
    results = []

    for i, (fname, agg, direction) in enumerate(all_features):
        key    = f"{fname}__{agg}"
        subset = paired[paired["feature_agg"] == key].dropna(subset=["delta"])
        is_primary = i < n_primary

        if len(subset) < 2:
            results.append(_empty_result(key, fname, agg, 0, is_primary))
            continue

        mm = run_mixed_model(subset[["delta", "patient_id"]].copy(), "delta ~ 1")

        p1 = one_sided_p(mm["p_2sided"], direction, mm["beta"])
        p_bonf = min(p1 * n_primary, 1.0) if is_primary else np.nan
        dz = cohen_dz(subset["delta"])
        sd_delta = float(subset["delta"].std(ddof=1))

        verdict = "VALIDADO" if (is_primary and not np.isnan(p_bonf) and p_bonf < 0.050) else (
                  "exploratory_sig" if (not is_primary and mm["p_2sided"] < 0.050) else "NS")

        results.append({
            "feature":      key,
            "feature_name": fname,
            "agg":          agg,
            "analysis":     "primary" if is_primary else "exploratory",
            "direction":    direction,
            "n_pairs":      len(subset),
            "n_patients":   subset["patient_id"].nunique(),
            "beta":         mm["beta"],
            "ic_lo":        mm["ic_lo"],
            "ic_hi":        mm["ic_hi"],
            "sd_delta":     sd_delta,
            "p_2sided":     mm["p_2sided"],
            "p_1sided":     p1,
            "p_bonferroni": p_bonf,
            "cohen_dz":     dz,
            "verdict":      verdict,
            "model_note":   mm["method"],
        })

        sym = "✓" if verdict == "VALIDADO" else "·"
        print(f"  {sym} {key:25s}  β={mm['beta']:+.4f} [{mm['ic_lo']:+.4f},{mm['ic_hi']:+.4f}]"
              f"  p_bonf={p_bonf:.3f}" if is_primary else
              f"  {sym} {key:25s}  β={mm['beta']:+.4f}  p={mm['p_2sided']:.3f} [expl]")

    return pd.DataFrame(results)


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
# PASO 4 – Q2b paired (pre vs post)
# ═══════════════════════════════════════════════════════════════════════════════

def build_paired_Q2b(events: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """
    For each clean event compute agg(feature, pre) and agg(feature, post).
    delta_post = value_post − value_pre  (positive = increase after bolus).
    """
    clean = events[events["status"] == "clean"].copy()
    print(f"\n[PASO 4] Building pre-post pairs for {len(clean)} events …")

    all_features = PRIMARY + EXPLORATORY
    rows = []
    for _, ev in clean.iterrows():
        pid     = str(ev["patient_id"])
        t_bolus = float(ev["t_bolus"])

        for fname, agg, direction in all_features:
            key = f"{fname}__{agg}"
            v_pre  = extract_window_value(feat, pid, t_bolus, PRE_START_S, PRE_END_S, fname, agg)
            v_post = extract_window_value(feat, pid, t_bolus, POST_START_S, POST_END_S, fname, agg)

            delta_post = float(v_post - v_pre) if not pd.isna(v_pre) and not pd.isna(v_post) else np.nan

            rows.append({
                "event_id":    int(ev["event_id"]),
                "patient_id":  pid,
                "drug":        str(ev["drug"]),
                "group":       str(ev["group"]),
                "feature_agg": key,
                "feature":     fname,
                "agg":         agg,
                "value_pre":   v_pre,
                "value_post":  v_post,
                "delta_post":  delta_post,
                "analysis":    "primary" if (fname, agg, direction) in PRIMARY else "exploratory",
            })

    df = pd.DataFrame(rows)
    print(f"[PASO 4] Pre-post rows: {len(df)}  valid: {df['delta_post'].notna().sum()}")
    return df


def run_Q2b_tests(paired_post: pd.DataFrame) -> pd.DataFrame:
    """
    PASO 4. Descriptive two-sided GLMM for each feature.
    Tests whether the feature changes from pre to post (bolus response).
    """
    all_features = PRIMARY + EXPLORATORY
    results = []
    print("\n[PASO 4] Q2b pre→post tests:")

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

        p_sig = mm["p_2sided"] < 0.050
        results.append({
            "feature":      key,
            "feature_name": fname,
            "agg":          agg,
            "analysis":     "primary" if is_primary else "exploratory",
            "n_pairs":      len(subset),
            "n_patients":   subset["patient_id"].nunique(),
            "beta_post":    mm["beta"],
            "ic_lo":        mm["ic_lo"],
            "ic_hi":        mm["ic_hi"],
            "p_2sided":     mm["p_2sided"],
            "cohen_dz":     dz,
            "model_note":   mm["method"],
        })
        sym = "✓" if p_sig else "·"
        print(f"  {sym} {key:25s}  Δpost={mm['beta']:+.4f} [{mm['ic_lo']:+.4f},{mm['ic_hi']:+.4f}]"
              f"  p={mm['p_2sided']:.3f}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# PASO 5 – Gradient and sensitivity analyses
# ═══════════════════════════════════════════════════════════════════════════════

def run_gradient_Q2(paired: pd.DataFrame) -> pd.DataFrame:
    """
    PASO 5a. Test whether the group (interescalenico vs supra_axilar) modifies
    the pre-bolus signature.  Contrary to Q1 (where interesc. attenuates pain
    response), here we expect interesc. to AMPLIFY the pre-hypotensive drop
    (less sympathetic reserve → deeper decompensation).
    Uses interaction: delta ~ group  (interescalenico coded as 1).
    """
    if not HAS_SM:
        return pd.DataFrame()

    validated = [
        (f, a, d) for f, a, d in PRIMARY
    ]
    results = []
    print("\n[PASO 5a] Gradient analysis (group × pre-ctrl delta):")

    for fname, agg, _ in validated:
        key    = f"{fname}__{agg}"
        subset = paired[paired["feature_agg"] == key].dropna(subset=["delta"]).copy()
        if len(subset) < 4:
            continue

        subset["is_interesk"] = (subset["group"] == "interescalenico").astype(float)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md  = smf.mixedlm("delta ~ is_interesk", subset, groups=subset["patient_id"])
                res = md.fit(method="bfgs", reml=True, maxiter=800, warn_convergence=False)
            ci  = res.conf_int()
            beta_int = float(res.params.get("is_interesk", np.nan))
            p_int    = float(res.pvalues.get("is_interesk", np.nan))
            results.append({
                "feature":       key,
                "beta_interesk": beta_int,
                "ic_lo":         float(ci.loc["is_interesk", 0]) if "is_interesk" in ci.index else np.nan,
                "ic_hi":         float(ci.loc["is_interesk", 1]) if "is_interesk" in ci.index else np.nan,
                "p_interaction": p_int,
                "interpretation": "interesk_amplifies" if beta_int < 0 else "interesk_attenuates",
            })
            print(f"  {key:25s}  β_interesk={beta_int:+.4f}  p={p_int:.3f}")
        except Exception as e:
            print(f"  [FAIL] {key}: {e}")

    return pd.DataFrame(results) if results else pd.DataFrame()


def run_sensitivity_Q2(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    feat: pd.DataFrame,
) -> dict:
    """
    PASO 5b. Three sensitivity analyses:
      SA1 – Exclude the partial patient (PARTIAL_PATIENT)
      SA2 – Efedrina only (no fenilefrina)
      SA3 – Interescalenico group only
    Returns dict of DataFrames keyed by SA label.
    """
    print("\n[PASO 5b] Sensitivity analyses …")
    sens_results = {}

    sa_defs = [
        ("SA1_no_partial",    "patient_id != @PARTIAL_PATIENT"),
        ("SA2_efedrina_only", "drug == 'efedrina'"),
        ("SA3_interesk_only", "group == 'interescalenico'"),
    ]

    for sa_label, filt_expr in sa_defs:
        try:
            ev_sub = events.query(filt_expr).copy() if filt_expr else events.copy()
        except Exception:
            ev_sub = events[~(events["patient_id"] == PARTIAL_PATIENT)] if "partial" in sa_label else events.copy()

        # Rebuild paired Q2a for this subset
        paired_sa = build_paired_Q2a(ev_sub, controls, feat)
        if len(paired_sa) < 2:
            print(f"  [SKIP] {sa_label}: n<2")
            continue

        results_sa = []
        for fname, agg, direction in PRIMARY:
            key    = f"{fname}__{agg}"
            subset = paired_sa[paired_sa["feature_agg"] == key].dropna(subset=["delta"])
            if len(subset) < 2:
                results_sa.append(_empty_result(key, fname, agg, len(subset), True))
                continue
            mm = run_mixed_model(subset[["delta", "patient_id"]].copy(), "delta ~ 1")
            p1 = one_sided_p(mm["p_2sided"], direction, mm["beta"])
            results_sa.append({
                "feature":      key,
                "n_pairs":      len(subset),
                "beta":         mm["beta"],
                "ic_lo":        mm["ic_lo"],
                "ic_hi":        mm["ic_hi"],
                "p_1sided":     p1,
                "cohen_dz":     cohen_dz(subset["delta"]),
                "model_note":   mm["method"],
                "analysis":     sa_label,
            })

        sens_results[sa_label] = pd.DataFrame(results_sa)
        print(f"  {sa_label}: {len(ev_sub[ev_sub['status']=='clean'])} events")

    return sens_results
