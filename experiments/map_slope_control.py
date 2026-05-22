#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAP SLOPE COMO CONTROL PREDICTIVO — ¿Cuánto del AUC es ya visible en la tendencia MAP?
========================================================================================
Experimento de control: calcula features de la MAP cruda (Solar8000/ART_MBP a 1 Hz)
sobre la ventana de predicción [window_start_s, window_end_s] para cada ventana del
parquet de VitalDB.  Compara el potencial predictivo (AUC en hipotensión e hipertensión)
de estas features de MAP contra las features autonómicas del modelo parsimónico.

Features de MAP calculadas:
  map_mean       : media de MAP en la ventana (nivel basal)
  map_slope      : pendiente lineal (mmHg/min) — negativa = MAP bajando
  map_slope_last10 : pendiente en los últimos 10 min (señal más reciente)
  map_std        : desviación estándar de MAP en la ventana
  map_end_mean   : media de MAP en los últimos 5 min
  map_start_mean : media de MAP en los primeros 5 min
  map_end_vs_start: map_end_mean - map_start_mean (cambio total)

Outputs:
  results/supplementary/map_slope_features.parquet   — features por ventana
  results/supplementary/map_slope_auc_comparison.csv — AUC MAP vs autonomic
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

CACHE_PATH   = Path("results/supplementary/map_slope_features.parquet")
WINDOWS_PATH = Path("results/cache/vitaldb_windows.parquet")
OUTCOMES     = ["hypotension", "hypertension"]

PARSIMONIOUS = {
    "hypotension":  ["std_pa_mean", "cv_pa_std", "rsa_max", "arv_std",
                     "brs_min", "arv_mean", "std_pa_max", "rsa_mean"],
    "hypertension": ["std_pa_std", "cv_pa_std", "brs_min", "cv_pa_mean",
                     "std_pa_max", "arv_std", "std_pa_slope", "sdnn_mean"],
}
MAP_FEATURES = ["map_mean", "map_slope", "map_slope_last10",
                "map_std", "map_end_vs_start"]

# ── helpers ────────────────────────────────────────────────────────────────────
def _slope_mmhg_per_min(arr, dt_s=1.0):
    """OLS slope of a 1-D signal; returns mmHg/min (or NaN if <10 points)."""
    valid = arr[~np.isnan(arr)]
    if len(valid) < 10:
        return np.nan
    t = np.arange(len(valid)) * dt_s / 60.0   # minutes
    slope, *_ = np.polyfit(t, valid, 1)
    return slope

def _safe_mean(arr):
    v = arr[~np.isnan(arr)]
    return float(v.mean()) if len(v) > 0 else np.nan

def extract_map_features(arr, window_len_s=1800):
    """Extract MAP features from a 1-Hz array segment."""
    last10  = arr[max(0, len(arr)-600):]
    first5  = arr[:300]
    last5   = arr[max(0, len(arr)-300):]
    return {
        "map_mean":          _safe_mean(arr),
        "map_slope":         _slope_mmhg_per_min(arr),
        "map_slope_last10":  _slope_mmhg_per_min(last10),
        "map_std":           float(arr[~np.isnan(arr)].std()) if (~np.isnan(arr)).sum() > 1 else np.nan,
        "map_end_mean":      _safe_mean(last5),
        "map_start_mean":    _safe_mean(first5),
        "map_end_vs_start":  _safe_mean(last5) - _safe_mean(first5),
    }

# ── download & compute ─────────────────────────────────────────────────────────
def build_map_features(windows: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Download ART_MBP for each patient and compute MAP features per window."""
    import vitaldb

    windows = windows.copy()
    windows["caseid"] = windows["patient_id"].str.lstrip("0").astype(int)

    results = []
    patients = windows["caseid"].unique()
    n = len(patients)

    for i, cid in enumerate(patients):
        if verbose and i % 50 == 0:
            print(f"  [{i+1}/{n}] case {cid}", flush=True)
        try:
            data = vitaldb.load_case(int(cid), ["Solar8000/ART_MBP"], interval=1)
            if data.shape[0] == 0:
                continue
            art = data[:, 0].astype(float)
            # Replace physiologically implausible values
            art[(art < 20) | (art > 250)] = np.nan
        except Exception:
            continue

        pt_wins = windows[windows["caseid"] == cid]
        for _, row in pt_wins.iterrows():
            ws_float = float(row["window_start_s"])
            ws = int(ws_float)
            we = int(float(row["window_end_s"]))
            seg = art[ws:we] if we <= len(art) else art[ws:]
            feats = extract_map_features(seg)
            feats["patient_id"]     = row["patient_id"]
            feats["event_type"]     = row["event_type"]
            feats["label"]          = int(row["label"])
            feats["window_start_s"] = ws_float   # keep original float for merge
            results.append(feats)

    return pd.DataFrame(results)


# ── evaluation ─────────────────────────────────────────────────────────────────
def auc_ci(y, x, B=1000, seed=0):
    """Pooled AUC with 1000-sample bootstrap CI."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); x = np.asarray(x)
    m = ~np.isnan(x) & ~np.isnan(y)
    y, x = y[m], x[m]
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    base = roc_auc_score(y, x)
    boots = [roc_auc_score(y[idx := rng.integers(0, len(y), len(y))],
                           x[idx]) for _ in range(B)]
    return base, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def compare_aucs(windows: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    """Build AUC comparison table: MAP features vs parsimonious autonomic features."""
    merged = windows.merge(
        map_df[["patient_id", "event_type", "window_start_s"] + MAP_FEATURES + ["map_end_mean", "map_start_mean"]],
        on=["patient_id", "event_type", "window_start_s"], how="left"
    )
    rows = []
    for outcome in OUTCOMES:
        sub = merged[merged["event_type"] == outcome].dropna(subset=["label"])
        y = sub["label"].astype(int).values

        # --- MAP features ---
        for f in MAP_FEATURES + ["map_end_mean", "map_start_mean"]:
            auc, lo, hi = auc_ci(y, sub[f].values)
            rows.append(dict(outcome=outcome, axis="MAP_signal", feature=f,
                             auc=round(auc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             n_event=int(y.sum()), n_control=int((y==0).sum())))

        # MAP composite (multivariate logistic on all MAP features)
        map_feats = [f for f in MAP_FEATURES if f in sub.columns]
        sub_map = sub[map_feats + ["label"]].dropna()
        if len(sub_map) > 10:
            sc = StandardScaler()
            Xm = sc.fit_transform(sub_map[map_feats].values)
            ym = sub_map["label"].astype(int).values
            lr = LogisticRegression(max_iter=1000).fit(Xm, ym)
            auc, lo, hi = auc_ci(ym, lr.predict_proba(Xm)[:, 1])
            rows.append(dict(outcome=outcome, axis="MAP_signal", feature="MAP_composite",
                             auc=round(auc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             n_event=int(ym.sum()), n_control=int((ym==0).sum())))

        # --- Autonomic parsimonious features (univariate) ---
        for f in PARSIMONIOUS[outcome]:
            if f not in sub.columns:
                continue
            auc, lo, hi = auc_ci(y, sub[f].values)
            rows.append(dict(outcome=outcome, axis="autonomic", feature=f,
                             auc=round(auc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             n_event=int(y.sum()), n_control=int((y==0).sum())))

        # Autonomic composite
        par_feats = [f for f in PARSIMONIOUS[outcome] if f in sub.columns]
        sub_par = sub[par_feats + ["label"]].dropna()
        if len(sub_par) > 10:
            sc = StandardScaler()
            Xa = sc.fit_transform(sub_par[par_feats].values)
            ya = sub_par["label"].astype(int).values
            lr = LogisticRegression(max_iter=1000).fit(Xa, ya)
            auc, lo, hi = auc_ci(ya, lr.predict_proba(Xa)[:, 1])
            rows.append(dict(outcome=outcome, axis="autonomic", feature="autonomic_composite",
                             auc=round(auc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             n_event=int(ya.sum()), n_control=int((ya==0).sum())))

        # MAP + Autonomic combined
        all_feats = [f for f in (map_feats + par_feats) if f in sub.columns]
        sub_all = sub[all_feats + ["label"]].dropna()
        if len(sub_all) > 10:
            sc = StandardScaler()
            Xc = sc.fit_transform(sub_all[all_feats].values)
            yc = sub_all["label"].astype(int).values
            lr = LogisticRegression(max_iter=1000).fit(Xc, yc)
            auc, lo, hi = auc_ci(yc, lr.predict_proba(Xc)[:, 1])
            rows.append(dict(outcome=outcome, axis="combined", feature="MAP+autonomic_composite",
                             auc=round(auc, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             n_event=int(yc.sum()), n_control=int((yc==0).sum())))

    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Force re-download even if cache exists")
    ap.add_argument("--out", default="results/supplementary/map_slope_auc_comparison.csv")
    args = ap.parse_args()

    windows = pd.read_parquet(WINDOWS_PATH)

    if CACHE_PATH.exists() and not args.rebuild:
        print(f"Loading cached MAP features from {CACHE_PATH}")
        map_df = pd.read_parquet(CACHE_PATH)
    else:
        print(f"Downloading ART_MBP for {windows['patient_id'].nunique()} patients…")
        map_df = build_map_features(windows)
        map_df.to_parquet(CACHE_PATH, index=False)
        print(f"Cached to {CACHE_PATH}  ({len(map_df)} rows)")

    print(f"\nMAP features extracted: {len(map_df)} windows, "
          f"{map_df['map_slope'].notna().sum()} with valid slope")

    print("\nComputing AUC comparison…")
    tab = compare_aucs(windows, map_df)

    pd.set_option("display.width", 180, "display.max_rows", 60)
    print("\n=== AUC COMPARISON: MAP signal vs autonomic features ===\n")
    for outcome in OUTCOMES:
        sub = tab[tab["outcome"] == outcome].sort_values(["axis", "auc"], ascending=[True, False])
        print(f"── {outcome.upper()} ──")
        print(sub[["axis", "feature", "auc", "ci_lo", "ci_hi",
                   "n_event", "n_control"]].to_string(index=False))
        print()

    tab.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")
