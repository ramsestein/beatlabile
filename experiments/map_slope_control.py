#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAP SLOPE COMO CONTROL PREDICTIVO — ¿Cuánto del AUC es ya visible en la tendencia MAP?
========================================================================================
Experimento de control: calcula features de la MAP cruda a 1 Hz sobre la ventana de
predicción [window_start_s, window_end_s] para cada ventana de VitalDB y Clínic.
Compara el potencial predictivo (AUC) de las features de MAP vs las features autonómicas
del modelo parsimónico, usando el test de DeLong pareado (mismos pacientes).

VitalDB: Solar8000/ART_MBP descargado vía API (vitaldb.load_case)
Clínic : Intellivue/ABP_MEAN de ficheros .vital locales (vitaldb.VitalFile)

Features de MAP calculadas:
  map_mean         : media de MAP en la ventana (nivel basal)
  map_slope        : pendiente lineal (mmHg/min)
  map_slope_last10 : pendiente en los últimos 10 min
  map_std          : desviación estándar de MAP
  map_end_mean     : media de MAP en los últimos 5 min
  map_start_mean   : media de MAP en los primeros 5 min
  map_end_vs_start : map_end_mean - map_start_mean

Modelos compuestos — validación cruzada estratificada:
  Los tres composites (MAP-solo, autonómico, MAP+variabilidad) se producen con
  cross_val_predict + StratifiedKFold(5) con StandardScaler dentro de cada fold
  (sin data leakage). Los scores out-of-fold se usan para AUC, CI bootstrap y
  el test de DeLong pareado.

Test de DeLong pareado (Sun & Xu 2014 FastDeLong):
  H0: AUC(MAP+variabilidad) == AUC(MAP-solo) en las mismas N ventanas.
  Los dos vectores de score provienen de los mismos folds CV → ROC correlacionadas
  → test pareado correcto para ΔAUC.

Resultados (AUC fuera de muestra, 5-fold CV):
  VitalDB — Hipotensi.  MAP-solo=0.700  MAP+var=0.791  ΔAUC=+0.09  p<0.0001
  VitalDB — Hipertensi. MAP-solo=0.728  MAP+var=0.837  ΔAUC=+0.11  p=0.0004
  Clínic  — Hipotensi.  MAP-solo=0.546  MAP+var=0.757  ΔAUC=+0.21  p<0.0001
  Clínic  — Hipertensi. MAP-solo=0.772  MAP+var=0.941  ΔAUC=+0.17  p=0.0005

Outputs:
  results/supplementary/map_slope_features_{vitaldb,clinic}.parquet
  results/supplementary/map_slope_auc_comparison_v2.csv
  results/supplementary/map_slope_delong_v2.csv
"""

import argparse
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

VITALDB_CACHE  = Path("results/supplementary/map_slope_features_vitaldb.parquet")
CLINIC_CACHE   = Path("results/supplementary/map_slope_features_clinic.parquet")
VITALDB_WINS   = Path("results/cache/vitaldb_windows.parquet")
CLINIC_WINS    = Path("results/cache/clinic_windows.parquet")
CLINIC_ROOT    = Path("clinic")
OUTCOMES       = ["hypotension", "hypertension"]

# Clinic MAP track (Intellivue monitor)
CLINIC_MAP_TRACK = "Intellivue/ABP_MEAN"

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

# ── download & compute (VitalDB) ───────────────────────────────────────────────
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


# ── clinic MAP feature builder ─────────────────────────────────────────────────
def _build_patient_file_index(clinic_root: Path) -> dict:
    """Return {patient_id: [path, ...]} mapping for all .vital files under clinic_root.

    Deduplicates by filename (same file may appear in multiple box subdirectories).
    Patient ID is identified as a 6-digit directory component in the path.
    """
    all_vitals = glob.glob(str(clinic_root / "**" / "*.vital"), recursive=True)
    seen: set = set()
    patient_files: dict = {}
    for v in all_vitals:
        name = Path(v).name
        if name in seen:
            continue
        seen.add(name)
        for part in Path(v).parts:
            if len(part) == 6 and part.isdigit():
                patient_files.setdefault(part, []).append(v)
                break
    return patient_files


def build_map_features_clinic(windows: pd.DataFrame, clinic_root: Path = CLINIC_ROOT,
                               verbose=True) -> pd.DataFrame:
    """Load Intellivue/ABP_MEAN from local .vital files and compute MAP features."""
    import vitaldb as vdb

    patient_files = _build_patient_file_index(clinic_root)
    results = []
    pids = windows["patient_id"].unique()
    n = len(pids)

    for i, pid in enumerate(pids):
        if verbose and (i % 50 == 0 or i == 0):
            print(f"  [{i+1}/{n}] patient {pid}", flush=True)

        files = patient_files.get(str(pid), [])
        if not files:
            continue

        # Load ABP_MEAN 1-Hz arrays for every file of this patient
        file_arrays: list = []
        for fp in files:
            try:
                vf = vdb.VitalFile(fp)
                if CLINIC_MAP_TRACK not in vf.get_track_names():
                    continue
                arr = vf.to_numpy(CLINIC_MAP_TRACK, interval=1.0)
                arr = np.asarray(arr).ravel().astype(float)
                arr[(arr < 20) | (arr > 250)] = np.nan
                if np.sum(~np.isnan(arr)) > 60:   # skip near-empty files
                    file_arrays.append(arr)
            except Exception:
                continue

        if not file_arrays:
            continue

        # Sort by descending length so we try the longest file first
        file_arrays.sort(key=len, reverse=True)

        pt_wins = windows[windows["patient_id"] == pid]
        for _, row in pt_wins.iterrows():
            ws_float = float(row["window_start_s"])
            ws = int(ws_float)
            we = int(float(row["window_end_s"]))
            win_len = we - ws

            seg = None
            for arr in file_arrays:
                if ws >= len(arr):
                    continue
                candidate = arr[ws: min(we, len(arr))]
                if np.sum(~np.isnan(candidate)) >= 0.5 * win_len:
                    seg = candidate
                    break

            if seg is None:
                continue

            feats = extract_map_features(seg)
            feats["patient_id"]     = row["patient_id"]
            feats["event_type"]     = row["event_type"]
            feats["label"]          = int(row["label"])
            feats["window_start_s"] = ws_float
            results.append(feats)

    return pd.DataFrame(results)


# ── DeLong paired AUC test ─────────────────────────────────────────────────────
def delong_paired_test(y_true, prob1, prob2):
    """Paired DeLong test: H\u2080: AUC(prob1) = AUC(prob2) on identical observations.

    Implements the FastDeLong estimator (Sun & Xu 2014, IEEE Signal Proc.).
    Returns (auc1, auc2, delta_auc, z, p_two_sided).
    """
    y = np.asarray(y_true, dtype=int)
    p1 = np.asarray(prob1, dtype=float)
    p2 = np.asarray(prob2, dtype=float)
    mask = ~np.isnan(p1) & ~np.isnan(p2) & (y == y)  # drops NaN labels too
    y, p1, p2 = y[mask], p1[mask], p2[mask]

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    def _placements(probs):
        """Return (auc, V10, V01) using vectorised kernel."""
        p_pos = probs[pos]          # (n_pos,)
        p_neg = probs[neg]          # (n_neg,)
        diff = p_pos[:, None] - p_neg[None, :]          # (n_pos, n_neg)
        psi = (diff > 0).astype(float) + 0.5 * (diff == 0).astype(float)
        V10 = psi.mean(axis=1)      # mean over controls for each case
        V01 = psi.mean(axis=0)      # mean over cases for each control
        return psi.mean(), V10, V01

    auc1, V10_1, V01_1 = _placements(p1)
    auc2, V10_2, V01_2 = _placements(p2)

    # Covariance structure of (AUC1 - AUC2)
    var1  = np.var(V10_1, ddof=1) / n_pos + np.var(V01_1, ddof=1) / n_neg
    var2  = np.var(V10_2, ddof=1) / n_pos + np.var(V01_2, ddof=1) / n_neg
    cov12 = (np.cov(V10_1, V10_2, ddof=1)[0, 1] / n_pos +
             np.cov(V01_1, V01_2, ddof=1)[0, 1] / n_neg)

    var_delta = var1 + var2 - 2 * cov12
    if var_delta <= 0:
        return auc1, auc2, auc1 - auc2, np.nan, np.nan

    z = (auc1 - auc2) / np.sqrt(var_delta)
    p = float(2 * stats.norm.sf(abs(z)))
    return float(auc1), float(auc2), float(auc1 - auc2), float(z), p


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


def _cv_scores(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
               seed: int = 0) -> np.ndarray:
    """Return out-of-fold P(y=1) from stratified k-fold CV.

    StandardScaler is fitted inside each fold to prevent data leakage.
    With small event counts (<50) the fold size is small but the estimate
    is unbiased; use the same CV scheme for all composites so the three
    score vectors align row-by-row for the DeLong paired test.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=seed))
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def compare_aucs(windows: pd.DataFrame, map_df: pd.DataFrame,
                 cohort: str = "cohort", B: int = 500, cv_folds: int = 5) -> pd.DataFrame:
    """AUC comparison table with CROSS-VALIDATED composites and paired DeLong test.

    All three composite scores (MAP-solo, autonomic, MAP+variabilidad) are
    produced by 5-fold stratified CV so absolute AUC values are out-of-sample.
    The DeLong test compares the two CV score vectors on the SAME observations,
    preserving the paired (correlated-ROC) structure.
    """
    merged = windows.merge(
        map_df[["patient_id", "event_type", "window_start_s"]
               + MAP_FEATURES + ["map_end_mean", "map_start_mean"]],
        on=["patient_id", "event_type", "window_start_s"], how="left"
    )
    rows = []
    delong_rows = []

    for outcome in OUTCOMES:
        sub = merged[merged["event_type"] == outcome].dropna(subset=["label"])
        y_all = sub["label"].astype(int).values

        # --- Univariate MAP features ---
        for f in MAP_FEATURES + ["map_end_mean", "map_start_mean"]:
            auc, lo, hi = auc_ci(y_all, sub[f].values, B=B)
            rows.append(dict(cohort=cohort, outcome=outcome, axis="MAP_signal",
                             feature=f, auc=round(auc, 3),
                             ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             n_event=int(y_all.sum()),
                             n_control=int((y_all == 0).sum())))

        # --- Univariate autonomic features ---
        for f in PARSIMONIOUS[outcome]:
            if f not in sub.columns:
                continue
            auc, lo, hi = auc_ci(y_all, sub[f].values, B=B)
            rows.append(dict(cohort=cohort, outcome=outcome, axis="autonomic",
                             feature=f, auc=round(auc, 3),
                             ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                             n_event=int(y_all.sum()),
                             n_control=int((y_all == 0).sum())))

        # ── Composite models on the COMMON observation set ──────────────────
        map_feats = [f for f in MAP_FEATURES if f in sub.columns]
        par_feats = [f for f in PARSIMONIOUS[outcome] if f in sub.columns]
        all_feats = map_feats + par_feats

        # Common set: windows where BOTH map AND autonomic features are present
        sub_common = sub[all_feats + ["label"]].dropna()
        if len(sub_common) < 20:
            continue
        y_c = sub_common["label"].astype(int).values
        if len(np.unique(y_c)) < 2:
            continue

        # ── 5-fold CV composites (out-of-sample scores, no data leakage) ────
        # All three models use the same CV splits so rows align for DeLong.
        score_map  = _cv_scores(sub_common[map_feats].values,  y_c, cv_folds)
        score_aut  = _cv_scores(sub_common[par_feats].values,  y_c, cv_folds)
        score_comb = _cv_scores(sub_common[all_feats].values,  y_c, cv_folds)

        auc_m,    lo_m,    hi_m    = auc_ci(y_c, score_map,  B=B)
        auc_a,    lo_a,    hi_a    = auc_ci(y_c, score_aut,  B=B)
        auc_comb, lo_comb, hi_comb = auc_ci(y_c, score_comb, B=B)

        rows.append(dict(cohort=cohort, outcome=outcome, axis="MAP_signal",
                         feature="MAP_composite", auc=round(auc_m, 3),
                         ci_lo=round(lo_m, 3), ci_hi=round(hi_m, 3),
                         n_event=int(y_c.sum()), n_control=int((y_c == 0).sum())))
        rows.append(dict(cohort=cohort, outcome=outcome, axis="autonomic",
                         feature="autonomic_composite", auc=round(auc_a, 3),
                         ci_lo=round(lo_a, 3), ci_hi=round(hi_a, 3),
                         n_event=int(y_c.sum()), n_control=int((y_c == 0).sum())))
        rows.append(dict(cohort=cohort, outcome=outcome, axis="combined",
                         feature="MAP+autonomic_composite", auc=round(auc_comb, 3),
                         ci_lo=round(lo_comb, 3), ci_hi=round(hi_comb, 3),
                         n_event=int(y_c.sum()), n_control=int((y_c == 0).sum())))

        # ── DeLong paired test (primary) ─────────────────────────────────────
        # H0: AUC(MAP+variabilidad) == AUC(MAP-solo) on the SAME N windows.
        # Both score vectors come from the same CV folds on the same rows, so
        # the ROC curves are correlated — correct paired test (Sun & Xu 2014).
        # Absolute AUC values are out-of-sample (5-fold CV).
        # δAUC > 0 means autonomic variables add information above raw MAP.
        a1, a2, delta, z, p = delong_paired_test(y_c, score_comb, score_map)
        delong_rows.append(dict(
            cohort=cohort, outcome=outcome,
            model_A="MAP+variabilidad", model_B="MAP_solo",
            auc_A=round(a1, 3), auc_B=round(a2, 3),
            delta_AUC=round(delta, 3), z=round(z, 3),
            p_delong=round(p, 4), n=len(y_c),
            n_event=int(y_c.sum())
        ))

    return pd.DataFrame(rows), pd.DataFrame(delong_rows)


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Force rebuild even if cache exists")
    ap.add_argument("--cohorts", nargs="+", default=["vitaldb", "clinic"],
                    choices=["vitaldb", "clinic"])
    ap.add_argument("--out", default="results/supplementary/map_slope_auc_comparison_v2.csv")
    ap.add_argument("--out-delong",
                    default="results/supplementary/map_slope_delong_v2.csv")
    args = ap.parse_args()

    all_auc_rows, all_delong_rows = [], []

    for cohort in args.cohorts:
        print(f"\n{'='*60}")
        print(f"  COHORT: {cohort.upper()}")
        print(f"{'='*60}")

        if cohort == "vitaldb":
            wins_path  = VITALDB_WINS
            cache_path = VITALDB_CACHE
        else:
            wins_path  = CLINIC_WINS
            cache_path = CLINIC_CACHE

        windows = pd.read_parquet(wins_path)
        print(f"Windows: {len(windows)} rows, "
              f"{windows['patient_id'].nunique()} patients")

        if cache_path.exists() and not args.rebuild:
            print(f"Loading cached MAP features from {cache_path}")
            map_df = pd.read_parquet(cache_path)
        else:
            if cohort == "vitaldb":
                print(f"Downloading ART_MBP for "
                      f"{windows['patient_id'].nunique()} patients…")
                map_df = build_map_features(windows)
            else:
                print(f"Loading ABP_MEAN from local .vital files for "
                      f"{windows['patient_id'].nunique()} patients…")
                map_df = build_map_features_clinic(windows)

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            map_df.to_parquet(cache_path, index=False)
            print(f"Cached to {cache_path}  ({len(map_df)} rows)")

        print(f"MAP features: {len(map_df)} windows, "
              f"{map_df['map_slope'].notna().sum()} with valid slope")

        print("Computing AUC comparison + DeLong tests (5-fold CV composites)…")
        tab, delong = compare_aucs(windows, map_df, cohort=cohort)
        all_auc_rows.append(tab)
        all_delong_rows.append(delong)

        pd.set_option("display.width", 200, "display.max_rows", 80)
        print(f"\n── AUC table ({cohort}) ──")
        for outcome in OUTCOMES:
            sub = (tab[tab["outcome"] == outcome]
                   .sort_values(["axis", "auc"], ascending=[True, False]))
            print(f"  {outcome.upper()}:")
            print(sub[["axis", "feature", "auc", "ci_lo", "ci_hi",
                        "n_event", "n_control"]].to_string(index=False))
            print()

        print(f"── DeLong paired tests ({cohort}) ──")
        print(delong[["outcome", "model_A", "model_B", "auc_A", "auc_B",
                       "delta_AUC", "z", "p_delong", "n"]].to_string(index=False))

    # Combine and save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_auc_rows, ignore_index=True).to_csv(args.out, index=False)
    print(f"\nAUC table saved → {args.out}")
    pd.concat(all_delong_rows, ignore_index=True).to_csv(args.out_delong, index=False)
    print(f"DeLong table saved → {args.out_delong}")
