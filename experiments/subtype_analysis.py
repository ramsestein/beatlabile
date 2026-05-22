#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUBTIPOS DE HIPOTENSIÓN — ¿el deterioro barorreflejo (brs-min) marca la hipotensión
"mala" (descompensatoria) frente a la anestésica/iatrogénica?  [3 definiciones]
================================================================================
Encuadre: queremos una señal que SOBREVIVE a confusores (ventilación, fármacos) y que
aun así separa la inestabilidad mala. Aquí probamos si el EJE BARORREFLEJO separa
subtipos de hipotensión, con TRES definiciones de subtipo:
  - PRIMARIA   : vasoactivo/cambio anestésico en ventana previa  -> anestésica/iatrogénica
  - sensib. 1  : proximidad a la inducción (<=15 min)            -> anestésica
  - sensib. 2  : gravedad/refractariedad del evento              -> descompensatoria
Correr las tres = robustez de la DEFINICIÓN (no p-hacking) si declaras la primaria.

Para cada definición y eje (barorreflejo / variabilidad): Mann-Whitney, Cohen's d,
AUC de discriminación del subtipo y separación (|AUC-0.5|*2). Autovalidación:
    python subtype_analysis.py --synthetic
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

# Column names match results/cache/{cohort}_windows.parquet
BARO_AXIS = ["brs_min", "brs_mean", "brs_slope"]
VAR_AXIS  = ["cv_pa_std", "std_pa_std", "std_pa_max", "arv_std"]

# --- definiciones de subtipo (requieren tus anotaciones) -------------------------
def def_primary_vasopressor(df):
    iatro = (df.get("vasoactive_in_prewindow", 0).astype(bool)
             | df.get("anesthetic_bolus_in_prewindow", 0).astype(bool))
    return np.where(iatro, "anesthetic", "decompensatory")

def def_induction_proximity(df, minutes=15):
    return np.where(df["min_since_induction"] <= minutes, "anesthetic", "decompensatory")

def def_severity(df, map_floor=45, min_duration=10):
    decomp = (df["event_min_map"] <= map_floor) & (df["event_duration_min"] >= min_duration)
    return np.where(decomp, "decompensatory", "anesthetic")

DEFINITIONS = [("PRIMARIA: vasoactivo en ventana previa", def_primary_vasopressor),
               ("sensib.1: proximidad a inducción",        def_induction_proximity),
               ("sensib.2: gravedad del evento",           def_severity)]

# --- tests ------------------------------------------------------------------------
def feature_test(df, feat, sub, pos="decompensatory"):
    y = (sub == pos).astype(int)
    x = pd.to_numeric(df[feat], errors="coerce").values
    m = ~np.isnan(x); x, y = x[m], y[m]
    a, b = x[y == 1], x[y == 0]
    if len(a) < 3 or len(b) < 3: return None
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    sp = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)/(len(a)+len(b)-2))
    d = (a.mean()-b.mean())/sp if sp > 0 else np.nan
    auc = roc_auc_score(y, x); sep = abs(auc-0.5)*2
    return dict(feature=feat, median_decomp=round(np.median(a),4),
                median_anesth=round(np.median(b),4), cohens_d=round(d,3),
                auc_subtype=round(auc,3), separation=round(sep,3), p=p,
                n_decomp=int(len(a)), n_anesth=int(len(b)))

def run_definition(df, sub, label):
    rows = []
    for axis, feats in [("baroreflex", BARO_AXIS), ("variability", VAR_AXIS)]:
        for f in feats:
            if f in df.columns:
                r = feature_test(df, f, sub)
                if r: r.update(axis=axis, definition=label); rows.append(r)
    return pd.DataFrame(rows)

def run_all(df, definitions=None):
    if definitions is None:
        definitions = DEFINITIONS
    tabs = []
    for label, fn in definitions:
        try:
            sub = pd.Series(fn(df))
        except Exception as e:
            print(f"[saltada] {label}: faltan anotaciones ({e})"); continue
        t = run_definition(df, sub, label); tabs.append(t)
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()

# --- autovalidación sintética -----------------------------------------------------
def synthetic(n=400, seed=3):
    rng = np.random.default_rng(seed)
    dec = np.r_[np.ones(n//2), np.zeros(n-n//2)]               # subtipo latente
    rng.shuffle(dec)
    # brs-min más deteriorado en descompensatoria; variabilidad NO separa
    brs_min_min  = 0.50 - 0.34*dec + rng.normal(0, 0.16, n)
    brs_min_mean = brs_min_min + 0.10 + rng.normal(0, 0.08, n)
    brs_min_slope = -0.020*dec + rng.normal(0, 0.030, n)
    cv_pa_std_mean = 0.045 + 0.004*(1-dec) + rng.normal(0, 0.020, n)
    std_pa_std_mean = 5.0 + rng.normal(0, 2.0, n)
    std_pa_max_mean = 20.0 + rng.normal(0, 5.0, n)
    arv_std_mean = 3.0 + rng.normal(0, 1.0, n)
    # anotaciones (ruidosamente ligadas al subtipo latente)
    vaso = (rng.uniform(size=n) < np.where(dec==1, 0.20, 0.70)).astype(int)
    bolus = (rng.uniform(size=n) < np.where(dec==1, 0.15, 0.50)).astype(int)
    min_since_induction = np.where(dec==1, rng.uniform(20,120,n), rng.uniform(2,25,n))
    event_min_map = np.where(dec==1, rng.normal(40,5,n), rng.normal(50,5,n))
    event_duration_min = np.where(dec==1, rng.normal(15,5,n), rng.normal(6,3,n))
    return pd.DataFrame(dict(
        patient=rng.integers(0,120,n),
        brs_min_min=brs_min_min, brs_min_mean=brs_min_mean, brs_min_slope=brs_min_slope,
        cv_pa_std_mean=cv_pa_std_mean, std_pa_std_mean=std_pa_std_mean,
        std_pa_max_mean=std_pa_max_mean, arv_std_mean=arv_std_mean,
        vaso_active_in_prewindow=vaso, vasoactive_in_prewindow=vaso,
        anesthetic_bolus_in_prewindow=bolus, min_since_induction=min_since_induction,
        event_min_map=event_min_map, event_duration_min=event_duration_min))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="subtype_analysis_result.csv")
    args = ap.parse_args()
    if args.synthetic:
        df = synthetic()
    else:
        # --- VitalDB real data loader ---
        vdb = pd.read_parquet("results/cache/vitaldb_windows.parquet")
        df = vdb[vdb["event_type"] == "hypotension"].copy()
        df = df.rename(columns={"patient_id": "patient"})

        # Merge cases.csv for clinical annotations (vasopressors, ASA, emergency)
        cases = pd.read_csv(
            "datasets/data/vitaldb/clinical_data/cases.csv",
            usecols=["caseid", "asa", "emop", "intraop_eph", "intraop_phe"],
        )
        df["caseid"] = pd.to_numeric(df["patient"].str.lstrip("0"), errors="coerce")
        df = df.merge(cases, on="caseid", how="left")

        # Proxy annotations -------------------------------------------------------
        # def_primary: patient received intraop vasopressors → iatrogénica/anestésica
        df["vasoactive_in_prewindow"] = (
            (df["intraop_eph"].fillna(0) > 0) | (df["intraop_phe"].fillna(0) > 0)
        ).astype(int)
        df["anesthetic_bolus_in_prewindow"] = 0  # not available per-window

        # def_induction_proximity: time since recording start ≈ time since induction
        df["min_since_induction"] = df["window_start_s"] / 60

        # def_severity (VitalDB): ASA ≥ 3 OR emergency operation → decompensatoria
        def def_severity_vitaldb(df_inner, **kwargs):
            decomp = (
                (df_inner["asa"].fillna(0) >= 3)
                | (df_inner["emop"].fillna(0) == 1)
            )
            return np.where(decomp, "decompensatory", "anesthetic")

        vdb_defs = [
            ("PRIMARIA: vasopressor intraop (case-level)",  def_primary_vasopressor),
            ("sensib.1: proximidad a inducción (≤15 min)",  def_induction_proximity),
            ("sensib.2: ASA≥3 o cirugía urgente",           def_severity_vitaldb),
        ]

        tab = run_all(df, definitions=vdb_defs)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    cols = ["definition","axis","feature","median_decomp","median_anesth",
            "cohens_d","auc_subtype","separation","p","n_decomp","n_anesth"]
    print(tab[cols].to_string(index=False))
    print("\n=== Separación media por DEFINICIÓN x EJE (mayor = separa mejor) ===")
    print(tab.groupby(["definition","axis"])["separation"].mean().round(3).to_string())
    tab.to_csv(args.out, index=False); print("\nGuardado en", args.out)
