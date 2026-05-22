#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DETERIORO EVENTO-vs-CONTROL ESTANDARIZADO — ¿hay gradiente entre cohortes?
================================================================================
Mide el tamaño de efecto del deterioro pre-evento de cada EJE (barorreflejo,
variabilidad), comparándolo entre cohortes de forma INDEPENDIENTE DEL HARDWARE:
Cohen's d y AUC son adimensionales, así que un brs en unidades distintas entre
Clínic y VitalDB no contamina la comparación (se demuestra abajo aplicando una
transformación afín distinta por cohorte y viendo que el d no cambia).

Hipótesis (el gradiente fisiológico que queremos):
  - HIPOTENSIÓN: el deterioro evento-vs-control es MAYOR en Clínic que en VitalDB
                 (la implicación autonómica escala con la acuidad).
  - HIPERTENSIÓN: el deterioro es IGUAL entre cohortes (precursor conservado).

Salida: por cohorte×desenlace, d y AUC (orientados a "deterioro", positivo=peor)
por característica y por eje, con IC95% por bootstrap agrupado por paciente; y un
TEST DE GRADIENTE (Clínic − VitalDB) por desenlace×eje con IC95% bootstrap.

Bloqueos NO entra aquí: es otro constructo (PTT, en torno a nocicepción, sin
ventanas evento/control). Queda como ancla cualitativa ("sin deterioro en sano").

Autovalidación:  python delta_event_control.py --synthetic
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ejes (ajusta nombres a tus columnas). orient: 'baro' (bajo=deterioro) | 'var' (alto=deterioro)
AXES = {
    "baroreflex":  (["brs_min", "brs_mean", "brs_slope"], "baro"),
    "variability": (["cv_pa_std", "std_pa_std", "std_pa_max", "arv_std"], "var"),
}

def _cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan

def _orient(df, feats, orient):
    """z-score DENTRO de la cohorte y orienta a deterioro (mayor=peor)."""
    z = (df[feats] - df[feats].mean()) / df[feats].std(ddof=0).replace(0, 1.0)
    return -z if orient == "baro" else z          # baro: bajo=peor -> invierte

def _axis_score(df, feats, orient):
    return _orient(df, feats, orient).mean(axis=1)

def effect(df, feats, orient, label="label"):
    """d y AUC del eje (evento vs control), orientados a deterioro."""
    s = _axis_score(df, feats, orient).values
    y = df[label].astype(int).values
    m = ~np.isnan(s); s, y = s[m], y[m]
    ev, ct = s[y == 1], s[y == 0]
    d = _cohens_d(pd.Series(ev), pd.Series(ct))
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    return d, auc, int((y == 1).sum()), int((y == 0).sum())

def feature_effects(df, label="label"):
    rows = []
    for axis, (feats, orient) in AXES.items():
        for f in feats:
            if f not in df.columns: continue
            x = _orient(df, [f], orient)[f].values
            y = df[label].astype(int).values
            m = ~np.isnan(x); x, y = x[m], y[m]
            d = _cohens_d(pd.Series(x[y == 1]), pd.Series(x[y == 0]))
            auc = roc_auc_score(y, x) if len(np.unique(y)) == 2 else np.nan
            rows.append(dict(axis=axis, feature=f, d_deterioration=round(d, 3),
                             auc=round(auc, 3)))
    return pd.DataFrame(rows)

def boot_axis_d(df, feats, orient, label="label", patient="patient", B=1000, seed=0):
    """distribución bootstrap (agrupado por paciente) del d del eje."""
    rng = np.random.default_rng(seed)
    pts = df[patient].unique()
    idx = {p: df.index[df[patient] == p].to_numpy() for p in pts}
    ds = []
    for _ in range(B):
        samp = rng.choice(pts, size=len(pts), replace=True)
        rows = np.concatenate([idx[p] for p in samp])
        d, *_ = effect(df.loc[rows], feats, orient, label)
        if not np.isnan(d): ds.append(d)
    return np.array(ds)

def run(loader, cohorts=("Clinic", "VitalDB"),
        outcomes=("hypotension", "hypertension"), B=1000):
    cell, boot = [], {}
    for c in cohorts:
        for o in outcomes:
            df = loader(c, o)
            for axis, (feats, orient) in AXES.items():
                d, auc, ne, nc = effect(df, feats, orient)
                bd = boot_axis_d(df, feats, orient, B=B)
                lo, hi = np.percentile(bd, [2.5, 97.5])
                boot[(c, o, axis)] = bd
                cell.append(dict(cohort=c, outcome=o, axis=axis,
                                 d_deterioration=round(d, 3),
                                 ci95=f"[{lo:.2f}, {hi:.2f}]", auc=round(auc, 3),
                                 n_event=ne, n_control=nc))
    cells = pd.DataFrame(cell)
    # ---- TEST DE GRADIENTE: Clínic − VitalDB por desenlace×eje ----
    grad = []
    for o in outcomes:
        for axis in AXES:
            bc, bv = boot.get(("Clinic", o, axis)), boot.get(("VitalDB", o, axis))
            if bc is None or bv is None: continue
            k = min(len(bc), len(bv))
            diff = bc[:k] - bv[:k]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())   # bootstrap 2-colas
            grad.append(dict(outcome=o, axis=axis,
                             clinic_minus_vitaldb=round(diff.mean(), 3),
                             ci95=f"[{lo:.2f}, {hi:.2f}]", p_boot=round(p, 3),
                             gradient=("Clínic>VitalDB" if lo > 0 else
                                       "VitalDB>Clínic" if hi < 0 else "no concluyente")))
    return cells, pd.DataFrame(grad)

# ----------------------------------------------------------------------------------
# AUTOVALIDACIÓN SINTÉTICA (planta gradiente en hipo, plano en hiper; + hardware afín)
# ----------------------------------------------------------------------------------
def synthetic_loader(cohort, outcome, n_pat=120, w=30, seed=7):
    rng = np.random.default_rng(seed + hash((cohort, outcome)) % 9999)
    # tamaño de efecto plantado (baro, var) evento-vs-control
    d_map = {("Clinic", "hypotension"): (0.80, 0.90),
             ("VitalDB", "hypotension"): (0.40, 0.50),
             ("Clinic", "hypertension"): (0.60, 0.70),
             ("VitalDB", "hypertension"): (0.62, 0.68)}
    d_baro, d_var = d_map[(cohort, outcome)]
    pid = np.repeat(np.arange(n_pat), w); n = len(pid)
    ev = (rng.uniform(size=n) < 0.30)
    baro_lat = rng.normal(0, 1, n) + d_baro * ev   # mayor = más deterioro
    var_lat = rng.normal(0, 1, n) + d_var * ev
    df = pd.DataFrame(dict(
        brs_min=-baro_lat + rng.normal(0, 0.5, n),         # brs bajo = deterioro
        brs_mean=-baro_lat + rng.normal(0, 0.6, n),
        brs_slope=-0.3*baro_lat + rng.normal(0, 0.8, n),
        cv_pa_std=var_lat + rng.normal(0, 0.5, n),
        std_pa_std=var_lat + rng.normal(0, 0.6, n),
        std_pa_max=0.9*var_lat + rng.normal(0, 0.6, n),
        arv_std=0.8*var_lat + rng.normal(0, 0.7, n)))
    # SIMULAR HARDWARE DISTINTO: transformación afín por cohorte (el d debe ser inmune)
    scale, offset = (2.0, 10.0) if cohort == "Clinic" else (0.5, -3.0)
    df = df * scale + offset
    df["label"] = ev.astype(int); df["patient"] = pid
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default="delta_event_control.csv")
    args = ap.parse_args()
    def real_loader(cohort, outcome):
        name_map = {"VitalDB": "vitaldb", "Clinic": "clinic", "MIMIC": "mimic"}
        key = name_map.get(cohort, cohort.lower())
        df = pd.read_parquet(f"results/cache/{key}_windows.parquet")
        df = df[df["event_type"] == outcome].copy()
        df = df.rename(columns={"patient_id": "patient"})
        drop = {"event_type", "window_start_s", "window_end_s", "event_onset_s"}
        return df.drop(columns=[c for c in drop if c in df.columns])

    loader = synthetic_loader if args.synthetic else real_loader
    cohorts = ("Clinic", "VitalDB")
    cells, grad = run(loader, cohorts=cohorts, B=args.B)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== DETERIORO POR COHORTE × DESENLACE × EJE (d evento-vs-control, +=peor) ===")
    print(cells.to_string(index=False))
    print("\n=== TEST DE GRADIENTE (Clínic − VitalDB) ===")
    print(grad.to_string(index=False))
    cells.to_csv(args.out, index=False)
    grad.to_csv(args.out.replace(".csv", "_gradient.csv"), index=False)
    print("\nGuardado en", args.out)
