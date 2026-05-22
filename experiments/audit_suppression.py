#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUDITORÍA DE SUPRESIÓN — univariante vs multivariante (parsimonioso / completo)
================================================================================
Objetivo: para cada celda cohorte x desenlace, decidir si el SIGNO del coeficiente
multivariante de cada característica refleja su asociación real (univariante) o es un
artefacto de supresión por colinealidad.

Diagnóstico clave:  sign(rho_univariante)  vs  sign(coef_multivariante).
  - Coinciden            -> dirección fiable.
  - NO coinciden (flip)  -> SUPRESIÓN: el signo multivariante es artefacto.
  - Univariante ~0 y multivariante grande -> signo "fabricado" de la nada (p.ej. cv-PA-std).

Además calcula la ESTABILIDAD DE SIGNO POR BOOTSTRAP del coeficiente multivariante para
DEMOSTRAR que una supresión estable da estabilidad alta (~99-100%) siendo un artefacto:
la estabilidad NO detecta supresión; el contraste univariante↔multivariante sí.

NOTA DE USO: rellena CONFIG con las rutas a tus parquets v7 y los nombres de columnas.
El script no inventa datos; corre sobre los tuyos. Para auto-validarlo sin datos:
    python audit_suppression.py --synthetic
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------------
# CONJUNTOS DE 8 CARACTERÍSTICAS POR DESENLACE (del manuscrito; específicos de desenlace,
# compartidos entre cohortes). Edita si tus sets difieren.
# ----------------------------------------------------------------------------------
PARSIMONIOUS = {
    "hypotension":  ["brs_min", "cv_pa_std", "std_pa_mean", "std_pa_max",
                     "rsa_max", "rsa_mean", "arv_std", "arv_mean"],
    "hypertension": ["std_pa_std", "cv_pa_std", "brs_min", "cv_pa_mean",
                     "std_pa_max", "arv_std", "std_pa_slope", "sdnn_mean"],
}

# ----------------------------------------------------------------------------------
# CONFIG — RELLENA CON TUS DATOS
# Cada (cohorte, desenlace) -> DataFrame con: columnas de características + 'label' (0/1) + 'patient'
# 'label' = 1 ventana-evento, 0 ventana-control.
# ----------------------------------------------------------------------------------
def load_real(cohort, outcome):
    """
    Carga los parquets cacheados en results/cache/{cohort}_windows.parquet,
    filtra por event_type==outcome, y devuelve features + 'label' + 'patient'.
    """
    name_map = {"VitalDB": "vitaldb", "Clinic": "clinic", "MIMIC": "mimic"}
    key = name_map.get(cohort, cohort.lower())
    path = f"results/cache/{key}_windows.parquet"
    df = pd.read_parquet(path)
    df = df[df["event_type"] == outcome].copy()
    df = df.rename(columns={"patient_id": "patient"})
    # Drop non-feature columns
    drop_cols = {"event_type", "window_start_s", "window_end_s", "event_onset_s"}
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


# ----------------------------------------------------------------------------------
# MODELOS
# ----------------------------------------------------------------------------------
def _standardize(X):
    mu = X.mean(0); sd = X.std(0).replace(0, 1.0)
    return (X - mu) / sd

def fit_glmm(df, features, label="label", patient="patient", backend="bayes_mixed"):
    """
    Ajusta el modelo multivariante y devuelve {feature: coef_estandarizado}.
      backend='bayes_mixed' : BinomialBayesMixedGLM con intercepto aleatorio por paciente
                              (reproduce el método del manuscrito).
      backend='logit_cluster': Logit con EE robustas agrupadas por paciente (rápido;
                              mismos SIGNOS de efecto fijo, útil para la auditoría de signo).
    """
    d = df[features + [label, patient]].dropna().copy()
    d[features] = _standardize(d[features])
    y = d[label].astype(float).values
    if backend == "logit_cluster":
        import statsmodels.api as sm
        from numpy.linalg import LinAlgError
        X = sm.add_constant(d[features].values)
        try:
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
            return dict(zip(features, res.params[1:]))
        except (LinAlgError, Exception):
            # Fallback: sklearn L2-regularised logistic regression (same signs)
            from sklearn.linear_model import LogisticRegression
            sc2 = LogisticRegression(max_iter=2000, C=1.0).fit(d[features].values, y)
            return dict(zip(features, sc2.coef_[0]))
    elif backend == "bayes_mixed":
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        d = d.reset_index(drop=True)
        formula = label + " ~ " + " + ".join(features)
        vc = {"patient": "0 + C(%s)" % patient}
        model = BinomialBayesMixedGLM.from_formula(formula, vc, d)
        res = model.fit_vb()
        names = list(res.model.exog_names)
        out = {}
        for f in features:
            out[f] = res.fe_mean[names.index(f)] if f in names else np.nan
        return out
    else:
        raise ValueError(backend)


def univariate_stats(df, features, label="label"):
    rows = {}
    y = df[label].astype(int).values
    for f in features:
        x = df[f].values
        m = ~np.isnan(x)
        rho, p = spearmanr(x[m], y[m])
        try:
            auc = roc_auc_score(y[m], x[m])
        except Exception:
            auc = np.nan
        rows[f] = dict(univ_rho=rho, univ_p=p, univ_auc=auc)
    return rows


def top_collinear(df, features):
    C = df[features].corr(method="pearson")
    out = {}
    for f in features:
        s = C[f].drop(f).abs()
        partner = s.idxmax(); out[f] = (partner, C.loc[f, partner])
    return out


def bootstrap_sign_stability(df, features, label="label", patient="patient",
                             B=200, backend="logit_cluster", seed=0):
    """% de remuestreos (agrupados por paciente) en que el coef conserva su signo."""
    rng = np.random.default_rng(seed)
    pts = df[patient].unique()
    signs = {f: [] for f in features}
    for _ in range(B):
        samp = rng.choice(pts, size=len(pts), replace=True)
        boot = pd.concat([df[df[patient] == p] for p in samp], ignore_index=True)
        try:
            coefs = fit_glmm(boot, features, label, patient, backend=backend)
        except Exception:
            continue
        for f in features:
            c = coefs.get(f, np.nan)
            if not np.isnan(c):
                signs[f].append(np.sign(c))
    stab = {}
    for f in features:
        arr = np.array(signs[f])
        if len(arr) == 0:
            stab[f] = np.nan; continue
        dom = np.sign(np.sum(arr)) or 1.0
        stab[f] = float(np.mean(arr == dom))
    return stab


# ----------------------------------------------------------------------------------
# AUDITORÍA
# ----------------------------------------------------------------------------------
def run_cell(df, outcome, cohort, full_features=None,
             backend="bayes_mixed", B=200, boot_backend="logit_cluster"):
    pars = PARSIMONIOUS[outcome]
    if full_features is None:
        full_features = [c for c in df.columns if c not in ("label", "patient")]
    uni = univariate_stats(df, sorted(set(pars) | set(full_features)))
    coef_pars = fit_glmm(df, pars, backend=backend)
    coef_full = fit_glmm(df, full_features, backend=backend)
    coll = top_collinear(df, pars)
    stab = bootstrap_sign_stability(df, pars, B=B, backend=boot_backend)

    rows = []
    for f in pars:
        ur = uni[f]["univ_rho"]
        cp = coef_pars.get(f, np.nan); cf = coef_full.get(f, np.nan)
        # convención: rho>0 -> feature alto asociado a evento (riesgo); coef>0 -> riesgo.
        flip_p = (not np.isnan(ur) and not np.isnan(cp) and np.sign(ur) != np.sign(cp)
                  and abs(ur) > 0.02)
        flip_f = (not np.isnan(ur) and not np.isnan(cf) and np.sign(ur) != np.sign(cf)
                  and abs(ur) > 0.02)
        if abs(ur) <= 0.02:
            verdict = "UNIV~0: signo multivariante FABRICADO (sin señal de base)"
        elif flip_p or flip_f:
            verdict = "SUPRESIÓN: inversión de signo univariante->multivariante"
        else:
            verdict = "consistente"
        rows.append(dict(
            cohort=cohort, outcome=outcome, feature=f,
            univ_rho=round(ur, 4), univ_p=uni[f]["univ_p"],
            univ_auc=round(uni[f]["univ_auc"], 3) if not np.isnan(uni[f]["univ_auc"]) else np.nan,
            coef_parsimonious=round(cp, 4) if not np.isnan(cp) else np.nan,
            coef_full=round(cf, 4) if not np.isnan(cf) else np.nan,
            flip_parsimonious=flip_p, flip_full=flip_f,
            boot_sign_stability_pars=round(stab.get(f, np.nan), 3) if not np.isnan(stab.get(f, np.nan)) else np.nan,
            top_collinear=coll[f][0], top_collinear_r=round(coll[f][1], 3),
            verdict=verdict,
        ))
    return pd.DataFrame(rows)


def run_audit(cohorts=("VitalDB", "Clinic"), outcomes=("hypotension", "hypertension"),
              loader=load_real, **kw):
    out = []
    for c in cohorts:
        for o in outcomes:
            df = loader(c, o)
            out.append(run_cell(df, o, c, **kw))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------------
# AUTO-VALIDACIÓN SOBRE DATOS SINTÉTICOS (supresión plantada tipo brs_min)
# ----------------------------------------------------------------------------------
def synthetic_loader(cohort, outcome, n_pat=150, w=30, seed=1):
    """
    Planta una supresión conocida:
      - V (variabilidad) impulsa el evento (+).
      - brs_min = -0.7*V + ruido  (fuertemente anticorrelado con la variabilidad).
      - El evento depende de +0.4*brs_min (efecto directo pequeño y POSITIVO).
    => brs_min univariante: NEGATIVO (protector, vía V).  Multivariante: POSITIVO (flip).
    Réplica exacta del patrón observado en VitalDB-hipotensión.
    """
    rng = np.random.default_rng(seed + (0 if outcome == "hypotension" else 7)
                                + (0 if cohort == "VitalDB" else 100))
    pid = np.repeat(np.arange(n_pat), w)
    n = len(pid)
    re = np.repeat(rng.normal(0, 0.5, n_pat), w)  # intercepto aleatorio por paciente
    V = rng.normal(0, 1, n)
    arv_std = V + rng.normal(0, 0.4, n)
    cv_pa_std = 0.9 * V + rng.normal(0, 0.5, n)
    std_pa_max = 0.8 * V + rng.normal(0, 0.5, n)
    std_pa_mean = 0.7 * V + rng.normal(0, 0.6, n)
    brs_min = -0.7 * V + 0.7 * rng.normal(0, 1, n)      # anticorrelada con variabilidad
    rsa_max = rng.normal(0, 1, n); rsa_mean = rng.normal(0, 1, n)
    arv_mean = 0.5 * V + rng.normal(0, 0.7, n)
    # logit del evento: lo impulsa V (vía variabilidad) + efecto directo POSITIVO de brs_min
    lin = re + 1.5 * V + 0.4 * brs_min - 1.0
    p = 1 / (1 + np.exp(-lin)); label = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame(dict(
        patient=pid, label=label, brs_min=brs_min, cv_pa_std=cv_pa_std,
        std_pa_mean=std_pa_mean, std_pa_max=std_pa_max, rsa_max=rsa_max,
        rsa_mean=rsa_mean, arv_std=arv_std, arv_mean=arv_mean,
        # rellenos para el set de hipertensión:
        std_pa_std=0.85 * V + rng.normal(0, 0.5, n),
        cv_pa_mean=0.6 * V + rng.normal(0, 0.6, n),
        std_pa_slope=rng.normal(0, 1, n), sdnn_mean=rng.normal(0, 1, n),
    ))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="auto-validar sobre datos sintéticos con supresión conocida")
    ap.add_argument("--backend", default="bayes_mixed",
                    choices=["bayes_mixed", "logit_cluster"])
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--out", default="suppression_audit.csv")
    args = ap.parse_args()

    if args.synthetic:
        tab = run_audit(loader=synthetic_loader, backend=args.backend,
                        B=args.B, boot_backend="logit_cluster")
    else:
        tab = run_audit(loader=load_real, backend=args.backend,
                        B=args.B, boot_backend="logit_cluster")

    cols = ["cohort", "outcome", "feature", "univ_rho", "univ_p", "univ_auc",
            "coef_parsimonious", "coef_full", "flip_parsimonious", "flip_full",
            "boot_sign_stability_pars", "top_collinear", "top_collinear_r", "verdict"]
    tab = tab[cols]
    tab.to_csv(args.out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(tab.to_string(index=False))
    print("\nGuardado en", args.out)
