"""
q1_stats.py
===========
PASO 5 — Tests confirmatorios pre-especificados.

Tests primarios (7, Bonferroni α=0.05/7≈0.00714):
  GLMM: feature ~ window_type + (1|patient_id) + (1|event_subcategory)
  Test one-sided según dirección pre-especificada.

Análisis de gradiente (secundario):
  feature ~ window_type * group + (1|patient_id) + (1|event_subcategory)

Análisis de sensibilidad (suplementario):
  1. Sin (1|event_subcategory)
  2. Solo estímulo "anclaje"
  3. Excluyendo paciente parcial 70767707
  4. Con covariables: edad, BMI, HTA, posición, grupo

Output: test_results.csv
"""

from __future__ import annotations

import logging
import warnings as pywarnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as sp_stats

from q1_config import (
    ALPHA_BONFERRONI,
    ALPHA_NOMINAL,
    EXPLORATORY_FEATURES,
    GLOBAL_SEED,
    N_PRIMARY_TESTS,
    PARCIAL,
    PRIMARY_FEATURES,
    RESULTS_DIR,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GLMM via statsmodels MixedLM
# ---------------------------------------------------------------------------

def _fit_mixedlm(
    df: pd.DataFrame,
    feature_col: str,
    random_effects: list[str],
    extra_covariates: list[str] | None = None,
) -> Optional[object]:
    """
    Ajusta Mixed LM:
      feature_col ~ window_type_bin [+ covariates] + random_effects

    window_type_bin: 1=event, 0=control
    random_effects: lista de columnas (e.g. ['patient_id', 'event_subcategory'])

    Returns: fitted model result o None si falla.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels no instalado") from exc

    work = df[
        [feature_col, "window_type", "patient_id", "event_subcategory"] +
        (extra_covariates or [])
    ].dropna(subset=[feature_col]).copy()

    if len(work) < 6:
        log.debug("Muy pocos datos para %s (n=%d)", feature_col, len(work))
        return None

    work["window_type_bin"] = (work["window_type"] == "event").astype(float)

    # Construir fórmula
    cov_str = ""
    if extra_covariates:
        cov_str = " + " + " + ".join(extra_covariates)
    formula = f"{feature_col} ~ window_type_bin{cov_str}"

    # Random effects: primero patient_id, luego event_subcategory si está en lista
    groups_col = "patient_id"
    re_formula  = None

    try:
        with pywarnings.catch_warnings():
            pywarnings.simplefilter("ignore")
            if "event_subcategory" in random_effects:
                # Nested random effects: add event_subcategory as variance component
                # statsmodels MixedLM soporta un solo re_formula adicional
                vcf = {"event_subcategory": "0 + C(event_subcategory)"}
                model = smf.mixedlm(
                    formula, data=work,
                    groups=work[groups_col],
                    vc_formula=vcf if work["event_subcategory"].nunique() > 1 else None,
                )
            else:
                model = smf.mixedlm(formula, data=work, groups=work[groups_col])
            result = model.fit(method="lbfgs", reml=False)
        return result
    except Exception as exc:
        log.debug("MixedLM falló para %s: %s", feature_col, exc)
        return None


def _one_sided_p(result, coef_name: str, direction: int) -> tuple[float, float, float, float]:
    """
    Extrae β, SE, IC 95% y p one-sided del coeficiente de interés.
    direction: +1 → H1: β > 0, -1 → H1: β < 0.
    Returns (beta, ci_lo, ci_hi, p_onesided).
    """
    try:
        beta = float(result.params[coef_name])
        se   = float(result.bse[coef_name])
        ci   = result.conf_int(alpha=0.05).loc[coef_name]
        ci_lo, ci_hi = float(ci.iloc[0]), float(ci.iloc[1])
        # Two-sided p → one-sided
        p_two = float(result.pvalues[coef_name])
        z = beta / (se + 1e-12)
        if direction > 0:
            # H1: β > 0 → p = P(Z > z)
            p_one = float(sp_stats.norm.sf(z))
        else:
            # H1: β < 0 → p = P(Z < z)
            p_one = float(sp_stats.norm.cdf(z))
        return beta, ci_lo, ci_hi, p_one
    except Exception as exc:
        log.debug("Error extrayendo coeficiente %s: %s", coef_name, exc)
        return np.nan, np.nan, np.nan, np.nan


# ---------------------------------------------------------------------------
# Veredicto
# ---------------------------------------------------------------------------

def _verdict(p_corr: float, beta: float, direction: int) -> str:
    obs_dir = np.sign(beta) if not np.isnan(beta) else 0
    if np.isnan(p_corr) or np.isnan(beta):
        return "indeterminado"
    if obs_dir != direction and obs_dir != 0:
        return "inconsistente"
    if p_corr < ALPHA_NOMINAL:
        return "validado"
    # CI cruza ampliamente cero → indeterminado
    return "no_validado"


# ---------------------------------------------------------------------------
# Tests primarios
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    feature:      str
    aggregation:  str
    col_name:     str
    direction:    int
    beta:         float = np.nan
    ci_lo:        float = np.nan
    ci_hi:        float = np.nan
    p_onesided:   float = np.nan
    p_bonferroni: float = np.nan
    direction_obs: str  = ""
    verdict:       str  = "indeterminado"
    n_event:       int  = 0
    n_control:     int  = 0
    analysis:      str  = "primary"


def run_primary_tests(event_windows: pd.DataFrame) -> list[TestResult]:
    """Corre los 7 tests primarios con GLMM y Bonferroni."""
    results: list[TestResult] = []

    for feat, agg, direction in PRIMARY_FEATURES:
        col = f"{feat}__{agg}"
        if col not in event_windows.columns:
            log.warning("Columna %s no encontrada en event_windows", col)
            results.append(TestResult(
                feature=feat, aggregation=agg, col_name=col,
                direction=direction, verdict="indeterminado",
                analysis="primary",
            ))
            continue

        n_ev  = int((event_windows["window_type"] == "event").sum())
        n_ct  = int((event_windows["window_type"] == "control").sum())

        res = _fit_mixedlm(
            event_windows, col,
            random_effects=["patient_id", "event_subcategory"],
        )
        if res is None:
            results.append(TestResult(
                feature=feat, aggregation=agg, col_name=col,
                direction=direction, verdict="indeterminado",
                n_event=n_ev, n_control=n_ct, analysis="primary",
            ))
            continue

        beta, ci_lo, ci_hi, p_one = _one_sided_p(res, "window_type_bin", direction)
        p_bonf = min(1.0, p_one * N_PRIMARY_TESTS) if not np.isnan(p_one) else np.nan
        obs_dir = ("+" if beta > 0 else "-") if not np.isnan(beta) else "?"
        verdict = _verdict(p_bonf, beta, direction)

        log.info(
            "PRIMARY [%s__%s]: β=%.3f [%.3f, %.3f], p1=%.4f, p_corr=%.4f → %s",
            feat, agg, beta, ci_lo, ci_hi, p_one, p_bonf, verdict
        )

        results.append(TestResult(
            feature=feat, aggregation=agg, col_name=col,
            direction=direction,
            beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
            p_onesided=p_one, p_bonferroni=p_bonf,
            direction_obs=obs_dir, verdict=verdict,
            n_event=n_ev, n_control=n_ct, analysis="primary",
        ))

    return results


# ---------------------------------------------------------------------------
# Análisis de gradiente (test de interacción)
# ---------------------------------------------------------------------------

def run_gradient_analysis(event_windows: pd.DataFrame,
                           validated_features: list[str]) -> list[TestResult]:
    """
    Para cada feature primaria validada:
      feature ~ window_type_bin * group_bin + (1|patient_id) + (1|event_subcategory)
    Test de interacción: |efecto evento| en interescalenico > supra_axilar
    """
    results: list[TestResult] = []

    df = event_windows.copy()
    df["group_bin"] = (df["group"] == "interescalenico").astype(float)

    for col in validated_features:
        if col not in df.columns:
            continue
        try:
            import statsmodels.formula.api as smf
            work = df[[col, "window_type", "patient_id", "event_subcategory", "group_bin"]
                       ].dropna(subset=[col]).copy()
            work["window_type_bin"] = (work["window_type"] == "event").astype(float)

            if len(work) < 8:
                continue

            formula = f"{col} ~ window_type_bin * group_bin"
            with pywarnings.catch_warnings():
                pywarnings.simplefilter("ignore")
                vcf = {"event_subcategory": "0 + C(event_subcategory)"}
                model = smf.mixedlm(
                    formula, data=work,
                    groups=work["patient_id"],
                    vc_formula=vcf if work["event_subcategory"].nunique() > 1 else None,
                )
                res = model.fit(method="lbfgs", reml=False)

            inter_key = "window_type_bin:group_bin"
            if inter_key not in res.params:
                inter_key = "window_type_bin:group_bin"   # statsmodels naming

            beta, ci_lo, ci_hi, p_one = _one_sided_p(res, inter_key, direction=+1)
            log.info(
                "GRADIENT [%s]: β_int=%.3f [%.3f, %.3f], p1=%.4f",
                col, beta, ci_lo, ci_hi, p_one
            )

            feat_parts = col.split("__")
            feat = feat_parts[0]
            agg  = feat_parts[1] if len(feat_parts) > 1 else ""
            results.append(TestResult(
                feature=feat, aggregation=agg, col_name=col,
                direction=+1,
                beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
                p_onesided=p_one, p_bonferroni=np.nan,
                direction_obs=("+" if beta > 0 else "-") if not np.isnan(beta) else "?",
                verdict=_verdict(p_one, beta, +1),
                n_event=int((work["window_type"] == "event").sum()),
                n_control=int((work["window_type"] == "control").sum()),
                analysis="gradient",
            ))
        except Exception as exc:
            log.warning("Gradient test falló para %s: %s", col, exc)

    return results


# ---------------------------------------------------------------------------
# Análisis de sensibilidad
# ---------------------------------------------------------------------------

def run_sensitivity_analyses(event_windows: pd.DataFrame,
                               clinical_df: pd.DataFrame | None) -> list[TestResult]:
    """Cuatro análisis de sensibilidad pre-especificados."""
    results: list[TestResult] = []

    # ── S1: Sin (1|event_subcategory) ──
    log.info("SENSIBILIDAD S1: sin (1|event_subcategory)")
    for feat, agg, direction in PRIMARY_FEATURES:
        col = f"{feat}__{agg}"
        if col not in event_windows.columns:
            continue
        res = _fit_mixedlm(event_windows, col, random_effects=["patient_id"])
        if res is None:
            continue
        beta, ci_lo, ci_hi, p_one = _one_sided_p(res, "window_type_bin", direction)
        p_bonf = min(1.0, p_one * N_PRIMARY_TESTS) if not np.isnan(p_one) else np.nan
        results.append(TestResult(
            feature=feat, aggregation=agg, col_name=col, direction=direction,
            beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
            p_onesided=p_one, p_bonferroni=p_bonf,
            direction_obs=("+" if beta > 0 else "-") if not np.isnan(beta) else "?",
            verdict=_verdict(p_bonf, beta, direction),
            n_event=int((event_windows["window_type"] == "event").sum()),
            n_control=int((event_windows["window_type"] == "control").sum()),
            analysis="sensitivity_no_subcat",
        ))

    # ── S2: Solo estímulo "anclaje" ──
    log.info("SENSIBILIDAD S2: solo estímulo anclaje")
    df_anc = event_windows[
        (event_windows["window_type"] == "control") |
        (event_windows["event_subcategory"] == "anclaje")
    ].copy()
    for feat, agg, direction in PRIMARY_FEATURES:
        col = f"{feat}__{agg}"
        if col not in df_anc.columns:
            continue
        res = _fit_mixedlm(df_anc, col, random_effects=["patient_id"])
        if res is None:
            continue
        beta, ci_lo, ci_hi, p_one = _one_sided_p(res, "window_type_bin", direction)
        p_bonf = min(1.0, p_one * N_PRIMARY_TESTS) if not np.isnan(p_one) else np.nan
        results.append(TestResult(
            feature=feat, aggregation=agg, col_name=col, direction=direction,
            beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
            p_onesided=p_one, p_bonferroni=p_bonf,
            direction_obs=("+" if beta > 0 else "-") if not np.isnan(beta) else "?",
            verdict=_verdict(p_bonf, beta, direction),
            n_event=int((df_anc["window_type"] == "event").sum()),
            n_control=int((df_anc["window_type"] == "control").sum()),
            analysis="sensitivity_anclaje_only",
        ))

    # ── S3: Excluyendo 70767707 ──
    log.info("SENSIBILIDAD S3: excluyendo paciente parcial 70767707")
    df_excl = event_windows[~event_windows["patient_id"].isin(PARCIAL)].copy()
    for feat, agg, direction in PRIMARY_FEATURES:
        col = f"{feat}__{agg}"
        if col not in df_excl.columns:
            continue
        res = _fit_mixedlm(df_excl, col, random_effects=["patient_id", "event_subcategory"])
        if res is None:
            continue
        beta, ci_lo, ci_hi, p_one = _one_sided_p(res, "window_type_bin", direction)
        p_bonf = min(1.0, p_one * N_PRIMARY_TESTS) if not np.isnan(p_one) else np.nan
        results.append(TestResult(
            feature=feat, aggregation=agg, col_name=col, direction=direction,
            beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
            p_onesided=p_one, p_bonferroni=p_bonf,
            direction_obs=("+" if beta > 0 else "-") if not np.isnan(beta) else "?",
            verdict=_verdict(p_bonf, beta, direction),
            n_event=int((df_excl["window_type"] == "event").sum()),
            n_control=int((df_excl["window_type"] == "control").sum()),
            analysis="sensitivity_no_partial",
        ))

    # ── S4: Con covariables clínicas (exploratorio) ──
    if clinical_df is not None and not clinical_df.empty:
        log.info("SENSIBILIDAD S4: con covariables clínicas")
        covs = []
        # Intentar mergear covariables disponibles
        df_merged = event_windows.copy()
        clin_work = clinical_df.copy()
        if "patient_id" in clin_work.columns:
            for cov_raw, cov_norm in [
                ("edad", "edad"), ("Edad", "edad"), ("Age", "edad"),
                ("BMI", "bmi"), ("bmi", "bmi"),
                ("HTA", "hta"), ("Hipertension", "hta"),
            ]:
                if cov_raw in clin_work.columns:
                    try:
                        df_merged = df_merged.merge(
                            clin_work[["patient_id", cov_raw]].rename(columns={cov_raw: cov_norm}),
                            on="patient_id", how="left"
                        )
                        covs.append(cov_norm)
                    except Exception:
                        pass
        if covs:
            for feat, agg, direction in PRIMARY_FEATURES:
                col = f"{feat}__{agg}"
                if col not in df_merged.columns:
                    continue
                valid_covs = [c for c in covs if c in df_merged.columns and df_merged[c].notna().any()]
                if not valid_covs:
                    break
                res = _fit_mixedlm(
                    df_merged, col,
                    random_effects=["patient_id"],
                    extra_covariates=valid_covs,
                )
                if res is None:
                    continue
                beta, ci_lo, ci_hi, p_one = _one_sided_p(res, "window_type_bin", direction)
                p_bonf = min(1.0, p_one * N_PRIMARY_TESTS) if not np.isnan(p_one) else np.nan
                results.append(TestResult(
                    feature=feat, aggregation=agg, col_name=col, direction=direction,
                    beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
                    p_onesided=p_one, p_bonferroni=p_bonf,
                    direction_obs=("+" if beta > 0 else "-") if not np.isnan(beta) else "?",
                    verdict=_verdict(p_bonf, beta, direction),
                    n_event=int((df_merged["window_type"] == "event").sum()),
                    n_control=int((df_merged["window_type"] == "control").sum()),
                    analysis=f"sensitivity_covariates_{','.join(valid_covs)}",
                ))
    return results


# ---------------------------------------------------------------------------
# Features exploratorias (descriptivo)
# ---------------------------------------------------------------------------

def describe_exploratory(event_windows: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas descriptivas de features exploratorias (sin tests formales)."""
    rows = []
    for feat, agg, direction in EXPLORATORY_FEATURES:
        col = f"{feat}__{agg}"
        if col not in event_windows.columns:
            continue
        ev  = event_windows[event_windows["window_type"] == "event"][col].dropna()
        ct  = event_windows[event_windows["window_type"] == "control"][col].dropna()
        rows.append({
            "feature":    col,
            "direction_expected": "+" if direction > 0 else "-",
            "event_mean":   float(ev.mean()) if len(ev) > 0 else np.nan,
            "event_std":    float(ev.std()) if len(ev) > 0 else np.nan,
            "control_mean": float(ct.mean()) if len(ct) > 0 else np.nan,
            "control_std":  float(ct.std()) if len(ct) > 0 else np.nan,
            "n_event":   len(ev),
            "n_control": len(ct),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline principal y guardado
# ---------------------------------------------------------------------------

def run_paso5(event_windows: pd.DataFrame,
              clinical_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Ejecuta todos los tests y devuelve DataFrame con resultados.
    """
    log.info("PASO 5 — Tests confirmatorios")

    primary    = run_primary_tests(event_windows)
    validated  = [r.col_name for r in primary if r.verdict == "validado"]
    gradient   = run_gradient_analysis(event_windows, validated)
    sensitivity = run_sensitivity_analyses(event_windows, clinical_df)

    all_results = primary + gradient + sensitivity

    # Resumen
    log.info("-- PASO 5 RESUMEN ------------------------------------")
    log.info("Tests primarios: %d", len(primary))
    for r in primary:
        log.info("  %-40s β=%.3f  p1=%.4f  p_corr=%.4f  → %s",
                 r.col_name, r.beta, r.p_onesided, r.p_bonferroni, r.verdict)
    log.info("Features validadas: %s", validated)
    log.info("Tests gradiente: %d", len(gradient))

    rows = []
    for r in all_results:
        rows.append({
            "analysis":       r.analysis,
            "feature":        r.feature,
            "aggregation":    r.aggregation,
            "col_name":       r.col_name,
            "direction_expected": "+" if r.direction > 0 else "-",
            "direction_obs":  r.direction_obs,
            "beta":           r.beta,
            "ci_lo":          r.ci_lo,
            "ci_hi":          r.ci_hi,
            "p_onesided":     r.p_onesided,
            "p_bonferroni":   r.p_bonferroni,
            "verdict":        r.verdict,
            "n_event":        r.n_event,
            "n_control":      r.n_control,
        })
    return pd.DataFrame(rows)


def save_paso5(test_results_df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "test_results.csv"
    if not test_results_df.empty:
        test_results_df.to_csv(path, index=False)
        log.info("Guardado: %s (%d filas)", path, len(test_results_df))
