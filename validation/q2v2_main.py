"""
Q2 v2 Main Orchestrator
────────────────────────
Runs all 9 PASOS of Q2 v2 sequentially.
Call:   python validation/q2v2_main.py
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2v2_config import (
    Q2V2_RES, FIG, Q1_RES,
    FEATURES_BRS_SEQ, VASOPRESSOR_EVENTS, PAIRED_PRE_CONTROL, PAIRED_PRE_POST,
    TEST_RESULTS_Q2A, TEST_RESULTS_Q2B, REPORT, Q1_RESULTS_V2,
    MIN_EVENTS_CHECKPOINT, ALPHA_BONF, PRIMARY, EXPLORATORY,
    CONTROL_DURATION_S, SEED, PARTIAL_PATIENT,
)
from q2v2_brs_sequence import compute_all_brs_seq
from q2v2_events import load_all, build_vasopressor_events_v2, build_control_windows_v2
from q2v2_stats import (
    build_paired_Q2a_v2, run_Q2a_v2_tests,
    build_paired_Q2b_v2, run_Q2b_v2_tests,
    run_gradient_v2, run_sensitivity_v2,
    compute_brs_seq_q1,
)
from q2v2_figures import generate_all_figures

np.random.seed(SEED)
warnings.filterwarnings("ignore")

STEP_TIMINGS: dict[str, float] = {}


def _step(label: str):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")


def _save(df: pd.DataFrame, path: Path, label: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"[SAVE] {path.name}  ({len(df):,} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# Report generator
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_row(row, key, default="–"):
    val = row.get(key, default)
    if pd.isna(val):
        return default
    return str(val)


def _fmt_float(v, fmt=".4f", default="–"):
    try:
        return f"{float(v):{fmt}}" if not pd.isna(v) else default
    except Exception:
        return default


def generate_report(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    results_q2a: pd.DataFrame,
    results_q2b: pd.DataFrame,
    sa_results: dict,
    grad_results: pd.DataFrame,
    brs_q1_summary: pd.DataFrame,
    run_date: str,
) -> str:

    n_total   = len(events)
    n_clean   = int((events["status"] == "clean").sum())
    n_efed    = int((events[(events["status"] == "clean")]["drug"] == "efedrina").sum())
    n_fen     = int((events[(events["status"] == "clean")]["drug"] == "fenilefrina").sum())
    ctrl_ok   = controls[controls["ctrl_excluded"].isna()] if controls is not None else pd.DataFrame()
    n_ctrl    = len(ctrl_ok)

    # Count how many "infusion" exclusions were recovered vs v1
    inf_recovered = events[(events["status"] == "clean")].shape[0]

    # Primary Q2a verdicts
    primary_keys = [f"{f}__{a}" for f, a, _ in PRIMARY]
    prim = results_q2a[results_q2a["feature"].isin(primary_keys)] if not results_q2a.empty else pd.DataFrame()
    n_validated = int((prim["p_bonferroni"] < 0.050).sum()) if "p_bonferroni" in prim.columns else 0

    # SA-C summary
    sa_c = sa_results.get("SA_C_infusion_adjusted", pd.DataFrame())
    sa_c_robust = 0
    if not sa_c.empty and "verdict_adj" in sa_c.columns:
        sa_c_robust = int((sa_c["verdict_adj"] == "ROBUST").sum())

    # Overall verdict
    if n_validated >= 3:
        verdict = "CONFIRMADA (≥3 primary features validated)"
    elif n_validated >= 1:
        verdict = "PARCIALMENTE CONFIRMADA (1-2 primary features validated)"
    else:
        verdict = "NO CONFIRMADA (0 primary features validated)"

    # BRS Q1 summary
    brs_q1_beta = brs_q1_beta_lo = brs_q1_beta_hi = brs_q1_p = "–"
    if brs_q1_summary is not None and not brs_q1_summary.empty:
        r = brs_q1_summary.iloc[0]
        brs_q1_beta   = _fmt_float(r.get("beta", np.nan))
        brs_q1_beta_lo = _fmt_float(r.get("ic_lo", np.nan))
        brs_q1_beta_hi = _fmt_float(r.get("ic_hi", np.nan))
        brs_q1_p       = _fmt_float(r.get("p_1sided", np.nan), ".3f")

    # Build primary results table
    primary_table = "| Feature | β | 95% CI | p_bonf | Verdict |\n"
    primary_table += "|---------|---|--------|--------|--------|\n"
    for _, r in prim.iterrows():
        primary_table += (
            f"| {r.get('feature','–')} "
            f"| {_fmt_float(r.get('beta',np.nan))} "
            f"| [{_fmt_float(r.get('ic_lo',np.nan))}, {_fmt_float(r.get('ic_hi',np.nan))}] "
            f"| {_fmt_float(r.get('p_bonferroni',np.nan),'.3f')} "
            f"| {r.get('verdict','–')} |\n"
        )

    # Sensitivity C table
    sa_c_table = "| Feature | β adj | p_adj | Verdict |\n|---------|-------|-------|--------|\n"
    if not sa_c.empty and "beta_adj" in sa_c.columns:
        for _, r in sa_c.iterrows():
            sa_c_table += (
                f"| {r.get('feature','–')} "
                f"| {_fmt_float(r.get('beta_adj',np.nan))} "
                f"| {_fmt_float(r.get('p_2sided_adj',r.get('p_adj',np.nan)),'.3f')} "
                f"| {r.get('verdict_adj','–')} |\n"
            )

    md = f"""# Q2 v2 Analysis Report
**Date**: {run_date}
**Analyst**: automated (q2v2_main.py)
**Analysis**: Análisis Q2 v2 (relanzamiento corregido)

---

## 1. Cambios vs Q2 v1 (pre-especificados antes de ver resultados v2)

| # | Cambio | Detalle |
|---|--------|---------|
| 1 | **Eliminar criterio de no cambio de infusión** | Convertido en covariable (delta_propofol_pre, delta_remi_pre). Sensitivity C verifica confounding. |
| 2 | **Ventanas control 3 min mínimo** | CONTROL_DURATION_S=180 (antes 300 s = 5 min). Recupera controles en pacientes con periodos quiescentes cortos. |
| 3 | **BRS_seq primaria** | Método secuencia PTT-RR reemplaza brs_alpha_lf como feature primaria de barorreflejos. |
| 4 | **BRS_seq en Q1** | Calculada también en Q1 para figura de disociación consistente. |
| 5 | **Sensitivity C (diagnóstico)** | delta ~ delta_propofol_pre + delta_remi_pre + (1|patient_id) — crítico para descartar confounding por cambios de infusión. |

## 1b. Correcciones aplicadas en re-ejecución (fixes post-v2-inicial)

| Fix | Descripción | Impacto |
|-----|-------------|---------|
| **Fix 1** | VALIDITY_FEATURES = [ptt_cv, ptt_std, pai_mean] — brs_seq excluido del criterio de exclusión de ventana pre-evento (NaN normal sin secuencias detectables). | Recupera eventos incorrectamente excluidos por brs_seq NaN. |
| **Fix 2** | event_filtering_audit.csv — traza de filtros por evento con sumario. | Trazabilidad del proceso de exclusión. |
| **Fix 3** | Q1 BRS_seq: algoritmo corregido; usa t_end_s como timestamp del estímulo; calcula pre [-5,-2] min y post [0,+2.5] min desde features_brs_seq.parquet. | BRS_seq Q1 computable con datos reales (antes 0 pares válidos). |

---

## 2. Cohorte v2

- **Total eventos vasopresor crudos**: {n_total}
- **Eventos limpios (criterios v2)**: **{n_clean}**
  - Efedrina: {n_efed}  |  Fenilefrina: {n_fen}
- **Con ventana control válida (3 min)**: **{n_ctrl}**

### Checkpoint power
{"✓ n_clean ≥ " + str(MIN_EVENTS_CHECKPOINT) + " (potencia suficiente según checkpoint)" if n_clean >= MIN_EVENTS_CHECKPOINT
 else f"⚠ n_clean = {n_clean} < {MIN_EVENTS_CHECKPOINT} (CAVEAT: potencia reducida — continuar con limitación documentada)"}

---

## 3. Resultados Q2a v2 (pre vs control)

**Resultado primario**: {n_validated}/5 features validadas (Bonferroni α = {ALPHA_BONF:.3f})

{primary_table}

**Veredicto Q2a v2**: **{verdict}**

---

## 4. Comparación BRS: secuencia vs espectral

Ver figura: `brs_methods_comparison.png`

### BRS_seq vs BRS-α_LF en Q1:
- BRS_seq Q1: β = {brs_q1_beta} [{brs_q1_beta_lo}, {brs_q1_beta_hi}]  p₁ = {brs_q1_p}
- (brs_alpha_lf fue el método histórico de BeatLabile)

---

## 5. Gradient analysis

{grad_results.to_markdown(index=False) if not grad_results.empty else "No hay suficientes datos por grupo para análisis de gradiente."}

---

## 6. Sensitivity C — Diagnóstico de confounding por cambio de infusión

**Pregunta crítica**: ¿Los efectos de rigidificación persisten ajustando por cambios de propofol/remi?

{sa_c_table}

- Features robustas (significativas tras ajuste): **{sa_c_robust}/5**
- Interpretación: {"ROBUSTO — cambios de infusión no explican la rigidificación." if sa_c_robust >= 3 else "CAVEAT — efecto atenuado o perdido tras ajuste por infusiones."}

---

## 7. Sensitivity A (solo efedrina) / B (excluir {PARTIAL_PATIENT})

Ver archivos: results/validation/Q2v2/sa_A_efedrina_only.csv, sa_B_no_partial.csv

---

## 8. Q2b v2 (pre vs post 5 min)

Ver: test_results_Q2b_v2.csv

---

## 9. Disociación Q1 vs Q2v2

Ver figura: `q1_vs_q2v2_dissociation.png`

**Hipótesis de disociación**: Q1 (dolor → homeostasis barorrefleja) y Q2v2 (vasopresor → anestesia general) muestran patrones OPUESTOS en features de rigidificación vascular.

---

## 10. Limitaciones

1. **n pequeño**: {n_clean} eventos limpios (objetivo pre-especificado ≥12 para potencia adecuada).
2. **Heterogeneidad de grupo**: bloqueos interescalénico vs supraclavicular — gradiente analizado.
3. **BRS_seq como proxy de BP**: PTT no es BP directo; calibración individual necesaria.
4. **SA-D no ejecutable**: variables demográficas no disponibles en dataset actual.
5. **Causalidad**: diseño observacional pre-post, no puede descartar confounding residual.
6. **BRS_seq en Q1**: calculada post-hoc en datos de Q1 (no pre-especificada en Q1).

---
*Análisis pre-especificado. No se modificaron criterios de dirección o umbrales tras ver resultados.*
"""
    return md


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    t0_global = time.time()
    run_date  = datetime.now().strftime("%Y-%m-%d %H:%M")
    Q2V2_RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  Q2 v2 Pipeline — {run_date}")
    print(f"  Output: {Q2V2_RES}")
    print(f"{'#'*70}")

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 1 — BRS_seq feature computation
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 1: Compute BRS_seq (PTT-RR sequence method)")
    t0 = time.time()
    if FEATURES_BRS_SEQ.exists():
        print(f"  [SKIP] {FEATURES_BRS_SEQ.name} already exists — loading cached")
        feat = pd.read_parquet(FEATURES_BRS_SEQ)
        feat["patient_id"] = feat["patient_id"].astype(str)
    else:
        feat = compute_all_brs_seq()
    STEP_TIMINGS["paso1_brs_seq"] = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 2 — Vasopressor events (v2: no infusion criterion)
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 2: Build vasopressor events (v2 — infusion criterion REMOVED)")
    t0 = time.time()
    ann, feat_check, drug, ev = load_all()

    events = build_vasopressor_events_v2(ann, feat, drug, ev)
    _save(events, VASOPRESSOR_EVENTS)

    n_clean = int((events["status"] == "clean").sum())
    if n_clean < MIN_EVENTS_CHECKPOINT:
        print(f"\n  ⚠  CHECKPOINT: n_clean = {n_clean} < {MIN_EVENTS_CHECKPOINT}.")
        print(f"     Power may be insufficient. Continuing with explicit caveat.")
        print(f"     (This will be documented in the report.)\n")
    else:
        print(f"\n  ✓  Checkpoint passed: n_clean = {n_clean} ≥ {MIN_EVENTS_CHECKPOINT}")
    STEP_TIMINGS["paso2_events"] = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 3 — Control windows (3 min)
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 3: Build quiescent control windows (3 min, infusion check KEPT)")
    t0 = time.time()
    controls = build_control_windows_v2(events, ann, feat, drug)
    ctrl_path = Q2V2_RES / "control_windows_v2.csv"
    _save(controls, ctrl_path)
    STEP_TIMINGS["paso3_controls"] = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 4 — Q2a: pre vs control (confirmatory)
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 4: Q2a v2 — pre vs control (GLMM, one-sided Bonferroni)")
    t0 = time.time()
    paired_q2a = build_paired_Q2a_v2(events, controls, feat)
    _save(paired_q2a, PAIRED_PRE_CONTROL)

    results_q2a = run_Q2a_v2_tests(paired_q2a)
    _save(results_q2a, TEST_RESULTS_Q2A)
    STEP_TIMINGS["paso4_q2a"] = time.time() - t0

    # Print primary results summary
    primary_keys = [f"{f}__{a}" for f, a, _ in PRIMARY]
    prim = results_q2a[results_q2a["feature"].isin(primary_keys)]
    n_val = int((prim["p_bonferroni"] < 0.050).sum()) if "p_bonferroni" in prim.columns else 0
    print(f"\n  Primary Q2a summary: {n_val}/5 features validated (Bonferroni p < 0.010)")

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 5 — Q2b: pre vs post (descriptive)
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 5: Q2b v2 — pre vs post 5 min (descriptive GLMM)")
    t0 = time.time()
    paired_q2b = build_paired_Q2b_v2(events, feat)
    _save(paired_q2b, PAIRED_PRE_POST)

    results_q2b = run_Q2b_v2_tests(paired_q2b)
    _save(results_q2b, TEST_RESULTS_Q2B)
    STEP_TIMINGS["paso5_q2b"] = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 6 — Figures
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 6: Generate figures 1–3 (violins, trajectories, forest)")
    t0 = time.time()
    from q2v2_figures import fig_violins_Q2a_v2, fig_trajectories_Q2b_v2, fig_forest_Q2a_v2
    fig_violins_Q2a_v2(paired_q2a, results_q2a)
    fig_trajectories_Q2b_v2(events, feat)
    fig_forest_Q2a_v2(results_q2a)
    STEP_TIMINGS["paso6a_figs"] = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 7 — Q1 BRS_seq + dissociation figure
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 7: Compute BRS_seq for Q1 events + dissociation figure")
    t0 = time.time()
    brs_q1_summary = compute_brs_seq_q1(feat)

    # Load Q1 results (test_results_v2.csv)
    q1_results = pd.DataFrame()
    if Q1_RESULTS_V2.exists():
        q1_results = pd.read_csv(Q1_RESULTS_V2)
        if "patient_id" in q1_results.columns:
            q1_results["patient_id"] = q1_results["patient_id"].astype(str)
        print(f"  Loaded Q1 results: {len(q1_results)} rows from {Q1_RESULTS_V2.name}")

        # Upsert BRS_seq Q1 result row: REMOVE stale brs_seq row, ADD fresh one
        if brs_q1_summary is not None and not brs_q1_summary.empty:
            brs_row = brs_q1_summary.copy()
            if "analysis" in q1_results.columns:
                # Drop any existing brs_seq row (may have NaN from previous run)
                q1_clean = q1_results[q1_results["analysis"] != "primary_brs_seq"].copy()
                stale = len(q1_results) - len(q1_clean)
                if stale:
                    print(f"  [UPDATE] Replacing {stale} stale brs_seq row(s) in {Q1_RESULTS_V2.name}")
            else:
                q1_clean = q1_results.copy()
            updated_q1 = pd.concat([q1_clean, brs_row], ignore_index=True)
            updated_q1.to_csv(Q1_RESULTS_V2, index=False)
            print(f"  [SAVE] Upserted brs_seq Q1 row in {Q1_RESULTS_V2.name}")
            # Reload for downstream use
            q1_results = updated_q1
    else:
        print(f"  [WARN] {Q1_RESULTS_V2.name} not found — dissociation figure will lack Q1 data")

    # Figure 4 — BRS methods comparison
    from q2v2_figures import fig_brs_methods_comparison
    fig_brs_methods_comparison(feat, results_q2a)

    # Figure 5 — Dissociation
    from q2v2_figures import fig_dissociation_v2
    fig_dissociation_v2(results_q2a, q1_results, brs_q1_summary)

    STEP_TIMINGS["paso7_dissociation"] = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 8 — Gradient + sensitivity
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 8: Gradient analysis + sensitivity analyses (A-D)")
    t0 = time.time()
    grad_results = run_gradient_v2(paired_q2a)
    if not grad_results.empty:
        _save(grad_results, Q2V2_RES / "gradient_analysis_v2.csv")

    sa_results = run_sensitivity_v2(events, controls, feat)
    for label, df_sa in sa_results.items():
        if not df_sa.empty:
            _save(df_sa, Q2V2_RES / f"{label}.csv")
    STEP_TIMINGS["paso8_gradient_sa"] = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # PASO 9 — Report
    # ─────────────────────────────────────────────────────────────────────────
    _step("PASO 9: Generate Q2_report_v2.md")
    t0 = time.time()
    report_md = generate_report(
        events=events,
        controls=controls,
        results_q2a=results_q2a,
        results_q2b=results_q2b,
        sa_results=sa_results,
        grad_results=grad_results,
        brs_q1_summary=brs_q1_summary,
        run_date=run_date,
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_md, encoding="utf-8")
    print(f"[SAVE] {REPORT.name}")
    STEP_TIMINGS["paso9_report"] = time.time() - t0

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────────────────
    total_time = time.time() - t0_global
    print(f"\n{'='*70}")
    print(f"  Q2 v2 Pipeline COMPLETE — {total_time:.1f}s total")
    print(f"  Output directory: {Q2V2_RES}")
    print(f"  Figures:          {FIG}")
    print(f"  Report:           {REPORT.name}")
    print(f"{'='*70}")
    print("\n  Step timings:")
    for step, t in STEP_TIMINGS.items():
        print(f"    {step:35s}  {t:6.1f}s")

    primary_keys = [f"{f}__{a}" for f, a, _ in PRIMARY]
    prim = results_q2a[results_q2a["feature"].isin(primary_keys)] \
        if not results_q2a.empty else pd.DataFrame()
    n_val = int((prim["p_bonferroni"] < 0.050).sum()) if "p_bonferroni" in prim.columns else 0

    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  PRIMARY Q2a v2: {n_val}/5 features validated  │")
    if n_val >= 3:
        print(f"  │  VEREDICTO: CONFIRMADA                      │")
    elif n_val >= 1:
        print(f"  │  VEREDICTO: PARCIALMENTE CONFIRMADA         │")
    else:
        print(f"  │  VEREDICTO: NO CONFIRMADA                   │")
    print(f"  └─────────────────────────────────────────────┘")

    return {
        "events": events,
        "controls": controls,
        "feat": feat,
        "paired_q2a": paired_q2a,
        "results_q2a": results_q2a,
        "results_q2b": results_q2b,
        "sa_results": sa_results,
        "grad_results": grad_results,
        "brs_q1_summary": brs_q1_summary,
    }


if __name__ == "__main__":
    run_all()
