# Q2 v2 Analysis Report
**Date**: 2026-05-10 13:01
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

- **Total eventos vasopresor crudos**: 19
- **Eventos limpios (criterios v2)**: **11**
  - Efedrina: 10  |  Fenilefrina: 1
- **Con ventana control válida (3 min)**: **10**

### Checkpoint power
⚠ n_clean = 11 < 12 (CAVEAT: potencia reducida — continuar con limitación documentada)

---

## 3. Resultados Q2a v2 (pre vs control)

**Resultado primario**: 0/5 features validadas (Bonferroni α = 0.010)

| Feature | β | 95% CI | p_bonf | Verdict |
|---------|---|--------|--------|--------|
| ptt_cv__mean | -0.0011 | [-0.1288, 0.1267] | 1.000 | NS |
| ptt_cv__std | -0.0176 | [-0.0762, 0.0409] | 1.000 | NS |
| ptt_std__max | -11.4617 | [-79.2624, 56.3390] | 1.000 | NS |
| pai_mean__mean | -1.1616 | [-5.4737, 3.1505] | 1.000 | NS |
| brs_seq__min | -0.1005 | [-0.3024, 0.1014] | 0.724 | NS |


**Veredicto Q2a v2**: **NO CONFIRMADA (0 primary features validated)**

---

## 4. Comparación BRS: secuencia vs espectral

Ver figura: `brs_methods_comparison.png`

### BRS_seq vs BRS-α_LF en Q1:
- BRS_seq Q1: β = 0.0588 [-0.4360, 0.5536]  p₁ = 0.594
- (brs_alpha_lf fue el método histórico de BeatLabile)

---

## 5. Gradient analysis

| feature        | feature_name   | agg   |   beta_interescalenico |   p_gradient |
|:---------------|:---------------|:------|-----------------------:|-------------:|
| ptt_cv__mean   | ptt_cv         | mean  |              0.210034  |    0.139601  |
| ptt_cv__std    | ptt_cv         | std   |              0.0338474 |    0.642716  |
| ptt_std__max   | ptt_std        | max   |             44.2772    |    0.602153  |
| pai_mean__mean | pai_mean       | mean  |              5.41304   |    0.0598004 |
| brs_seq__min   | brs_seq        | min   |             -0.143579  |    0.473964  |

---

## 6. Sensitivity C — Diagnóstico de confounding por cambio de infusión

**Pregunta crítica**: ¿Los efectos de rigidificación persisten ajustando por cambios de propofol/remi?

| Feature | β adj | p_adj | Verdict |
|---------|-------|-------|--------|
| ptt_cv__mean | -0.0011 | 0.985 | NS_after_adj |
| ptt_cv__std | -0.0176 | 0.513 | NS_after_adj |
| ptt_std__max | -11.4617 | 0.711 | NS_after_adj |
| pai_mean__mean | -1.1616 | 0.557 | NS_after_adj |
| brs_seq__min | -0.1005 | 0.289 | NS_after_adj |


- Features robustas (significativas tras ajuste): **0/5**
- Interpretación: CAVEAT — efecto atenuado o perdido tras ajuste por infusiones.

---

## 7. Sensitivity A (solo efedrina) / B (excluir 70767707)

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

1. **n pequeño**: 11 eventos limpios (objetivo pre-especificado ≥12 para potencia adecuada).
2. **Heterogeneidad de grupo**: bloqueos interescalénico vs supraclavicular — gradiente analizado.
3. **BRS_seq como proxy de BP**: PTT no es BP directo; calibración individual necesaria.
4. **SA-D no ejecutable**: variables demográficas no disponibles en dataset actual.
5. **Causalidad**: diseño observacional pre-post, no puede descartar confounding residual.
6. **BRS_seq en Q1**: calculada post-hoc en datos de Q1 (no pre-especificada en Q1).

---
*Análisis pre-especificado. No se modificaron criterios de dirección o umbrales tras ver resultados.*
