# Sensitivity Analysis: MAP<65 vs MAP<55 Threshold Comparison

## Table: Hypotension Definition Sensitivity Analysis

| Parameter | MAP<55 mmHg ≥3 min (primary) | MAP<65 mmHg ≥3 min (sensitivity) |
|-----------|-------------------------------|-----------------------------------|
| Prevalence – Clínic Barcelona | 7.0% | 34.1% |
| Prevalence – VitalDB Seoul | 51.1% | 76.5% |
| GLMM AUC VitalDB (70/30 split) | 0.844 [0.806–0.883] | 0.715 [0.662–0.767] |
| Cross-cohort AUC rank correlation (ρ) | reported in main text | 0.202 (p=0.2120) |
| MILP Sensitivity (VitalDB test) | 0.513 | 0.375 |
| MILP Specificity (VitalDB test) | 0.77 | 0.817 |
| MILP PPV (VitalDB test) | 0.716 | 0.872 |
| MILP NPV (VitalDB test) | 0.584 | 0.283 |

**Note:** GLMM uses the same 8 pre-specified parsimonious features (no re-selection).
MILP rule was trained on MAP<55 Clínic data and applied unchanged to MAP<65 windows.
70/30 patient-level stratified split with seed=42.
