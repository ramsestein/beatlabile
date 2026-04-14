# Table: BeatLabile performance — MAP<55 vs MAP<65 (VitalDB external validation)

| Metric | MAP<55 (primary) | MAP<65 (sensitivity) |
|:-------|:----------------:|:--------------------:|
| Prevalence (% pre-event windows) | 51.1% | 76.5% |
| GLMM AUROC [95% CI] | 0.844 [0.806–0.883] | 0.629 [0.567–0.691] |
| MILP sensitivity | 0.513 | 0.375 |
| MILP specificity | 0.770 | 0.817 |
| MILP PPV | 0.716 | 0.872 |
| MILP NPV | 0.584 | 0.283 |

**Note:** MAP<65 GLMM uses the MAP<55-trained model without retraining.
MILP rule is the MAP<55 rule applied directly to MAP<65 labelled windows.