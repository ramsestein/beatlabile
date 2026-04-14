# Table: Clinical Utility of BeatLabile at Different Outcome Prevalences

## Hypotension (MAP<55 ≥3 min)
| Population | Model | Prevalence | Sensitivity | Specificity | PPV | NPV | LR+ | LR− | NNS |
|------------|-------|-----------|-------------|-------------|-----|-----|-----|-----|-----|
| Clínic Barcelona (dev.) | MILP | 7% | 0.513 | 0.770 | 0.144 | 0.955 | 2.23 | 0.632 | 7 |
| Typical major surgery | MILP | 15% | 0.513 | 0.770 | 0.282 | 0.900 | 2.23 | 0.632 | 4 |
| VitalDB (validation) | MILP | 51% | 0.513 | 0.770 | 0.700 | 0.602 | 2.23 | 0.632 | 2 |
| Clínic Barcelona (dev.) | GLMM (threshold=0.5) | 7% | 0.090 | 0.988 | 0.369 | 0.935 | 7.76 | 0.920 | 3 |
| Typical major surgery | GLMM (threshold=0.5) | 15% | 0.090 | 0.988 | 0.578 | 0.860 | 7.76 | 0.920 | 2 |
| VitalDB (validation) | GLMM (threshold=0.5) | 51% | 0.090 | 0.988 | 0.890 | 0.510 | 7.76 | 0.920 | 2 |


## Hypertension (MAP<55 ≥3 min)
| Population | Model | Prevalence | Sensitivity | Specificity | PPV | NPV | LR+ | LR− | NNS |
|------------|-------|-----------|-------------|-------------|-----|-----|-----|-----|-----|
| Clínic Barcelona (dev.) | MILP | 7% | 0.087 | 0.965 | 0.158 | 0.934 | 2.49 | 0.946 | 7 |
| Typical major surgery | MILP | 15% | 0.087 | 0.965 | 0.305 | 0.857 | 2.49 | 0.946 | 4 |
| VitalDB (validation) | MILP | 51% | 0.087 | 0.965 | 0.722 | 0.503 | 2.49 | 0.946 | 2 |
| Clínic Barcelona (dev.) | GLMM (threshold=0.5) | 7% | 0.056 | 0.953 | 0.082 | 0.931 | 1.18 | 0.991 | 13 |
| Typical major surgery | GLMM (threshold=0.5) | 15% | 0.056 | 0.953 | 0.173 | 0.851 | 1.18 | 0.991 | 6 |
| VitalDB (validation) | GLMM (threshold=0.5) | 51% | 0.056 | 0.953 | 0.553 | 0.491 | 1.18 | 0.991 | 2 |

