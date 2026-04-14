# Lead Time Summary

## Effective prediction horizon

The features are aggregated over a **30-minute sliding window** ending at event onset. In the current analysis design, the model receives all autonomic information from the 30 minutes immediately preceding each event.

| Model | Event type | Subset | Metric | Mean (min) | Median [IQR] (min) |
|-------|------------|--------|--------|------------|--------------------|
| GLMM-hypotension | hypotension | All event windows | Lead time from window start (min) | 30.0 | 30.0 [30–30] |
| GLMM-hypotension | hypotension | All event windows | Lead time from window centre (min) | 15.0 | 15.0 [15–15] |
| GLMM-hypotension | hypotension | All event windows | Lead time from window end (min) | 0.0 | 0.0 [0–0] |
| GLMM-hypotension | hypotension | True positives only | Lead time from window start (min) | 30.0 | 30.0 [30–30] |
| GLMM-hypotension | hypotension | True positives only | Lead time from window centre (min) | 15.0 | 15.0 [15–15] |
| GLMM-hypotension | hypotension | True positives only | Lead time from window end (min) | 0.0 | 0.0 [0–0] |
| MILP-hypotension | hypotension | All event windows | Lead time from window start (min) | 30.0 | 30.0 [30–30] |
| MILP-hypotension | hypotension | All event windows | Lead time from window centre (min) | 15.0 | 15.0 [15–15] |
| MILP-hypotension | hypotension | All event windows | Lead time from window end (min) | 0.0 | 0.0 [0–0] |
| MILP-hypotension | hypotension | True positives only | Lead time from window start (min) | 30.0 | 30.0 [30–30] |
| MILP-hypotension | hypotension | True positives only | Lead time from window centre (min) | 15.0 | 15.0 [15–15] |
| MILP-hypotension | hypotension | True positives only | Lead time from window end (min) | 0.0 | 0.0 [0–0] |
| GLMM-hypertension | hypertension | All event windows | Lead time from window start (min) | 30.0 | 30.0 [30–30] |
| GLMM-hypertension | hypertension | All event windows | Lead time from window centre (min) | 15.0 | 15.0 [15–15] |
| GLMM-hypertension | hypertension | All event windows | Lead time from window end (min) | 0.0 | 0.0 [0–0] |
| GLMM-hypertension | hypertension | True positives only | Lead time from window start (min) | 30.0 | 30.0 [30–30] |
| GLMM-hypertension | hypertension | True positives only | Lead time from window centre (min) | 15.0 | 15.0 [15–15] |
| GLMM-hypertension | hypertension | True positives only | Lead time from window end (min) | 0.0 | 0.0 [0–0] |
| MILP-hypertension | hypertension | All event windows | Lead time from window start (min) | 30.0 | 30.0 [30–30] |
| MILP-hypertension | hypertension | All event windows | Lead time from window centre (min) | 15.0 | 15.0 [15–15] |
| MILP-hypertension | hypertension | All event windows | Lead time from window end (min) | 0.0 | 0.0 [0–0] |
| MILP-hypertension | hypertension | True positives only | Lead time from window start (min) | 30.0 | 30.0 [30–30] |
| MILP-hypertension | hypertension | True positives only | Lead time from window centre (min) | 15.0 | 15.0 [15–15] |
| MILP-hypertension | hypertension | True positives only | Lead time from window end (min) | 0.0 | 0.0 [0–0] |

## AUC vs prediction horizon (from act_lead_time.py, linear extrapolation)

| Event type | Lead time (min) | AUC | 95% CI |
|------------|-----------------|-----|--------|
| hypotension | 0 | 0.784 | [0.678–0.855] |
| hypotension | 5 | 0.668 | [0.505–0.783] |
| hypotension | 10 | 0.625 | [0.462–0.758] |
| hypotension | 15 | 0.605 | [0.436–0.746] |
| hypotension | 20 | 0.593 | [0.420–0.737] |
| hypotension | 30 | 0.568 | [0.384–0.721] |
| hypertension | 0 | 0.899 | [0.791–0.966] |
| hypertension | 5 | 0.853 | [0.704–0.941] |
| hypertension | 10 | 0.804 | [0.623–0.915] |
| hypertension | 15 | 0.779 | [0.570–0.902] |
| hypertension | 20 | 0.766 | [0.555–0.894] |
| hypertension | 30 | 0.749 | [0.535–0.885] |
| variability | 0 | 0.544 | [0.516–0.572] |
| variability | 5 | 0.509 | [0.482–0.537] |
| variability | 10 | 0.505 | [0.478–0.533] |
| variability | 15 | 0.501 | [0.474–0.529] |
| variability | 20 | 0.498 | [0.470–0.526] |
| variability | 30 | 0.492 | [0.464–0.520] |

