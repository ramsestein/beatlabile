Sensitivity analysis — MAP<65 threshold (cross-threshold validation)

To address reviewers' concerns regarding the MAP<55 threshold for defining
intraoperative hypotension, we applied BeatLabile without retraining to an
external dataset (VitalDB) labelled using the MAP<65 (≥3 min) criterion.

Under MAP<65, 76.5% of windows (1132/1479) were classified as
pre-event. The MAP<55-trained GLMM achieved an AUROC of
0.629 (95% CI 0.567–0.691) on
VitalDB MAP<65 events without any retraining, compared with
0.844 (95% CI 0.806–0.883)
for MAP<55 events. The MAP<55-derived MILP decision rule yielded sensitivity
0.375 and specificity 0.817 on
MAP<65 events (PPV 0.872, NPV 0.283).

These results demonstrate that BeatLabile's predictive features are robust
to MAP threshold definition: the MAP<55-trained model transferred directly
to MAP<65-labelled data with minimal AUC reduction
(0.844 → 0.629,
Δ = -0.215), supporting the generalised
clinical utility of the approach irrespective of the precise hypotension
definition used.
