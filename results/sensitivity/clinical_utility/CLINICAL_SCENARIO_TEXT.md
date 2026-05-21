# Clinical Scenario and Lead-Time Text for A&A Manuscript

## Results: Lead time analysis

The 30-minute autonomic feature window aggregated immediately before event
onset provides a maximum advance warning of 30 minutes from the first beat
of the prediction window to event onset (15 minutes from the window centroid).
To characterise the clinically relevant prediction horizon, features were
back-extrapolated using linear approximation to simulate physiological states
at 5, 10, 15, 20, and 30 minutes before event onset (Supplementary Methods).
For hypotension (MAP<55 ≥3 min), the parsimonious GLMM maintained an AUC of
0.605 at 15 minutes before event onset (bootstrap 95% CI in
Supplementary Table). Discriminatory ability declined with increasing lead
time, reaching AUC=0.568 at 30 minutes before event onset. For
hypertension, discriminatory ability was more sustained, with AUC=0.779
at 15 minutes and 0.749 at 30 minutes, consistent with the observation
that hypertensive autonomic signatures accumulate over a longer pre-event
window.

## Discussion: Clinical utility framing

### Ready-to-paste paragraph (intermediate-prevalence scenario, ~15%)

Consider an anaesthesiologist managing a patient during major elective abdominal
surgery (background hypotension prevalence approximately 15%, MAP<55 ≥3 min).
When the BeatLabile autonomic rule does not alert (negative test), the posterior
probability that sustained hypotension will NOT occur in the subsequent 30 minutes
is 90.0% (NPV=0.9), providing actionable haemodynamic reassurance.
When the rule fires (positive test), the positive predictive value is 28.2%
(PPV=0.282), meaning approximately 1 in 4 screen-positive patients
will experience a true sustained hypotensive event; the remaining alerts
represent opportunities for heightened vigilance rather than mandatory
pharmacological intervention. The likelihood ratios (LR+=2.23,
LR−=0.632) quantify the diagnostic shift: a positive result
approximately 2-fold increases the odds of hypotension, while a
negative result reduces them by ~1.6-fold.
The clinical utility of BeatLabile is therefore best understood as a
high-specificity, moderate-sensitivity rule-out tool: negative alerts provide
actionable reassurance and may reduce unnecessary pre-emptive interventions,
while positive alerts should prompt intensified monitoring and preparation
rather than mandating vasopressor administration.

### Table: BeatLabile clinical utility at different outcome prevalences (MAP<55 ≥3 min)
*(See clinical_utility_table.csv for full table)*

| Scenario | Prevalence | Sensitivity | Specificity | PPV | NPV | LR+ | LR− | NNS |
|----------|-----------|-------------|-------------|-----|-----|-----|-----|-----|
| Clínic Barcelona (dev.) | 7% | 0.513 | 0.770 | 0.144 | 0.955 | 2.23 | 0.632 | 7 |
| Typical major surgery | 15% | 0.513 | 0.770 | 0.282 | 0.900 | 2.23 | 0.632 | 4 |
| VitalDB (validation) | 51% | 0.513 | 0.770 | 0.700 | 0.602 | 2.23 | 0.632 | 2 |

*Sensitivity and specificity derived from MAP<55 MILP rule applied to VitalDB
test set (30% hold-out, n≈756 patients). PPV, NPV, LR projected to hypothetical
prevalences using Bayes theorem. NNS = number of screen-positive patients needed
to identify one true hypotension event.*

## Methods: Lead time analysis

Prediction horizon was characterised using back-extrapolation of 30-minute
aggregate features. For each event window, *_mean, *_min, and *_max statistics
of autonomic metrics were shifted backward in time by L minutes using the
within-window linear slope (f̂(−L) ≈ f(0) − slope · L · 60 s). Slope features
and *_std statistics were held constant since within-window variability structure
is assumed stationary over the evaluation horizon. Lead times of 5, 10, 15, 20,
and 30 minutes were evaluated. Bootstrap 95% confidence intervals (B=500) were
computed by cluster resampling at patient level. This analysis provides a
conservative approximation of real-time alerting performance; prospective
evaluation in an alert-triggering system is required to validate clinical utility.
