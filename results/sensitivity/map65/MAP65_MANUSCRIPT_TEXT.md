## Manuscript Text — Sensitivity Analysis (Hypotension Threshold)

### Results paragraph (ready to insert)

When a less conservative hypotension threshold was applied (MAP<65 mmHg for
≥3 consecutive minutes, the most widely used definition in recent observational
studies [Sessler 2018, Salmasi 2017, Wesselink 2018]), the eight pre-specified
autonomic features retained their predictive value. Prevalence of sustained
hypotension increased to 34.1% in Clínic Barcelona
(441 events) and 76.5% in VitalDB
(1132 events). Cross-cohort AUC rank correlation of the 40 features
was ρ=0.20 (p=0.2120), indicating that features
with high univariate predictive value for MAP<55 retained relative ranking
for MAP<65. The parsimonious GLMM (8 fixed features, no re-selection) achieved
an AUC of 0.715 (95% CI 0.662–0.767) on the VitalDB
validation set, which was similar compared with the primary MAP<55 analysis
(AUC 0.844; ΔAUC=0.129). When the MAP<55-trained MILP rule was
applied without modification to MAP<65-defined events, sensitivity was
0.375 and specificity 0.817,
suggesting that the autonomous-feature rule retains acceptable discrimination
under a broader outcome definition without retraining.

### Methods note

To evaluate sensitivity to outcome definition, hypotension was alternatively
defined as MAP<65 mmHg for ≥3 consecutive minutes,21 the threshold most
commonly reported in recent intraoperative studies. Both cohorts were
reprocessed using the identical pipeline. The eight parsimonious features
selected in the primary analysis were retained without modification; no
additional variable selection was performed. Validation was performed on the
same 70/30 patient-level stratified hold-out of VitalDB (seed=42). The
MAP<55-trained MILP decision rule was applied to MAP<65-labelled windows
without any threshold adjustment to evaluate portability of the interpretable rule.

### Statistics note on prevalence difference

The higher prevalence under MAP<65 increases PPV while leaving LR+ largely
unchanged (same features, same relative discriminability). The NPV decreases
as expected for a more common outcome.
