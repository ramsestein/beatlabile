# Analysis 4: BRS Calculability — Methods & Results Text

## Methods (2–3 sentences for Supplementary Methods)

Baroreflex sensitivity (BRS) was estimated using the sequence method applied to
30-second sliding windows (step = 1 beat). A sequence was considered valid if
≥3 consecutive beats showed concordant directional change in systolic arterial
pressure and RR interval with Pearson correlation r≥0.6. BRS was designating as
non-calculable (NaN) for any 30-second window in which no qualifying sequence
was identified; the corresponding sliding-window metric was excluded from the
30-minute aggregate features. Prediction windows in which BRS was non-calculable
across all constituent 30-second windows (brs_mean = NaN) were retained in
analyses, with missing BRS features imputed by column median.

## Results: BRS calculability (ready to insert)

BRS was calculable in 96.6% of 30-minute prediction windows in
Clínic Barcelona and 95.8% in VitalDB (hypotension windows; all
windows: Clínic 97.0% [hypertension] and VitalDB
97.3%). At the patient level, 96% of Clínic
patients (195/204) and 95% of VitalDB patients
(550/580) had BRS calculable in ≥50% of their prediction windows.
In a sensitivity analysis excluding patients with <50% BRS calculability,
GLMM AUC on the VitalDB hold-out was 0.768 [0.712–0.824] for hypotension and
0.839 [0.752–0.919] for hypertension (vs 0.750 [0.692–0.805] and 0.817 [0.687–0.907] respectively
in the full cohort), demonstrating that BRS calculability had negligible
influence on model performance.

## Key interpretation

- If brs_calculable_pct > 90%: "BRS was calculable in >90% of prediction
  windows in both cohorts, confirming that the signal duration (30 seconds,
  approximately 30–80 beats) is sufficient for the sequence method in the
  vast majority of perioperative patients."

- If brs_calculable_pct 70–90%: "BRS calculability was acceptable but reduced
  in a minority of windows. Imputation by column median minimised data loss."

- If brs_calculable_pct < 70%: "BRS calculability was limited, potentially
  reflecting high ectopic burden, arrhythmia, or signal artefact in a subset
  of patients. Given that BRS features appear only once in the parsimonious
  hypotension model (brs_min) and once in the hypertension model (brs_min),
  and given that the sensitivity analysis excluding low-BRS patients shows
  similar AUC, the results are robust to this limitation."
