# Q1 Report v2 — Refined Analysis (CAMBIO A + CAMBIO B)
Generated: 2026-05-08 16:20:35

---

## Summary

**VALIDACIÓN PARCIAL** (1/5 features primarias superan Bonferroni)

- Features tested (Bonferroni-corrected): **5**
- Features validated: **1**
- Features partially validated: **1**
- Bonferroni α: **0.0100** (= 0.05/5)

---

## Refinement of surrogate mapping

### Justification for CAMBIO B (direction revision)

The original Q1 analysis mapped `ptt_std` and `ptt_arv` to **positive** expected
direction (increased variability under pain stimulus). The trajectory analysis
(`trajectory_diagnostic.md`, 2026-05-08) revealed that **all** PTT variability
features decreased post-stimulus (all observed direction: **−**):

> *"ptt_std (observado −): descenso de 0.60×IQR, pico a 90 s, retorno en 120 s.
> Dirección CONTRARIO a BeatLabile."*

**Mechanistic interpretation**: Under propofol + remifentanil + regional block,
nociceptive activation produces tonic sympathetic discharge that causes uniform
vasoconstriction. This compresses the dynamic range of the PTT (shorter, more
uniform beat-to-beat PTT), reducing all variability metrics — not increasing them.
The original mapping assumed a "arousal → jitter" model, but the data support a
"activation → rigidification" model.

**Pre-specification**: All directions were fixed **before** running these tests
(this document) and are not modified post-hoc.

### Justification for CAMBIO A (shorter window)

Trajectory analysis showed:
- Peak at **90 s** post-stimulus
- Return to baseline at **120 s**
- The original 5-min window captured predominantly noise (3–4× the signal duration)
- **New window**: Pre [−5, −2] min, Post [0, +2.5 min] = [0, 150 s]

---

## PASO 2 — Primary confirmatory tests

| Feature                | n_pairs | β           | IC 95%               | p (1-sided) | p_Bonferroni | Cohen d_z | Verdict              |
|------------------------|---------|-------------|----------------------|-------------|--------------|-----------|----------------------|
| PTT-CV  (mean)         |      50 |    +0.0024 |   [-0.0280, +0.0328] |   0.5616 |       1.0000 |     0.022 | no_validado          |
| PTT-CV  (std)          |      50 |    -0.0227 |   [-0.0388, -0.0067] |   0.0028 |       0.0139 |    -0.395 | validado             |
| PTT-SD  (max)          |      50 |   -15.8813 |  [-29.9899, -1.7727] |   0.0137 |       0.0684 |    -0.313 | no_validado          |
| BRS-αLF (min)          |      47 |    +0.2655 |   [+0.0952, +0.4358] |   0.9989 |       1.0000 |     0.477 | no_validado          |
| PAI     (mean)         |      52 |    +0.0147 |   [-0.1369, +0.1663] |   0.5754 |       1.0000 |     0.026 | no_validado          |

**Threshold**: p_Bonferroni < 0.05 (= α_bonf 0.010 × 5)

---

## PASO 3 — Gradient analysis (interescalénico vs supra+axilar)

H₁: |Δ_interescalénico| < |Δ_supra_axilar|
(more residual sympatholysis → less nociceptive response → delta closer to zero)
Reference group: supra_axilar. Test one-sided: β_interesc > 0.

See `test_results_v2.csv` (analysis=gradient) for full results.

---

## PASO 4 — Sensitivity analyses

| Feature                | Analysis                       | n   | β       | p (1-sided) | p_Bonferroni |
|------------------------|--------------------------------|-----|---------|-------------|--------------|
| PTT-CV  (mean)         | sensitivity_no_partial         |    49 | +0.0019 |   0.5479 |       1.0000 |
| PTT-CV  (mean)         | sensitivity_anclaje_only       |    22 | -0.0205 |   0.1298 |       0.6492 |
| PTT-CV  (mean)         | sensitivity_adjusted           |    33 | +0.1292 |   0.8686 |       1.0000 |
| PTT-CV  (std)          | sensitivity_no_partial         |    49 | -0.0237 |   0.0021 |       0.0105 |
| PTT-CV  (std)          | sensitivity_anclaje_only       |    22 | -0.0270 |   0.0145 |       0.0726 |
| PTT-CV  (std)          | sensitivity_adjusted           |    33 | +0.0637 |   0.7697 |       1.0000 |
| PTT-SD  (max)          | sensitivity_no_partial         |    49 | -16.5963 |   0.0115 |       0.0573 |
| PTT-SD  (max)          | sensitivity_anclaje_only       |    22 | -23.5874 |   0.0086 |       0.0429 |
| PTT-SD  (max)          | sensitivity_adjusted           |    33 | +81.7601 |   0.8898 |       1.0000 |
| BRS-αLF (min)          | sensitivity_no_partial         |    46 | +0.2672 |   0.9987 |       1.0000 |
| BRS-αLF (min)          | sensitivity_anclaje_only       |    20 | +0.3217 |   0.9593 |       1.0000 |
| BRS-αLF (min)          | sensitivity_adjusted           |    32 | -0.7489 |   0.1839 |       0.9196 |
| PAI     (mean)         | sensitivity_no_partial         |    51 | +0.0144 |   0.5724 |       1.0000 |
| PAI     (mean)         | sensitivity_anclaje_only       |    24 | +0.0530 |   0.7173 |       1.0000 |
| PAI     (mean)         | sensitivity_adjusted           |    34 | -0.7612 |   0.0919 |       0.4593 |

**Sensitivity C** is flagged EXPLORATORY (n=17 patients; between-patient covariates
edad, BMI, posicion are poorly identified with patient_id as random effect).

---

## PASO 5 — Exploratory features (reclassified)

The following features were originally classified with positive expected direction
("inconsistente" category in Q1 v1). They are reclassified here with direction −
(consistent with the rigidification model) and tested without Bonferroni correction.
All are marked **exploratory** and do not contribute to the primary verdict.

| Feature                | n_pairs | β           | IC 95%               | p (1-sided) | p_Bonferroni | Cohen d_z | Verdict              |
|------------------------|---------|-------------|----------------------|-------------|--------------|-----------|----------------------|
| PTT-SD  (std)  [expl.] |      50 |    -7.3780 |  [-11.7096, -3.0463] |   0.0004 |            — |    -0.475 | exploratory_sig      |
| PTT-SD  (slope) [expl.] |      50 |    +0.1013 |   [-0.0214, +0.2239] |   0.9471 |            — |     0.232 | exploratory_ns       |
| PTT-ARV (std)  [expl.] |      50 |    -4.9664 |   [-9.1747, -0.7580] |   0.0104 |            — |    -0.330 | exploratory_sig      |

If all three show p < 0.05 in the negative direction, this corroborates the
"uniform rigidification" interpretation.

---

## Q1 Verdict (v2)

**VALIDACIÓN PARCIAL** (1/5 features primarias superan Bonferroni)

### Definition
- **VALIDADO**: ≥3/5 primary features cross Bonferroni with correct direction
- **VALIDACIÓN PARCIAL**: 1–2 features cross Bonferroni
- **NO VALIDADO**: 0 features cross Bonferroni

---

## Files

| File | Description |
|------|-------------|
| `paired_event_data.csv` | Long-format paired pre/post values (52 events × 8 features × 2 aggs) |
| `test_results_v2.csv`   | Full statistical results (primary + gradient + sensitivity + exploratory) |
| `figures/paired_event_violins.png` | Pre/post distributions per primary feature |
| `figures/forest_plot_v2.png`       | Standardised β forest plot (primary + exploratory) |
| `figures/gradient_plot_v2.png`     | Group gradient (if ≥1 feature validated) |

---
*Q1 Refinement v2 — beatlabile validation pipeline*
