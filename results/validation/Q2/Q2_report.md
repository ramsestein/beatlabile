# Q2 Analysis Report

Generated: 2026-05-08 17:42:01  
Elapsed: 4s

## 1. Hypothesis

**Q2**: Validate the autonomic pre-hypotension signature of BeatLabile
in the minutes preceding a vasopressor bolus.

- **Pre-specified direction**: ALL features expected to DECREASE (β < 0)
  relative to a quiescent control window.
- **Bonferroni α** = 0.010 (5 primary features)

## 2. Events (PASO 1)

| Metric | N |
|---|---|
| Raw vasopressor boluses | 19 |
| Clean (all criteria) | 10 |
| With valid control window | 6 |

### 2.1 Exclusion log

| patient_id | drug | t_bolus (s) | reason |
|---|---|---|---|
| 4247699 | efedrina | 714 | infusion_change_pm5min |
| 4247699 | efedrina | 1154 | infusion_change_pm5min |
| 5020549 | efedrina | 3746 | invalid_pre_features (ptt_cv_valid_pre=0.00) |
| 5431482 | efedrina | 3433 | invalid_pre_features (ptt_cv_valid_pre=0.13) |
| 5431482 | efedrina | 4290 | invalid_pre_features (ptt_cv_valid_pre=0.13) |
| 5582912 | efedrina | 3233 | pain_stimulus_in_pre5min |
| 5589679 | efedrina | 1767 | infusion_change_pm5min |
| 70297385 | fenilefrina | 3290 | outside_AG_window (t=3290, AG=[343,2967]) |
| 70551555 | fenilefrina | 1257 | invalid_pre_features (ptt_cv_valid_pre=0.00) |

### 2.2 Clean events

| pid | drug | group | t_bolus (s) | t_from_AG (s) | cum_eph (mg) | cum_phen (µg) |
|---|---|---|---|---|---|---|
| 4214722 | efedrina | interescalenico | 2765 | 2501 | 0.0 | 0.0 |
| 4247699 | efedrina | interescalenico | 2130 | 2072 | 0.0 | 0.0 |
| 4912692 | efedrina | supra_axilar | 747 | 185 | 0.0 | 0.0 |
| 4912692 | efedrina | supra_axilar | 1177 | 615 | 0.0 | 0.0 |
| 5362391 | efedrina | interescalenico | 1729 | 1383 | 0.0 | 0.0 |
| 5582912 | efedrina | interescalenico | 714 | 405 | 0.0 | 0.0 |
| 5589679 | efedrina | interescalenico | 2385 | 1709 | 0.0 | 0.0 |
| 5589679 | efedrina | interescalenico | 2811 | 2135 | 0.0 | 0.0 |
| 70551555 | fenilefrina | supra_axilar | 1730 | 1176 | 0.0 | 0.0 |
| 720142 | efedrina | interescalenico | 2052 | 1267 | 0.0 | 0.0 |

## 3. Q2a – Confirmatory Tests (PASO 3)

**Design**: GLMM (random intercept per patient), one-sided H1: β < 0
**delta** = feature_pre − feature_control

### 3.1 Primary features

| Feature | n_pairs | β | IC 95% | p_1sided | p_Bonf | d_z | Verdict |
|---|---|---|---|---|---|---|---|
| ptt_cv__mean | 6 | -0.0214 | [-0.2118, +0.1691] | 0.3924 | 1.0000 | -0.118 | **NS** |
| ptt_cv__std | 6 | -0.0328 | [-0.1210, +0.0554] | 0.1916 | 0.9579 | -0.390 | **NS** |
| ptt_std__max | 6 | -42.5499 | [-129.4323, +44.3326] | 0.1318 | 0.6591 | -0.514 | **NS** |
| pai_mean__mean | 6 | -0.9691 | [-9.2806, +7.3425] | 0.3882 | 1.0000 | -0.122 | **NS** |
| brs_alpha_lf__min | 6 | +0.1109 | [-0.6080, +0.8298] | 0.6460 | 1.0000 | +0.162 | **NS** |

### 3.2 Summary

**0 / 5 primary features validated** (Bonferroni α=0.010)

**Overall Q2a verdict: NO VALIDADO (0/5)**

### 3.3 Exploratory features

| Feature | n_pairs | β | p_2sided | d_z |
|---|---|---|---|---|
| ptt_std__std | 6 | -15.3095 | 0.1743 | -0.646 |
| ptt_std__slope | 6 | -0.1799 | 0.0181 | -1.412 |
| ptt_arv__std | 6 | -15.9704 | 0.3198 | -0.451 |

## 4. Q2b – Descriptive Trajectories (PASO 4)

**delta_post** = feature_post − feature_pre (vasopresor response)

| Feature | n_pairs | β_post | IC 95% | p_2sided | d_z |
|---|---|---|---|---|---|
| ptt_cv__mean | 10 | -0.0839 | [-0.2312, +0.0634] | 0.2296 | -0.408 |
| ptt_cv__std | 10 | -0.0160 | [-0.0877, +0.0557] | 0.6264 | -0.159 |
| ptt_std__max | 10 | -8.2141 | [-75.3275, +58.8994] | 0.7881 | -0.088 |
| pai_mean__mean | 10 | -1.3560 | [-5.8694, +3.1574] | 0.5138 | -0.215 |
| brs_alpha_lf__min | 10 | -0.0062 | [-0.1747, +0.1623] | 0.9357 | -0.026 |
| ptt_std__std | 10 | -1.0779 | [-19.0135, +16.8577] | 0.8949 | -0.043 |
| ptt_std__slope | 10 | +0.0687 | [-0.2630, +0.4004] | 0.6505 | +0.148 |
| ptt_arv__std | 10 | -6.1581 | [-28.5594, +16.2433] | 0.5495 | -0.197 |

## 5. Gradient Analysis (PASO 5a)

Group modifier: interescalénico vs supra-axilar.
Expected: interescalénico **amplifies** pre-hypotensive signature (β_interesk < 0).

| Feature | β_interesk | IC 95% | p_interaction | Interpretation |
|---|---|---|---|---|
| ptt_cv__mean | -0.0646 | [-0.6926, +0.5634] | 0.8402 | interesk_amplifies |
| ptt_cv__std | -0.0773 | [-0.2552, +0.1007] | 0.3946 | interesk_amplifies |
| ptt_std__max | -35.2679 | [-209.8396, +139.3037] | 0.6921 | interesk_amplifies |
| pai_mean__mean | +8.8316 | [-0.9554, +18.6187] | 0.0770 | interesk_attenuates |
| brs_alpha_lf__min | +0.5899 | [-1.1730, +2.3528] | 0.5119 | interesk_attenuates |

## 6. Sensitivity Analyses (PASO 5b)

### SA1_no_partial
| Feature | n_pairs | β | p_1sided | d_z |
|---|---|---|---|---|
| ptt_cv__mean | 6 | -0.0214 | 0.3924 | -0.118 |
| ptt_cv__std | 6 | -0.0328 | 0.1916 | -0.390 |
| ptt_std__max | 6 | -42.5499 | 0.1318 | -0.514 |
| pai_mean__mean | 6 | -0.9691 | 0.3882 | -0.122 |
| brs_alpha_lf__min | 6 | +0.1109 | 0.6460 | +0.162 |

### SA2_efedrina_only
| Feature | n_pairs | β | p_1sided | d_z |
|---|---|---|---|---|
| ptt_cv__mean | 5 | -0.0145 | 0.4403 | -0.072 |
| ptt_cv__std | 5 | -0.0272 | 0.2738 | -0.293 |
| ptt_std__max | 5 | -29.5379 | 0.2413 | -0.346 |
| pai_mean__mean | 5 | -0.9307 | 0.4129 | -0.105 |
| brs_alpha_lf__min | 5 | +0.1270 | 0.6354 | +0.166 |

### SA3_interesk_only
| Feature | n_pairs | β | p_1sided | d_z |
|---|---|---|---|---|
| ptt_cv__mean | 3 | -0.0551 | 0.3707 | -0.219 |
| ptt_cv__std | 3 | -0.0716 | 0.1649 | -0.737 |
| ptt_std__max | 3 | -67.3864 | 0.1731 | -0.706 |
| pai_mean__mean | 3 | +3.4468 | 0.9472 | +1.632 |
| brs_alpha_lf__min | 3 | +0.2685 | 0.6500 | +0.257 |

## 7. Dissociation Q1 vs Q2 (PASO 6)

See figure `figures/q1_vs_q2_dissociation.png`.

| Feature | Q1 β (pain) | Q2 β (pre-HTN) | Dissociation |
|---|---|---|---|
| ptt_cv__mean | +0.0024 | -0.0214 | OPPOSITE |
| ptt_cv__std | -0.0227 | -0.0328 | SAME_SIGN |
| ptt_std__max | -15.8813 | -42.5499 | SAME_SIGN |
| pai_mean__mean | +0.0147 | -0.9691 | OPPOSITE |
| brs_alpha_lf__min | +0.2655 | +0.1109 | SAME_SIGN |

## 8. Figures

| File | Description |
|---|---|
| figures/q2_pre_vs_control_violins.png | Q2a pre vs control violins |
| figures/q2_pre_vs_post_trajectories.png | Q2b ±5 min trajectories |
| figures/q2_forest_plot.png | Q2a forest plot |
| figures/q1_vs_q2_dissociation.png | Q1 vs Q2 dissociation (KEY) |

---
*Report generated by `validation/q2_main.py` — pre-specified analysis.*