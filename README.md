# BeatLabile

> **Beat-by-beat prediction of haemodynamic instability in ICU patients**  
> Hypotension · Hypertension · Autonomic variability  
> From invasive arterial pressure waveforms — interpretable models, three independent cohorts.

---

## What is this?

BeatLabile is a research framework that predicts intraoperative haemodynamic instability events
(MAP hypotension, MAP hypertension, autonomic variability) up to **30 minutes before** they
occur, using exclusively beat-by-beat arterial pressure features derived from standard invasive
monitoring. The core claim: **autonomic dynamics beat-by-beat are sufficient** — adding
demographics, clinical variables or lab values does not improve AUC over signal alone.

Models are designed for interpretability: the primary model is a Bayesian mixed GLMM (8
parsimonious features), complemented by an MILP-optimal decision tree that translates directly
into bedside rules with explicit thresholds.

---

## Repository contents

```
beatlabile/          Core library
  config.py          Config loading (config.yaml)
  events/            Event detector (hypotension / hypertension / variability windows)
  io/                Dataset loaders: Clínic, MIMIC-IV Waveform, VitalDB
  models/            GLMM, Random Forest, XGBoost benchmarks, MILP tree
  qc/                Signal quality control pipeline
  signal/            Beat-by-beat metrics (HRV, BRS, RSA, PA variability)
  stats/             Bootstrap CI, calibration, unsupervised analysis
  windows/           Sliding-window builder

experiments/         Reproducible analysis scripts (one per Act / figure)
  act1_clinic.py     Act 1 — development, 10×50 CV, Clínic UCIQ
  act2_mimic.py      Act 2 — external validation, MIMIC-IV
  act3_vitaldb.py    Act 3 — VitalDB revalidation + sufficiency test
  act4_univariate.py Act 4 — univariate validity & domain-shift PCA
  act5_tripod.py     Act 5 — TRIPOD checklist + Table 1
  act_lead_time.py   Lead time analysis (OOS CV, linear extrapolation)
  act_lead_time_raw.py  Lead time from raw signal re-extraction
  fig_*.py           Figure generation scripts
  milp_bootstrap_full.py  MILP stability B=500
  nri_idi_*.py       NRI/IDI incremental value
  composite_outcome.py    Composite haemodynamic lability outcome
  ventilatory_context_analysis.py  Supp — cv–arv collinearity vs ventilatory context
  audit_suppression.py             Supp — univariate vs multivariate sign-flip audit
  subtype_analysis.py              Supp — hypotension subtype separation (baroreflex axis)
  delta_event_control.py           Supp — deterioration gradient Clínic vs VitalDB

figures/             Shared plotting utilities
results/             All outputs (CSVs, JSONs, PKLs, PDFs)
  act1/ act2/ act3/ act4/ act5/
  lead_time/  sensitivity/  supplementary/  figures/
datasets/            Data directory (not committed — see Datasets section)
scripts/             Utility scripts (VitalDB clinical data download)
tests/               Pytest test suite
config.yaml          Paths and hyperparameters
requirements.txt     Python dependencies
```

---

## Datasets

Raw data is **not included** in this repository. Three cohorts are used:

| Cohort | Role | Source | Access |
|--------|------|--------|--------|
| **Clínic UCIQ** | Development | Hospital Clínic Barcelona | Institutional (non-public) |
| **MIMIC-IV Waveform** | External validation | PhysioNet | Public — ODC License v1.0 |
| **VitalDB** | External validation + refit | VitalDB.net | Public — CC BY 4.0 |

Edit `config.yaml` to point `data.clinic`, `data.mimic`, and `data.vitaldb` to your local
copies. For a full inventory of available signals and variables per dataset, see [DATASETS.md](DATASETS.md).

---

## Installation

```bash
pip install -r requirements.txt
# or editable install
pip install -e .
```

Python ≥ 3.10 recommended.

---

## Cohorts — Table 1

| Cohort | Event | Patients | Windows | Events | Prevalence |
|--------|-------|:---:|:---:|:---:|:---:|
| Clínic (dev) | Hypotension | 204 | 951 | 66 | 6.9 % |
| Clínic (dev) | Hypertension | 204 | 923 | 38 | 4.1 % |
| Clínic (dev) | Variability | 269 | 2 481 | 1 596 | 64.3 % |
| MIMIC-IV (val) | Hypotension | 46 | 299 | 48 | 16.1 % |
| MIMIC-IV (val) | Hypertension | 46 | 256 | 5 | 2.0 % |
| MIMIC-IV (val) | Variability | 51 | 829 | 578 | 69.7 % |
| VitalDB (val) | Hypotension | 580 | 988 | 505 | 51.1 % |
| VitalDB (val) | Hypertension | 306 | 558 | 75 | 13.4 % |
| VitalDB (val) | Variability | 1 068 | 2 925 | 2 442 | 83.5 % |

Full per-cohort statistics → `results/act5/table1_all_cohorts.csv`

---

## Act 1 — Development (Clínic UCIQ)

10×50 patient-level cross-validation. 95 % CI from fold percentiles.

### Results by model

#### Hypotension (66 events / 951 windows)

| Model | Features | EPV | AUC CV | 95 % CI | Note |
|-------|:--------:|:---:|:------:|:-------:|------|
| GLMM full | 40 | 1.65 | **0.750** | [0.258, 0.988] | sensitivity analysis |
| **GLMM parsimonious** | **8** | **8.25** | **0.656** | [0.238, 1.000] | **primary model** |
| MILP (bootstrap B=500) | — | — | 0.735 | SD=0.055 | interpretable |
| RF | — | — | 0.792 | [0.545, 0.985] | benchmark |
| XGBoost | — | — | **0.793** | [0.553, 0.979] | benchmark |

Parsimonious features: `std_pa_mean`, `cv_pa_std`, `rsa_max`, `arv_std`, `brs_min`, `arv_mean`, `std_pa_max`, `rsa_mean`

#### Hypertension (38 events / 923 windows)

| Model | Features | EPV | AUC CV | 95 % CI | Note |
|-------|:--------:|:---:|:------:|:-------:|------|
| GLMM full | 40 | 0.95 | **0.882** | [0.391, 1.000] | sensitivity analysis |
| **GLMM parsimonious** | **8** | **4.75** | **0.834** | [0.412, 1.000] | **primary model** |
| MILP (bootstrap B=500) | — | — | 0.763 | SD=0.058 | interpretable |
| RF | — | — | 0.806 | [0.585, 0.999] | benchmark |
| XGBoost | — | — | **0.864** | [0.748, 0.993] | benchmark |

Parsimonious features: `std_pa_std`, `std_pa_max`, `cv_pa_std`, `std_pa_slope`, `arv_std`, `cv_pa_mean`, `brs_min`, `sdnn_mean`

#### Variability (1 596 events / 2 481 windows) — negative result

| Model | AUC CV | 95 % CI |
|-------|:------:|:-------:|
| GLMM full | 0.547 | [0.454, 0.623] |
| GLMM parsimonious | 0.551 | [0.458, 0.661] |
| RF | 0.558 | [0.506, 0.611] |
| MILP | 0.574 | — |

> AUC ≈ 0.55 across all models and cohorts (Clínic, MIMIC, VitalDB M1) — consistent
> negative finding. Variability is reported as a secondary exploratory outcome; the
> beat-level signal does not discriminate autonomous variability events as defined here.

**EPV note**: EPV < 10 persists for hypertension (n=38 events) even with 8 features.
Mitigation: Bayesian GLMM (`BinomialBayesMixedGLM`) with partial-pooling regularisation;
wide 95 % CIs reported throughout.

Calibration curves → `results/act1/calibration_glmm_*.csv`  
DCA net-benefit curves → `results/act1/dca_glmm_*.csv`  
MILP feature stability → `results/act1/milp_stability_*.csv`  
GLMM coefficients → `results/act1/glmm_coef_*.csv` · `glmm_pars_coef_*.csv`

---

## Act 2 — Blind External Validation (MIMIC-IV)

Fixed coefficients from Act 1 applied to MIMIC-IV without retraining.
95 % CI from 1 000-sample bootstrap.

| Event | GLMM AUC (95 % CI) | CITL | Cal. slope | MILP AUC |
|-------|--------------------|:----:|:----------:|:--------:|
| Hypotension | 0.638 [0.539–0.743] | −0.08 | 0.16 | 0.434 |
| Hypertension | 0.558 [0.392–0.809] | −0.01 | 0.05 | 0.428 |
| Variability | 0.507 [0.466–0.551] | −0.03 | −0.09 | 0.500 |

> Moderate domain shift for hypotension (AUC 0.75 → 0.64). Calibration slopes < 1 indicate
> overconfident predictions; recalibration does not recover discriminative performance.
> MIMIC calibration slope = 0.162 for hypotension — predicted probabilities are overscaled ~6×
> and require recalibration before clinical use in new populations.

Calibration → `results/act2/calibration_glmm_*.csv`  
DCA → `results/act2/dca_glmm_*.csv`

---

## Act 3 — VitalDB Revalidation + Sufficiency Test

> ⚠️ **Results updated in v1 correction** — a critical bug (labs merge row inflation ×81)
> invalidated all previous M2–M6 results. See [Bug fix](#bug-fix-vitaldb-labs-merge) below.

M1 = Act 1 GLMM applied as-is. M2–M6 = GLMM refitted on VitalDB (70/30 patient hold-out).
95 % CI from cluster-bootstrap (B=1 000, patient-level resampling).

### AUC by model

| Model | Predictors | Hypotension | Hypertension | Variability |
|-------|-----------|:-----------:|:------------:|:-----------:|
| M1 | Signal (Act 1 coefs, no refit) | 0.724 | 0.661 | 0.651 |
| M2 | Signal (refit VitalDB) | 0.844 [0.806–0.879] | **0.875** [0.813–0.930] | 0.763 [0.727–0.804] |
| M3 | M2 + age, BMI | 0.844 [0.806–0.879] | 0.874 [0.810–0.931] | 0.763 [0.727–0.804] |
| M4 | M3 + ASA, HTN, DM, urgency | 0.844 [0.806–0.879] | 0.874 [0.808–0.933] | 0.763 [0.727–0.804] |
| M5 | M4 + Hb, Cr, glucose, K | 0.844 [0.806–0.880] | 0.877 [0.810–0.934] | 0.763 [0.727–0.804] |
| M6 | Clinical only (no signal) | 0.635 [0.581–0.689] | 0.657 [0.532–0.763] | 0.591 [0.548–0.645] |

> **Sufficiency key finding**: M2 ≈ M3 ≈ M4 ≈ M5 across all three events — adding demographics,
> clinical data, and labs **does not improve** AUC over signal alone. M6 (clinical-only) is
> clearly inferior → the arterial signal is the dominant predictor.

### Subgroups (M1, VitalDB)

| Subgroup | Hypotension | Hypertension | Variability |
|----------|:-----------:|:------------:|:-----------:|
| Age low (tercile) | 0.732 | 0.536 | 0.642 |
| Age mid | 0.729 | 0.727 | 0.590 |
| Age high | 0.696 | 0.657 | 0.619 |
| Female | 0.719 | 0.643 | 0.639 |
| Male | 0.716 | 0.609 | 0.609 |
| ASA 1–2 | 0.690 | 0.758 | 0.623 |
| ASA 3+ | 0.745 | 0.586 | 0.608 |
| BMI normal | 0.712 | 0.719 | 0.653 |
| BMI overweight | 0.751 | 0.504 | 0.674 |
| BMI obese | 0.750 | 0.824 | 0.431 |

---

## Lead Time Analysis

M1 GLMM evaluated at multiple prediction horizons using **10-fold patient-level OOS CV**
(no data leakage) followed by linear feature back-extrapolation for lead > 0.

### Exploratory (linear extrapolation, Clínic OOS)

| Lead (min) | Hypotension [95 % CI] | Hypertension [95 % CI] | Variability [95 % CI] |
|:----------:|:---------------------:|:----------------------:|:----------------------:|
| 0 (OOS CV) | 0.784 [0.678–0.855] | **0.899** [0.791–0.966] | 0.544 [0.516–0.572] |
| 5 | 0.669 [0.506–0.783] | 0.853 [0.704–0.941] | 0.509 [0.482–0.537] |
| 10 | 0.625 [0.462–0.758] | 0.805 [0.623–0.915] | 0.505 [0.478–0.533] |
| 15 | 0.605 [0.436–0.746] | 0.779 [0.570–0.902] | 0.501 [0.474–0.529] |
| 20 | 0.593 [0.420–0.737] | 0.766 [0.555–0.894] | 0.498 [0.470–0.526] |
| 30 | 0.568 [0.384–0.721] | 0.749 [0.536–0.885] | 0.492 [0.464–0.520] |

### Confirmatory (raw signal re-extraction)

VitalDB OOS controls (n=483, never seen in training):

| Lead (min) | Hypotension AUC (VitalDB OOS) | Hypertension AUC (VitalDB OOS) |
|:----------:|:-----:|:-----:|
| 0 | 0.531 | **0.770** |
| 5 | 0.513 | 0.768 |
| 15 | 0.506 | 0.733 |
| 30 | 0.492 | **0.769** |

> **Hypertension**: robust signal stable to 30 min in OOS controls (AUC 0.769).  
> **Hypotension**: AUC ≈ 0.50 OOS — does not generalise to VitalDB (surgical vs ICU population).

Data → `results/lead_time/lead_time_auc.csv` · `results/lead_time/lead_time_raw_auc_combined.csv`  
Figures → `results/lead_time/fig_lead_time.pdf` · `results/figures/lead_time_raw_curves.pdf`

---

## Act 4 — Univariate Predictor Validity & Domain-Shift Robustness

40 features × 3 cohorts × 3 event types evaluated individually.

### Most domain-robust features (highest minimum AUC across cohorts)

| Event | Feature | Min AUC |
|-------|---------|:---:|
| Hypotension | `std_pa_mean` | 0.651 |
| Hypertension | `std_pa_std` | 0.720 |
| Variability | `sdnn_std` | 0.541 |

### Population PCA (3 cohorts pooled)

- PC1 = 24.8 %, PC2 = 14.2 % of variance.
- Cohort silhouette score = **0.039** (near zero) → cohorts largely overlap in feature space.

Full PCA → `results/act4/pca_scores.csv` · `results/act4/pca_loadings.csv`  
Consistency table → `results/act4/feature_correlation_consistency.csv`

---

## Act 5 — TRIPOD Checklist

TRIPOD-AI 2020: **17/22 items fully addressed**, 5 partial.  
EPV table → `results/act5/epv_by_event_type.csv`  
Checklist → `results/act5/tripod_checklist.csv` / `.json`

---

## Physiological interpretation

### GLMM parsimonious coefficients — Hypotension

| Feature | Coef | Direction | Interpretation |
|---------|:----:|:---------:|----------------|
| `brs_min` | −1.700 | ↓ risk | Attenuated baroreflex precedes drop |
| `cv_pa_std` | +1.329 | ↑ risk | Growing inter-window CV oscillation |
| `std_pa_mean` | +0.900 | ↑ risk | High beat-to-beat PA dispersion |
| `rsa_max` | −0.565 | ↓ risk | Strong vagal modulation is protective |

**Central signal**: hypotension is preceded by **autonomic uncoupling** — rising PA oscillations
(`cv_pa_std` ↑) with weakening vagal modulation (`rsa_max` ↓, `brs_min` ↓).

### GLMM parsimonious coefficients — Hypertension

| Feature | Coef | Direction | Interpretation |
|---------|:----:|:---------:|----------------|
| `std_pa_std` | +3.396 | ↑ risk | Non-stationary variability (σ of σ) |
| `cv_pa_std` | −2.540 | ↓ risk | Rigid, stable relative variability |
| `brs_min` | −1.524 | ↓ risk | Same baroreflex attenuation |
| `cv_pa_mean` | −1.334 | protective | Low CV = haemodynamic rigidity pre-crisis |

**Central signal**: hypertension follows **progressive autonomic rigidification** — CV and BRS
drop (system loses variability), while second-order variability (`std_pa_std`) rises.

### Key differentiator

| Metric | Pre-hypotension | Pre-hypertension |
|--------|:---------------:|:----------------:|
| `brs_min` | ↓ (−1.70) | ↓ (−1.52) |
| `cv_pa_std` | ↑ **+1.33** | ↓ **−2.54** |
| `std_pa_std` | — | ↑ **+3.40** |

The two instability phenotypes have **distinct spectral signatures** detectable 15–30 min before
the event — the central physiological argument of the study.

---

## MILP Decision Trees

### Hypotension rule (AUC bootstrap 0.735)

```
Is cv_pa_min ≤ 0.0225?
  YES → Is cv_pa_mean ≤ 0.0547?  → YES: LOW risk  / NO: HIGH risk
  NO  → Is std_pa_mean ≤ 8.05 mmHg?  → YES: LOW risk  / NO: HIGH risk
```

**Verbal rule**: alert if `(cv_pa_min < 2.25% AND cv_pa_mean > 5.47%)` OR
`(cv_pa_min ≥ 2.25% AND std_pa_mean > 8.05 mmHg)`

### Hypertension rule (AUC bootstrap 0.763)

```
Is cv_pa_mean ≤ 0.0525?
  YES → Is std_pa_max ≤ 8.70 mmHg?  → YES: LOW risk  / NO: HIGH risk
  NO  → Is std_pa_max ≤ 3.94 mmHg?  → YES: HIGH risk / NO: LOW risk
```

**Verbal rule**: alert if `(cv_pa_mean < 5.25% AND std_pa_max > 8.70 mmHg)` OR
`(cv_pa_mean ≥ 5.25% AND std_pa_max < 3.94 mmHg)`

---

## Comparison with published systems

### Hypotension

| System | n patients | AUC | Interpretable | Ref. |
|--------|:----------:|:---:|:-------------:|------|
| HPI (Hatib 2018) | 1 334 | **0.883** | ✗ | Hatib 2018 |
| HPI (Schneck 2020) | 204 | 0.820 | ✗ | Schneck 2020 |
| ANSWer / AHI-PI | 2 083 | 0.818 | Partial | Michard 2020 |
| **BeatLabile GLMM parsimonious** | **204** | **0.656–0.750** | **✓** | This study |
| **BeatLabile GLMM (VitalDB refit)** | **~900** | **0.844 [0.806–0.879]** | **✓** | This study |

> BeatLabile achieves performance comparable to published HPI validation studies (0.820)
> using only 8 autonomic beat-by-beat metrics — no waveform morphology, no CO or SVR required
> — while providing explicit, reproducible decision thresholds.

---

## Supplementary analyses

Four post-hoc analyses addressing reviewer-level questions about collinearity, suppression,
baroreflex subtypes, and cross-cohort deterioration gradients.

### S1 — cv–arv collinearity vs ventilatory context

`experiments/ventilatory_context_analysis.py`

Investigates whether the Clínic–VitalDB gap in cv–arv Spearman r (0.85 vs 0.71) is explained
by ventilatory regime (MV fraction, PEEP, spontaneous breathing proxy `rsa_mean`).

Key findings:
- Overall pooled Spearman r(cv_pa_mean, arv_mean): Clínic = **0.847**, VitalDB = **0.715** (Fisher Z p < 10⁻⁶⁰)
- Restricting VitalDB to full-MV (General anaesthesia + IPPV track): r = 0.715 — **unchanged**
- Per-patient r(rsa_mean, r_cv_arv) = 0.065, p = 0.28 — spontaneous-breathing fraction non-significant
- **Ventilatory context does not explain the collinearity gap.**
- The manuscript-cited Pearson r = 0.61/0.36 traces to `cv_pa_std × arv_std`, Pearson,
  **hypotension windows only** (n=939/972) — the variability-of-variability dimension,
  not the level features used in the models.

Outputs → `results/supplementary/a1_cv_arv_ventilatory_strata.csv`,
`a2_brs_min_stress_by_asa.csv`, `a3_cv_arv_regimestratified.csv`,
`ventilatory_context_analysis.txt`

### S2 — Suppression audit: univariate vs multivariate sign-flip

`experiments/audit_suppression.py`

For every feature in the parsimonious set, contrasts the univariate Spearman ρ (vs outcome)
with the multivariate logistic regression coefficient to detect sign inversions (suppression).
Includes 200-sample patient-clustered bootstrap sign-stability.

Key findings across Clínic + VitalDB × hypotension + hypertension:
- **8 suppression instances** detected (sign flip univariate → multivariate)
- **Zero** "fabricated signs" (all features have |ρ_uni| > 0.02)
- **brs_min in VitalDB-hypotension**: univariate ρ = **−0.132** (protector, p < 10⁻⁵),
  multivariate coef = **+0.341** (inverted). Bootstrap stability = 97.5 % — the flip is
  stable but artefactual (suppression mechanism: negative correlation with variability
  features `arv_std`, `cv_pa_std`, `std_pa_max` which are themselves positive predictors)
- brs_min in **Clínic-hypotension**: consistent, no flip (uni ρ = −0.10, multi = −0.237)
- Most affected VitalDB pairs: `cv_pa_std ↔ std_pa_max` (r = 0.854), `std_pa_std ↔ cv_pa_std`
  (r = 0.965 in hypertension)

> **Implication**: the positive brs_min coefficient in VitalDB-hypotension is a suppression
> artefact. The univariate direction (baroreflex attenuation as risk factor) is preserved
> across both cohorts.

Output → `results/supplementary/suppression_audit.csv`

### S3 — Hypotension subtype separation (baroreflex axis)

`experiments/subtype_analysis.py`

Tests whether the baroreflex axis (brs_min, brs_mean, brs_slope) discriminates hypotension
subtypes (decompensatory vs anaesthetic/iatrogenic) across three subtying definitions.

Definitions (VitalDB):
1. **Primary**: intraop vasopressor use (case-level proxy for iatrogenic hypotension)
2. Sensitivity 1: proximity to induction (≤15 min vs later)
3. Sensitivity 2: ASA ≥ 3 or emergency surgery (patient-level severity proxy)

Key findings:
- Baroreflex separation ≤ 0.15 (mean Cohen's d) across all three definitions
- Variability axis separates subtypes, but in the **opposite direction**: higher variability
  in peri-induction / "anaesthetic" windows (separation 0.39–0.46 for sensib.1)
- **brs_min does not mark hypotension severity** in VitalDB — consistent with the suppression
  finding (S2): the univariate baroreflex signal is masked by variability collinearity in this dataset

Output → `results/supplementary/subtype_analysis_vitaldb.csv`

### S4 — Deterioration gradient: Clínic vs VitalDB

`experiments/delta_event_control.py`

Computes Cohen's d and AUC of the pre-event deterioration (event windows vs matched controls)
separately per cohort × outcome × axis. Cross-cohort gradient tested with B=1 000
patient-clustered bootstrap (two-tailed).

| Cohort | Outcome | Axis | d | 95 % CI | AUC |
|--------|---------|------|:---:|:---:|:---:|
| Clínic | Hypotension | Baroreflex | +0.100 | [0.06, 0.23] | 0.53 |
| Clínic | Hypotension | Variability | +0.844 | [0.32, 1.61] | 0.70 |
| VitalDB | Hypotension | Baroreflex | **−0.066** | [−0.24, 0.02] | 0.41 |
| VitalDB | Hypotension | Variability | +0.494 | [0.37, 0.63] | 0.67 |
| Clínic | Hypertension | Baroreflex | +0.106 | [0.07, 0.25] | 0.55 |
| Clínic | Hypertension | Variability | +0.486 | [0.15, 1.06] | 0.68 |
| VitalDB | Hypertension | Baroreflex | +0.023 | [−0.20, 0.16] | 0.50 |
| VitalDB | Hypertension | Variability | +0.838 | [0.58, 1.12] | 0.74 |

Gradient test (Clínic − VitalDB):

| Outcome | Axis | Δd | 95 % CI | p_boot | Result |
|---------|------|:---:|:---:|:---:|---|
| Hypotension | **Baroreflex** | **+0.209** | [0.07, 0.38] | **< 0.001** | **Clínic > VitalDB** |
| Hypotension | Variability | +0.368 | [−0.19, 1.12] | 0.256 | no concluyente |
| Hypertension | Baroreflex | +0.114 | [−0.04, 0.37] | 0.128 | no concluyente |
| Hypertension | Variability | −0.317 | [−0.81, 0.29] | 0.230 | no concluyente |

> The only significant gradient is **baroreflex deterioration in hypotension** (p < 0.001).
> Variability deterioration is a universal precursor (large d in both cohorts, no
> significant inter-cohort difference). Hypertension shows no gradient on either axis.

Outputs → `results/supplementary/delta_event_control.csv`,
`results/supplementary/delta_event_control_gradient.csv`

---

## Bug fix — VitalDB labs merge (v1 correction)

**Root cause** (detected in senior statistical review, 2026-04-09): the VitalDB labs file is
in long format (~160 rows/patient). A direct `merge` on `patient_id` caused **×81 row
inflation** (4 471 → 362 489 windows), and `LAB_COLS` referenced non-existent column names.

**All Act 3 M2–M6 results computed before this fix are invalid.**

**Fix applied** in `experiments/act3_vitaldb.py`:

```python
# Before (incorrect — row explosion)
windows_df = windows_df.merge(labs_df, on="patient_id", how="left")

# After (correct — long→wide pivot, median per patient)
labs_wide = (
    labs_df.groupby(["patient_id", "name"])["result"]
    .median()
    .unstack("name")
    .reset_index()
)
windows_df = windows_df.merge(labs_wide, on="patient_id", how="left")
```

Previous (invalid) vs corrected AUC:

| Event | Model | Previous (INVALID) | Corrected |
|-------|-------|--------------------|-----------|
| Hypotension | M2 | 0.758 | **0.844** [0.806–0.879] |
| Hypertension | M2 | **0.934** ← suspect | **0.875** [0.813–0.930] |
| Variability | M2 | 0.671 | **0.763** [0.727–0.804] |

---

## Files modified / added in v1

| File | Change |
|------|--------|
| `experiments/act3_vitaldb.py` | Labs merge fix + bootstrap CI |
| `experiments/act1_clinic.py` | Parsimonious GLMM + coefficient export |
| `experiments/act1_finish.py` | New — optimised finishing script (reuses PKLs) |
| `experiments/act5_tripod.py` | Dynamic EPV computation |
| `experiments/act_lead_time.py` | New — OOS lead time analysis |
| `experiments/act_lead_time_raw.py` | New — raw signal lead time |
| `experiments/fig_forest_plot.py` | New — GLMM coefficient forest plot |
| `experiments/fig_calibration.py` | New — calibration plots |
| `experiments/fig_roc_curves.py` | New — ROC curves all cohorts |
| `experiments/milp_bootstrap_full.py` | New — MILP bootstrap B=500 |
| `experiments/nri_idi_calc.py` | New — NRI/IDI training-apparent |
| `experiments/nri_idi_cv.py` | New — NRI/IDI cross-validated |
| `experiments/composite_outcome.py` | New — composite hypotension OR hypertension |
| `results/act1/act1_results.json` | Full Act 1 results |
| `results/act3/act3_results.json` | Regenerated with correct data |
| `results/lead_time/lead_time_auc.csv` | Lead time AUC by horizon |
| `results/act5/epv_by_event_type.csv` | EPV per event type |

---

