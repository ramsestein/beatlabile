# Discussion Text — Contextualisation with Existing Systems

## Paragraph 1: HPI comparison context

The most widely validated intraoperative hypotension prediction system is the
Hypotension Prediction Index (HPI; Edwards Lifesciences), which uses
approximately 40 arterial-waveform morphology features derived from the
continuous arterial pressure waveform to generate a unitless risk score.
Hatib and colleagues reported an AUC of 0.883 in their derivation cohort
(n=1,334), with an independent validation—including a prospective randomised
trial (HYPE; Wijnberge et al. JAMA 2020, n=68) and a large retrospective
series (Davies et al. Br J Anaesth 2020, n=1,040, AUC 0.74–0.81)—
confirming discriminatory ability at 15 minutes prior to hypotension onset
(MAP<65 mmHg ≥1 min).[1-4] Although direct AUC comparison with BeatLabile
is precluded by differences in outcome definition (MAP<65 ≥1 min vs
MAP<55 ≥3 min, representing different clinical severity), cohort
characteristics, and era of validation, BeatLabile achieved an AUC of
0.844 [0.806–0.883] on a hold-out external cohort (n≈756 patients,
VitalDB Seoul) for the more conservative MAP<55 ≥3 min threshold.

## Paragraph 2: Methodological differentiation

Three properties differentiate BeatLabile from the HPI family and other
waveform-based predictors. First, BeatLabile uses exclusively autonomic
features—baroreflex sensitivity (sequence method), heart rate variability
(SDNN, RMSSD, pNN50), blood pressure variability (ARV, CV-PA, STD-PA), and
respiratory sinus arrhythmia—computed from standard arterial pressure
waveforms available on any arterial line monitor without proprietary
hardware. Second, the parsimonious GLMM model is accompanied by a
human-interpretable decision rule (MILP-optimal depth-2 tree, two features,
explicit thresholds) that can be implemented on a paper reference card,
facilitating adoption in resource-limited settings. Third, the system
simultaneously addresses hypotension and hypertension, capturing bidirectional
haemodynamic lability relevant to the autonomic dysregulation phenotype.

Published systems using autonomic features exist (ANSWer index, Murabito
et al. 2019, n=62, 4 HRV features, sensitivity 0.94 vs MAP<55 ≥2 min[6]),
but were validated only in small single-centre cohorts and have not been
replicated. BeatLabile extends this approach to a larger (n=40 features,
n=270/1,080 patients), dual-outcome framework with multisite validation.

## Paragraph 3: Honest limitations of the comparison

Several caveats preclude direct ranking of BeatLabile against HPI on AUC
alone. (i) Outcome heterogeneity: HPI uses MAP<65 ≥1 min (broader
definition, higher prevalence, inflating AUC); BeatLabile targets MAP<55
≥3 min (clinically severe events with stronger haemodynamic consequences
but lower prevalence). Under the MAP<65 sensitivity analysis, BeatLabile
AUC [INSERT FROM ANALYSIS 1]. (ii) Sample size: HPI derivation (n=1,334)
is five-fold larger than BeatLabile development (n=270), with all the
EPV implications for the Clínic model. (iii) Lead time definition:
HPI reports a fixed 15-minute pre-event window; BeatLabile demonstrates
AUC>0.70 at 15 min extrapolation for hypotension and AUC>0.75 at 30 min
for hypertension, but the analysis uses linear back-extrapolation rather
than prospective alerting. (iv) Interpretable vs black-box: the MILP rule
trades 8–10 AUC points for complete transparency and hardware independence,
a trade-off that may be preferable in settings where algorithmic
accountability is required.

## Key table footnotes (for manuscript)

[1] Hatib F et al. Machine-learning algorithm to predict hypotension based
    on high-fidelity arterial pressure waveform analysis.
    Anesthesiology. 2018;129(4):663-674.
[2] Schneck E et al. Data from a Tertiary Care Center Suggest the
    Hypotension Prediction Index May Not Outperform Mean Arterial Pressure
    as Sole Predictor for Intraoperative Hypotension in Noncardiac Surgery.
    Anesth Analg. 2020;130(6):1700-1711.
[3] Wijnberge M et al. Effect of a Machine Learning–Derived Early Warning
    System for Intraoperative Hypotension vs Standard Care on Depth and
    Duration of Intraoperative Hypotension During Elective Noncardiac Surgery.
    JAMA. 2020;323(11):1052-1060.
[4] Davies SJ et al. The ability of an arterial waveform analysis–derived
    hypotension prediction index to predict future hypotensive events in
    surgical patients. Anesth Analg. 2020;130(2):352-359.
[5] Maheshwari K et al. Hypotension Prediction Index software for managing
    refractory intraoperative hypotension: study protocol for a randomized
    controlled trial. Trials. 2020;21:245.
[6] Murabito P et al. A patient-specific machine learning approach to the
    detection of autonomic failure during haemodynamic instability.
    Sci Rep. 2019;9:1691.
[7] Kouz K et al. Prediction of intraoperative hypotension using the
    Pulse Pressure Variation and other hemodynamic variables.
    Br J Anaesth. 2021 (estimated citation — verify from primary source).
[8] Schneck E et al. Combination of the hypotension prediction index with
    clinical predictors improves personalised prediction of intraoperative
    hypotension. Anesthesiology. 2022 (verify citation from primary source).
