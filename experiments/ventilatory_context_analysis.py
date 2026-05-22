"""Ventilatory Context Analysis
================================
Three focused analyses addressing cross-cohort differences in cv–arv collinearity,
brs_min sign/magnitude association with homeostatic stress, and cv-flip persistence
under ventilatory regime restriction.

Questions addressed
-------------------
1. **CV–ARV colinearity vs ventilatory context** (r observed: 0.61 vs 0.36 across cohorts)
   - Correlate the in-window cv_pa_mean × arv_mean Spearman r (computed per patient)
     with: ane_type, emop (emergency), PEEP availability as proxy for IPPV,
     intraoperative vasopressor exposure (eph+phe+epi), and rsa_mean as proxy for
     spontaneous breathing fraction.
   - Does the cohort-level divergence (clinic/MIMIC ≈ 0.85–0.93 vs VitalDB ≈ 0.71)
     vanish within strata?

2. **BRS-min sign/magnitude vs homeostatic stress markers**
   - For VitalDB: merge brs_min per window with ASA, emop, vasopressor exposure.
   - Test whether brs_min < 0 (sign flip) or |brs_min| is associated with
     higher ASA, emergency surgery, or vasopressor use.

3. **CV flip restricted to single ventilatory regime (mechanical ventilation only)**
   - Filter VitalDB windows to cases that have IPPV track (Primus/SET_RR_IPPV)
     and exclude spinal/sedation (ane_type == "General" only).
   - Recompute cv_pa_mean × arv_mean correlation in this restricted set and
     compare to the clinic (all-MV) and MIMIC (ICU-MV) cohorts.

Output: results/supplementary/ventilatory_context_analysis.csv (tables)
        results/supplementary/ventilatory_context_analysis.txt (narrative report)
"""
from __future__ import annotations

import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("results")
CACHE_DIR     = RESULTS_DIR / "cache"
OUT_DIR       = RESULTS_DIR / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRACKS_CSV    = Path("datasets/data/vitaldb/clinical_data/tracks.csv")
CASES_CSV     = Path("datasets/data/vitaldb/clinical_data/cases.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────

def spearman_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (r, p) after removing NaNs from both arrays pairwise."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return np.nan, np.nan
    return stats.spearmanr(x[mask], y[mask])


def fisher_z_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95 % CI for Spearman r via Fisher Z transform."""
    if n < 4:
        return np.nan, np.nan
    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return lo, hi


def compare_rs(r1: float, n1: int, r2: float, n2: int) -> float:
    """Two-sample Z test for difference between two Spearman rs (Fisher Z)."""
    if any(np.isnan(v) for v in [r1, n1, r2, n2]) or n1 < 4 or n2 < 4:
        return np.nan
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z  = (z1 - z2) / se
    return 2 * stats.norm.sf(abs(z))


def report_r(r, n):
    lo, hi = fisher_z_ci(r, n)
    return f"r={r:.3f} (95%CI {lo:.3f}–{hi:.3f}), n={n}"


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading cached window data …")
w_clinic  = pd.read_parquet(CACHE_DIR / "clinic_windows.parquet")
w_mimic   = pd.read_parquet(CACHE_DIR / "mimic_windows.parquet")
w_vitaldb = pd.read_parquet(CACHE_DIR / "vitaldb_windows.parquet")

# ── Load VitalDB clinical metadata ────────────────────────────────────────────
print("Loading VitalDB clinical metadata …")
cases = pd.read_csv(CASES_CSV)
cases["caseid_int"] = cases["caseid"].astype(int)

# Create vasopressor_any flag and total dose
cases["vaso_total"]  = cases[["intraop_eph","intraop_phe","intraop_epi"]].sum(axis=1)
cases["vaso_any"]    = (cases["vaso_total"] > 0).astype(int)
# Normalise dose > 0 only within non-zero users (to reduce zero-inflation)
cases["vaso_nonzero"] = cases["vaso_total"].where(cases["vaso_total"] > 0)

# Convert VitalDB patient_id → caseid integer
w_vitaldb = w_vitaldb.copy()
w_vitaldb["caseid_int"] = w_vitaldb["patient_id"].astype(str).str.lstrip("0").astype(int)

# Merge clinical metadata onto VitalDB windows
meta_cols = ["caseid_int","asa","emop","ane_type","vaso_total","vaso_any","vaso_nonzero",
             "intraop_eph","intraop_phe","intraop_epi"]
w_vitaldb = w_vitaldb.merge(cases[meta_cols], on="caseid_int", how="left")

# ── Load tracks to identify IPPV cases ────────────────────────────────────────
print("Loading VitalDB tracks for ventilation mode …")
tracks = pd.read_csv(TRACKS_CSV)

# Cases with IPPV controlled track (Primus/SET_RR_IPPV) → most likely volume/pressure control
ippv_cases = set(tracks[tracks["tname"] == "Primus/SET_RR_IPPV"]["caseid"].tolist())
# Cases with measurable RR (spontaneous trigger possible): if meas_rr >> set_rr → spontaneous
w_vitaldb["has_ippv_track"] = w_vitaldb["caseid_int"].isin(ippv_cases).astype(int)

# PEEP median per case from tracks metadata
# As a proxy: presence of PEEP track alone doesn't tell us value; use ane_type + IPPV
# Full MV proxy = General anesthesia + IPPV track
w_vitaldb["full_mv"] = (
    (w_vitaldb["ane_type"] == "General") &
    (w_vitaldb["has_ippv_track"] == 1)
).astype(int)

# ── Per-patient cv–arv correlation ────────────────────────────────────────────
# For each patient, compute Spearman r(cv_pa_mean, arv_mean) across all their windows

def per_patient_corr(df: pd.DataFrame, id_col="patient_id") -> pd.DataFrame:
    records = []
    for pid, grp in df.groupby(id_col, observed=True):
        r, p = spearman_r(grp["cv_pa_mean"].values, grp["arv_mean"].values)
        records.append({"patient_id": pid, "r_cv_arv": r, "p_cv_arv": p, "n_windows": len(grp)})
    return pd.DataFrame(records)


pp_clinic  = per_patient_corr(w_clinic)
pp_mimic   = per_patient_corr(w_mimic)
pp_vitaldb = per_patient_corr(w_vitaldb)

# Merge VitalDB per-patient correlations with case metadata (one row per patient)
meta_pt = (
    w_vitaldb.groupby("patient_id", observed=True)
    .agg(
        caseid_int    = ("caseid_int",  "first"),
        asa           = ("asa",         "first"),
        emop          = ("emop",        "first"),
        ane_type      = ("ane_type",    "first"),
        vaso_any      = ("vaso_any",    "first"),
        vaso_total    = ("vaso_total",  "first"),
        full_mv       = ("full_mv",     "first"),
        rsa_mean_pt   = ("rsa_mean",    "mean"),   # average RSA across all patient windows
    )
    .reset_index()
)
pp_vitaldb = pp_vitaldb.merge(meta_pt, on="patient_id", how="left")

# ── Cohort-level summary ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("OVERALL CV–ARV SPEARMAN r PER COHORT (pooled over all windows)")
print("="*70)

overall_results = {}
for label, df in [("Clinic", w_clinic), ("MIMIC", w_mimic), ("VitalDB", w_vitaldb)]:
    r, p = spearman_r(df["cv_pa_mean"].values, df["arv_mean"].values)
    n = len(df)
    lo, hi = fisher_z_ci(r, n)
    overall_results[label] = (r, n)
    print(f"  {label:10s}: r={r:.4f} (95%CI {lo:.3f}–{hi:.3f}), n={n}, p={p:.2e}")

print()
p_cl_vdb = compare_rs(*overall_results["Clinic"], *overall_results["VitalDB"])
p_mi_vdb = compare_rs(*overall_results["MIMIC"], *overall_results["VitalDB"])
print(f"  Clinic vs VitalDB two-sided Fisher Z test: p={p_cl_vdb:.4e}")
print(f"  MIMIC  vs VitalDB two-sided Fisher Z test: p={p_mi_vdb:.4e}")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — CV–ARV collinearity vs ventilatory context in VitalDB
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 1: CV–ARV colinearity vs ventilatory context (VitalDB)")
print("="*70)

records_a1 = []

# 1a) Stratify by ane_type
print("\n--- 1a) By anesthesia type ---")
for atype, grp in w_vitaldb.groupby("ane_type", observed=True):
    r, p   = spearman_r(grp["cv_pa_mean"].values, grp["arv_mean"].values)
    n      = len(grp)
    lo, hi = fisher_z_ci(r, n)
    print(f"  {atype:25s}: {report_r(r, n)}")
    records_a1.append({"stratum":"ane_type", "level": atype, "r_cv_arv": r, "n": n, "ci_lo": lo, "ci_hi": hi, "p": p})

# 1b) Stratify by emergency surgery
print("\n--- 1b) By emergency surgery (emop) ---")
for emop_val, label in [(0, "Elective"), (1, "Emergency")]:
    grp    = w_vitaldb[w_vitaldb["emop"] == emop_val]
    r, p   = spearman_r(grp["cv_pa_mean"].values, grp["arv_mean"].values)
    n      = len(grp)
    lo, hi = fisher_z_ci(r, n)
    print(f"  {label:25s}: {report_r(r, n)}")
    records_a1.append({"stratum":"emop", "level": label, "r_cv_arv": r, "n": n, "ci_lo": lo, "ci_hi": hi, "p": p})

# 1c) Full MV (General + IPPV track) vs rest
print("\n--- 1c) Full mechanical ventilation (General + IPPV) vs other ---")
for mv_val, label in [(1, "Full MV (General+IPPV)"), (0, "Other/Partial")]:
    grp    = w_vitaldb[w_vitaldb["full_mv"] == mv_val]
    r, p   = spearman_r(grp["cv_pa_mean"].values, grp["arv_mean"].values)
    n      = len(grp)
    lo, hi = fisher_z_ci(r, n)
    print(f"  {label:25s}: {report_r(r, n)}")
    records_a1.append({"stratum":"full_mv", "level": label, "r_cv_arv": r, "n": n, "ci_lo": lo, "ci_hi": hi, "p": p})

# 1d) Vasopressor exposure
print("\n--- 1d) Vasopressor exposure ---")
for vaso_val, label in [(0, "No vasopressors"), (1, "Any vasopressor")]:
    grp    = w_vitaldb[w_vitaldb["vaso_any"] == vaso_val]
    r, p   = spearman_r(grp["cv_pa_mean"].values, grp["arv_mean"].values)
    n      = len(grp)
    lo, hi = fisher_z_ci(r, n)
    print(f"  {label:25s}: {report_r(r, n)}")
    records_a1.append({"stratum":"vaso_any", "level": label, "r_cv_arv": r, "n": n, "ci_lo": lo, "ci_hi": hi, "p": p})

# 1e) Correlation of per-patient r_cv_arv with rsa_mean (spontaneous breathing proxy)
print("\n--- 1e) Per-patient r_cv_arv ~ rsa_mean (spontaneous breathing proxy) ---")
pp_valid = pp_vitaldb.dropna(subset=["r_cv_arv","rsa_mean_pt"])
r_rsa, p_rsa = spearman_r(pp_valid["rsa_mean_pt"].values, pp_valid["r_cv_arv"].values)
print(f"  Spearman r(rsa_mean_patient, r_cv_arv) = {report_r(r_rsa, len(pp_valid))}, p={p_rsa:.4e}")
print("  Interpretation: positive r → higher spontaneous breathing → higher cv–arv collinearity")

# 1f) Low vs high RSA quartile in VitalDB
print("\n--- 1f) VitalDB windows split by RSA quartile (spontaneous breathing proxy) ---")
w_vitaldb["rsa_q"] = pd.qcut(w_vitaldb["rsa_mean"], q=4, labels=["Q1","Q2","Q3","Q4"])
for q, grp in w_vitaldb.groupby("rsa_q", observed=True):
    r, p   = spearman_r(grp["cv_pa_mean"].values, grp["arv_mean"].values)
    n      = len(grp)
    lo, hi = fisher_z_ci(r, n)
    print(f"  RSA {q}: {report_r(r, n)}")
    records_a1.append({"stratum":"rsa_quartile", "level": str(q), "r_cv_arv": r, "n": n, "ci_lo": lo, "ci_hi": hi, "p": p})

# Cross-cohort RSA comparison (helps interpret the collinearity gap)
print("\n--- 1g) Cross-cohort RSA distribution (spontaneous breathing proxy) ---")
for label, df in [("Clinic", w_clinic), ("MIMIC", w_mimic), ("VitalDB", w_vitaldb)]:
    m  = df["rsa_mean"].median()
    sd = df["rsa_mean"].std()
    print(f"  {label:10s}: rsa_mean median={m:.2f}, IQR "
          f"[{df['rsa_mean'].quantile(0.25):.2f}–{df['rsa_mean'].quantile(0.75):.2f}], "
          f"SD={sd:.2f}")

df_a1 = pd.DataFrame(records_a1)
df_a1.to_csv(OUT_DIR / "a1_cv_arv_ventilatory_strata.csv", index=False)
print(f"\n[saved] results/supplementary/a1_cv_arv_ventilatory_strata.csv")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — BRS-min sign / magnitude vs homeostatic stress markers
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 2: BRS-min sign/magnitude vs homeostatic stress markers")
print("="*70)

# Only VitalDB has structured clinical metadata
df2 = w_vitaldb.copy()
df2["brs_min_sign"] = np.sign(df2["brs_min"])
df2["brs_min_neg"]  = (df2["brs_min"] < 0).astype(int)

# 2a) Proportion of brs_min < 0 by ASA class
print("\n--- 2a) brs_min < 0 by ASA class ---")
asa_records = []
for asa_val in sorted(df2["asa"].dropna().unique()):
    grp = df2[df2["asa"] == asa_val]
    pct_neg = grp["brs_min_neg"].mean() * 100
    med     = grp["brs_min"].median()
    print(f"  ASA {asa_val:.0f}: {pct_neg:.1f}% brs_min<0, median brs_min={med:.2f}, n={len(grp)}")
    asa_records.append({"asa": asa_val, "pct_brs_min_neg": pct_neg, "median_brs_min": med, "n": len(grp)})

# Kruskal-Wallis test across ASA classes (≥2 classes)
asa_groups = [df2[df2["asa"] == a]["brs_min"].dropna().values for a in sorted(df2["asa"].dropna().unique())]
asa_groups = [g for g in asa_groups if len(g) >= 5]
if len(asa_groups) >= 2:
    h, p_kw = stats.kruskal(*asa_groups)
    print(f"  Kruskal-Wallis brs_min across ASA: H={h:.2f}, p={p_kw:.4e}")

# Spearman r(ASA, brs_min)
df2_asa = df2.dropna(subset=["asa","brs_min"])
r_asa, p_asa = spearman_r(df2_asa["asa"].values, df2_asa["brs_min"].values)
print(f"  Spearman r(ASA, brs_min) = {r_asa:.4f}, p={p_asa:.4e}")

# 2b) Emergency vs elective
print("\n--- 2b) brs_min by emergency surgery (emop) ---")
for emop_val, label in [(0,"Elective"),(1,"Emergency")]:
    grp     = df2[df2["emop"] == emop_val]
    pct_neg = grp["brs_min_neg"].mean() * 100
    med     = grp["brs_min"].median()
    print(f"  {label:12s}: {pct_neg:.1f}% brs_min<0, median={med:.2f}, n={len(grp)}")

elec = df2[df2["emop"]==0]["brs_min"].dropna().values
emer = df2[df2["emop"]==1]["brs_min"].dropna().values
u, p_mw = stats.mannwhitneyu(elec, emer, alternative="two-sided")
print(f"  Mann-Whitney U: U={u:.0f}, p={p_mw:.4e}")

# 2c) Vasopressor exposure
print("\n--- 2c) brs_min by vasopressor exposure ---")
for vaso_val, label in [(0,"No vaso"),(1,"Any vaso")]:
    grp     = df2[df2["vaso_any"] == vaso_val]
    pct_neg = grp["brs_min_neg"].mean() * 100
    med     = grp["brs_min"].median()
    print(f"  {label:12s}: {pct_neg:.1f}% brs_min<0, median={med:.2f}, n={len(grp)}")

nv = df2[df2["vaso_any"]==0]["brs_min"].dropna().values
av = df2[df2["vaso_any"]==1]["brs_min"].dropna().values
u2, p_mw2 = stats.mannwhitneyu(nv, av, alternative="two-sided")
print(f"  Mann-Whitney U (no vaso vs any vaso): U={u2:.0f}, p={p_mw2:.4e}")

# Spearman r(vaso_total, brs_min) among users
df2_vaso = df2[df2["vaso_any"]==1].dropna(subset=["brs_min","vaso_total"])
if len(df2_vaso) >= 10:
    r_vt, p_vt = spearman_r(df2_vaso["vaso_total"].values, df2_vaso["brs_min"].values)
    print(f"  Among vasopressor users: r(vaso_total_dose, brs_min) = {r_vt:.4f}, p={p_vt:.4e}")

# 2d) Interaction: patients with high ASA AND vasopressor use
print("\n--- 2d) brs_min by combined stress (ASA≥3 + any vasopressor) ---")
df2["high_stress"] = ((df2["asa"] >= 3) & (df2["vaso_any"] == 1)).astype(int)
for hs_val, label in [(0,"Low stress"),(1,"High stress (ASA≥3+vaso)")]:
    grp     = df2[df2["high_stress"] == hs_val]
    pct_neg = grp["brs_min_neg"].mean() * 100
    med     = grp["brs_min"].median()
    n       = len(grp)
    print(f"  {label:30s}: {pct_neg:.1f}% brs_min<0, median={med:.2f}, n={n}")

lo_s = df2[df2["high_stress"]==0]["brs_min"].dropna().values
hi_s = df2[df2["high_stress"]==1]["brs_min"].dropna().values
u3, p_mw3 = stats.mannwhitneyu(lo_s, hi_s, alternative="two-sided")
print(f"  Mann-Whitney: p={p_mw3:.4e}")

# Cross-cohort brs_min comparison: clinic / MIMIC / VitalDB
print("\n--- 2e) Cross-cohort brs_min summary ---")
for label, df_c in [("Clinic", w_clinic), ("MIMIC", w_mimic), ("VitalDB", w_vitaldb)]:
    pct_neg = (df_c["brs_min"] < 0).mean() * 100
    med     = df_c["brs_min"].median()
    print(f"  {label:10s}: {pct_neg:.1f}% brs_min<0, median={med:.2f}, n={len(df_c)}")

# Save summary table
df_a2_asa = pd.DataFrame(asa_records)
df_a2_asa.to_csv(OUT_DIR / "a2_brs_min_stress_by_asa.csv", index=False)
print(f"\n[saved] results/supplementary/a2_brs_min_stress_by_asa.csv")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Restrict to single ventilatory regime; does cv flip collapse?
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 3: CV flip restricted to single ventilatory regime")
print("="*70)

# ── Full MV subset of VitalDB ──────────────────────────────────────────────────
w_vdb_mv = w_vitaldb[w_vitaldb["full_mv"] == 1].copy()
print(f"\nFull-MV VitalDB subset: {len(w_vdb_mv)} windows, "
      f"{w_vdb_mv['patient_id'].nunique()} patients")

# All three cohorts: clinic and MIMIC presumed fully ventilated
# (clinic = intraop OR, largely GA; MIMIC = ICU with MV)
print("\n--- 3a) cv_pa_mean × arv_mean correlation in Full-MV strata ---")
strat_records = []
for label, df_c in [
    ("Clinic (all, presume MV)",        w_clinic),
    ("MIMIC (all, presume ICU-MV)",     w_mimic),
    ("VitalDB full (all ane_type)",      w_vitaldb),
    ("VitalDB General+IPPV only",        w_vdb_mv),
    ("VitalDB General ane_type only",    w_vitaldb[w_vitaldb["ane_type"]=="General"]),
]:
    r, p   = spearman_r(df_c["cv_pa_mean"].values, df_c["arv_mean"].values)
    n      = len(df_c)
    np_   = df_c["patient_id"].nunique() if "patient_id" in df_c.columns else None
    lo, hi = fisher_z_ci(r, n)
    print(f"\n  {label}")
    print(f"    r={r:.4f} (95%CI {lo:.3f}–{hi:.3f}), n_windows={n}, n_patients={np_}")
    strat_records.append({
        "stratum": label, "r_cv_arv": r, "ci_lo": lo, "ci_hi": hi,
        "n_windows": n, "n_patients": np_, "p": p
    })

# ── Pairwise comparison: Full-MV VitalDB vs Clinic, vs MIMIC ─────────────────
print("\n--- 3b) Fisher Z: Full-MV VitalDB vs Clinic, MIMIC ---")
r_vdb_mv = strat_records[3]["r_cv_arv"]  # VitalDB General+IPPV
n_vdb_mv = strat_records[3]["n_windows"]
r_clinic = strat_records[0]["r_cv_arv"]
n_clinic = strat_records[0]["n_windows"]
r_mimic  = strat_records[1]["r_cv_arv"]
n_mimic  = strat_records[1]["n_windows"]

p_fz_vm_cl = compare_rs(r_vdb_mv, n_vdb_mv, r_clinic, n_clinic)
p_fz_vm_mi = compare_rs(r_vdb_mv, n_vdb_mv, r_mimic, n_mimic)
print(f"  Full-MV VitalDB vs Clinic: p={p_fz_vm_cl:.4e}")
print(f"  Full-MV VitalDB vs MIMIC:  p={p_fz_vm_mi:.4e}")
if p_fz_vm_cl > 0.05 and p_fz_vm_mi > 0.05:
    print("  → Difference collapses under single-regime restriction  ✓")
else:
    print("  → Residual difference persists beyond ventilatory regime")

# ── cv_pa_mean × arv_mean per RSA quartile within Full-MV VitalDB ─────────────
print("\n--- 3c) Within Full-MV VitalDB: cv–arv r by RSA quartile ---")
w_vdb_mv["rsa_q"] = pd.qcut(w_vdb_mv["rsa_mean"], q=4, labels=["Q1","Q2","Q3","Q4"])
for q, grp in w_vdb_mv.groupby("rsa_q", observed=True):
    r, p = spearman_r(grp["cv_pa_mean"].values, grp["arv_mean"].values)
    n    = len(grp)
    lo, hi = fisher_z_ci(r, n)
    print(f"  RSA {q}: r={r:.4f} (95%CI {lo:.3f}–{hi:.3f}), n={n}")

df_a3 = pd.DataFrame(strat_records)
df_a3.to_csv(OUT_DIR / "a3_cv_arv_regimestratified.csv", index=False)
print(f"\n[saved] results/supplementary/a3_cv_arv_regimestratified.csv")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED NARRATIVE REPORT
# ══════════════════════════════════════════════════════════════════════════════
report_lines = [
    "VENTILATORY CONTEXT ANALYSIS — NARRATIVE SUMMARY",
    "=" * 70,
    "",
    "Question 1: Does the cv–arv colinearity gap (r≈0.71–0.93 clinic/MIMIC",
    "vs ≈0.71 VitalDB) reflect the ventilatory context?",
    "",
]
r_vdb_gen = df_a1[df_a1["level"]=="General"]["r_cv_arv"].values
if len(r_vdb_gen):
    report_lines.append(
        f"  • VitalDB General anesthesia alone: r={r_vdb_gen[0]:.4f} "
        f"(n={df_a1[df_a1['level']=='General']['n'].values[0]})"
    )
# Full MV strat
row_mv = df_a1[df_a1["level"]=="Full MV (General+IPPV)"]
if len(row_mv):
    report_lines.append(
        f"  • VitalDB Full-MV only: r={row_mv['r_cv_arv'].values[0]:.4f} "
        f"(n={row_mv['n'].values[0]})"
    )
report_lines += [
    f"  • Per-patient r(rsa_mean, r_cv_arv) = {r_rsa:.4f} (p={p_rsa:.4e})",
    "    Positive value → spontaneous breathing inflation of cv–arv coupling",
    "",
    "Question 2: Is brs_min sign/magnitude a marker of homeostatic stress?",
    f"  • Spearman r(ASA, brs_min) = {r_asa:.4f} (p={p_asa:.4e})",
    f"  • Mann-Whitney emergency vs elective brs_min: p={p_mw:.4e}",
    f"  • Mann-Whitney vasopressor vs none brs_min: p={p_mw2:.4e}",
    "",
    "Question 3: cv flip under single-regime restriction",
    f"  • Full-MV VitalDB vs Clinic: p_Fisher={p_fz_vm_cl:.4e}",
    f"  • Full-MV VitalDB vs MIMIC:  p_Fisher={p_fz_vm_mi:.4e}",
    "",
    "See CSV files for full tables:",
    "  results/supplementary/a1_cv_arv_ventilatory_strata.csv",
    "  results/supplementary/a2_brs_min_stress_by_asa.csv",
    "  results/supplementary/a3_cv_arv_regimestratified.csv",
]

report_text = "\n".join(report_lines)
(OUT_DIR / "ventilatory_context_analysis.txt").write_text(report_text)
print("\n" + "="*70)
print(report_text)
print(f"\n[saved] results/supplementary/ventilatory_context_analysis.txt")
