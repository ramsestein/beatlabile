"""
Q2 Main Orchestrator
─────────────────────
Runs the full Q2 pipeline:
  PASO 1 – Vasopressor event identification + exclusion
  PASO 2 – Quiescent control window matching
  PASO 3 – Q2a confirmatory tests (pre vs control)
  PASO 4 – Q2b descriptive tests (pre vs post)
  PASO 5 – Gradient + sensitivity analyses
  PASO 6 – Dissociation figure (Q1 vs Q2)
  PASO 7 – Q2_report.md

CHECKPOINT: if clean events < MIN_EVENTS (10) → STOP after PASO 1.
"""
import sys
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_config import (
    Q2_RES, FIG,
    VASOPRESSOR_EVENTS, PAIRED_PRE_CONTROL, PAIRED_PRE_POST,
    TEST_RESULTS_Q2A, TEST_RESULTS_Q2B, REPORT,
    MIN_EVENTS, ALPHA_BONF, PRIMARY, PARTIAL_PATIENT,
)
from q2_events import load_all, build_vasopressor_events, build_control_windows
from q2_stats  import (
    build_paired_Q2a, run_Q2a_tests,
    build_paired_Q2b, run_Q2b_tests,
    run_gradient_Q2, run_sensitivity_Q2,
)
from q2_figures import (
    fig_violins_Q2a, fig_trajectories_Q2b,
    fig_forest_Q2a, fig_dissociation,
)


def main():
    ts_start = datetime.datetime.now()
    print("=" * 65)
    print(f"Q2 Analysis  |  {ts_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Create output dirs ───────────────────────────────────────────────────
    Q2_RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[LOAD] Loading data …")
    ann, feat, drug, ev_win = load_all()

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 1 – Vasopressor events
    # ══════════════════════════════════════════════════════════════════════════
    events = build_vasopressor_events(ann, feat, drug, ev_win)
    events.to_csv(VASOPRESSOR_EVENTS, index=False)
    print(f"[SAVE] {VASOPRESSOR_EVENTS.name}")

    n_clean = (events["status"] == "clean").sum()

    # CHECKPOINT
    if n_clean < MIN_EVENTS:
        msg = (f"\n⚠  CHECKPOINT FAILED: only {n_clean} clean events "
               f"(required ≥ {MIN_EVENTS}).\n"
               f"   Writing partial report and stopping.\n")
        print(msg)
        _write_checkpoint_report(events, n_clean, ts_start)
        return

    print(f"\n✓ CHECKPOINT passed: {n_clean} clean events (≥ {MIN_EVENTS})")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2 – Control windows
    # ══════════════════════════════════════════════════════════════════════════
    controls = build_control_windows(events, ann, feat, drug)

    n_with_ctrl = controls["ctrl_excluded"].isna().sum()
    print(f"[PASO 2] Events with valid control: {n_with_ctrl} / {n_clean}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 3 – Q2a confirmatory
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[PASO 3] Q2a – pre-bolus vs quiescent control:")
    paired_ctrl = build_paired_Q2a(events, controls, feat)
    paired_ctrl.to_csv(PAIRED_PRE_CONTROL, index=False)
    print(f"[SAVE] {PAIRED_PRE_CONTROL.name}")

    print("\n[PASO 3] Running Q2a tests …")
    results_q2a = run_Q2a_tests(paired_ctrl)
    results_q2a.to_csv(TEST_RESULTS_Q2A, index=False)
    print(f"[SAVE] {TEST_RESULTS_Q2A.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 4 – Q2b descriptive
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[PASO 4] Q2b – pre vs post trajectories:")
    paired_post = build_paired_Q2b(events, feat)
    paired_post.to_csv(PAIRED_PRE_POST, index=False)
    print(f"[SAVE] {PAIRED_PRE_POST.name}")

    print("\n[PASO 4] Running Q2b tests …")
    results_q2b = run_Q2b_tests(paired_post)
    results_q2b.to_csv(TEST_RESULTS_Q2B, index=False)
    print(f"[SAVE] {TEST_RESULTS_Q2B.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 5 – Gradient + sensitivity
    # ══════════════════════════════════════════════════════════════════════════
    gradient_df   = run_gradient_Q2(paired_ctrl)
    sensitivity   = run_sensitivity_Q2(events, controls, feat)

    if not gradient_df.empty:
        gradient_df.to_csv(Q2_RES / "gradient_Q2.csv", index=False)
    for sa_label, df_sa in sensitivity.items():
        df_sa.to_csv(Q2_RES / f"{sa_label}.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 6 – Figures
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[PASO 6] Generating figures …")

    prim_q2a = results_q2a[results_q2a["analysis"] == "primary"].copy()

    fig_violins_Q2a(paired_ctrl, prim_q2a, FIG)
    fig_trajectories_Q2b(events, feat, FIG)
    fig_forest_Q2a(results_q2a, paired_ctrl, FIG)
    fig_dissociation(prim_q2a, paired_ctrl, FIG)

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 7 – Report
    # ══════════════════════════════════════════════════════════════════════════
    ts_end = datetime.datetime.now()
    write_Q2_report(
        events, controls, paired_ctrl, paired_post,
        results_q2a, results_q2b, gradient_df, sensitivity,
        ts_start, ts_end,
    )
    print(f"\n{'='*65}")
    print(f"Q2 pipeline complete  |  {ts_end.strftime('%H:%M:%S')}")
    print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# Report writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_Q2_report(
    events, controls, paired_ctrl, paired_post,
    results_q2a, results_q2b, gradient_df, sensitivity,
    ts_start, ts_end,
):
    """Write Q2_report.md with all results."""
    n_raw   = len(events)
    n_clean = (events["status"] == "clean").sum()
    n_ctrl  = controls["ctrl_excluded"].isna().sum() if controls is not None else 0
    n_pairs_q2a = paired_ctrl["delta"].notna().sum() if paired_ctrl is not None else 0
    n_pairs_q2b = paired_post["delta_post"].notna().sum() if paired_post is not None else 0

    lines = [
        f"# Q2 Analysis Report",
        f"",
        f"Generated: {ts_start.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Elapsed: {(ts_end - ts_start).seconds}s",
        f"",
        f"## 1. Hypothesis",
        f"",
        f"**Q2**: Validate the autonomic pre-hypotension signature of BeatLabile",
        f"in the minutes preceding a vasopressor bolus.",
        f"",
        f"- **Pre-specified direction**: ALL features expected to DECREASE (β < 0)",
        f"  relative to a quiescent control window.",
        f"- **Bonferroni α** = {ALPHA_BONF:.3f} (5 primary features)",
        f"",
        f"## 2. Events (PASO 1)",
        f"",
        f"| Metric | N |",
        f"|---|---|",
        f"| Raw vasopressor boluses | {n_raw} |",
        f"| Clean (all criteria) | {n_clean} |",
        f"| With valid control window | {n_ctrl} |",
        f"",
        f"### 2.1 Exclusion log",
        f"",
    ]

    excl = events[events["status"] == "excluded"][
        ["patient_id", "drug", "t_bolus", "exclusion_reason"]
    ].copy()
    if len(excl) > 0:
        lines.append("| patient_id | drug | t_bolus (s) | reason |")
        lines.append("|---|---|---|---|")
        for _, r in excl.iterrows():
            lines.append(f"| {r['patient_id']} | {r['drug']} | {r['t_bolus']:.0f} | {r['exclusion_reason']} |")
    else:
        lines.append("No exclusions.")

    lines += [
        f"",
        f"### 2.2 Clean events",
        f"",
    ]
    clean_ev = events[events["status"] == "clean"][
        ["patient_id", "drug", "group", "t_bolus", "t_from_ag_start_s",
         "cum_efedrina_mg", "cum_fenilefrina_mcg"]
    ]
    if len(clean_ev) > 0:
        lines.append("| pid | drug | group | t_bolus (s) | t_from_AG (s) | cum_eph (mg) | cum_phen (µg) |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in clean_ev.iterrows():
            lines.append(
                f"| {r['patient_id']} | {r['drug']} | {r['group']} |"
                f" {r['t_bolus']:.0f} | {r['t_from_ag_start_s']:.0f} |"
                f" {r['cum_efedrina_mg']:.1f} | {r['cum_fenilefrina_mcg']:.1f} |"
            )

    lines += [
        f"",
        f"## 3. Q2a – Confirmatory Tests (PASO 3)",
        f"",
        f"**Design**: GLMM (random intercept per patient), one-sided H1: β < 0",
        f"**delta** = feature_pre − feature_control",
        f"",
        f"### 3.1 Primary features",
        f"",
        f"| Feature | n_pairs | β | IC 95% | p_1sided | p_Bonf | d_z | Verdict |",
        f"|---|---|---|---|---|---|---|---|",
    ]

    prim = results_q2a[results_q2a["analysis"] == "primary"]
    for _, r in prim.iterrows():
        lines.append(
            f"| {r['feature']} | {r['n_pairs']} | {r.get('beta', np.nan):+.4f} |"
            f" [{r.get('ic_lo', np.nan):+.4f}, {r.get('ic_hi', np.nan):+.4f}] |"
            f" {r.get('p_1sided', np.nan):.4f} | {r.get('p_bonferroni', np.nan):.4f} |"
            f" {r.get('cohen_dz', np.nan):+.3f} | **{r.get('verdict', 'NS')}** |"
        )

    n_validated = (prim["verdict"] == "VALIDADO").sum()
    overall = (
        "VALIDACIÓN COMPLETA (≥3/5)" if n_validated >= 3 else
        "VALIDACIÓN PARCIAL (1-2/5)" if n_validated >= 1 else
        "NO VALIDADO (0/5)"
    )

    lines += [
        f"",
        f"### 3.2 Summary",
        f"",
        f"**{n_validated} / {len(prim)} primary features validated** (Bonferroni α={ALPHA_BONF:.3f})",
        f"",
        f"**Overall Q2a verdict: {overall}**",
        f"",
        f"### 3.3 Exploratory features",
        f"",
        f"| Feature | n_pairs | β | p_2sided | d_z |",
        f"|---|---|---|---|---|",
    ]

    expl = results_q2a[results_q2a["analysis"] == "exploratory"]
    for _, r in expl.iterrows():
        lines.append(
            f"| {r['feature']} | {r['n_pairs']} | {r.get('beta', np.nan):+.4f} |"
            f" {r.get('p_2sided', np.nan):.4f} | {r.get('cohen_dz', np.nan):+.3f} |"
        )

    lines += [
        f"",
        f"## 4. Q2b – Descriptive Trajectories (PASO 4)",
        f"",
        f"**delta_post** = feature_post − feature_pre (vasopresor response)",
        f"",
        f"| Feature | n_pairs | β_post | IC 95% | p_2sided | d_z |",
        f"|---|---|---|---|---|---|",
    ]
    for _, r in results_q2b.iterrows():
        lines.append(
            f"| {r['feature']} | {r['n_pairs']} | {r.get('beta_post', np.nan):+.4f} |"
            f" [{r.get('ic_lo', np.nan):+.4f}, {r.get('ic_hi', np.nan):+.4f}] |"
            f" {r.get('p_2sided', np.nan):.4f} | {r.get('cohen_dz', np.nan):+.3f} |"
        )

    lines += [
        f"",
        f"## 5. Gradient Analysis (PASO 5a)",
        f"",
        f"Group modifier: interescalénico vs supra-axilar.",
        f"Expected: interescalénico **amplifies** pre-hypotensive signature (β_interesk < 0).",
        f"",
    ]
    if gradient_df is not None and not gradient_df.empty:
        lines += [
            f"| Feature | β_interesk | IC 95% | p_interaction | Interpretation |",
            f"|---|---|---|---|---|",
        ]
        for _, r in gradient_df.iterrows():
            lines.append(
                f"| {r['feature']} | {r.get('beta_interesk', np.nan):+.4f} |"
                f" [{r.get('ic_lo', np.nan):+.4f}, {r.get('ic_hi', np.nan):+.4f}] |"
                f" {r.get('p_interaction', np.nan):.4f} | {r.get('interpretation', '')} |"
            )
    else:
        lines.append("Gradient analysis not run (insufficient data).")

    lines += [
        f"",
        f"## 6. Sensitivity Analyses (PASO 5b)",
        f"",
    ]
    for sa_label, df_sa in sensitivity.items():
        lines.append(f"### {sa_label}")
        if df_sa.empty:
            lines.append("Insufficient data.")
        else:
            lines += [
                f"| Feature | n_pairs | β | p_1sided | d_z |",
                f"|---|---|---|---|---|",
            ]
            for _, r in df_sa.iterrows():
                lines.append(
                    f"| {r.get('feature', '')} | {r.get('n_pairs', 0)} |"
                    f" {r.get('beta', np.nan):+.4f} | {r.get('p_1sided', np.nan):.4f} |"
                    f" {r.get('cohen_dz', np.nan):+.3f} |"
                )
        lines.append("")

    lines += [
        f"## 7. Dissociation Q1 vs Q2 (PASO 6)",
        f"",
        f"See figure `figures/q1_vs_q2_dissociation.png`.",
        f"",
        f"| Feature | Q1 β (pain) | Q2 β (pre-HTN) | Dissociation |",
        f"|---|---|---|---|",
    ]
    # Load Q1 for comparison
    try:
        q1 = pd.read_csv(str(REPORT).replace("Q2_report.md", "../Q1/test_results_v2.csv"))
        q1_prim = q1[q1["analysis"] == "primary"] if "analysis" in q1.columns else q1
        for fname, agg, _ in PRIMARY:
            key = f"{fname}__{agg}"
            # Q1 CSV: feature=name only + agg separate
            if "agg" in q1_prim.columns:
                r1 = q1_prim[(q1_prim["feature"] == fname) & (q1_prim["agg"] == agg)]
            else:
                r1 = q1_prim[q1_prim["feature"] == key]
            r2  = results_q2a[results_q2a["feature"] == key]
            b1  = r1.iloc[0]["beta"] if not r1.empty else np.nan
            b2  = r2.iloc[0].get("beta", np.nan) if not r2.empty else np.nan
            diss = ("OPPOSITE" if not np.isnan(b1) and not np.isnan(b2) and
                    np.sign(b1) != np.sign(b2) else
                    "SAME_SIGN" if not np.isnan(b1) and not np.isnan(b2) else "NA")
            lines.append(f"| {key} | {b1:+.4f} | {b2:+.4f} | {diss} |"
                         if not np.isnan(b1) else f"| {key} | NA | {b2:+.4f} | NA |")
    except Exception as e:
        lines.append(f"Q1 comparison unavailable: {e}")

    lines += [
        f"",
        f"## 8. Figures",
        f"",
        f"| File | Description |",
        f"|---|---|",
        f"| figures/q2_pre_vs_control_violins.png | Q2a pre vs control violins |",
        f"| figures/q2_pre_vs_post_trajectories.png | Q2b ±5 min trajectories |",
        f"| figures/q2_forest_plot.png | Q2a forest plot |",
        f"| figures/q1_vs_q2_dissociation.png | Q1 vs Q2 dissociation (KEY) |",
        f"",
        f"---",
        f"*Report generated by `validation/q2_main.py` — pre-specified analysis.*",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {REPORT.name}")


def _write_checkpoint_report(events, n_clean, ts_start):
    Q2_RES.mkdir(parents=True, exist_ok=True)
    excl_summary = (
        events[events["status"] == "excluded"]["exclusion_reason"]
        .value_counts()
        .to_string()
    )
    content = (
        f"# Q2 Analysis – CHECKPOINT FAILED\n\n"
        f"Generated: {ts_start.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"Clean events: **{n_clean}** (required ≥ {MIN_EVENTS})\n\n"
        f"## Exclusion reason summary\n\n```\n{excl_summary}\n```\n"
    )
    REPORT.write_text(content, encoding="utf-8")
    print(f"[SAVE] Checkpoint report → {REPORT.name}")


if __name__ == "__main__":
    main()
