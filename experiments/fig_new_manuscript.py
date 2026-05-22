#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_new_manuscript.py — Figuras nuevas para la revisión del paper
==================================================================
Genera cuatro figuras para el manuscrito revisado:

  Fig 1 — STROBE flowchart (Clínic + VitalDB, con conteos actualizados)
  Fig 2 — Incremento sobre MAP (AUC 5-fold CV + DeLong paired, 2 cohortes × 2 desenlaces)
  Fig 3 — Δ evento vs control + gradiente inter-cohorte (baroreflex vs variabilidad)
  Fig 4 — Persistencia temporal del AUC (horizonte 0–30 min, Clínic OOS-CV + VitalDB OOS)

Outputs:
  results/figures/fig1_strobe_v2.{png,pdf}
  results/figures/fig2_map_increment.{png,pdf}
  results/figures/fig3_delta_gradient.{png,pdf}
  results/figures/fig4_temporal_persistence.{png,pdf}

Usage (from project root, venv activo):
  python experiments/fig_new_manuscript.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "map":      "#6baed6",   # blue
    "autonomic":"#fd8d3c",   # orange
    "combined": "#31a354",   # green
    "clinic":   "#756bb1",   # purple
    "vitaldb":  "#2c7fb8",   # teal-blue
    "baro":     "#d73027",   # red
    "var":      "#4575b4",   # blue
}

EVENT_LABEL = {"hypotension": "Hypotension", "hypertension": "Hypertension"}
COHORT_LABEL = {"clinic": "Clínic", "vitaldb": "VitalDB"}


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — STROBE flowchart (updated counts)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_strobe():
    """STROBE participant flow for Clínic (development) and VitalDB (validation)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 9))

    def box(ax, x, y, text, w=0.38, h=0.07, fc="white", ec="steelblue", fs=8.5):
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.015", fc=fc, ec=ec, lw=1.3, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                wrap=True, zorder=4, multialignment="center",
                linespacing=1.4)

    def arrow(ax, x, y0, y1):
        ax.annotate("", xy=(x, y1 + 0.035), xytext=(x, y0 - 0.035),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.2), zorder=2)

    def exclusion(ax, x_main, y, text):
        ax.annotate("", xy=(x_main + 0.21, y), xytext=(x_main, y),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.0), zorder=2)
        rect = mpatches.FancyBboxPatch(
            (x_main + 0.21, y - 0.03), 0.30, 0.06,
            boxstyle="round,pad=0.01", fc="#fff7f0", ec="#888", lw=0.9, zorder=3)
        ax.add_patch(rect)
        ax.text(x_main + 0.215 + 0.15, y, text, ha="center", va="center",
                fontsize=7.5, color="#555", zorder=4, multialignment="center",
                linespacing=1.3)

    # ── Clínic panel ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Clínic UCIQ  (Development cohort)", fontsize=10,
                 fontweight="bold", pad=6)

    ys = [0.92, 0.80, 0.68, 0.56, 0.40, 0.26, 0.12]
    x = 0.5

    box(ax, x, ys[0], "Surgical ICU admissions screened\nn = 492", fc="#eaf3fb")
    arrow(ax, x, ys[0], ys[1])
    exclusion(ax, x, (ys[0] + ys[1]) / 2,
              "Excluded (n = 222):\n• No invasive arterial line\n• Duration < 30 min\n• Signal QC fail")
    box(ax, x, ys[1], "Recordings with valid signal\nn = 270", fc="#eaf3fb")
    arrow(ax, x, ys[1], ys[2])
    box(ax, x, ys[2], "Patients with ≥ 1 prediction window\nn = 270  |  4 355 windows", fc="#ddeedd")
    arrow(ax, x, ys[2], ys[3])
    exclusion(ax, x, (ys[2] + ys[3]) / 2 - 0.01,
              "Excluded (event analysis):\n• No outcome event\n  (variability excl.)")
    box(ax, x, ys[3], "Included in primary analysis\nn = 204 patients", fc="#ddeedd")
    arrow(ax, x, ys[3], ys[4])

    # Event boxes
    box(ax, 0.28, ys[4], "Hypotension\n66 events\n885 controls", w=0.26, fc="#fef0d9")
    box(ax, 0.72, ys[4], "Hypertension\n38 events\n885 controls", w=0.26, fc="#fef0d9")

    arrow(ax, 0.28, ys[4], ys[5])
    arrow(ax, 0.72, ys[4], ys[5])

    box(ax, x, ys[5],
        "Model development\n10 × 50 patient-level CV\nGLMM parsimonious (8 features)",
        fc="#f5f0ff")
    arrow(ax, x, ys[5], ys[6])
    box(ax, x, ys[6],
        "AUC hypotension 0.75 [0.26–0.99]\nAUC hypertension 0.83 [0.41–1.00]",
        fc="#fff0f5", ec="#c994c7")

    # ── VitalDB panel ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("VitalDB  (External validation cohort)", fontsize=10,
                 fontweight="bold", pad=6)

    box(ax, x, ys[0], "VitalDB cases available\nn = 6 388", fc="#eaf3fb")
    arrow(ax, x, ys[0], ys[1])
    exclusion(ax, x, (ys[0] + ys[1]) / 2,
              "Excluded (n = 5 308):\n• No invasive ART line\n• Duration < 30 min\n• QC fail / no windows")
    box(ax, x, ys[1], "Cases with valid arterial signal\nn = 1 080", fc="#eaf3fb")
    arrow(ax, x, ys[1], ys[2])
    box(ax, x, ys[2], "Patients with ≥ 1 prediction window\nn = 1 080  |  4 471 windows", fc="#ddeedd")
    arrow(ax, x, ys[2], ys[3])
    box(ax, x, ys[3], "Included in validation analysis\nn = 1 080 patients", fc="#ddeedd")
    arrow(ax, x, ys[3], ys[4])

    box(ax, 0.28, ys[4], "Hypotension\n505 events\n483 controls", w=0.26, fc="#fef0d9")
    box(ax, 0.72, ys[4], "Hypertension\n75 events\n483 controls", w=0.26, fc="#fef0d9")

    arrow(ax, 0.28, ys[4], ys[5])
    arrow(ax, 0.72, ys[4], ys[5])

    box(ax, x, ys[5],
        "External validation (M1)\n+ Refit analysis M2–M6 (70/30 hold-out)\n+ MAP slope control (S5)",
        fc="#f5f0ff")
    arrow(ax, x, ys[5], ys[6])
    box(ax, x, ys[6],
        "AUC hypo (M2 refit) 0.844 [0.806–0.879]\nAUC hyper (M2 refit) 0.875 [0.813–0.930]",
        fc="#fff0f5", ec="#c994c7")

    fig.suptitle("Fig 1 — STROBE participant flow", fontsize=12,
                 fontweight="bold", y=1.0)
    _save(fig, "fig1_strobe_v2")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — MAP increment (headline result)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_map_increment():
    """2×2 panel: AUC bars (MAP-solo / autonomic / MAP+var) + DeLong annotation."""
    auc_df   = pd.read_csv("results/supplementary/map_slope_auc_comparison_v2.csv")
    delong   = pd.read_csv("results/supplementary/map_slope_delong_v2.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=False)
    fig.suptitle(
        "Fig 2 — Autonomic variability adds predictive value above raw MAP\n"
        "(5-fold CV composites, paired DeLong test)",
        fontsize=11, fontweight="bold", y=1.01)

    cohorts  = ["clinic", "vitaldb"]
    outcomes = ["hypotension", "hypertension"]

    for row, cohort in enumerate(cohorts):
        for col, outcome in enumerate(outcomes):
            ax = axes[row][col]

            sub = auc_df[
                (auc_df.cohort == cohort) &
                (auc_df.outcome == outcome) &
                (auc_df.feature.isin(
                    ["MAP_composite", "autonomic_composite", "MAP+autonomic_composite"]))
            ].copy()

            feature_order = ["MAP_composite", "autonomic_composite",
                             "MAP+autonomic_composite"]
            labels_short  = ["MAP\n(5-fold CV)", "Autonomic\n(5-fold CV)",
                              "MAP +\nVariabilidad\n(5-fold CV)"]
            bar_colors    = [COLORS["map"], COLORS["autonomic"], COLORS["combined"]]

            sub = sub.set_index("feature").loc[feature_order]
            aucs = sub["auc"].values
            los  = sub["ci_lo"].values
            his  = sub["ci_hi"].values
            errs = np.array([aucs - los, his - aucs])

            xs = np.arange(len(feature_order))
            bars = ax.bar(xs, aucs, color=bar_colors, alpha=0.85, width=0.55,
                          zorder=3)
            ax.errorbar(xs, aucs, yerr=errs, fmt="none",
                        color="k", capsize=4, capthick=1.2, lw=1.2, zorder=4)

            # AUC labels on bars
            for i, (a, lo, hi) in enumerate(zip(aucs, los, his)):
                ax.text(i, hi + 0.015, f"{a:.3f}", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

            # DeLong annotation (bracket between bar 0 and bar 2)
            dl = delong[(delong.cohort == cohort) & (delong.outcome == outcome)]
            if not dl.empty:
                row_d = dl.iloc[0]
                y_top = max(his) + 0.055
                # horizontal line
                ax.plot([0, 2], [y_top, y_top], color="#333", lw=1.2)
                ax.plot([0, 0], [y_top - 0.01, y_top], color="#333", lw=1.2)
                ax.plot([2, 2], [y_top - 0.01, y_top], color="#333", lw=1.2)
                p_str = "p < 0.0001" if row_d.p_delong < 0.0001 \
                    else f"p = {row_d.p_delong:.4f}"
                ax.text(1, y_top + 0.01,
                        f"ΔAUC = +{row_d.delta_AUC:.3f}   {p_str}",
                        ha="center", va="bottom", fontsize=8,
                        color="#333", fontweight="bold")

            ax.axhline(0.5, color="gray", lw=0.8, ls="--", zorder=1)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels_short, fontsize=8)
            ax.set_ylim(0.35, min(1.05, max(his) + 0.18))
            ax.set_ylabel("AUC (95% CI)" if col == 0 else "", fontsize=9)
            ax.set_title(
                f"{COHORT_LABEL[cohort]}  —  {EVENT_LABEL[outcome]}\n"
                f"(n events = {int(sub.iloc[0].n_event)})",
                fontsize=9, pad=4)
            ax.yaxis.grid(True, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS["map"],      label="MAP composite"),
        mpatches.Patch(color=COLORS["autonomic"], label="Autonomic composite"),
        mpatches.Patch(color=COLORS["combined"],  label="MAP + Variabilidad"),
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=9,
               frameon=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "fig2_map_increment")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Δ event-vs-control + inter-cohort gradient
# ══════════════════════════════════════════════════════════════════════════════
def fig3_delta_gradient():
    """Left: Cohen's d pre-event deterioration per cohort/outcome/axis.
    Right: inter-cohort gradient (Clínic – VitalDB) with bootstrap CI."""
    delta = pd.read_csv("results/supplementary/delta_event_control.csv")
    grad  = pd.read_csv("results/supplementary/delta_event_control_gradient.csv")

    fig = plt.figure(figsize=(13, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38,
                            width_ratios=[1.8, 1])

    # ── Left: grouped bar (d) ─────────────────────────────────────────────────
    ax_left = fig.add_subplot(gs[0])

    outcomes = ["hypotension", "hypertension"]
    axes_types = ["baroreflex", "variability"]
    cohorts = ["Clinic", "VitalDB"]
    n_groups = len(outcomes) * len(axes_types)  # 4 groups
    group_labels = [f"{EVENT_LABEL[o].lower()}\n({a[:4].title()})"
                    for o in outcomes for a in axes_types]

    x = np.arange(n_groups)
    width = 0.32
    offsets = [-width / 2, width / 2]
    cohort_colors = [COLORS["clinic"], COLORS["vitaldb"]]

    for ci, (cohort, color) in enumerate(zip(cohorts, cohort_colors)):
        ds = []
        lo_errs, hi_errs = [], []
        for outcome in outcomes:
            for axis in axes_types:
                row = delta[(delta.cohort == cohort) &
                            (delta.outcome == outcome) &
                            (delta.axis == axis)]
                if row.empty:
                    ds.append(0); lo_errs.append(0); hi_errs.append(0)
                    continue
                d_val = float(row["d_deterioration"].iloc[0])
                ci95  = str(row["ci95"].iloc[0]).strip("[]").split(",")
                lo_v  = float(ci95[0])
                hi_v  = float(ci95[1])
                ds.append(d_val)
                lo_errs.append(d_val - lo_v)
                hi_errs.append(hi_v - d_val)

        bars = ax_left.bar(x + offsets[ci], ds, width, color=color,
                           alpha=0.82, label=cohort, zorder=3)
        ax_left.errorbar(x + offsets[ci], ds,
                         yerr=[lo_errs, hi_errs],
                         fmt="none", color="k", capsize=3, lw=1.0, zorder=4)
        for xi, d_v in zip(x, ds):
            ax_left.text(xi + offsets[ci], max(0, d_v) + 0.02,
                         f"{d_v:.2f}", ha="center", va="bottom",
                         fontsize=7, color="#333")

    ax_left.axhline(0, color="k", lw=0.8)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(group_labels, fontsize=8)
    ax_left.set_ylabel("Cohen's d  (event vs matched control)", fontsize=9)
    ax_left.set_title("Pre-event deterioration\n(event windows vs matched controls)",
                      fontsize=10, pad=5)
    ax_left.legend(handles=[
        mpatches.Patch(color=COLORS["clinic"],  label="Clínic"),
        mpatches.Patch(color=COLORS["vitaldb"], label="VitalDB"),
    ], fontsize=8, loc="upper left")
    ax_left.yaxis.grid(True, alpha=0.4, zorder=0)
    ax_left.set_axisbelow(True)

    # shade axis-type bands
    for xi in [1, 3]:
        ax_left.axvspan(xi - 0.6, xi + 0.6, alpha=0.05, color="gray", zorder=0)

    # ── Right: gradient (Clínic – VitalDB) ───────────────────────────────────
    ax_right = fig.add_subplot(gs[1])

    grad["label"] = grad.apply(
        lambda r: f"{EVENT_LABEL[r.outcome]}\n({r.axis[:3].title()})",
        axis=1)
    n = len(grad)
    y_pos = np.arange(n)

    deltas   = grad["clinic_minus_vitaldb"].values
    ci_strs  = grad["ci95"].values
    p_boots  = grad["p_boot"].values
    labels   = grad["label"].values

    lo_errs = []
    hi_errs = []
    for d_v, ci95 in zip(deltas, ci_strs):
        parts = str(ci95).strip("[]").split(",")
        lo_errs.append(d_v - float(parts[0]))
        hi_errs.append(float(parts[1]) - d_v)

    sig_colors = [COLORS["baro"] if p < 0.05 else "#aaa" for p in p_boots]

    ax_right.barh(y_pos, deltas, xerr=[lo_errs, hi_errs],
                  color=sig_colors, alpha=0.85, height=0.55, capsize=4,
                  error_kw={"lw": 1.2}, zorder=3)
    ax_right.axvline(0, color="k", lw=0.9)

    for yi, (d_v, p) in enumerate(zip(deltas, p_boots)):
        pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        star = "★" if p < 0.05 else ""
        ax_right.text(d_v + 0.01, yi, f"{star} {pstr}",
                      va="center", fontsize=7.5,
                      color=COLORS["baro"] if p < 0.05 else "#666")

    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(labels, fontsize=8)
    ax_right.set_xlabel("Δd  (Clínic − VitalDB)", fontsize=9)
    ax_right.set_title("Inter-cohort gradient\n(bootstrap p, n=1000)",
                       fontsize=10, pad=5)
    ax_right.xaxis.grid(True, alpha=0.4, zorder=0)
    ax_right.set_axisbelow(True)

    # Legend for significance
    ax_right.legend(handles=[
        mpatches.Patch(color=COLORS["baro"], label="p < 0.05 (significant)"),
        mpatches.Patch(color="#aaa",          label="p ≥ 0.05"),
    ], fontsize=7.5, loc="lower right")

    fig.suptitle("Fig 3 — Pre-event deterioration and inter-cohort gradient",
                 fontsize=11, fontweight="bold", y=1.01)
    _save(fig, "fig3_delta_gradient")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Temporal persistence (lead time 0–30 min)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_temporal_persistence():
    """AUC vs lead time for hypotension and hypertension.
    Left panel: Clínic OOS-CV (10×50 folds, linear extrapolation).
    Right panel: VitalDB OOS (models trained on Clínic, scored on VitalDB controls).
    """
    lt_clinic  = pd.read_csv("results/lead_time/lead_time_auc.csv")
    lt_raw     = pd.read_csv("results/lead_time/lead_time_raw_auc_combined.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    fig.suptitle(
        "Fig 4 — Temporal persistence of prediction performance (0–30 min lead time)",
        fontsize=11, fontweight="bold", y=1.01)

    outcomes   = ["hypotension", "hypertension"]
    out_colors = {
        "hypotension":  "#d73027",
        "hypertension": "#4575b4",
    }
    out_ls = {"hypotension": "-", "hypertension": "--"}

    # ── Clínic OOS-CV ─────────────────────────────────────────────────────────
    ax = axes[0]
    for outcome in outcomes:
        sub = lt_clinic[lt_clinic.event_type == outcome].sort_values("lead_min")
        color = out_colors[outcome]
        ax.plot(sub.lead_min, sub.auc, color=color, lw=2.0,
                ls=out_ls[outcome], marker="o", ms=5,
                label=EVENT_LABEL[outcome], zorder=3)
        ax.fill_between(sub.lead_min, sub.ci_lo, sub.ci_hi,
                        color=color, alpha=0.15, zorder=2)

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-1, 32)
    ax.set_ylim(0.40, 1.02)
    ax.set_xticks([0, 5, 10, 15, 20, 30])
    ax.set_xlabel("Lead time (minutes before event)", fontsize=9)
    ax.set_ylabel("AUC (95 % CI)", fontsize=9)
    ax.set_title("Clínic — OOS cross-validation\n(10 × 50 patient-level folds)",
                 fontsize=9, pad=5)
    ax.legend(fontsize=8, loc="lower left")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Annotate method change
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.text(0.5, 0.415, "← OOS CV  |  linear extrap →",
            fontsize=7, color="#666", va="bottom", ha="left")

    # ── VitalDB OOS ───────────────────────────────────────────────────────────
    ax = axes[1]
    vdb = lt_raw[lt_raw.ctrl_set == "vitaldb_oos"]
    ctr = lt_raw[lt_raw.ctrl_set == "clinic_training"]

    for outcome in outcomes:
        color = out_colors[outcome]
        sub_v = vdb[vdb.event_type == outcome].sort_values("lead_min")
        sub_c = ctr[ctr.event_type == outcome].sort_values("lead_min")

        ax.plot(sub_v.lead_min, sub_v.auc,
                color=color, lw=2.0, ls=out_ls[outcome],
                marker="s", ms=5, label=f"{EVENT_LABEL[outcome]} (VitalDB OOS)",
                zorder=3)
        ax.plot(sub_c.lead_min, sub_c.auc,
                color=color, lw=1.4, ls=out_ls[outcome],
                marker="^", ms=4, alpha=0.5,
                label=f"{EVENT_LABEL[outcome]} (Clínic train)",
                zorder=3)

    ax.axhline(0.5, color="gray", lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-1, 32)
    ax.set_ylim(0.40, 1.02)
    ax.set_xticks([0, 5, 10, 15, 30])
    ax.set_xlabel("Lead time (minutes before event)", fontsize=9)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("VitalDB — Out-of-sample\n(Clínic-trained model, raw re-extraction)",
                 fontsize=9, pad=5)

    # Custom legend (group by outcome, not by cohort)
    handles = []
    for outcome in outcomes:
        color = out_colors[outcome]
        handles.append(mpatches.Patch(color=color, alpha=0.85,
                                      label=EVENT_LABEL[outcome]))
    handles += [
        plt.Line2D([0], [0], color="k", marker="s", ms=5, lw=1.5,
                   label="VitalDB OOS"),
        plt.Line2D([0], [0], color="k", marker="^", ms=4, lw=1.2,
                   alpha=0.55, label="Clínic train"),
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="lower left",
              ncol=2, columnspacing=0.8)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 1])
    _save(fig, "fig4_temporal_persistence")


# ── helpers ────────────────────────────────────────────────────────────────────
def _save(fig, name: str):
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  Saved → {path}")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Fig 1 — STROBE…")
    fig1_strobe()

    print("Generating Fig 2 — MAP increment…")
    fig2_map_increment()

    print("Generating Fig 3 — Δ event-vs-control + gradient…")
    fig3_delta_gradient()

    print("Generating Fig 4 — Temporal persistence…")
    fig4_temporal_persistence()

    print("\nDone. All figures saved to results/figures/")
