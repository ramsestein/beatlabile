#!/usr/bin/env python
"""
Generate three manuscript figures for BeatLabile validation cohort.

Figures:
  figure_6_dissociation.png        — Q1 vs Q2 dissociation forest plot
  figure_5_conceptual_model.png    — Two-tier autonomic model diagram
  sup_fig_validation_flowchart.png — CONSORT/STROBE-style patient flowchart

Output: results/validation/manuscript_figures/
All text in English.  300 dpi.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
Q1_RES   = ROOT / "results" / "validation" / "Q1"
Q2V2_RES = ROOT / "results" / "validation" / "Q2v2"
OUT      = ROOT / "results" / "validation" / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared palette ────────────────────────────────────────────────────────────
C_Q1   = "#555555"   # dark grey  — Q1 (nociceptive)
C_Q2   = "#C0392B"   # red        — Q2 (pre-vasopressor)
C_T1   = "#AED6F1"   # light blue — Tier 1
C_T2   = "#F1948A"   # light red  — Tier 2
C_EXCL = "#FDEDEC"   # pale red   — exclusion boxes

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Dissociation Q1 vs Q2  (Figure 6 of manuscript)
# ══════════════════════════════════════════════════════════════════════════════
def fig_A():
    """Forest-plot dissociation panel: 6 features × 2 cohorts (Q1 grey, Q2 red)."""

    q1 = pd.read_csv(Q1_RES / "test_results_v2.csv")
    q2 = pd.read_csv(Q2V2_RES / "test_results_Q2a_v2.csv")

    q1_prim = q1[q1["analysis"].isin(["primary", "primary_brs_seq"])].copy()

    # 6-panel order: PTT-CV mean, PTT-CV std, PTT-SD max, PAI mean, BRS-seq, BRS-αLF
    panels = [
        # (feat_q1, agg_q1,  feat_q2_key,        display_title,     x_label)
        ("ptt_cv",       "mean", "ptt_cv__mean",      "PTT-CV (mean)",   "Δ PTT-CV [a.u.]"),
        ("ptt_cv",       "std",  "ptt_cv__std",        "PTT-CV (std)",    "Δ PTT-CV std [a.u.]"),
        ("ptt_std",      "max",  "ptt_std__max",       "PTT-SD (max)",    "Δ PTT-SD [ms]"),
        ("pai_mean",     "mean", "pai_mean__mean",     "PAI (mean)",      "Δ PAI [a.u.]"),
        ("brs_seq",      "min",  "brs_seq__min",       "BRS-seq (min)",   "Δ BRS-seq [ms/mmHg]"),
        ("brs_alpha_lf", "min",  "brs_alpha_lf__min",  "BRS-α LF (min)", "Δ BRS-α LF [ms/mmHg]"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8), constrained_layout=True)
    axes = axes.flatten()

    for i, (feat1, agg1, feat2_key, title, xlabel) in enumerate(panels):
        ax = axes[i]

        # ── Retrieve rows ──────────────────────────────────────────────────
        if feat1 == "brs_seq":
            r1_df = q1_prim[q1_prim["analysis"] == "primary_brs_seq"]
        else:
            r1_df = q1_prim[
                (q1_prim["feature"] == feat1) & (q1_prim["agg"] == agg1)
            ]
        r1 = r1_df.iloc[0] if len(r1_df) else None

        r2_df = q2[q2["feature"] == feat2_key]
        r2 = r2_df.iloc[0] if len(r2_df) else None

        # ── Plot each row ──────────────────────────────────────────────────
        y_pos    = [1.0, 0.0]
        y_labels = ["Q1\n(nociceptive)", "Q2\n(pre-vasopressor)"]
        colors   = [C_Q1, C_Q2]
        rows     = [r1, r2]
        xlims    = []

        for yi, (row, color, yp) in enumerate(zip(rows, colors, y_pos)):
            if row is None:
                ax.text(0, yp, "N/A", ha="center", va="center",
                        color=color, fontsize=8, style="italic")
                continue

            beta = float(row["beta"])
            lo   = float(row["ic_lo"])
            hi   = float(row["ic_hi"])

            # Significance: compare p_bonferroni < 0.05
            try:
                pb = float(row["p_bonferroni"])
                is_sig = pb < 0.05
            except (TypeError, ValueError):
                is_sig = False

            try:
                p_1s = float(row["p_1sided"])
            except (TypeError, ValueError):
                p_1s = np.nan

            cap_h = 0.06

            # CI bar
            ax.plot([lo, hi], [yp, yp], color=color, lw=2.2,
                    solid_capstyle="butt", zorder=3)
            # End caps
            for xc in [lo, hi]:
                ax.plot([xc, xc], [yp - cap_h, yp + cap_h],
                        color=color, lw=1.5, zorder=3)
            # Central point: diamond if significant, circle otherwise
            marker = "D" if is_sig else "o"
            ms     = 9  if is_sig else 8
            ax.plot(beta, yp, marker=marker, ms=ms, color=color,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=5)

            # p-value annotation (one-sided, uncorrected) above point
            if not np.isnan(p_1s):
                p_str = "p<0.001" if p_1s < 0.001 else f"p={p_1s:.3f}"
                ax.text(beta, yp + 0.27, p_str,
                        ha="center", va="bottom", fontsize=6.2, color=color)

            xlims += [lo, hi, beta]

        # Dashed reference line at β = 0
        ax.axvline(0, color="#333333", lw=0.8, ls="--", zorder=2, alpha=0.75)

        # Shaded band around zero — width based on smaller SE of the two rows
        # Acts as a visual "null zone" guide
        half_null = 0.0
        for row in [r1, r2]:
            if row is not None:
                try:
                    se = (float(row["ic_hi"]) - float(row["ic_lo"])) / (2 * 1.96)
                    if half_null == 0.0:
                        half_null = se
                    else:
                        half_null = min(half_null, se)
                except Exception:
                    pass
        if half_null > 0:
            ax.axvspan(-half_null, half_null, color="lightgrey",
                       alpha=0.25, zorder=1, lw=0)

        # Axes formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=7.5)
        ax.set_ylim(-0.55, 1.70)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False, axis="x", labelsize=7.5)

        # Auto x-limits
        if xlims:
            xmin, xmax = min(xlims), max(xlims)
            rng = max(xmax - xmin, 1e-8)
            pad = rng * 0.25
            ax.set_xlim(xmin - pad, xmax + pad)

    # ── Figure-level legend & titles ──────────────────────────────────────
    legend_elems = [
        mlines.Line2D([0], [0], color=C_Q1, marker="o", ms=7, ls="-",
                      label="Q1 — nociceptive stimuli (n=52, intact homeostasis)"),
        mlines.Line2D([0], [0], color=C_Q2, marker="o", ms=7, ls="-",
                      label="Q2 — pre-vasopressor (n=10, decompensating)"),
        mlines.Line2D([0], [0], color="gray", marker="D", ms=7, ls="none",
                      label="Bonferroni-corrected significant (◆)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Q1 vs Q2 — Autonomic Feature Dissociation",
        fontsize=13, fontweight="bold", y=1.03,
    )
    fig.text(
        0.5, 1.005,
        "Bars: β ± 95% CI.   Grey band: null zone (±1 SE).   "
        "p values: one-sided, uncorrected.",
        ha="center", va="bottom", fontsize=7.5, style="italic", color="dimgray",
    )

    out_path = OUT / "figure_6_dissociation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — Two-tier conceptual model  (Figure 5 of manuscript)
# ══════════════════════════════════════════════════════════════════════════════
def _rbox(ax, x, y, w, h, facecolor, edgecolor="#444444",
          lw=1.3, radius=0.25, alpha=0.90, zorder=2):
    """Draw a rounded rectangle patch."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _text(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("zorder", 10)
    ax.text(x, y, s, **kw)


def _arrow(ax, x0, y0, x1, y1, color="#444444", lw=1.4,
           style="->", zorder=6):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=zorder,
    )


def fig_B():
    """Conceptual two-tier model with phenotype panels."""

    fig = plt.figure(figsize=(14, 7.5), facecolor="white")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # ── Figure title ──────────────────────────────────────────────────────
    _text(ax, 7, 7.25,
          "BeatLabile Two-Tier Autonomic Model",
          fontsize=14, fontweight="bold", color="#1A1A1A")
    _text(ax, 7, 6.95,
          "Sympathetic activation (Tier 1) is universal; "
          "homeostatic failure (Tier 2) is state-specific.",
          fontsize=9, style="italic", color="dimgray")

    # ─────────────────────────────────────────────────────────────────────
    # LEFT COLUMN — Tier boxes
    # ─────────────────────────────────────────────────────────────────────
    BX, BW, BH = 0.25, 4.6, 2.1

    # Tier 1 box  (top, blue)
    T1_Y = 4.15
    _rbox(ax, BX, T1_Y, BW, BH, facecolor=C_T1, edgecolor="#2471A3", lw=1.8)
    _text(ax, BX + BW/2, T1_Y + BH - 0.32,
          "TIER 1 — Sympathetic Activation", fontsize=10, fontweight="bold",
          color="#154360")
    _text(ax, BX + BW/2, T1_Y + BH - 0.70,
          "(variability rigidification)", fontsize=8.5, style="italic",
          color="#1F618D")
    _text(ax, BX + BW/2, T1_Y + 0.95,
          "Trigger: nociceptive stimulus · stress · haemodynamic perturbation",
          fontsize=8, color="#2C3E50")
    _text(ax, BX + BW/2, T1_Y + 0.60,
          "Signature:  PTT-CV std ↓    PTT-SD max ↓    ARV std ↑",
          fontsize=8.5, color="#1F618D", fontweight="bold")
    _text(ax, BX + BW/2, T1_Y + 0.24,
          "Always present when sympathetic outflow rises",
          fontsize=8, style="italic", color="#555555")

    # Tier 2 box  (below, red)
    T2_Y = 1.65
    _rbox(ax, BX, T2_Y, BW, BH, facecolor=C_T2, edgecolor="#922B21", lw=1.8)
    _text(ax, BX + BW/2, T2_Y + BH - 0.32,
          "TIER 2 — Homeostatic Failure", fontsize=10, fontweight="bold",
          color="#641E16")
    _text(ax, BX + BW/2, T2_Y + BH - 0.70,
          "(baroreflex deterioration)", fontsize=8.5, style="italic",
          color="#922B21")
    _text(ax, BX + BW/2, T2_Y + 0.95,
          "Trigger: sympathetic compensation overwhelmed",
          fontsize=8, color="#2C3E50")
    _text(ax, BX + BW/2, T2_Y + 0.60,
          "Signature:  BRS-seq min ↓    BRS-α LF ↓",
          fontsize=8.5, color="#922B21", fontweight="bold")
    _text(ax, BX + BW/2, T2_Y + 0.24,
          "Specific to pathological pre-event states",
          fontsize=8, style="italic", color="#555555")

    # Arrow Tier 1 → Tier 2
    arr_x = BX + BW/2
    _arrow(ax, arr_x, T1_Y, arr_x, T2_Y + BH,
           color="#555555", lw=2.0, style="-|>")
    _text(ax, arr_x + 0.35, (T1_Y + T2_Y + BH) / 2,
          "if severe\nenough", fontsize=7.5, ha="left",
          style="italic", color="#777777")

    # Label above left column
    _text(ax, BX + BW/2, T1_Y + BH + 0.22,
          "Universal cascade", fontsize=8.5, fontweight="bold",
          color="#333333")

    # ─────────────────────────────────────────────────────────────────────
    # RIGHT COLUMN — Phenotype panels (A, B, C)
    # ─────────────────────────────────────────────────────────────────────
    PX  = 5.3    # left edge of phenotype section
    PW  = 8.4    # total width of phenotype section
    PH  = 1.55   # height of each phenotype block
    GAP = 0.28   # vertical gap between blocks

    # Vertical position of top of each block
    TOP_A = T1_Y + BH - 0.0         # align with top of Tier 1
    TOP_B = TOP_A - PH - GAP
    TOP_C = TOP_B - PH - GAP

    phenotypes = [
        {
            "label": "A",
            "title": "Phenotype A — Nociceptive Response, Preserved Homeostasis",
            "tiers": [
                ("Tier 1", True,  C_T1, "#2471A3"),
                ("Tier 2", False, "#EEEEEE", "#AAAAAA"),
            ],
            "effect":  "PTT-CV std ↓   (rigidification only)",
            "example": "Q1 validation cohort\n(pain stimuli under anaesthesia)",
            "edge":    "#2471A3",
            "top":     TOP_A,
        },
        {
            "label": "B",
            "title": "Phenotype B — Pre-Hypertensive Event",
            "tiers": [
                ("Tier 1", True, C_T1,  "#2471A3"),
                ("Tier 2", True, C_T2,  "#922B21"),
            ],
            "effect":  "PTT-CV std ↓   +   BRS-seq ↓   (rigidification + decompensation)",
            "example": "VitalDB primary cohort\n(hypertensive events)",
            "edge":    "#E59866",
            "top":     TOP_B,
        },
        {
            "label": "C",
            "title": "Phenotype C — Pre-Hypotensive Event",
            "tiers": [
                ("Tier 1", True, C_T1,  "#2471A3"),
                ("Tier 2", True, C_T2,  "#922B21"),
            ],
            "effect":  "PTT-CV ↑ (oscillates)   +   BRS-seq ↓   (vasodilatory context)",
            "example": "VitalDB primary cohort\n(hypotensive events)",
            "edge":    "#5DADE2",
            "top":     TOP_C,
        },
    ]

    for ph in phenotypes:
        px    = PX
        py    = ph["top"] - PH      # bottom-left corner
        pw    = PW
        ph_h  = PH
        ecol  = ph["edge"]

        # Background box
        _rbox(ax, px, py, pw, ph_h,
              facecolor="#F9F9F9", edgecolor=ecol, lw=1.5, radius=0.18)

        # Label letter on left
        ax.text(px + 0.25, py + ph_h/2, ph["label"],
                ha="center", va="center", fontsize=16, fontweight="bold",
                color=ecol, zorder=10)

        # Title
        ax.text(px + 0.55, py + ph_h - 0.21, ph["title"],
                ha="left", va="top", fontsize=8.5, fontweight="bold",
                color="#222222", zorder=10)

        # Tier badges
        badge_y = py + 0.14
        badge_w, badge_h = 1.55, 0.38
        for bi, (tier_lbl, active, bface, bedge) in enumerate(ph["tiers"]):
            bx = px + 0.55 + bi * (badge_w + 0.15)
            _rbox(ax, bx, badge_y, badge_w, badge_h,
                  facecolor=bface if active else "#F0F0F0",
                  edgecolor=bedge, lw=0.9, radius=0.08,
                  alpha=0.95 if active else 0.6, zorder=8)
            status = "ACTIVE" if active else "inactive"
            ax.text(bx + badge_w/2, badge_y + badge_h/2,
                    f"{tier_lbl}: {status}",
                    ha="center", va="center", fontsize=7.0,
                    fontweight="bold" if active else "normal",
                    color="#222222" if active else "#999999", zorder=9)

        # Effect text (centre-left)
        ax.text(px + 0.55, py + 0.60,
                "Effect:  " + ph["effect"],
                ha="left", va="center", fontsize=7.5,
                color="#333333", style="italic", zorder=10)

        # Example (right side)
        ax.text(px + pw - 0.25, py + ph_h/2,
                ph["example"],
                ha="right", va="center", fontsize=7.5,
                color="#666666", zorder=10)

    # Arrow from Tier boxes to phenotype section
    mid_tier_y = (T1_Y + T2_Y + BH) / 2
    _arrow(ax, BX + BW + 0.12, mid_tier_y + 0.4,
           PX - 0.08, TOP_A - PH/2 + 0.1,
           color="#999999", lw=1.2, style="-|>")
    _arrow(ax, BX + BW + 0.12, mid_tier_y,
           PX - 0.08, TOP_B - PH/2 + 0.1,
           color="#999999", lw=1.2, style="-|>")
    _arrow(ax, BX + BW + 0.12, mid_tier_y - 0.4,
           PX - 0.08, TOP_C - PH/2 + 0.1,
           color="#999999", lw=1.2, style="-|>")

    # Right column label
    _text(ax, PX + PW/2, TOP_A + 0.22,
          "Clinical phenotypes", fontsize=8.5, fontweight="bold",
          color="#333333")

    out_path = OUT / "figure_5_conceptual_model.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Validation cohort flowchart  (Supplementary Figure)
# ══════════════════════════════════════════════════════════════════════════════
def fig_C():
    """CONSORT/STROBE-style participant and event flowchart."""

    fig = plt.figure(figsize=(12, 14), facecolor="white")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # ── Helpers ───────────────────────────────────────────────────────────
    MAIN_W = 4.8   # width of main flow boxes
    EXCL_W = 3.8   # width of exclusion boxes
    BOX_H  = 0.72  # height of single-line main boxes
    MCX    = 6.0   # centre-x for main flow column

    def main_box(cx, cy, lines, facecolor="#EAF4FB", edgecolor="#2471A3"):
        n   = len(lines)
        bh  = BOX_H + 0.32 * max(n - 1, 0)
        x   = cx - MAIN_W / 2
        y   = cy - bh / 2
        _rbox(ax, x, y, MAIN_W, bh,
              facecolor=facecolor, edgecolor=edgecolor, lw=1.5, radius=0.15)
        for j, line in enumerate(lines):
            offset = (n - 1) / 2 - j
            fw = "bold" if j == 0 else "normal"
            ax.text(cx, cy + offset * 0.30, line,
                    ha="center", va="center", fontsize=8.5,
                    fontweight=fw, zorder=5)
        return bh

    def excl_box(right_edge_x, cy, lines):
        n   = len(lines)
        eh  = 0.55 + 0.28 * max(n - 1, 0)
        x   = right_edge_x
        y   = cy - eh / 2
        ex  = x + EXCL_W
        _rbox(ax, x, y, EXCL_W, eh,
              facecolor=C_EXCL, edgecolor="#C0392B", lw=1.0, radius=0.12)
        for j, line in enumerate(lines):
            offset = (n - 1) / 2 - j
            ax.text(x + EXCL_W / 2, cy + offset * 0.26, line,
                    ha="center", va="center", fontsize=7.5,
                    color="#C0392B", zorder=5)

    def v_arrow(cx, y_top, y_bot):
        ax.annotate(
            "", xy=(cx, y_bot + 0.04), xytext=(cx, y_top - 0.04),
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.3),
            zorder=6,
        )

    def h_branch(from_x, branch_y, to_x):
        """Horizontal line segment (no arrow) for the branch fork."""
        ax.plot([from_x, to_x], [branch_y, branch_y],
                color="#444444", lw=1.3, zorder=5)

    def excl_connector(cx, mid_y, excl_left_x):
        """
        Draw a line from the main downward arrow at mid_y going right
        to the exclusion box.
        """
        ax.plot([cx, excl_left_x], [mid_y, mid_y],
                color="#C0392B", lw=1.1, ls="-", zorder=4)
        ax.annotate(
            "", xy=(excl_left_x - 0.04, mid_y),
            xytext=(excl_left_x - 0.04, mid_y),
            arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.0),
            zorder=5,
        )
        # small arrow tip at box edge
        ax.annotate(
            "", xy=(excl_left_x + 0.02, mid_y),
            xytext=(excl_left_x - 0.25, mid_y),
            arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.0),
            zorder=5,
        )

    # ── Title ─────────────────────────────────────────────────────────────
    ax.text(MCX, 13.65,
            "Validation Cohort — Patient and Event Selection Flowchart",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color="#1A1A1A", zorder=5)
    ax.text(MCX, 13.32,
            "Shoulder arthroplasty patients — total intravenous anaesthesia (TIVA)",
            ha="center", va="top", fontsize=9, style="italic",
            color="dimgray", zorder=5)

    # ── Main patient screening flow ───────────────────────────────────────
    y_scr  = 12.70
    y_usa  = 11.30
    y_inc  = 9.90
    y_note = 9.22  # italic sub-note
    y_bra  = 8.85  # branch fork y

    # 1. Screened
    main_box(MCX, y_scr, ["21 patients screened"])

    # Exclusion 1 (between screened → usable)
    exc1_y   = (y_scr + y_usa) / 2 + 0.05
    exc1_lx  = MCX + MAIN_W / 2 + 0.18
    excl_connector(MCX, exc1_y, exc1_lx)
    excl_box(exc1_lx, exc1_y,
             ["2 excluded — no .vital file",
              "(patients 4234018, 70078466)"])
    v_arrow(MCX, y_scr - BOX_H/2, y_usa + BOX_H/2)

    # 2. Usable signal
    main_box(MCX, y_usa, ["19 patients with usable signal"])

    # Exclusion 2 (between usable → included)
    exc2_y   = (y_usa + y_inc) / 2 + 0.05
    exc2_lx  = MCX + MAIN_W / 2 + 0.18
    excl_connector(MCX, exc2_y, exc2_lx)
    excl_box(exc2_lx, exc2_y,
             ["1 excluded — PPG quality <70%",
              "(patient 5684023)"])
    v_arrow(MCX, y_usa - BOX_H/2, y_inc + BOX_H/2)

    # 3. Included
    main_box(MCX, y_inc,
             ["18 patients included in analysis"],
             facecolor="#D5F5E3", edgecolor="#1E8449")

    # Sub-note on recording duration
    ax.text(MCX, y_note,
            "17 full-duration recordings (≥60 min)  ·  1 partial recording "
            "(patient 70767707, 23 min)",
            ha="center", va="center", fontsize=7.8,
            style="italic", color="#555555", zorder=5)

    # Arrow to branch
    v_arrow(MCX, y_inc - BOX_H/2 - 0.15, y_bra + 0.05)

    # ── Branch fork ───────────────────────────────────────────────────────
    Q1_CX = 2.9    # centre-x of Q1 branch (left)
    Q2_CX = 9.1    # centre-x of Q2 branch (right)

    h_branch(Q1_CX, y_bra, Q2_CX)

    # Arrows downward into each branch
    y_q1_head = y_bra - 0.05
    y_q2_head = y_bra - 0.05

    # ── Q1 branch  (left) ─────────────────────────────────────────────────
    y_q1_raw   = 7.85
    y_q1_excl  = 6.75
    y_q1_clean = 5.42

    v_arrow(Q1_CX, y_q1_head, y_q1_raw + BOX_H/2 + 0.05)

    main_box(Q1_CX, y_q1_raw,
             ["Q1: 110 nociceptive stimuli annotated"],
             facecolor="#EBF5FB", edgecolor="#2471A3")

    v_arrow(Q1_CX, y_q1_raw - BOX_H/2,
            y_q1_excl + 0.60)  # top of exclusion block

    # Q1 exclusion block (centred, not offset right for readability)
    eq1_h = 1.15
    eq1_x = Q1_CX - EXCL_W / 2
    eq1_y = y_q1_excl - eq1_h / 2
    _rbox(ax, eq1_x, eq1_y, EXCL_W, eq1_h,
          facecolor=C_EXCL, edgecolor="#C0392B", lw=1.0, radius=0.12)
    excl_q1_lines = [
        "58 stimuli excluded:",
        "· overlap with other stimuli (±5 min)",
        "· vasopressor bolus within ±5 min",
        "· major infusion change within ±5 min",
        "· invalid PPG signal",
        "· outside anaesthetic gas (AG) window",
    ]
    for j, line in enumerate(excl_q1_lines):
        offset = (len(excl_q1_lines) - 1) / 2 - j
        fw = "bold" if j == 0 else "normal"
        ax.text(Q1_CX, y_q1_excl + offset * 0.175,
                line, ha="center", va="center",
                fontsize=7.2, color="#C0392B",
                fontweight=fw, zorder=5)

    v_arrow(Q1_CX, eq1_y, y_q1_clean + BOX_H/2 + 0.04)

    main_box(Q1_CX, y_q1_clean,
             ["Q1: 52 clean events", "(in 18 patients)"],
             facecolor="#D5F5E3", edgecolor="#1E8449")

    # Q1 design note
    ax.text(Q1_CX, y_q1_clean - BOX_H/2 - 0.48,
            "Design: pre-window [−5, 0] min  vs  post-window [0, +5] min\n"
            "within each stimulus event (paired GLMM).",
            ha="center", va="center", fontsize=7.5,
            style="italic", color="#555555", zorder=5)

    # ── Q2 branch  (right) ────────────────────────────────────────────────
    y_q2_raw   = 7.85
    y_q2_excl  = 6.82
    y_q2_clean = 5.72
    y_q2_ctrl  = 5.10

    v_arrow(Q2_CX, y_q2_head, y_q2_raw + BOX_H/2 + 0.05)

    main_box(Q2_CX, y_q2_raw,
             ["Q2: 19 vasopressor boluses"],
             facecolor="#FEF9E7", edgecolor="#D4AC0D")

    v_arrow(Q2_CX, y_q2_raw - BOX_H/2,
            y_q2_excl + 0.55)

    # Q2 exclusion block
    eq2_h = 0.98
    eq2_x = Q2_CX - EXCL_W / 2
    eq2_y = y_q2_excl - eq2_h / 2
    _rbox(ax, eq2_x, eq2_y, EXCL_W, eq2_h,
          facecolor=C_EXCL, edgecolor="#C0392B", lw=1.0, radius=0.12)
    excl_q2_lines = [
        "8 boluses excluded:",
        "· other vasopressor within ±10 min",
        "· pain stimulus in pre-window (±2 min)",
        "· invalid features (PTT / PPG)",
        "· outside AG window",
    ]
    for j, line in enumerate(excl_q2_lines):
        offset = (len(excl_q2_lines) - 1) / 2 - j
        fw = "bold" if j == 0 else "normal"
        ax.text(Q2_CX, y_q2_excl + offset * 0.182,
                line, ha="center", va="center",
                fontsize=7.2, color="#C0392B",
                fontweight=fw, zorder=5)

    v_arrow(Q2_CX, eq2_y, y_q2_clean + BOX_H/2 + 0.04)

    main_box(Q2_CX, y_q2_clean,
             ["Q2: 11 clean events"],
             facecolor="#FEF9E7", edgecolor="#D4AC0D")

    # Sub-step: control window
    v_arrow(Q2_CX, y_q2_clean - BOX_H/2, y_q2_ctrl + BOX_H/2 + 0.04)
    excl_box(Q2_CX + MAIN_W/2 + 0.18, (y_q2_clean + y_q2_ctrl)/2,
             ["1 event excluded:", "no valid control window"])
    excl_connector(Q2_CX, (y_q2_clean + y_q2_ctrl)/2,
                   Q2_CX + MAIN_W/2 + 0.18)

    main_box(Q2_CX, y_q2_ctrl,
             ["Q2: 10 events with valid", "control window (in 7 patients)"],
             facecolor="#D5F5E3", edgecolor="#1E8449")

    # Q2 design note
    ax.text(Q2_CX, y_q2_ctrl - BOX_H/2 - 0.55,
            "Design: pre-event window [−5, 0] min  vs\n"
            "quiescent control window [≥3 min] (paired GLMM).",
            ha="center", va="center", fontsize=7.5,
            style="italic", color="#555555", zorder=5)

    # ── Branch labels ─────────────────────────────────────────────────────
    ax.text(Q1_CX, y_bra + 0.14, "Q1 — Nociceptive analysis",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color="#2471A3", zorder=5)
    ax.text(Q2_CX, y_bra + 0.14, "Q2 — Vasopressor analysis",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color="#D4AC0D", zorder=5)

    out_path = OUT / "sup_fig_validation_flowchart.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {OUT}\n")
    fig_A()
    fig_B()
    fig_C()
    print(f"\nDone. All figures saved to:\n  {OUT}")
