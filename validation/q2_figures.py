"""
Q2 Figures  (PASO 4 + PASO 6)
──────────────────────────────
fig_violins_Q2a()       – pre vs control violins (Q2a)
fig_trajectories_Q2b()  – aligned ±5 min trajectories (Q2b)
fig_forest_Q2a()        – forest plot Q2a
fig_dissociation()      – KEY: Q1 vs Q2 β comparison (PASO 6)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_config import (
    FEATURES_LONG, Q1_RESULTS_V2, Q1_PAIRED_DATA,
    FIG, PRIMARY, FEAT_LABELS, ALPHA_BONF,
    PRE_START_S, PRE_END_S, POST_START_S, POST_END_S,
)

# ── colour palette ────────────────────────────────────────────────────────────
CLR_PRE     = "#4e79a7"
CLR_CTRL    = "#76b7b2"
CLR_POST    = "#f28e2b"
CLR_INTER   = "#e15759"
CLR_SUPRA   = "#59a14f"
CLR_Q1      = "#aaaaaa"
CLR_Q2      = "#e15759"


def _savefig(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Pre vs Control violins (Q2a confirmatory)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_violins_Q2a(paired: pd.DataFrame, results: pd.DataFrame, out_dir: Path):
    """Paired violin plot: pre-bolus (blue) vs quiescent control (teal)."""
    feats = [(f, a) for f, a, _ in PRIMARY]
    n     = len(feats)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (fname, agg) in zip(axes, feats):
        key = f"{fname}__{agg}"
        sub = paired[paired["feature_agg"] == key].dropna(subset=["value_pre", "value_ctrl"])

        data_pre  = sub["value_pre"].values
        data_ctrl = sub["value_ctrl"].values

        parts = ax.violinplot([data_ctrl, data_pre], positions=[0, 1],
                               showmedians=True, widths=0.7)
        parts["bodies"][0].set_facecolor(CLR_CTRL)
        parts["bodies"][1].set_facecolor(CLR_PRE)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)

        # Connecting lines
        for pre, ctrl in zip(data_pre, data_ctrl):
            ax.plot([0, 1], [ctrl, pre], color="grey", alpha=0.35, lw=0.8, zorder=1)

        # Jitter dots
        jit = np.random.RandomState(42).uniform(-0.06, 0.06, len(data_ctrl))
        ax.scatter(0 + jit, data_ctrl, color=CLR_CTRL, s=22, alpha=0.8, zorder=3)
        ax.scatter(1 + jit, data_pre,  color=CLR_PRE,  s=22, alpha=0.8, zorder=3)

        # Annotation
        res_row = results[results["feature"] == key]
        if not res_row.empty:
            r = res_row.iloc[0]
            sig = ("**" if r.get("p_bonferroni", 1) < ALPHA_BONF
                   else "*"  if r.get("p_bonferroni", 1) < 0.05
                   else "ns")
            ax.set_title(f"{FEAT_LABELS.get(key, key)}\n"
                         f"β={r.get('beta', np.nan):+.3f}  {sig}", fontsize=9)
        else:
            ax.set_title(FEAT_LABELS.get(key, key), fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Quiescent\n(control)", "Pre-bolus\n(−5 to 0 min)"], fontsize=8)
        ax.set_ylabel("Feature value", fontsize=8)

    fig.suptitle("Q2a – Pre-bolus vs Quiescent Control", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir / "q2_pre_vs_control_violins.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Aligned ±5 min trajectories (Q2b descriptive)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_trajectories_Q2b(
    events: pd.DataFrame,
    feat: pd.DataFrame,
    out_dir: Path,
    bin_s: int = 60,
):
    """
    Grid of ±5 min time series aligned to bolus for each primary feature.
    Individual events as thin lines, median as thick line, stratified by group.
    """
    feat = feat.copy()
    feat["patient_id"] = feat["patient_id"].astype(str)
    clean = events[events["status"] == "clean"].copy()
    feats = [(f, a) for f, a, _ in PRIMARY]
    n     = len(feats)

    bins   = np.arange(PRE_START_S, POST_END_S + bin_s, bin_s)
    bin_c  = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (fname, agg) in zip(axes, feats):
        key = f"{fname}__{agg}"
        if fname not in feat.columns:
            ax.set_title(f"{key}\n(no data)")
            continue

        medians_inter  = np.full(len(bin_c), np.nan)
        medians_supra  = np.full(len(bin_c), np.nan)

        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            vals_i, vals_s = [], []
            for _, ev in clean.iterrows():
                pid = str(ev["patient_id"])
                t_b = float(ev["t_bolus"])
                grp = str(ev["group"])
                pat = feat[feat["patient_id"] == pid]
                w   = pat[
                    (pat["t_window_start_s"] >= t_b + lo)
                    & (pat["t_window_start_s"] <  t_b + hi)
                    & (pat[fname].notna())
                ]
                if len(w) == 0:
                    continue
                val = float(w[fname].median())
                (vals_i if grp == "interescalenico" else vals_s).append(val)

            if vals_i:
                medians_inter[i] = np.median(vals_i)
            if vals_s:
                medians_supra[i] = np.median(vals_s)

        valid_i = ~np.isnan(medians_inter)
        valid_s = ~np.isnan(medians_supra)
        if valid_i.any():
            ax.plot(bin_c[valid_i], medians_inter[valid_i],
                    color=CLR_INTER, lw=2.5, label="Interescalénico", zorder=3)
        if valid_s.any():
            ax.plot(bin_c[valid_s], medians_supra[valid_s],
                    color=CLR_SUPRA, lw=2.5, linestyle="--", label="Supra-axilar", zorder=3)

        ax.axvline(0, color="red", lw=1.2, linestyle=":", alpha=0.8, label="Bolus")
        ax.axvspan(PRE_START_S, 0, alpha=0.06, color=CLR_PRE)
        ax.set_xlabel("Time re. bolus (s)", fontsize=8)
        ax.set_title(FEAT_LABELS.get(key, key), fontsize=9)
        ax.set_xlim(PRE_START_S, POST_END_S)

    axes[0].set_ylabel("Median feature value", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Q2b – Trajectories ±5 min around vasopresor bolus", fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir / "q2_pre_vs_post_trajectories.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Forest plot Q2a
# ═══════════════════════════════════════════════════════════════════════════════

def fig_forest_Q2a(results: pd.DataFrame, paired: pd.DataFrame, out_dir: Path):
    """
    Horizontal forest plot of standardised β (= Cohen's d_z) for primary
    features.  Colour encodes significance vs Bonferroni threshold.
    """
    prim = results[results["analysis"] == "primary"].copy()
    if prim.empty:
        print("  [WARN] No primary results for forest plot")
        return

    # Standardise IC by sd_delta
    prim = prim.copy()
    prim["dz"]     = prim["cohen_dz"].values
    prim["dz_lo"]  = np.where(
        prim["sd_delta"] > 0,
        prim["ic_lo"] / prim["sd_delta"],
        np.nan,
    )
    prim["dz_hi"]  = np.where(
        prim["sd_delta"] > 0,
        prim["ic_hi"] / prim["sd_delta"],
        np.nan,
    )

    labels = [FEAT_LABELS.get(r["feature"], r["feature"]) for _, r in prim.iterrows()]
    n      = len(prim)

    fig, ax = plt.subplots(figsize=(7, max(3, 1.1 * n)))
    ys = np.arange(n)

    for i, (_, row) in enumerate(prim.iterrows()):
        pval  = row.get("p_bonferroni", np.nan)
        color = (CLR_Q2 if not np.isnan(pval) and pval < ALPHA_BONF
                 else CLR_Q1)
        ax.errorbar(
            row["dz"],
            ys[i],
            xerr=[[row["dz"] - row["dz_lo"]], [row["dz_hi"] - row["dz"]]],
            fmt="o",
            color=color,
            markersize=7,
            capsize=4,
            lw=1.5,
        )

    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d_z  (standardised β)", fontsize=9)
    ax.set_title("Q2a Forest Plot – Pre-bolus vs Quiescent Control", fontweight="bold")

    sig_patch  = mpatches.Patch(color=CLR_Q2,  label=f"p_Bonf < {ALPHA_BONF:.3f}")
    ns_patch   = mpatches.Patch(color=CLR_Q1,  label="NS")
    ax.legend(handles=[sig_patch, ns_patch], fontsize=8)

    fig.tight_layout()
    _savefig(fig, out_dir / "q2_forest_plot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Q1 vs Q2 Dissociation  (PASO 6 – KEY FIGURE)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_dissociation(
    q2_results: pd.DataFrame,
    q2_paired: pd.DataFrame,
    out_dir: Path,
):
    """
    PASO 6. KEY FIGURE.
    Horizontal forest plot with two points per primary feature:
      grey = Q1 (pain response / homeostasis)
      red  = Q2 (pre-hypotension)
    Connected by a thin line.  Vertical dashed line at x=0.
    The expected dissociation for BRS-αLF:
      Q1: β > 0  (homeostasis preserved → BRS increases with pain)
      Q2: β < 0  (pre-hypotension → BRS deteriorates)
    """
    # Load Q1 primary results
    try:
        q1 = pd.read_csv(Q1_RESULTS_V2)
        q1 = q1[q1["analysis"] == "primary"].copy()
        q1_paired = pd.read_csv(Q1_PAIRED_DATA)
    except FileNotFoundError as e:
        print(f"  [WARN] Cannot load Q1 data: {e}")
        return

    # Compute Q1 sd_delta per feature
    q1_sd = {}
    for (f, a, _) in PRIMARY:
        key = f"{f}__{a}"
        sub = q1_paired[q1_paired["feature_agg"] == key]["delta"].dropna() \
            if "feature_agg" in q1_paired.columns \
            else q1_paired[q1_paired.get("feature", "") == key]["delta"].dropna() \
            if "feature" in q1_paired.columns else pd.Series(dtype=float)
        if len(sub) > 1:
            q1_sd[key] = float(sub.std(ddof=1))

    # Compute Q2 sd_delta per feature
    q2_sd = {}
    for (f, a, _) in PRIMARY:
        key = f"{f}__{a}"
        sub = q2_paired[q2_paired["feature_agg"] == key]["delta"].dropna()
        if len(sub) > 1:
            q2_sd[key] = float(sub.std(ddof=1))

    def _std_row(row, sd_map, key):
        """Return (dz, dz_lo, dz_hi)."""
        sd = sd_map.get(key, np.nan)
        if np.isnan(sd) or sd == 0:
            return (row.get("cohen_dz", np.nan), np.nan, np.nan)
        return (
            row.get("cohen_dz", np.nan),
            float(row.get("ic_lo", np.nan)) / sd,
            float(row.get("ic_hi", np.nan)) / sd,
        )

    feats  = [(f, a, d) for f, a, d in PRIMARY]
    n      = len(feats)
    labels = [FEAT_LABELS.get(f"{f}__{a}", f"{f}__{a}") for f, a, _ in feats]

    fig, ax = plt.subplots(figsize=(8, max(4, 1.3 * n)))
    ys   = np.arange(n)
    dy   = 0.2  # vertical offset between Q1 and Q2

    for i, (fname, agg, _) in enumerate(feats):
        key = f"{fname}__{agg}"

        # ── Q1 point ──────────────────────────────────────────────────────────
        # Q1 CSV: columns 'feature' (name only) and 'agg' (separate)
        if "agg" in q1.columns:
            q1_row = q1[(q1["feature"] == fname) & (q1["agg"] == agg)]
        else:
            q1_row = q1[q1["feature"] == key]

        if not q1_row.empty:
            r1 = q1_row.iloc[0]
            dz1, lo1, hi1 = _std_row(r1, q1_sd, key)
            if not np.isnan(dz1):
                ax.errorbar(
                    dz1, ys[i] + dy,
                    xerr=[[dz1 - lo1], [hi1 - dz1]] if not np.isnan(lo1) else None,
                    fmt="s", color=CLR_Q1, markersize=8, capsize=3, lw=1.5,
                    label="Q1 (dolor)" if i == 0 else "_",
                )

        # ── Q2 point ──────────────────────────────────────────────────────────
        q2_row = q2_results[q2_results["feature"] == key]
        if q2_row.empty and "feature_name" in q2_results.columns:
            q2_row = q2_results[
                (q2_results["feature_name"] == fname) &
                (q2_results["agg"] == agg)
            ]

        if not q2_row.empty:
            r2  = q2_row.iloc[0]
            dz2 = r2.get("cohen_dz", np.nan)
            sd2 = q2_sd.get(key, np.nan)
            lo2 = (float(r2.get("ic_lo", np.nan)) / sd2
                   if not np.isnan(sd2) and sd2 != 0 else np.nan)
            hi2 = (float(r2.get("ic_hi", np.nan)) / sd2
                   if not np.isnan(sd2) and sd2 != 0 else np.nan)

            pval = r2.get("p_bonferroni", np.nan)
            clr  = CLR_Q2 if not np.isnan(pval) and pval < ALPHA_BONF else "#f4a460"

            if not np.isnan(dz2):
                ax.errorbar(
                    dz2, ys[i] - dy,
                    xerr=[[dz2 - lo2], [hi2 - dz2]] if not np.isnan(lo2) else None,
                    fmt="o", color=clr, markersize=8, capsize=3, lw=1.5,
                    label="Q2 (pre-hipotensión)" if i == 0 else "_",
                )

        # ── Connecting line ────────────────────────────────────────────────────
        if not q1_row.empty and not q2_row.empty:
            dz1_c = (q1_row.iloc[0].get("cohen_dz", np.nan)
                     if not q1_row.empty else np.nan)
            dz2_c = (q2_row.iloc[0].get("cohen_dz", np.nan)
                     if not q2_row.empty else np.nan)
            if not np.isnan(dz1_c) and not np.isnan(dz2_c):
                ax.plot([dz1_c, dz2_c], [ys[i] + dy, ys[i] - dy],
                        color="grey", lw=0.8, alpha=0.6, zorder=0)

    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's d_z  (standardised β)", fontsize=10)
    ax.set_title(
        "Disociación autonómica Q1 (dolor) vs Q2 (pre-hipotensión)",
        fontweight="bold",
    )

    # Legend
    q1_patch = mpatches.Patch(color=CLR_Q1,  label="Q1 – Respuesta al dolor")
    q2_patch = mpatches.Patch(color=CLR_Q2,  label="Q2 – Firma pre-hipotensión")
    ax.legend(handles=[q1_patch, q2_patch], fontsize=9, loc="lower right")

    # Annotation guide
    ax.text(0.01, 0.01,
            "← Disminución pre-hipotensión\nAumento →",
            transform=ax.transAxes, fontsize=7, color="grey", va="bottom")

    fig.tight_layout()
    _savefig(fig, out_dir / "q1_vs_q2_dissociation.png")

    # ── Dissociation verdict ───────────────────────────────────────────────────
    _print_dissociation_verdict(q2_results)


def _print_dissociation_verdict(q2_results: pd.DataFrame):
    """Print pre-specified dissociation verdict."""
    print("\n[PASO 6] Dissociation verdict:")
    # Criterion A: PTT_cv std OR PTT_std max crosses Bonferroni in Q2 (negative)
    a_feats = ["ptt_cv__std", "ptt_std__max"]
    crit_a  = False
    for key in a_feats:
        row = q2_results[q2_results["feature"] == key]
        if not row.empty:
            r = row.iloc[0]
            if r.get("beta", 0) < 0 and r.get("p_bonferroni", 1) < ALPHA_BONF:
                crit_a = True
                print(f"  Criterion A met by {key} (β={r['beta']:+.4f}, p_Bonf={r['p_bonferroni']:.3f})")

    # Criterion B: brs_alpha_lf min in Q2 negative p<0.05
    brs_row = q2_results[q2_results["feature"] == "brs_alpha_lf__min"]
    crit_b  = False
    if not brs_row.empty:
        r = brs_row.iloc[0]
        if r.get("beta", 0) < 0 and r.get("p_2sided", 1) < 0.05:
            crit_b = True
            print(f"  Criterion B met: BRS-αLF β={r['beta']:+.4f}, p={r['p_2sided']:.3f}")

    # Also check if BRS is positive in Q2 (alternative result)
    brs_positive_q2 = (not brs_row.empty and brs_row.iloc[0].get("beta", 0) > 0
                       and brs_row.iloc[0].get("p_2sided", 1) < 0.05)

    if crit_a and crit_b:
        verdict = "**DISOCIACIÓN CONFIRMADA**"
    elif crit_a or crit_b:
        verdict = "DISOCIACIÓN PARCIAL"
    elif brs_positive_q2:
        verdict = "RESULTADO ALTERNATIVO (BRS positiva también en Q2)"
    else:
        verdict = "NO CONFIRMADA"

    print(f"\n  VERDICT: {verdict}\n"
          f"  (Criterion A={crit_a}, Criterion B={crit_b}, BRS+Q2={brs_positive_q2})")
