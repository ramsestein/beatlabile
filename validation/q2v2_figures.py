"""
Q2 v2 Figures  (PASO 6 + PASO 7)
──────────────────────────────────
Five figures:
  1. q2v2_pre_vs_control_violins.png    – Q2a violins (pre vs control)
  2. q2v2_pre_vs_post_trajectories.png  – Q2b ±5 min trajectories
  3. q2v2_forest_plot.png               – Q2a forest plot (primary features)
  4. brs_methods_comparison.png         – Scatter brs_seq vs brs_alpha_lf + forest
  5. q1_vs_q2v2_dissociation.png        – KEY: Q1 vs Q2v2 6-feature comparison
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr, sem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2v2_config import (
    FIG, PRIMARY, EXPLORATORY, ALPHA_BONF,
    PRE_START_S, PRE_END_S, POST_START_S, POST_END_S, WINDOW_S,
    CONTROL_DURATION_S, FEATURES_BRS_SEQ,
)

# Colour palette
C_PRE    = "#4e79a7"   # blue
C_CTRL   = "#8cd17d"   # green
C_POST   = "#e15759"   # red
C_Q1     = "#888888"   # grey
C_Q2V2   = "#c0392b"   # crimson

FEATURE_LABELS = {
    "ptt_cv__mean":      "PTT-CV mean",
    "ptt_cv__std":       "PTT-CV std",
    "ptt_std__max":      "PTT-std max",
    "pai_mean__mean":    "PAI mean",
    "brs_seq__min":      "BRS-seq min",
    "brs_alpha_lf__min": "BRS-α LF min",
    "ptt_std__std":      "PTT-std std",
    "ptt_std__slope":    "PTT-std slope",
    "ptt_arv__std":      "PTT-ARV std",
}

FIG.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Q2a violins
# ═══════════════════════════════════════════════════════════════════════════════

def fig_violins_Q2a_v2(paired: pd.DataFrame, results: pd.DataFrame):
    """
    Violin plots: paired[value_pre] vs paired[value_ctrl], one panel per primary feature.
    Colour: blue = pre, green = control.
    Mark validated features with (*).
    """
    n_feat = len(PRIMARY)
    fig, axes = plt.subplots(1, n_feat, figsize=(3 * n_feat, 4))
    if n_feat == 1:
        axes = [axes]

    for ax, (fname, agg, direction) in zip(axes, PRIMARY):
        key    = f"{fname}__{agg}"
        sub    = paired[(paired["feature_agg"] == key)].dropna(subset=["value_pre", "value_ctrl"])
        label  = FEATURE_LABELS.get(key, key)

        vp = sub[["value_pre", "value_ctrl"]].values
        if len(vp) < 2:
            ax.set_title(f"{label}\n(no data)", fontsize=8)
            continue

        parts = ax.violinplot(
            [sub["value_pre"].values, sub["value_ctrl"].values],
            positions=[1, 2],
            showmedians=True, showextrema=True,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([C_PRE, C_CTRL][i])
            pc.set_alpha(0.55)

        # Overlay paired lines
        for _, r in sub.iterrows():
            ax.plot([1, 2], [r["value_pre"], r["value_ctrl"]],
                    color="grey", alpha=0.4, lw=0.8, zorder=2)
        ax.plot([1, 2],
                [sub["value_pre"].median(), sub["value_ctrl"].median()],
                color="black", lw=1.5, zorder=3)

        # Verdict star
        res_row = results[results["feature"] == key]
        is_val = (not res_row.empty
                  and not pd.isna(res_row.iloc[0].get("p_bonferroni", np.nan))
                  and res_row.iloc[0]["p_bonferroni"] < 0.050)
        suffix = " (*)" if is_val else ""
        ax.set_title(f"{label}{suffix}", fontsize=8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Pre", "Control"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle("Q2a v2 — Pre vs Control windows  (violins)", fontsize=10, y=1.02)
    plt.tight_layout()
    out = FIG / "q2v2_pre_vs_control_violins.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG 1] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Q2b trajectories
# ═══════════════════════════════════════════════════════════════════════════════

def fig_trajectories_Q2b_v2(events: pd.DataFrame, feat: pd.DataFrame):
    """
    For each primary feature: median trajectory in [-5, +5] min around bolus.
    One panel per feature. X = time re bolus (min). Shaded = IQR.
    """
    all_features = PRIMARY  # 5 panels
    n_feat  = len(all_features)
    fig, axes = plt.subplots(1, n_feat, figsize=(3 * n_feat, 3.5), sharey=False)
    if n_feat == 1:
        axes = [axes]

    clean = events[events["status"] == "clean"].copy()
    if clean.empty:
        print("[FIG 2] no clean events — skipping trajectories")
        plt.close(fig)
        return

    feat_sub = feat.copy()
    feat_sub["patient_id"] = feat_sub["patient_id"].astype(str)

    for ax, (fname, agg, direction) in zip(axes, all_features):
        key   = f"{fname}__{agg}"
        label = FEATURE_LABELS.get(key, key)

        if fname not in feat_sub.columns:
            ax.set_title(f"{label}\n(no data)", fontsize=8)
            continue

        rows = []
        for _, ev in clean.iterrows():
            pid     = str(ev["patient_id"])
            t_bolus = float(ev["t_bolus"])
            pat     = feat_sub[feat_sub["patient_id"] == pid].copy()
            pat["t_rel_min"] = (pat["t_window_start_s"] - t_bolus) / 60.0

            window_sub = pat[
                (pat["t_rel_min"] >= PRE_START_S / 60.0 - 0.5)
                & (pat["t_rel_min"] <= POST_END_S / 60.0 + 0.5)
            ]
            for _, w in window_sub.iterrows():
                if pd.isna(w[fname]):
                    continue
                rows.append({"t_rel": float(w["t_rel_min"]), "value": float(w[fname]),
                             "event_id": int(ev["event_id"])})

        if not rows:
            ax.set_title(f"{label}\n(no data)", fontsize=8)
            continue

        df_t = pd.DataFrame(rows)
        # Bin into 1-min bins
        df_t["t_bin"] = np.round(df_t["t_rel"], 0).astype(float)
        gb = df_t.groupby("t_bin")["value"]
        t_bins  = np.array(sorted(gb.groups.keys()))
        medians = gb.median().reindex(t_bins).values
        q25     = gb.quantile(0.25).reindex(t_bins).values
        q75     = gb.quantile(0.75).reindex(t_bins).values

        ax.fill_between(t_bins, q25, q75, alpha=0.25, color=C_PRE)
        ax.plot(t_bins, medians, color=C_PRE, lw=1.5)
        ax.axvline(0, color="black", lw=0.8, linestyle="--", label="Bolus")
        ax.axvline(PRE_START_S / 60.0, color="grey", lw=0.6, linestyle=":")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Time (min)", fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle("Q2b v2 — Feature trajectories ±5 min around vasopresor bolus",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    out = FIG / "q2v2_pre_vs_post_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG 2] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Q2a forest plot
# ═══════════════════════════════════════════════════════════════════════════════

def fig_forest_Q2a_v2(results: pd.DataFrame, title_suffix: str = "v2"):
    """Forest plot for all tested features (primary + exploratory)."""
    df = results.copy()
    df = df.sort_values(["analysis", "feature"]).reset_index(drop=True)

    if df.empty or df["beta"].isna().all():
        print("[FIG 3] no results for forest plot")
        return

    n = len(df)
    fig, ax = plt.subplots(figsize=(8, 0.5 * n + 1.5))

    for i, row in df.iterrows():
        color = C_Q2V2 if row["analysis"] == "primary" else "#aaa"
        lw    = 1.8 if row["analysis"] == "primary" else 1.2
        if pd.isna(row["beta"]):
            continue
        ax.plot(row["beta"], i, "s", color=color, markersize=6, zorder=3)
        if not pd.isna(row["ic_lo"]) and not pd.isna(row["ic_hi"]):
            ax.plot([row["ic_lo"], row["ic_hi"]], [i, i], color=color, lw=lw, zorder=2)

        # Annotate p
        p_str = ""
        if row["analysis"] == "primary" and not pd.isna(row.get("p_bonferroni", np.nan)):
            p_val = row["p_bonferroni"]
            p_str = f" p_bonf={p_val:.3f}"
            if p_val < 0.050:
                p_str += " ✓"
        elif not pd.isna(row.get("p_2sided", np.nan)):
            p_str = f" p={row['p_2sided']:.3f}"

        label = FEATURE_LABELS.get(row["feature"], row["feature"])
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -0.1, i,
                f"{label}{p_str}", va="center", ha="right", fontsize=7)

    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [FEATURE_LABELS.get(r["feature"], r["feature"]) for _, r in df.iterrows()],
        fontsize=7,
    )
    ax.set_xlabel("β (pre − control)", fontsize=8)
    ax.set_title(f"Q2a {title_suffix} — Forest plot  (primary in crimson)", fontsize=9)

    patches = [mpatches.Patch(color=C_Q2V2, label="Primary (Bonf. α=0.010)"),
               mpatches.Patch(color="#aaa",   label="Exploratory")]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    plt.tight_layout()
    out = FIG / "q2v2_forest_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG 3] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – BRS methods comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig_brs_methods_comparison(feat: pd.DataFrame, results_q2a: pd.DataFrame):
    """
    Panel A: scatter brs_seq vs brs_alpha_lf (window-level, Spearman r).
    Panel B: forest plot of β/IC for brs_seq and brs_alpha_lf from Q2a.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Panel A ───────────────────────────────────────────────────────────────
    sub = feat[feat["brs_seq"].notna() & feat["brs_alpha_lf"].notna()].copy()

    if len(sub) >= 10:
        x = sub["brs_alpha_lf"].values
        y = sub["brs_seq"].values
        # Clip to remove extreme outliers for display
        x_clip = np.clip(x, *np.percentile(x, [1, 99]))
        y_clip = np.clip(y, *np.percentile(y, [1, 99]))

        rho, p_rho = spearmanr(x, y)
        ax_a.hexbin(x_clip, y_clip, gridsize=40, cmap="Blues", mincnt=1, linewidths=0.2)
        ax_a.set_xlabel("BRS-α LF (ms/mmHg equivalent)", fontsize=9)
        ax_a.set_ylabel("BRS-seq (ms/ms)", fontsize=9)
        ax_a.set_title(f"Panel A — BRS methods window correlation\n"
                       f"Spearman ρ = {rho:+.3f}  p = {p_rho:.3e}  n = {len(sub):,}",
                       fontsize=9)
    else:
        ax_a.text(0.5, 0.5, "Insufficient data\nfor scatter",
                  ha="center", va="center", fontsize=9, transform=ax_a.transAxes)
        ax_a.set_title("Panel A — BRS methods correlation", fontsize=9)

    # ── Panel B ───────────────────────────────────────────────────────────────
    keys_b = ["brs_seq__min", "brs_alpha_lf__min"]
    colors_b = {k: c for k, c in zip(keys_b, [C_Q2V2, "#888888"])}

    plotted = 0
    for i, key in enumerate(keys_b):
        r = results_q2a[results_q2a["feature"] == key]
        if r.empty or r.iloc[0]["beta"] != r.iloc[0]["beta"]:  # NaN check
            continue
        row = r.iloc[0]
        lbl  = FEATURE_LABELS.get(key, key)
        ax_b.plot(row["beta"], i, "s", color=colors_b[key], markersize=8, zorder=3)
        if not pd.isna(row["ic_lo"]) and not pd.isna(row["ic_hi"]):
            ax_b.plot([row["ic_lo"], row["ic_hi"]], [i, i], color=colors_b[key], lw=2.0)
        # p label
        p_str = ""
        if not pd.isna(row.get("p_bonferroni", np.nan)):
            p_str = f"  p_bonf={row['p_bonferroni']:.3f}"
        elif not pd.isna(row.get("p_2sided", np.nan)):
            p_str = f"  p={row['p_2sided']:.3f}"
        ax_b.text(row["beta"], i + 0.2, f"{lbl}{p_str}", ha="center",
                  fontsize=8, color=colors_b[key])
        plotted += 1

    ax_b.axvline(0, color="black", lw=0.8, linestyle="--")
    ax_b.set_yticks(range(len(keys_b)))
    ax_b.set_yticklabels([FEATURE_LABELS.get(k, k) for k in keys_b], fontsize=9)
    ax_b.set_xlabel("β (pre − control)", fontsize=9)
    ax_b.set_title("Panel B — BRS methods Q2a β comparison", fontsize=9)
    if plotted == 0:
        ax_b.text(0.5, 0.5, "No BRS results available",
                  ha="center", va="center", fontsize=9, transform=ax_b.transAxes)

    plt.suptitle("BRS Methods Comparison (sequence vs spectral)", fontsize=10)
    plt.tight_layout()
    out = FIG / "brs_methods_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG 4] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Q1 vs Q2v2 dissociation (KEY FIGURE)
# ═══════════════════════════════════════════════════════════════════════════════

def _dissociation_verdict(q1_beta, q2_beta, direction, p_q1_1s, p_q2_1s) -> str:
    """
    Classify feature direction and significance for dissociation labelling.
    direction = -1 means the pre-specified expected direction for Q2 is negative.
    For Q1 (pain): the expected direction for rigidification features is OPPOSITE
    (i.e., +1 for ptt_cv, ptt_std, pai = these should be LOWER with pain, i.e. response).
    
    Returns one of:
      "DISSOCIATION"  – significant Q1 and significant Q2 but in same expected direction
                         (no dissociation, or confirming)
      "Q2_NS"         – Q2 not significant (main finding if Q1 was significant)
      "Q1_NS_Q2_SIG"  – Q1 not significant, Q2 significant
      "BOTH_NS"       – neither significant
    """
    q1_sig = (not np.isnan(p_q1_1s)) and p_q1_1s < 0.050
    q2_sig = (not np.isnan(p_q2_1s)) and p_q2_1s < ALPHA_BONF

    if q1_sig and not q2_sig:
        return "Q2_NS (dissociation)"
    if q1_sig and q2_sig:
        return "BOTH_SIG"
    if not q1_sig and q2_sig:
        return "Q2_SIG_Q1_NS"
    return "BOTH_NS"


def fig_dissociation_v2(
    results_q2a: pd.DataFrame,
    q1_results: pd.DataFrame,    # from results/validation/Q1/test_results_v2.csv
    brs_q1_summary: pd.DataFrame,  # from compute_brs_seq_q1 return
):
    """
    6-feature dissociation figure.
    Features: ptt_cv mean/std, ptt_std max, pai_mean mean, brs_seq min, brs_alpha_lf min.
    Q1 = grey squares, Q2v2 = crimson circles.
    X axis = normalised β (standardised to SD of pre-window values).
    Verdict label per feature.
    """
    features_6 = [
        ("ptt_cv__mean",      "PTT-CV mean"),
        ("ptt_cv__std",       "PTT-CV std"),
        ("ptt_std__max",      "PTT-std max"),
        ("pai_mean__mean",    "PAI mean"),
        ("brs_seq__min",      "BRS-seq min"),
        ("brs_alpha_lf__min", "BRS-α LF min"),
    ]

    # Build Q2v2 beta/CI from results_q2a
    q2_map = {}
    for _, r in results_q2a.iterrows():
        q2_map[r["feature"]] = {
            "beta": r.get("beta", np.nan),
            "ic_lo": r.get("ic_lo", np.nan),
            "ic_hi": r.get("ic_hi", np.nan),
            "p_1sided": r.get("p_1sided", np.nan),
        }

    # Build Q1 beta/CI from q1_results (test_results_v2.csv)
    q1_map = {}
    if q1_results is not None and not q1_results.empty:
        for _, r in q1_results.iterrows():
            # Handle both formats: 'feature' already as 'fname__agg', or separate columns
            if "feature_name" in r.index and "agg" in r.index:
                key = f"{r['feature_name']}__{r['agg']}"
            elif "feature" in r.index and "__" in str(r.get("feature", "")):
                key = str(r["feature"])
            else:
                continue
            q1_map[key] = {
                "beta": r.get("beta", np.nan),
                "ic_lo": r.get("ic_lo", np.nan),
                "ic_hi": r.get("ic_hi", np.nan),
                "p_1sided": r.get("p_1sided", r.get("p_2sided", np.nan)),
            }

    # For BRS_seq Q1, use brs_q1_summary if available
    if brs_q1_summary is not None and not brs_q1_summary.empty and "beta" in brs_q1_summary.columns:
        row = brs_q1_summary.iloc[0]
        q1_map["brs_seq__min"] = {
            "beta": row.get("beta", np.nan),
            "ic_lo": row.get("ic_lo", np.nan),
            "ic_hi": row.get("ic_hi", np.nan),
            "p_1sided": row.get("p_1sided", np.nan),
        }

    n_feat = len(features_6)
    fig, axes = plt.subplots(1, n_feat, figsize=(2.5 * n_feat, 4.5), sharey=False)
    if n_feat == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, features_6):
        q2_d = q2_map.get(key, {})
        q1_d = q1_map.get(key, {})

        y_positions = [0.7, 0.3]  # Q1=0.7, Q2v2=0.3 on y-axis
        ax.set_ylim(0, 1)
        ax.axvline(0, color="black", lw=0.7, linestyle="--", alpha=0.7)

        # Q1 point
        if q1_d.get("beta", np.nan) == q1_d.get("beta", np.nan):  # not NaN
            beta_q1 = q1_d["beta"]
            ax.plot(beta_q1, y_positions[0], "s", color=C_Q1, markersize=9, zorder=3)
            if not pd.isna(q1_d.get("ic_lo", np.nan)):
                ax.plot([q1_d["ic_lo"], q1_d["ic_hi"]], [y_positions[0]] * 2,
                        color=C_Q1, lw=1.5, zorder=2)
        else:
            ax.text(0, y_positions[0], "Q1 N/A", ha="center", va="center",
                    fontsize=6, color=C_Q1)

        # Q2v2 point
        if q2_d.get("beta", np.nan) == q2_d.get("beta", np.nan):
            beta_q2 = q2_d["beta"]
            ax.plot(beta_q2, y_positions[1], "o", color=C_Q2V2, markersize=9, zorder=3)
            if not pd.isna(q2_d.get("ic_lo", np.nan)):
                ax.plot([q2_d["ic_lo"], q2_d["ic_hi"]], [y_positions[1]] * 2,
                        color=C_Q2V2, lw=1.5, zorder=2)
        else:
            ax.text(0, y_positions[1], "Q2 N/A", ha="center", va="center",
                    fontsize=6, color=C_Q2V2)

        ax.set_title(label, fontsize=8, pad=4)
        ax.set_xlabel("β (ms)", fontsize=7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(["Q1", "Q2v2"], fontsize=7)
        ax.tick_params(axis="x", labelsize=6)

        # Verdict label
        verdict = _dissociation_verdict(
            q1_d.get("beta", np.nan), q2_d.get("beta", np.nan),
            direction=-1,
            p_q1_1s=q1_d.get("p_1sided", np.nan),
            p_q2_1s=q2_d.get("p_1sided", np.nan),
        )
        color_v = {"Q2_NS (dissociation)": "#c0392b",
                   "BOTH_SIG": "#27ae60",
                   "Q2_SIG_Q1_NS": "#e67e22",
                   "BOTH_NS": "#888888"}.get(verdict, "#444")
        ax.text(0.5, -0.08, verdict.replace(" (dissociation)", "\n(dissociation)"),
                ha="center", va="top", fontsize=6, color=color_v,
                transform=ax.transAxes)

    legend_patches = [
        mpatches.Patch(color=C_Q1,   label="Q1 (dolor ortopédico)"),
        mpatches.Patch(color=C_Q2V2, label="Q2v2 (vasopresor AG)"),
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               fontsize=8, ncol=2, bbox_to_anchor=(0.5, 1.01))
    plt.suptitle("Q1 vs Q2v2 — Feature dissociation\n"
                 "(β direction/magnitude: pain homeostasis vs vascular rigidification)",
                 fontsize=9, y=1.10)
    plt.tight_layout()
    out = FIG / "q1_vs_q2v2_dissociation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG 5] {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper: all figures
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_figures(
    paired_q2a: pd.DataFrame,
    results_q2a: pd.DataFrame,
    events: pd.DataFrame,
    feat: pd.DataFrame,
    q1_results: pd.DataFrame,
    brs_q1_summary: pd.DataFrame,
):
    FIG.mkdir(parents=True, exist_ok=True)
    print("\n[PASO 6] Generating all figures …")
    fig_violins_Q2a_v2(paired_q2a, results_q2a)
    fig_trajectories_Q2b_v2(events, feat)
    fig_forest_Q2a_v2(results_q2a)
    fig_brs_methods_comparison(feat, results_q2a)
    fig_dissociation_v2(results_q2a, q1_results, brs_q1_summary)
    print("[PASO 6] All figures saved.")
