"""Generate Figure 5 — MILP optimal decision trees (hypotension + hypertension).

Produces:
  results/figures/fig5_milp_trees.pdf   (publication-quality, 2-panel)
  results/figures/fig5_milp_trees.png   (300 dpi raster)
  results/figures/fig5a_milp_hypotension.pdf  (standalone panels)
  results/figures/fig5b_milp_hypertension.pdf

Run
---
python experiments/fig_milp_tree.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from beatlabile.config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ACT1_DIR = RESULTS_DIR / "act1"

# ── Colour palette ────────────────────────────────────────────────────────────
C_LOW   = "#2196F3"   # blue  — low risk leaf
C_HIGH  = "#F44336"   # red   — high risk leaf
C_NODE  = "#ECEFF1"   # light grey — decision node
C_EDGE  = "#546E7A"   # slate — edges
C_TITLE = "#263238"   # dark  — title text
FONT    = "DejaVu Sans"


# ── Feature label map ─────────────────────────────────────────────────────────
FEAT_LABELS = {
    "cv_pa_min":   "CV(PA)_min",
    "cv_pa_mean":  "CV(PA)_mean",
    "cv_pa_std":   "CV(PA)_std",
    "std_pa_mean": "SD(PA)_mean\n(mmHg)",
    "std_pa_max":  "SD(PA)_max\n(mmHg)",
    "std_pa_std":  "SD(PA)_std",
    "brs_min":     "BRS_min\n(ms/mmHg)",
    "rsa_max":     "RSA_max",
    "arv_std":     "ARV_std",
}

UNIT_LABELS = {
    "cv_pa_min":   "adim.",
    "cv_pa_mean":  "adim.",
    "std_pa_mean": "mmHg",
    "std_pa_max":  "mmHg",
}


def _feat_label(name: str) -> str:
    return FEAT_LABELS.get(name, name)


def _thr_str(feat: str, thr: float) -> str:
    if "std_pa" in feat:
        return f"≤ {thr:.2f} mmHg"
    if "cv_pa" in feat:
        return f"≤ {thr:.4f}"
    return f"≤ {thr:.4f}"


# ── Tree traversal → flat node list ──────────────────────────────────────────
def _collect_nodes(node, x=0.5, y=0.92, dx=0.22, depth=0, nodes=None, edges=None):
    """Convert TreeNode hierarchy to flat position lists."""
    if nodes is None:
        nodes = []
    if edges is None:
        edges = []

    feat  = getattr(node, "feature", None)
    thr   = getattr(node, "threshold", None)
    pred  = getattr(node, "prediction", None)
    left  = getattr(node, "left", None)
    right = getattr(node, "right", None)

    is_leaf = feat is None
    node_id = len(nodes)
    nodes.append(dict(id=node_id, x=x, y=y, feat=feat, thr=thr, pred=pred,
                      is_leaf=is_leaf, depth=depth))

    if not is_leaf:
        child_dy = 0.26
        # Left child (condition TRUE: ≤ threshold)
        lx = x - dx
        ly = y - child_dy
        left_id = len(nodes)
        _collect_nodes(left, lx, ly, dx / 2, depth + 1, nodes, edges)
        edges.append(dict(parent=node_id, child=left_id, label="YES  (≤)"))

        # Right child (condition FALSE: > threshold)
        rx = x + dx
        ry = y - child_dy
        right_id = len(nodes)
        _collect_nodes(right, rx, ry, dx / 2, depth + 1, nodes, edges)
        edges.append(dict(parent=node_id, child=right_id, label="NO  (>)"))

    return nodes, edges


def _draw_tree(ax, milp, title: str, outcome_label: str,
               train_auc: float, boot_auc: float, n_events: int) -> None:
    """Draw a single MILP tree onto ax."""
    nodes, edges = _collect_nodes(milp.root)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.32, 1.06)
    ax.axis("off")

    # ── Draw edges first ──
    for e in edges:
        px, py = nodes[e["parent"]]["x"], nodes[e["parent"]]["y"]
        cx, cy = nodes[e["child"]]["x"],  nodes[e["child"]]["y"]
        ax.annotate("",
                    xy=(cx, cy + 0.035), xytext=(px, py - 0.035),
                    arrowprops=dict(arrowstyle="-|>", color=C_EDGE,
                                   lw=1.4, mutation_scale=14))
        mx = (px + cx) / 2
        my = (py + cy) / 2 + 0.005
        is_yes = e["label"].startswith("YES")
        ax.text(mx, my, e["label"],
                ha="center", va="center", fontsize=7.5,
                color=C_EDGE, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    # ── Draw nodes ──
    for n in nodes:
        x, y = n["x"], n["y"]
        if n["is_leaf"]:
            risk = n["pred"]
            color = C_HIGH if risk > 0.5 else C_LOW
            label = "HIGH\nRISK" if risk > 0.5 else "LOW\nRISK"
            sub   = f"P ≈ {risk:.0f}"
            box = FancyBboxPatch((x - 0.09, y - 0.038), 0.18, 0.076,
                                 boxstyle="round,pad=0.01",
                                 fc=color, ec="white", lw=1.5, alpha=0.92,
                                 transform=ax.transData, zorder=3)
            ax.add_patch(box)
            ax.text(x, y + 0.012, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=4)
            ax.text(x, y - 0.018, sub, ha="center", va="center",
                    fontsize=7.5, color="white", alpha=0.9, zorder=4)
        else:
            feat_str = _feat_label(n["feat"])
            thr_str  = _thr_str(n["feat"], n["thr"])
            box = FancyBboxPatch((x - 0.13, y - 0.042), 0.26, 0.084,
                                 boxstyle="round,pad=0.01",
                                 fc=C_NODE, ec=C_EDGE, lw=1.4, alpha=0.95,
                                 transform=ax.transData, zorder=3)
            ax.add_patch(box)
            ax.text(x, y + 0.014, feat_str, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=C_TITLE, zorder=4)
            ax.text(x, y - 0.016, thr_str, ha="center", va="center",
                    fontsize=8, color="#37474F", zorder=4)

    # ── Title + stats banner ──
    banner = (f"n eventos = {n_events}  │  AUC train = {train_auc:.3f}"
              f"  │  AUC bootstrap (B=50) = {boot_auc:.3f}")
    ax.set_title(f"{title}\n{banner}",
                 fontsize=10.5, fontweight="bold", color=C_TITLE, pad=4,
                 linespacing=1.8)

    # ── Legend ──
    ax.legend(handles=[
        mpatches.Patch(fc=C_HIGH, label="High risk"),
        mpatches.Patch(fc=C_LOW,  label="Low risk"),
    ], loc="lower center", ncol=2, fontsize=8, framealpha=0.7,
       bbox_to_anchor=(0.5, 0.0))


def run_fig5() -> None:
    import json, numpy as np

    with open(ACT1_DIR / "act1_results.json") as fh:
        r1 = json.load(fh)

    event_config = {
        "hypotension": {
            "title": "A  Hypotension — Optimal MILP tree",
            "n_events": r1["hypotension"]["n_events"],
            "train_auc": r1["hypotension"]["milp_train_auc"],
            "boot_auc":  r1["hypotension"]["milp_bootstrap_auc_mean"],
        },
        "hypertension": {
            "title": "B  Hypertension — Optimal MILP tree",
            "n_events": r1["hypertension"]["n_events"],
            "train_auc": r1["hypertension"]["milp_train_auc"],
            "boot_auc":  r1["hypertension"]["milp_bootstrap_auc_mean"],
        },
    }

    # ── Combined 2-panel figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")

    for ax, etype in zip(axes, ["hypotension", "hypertension"]):
        with open(ACT1_DIR / f"milp_{etype}.pkl", "rb") as fh:
            milp = pickle.load(fh)
        cfg = event_config[etype]
        _draw_tree(ax, milp, cfg["title"], etype,
                   cfg["train_auc"], cfg["boot_auc"], cfg["n_events"])

    plt.suptitle(
        "Fig. 5 — MILP decision rules: beat-by-beat BP autonomic metrics",
        fontsize=13, fontweight="bold", color=C_TITLE, y=1.01,
    )
    plt.tight_layout(pad=1.2)

    for suffix in [".pdf", ".png"]:
        out = OUT_DIR / f"fig5_milp_trees{suffix}"
        dpi = 300 if suffix == ".png" else None
        plt.savefig(out, bbox_inches="tight", dpi=dpi)
        print(f"Saved: {out}")

    plt.close(fig)

    # ── Standalone panels ────────────────────────────────────────────────────
    for etype, panel in [("hypotension", "a"), ("hypertension", "b")]:
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5.5))
        fig2.patch.set_facecolor("white")
        with open(ACT1_DIR / f"milp_{etype}.pkl", "rb") as fh:
            milp = pickle.load(fh)
        cfg = event_config[etype]
        _draw_tree(ax2, milp, cfg["title"], etype,
                   cfg["train_auc"], cfg["boot_auc"], cfg["n_events"])
        plt.tight_layout(pad=1.2)
        out = OUT_DIR / f"fig5{panel}_milp_{etype}.pdf"
        plt.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig2)

    print("Fig 5 complete.")


if __name__ == "__main__":
    run_fig5()
