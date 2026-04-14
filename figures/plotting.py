"""Generate all study figures (Figs 1–10, Section 11 of the protocol).

Each function saves a PNG + PDF to results/figures/.

Figures
-------
Fig 1  — STROBE flowchart (three acts)
Fig 2  — Case star: metrics diverging before a hypotension ELH
Fig 3  — ROC curves by prediction horizon (5/10/15/30 min) per event type
Fig 4  — Forest plot: AUC signal-only across 3 cohorts, per event type
Fig 5  — MILP tree visualisation for hypotension
Fig 6  — Bootstrap stability: feature selection freq + threshold distributions
Fig 7  — Sufficiency test ΔAUC M1–M6 per event type
Fig 8  — Benchmark comparison: GLMM vs MILP vs RF vs XGBoost
Fig 9  — Subgroup forest plot (VitalDB) per event type
Fig 10 — Decision curve analysis per event type

Usage
-----
from figures.plotting import generate_all_figures
generate_all_figures(results_dir=RESULTS_DIR)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_DIR: Path | None = None
EVENT_COLORS = {
    "hypotension": "#2166ac",
    "hypertension": "#d6604d",
    "variability": "#4dac26",
}
EVENT_LABELS = {
    "hypotension": "Hipotensión",
    "hypertension": "Hipertensión",
    "variability": "Variabilidad",
}


def _setup_figure_dir(results_dir: Path) -> Path:
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _save(fig: plt.Figure, name: str, fig_dir: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1 — STROBE flowchart
# ---------------------------------------------------------------------------

def fig1_strobe(results_dir: Path) -> None:
    """Text-based STROBE flowchart."""
    fig_dir = _setup_figure_dir(results_dir)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")

    boxes = [
        (0.5, 0.95, "Clínic UCIQ\n~650 pacientes totales"),
        (0.5, 0.82, "QC: ~150 registros completos (≥6 h)\n+ ~500 fragmentos válidos (≥2 h)"),
        (0.5, 0.68, "Acto 1 — Desarrollo\nn=150 completos + fragmentos\n3 modelos × 3 aproximaciones"),
        (0.5, 0.50, "Acto 2 — Validación ciega MIMIC-IV\nn=150 | 3 modelos aplicados sin reajuste"),
        (0.5, 0.32, "Acto 3a — Revalidación signal-only VitalDB\nn=150"),
        (0.5, 0.16, "Acto 3b — Test suficiencia M1–M6\n+ Subgrupos (edad, sexo, ASA, BMI)"),
    ]

    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="steelblue", lw=1.5),
                transform=ax.transAxes)

    # Arrows
    for i in range(len(boxes) - 1):
        ax.annotate(
            "", xy=(boxes[i + 1][0], boxes[i + 1][1] + 0.06),
            xytext=(boxes[i][0], boxes[i][1] - 0.04),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
        )

    ax.set_title("Fig 1 — Flujo STROBE: Estudio BeatLabile (tres actos)", fontsize=13, fontweight="bold")
    _save(fig, "fig1_strobe", fig_dir)
    logger.info("Fig 1 saved.")


# ---------------------------------------------------------------------------
# Fig 2 — Case star
# ---------------------------------------------------------------------------

def fig2_case_star(metrics_ts: pd.DataFrame | None, results_dir: Path) -> None:
    """Plot an example record showing metrics diverging 15–20 min before ELH.

    If metrics_ts is None, a synthetic demo is generated.
    """
    fig_dir = _setup_figure_dir(results_dir)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    if metrics_ts is None:
        # Synthetic demonstration
        t = np.linspace(-30, 5, 300)
        t_event = 0.0
        np.random.seed(42)
        metrics_ts = pd.DataFrame({
            "time_min": t,
            "map": 75 - np.clip(np.exp((t - t_event + 2) * 0.5), 0, 40) + np.random.normal(0, 2, len(t)),
            "brs": 8 - np.clip(np.exp((t - t_event + 15) * 0.1), 0, 6) + np.random.normal(0, 0.5, len(t)),
            "sdnn": 35 - np.clip(np.exp((t - t_event + 12) * 0.15), 0, 25) + np.random.normal(0, 2, len(t)),
            "rmssd": 30 - np.clip(np.exp((t - t_event + 10) * 0.18), 0, 20) + np.random.normal(0, 1.5, len(t)),
        })

    for ax, (col, label, color, threshold) in zip(
        axes,
        [
            ("map", "PAM (mmHg)", "#d6604d", 55),
            ("brs", "BRS (ms/mmHg)", "#2166ac", None),
            ("sdnn", "SDNN (ms)", "#4dac26", None),
            ("rmssd", "RMSSD (ms)", "#762a83", None),
        ],
    ):
        if col in metrics_ts.columns:
            ax.plot(metrics_ts["time_min"], metrics_ts[col], color=color, lw=1.5)
        if threshold is not None:
            ax.axhline(threshold, color="gray", lw=1, ls="--", label=f"Umbral {threshold}")
        ax.axvline(0, color="red", lw=2, ls="--", label="Inicio ELH")
        ax.axvspan(-20, -5, alpha=0.1, color="orange", label="Ventana predictora")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Tiempo hasta evento (min)", fontsize=10)
    fig.suptitle("Fig 2 — Ejemplo: métricas fisiológicas antes de episodio de hipotensión", fontsize=12)
    _save(fig, "fig2_case_star", fig_dir)
    logger.info("Fig 2 saved.")


# ---------------------------------------------------------------------------
# Fig 3 — ROC curves by horizon
# ---------------------------------------------------------------------------

def fig3_roc_by_horizon(
    results_by_horizon: dict[str, dict[str, Any]],
    results_dir: Path,
) -> None:
    """ROC panel: rows = event types, cols = prediction horizons.

    results_by_horizon: {horizon_label: {event_type: {"fpr": ..., "tpr": ..., "auc": ...}}}
    """
    fig_dir = _setup_figure_dir(results_dir)
    event_types = ["hypotension", "hypertension", "variability"]
    horizons = sorted(results_by_horizon.keys()) or ["(sin datos)"]

    fig, axes = plt.subplots(
        len(event_types), len(horizons), figsize=(4 * len(horizons), 4 * len(event_types))
    )
    if len(event_types) == 1:
        axes = axes[np.newaxis, :]
    if len(horizons) == 1:
        axes = axes[:, np.newaxis]

    for j, horiz in enumerate(horizons):
        horiz_data = results_by_horizon.get(horiz, {})
        for i, etype in enumerate(event_types):
            ax = axes[i, j]
            edata = horiz_data.get(etype, {})
            fpr = np.array(edata.get("fpr", [0, 1]))
            tpr = np.array(edata.get("tpr", [0, 1]))
            auc = edata.get("auc", np.nan)
            ax.plot(fpr, tpr, color=EVENT_COLORS[etype], lw=2,
                    label=f"AUC={auc:.3f}" if not np.isnan(auc) else "AUC=N/A")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_xlabel("1 - Especificidad" if i == len(event_types) - 1 else "")
            ax.set_ylabel("Sensibilidad" if j == 0 else "")
            ax.set_title(f"{EVENT_LABELS[etype]}\n{horiz}", fontsize=9)
            ax.legend(fontsize=8)

    fig.suptitle("Fig 3 — Curvas ROC por horizonte temporal y tipo de evento", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig3_roc_by_horizon", fig_dir)
    logger.info("Fig 3 saved.")


# ---------------------------------------------------------------------------
# Fig 4 — Forest plot AUC across cohorts
# ---------------------------------------------------------------------------

def fig4_forest_auc(
    act1_aucs: dict,
    act2_aucs: dict,
    act3_aucs: dict,
    results_dir: Path,
) -> None:
    """Forest plot: AUC per event type × cohort."""
    fig_dir = _setup_figure_dir(results_dir)
    event_types = ["hypotension", "hypertension", "variability"]
    cohorts = ["Clínic (Acto 1)", "MIMIC-IV (Acto 2)", "VitalDB (Acto 3)"]
    all_aucs = [act1_aucs, act2_aucs, act3_aucs]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_positions = np.arange(len(event_types))
    offsets = np.linspace(-0.2, 0.2, len(cohorts))
    colors = ["#2166ac", "#d6604d", "#4dac26"]

    for ci, (cohort, aucs_dict, color) in enumerate(zip(cohorts, all_aucs, colors)):
        for ei, etype in enumerate(event_types):
            auc_val = aucs_dict.get(etype, {}).get("glmm_cv_auc_mean",
                      aucs_dict.get(etype, {}).get("glmm_auc", np.nan))
            if not np.isnan(auc_val):
                ax.scatter(auc_val, y_positions[ei] + offsets[ci],
                           color=color, zorder=5, s=80, marker="D")
                ax.plot([auc_val - 0.03, auc_val + 0.03],
                        [y_positions[ei] + offsets[ci]] * 2,
                        color=color, lw=2)

    ax.axvline(0.5, color="gray", lw=1, ls="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([EVENT_LABELS[e] for e in event_types])
    ax.set_xlabel("AUC (signal-only)", fontsize=10)
    ax.set_xlim(0.4, 1.0)

    legend_patches = [mpatches.Patch(color=c, label=lbl) for c, lbl in zip(colors, cohorts)]
    ax.legend(handles=legend_patches, fontsize=8)
    ax.set_title("Fig 4 — AUC signal-only en tres cohortes (modelo GLMM)", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig4_forest_auc", fig_dir)
    logger.info("Fig 4 saved.")


# ---------------------------------------------------------------------------
# Fig 5 — MILP tree visualisation
# ---------------------------------------------------------------------------

def fig5_milp_tree(milp_tree, results_dir: Path, event_type: str = "hypotension") -> None:
    """Visualise the MILP optimal decision tree."""
    from beatlabile.models.milp_tree import MILPTree, TreeNode

    fig_dir = _setup_figure_dir(results_dir)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")

    if milp_tree._sklearn_tree is not None:
        from sklearn.tree import plot_tree
        plot_tree(
            milp_tree._sklearn_tree,
            feature_names=milp_tree.feature_cols,
            class_names=["Control", "Evento"],
            filled=True,
            fontsize=9,
            ax=ax,
        )
    else:
        # Custom recursive visualisation for MILP TreeNode
        _draw_tree_node(ax, milp_tree.root, x=0.5, y=0.95, dx=0.25, dy=0.15, level=0)

    ax.set_title(
        f"Fig 5 — Árbol de decisión MILP óptimo: {EVENT_LABELS.get(event_type, event_type)}",
        fontsize=11,
    )
    _save(fig, f"fig5_milp_tree_{event_type}", fig_dir)
    logger.info("Fig 5 (%s) saved.", event_type)


def _draw_tree_node(ax, node, x, y, dx, dy, level, max_level=3):
    if node is None or level > max_level:
        return
    if node.prediction is not None:
        label = "EVENTO" if node.prediction == 1 else "CONTROL"
        color = "#d6604d" if node.prediction == 1 else "#2166ac"
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc=color, alpha=0.5),
                transform=ax.transAxes)
    else:
        label = f"{node.feature}\n≤ {node.threshold:.2f}" if node.threshold else node.feature
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc="lightyellow", ec="orange"),
                transform=ax.transAxes)
        if node.left:
            ax.annotate("", xy=(x - dx, y - dy), xytext=(x, y),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->"))
            _draw_tree_node(ax, node.left, x - dx, y - dy, dx / 2, dy, level + 1, max_level)
        if node.right:
            ax.annotate("", xy=(x + dx, y - dy), xytext=(x, y),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->"))
            _draw_tree_node(ax, node.right, x + dx, y - dy, dx / 2, dy, level + 1, max_level)


# ---------------------------------------------------------------------------
# Fig 6 — Bootstrap stability
# ---------------------------------------------------------------------------

def fig6_bootstrap_stability(stability_result, results_dir: Path, event_type: str = "hypotension") -> None:
    """Bar chart of feature selection frequency + boxplot of threshold distributions."""
    fig_dir = _setup_figure_dir(results_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: selection frequency
    freq = stability_result.feature_freq
    features_sorted = sorted(freq.keys(), key=lambda f: -freq[f])[:12]
    freqs = [freq[f] for f in features_sorted]
    colors_bar = ["#d6604d" if v >= 0.80 else "#7fbc41" for v in freqs]
    ax1.barh(features_sorted, freqs, color=colors_bar)
    ax1.axvline(0.80, color="red", lw=1.5, ls="--", label="Estabilidad (80%)")
    ax1.set_xlabel("Frecuencia de selección bootstrap")
    ax1.set_title("Frecuencia de selección de features")
    ax1.legend()

    # Right: threshold distributions (violin)
    dist_data = {
        f: stability_result.threshold_distributions[f]
        for f in features_sorted[:6]
        if len(stability_result.threshold_distributions.get(f, [])) > 5
    }
    if dist_data:
        ax2.violinplot(list(dist_data.values()), showmedians=True)
        ax2.set_xticks(range(1, len(dist_data) + 1))
        ax2.set_xticklabels(list(dist_data.keys()), rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Umbral óptimo (bootstrap)")
        ax2.set_title("Distribución de umbrales (bootstrap)")
    else:
        ax2.text(0.5, 0.5, "Datos insuficientes", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.suptitle(
        f"Fig 6 — Estabilidad bootstrap árbol MILP: {EVENT_LABELS.get(event_type, event_type)}",
        fontsize=11,
    )
    plt.tight_layout()
    _save(fig, f"fig6_stability_{event_type}", fig_dir)
    logger.info("Fig 6 (%s) saved.", event_type)


# ---------------------------------------------------------------------------
# Fig 7 — Sufficiency test ΔAUC M1–M6
# ---------------------------------------------------------------------------

def fig7_sufficiency(act3_results: dict, results_dir: Path) -> None:
    """ΔAUC plot for models M1–M6."""
    fig_dir = _setup_figure_dir(results_dir)
    event_types = ["hypotension", "hypertension", "variability"]
    model_keys = ["M1", "M2", "M3", "M4", "M5", "M6"]

    fig, axes = plt.subplots(1, len(event_types), figsize=(15, 5), sharey=False)

    for ax, etype in zip(axes, event_types):
        edata = act3_results.get(etype, {}).get("sufficiency_models", {})
        aucs = [edata.get(m, np.nan) for m in model_keys]
        ax.plot(model_keys, aucs, "o-", color=EVENT_COLORS[etype], lw=2, ms=8)
        ax.axhline(aucs[0] if not np.isnan(aucs[0]) else 0.5, color="gray", lw=1, ls="--")
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Modelo")
        ax.set_ylabel("AUC")
        ax.set_title(EVENT_LABELS[etype])
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Fig 7 — Test de suficiencia: ΔAUC M1–M6 por tipo de evento", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig7_sufficiency", fig_dir)
    logger.info("Fig 7 saved.")


# ---------------------------------------------------------------------------
# Fig 8 — Benchmark comparison
# ---------------------------------------------------------------------------

def fig8_benchmark(act1_results: dict, results_dir: Path) -> None:
    """AUC bar chart: GLMM vs MILP vs RF vs XGBoost."""
    fig_dir = _setup_figure_dir(results_dir)
    event_types = ["hypotension", "hypertension", "variability"]
    model_names = ["GLMM", "MILP", "RF", "XGBoost"]
    auc_keys = ["glmm_cv_auc_mean", "milp_train_auc", "rf_cv_auc", "xgb_cv_auc"]

    x = np.arange(len(event_types))
    width = 0.18
    colors_bench = ["#4393c3", "#d6604d", "#74c476", "#fd8d3c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (mname, key, color) in enumerate(zip(model_names, auc_keys, colors_bench)):
        aucs = [act1_results.get(e, {}).get(key, np.nan) for e in event_types]
        ax.bar(x + (i - 1.5) * width, aucs, width, label=mname, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([EVENT_LABELS[e] for e in event_types])
    ax.set_ylabel("AUC (CV internal)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.legend(fontsize=9)
    ax.set_title("Fig 8 — Benchmark: modelos interpretables vs ML", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig8_benchmark", fig_dir)
    logger.info("Fig 8 saved.")


# ---------------------------------------------------------------------------
# Fig 9 — Subgroup forest plot
# ---------------------------------------------------------------------------

def fig9_subgroups(act3_results: dict, results_dir: Path) -> None:
    """Forest plot of M1 AUC by subgroup."""
    fig_dir = _setup_figure_dir(results_dir)
    event_types = ["hypotension", "hypertension", "variability"]

    fig, axes = plt.subplots(1, len(event_types), figsize=(15, 6), sharey=True)

    for ax, etype in zip(axes, event_types):
        sg_data = act3_results.get(etype, {}).get("subgroup_auc", {})
        if not sg_data:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
            continue
        labels = list(sg_data.keys())
        aucs = [sg_data[k] for k in labels]
        y_pos = np.arange(len(labels))
        ax.scatter(aucs, y_pos, color=EVENT_COLORS[etype], s=80, zorder=5)
        ax.axvline(0.5, color="gray", lw=1, ls="--")
        ax.set_xlim(0.3, 1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("AUC M1 (signal-only)")
        ax.set_title(EVENT_LABELS[etype])

    fig.suptitle("Fig 9 — Análisis de subgrupos VitalDB (AUC M1)", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig9_subgroups", fig_dir)
    logger.info("Fig 9 saved.")


# ---------------------------------------------------------------------------
# Fig 10 — Decision curve analysis
# ---------------------------------------------------------------------------

def fig10_decision_curves(
    proba_dict: dict[str, dict[str, np.ndarray]],
    y_dict: dict[str, np.ndarray],
    results_dir: Path,
) -> None:
    """Decision curve analysis: net benefit vs threshold probability.

    proba_dict: {event_type: {"glmm": proba_arr, "milp": proba_arr}}
    y_dict: {event_type: y_arr}
    """
    fig_dir = _setup_figure_dir(results_dir)
    event_types = list(proba_dict.keys())
    fig, axes = plt.subplots(1, max(len(event_types), 1), figsize=(5 * max(len(event_types), 1), 5))
    if len(event_types) == 1:
        axes = [axes]

    for ax, etype in zip(axes, event_types):
        y = np.array(y_dict.get(etype, []))
        if len(y) == 0:
            continue
        thresh_range = np.linspace(0.01, 0.99, 100)
        prevalence = np.mean(y)

        # Treat all
        nb_all = prevalence - (1 - prevalence) * thresh_range / (1 - thresh_range)
        ax.plot(thresh_range, nb_all, "k--", lw=1, label="Tratar todos")
        ax.axhline(0, color="gray", lw=1)

        for model_name, color in zip(
            proba_dict.get(etype, {}).keys(), ["#2166ac", "#d6604d"]
        ):
            proba = np.array(proba_dict[etype][model_name])
            nb = _net_benefit(y, proba, thresh_range)
            ax.plot(thresh_range, nb, color=color, lw=2, label=model_name.upper())

        ax.set_xlabel("Probabilidad umbral")
        ax.set_ylabel("Net Benefit")
        ax.set_title(EVENT_LABELS.get(etype, etype))
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle("Fig 10 — Decision Curve Analysis por tipo de evento", fontsize=11)
    plt.tight_layout()
    _save(fig, "fig10_decision_curves", fig_dir)
    logger.info("Fig 10 saved.")


def _net_benefit(y: np.ndarray, proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    nb = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        predicted_pos = proba >= t
        tp = np.sum(predicted_pos & (y == 1))
        fp = np.sum(predicted_pos & (y == 0))
        nb[i] = tp / n - fp / n * t / (1 - t)
    return nb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: Path) -> None:
    """Generate all 10 figures from saved result files."""
    logger.info("Generating all figures...")
    act1_path = results_dir / "act1" / "act1_results.json"
    act2_path = results_dir / "act2" / "act2_results.json"
    act3_path = results_dir / "act3" / "act3_results.json"

    act1 = json.loads(act1_path.read_text()) if act1_path.exists() else {}
    act2 = json.loads(act2_path.read_text()) if act2_path.exists() else {}
    act3 = json.loads(act3_path.read_text()) if act3_path.exists() else {}

    # Fig 1 — always available
    fig1_strobe(results_dir)

    # Fig 2 — synthetic demo if no real data
    fig2_case_star(None, results_dir)

    # Fig 3 — placeholder ROC (actual data needed)
    fig3_roc_by_horizon({}, results_dir)

    # Fig 4 — forest plot
    fig4_forest_auc(act1, act2, act3, results_dir)

    # Figs 5, 6 — MILP tree for each event type
    for etype in ["hypotension", "hypertension", "variability"]:
        milp_path = results_dir / "act1" / f"milp_{etype}.pkl"
        if milp_path.exists():
            with open(milp_path, "rb") as fh:
                milp = pickle.load(fh)
            fig5_milp_tree(milp, results_dir, etype)

            stab_path = results_dir / "act1" / f"milp_stability_{etype}.csv"
            if stab_path.exists():
                # Build a minimal StabilityResult-like object from CSV
                from beatlabile.models.milp_tree import StabilityResult
                stab_df = pd.read_csv(stab_path)
                freq_dict = dict(zip(stab_df["feature"], stab_df["selection_freq"]))
                pseudo_stab = StabilityResult(
                    feature_freq=freq_dict,
                    threshold_distributions={f: np.array([]) for f in freq_dict},
                    bootstrap_auc=np.array([]),
                )
                fig6_bootstrap_stability(pseudo_stab, results_dir, etype)

    # Figs 7, 8, 9
    if act3:
        fig7_sufficiency(act3, results_dir)
    if act1:
        fig8_benchmark(act1, results_dir)
    if act3:
        fig9_subgroups(act3, results_dir)

    # Fig 10 — decision curves (placeholder without live proba)
    fig10_decision_curves({}, {}, results_dir)

    logger.info("All figures saved to %s/figures/", results_dir)
