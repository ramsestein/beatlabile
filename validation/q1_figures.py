"""
q1_figures.py
=============
PASO 6 — Figuras para el informe Q1.

Genera:
  1. feature_distributions_by_event.png  — boxplots control vs evento
  2. per_patient_event_response.png      — líneas por paciente
  3. group_gradient_plot.png             — gradiente por grupo
  4. forest_plot_directional_tests.png   — forest plot 7 tests primarios
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from q1_config import (
    ALPHA_BONFERRONI,
    ALPHA_NOMINAL,
    FIGURES_DIR,
    PRIMARY_FEATURES,
)

log = logging.getLogger(__name__)

# Paleta de colores consistente
CONTROL_COLOR = "#4C8BC2"
EVENT_COLOR   = "#D94F3D"
INT_COLOR     = "#2CA02C"
SUP_COLOR     = "#FF7F0E"

FEAT_LABELS = {
    "ptt_std__std":      "PTT-std (std)\n[std-PA-std]",
    "ptt_cv__std":       "PTT-CV (std)\n[cv-PA-std]",
    "brs_alpha_lf__min": "BRS-α LF (min)\n[brs-min]",
    "ptt_cv__mean":      "PTT-CV (mean)\n[cv-PA-mean]",
    "ptt_arv__std":      "PTT-ARV (std)\n[arv-std]",
    "ptt_std__slope":    "PTT-std (slope)\n[std-PA-slope]",
    "ptt_std__max":      "PTT-std (max)\n[std-PA-max]",
}


# ---------------------------------------------------------------------------
# 1. Boxplots control vs evento
# ---------------------------------------------------------------------------

def plot_feature_distributions(event_windows: pd.DataFrame) -> Path:
    """Boxplots side-by-side para las 7 features primarias."""
    cols = [f"{f}__{a}" for f, a, _ in PRIMARY_FEATURES]
    n_feat = len(cols)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    # Colores por paciente (para jitter)
    pids = sorted(event_windows["patient_id"].unique())
    cmap = plt.cm.get_cmap("tab20", len(pids))
    pid_color = {pid: cmap(i) for i, pid in enumerate(pids)}

    for ax_idx, col in enumerate(cols):
        ax = axes[ax_idx]
        if col not in event_windows.columns:
            ax.set_visible(False)
            continue

        for pos, wtype, color in [(0, "control", CONTROL_COLOR), (1, "event", EVENT_COLOR)]:
            sub = event_windows[event_windows["window_type"] == wtype][col].dropna()
            ax.boxplot(sub, positions=[pos], widths=0.35,
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.5),
                       medianprops=dict(color="black", linewidth=2),
                       flierprops=dict(marker="", linestyle="none"),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))

        # Jitter por paciente
        for pos, wtype in [(0, "control"), (1, "event")]:
            sub = event_windows[event_windows["window_type"] == wtype][[col, "patient_id"]].dropna()
            for pid, grp in sub.groupby("patient_id"):
                jitter = np.random.RandomState(42).uniform(-0.08, 0.08, len(grp))
                ax.scatter(np.full(len(grp), pos) + jitter, grp[col].values,
                           s=18, alpha=0.7, color=pid_color.get(pid, "gray"), zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Evento"], fontsize=9)
        ax.set_title(FEAT_LABELS.get(col, col), fontsize=8.5, pad=4)
        ax.set_ylabel("Valor", fontsize=8)
        ax.tick_params(labelsize=8)

    # Ocultar subplot extra
    for i in range(n_feat, len(axes)):
        axes[i].set_visible(False)

    # Leyenda de pacientes
    patches = [mpatches.Patch(color=pid_color[p], label=str(p)) for p in pids]
    fig.legend(handles=patches, title="Paciente", loc="lower right",
               fontsize=6, ncol=2, bbox_to_anchor=(1.0, 0.02))

    fig.suptitle("Distribución de features: Control vs Evento", fontsize=13, y=1.01)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "feature_distributions_by_event.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Guardado: %s", out)
    return out


# ---------------------------------------------------------------------------
# 2. Líneas por paciente (mediana control vs mediana evento)
# ---------------------------------------------------------------------------

def plot_per_patient_response(event_windows: pd.DataFrame) -> Path:
    cols = [f"{f}__{a}" for f, a, _ in PRIMARY_FEATURES]
    n_feat = len(cols)
    pids = sorted(event_windows["patient_id"].unique())
    cmap = plt.cm.get_cmap("tab20", len(pids))
    pid_color = {pid: cmap(i) for i, pid in enumerate(pids)}

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    for ax_idx, col in enumerate(cols):
        ax = axes[ax_idx]
        if col not in event_windows.columns:
            ax.set_visible(False)
            continue

        for pid in pids:
            sub = event_windows[event_windows["patient_id"] == pid]
            ctrl_med = sub[sub["window_type"] == "control"][col].median()
            ev_med   = sub[sub["window_type"] == "event"][col].median()
            if pd.isna(ctrl_med) or pd.isna(ev_med):
                continue
            ax.plot([0, 1], [ctrl_med, ev_med], "o-",
                    color=pid_color[pid], alpha=0.75, lw=1.5, ms=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control\n(mediana)", "Evento\n(mediana)"], fontsize=9)
        ax.set_title(FEAT_LABELS.get(col, col), fontsize=8.5, pad=4)
        ax.set_ylabel("Valor", fontsize=8)
        ax.tick_params(labelsize=8)

    for i in range(n_feat, len(axes)):
        axes[i].set_visible(False)

    patches = [mpatches.Patch(color=pid_color[p], label=str(p)) for p in pids]
    fig.legend(handles=patches, title="Paciente", loc="lower right",
               fontsize=6, ncol=2, bbox_to_anchor=(1.0, 0.02))

    fig.suptitle("Respuesta por paciente: mediana Control vs Evento", fontsize=13, y=1.01)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "per_patient_event_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Guardado: %s", out)
    return out


# ---------------------------------------------------------------------------
# 3. Gradiente por grupo
# ---------------------------------------------------------------------------

def plot_group_gradient(event_windows: pd.DataFrame,
                         validated_cols: list[str]) -> Optional[Path]:
    """Tres puntos: control, evento supra+axilar, evento interescalénico."""
    if not validated_cols:
        log.warning("No hay features validadas para gradiente plot")
        return None

    cols = [c for c in validated_cols if c in event_windows.columns]
    if not cols:
        return None

    n_cols = len(cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        # Control (todos)
        ctrl = event_windows[event_windows["window_type"] == "control"][col].dropna()
        # Evento por grupo
        ev_int = event_windows[
            (event_windows["window_type"] == "event") &
            (event_windows["group"] == "interescalenico")
        ][col].dropna()
        ev_sup = event_windows[
            (event_windows["window_type"] == "event") &
            (event_windows["group"] == "supra_axilar")
        ][col].dropna()

        groups = [("Control", ctrl, CONTROL_COLOR),
                  ("Evento\nSupra+axilar", ev_sup, SUP_COLOR),
                  ("Evento\nInterescalénico", ev_int, INT_COLOR)]

        for x_pos, (label, vals, color) in enumerate(groups):
            if len(vals) == 0:
                continue
            m  = vals.mean()
            ci = 1.96 * vals.sem() if len(vals) > 1 else 0.0
            ax.errorbar(x_pos, m, yerr=ci, fmt="o", color=color,
                        ms=9, capsize=5, capthick=2, linewidth=2)

        ax.plot([0, 1, 2],
                [
                    ctrl.mean() if len(ctrl) > 0 else np.nan,
                    ev_sup.mean() if len(ev_sup) > 0 else np.nan,
                    ev_int.mean() if len(ev_int) > 0 else np.nan,
                ],
                "k--", lw=1, alpha=0.4)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Control", "Evento\nSupra+axilar", "Evento\nInterescalénico"],
                           fontsize=9)
        ax.set_title(FEAT_LABELS.get(col, col), fontsize=9)
        ax.set_ylabel("Media ± IC95%", fontsize=9)

    fig.suptitle("Gradiente por grupo de bloqueo", fontsize=13)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "group_gradient_plot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Guardado: %s", out)
    return out


# ---------------------------------------------------------------------------
# 4. Forest plot de 7 tests primarios
# ---------------------------------------------------------------------------

def plot_forest(test_results_df: pd.DataFrame) -> Path:
    """Forest plot con β estandarizada e IC 95% para los 7 tests primarios."""
    primary = test_results_df[test_results_df["analysis"] == "primary"].copy()
    if primary.empty:
        log.warning("No hay tests primarios para forest plot")
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / "forest_plot_directional_tests.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # Estandarizar β (dividir por pooled SD de la feature)
    primary = primary.sort_values("beta", ascending=True).reset_index(drop=True)
    n = len(primary)

    # Normalizar: beta_std = beta / max(|ci_hi - ci_lo|)
    span = primary["ci_hi"] - primary["ci_lo"]
    primary["beta_std"] = primary["beta"] / span.replace(0, np.nan)
    primary["ci_lo_std"] = primary["ci_lo"] / span.replace(0, np.nan)
    primary["ci_hi_std"] = primary["ci_hi"] / span.replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5 + n * 0.5))

    y_positions = np.arange(n, dtype=float)

    for i, row in primary.iterrows():
        y = y_positions[i]
        beta = row["beta_std"]
        lo   = row["ci_lo_std"]
        hi   = row["ci_hi_std"]
        verdict = row["verdict"]

        color = {
            "validado":       EVENT_COLOR,
            "no_validado":    CONTROL_COLOR,
            "inconsistente":  "darkorange",
            "indeterminado":  "gray",
        }.get(verdict, "gray")

        if not (np.isnan(lo) or np.isnan(hi)):
            ax.plot([lo, hi], [y, y], "-", color=color, lw=2.5, solid_capstyle="round")
        if not np.isnan(beta):
            ax.scatter([beta], [y], color=color, s=80, zorder=5)

        label = FEAT_LABELS.get(row["col_name"], row["col_name"])
        ax.text(-0.05, y, label, ha="right", va="center", fontsize=8.5,
                transform=ax.get_yaxis_transform())

        # p-valor como texto
        p_str = f"p={row['p_bonferroni']:.4f}" if not np.isnan(row['p_bonferroni']) else "p=N/A"
        ax.text(1.02, y, p_str, ha="left", va="center", fontsize=8,
                transform=ax.get_yaxis_transform())

    # Línea en 0
    ax.axvline(0, color="black", lw=1, ls="-")

    # Banda Bonferroni (ninguna recta clara, simplemente anotar)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * n)
    ax.set_xlabel("β estandarizada (IC 95%)", fontsize=10)
    ax.set_title(
        f"Forest plot — Tests primarios confirmatorios\n"
        f"(Bonferroni α = {ALPHA_BONFERRONI:.5f}; rojo = validado)",
        fontsize=11,
    )

    # Leyenda de veredictos
    patches = [
        mpatches.Patch(color=EVENT_COLOR,   label="Validado"),
        mpatches.Patch(color=CONTROL_COLOR, label="No validado"),
        mpatches.Patch(color="darkorange",  label="Inconsistente"),
        mpatches.Patch(color="gray",        label="Indeterminado"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "forest_plot_directional_tests.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Guardado: %s", out)
    return out


# ---------------------------------------------------------------------------
# Función orquestadora
# ---------------------------------------------------------------------------

def run_paso6(event_windows: pd.DataFrame, test_results_df: pd.DataFrame) -> None:
    """Genera todas las figuras del PASO 6."""
    log.info("PASO 6 — Generando figuras")
    np.random.seed(42)

    validated_cols = []
    if not test_results_df.empty:
        validated_cols = list(
            test_results_df[
                (test_results_df["analysis"] == "primary") &
                (test_results_df["verdict"] == "validado")
            ]["col_name"]
        )

    if not event_windows.empty:
        plot_feature_distributions(event_windows)
        plot_per_patient_response(event_windows)
        if validated_cols:
            plot_group_gradient(event_windows, validated_cols)

    if not test_results_df.empty:
        plot_forest(test_results_df)

    log.info("PASO 6 completado")
