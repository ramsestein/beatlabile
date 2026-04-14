"""fig_roc_curves.py
Generate composite ROC curve figure (Fig 4) for BeatLabile manuscript.

Panels A (hypotension) and B (hypertension), each showing:
  - Clínic GLMM parsimonious — training apparent  (dashed, grey)
  - Clínic GLMM parsimonious — leave-one-patient-out CV  (solid, blue)
  - MIMIC M1    — direct transfer, external  (solid, green)
  - VitalDB M1  — direct transfer, external  (solid, orange)
  - VitalDB M2  — refit 70/30 hold-out       (solid, red)

AUC ± CI displayed in legend.  CI via DeLong (single ROC) or bootstrap (CV).

Outputs:
  results/figures/fig_roc_curves.pdf
  results/figures/fig_roc_curves.png
  results/figures/roc_data.csv   (raw FPR/TPR per curve for reproducibility)
"""

import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut

from beatlabile.models.mixed_logistic import MixedLogisticModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cache"
ACT1_DIR = ROOT / "results" / "act1"
ACT2_DIR = ROOT / "results" / "act2"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ETYPES = ["hypotension", "hypertension"]

PARSIMONIOUS_FEATURES: dict[str, list[str]] = {
    "hypotension": [
        "std_pa_mean", "cv_pa_std", "rsa_max", "arv_std",
        "brs_min", "arv_mean", "std_pa_max", "rsa_mean",
    ],
    "hypertension": [
        "std_pa_std", "std_pa_max", "cv_pa_std", "std_pa_slope",
        "arv_std", "cv_pa_mean", "brs_min", "sdnn_mean",
    ],
}

# Colours per curve
COLORS = {
    "clinic_apparent":  "#999999",   # grey dashed
    "clinic_lopocv":    "#1f77b4",   # blue
    "mimic_m1":         "#2ca02c",   # green
    "vitaldb_m1":       "#ff7f0e",   # orange
    "vitaldb_m2":       "#d62728",   # red
}

# ---------------------------------------------------------------------------
# Bootstrap DeLong-like CI for AUC
# ---------------------------------------------------------------------------

def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                     n_boot: int = 1000, seed: int = 42,
                     groups: np.ndarray | None = None) -> tuple[float, float]:
    """Bootstrap CI on AUC.  If groups provided, resample at group level."""
    rng = np.random.default_rng(seed)
    aucs = []
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if groups is not None:
        unique_g = np.unique(groups)
        for _ in range(n_boot):
            sampled_g = rng.choice(unique_g, size=len(unique_g), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in sampled_g])
            yt, ys = y_true[idx], y_score[idx]
            if len(np.unique(yt)) < 2:
                continue
            aucs.append(roc_auc_score(yt, ys))
    else:
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            yt, ys = y_true[idx], y_score[idx]
            if len(np.unique(yt)) < 2:
                continue
            aucs.append(roc_auc_score(yt, ys))

    if len(aucs) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_glmm(etype: str):
    """Load parsimonious GLMM pkl (beatlabile must be imported first)."""
    pkl = ACT1_DIR / f"glmm_parsimonious_{etype}.pkl"
    with open(pkl, "rb") as f:
        return pickle.load(f)


def load_windows(cohort: str) -> pd.DataFrame:
    p = CACHE / f"{cohort}_windows.parquet"
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# ROC computation per cohort / model
# ---------------------------------------------------------------------------

def roc_clinic_apparent(etype: str, windows: pd.DataFrame, glmm) -> dict:
    sub = windows[windows["event_type"] == etype].copy()
    sub["pred"] = glmm.predict_proba(sub)
    auc = roc_auc_score(sub["label"], sub["pred"])
    fpr, tpr, _ = roc_curve(sub["label"], sub["pred"])
    ci_lo, ci_hi = bootstrap_auc_ci(sub["label"].values, sub["pred"].values,
                                     groups=sub["patient_id"].values)
    logger.info("Clínic apparent  %s  AUC=%.4f [%.3f–%.3f]", etype, auc, ci_lo, ci_hi)
    return {"label": "Clínic (training apparent)", "key": "clinic_apparent",
            "fpr": fpr, "tpr": tpr, "auc": auc, "ci": (ci_lo, ci_hi), "ls": "--"}


def roc_clinic_lopocv(etype: str, windows: pd.DataFrame, glmm) -> dict | None:
    """Leave-one-patient-out CV AUC — uses feature cols from the fitted model."""
    feat_cols = PARSIMONIOUS_FEATURES[etype]
    sub = windows[windows["event_type"] == etype].copy()
    avail = [c for c in feat_cols if c in sub.columns]
    if len(avail) < len(feat_cols) // 2:
        return None

    pids = sub["patient_id"].values
    y_true_all, y_pred_all, groups_all = [], [], []
    logo = LeaveOneGroupOut()

    for tr_idx, te_idx in logo.split(sub, groups=pids):
        tr = sub.iloc[tr_idx]; te = sub.iloc[te_idx]
        if tr["label"].nunique() < 2:
            continue
        try:
            m = MixedLogisticModel(feature_cols=avail)
            m.fit(tr[avail], tr["label"].values, tr["patient_id"].values)
            preds = m.predict_proba(te[avail])
            y_true_all.extend(te["label"].values)
            y_pred_all.extend(preds)
            groups_all.extend(te["patient_id"].values)
        except Exception:
            continue

    if len(np.unique(y_true_all)) < 2 or sum(y_true_all) < 5:
        return None

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    groups_all = np.asarray(groups_all)
    auc = roc_auc_score(y_true_all, y_pred_all)
    fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
    ci_lo, ci_hi = bootstrap_auc_ci(y_true_all, y_pred_all, groups=groups_all)
    logger.info("Clínic LOPO-CV  %s  AUC=%.4f [%.3f–%.3f]", etype, auc, ci_lo, ci_hi)
    return {"label": "Clínic (LOPO-CV)", "key": "clinic_lopocv",
            "fpr": fpr, "tpr": tpr, "auc": auc, "ci": (ci_lo, ci_hi), "ls": "-"}


def roc_mimic_m1(etype: str, windows: pd.DataFrame, glmm) -> dict | None:
    sub = windows[windows["event_type"] == etype].copy()
    if sub["label"].nunique() < 2 or sub["label"].sum() < 5:
        return None
    sub["pred"] = glmm.predict_proba(sub)
    auc = roc_auc_score(sub["label"], sub["pred"])
    fpr, tpr, _ = roc_curve(sub["label"], sub["pred"])
    ci_lo, ci_hi = bootstrap_auc_ci(sub["label"].values, sub["pred"].values,
                                     groups=sub.get("patient_id", None)
                                     if "patient_id" in sub.columns else None)
    logger.info("MIMIC M1  %s  AUC=%.4f [%.3f–%.3f]", etype, auc, ci_lo, ci_hi)
    return {"label": "MIMIC M1 (direct transfer)", "key": "mimic_m1",
            "fpr": fpr, "tpr": tpr, "auc": auc, "ci": (ci_lo, ci_hi), "ls": "-"}


def roc_vitaldb_m1(etype: str, windows: pd.DataFrame, glmm) -> dict | None:
    sub = windows[windows["event_type"] == etype].copy()
    if sub["label"].nunique() < 2 or sub["label"].sum() < 5:
        return None
    sub["pred"] = glmm.predict_proba(sub)
    auc = roc_auc_score(sub["label"], sub["pred"])
    fpr, tpr, _ = roc_curve(sub["label"], sub["pred"])
    ci_lo, ci_hi = bootstrap_auc_ci(sub["label"].values, sub["pred"].values,
                                     groups=sub["patient_id"].values
                                     if "patient_id" in sub.columns else None)
    logger.info("VitalDB M1  %s  AUC=%.4f [%.3f–%.3f]", etype, auc, ci_lo, ci_hi)
    return {"label": "VitalDB M1 (direct transfer)", "key": "vitaldb_m1",
            "fpr": fpr, "tpr": tpr, "auc": auc, "ci": (ci_lo, ci_hi), "ls": "-"}


def roc_vitaldb_m2(etype: str, windows: pd.DataFrame, seed: int = 42) -> dict | None:
    """70/30 patient-level hold-out refit on VitalDB."""
    feat_cols = PARSIMONIOUS_FEATURES[etype]
    sub = windows[windows["event_type"] == etype].copy()
    avail = [c for c in feat_cols if c in sub.columns]

    pids = sub["patient_id"].values
    unique_pts = np.unique(pids)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_pts)
    split = int(0.7 * len(shuffled))
    train_pts = set(shuffled[:split])
    tr_mask = np.array([p in train_pts for p in pids])
    te_mask = ~tr_mask

    tr = sub.iloc[tr_mask]; te = sub.iloc[te_mask]
    if te["label"].nunique() < 2 or te["label"].sum() < 5:
        return None

    m = MixedLogisticModel(feature_cols=avail)
    m.fit(tr[avail], tr["label"].values, tr["patient_id"].values)
    preds = m.predict_proba(te[avail])
    auc = roc_auc_score(te["label"].values, preds)
    fpr, tpr, _ = roc_curve(te["label"].values, preds)
    ci_lo, ci_hi = bootstrap_auc_ci(te["label"].values, preds,
                                     groups=te["patient_id"].values)
    logger.info("VitalDB M2  %s  AUC=%.4f [%.3f–%.3f]", etype, auc, ci_lo, ci_hi)
    return {"label": "VitalDB M2 (refit 70/30)", "key": "vitaldb_m2",
            "fpr": fpr, "tpr": tpr, "auc": auc, "ci": (ci_lo, ci_hi), "ls": "-"}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _auc_label(info: dict) -> str:
    auc = info["auc"]
    ci = info["ci"]
    if not (np.isnan(ci[0]) or np.isnan(ci[1])):
        return f"{info['label']} (AUC {auc:.3f} [{ci[0]:.3f}–{ci[1]:.3f}])"
    return f"{info['label']} (AUC {auc:.3f})"


def plot_roc(curves_by_etype: dict[str, list[dict]]) -> None:
    etype_titles = {"hypotension": "A — Hypotension", "hypertension": "B — Hypertension"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    all_rows = []
    for ax, etype in zip(axes, ETYPES):
        curves = [c for c in curves_by_etype.get(etype, []) if c is not None]
        for info in curves:
            color = COLORS.get(info["key"], "#888888")
            ax.plot(info["fpr"], info["tpr"],
                    color=color, lw=1.8, ls=info["ls"],
                    label=_auc_label(info))
            # Store raw data
            for fp, tp in zip(info["fpr"], info["tpr"]):
                all_rows.append({"event_type": etype, "curve": info["key"],
                                  "fpr": fp, "tpr": tp, "auc": info["auc"]})

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance")
        ax.set_xlabel("1 − Specificity (FPR)", fontsize=10)
        ax.set_ylabel("Sensitivity (TPR)", fontsize=10)
        ax.set_title(etype_titles[etype], fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, lw=0.5)

    note = ("Clínic = training apparent (dashed) and LOPO-CV (solid).\n"
            "External: MIMIC (n=251 controls), VitalDB M1 direct transfer, M2 refit.\n"
            "CI 95% via patient-cluster bootstrap (B=1000).")
    fig.text(0.5, -0.04, note, ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"fig_roc_curves.{ext}"
        fig.savefig(p, dpi=180 if ext == "png" else None, bbox_inches="tight")
        logger.info("Saved: %s", p)
    plt.close(fig)

    # Save raw data
    csv_path = FIG_DIR / "roc_data.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== ROC Curves ===")

    clinic_w = load_windows("clinic")
    mimic_w  = load_windows("mimic")
    vdb_w    = load_windows("vitaldb")

    curves_by_etype: dict[str, list] = {e: [] for e in ETYPES}

    for etype in ETYPES:
        logger.info("--- %s ---", etype)
        glmm = load_glmm(etype)

        curves_by_etype[etype].append(roc_clinic_apparent(etype, clinic_w, glmm))
        curves_by_etype[etype].append(roc_clinic_lopocv(etype, clinic_w, glmm))
        curves_by_etype[etype].append(roc_mimic_m1(etype, mimic_w, glmm))
        curves_by_etype[etype].append(roc_vitaldb_m1(etype, vdb_w, glmm))
        curves_by_etype[etype].append(roc_vitaldb_m2(etype, vdb_w))

    plot_roc(curves_by_etype)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
