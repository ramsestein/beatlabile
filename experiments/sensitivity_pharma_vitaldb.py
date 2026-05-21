"""sensitivity_pharma_vitaldb.py
Pharmacology sensitivity analysis for VitalDB cohort.

Stratifies VitalDB windows by vasoactive drug administration and computes
GLMM parsimonious AUC within each pharmacological stratum.

Data sources:
  - datasets/data/vitaldb/clinical_data/cases.csv  (bolus totals: eph/phe/epi/ca)
  - datasets/data/vitaldb/clinical_data/tracks.csv (infusion presence: NEPI/PHEN/...)
  - results/cache/vitaldb_windows.parquet
  - results/act1/glmm_parsimonious_{etype}.pkl

Outputs:
  - results/sensitivity/pharma_vitaldb_auc.csv
  - results/figures/fig_pharma_sensitivity.{pdf,png}
"""

import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from beatlabile.models.mixed_logistic import MixedLogisticModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
BASE = Path(__file__).resolve().parents[1]
CASES_CSV  = BASE / "datasets/data/vitaldb/clinical_data/cases.csv"
TRACKS_CSV = BASE / "datasets/data/vitaldb/clinical_data/tracks.csv"
WINDOWS_PQ = BASE / "results/cache/vitaldb_windows.parquet"
GLMM_DIR   = BASE / "results/act1"
OUT_DIR    = BASE / "results/sensitivity"
FIG_DIR    = BASE / "results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ETYPES = ["hypotension", "hypertension"]

# Parsimonious feature sets (same as Act 1 training)
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

# Vasoactive bolus columns in cases.csv
VASO_BOLUS = {
    "eph": "intraop_eph",   # ephedrine
    "phe": "intraop_phe",   # phenylephrine
    "epi": "intraop_epi",   # epinephrine
    "ca":  "intraop_ca",    # calcium (vasopressor use)
}

# Continuous infusion tracks (from tracks.csv)
VASO_INFUSION_TRACKS = [
    "Orchestra/NEPI_RATE",   # norepinephrine (most severe)
    "Orchestra/PHEN_RATE",   # phenylephrine infusion
    "Orchestra/DOPA_RATE",   # dopamine
    "Orchestra/DOBU_RATE",   # dobutamine
    "Orchestra/EPI_RATE",    # epinephrine
]


def fit_m2_predict(windows_stratum: pd.DataFrame, etype: str,
                   seed: int = 42) -> np.ndarray | None:
    """Fit M2 (GLMM refit on VitalDB, 70/30 patient hold-out) within a stratum.

    Returns predicted probabilities for the held-out 30% test set, or None
    if the stratum is too small to split.
    Labels and patient_ids are taken from windows_stratum columns.
    Returns (y_test, proba_test) tuple.
    """
    feat_cols = [c for c in PARSIMONIOUS_FEATURES.get(etype, [])
                 if c in windows_stratum.columns]
    if len(feat_cols) < 3:
        return None

    sub = windows_stratum[windows_stratum["event_type"] == etype].copy()
    X = sub[feat_cols].copy()
    y = sub["label"].values
    pids = sub["patient_id"].values

    unique_pts = np.unique(pids)
    if len(unique_pts) < 10 or (y == 1).sum() < 5 or (y == 0).sum() < 5:
        return None

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_pts)
    split = int(0.7 * len(shuffled))
    train_pts = set(shuffled[:split])
    tr = np.array([p in train_pts for p in pids])
    te = ~tr

    if (y[te] == 1).sum() < 3 or (y[te] == 0).sum() < 3:
        return None

    try:
        m = MixedLogisticModel(feature_cols=feat_cols)
        m.fit(X.iloc[tr], y[tr], pids[tr])
        proba_te = m.predict_proba(X.iloc[te])
        return y[te], proba_te, (te.sum() - (y[te] == 1).sum())
    except Exception as e:
        logger.debug("M2 fit failed: %s", e)
        return None


def build_pharma_flags(windows: pd.DataFrame) -> pd.DataFrame:
    """Attach pharmacology strata flags to windows DataFrame."""
    logger.info("Loading cases.csv ...")
    cases = pd.read_csv(CASES_CSV)
    cases["patient_id"] = cases["caseid"].apply(lambda x: f"{int(x):04d}")

    # Bolus flags
    for short, col in VASO_BOLUS.items():
        cases[f"bolus_{short}"] = (cases[col].fillna(0) > 0)

    cases["any_bolus_vaso"] = cases[[f"bolus_{s}" for s in VASO_BOLUS]].any(axis=1)

    logger.info("Loading tracks.csv ...")
    tracks = pd.read_csv(TRACKS_CSV)
    # Infusion flag: any NEPI/PHEN/DOPA/DOBU/EPI infusion track present
    infusion_cases = (
        tracks[tracks["tname"].isin(VASO_INFUSION_TRACKS)]
        .assign(patient_id=lambda d: d["caseid"].apply(lambda x: f"{int(x):04d}"))
        ["patient_id"]
        .unique()
    )
    cases["has_infusion_vaso"] = cases["patient_id"].isin(infusion_cases)

    # NEPI specifically (most clinically relevant)
    nepi_cases = (
        tracks[tracks["tname"] == "Orchestra/NEPI_RATE"]
        .assign(patient_id=lambda d: d["caseid"].apply(lambda x: f"{int(x):04d}"))
        ["patient_id"]
        .unique()
    )
    cases["has_nepi"] = cases["patient_id"].isin(nepi_cases)

    # Composite strata
    cases["any_vaso"] = cases["any_bolus_vaso"] | cases["has_infusion_vaso"]

    # Master stratum label for reporting
    def stratum_label(row):
        if row["has_nepi"]:
            return "nepi_infusion"
        if row["has_infusion_vaso"]:
            return "other_infusion"
        if row["bolus_phe"] or row["bolus_epi"]:
            return "phe_epi_bolus"
        if row["bolus_eph"]:
            return "eph_only"
        if row["bolus_ca"]:
            return "ca_bolus"
        return "no_vaso"

    cases["pharma_stratum"] = cases.apply(stratum_label, axis=1)

    flag_cols = [c for c in cases.columns
                 if c.startswith("bolus_") or c.startswith("has_") or c == "any_vaso"
                 or c == "pharma_stratum"]
    merged = windows.merge(cases[["patient_id"] + flag_cols], on="patient_id", how="left")

    logger.info("Pharma strata distribution (unique patients):")
    pts = merged.drop_duplicates("patient_id")
    for s, n in pts["pharma_stratum"].value_counts().items():
        logger.info("  %s: %d patients", s, n)

    return merged


def load_glmm(etype: str):
    pkl = GLMM_DIR / f"glmm_parsimonious_{etype}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(pkl)
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                       groups: np.ndarray, n_boot: int = 500,
                       seed: int = 42) -> tuple[float, float]:
    """Patient-cluster bootstrap CI for AUC."""
    rng = np.random.default_rng(seed)
    unique_g = np.unique(groups)
    aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_g, size=len(unique_g), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if len(aucs) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_auc_strata(windows: pd.DataFrame, etype: str, glmm) -> list[dict]:
    """Compute AUC (+ bootstrap CI) for M1 and M2 in each pharmacological stratum."""
    sub = windows[windows["event_type"] == etype].copy()
    feat_cols = glmm.feature_cols
    avail = [c for c in feat_cols if c in sub.columns]
    if len(avail) < len(feat_cols) // 2:
        logger.warning("Too few feature cols for %s (%d/%d)", etype, len(avail), len(feat_cols))
        return []

    try:
        sub["m1_pred"] = glmm.predict_proba(sub)
    except Exception as e:
        logger.error("predict_proba failed for %s: %s", etype, e)
        return []

    rows = []

    def _auc_row(mask, stratum_name):
        s = sub if isinstance(mask, slice) else sub[mask]
        n_pos = (s["label"] == 1).sum()
        n_neg = (s["label"] == 0).sum()
        if n_pos < 5 or n_neg < 5 or s["label"].nunique() < 2:
            return

        # M1: direct transfer (no refit)
        m1_auc = roc_auc_score(s["label"], s["m1_pred"])
        # Bootstrap CI for M1 (skip for tiny strata to save time)
        if n_pos >= 10 and "patient_id" in s.columns:
            m1_ci = _bootstrap_auc_ci(s["label"].values, s["m1_pred"].values,
                                       s["patient_id"].values, n_boot=500)
        else:
            m1_ci = (float("nan"), float("nan"))

        # M2: refit on this stratum's VitalDB data (70/30 hold-out)
        stratum_windows = sub if isinstance(mask, slice) else sub[mask]
        m2_result = fit_m2_predict(stratum_windows, etype)
        if m2_result is not None:
            y_te, p_te, n_ctrl_te = m2_result
            m2_auc = round(float(roc_auc_score(y_te, p_te)), 4)
            n_ev_m2 = int((y_te == 1).sum())
        else:
            m2_auc = float("nan")
            n_ev_m2 = 0

        # Flag small strata (< 50 events or < 30 unique patients)
        n_pts = s["patient_id"].nunique() if "patient_id" in s.columns else n_pos
        small = (n_pos < 50) or (n_pts < 30)

        rows.append({
            "event_type": etype,
            "stratum": stratum_name,
            "n_events": int(n_pos),
            "n_controls": int(n_neg),
            "m1_auc": round(m1_auc, 4),
            "m1_ci_lo": round(m1_ci[0], 4) if not np.isnan(m1_ci[0]) else float("nan"),
            "m1_ci_hi": round(m1_ci[1], 4) if not np.isnan(m1_ci[1]) else float("nan"),
            "m2_auc": m2_auc,
            "m2_n_events_test": n_ev_m2,
            "small_n": small,
        })

    # 1. Overall
    _auc_row(slice(None), "all")

    # 2. any_vaso vs no_vaso
    for flag_val, label in [(True, "any_vaso"), (False, "no_vaso")]:
        _auc_row(sub["any_vaso"] == flag_val, label)

    # 3. Fine strata
    for s_name in ["eph_only", "phe_epi_bolus", "ca_bolus", "other_infusion",
                   "nepi_infusion", "no_vaso"]:
        _auc_row(sub["pharma_stratum"] == s_name, f"stratum_{s_name}")

    return rows



def plot_results(df: pd.DataFrame) -> None:
    """Side-by-side M1 vs M2 AUC by stratum × outcome."""
    strata_order = ["all", "no_vaso", "any_vaso",
                    "eph_only", "phe_epi_bolus", "ca_bolus",
                    "other_infusion", "nepi_infusion"]
    strata_labels = {
        "all":             "All VitalDB",
        "no_vaso":         "No vasoactive drug",
        "any_vaso":        "Any vasoactive drug",
        "eph_only":        "Ephedrine bolus only",
        "phe_epi_bolus":   "Phenylephrine / Epinephrine bolus",
        "ca_bolus":        "Calcium bolus",
        "other_infusion":  "Vasopressor infusion (no NEPI)",
        "nepi_infusion":   "Norepinephrine infusion",
    }
    etype_colors = {
        "hypotension":  {"M1": "#74afd1", "M2": "#2166ac"},
        "hypertension": {"M1": "#f4a582", "M2": "#d6604d"},
    }
    offsets = {("hypotension","M1"): -0.27, ("hypotension","M2"): -0.09,
               ("hypertension","M1"): +0.09, ("hypertension","M2"): +0.27}

    present_strata = df["stratum"].unique()
    strata_plot = [s for s in strata_order if s in present_strata]
    y_pos = {s: i for i, s in enumerate(reversed(strata_plot))}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    legend_handles, legend_labels = [], []
    for etype in TARGET_ETYPES:
        sub = df[df["event_type"] == etype]
        for model, col in [("M1", "m1_auc"), ("M2", "m2_auc")]:
            color = etype_colors[etype][model]
            ls = "-" if model == "M2" else "--"
            added_legend = False
            for _, row in sub.iterrows():
                s = row["stratum"]
                if s not in y_pos:
                    continue
                auc = row[col]
                if np.isnan(auc):
                    continue
                y = y_pos[s] + offsets[(etype, model)]
                n_ev = row["n_events"] if model == "M1" else row["m2_n_events_test"]
                ax.barh(y, auc - 0.5, left=0.5, height=0.17,
                        color=color, alpha=0.85)
                ax.text(auc + 0.005, y, f"{auc:.3f}",
                        va="center", ha="left", fontsize=6.5)
                if not added_legend:
                    legend_handles.append(
                        plt.Rectangle((0,0),1,1, color=color, alpha=0.85))
                    legend_labels.append(f"{etype.capitalize()} {model}")
                    added_legend = True

    ax.axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([strata_labels.get(s, s) for s in reversed(strata_plot)], fontsize=8.5)
    ax.set_xlabel("AUC")
    ax.set_title("M1 (direct transfer) vs M2 (VitalDB refit) by Pharmacological Stratum",
                 fontsize=10.5)
    ax.set_xlim(0.3, 1.05)
    ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=8,
              ncol=2, framealpha=0.9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"fig_pharma_sensitivity.{ext}"
        fig.savefig(p, dpi=180 if ext == "png" else None)
        logger.info("Saved: %s", p)
    plt.close(fig)


def main():
    logger.info("=== Pharmacology Sensitivity Analysis — VitalDB ===")

    windows = pd.read_parquet(WINDOWS_PQ)
    logger.info("Windows loaded: %d rows, %d patients",
                len(windows), windows["patient_id"].nunique())

    windows = build_pharma_flags(windows)

    all_rows = []
    for etype in TARGET_ETYPES:
        logger.info("--- %s ---", etype)
        glmm = load_glmm(etype)
        rows = compute_auc_strata(windows, etype, glmm)
        for r in rows:
            ci_str = (f"[{r['m1_ci_lo']:.3f}–{r['m1_ci_hi']:.3f}]"
                      if not np.isnan(r.get("m1_ci_lo", float("nan"))) else "")
            logger.info("  %-30s  n_ev=%-4d  M1=%.4f%s  M2=%s%s",
                        r["stratum"], r["n_events"],
                        r["m1_auc"], f" {ci_str}" if ci_str else "",
                        f"{r['m2_auc']:.4f}" if not np.isnan(r["m2_auc"]) else "n/a",
                        " ⚠small" if r.get("small_n") else "")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "pharma_vitaldb_auc.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved: %s", out_csv)

    plot_results(df)
    logger.info("=== DONE ===")
    return df


if __name__ == "__main__":
    main()
