"""sensitivity_vent_vitaldb.py
Ventilation sensitivity analysis for VitalDB cohort.

Questions:
  1. Does PEEP level affect the autonomic signal quality / AUC?
  2. Does ventilation mode (volume-control vs pressure-control vs
     pressure-support/spontaneous) affect the results?

Both confounders are especially relevant for RSA-based features (rsa_max,
rsa_mean), which are present in the parsimonious model for hypotension.

Strategy
--------
  a. One pass over .vital files to extract per-case ventilation parameters:
       - median PEEP  (Primus/PEEP_MBAR, fallback Solar8000/VENT_MAWP proxy)
       - median PIP   (Solar8000/VENT_PIP or Primus/PIP_MBAR)
       - fraction of time with SET_INSP_PRES > 0  → pressure-support fraction
       - fraction of time with VENT_SET_PCP > 0   → pressure-control fraction
     PEEP is essentially constant within a case (anaesthesiologist sets it once);
     using the per-case median is therefore a valid approximation.
  b. Classify each case:
       peep_group  : "zero_low" (<5 cmH2O) | "moderate" (5-8) | "high" (>8)
       vent_mode   : "pressure_support" | "pressure_control" | "volume_control"
  c. Merge classification with the windows cache (patient_id key).
  d. Compute M1 AUC (direct transfer of Act-1 GLMM parsimonious) per stratum.
  e. Bootstrap CI and forest-plot figure.

Data sources:
  - datasets/data/vitaldb/vital_full_cases/*.vital  (waveform files)
  - results/cache/vitaldb_windows.parquet
  - results/act1/glmm_parsimonious_{etype}.pkl

Outputs:
  - results/sensitivity/vent_vitaldb_case_params.csv   (per-case summary)
  - results/sensitivity/vent_vitaldb_auc.csv           (AUC table)
  - results/figures/fig_vent_sensitivity.{pdf,png}

Run
---
  python experiments/sensitivity_vent_vitaldb.py
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from beatlabile.config import DATA_VITALDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parents[1]

# Windows cache: primary location, then fallback to alternate beatlabile results
_WINDOWS_CANDIDATES = [
    BASE / "results" / "cache" / "vitaldb_windows.parquet",
    Path(r"D:\beatlabile\results\sensitivity\map65\vitaldb_windows_map65.parquet"),
]
WINDOWS_PQ = next((p for p in _WINDOWS_CANDIDATES if p.exists()), _WINDOWS_CANDIDATES[0])

# GLMM models: primary location, then fallback to alternate beatlabile results
_GLMM_CANDIDATES = [
    BASE / "results" / "act1",
    Path(r"D:\beatlabile\results\act1"),
]
GLMM_DIR  = next((p for p in _GLMM_CANDIDATES if p.exists()), _GLMM_CANDIDATES[0])
OUT_DIR   = BASE / "results" / "sensitivity"
FIG_DIR   = BASE / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ETYPES = ["hypotension", "hypertension"]

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

# Ventilation channels to extract (in priority order)
PEEP_CHANNELS = ["Primus/PEEP_MBAR", "Primus/SET_INTER_PEEP", "Solar8000/VENT_MAWP"]
PIP_CHANNELS  = ["Solar8000/VENT_PIP", "Primus/PIP_MBAR"]
PS_CHANNELS   = ["Primus/SET_INSP_PRES"]           # pressure support set value
PC_CHANNELS   = ["Solar8000/VENT_SET_PCP"]          # pressure-control set value

# PEEP thresholds (cmH₂O / mbar — units are equivalent for practical purposes)
PEEP_LOW_THR  = 5.0
PEEP_HIGH_THR = 8.0


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Extract ventilation parameters from .vital files
# ══════════════════════════════════════════════════════════════════════════════

def _read_track_median(vf, candidates: list[str], valid_lo: float = 0.0,
                       valid_hi: float = 1e6) -> float | None:
    """Return median of the first available channel in *candidates*, or None."""
    track_names = list(vf.trks.keys()) if hasattr(vf, "trks") else []
    suffix_map = {t.split("/")[-1].upper(): t for t in track_names}

    for ch in candidates:
        # Try exact match first, then suffix match
        real_ch = None
        if ch in track_names:
            real_ch = ch
        else:
            suffix = ch.split("/")[-1].upper()
            if suffix in suffix_map:
                real_ch = suffix_map[suffix]

        if real_ch is None:
            continue

        try:
            arr = vf.to_numpy(real_ch, interval=1.0)   # 1-second samples
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float).ravel()
            arr = arr[(arr >= valid_lo) & (arr <= valid_hi)]
            if len(arr) == 0:
                continue
            return float(np.nanmedian(arr))
        except Exception:
            continue
    return None


def _pressure_support_fraction(vf, ps_channels: list[str]) -> float:
    """Fraction of 1-s samples where pressure-support set value > 0."""
    track_names = list(vf.trks.keys()) if hasattr(vf, "trks") else []
    suffix_map = {t.split("/")[-1].upper(): t for t in track_names}

    for ch in ps_channels:
        real_ch = None
        if ch in track_names:
            real_ch = ch
        else:
            suffix = ch.split("/")[-1].upper()
            if suffix in suffix_map:
                real_ch = suffix_map[suffix]
        if real_ch is None:
            continue
        try:
            arr = vf.to_numpy(real_ch, interval=1.0)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float).ravel()
            valid = arr[~np.isnan(arr)]
            if len(valid) == 0:
                continue
            return float((valid > 0).mean())
        except Exception:
            continue
    return 0.0


def extract_vent_params(vitaldb_root: Path) -> pd.DataFrame:
    """One pass over .vital files → DataFrame with per-case ventilation summary.

    Columns: patient_id, peep_median, pip_median, ps_fraction, pc_fraction,
             peep_group, vent_mode
    """
    try:
        import vitaldb as vdb
    except ImportError:
        raise ImportError("vitaldb package required: pip install vitaldb")

    rows = []
    vital_files = sorted(vitaldb_root.glob("*.vital"))
    logger.info("Scanning %d .vital files for ventilation parameters ...", len(vital_files))

    for i, p in enumerate(vital_files):
        if i % 50 == 0:
            logger.info("  %d / %d", i, len(vital_files))
        try:
            vf = vdb.VitalFile(str(p))
        except Exception as e:
            logger.debug("Cannot open %s: %s", p.name, e)
            continue

        patient_id = p.stem.zfill(4)

        peep = _read_track_median(vf, PEEP_CHANNELS, valid_lo=0.0, valid_hi=30.0)
        pip  = _read_track_median(vf, PIP_CHANNELS,  valid_lo=0.0, valid_hi=60.0)
        ps_frac = _pressure_support_fraction(vf, PS_CHANNELS)
        pc_frac = _pressure_support_fraction(vf, PC_CHANNELS)   # reuse logic

        rows.append({
            "patient_id": patient_id,
            "peep_median": peep,
            "pip_median":  pip,
            "ps_fraction": ps_frac,   # fraction of time with PS > 0
            "pc_fraction": pc_frac,   # fraction of time with PC set > 0
        })

    df = pd.DataFrame(rows)

    # ── PEEP group ──────────────────────────────────────────────────────────
    def _peep_group(x):
        if pd.isna(x):
            return "unknown"
        if x < PEEP_LOW_THR:
            return "zero_low"          # 0-4 cmH₂O
        if x <= PEEP_HIGH_THR:
            return "moderate"          # 5-8 cmH₂O
        return "high"                  # >8 cmH₂O

    df["peep_group"] = df["peep_median"].apply(_peep_group)

    # ── Ventilation mode ────────────────────────────────────────────────────
    # Classify by dominant mode:
    #   pressure_support  → PS channel active >20% of time (spontaneous effort)
    #   pressure_control  → PC set point active >20% and PS ≤20%
    #   volume_control    → neither → classic IPPV
    def _vent_mode(row):
        if row["ps_fraction"] > 0.20:
            return "pressure_support"
        if row["pc_fraction"] > 0.20:
            return "pressure_control"
        return "volume_control"

    df["vent_mode"] = df.apply(_vent_mode, axis=1)

    logger.info("PEEP group distribution:\n%s", df["peep_group"].value_counts().to_string())
    logger.info("Ventilation mode distribution:\n%s", df["vent_mode"].value_counts().to_string())
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Merge vent flags with windows cache
# ══════════════════════════════════════════════════════════════════════════════

def build_vent_windows(windows: pd.DataFrame, vent_df: pd.DataFrame) -> pd.DataFrame:
    merged = windows.merge(
        vent_df[["patient_id", "peep_median", "pip_median",
                 "ps_fraction", "pc_fraction", "peep_group", "vent_mode"]],
        on="patient_id",
        how="left",
    )
    n_missing = merged["peep_group"].isna().sum()
    if n_missing:
        logger.warning("%d windows have no ventilation data (patient not in scan).", n_missing)
        merged["peep_group"] = merged["peep_group"].fillna("unknown")
        merged["vent_mode"]  = merged["vent_mode"].fillna("unknown")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — AUC per stratum
# ══════════════════════════════════════════════════════════════════════════════

class _FallbackModel:
    """Logistic regression fitted on VitalDB windows (M2 surrogate).

    Used when the Clinic-trained GLMM pkl is unavailable.  The model is
    fitted once on *all* available windows for a given etype and then
    applied per stratum — this is an optimistic (in-sample) estimate but
    relative comparisons across strata remain informative.
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self._scaler = StandardScaler()
        self._lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self._medians: pd.Series | None = None
        self._fitted = False

    def fit(self, windows: pd.DataFrame, etype: str) -> "_FallbackModel":
        sub = windows[windows["event_type"] == etype].copy()
        avail = [c for c in self.feature_cols if c in sub.columns]
        X = sub[avail].copy()
        self._medians = X.median()
        X = X.fillna(self._medians).fillna(0.0)
        y = sub["label"].values
        Xs = self._scaler.fit_transform(X)
        self._lr.fit(Xs, y)
        self.feature_cols = avail
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        avail = [c for c in self.feature_cols if c in X.columns]
        Xsub = X[avail].fillna(self._medians).fillna(0.0)
        Xs = self._scaler.transform(Xsub)
        return self._lr.predict_proba(Xs)[:, 1]


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


def compute_auc_strata(windows: pd.DataFrame, etype: str, glmm,
                        stratum_col: str, stratum_order: list[str]) -> list[dict]:
    """Compute M1 AUC + bootstrap CI for each level of *stratum_col*."""
    sub = windows[windows["event_type"] == etype].copy()
    feat_cols = [c for c in glmm.feature_cols if c in sub.columns]

    try:
        sub["m1_pred"] = glmm.predict_proba(sub)
    except Exception as e:
        logger.error("predict_proba failed (%s): %s", etype, e)
        return []

    rows = []

    # Overall (all windows, regardless of stratum)
    if sub["label"].nunique() == 2 and (sub["label"] == 1).sum() >= 5:
        auc_all = roc_auc_score(sub["label"], sub["m1_pred"])
        ci_all  = _bootstrap_auc_ci(sub["label"].values, sub["m1_pred"].values,
                                     sub["patient_id"].values)
        rows.append({
            "event_type": etype, "stratum_col": stratum_col,
            "stratum": "all",
            "n_windows": len(sub), "n_events": int((sub["label"] == 1).sum()),
            "n_patients": sub["patient_id"].nunique(),
            "auc": auc_all, "ci_lo": ci_all[0], "ci_hi": ci_all[1],
        })

    # Per stratum
    for stratum in stratum_order:
        mask = sub[stratum_col] == stratum
        s = sub[mask]
        n_pos = (s["label"] == 1).sum()
        n_neg = (s["label"] == 0).sum()
        if n_pos < 5 or n_neg < 5 or s["label"].nunique() < 2:
            logger.info("  Skip stratum %s=%s: n_pos=%d n_neg=%d", stratum_col, stratum, n_pos, n_neg)
            rows.append({
                "event_type": etype, "stratum_col": stratum_col,
                "stratum": stratum,
                "n_windows": len(s), "n_events": int(n_pos),
                "n_patients": s["patient_id"].nunique(),
                "auc": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
            })
            continue
        auc = roc_auc_score(s["label"], s["m1_pred"])
        ci  = _bootstrap_auc_ci(s["label"].values, s["m1_pred"].values,
                                 s["patient_id"].values)
        rows.append({
            "event_type": etype, "stratum_col": stratum_col,
            "stratum": stratum,
            "n_windows": len(s), "n_events": int(n_pos),
            "n_patients": s["patient_id"].nunique(),
            "auc": auc, "ci_lo": ci[0], "ci_hi": ci[1],
        })
        logger.info("  [%s] %s=%s: AUC=%.3f [%.3f–%.3f]  n_pos=%d  n_pts=%d",
                    etype, stratum_col, stratum, auc, ci[0], ci[1],
                    n_pos, s["patient_id"].nunique())
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Figure
# ══════════════════════════════════════════════════════════════════════════════

PEEP_ORDER = ["zero_low", "moderate", "high"]
MODE_ORDER = ["volume_control", "pressure_control", "pressure_support"]

PEEP_LABELS = {
    "zero_low":  "Low PEEP\n(<5 cmH₂O)",
    "moderate":  "Moderate PEEP\n(5–8 cmH₂O)",
    "high":      "High PEEP\n(>8 cmH₂O)",
    "all":       "All cases",
    "unknown":   "No PEEP data",
}
MODE_LABELS = {
    "volume_control":    "Volume-controlled",
    "pressure_control":  "Pressure-controlled",
    "pressure_support":  "Pressure-support\n(spontaneous)",
    "all":               "All cases",
    "unknown":           "Mode unknown",
}

ETYPE_COLORS = {
    "hypotension":  "#2166ac",
    "hypertension": "#d6604d",
}
ETYPE_TITLES = {
    "hypotension":  "Hypotension",
    "hypertension": "Hypertension",
}


def _forest_ax(ax: plt.Axes, rows: list[dict], strata_order: list[str],
               label_map: dict, etype: str, title: str, show_ylabel: bool) -> None:
    """Draw a single horizontal forest-plot panel."""
    color = ETYPE_COLORS[etype]

    # Filter to this etype's rows (excluding 'all' for plotting)
    plot_rows = [r for r in rows if r["event_type"] == etype
                 and r["stratum"] != "all" and r["stratum"] != "unknown"]
    # Preserve requested order
    order = [s for s in strata_order if s in {r["stratum"] for r in plot_rows}]
    row_map = {r["stratum"]: r for r in plot_rows}

    ys = np.arange(len(order))

    for y, stratum in zip(ys, order):
        r = row_map.get(stratum)
        if r is None or np.isnan(r["auc"]):
            ax.scatter([np.nan], [y], color=color, s=60, zorder=5)
            continue
        ax.scatter([r["auc"]], [y], color=color, s=60, zorder=5)
        if not (np.isnan(r["ci_lo"]) or np.isnan(r["ci_hi"])):
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=color, lw=2, zorder=4)
        # Sample sizes
        note = f"n={r['n_patients']} pts, {r['n_events']} events"
        ax.text(0.99, y, note, ha="right", va="center", fontsize=6.5,
                color="#555", transform=ax.get_yaxis_transform())

    # Overall AUC reference line
    all_row = next((r for r in rows if r["event_type"] == etype
                    and r["stratum"] == "all"), None)
    if all_row and not np.isnan(all_row["auc"]):
        ax.axvline(all_row["auc"], color=color, lw=1.2, ls="--", alpha=0.6,
                   label=f"Overall AUC={all_row['auc']:.3f}")

    ax.axvline(0.5, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlim(0.35, 1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([label_map.get(s, s) for s in order], fontsize=8)
    ax.set_xlabel("AUC M1 (direct transfer)", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    if all_row and not np.isnan(all_row["auc"]):
        ax.legend(fontsize=7, loc="lower right", framealpha=0.8)


def make_figure(auc_df: pd.DataFrame) -> None:
    """2-column × 2-row figure: rows = PEEP / vent mode, cols = event types."""
    rows_all = auc_df.to_dict("records")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    panel_specs = [
        # (row, col, stratum_col, order, label_map, subtitle)
        (0, 0, "peep_group",  PEEP_ORDER, PEEP_LABELS, "A  Hypotension — PEEP level"),
        (0, 1, "peep_group",  PEEP_ORDER, PEEP_LABELS, "B  Hypertension — PEEP level"),
        (1, 0, "vent_mode",   MODE_ORDER, MODE_LABELS, "C  Hypotension — Ventilation mode"),
        (1, 1, "vent_mode",   MODE_ORDER, MODE_LABELS, "D  Hypertension — Ventilation mode"),
    ]
    etypes = ["hypotension", "hypertension", "hypotension", "hypertension"]

    for (r, c, scol, order, lmap, subtitle), etype in zip(panel_specs, etypes):
        ax = axes[r, c]
        panel_rows = [row for row in rows_all if row["stratum_col"] == scol]
        _forest_ax(ax, panel_rows, order, lmap, etype,
                   title=subtitle, show_ylabel=(c == 0))

    fig.suptitle(
        "Ventilation sensitivity — VitalDB  |  M1 GLMM direct transfer\n"
        "PEEP level and ventilation mode stratification",
        fontsize=10, fontweight="bold", y=1.01,
    )

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig_vent_sensitivity.{ext}"
        fig.savefig(out, dpi=180 if ext == "png" else None, bbox_inches="tight")
        logger.info("Saved: %s", out)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=== Ventilation Sensitivity Analysis — VitalDB ===")

    # ── 1. Load or build per-case ventilation parameters ──────────────────
    case_params_path = OUT_DIR / "vent_vitaldb_case_params.csv"

    if case_params_path.exists():
        logger.info("Loading cached case ventilation params: %s", case_params_path)
        vent_df = pd.read_csv(case_params_path, dtype={"patient_id": str})
        vent_df["patient_id"] = vent_df["patient_id"].str.zfill(4)
    else:
        vitaldb_root = Path(DATA_VITALDB) if not isinstance(DATA_VITALDB, Path) else DATA_VITALDB
        if not vitaldb_root.exists():
            logger.error(
                "VitalDB root not found: %s\n"
                "Set DATA_VITALDB in beatlabile/config.py or create "
                "the directory with .vital files.", vitaldb_root
            )
            return
        vent_df = extract_vent_params(vitaldb_root)
        vent_df.to_csv(case_params_path, index=False)
        logger.info("Saved case ventilation params → %s", case_params_path)

    # ── 2. Load windows cache ──────────────────────────────────────────────
    if not WINDOWS_PQ.exists():
        logger.error(
            "Windows cache not found. Tried:\n%s\n"
            "Run experiments/act3_vitaldb.py first to build the cache.",
            "\n".join(f"  {p}" for p in _WINDOWS_CANDIDATES)
        )
        return
    logger.info("Using windows cache: %s", WINDOWS_PQ)

    windows = pd.read_parquet(WINDOWS_PQ)
    windows["patient_id"] = windows["patient_id"].astype(str).str.zfill(4)
    logger.info("Loaded %d windows (%d patients)", len(windows),
                windows["patient_id"].nunique())

    windows = build_vent_windows(windows, vent_df)

    # Distribution report
    logger.info("\nPEEP group distribution (windows):\n%s",
                windows["peep_group"].value_counts().to_string())
    logger.info("\nVent mode distribution (windows):\n%s",
                windows["vent_mode"].value_counts().to_string())

    # ── 3. Load GLMM models and compute AUC ───────────────────────────────
    all_rows: list[dict] = []

    for etype in TARGET_ETYPES:
        glmm_path = GLMM_DIR / f"glmm_parsimonious_{etype}.pkl"
        if glmm_path.exists():
            with open(glmm_path, "rb") as fh:
                glmm = pickle.load(fh)
            model_label = "M1 (Clinic GLMM)"
        else:
            logger.warning(
                "GLMM pkl not found for %s — using M2 fallback "
                "(LogisticRegression fitted on VitalDB windows).", etype
            )
            feat_cols = PARSIMONIOUS_FEATURES[etype]
            glmm = _FallbackModel(feat_cols).fit(windows, etype)
            model_label = "M2 (VitalDB fit)"
        logger.info("\n── %s [%s] ──", etype.upper(), model_label)

        for stratum_col, order in [
            ("peep_group", PEEP_ORDER + ["unknown"]),
            ("vent_mode",  MODE_ORDER + ["unknown"]),
        ]:
            logger.info("  Stratum: %s", stratum_col)
            r = compute_auc_strata(windows, etype, glmm, stratum_col, order)
            all_rows.extend(r)

    if not all_rows:
        logger.error("No AUC results computed. Check data availability.")
        return

    auc_df = pd.DataFrame(all_rows)
    auc_csv = OUT_DIR / "vent_vitaldb_auc.csv"
    auc_df.to_csv(auc_csv, index=False)
    logger.info("\nAUC table saved → %s", auc_csv)

    # Pretty-print summary
    print("\n" + "=" * 65)
    print("  PEEP level vs AUC")
    print("=" * 65)
    for _, row in auc_df[auc_df["stratum_col"] == "peep_group"].iterrows():
        if row["stratum"] == "unknown":
            continue
        ci_str = (f"[{row['ci_lo']:.3f}–{row['ci_hi']:.3f}]"
                  if not np.isnan(row["ci_lo"]) else "[n/a]")
        print(f"  {row['event_type']:12s}  {row['stratum']:12s}  "
              f"AUC={row['auc']:.3f} {ci_str}  "
              f"pts={row['n_patients']}  events={row['n_events']}")

    print("\n" + "=" * 65)
    print("  Ventilation mode vs AUC")
    print("=" * 65)
    for _, row in auc_df[auc_df["stratum_col"] == "vent_mode"].iterrows():
        if row["stratum"] == "unknown":
            continue
        ci_str = (f"[{row['ci_lo']:.3f}–{row['ci_hi']:.3f}]"
                  if not np.isnan(row["ci_lo"]) else "[n/a]")
        print(f"  {row['event_type']:12s}  {row['stratum']:20s}  "
              f"AUC={row['auc']:.3f} {ci_str}  "
              f"pts={row['n_patients']}  events={row['n_events']}")

    # ── 4. Figure ──────────────────────────────────────────────────────────
    make_figure(auc_df)
    logger.info("=== DONE ===")


if __name__ == "__main__":
    main()
