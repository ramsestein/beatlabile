#!/usr/bin/env python3
"""
Q1 Event-Locked Trajectory Analysis
=====================================
Lee features_long.parquet y event_windows.csv (ya generados).
Produce trayectorias alineadas al estímulo, figuras diagnósticas,
trajectory_summary.csv y trajectory_diagnostic.md.

NO recalcula features, NO corre tests estadísticos.
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("trajectory")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("../results/validation/Q1")
FEATURES_FILE = RESULTS_DIR / "features_long.parquet"
EVENTS_FILE   = RESULTS_DIR / "event_windows.csv"
FIGURES_DIR   = RESULTS_DIR / "figures"
OUT_SUMMARY   = RESULTS_DIR / "trajectory_summary.csv"
OUT_REPORT    = RESULTS_DIR / "trajectory_diagnostic.md"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_STEP   = 30           # segundos
HALF_WINDOW = 300          # ±5 minutos
GRID        = np.arange(-HALF_WINDOW, HALF_WINDOW + 1, GRID_STEP)  # -300…+300
MATCH_TOL   = 15           # ±15 s para buscar el timepoint más cercano
BL_MIN      = -300         # baseline start (−5 min)
BL_MAX      = -120         # baseline end   (−2 min)

# ── Features ──────────────────────────────────────────────────────────────────
PRIMARY_FEATURES = ["ptt_cv", "ptt_std", "brs_alpha_lf", "ptt_arv", "pai_mean"]
CONTROL_FEATURES = ["hr_mean", "hrv_rmssd", "hrv_sdnn", "ptt_mean"]
ALL_FEATURES     = PRIMARY_FEATURES + CONTROL_FEATURES

# Dirección esperada por BeatLabile (signo del efecto post-estímulo)
EXPECTED_DIR = {
    "ptt_cv":       "-",   # variabilidad PTT disminuye (simpático → vasoconstr)
    "ptt_std":      "+",   # idem
    "brs_alpha_lf": "-",   # BRS disminuye bajo simpático
    "ptt_arv":      "+",   # variabilidad PTT sube
    "pai_mean":     "-",   # amplitud PPG cae con vasoconstricción
    "hr_mean":      "+",   # FC sube
    "hrv_rmssd":    "-",   # RMSSD cae
    "hrv_sdnn":     "-",   # SDNN cae
    "ptt_mean":     "-",   # PTT acorta con vasoconstricción
}

FEATURE_LABELS = {
    "ptt_cv":       "PTT CV",
    "ptt_std":      "PTT Std (ms)",
    "brs_alpha_lf": "BRS α-LF (ms/mmHg)",
    "ptt_arv":      "PTT ARV (ms)",
    "pai_mean":     "PAI mean (a.u.)",
    "hr_mean":      "HR mean (bpm)*",
    "hrv_rmssd":    "HRV RMSSD (ms)",
    "hrv_sdnn":     "HRV SDNN (ms)",
    "ptt_mean":     "PTT mean (ms)",
}

# ── Load data ─────────────────────────────────────────────────────────────────
log.info("Cargando features_long.parquet …")
feat_df = pd.read_parquet(FEATURES_FILE)

# Derivar hr_mean (aprox) = n_rr / 30 * 60
feat_df["hr_mean"] = feat_df["n_rr"] / 30.0 * 60.0

# Enmascarar brs_alpha_lf cuando no es válido
feat_df.loc[~feat_df["brs_valid"], "brs_alpha_lf"] = np.nan

log.info("Cargando event_windows.csv …")
ev_df = pd.read_csv(EVENTS_FILE)
events = ev_df[ev_df["window_type"] == "event"].reset_index(drop=True)
log.info(f"  {len(events)} eventos limpios")

# Verificar features disponibles
missing_feats = [f for f in ALL_FEATURES if f not in feat_df.columns]
if missing_feats:
    log.warning(f"Features NO disponibles en parquet: {missing_feats}")
available_feats = [f for f in ALL_FEATURES if f in feat_df.columns]

# Pre-cachear datos por paciente para velocidad
log.info("Pre-cacheando datos por paciente …")
patient_cache: dict[str, pd.DataFrame] = {}
for pid in feat_df["patient_id"].unique():
    patient_cache[pid] = feat_df[feat_df["patient_id"] == pid].copy()


# ── Build event-locked grid ───────────────────────────────────────────────────
def build_grids() -> tuple[dict, list]:
    """
    Retorna:
      grids: {feat: np.array(n_events, n_grid)}
      meta:  list of dicts con info de cada evento
    """
    n_events = len(events)
    n_grid   = len(GRID)
    grids = {f: np.full((n_events, n_grid), np.nan) for f in available_feats}
    meta  = []

    for i, row in events.iterrows():
        pid = str(row["patient_id"])
        t0  = float(row["t_start_s"])   # t_stimulus
        grp = row["group"]
        sub = row["event_subcategory"]
        meta.append({"event_idx": i, "patient_id": pid,
                      "group": grp, "subcategory": sub, "t_stimulus": t0})

        if pid not in patient_cache:
            log.warning(f"  Paciente {pid} no encontrado en features_long")
            continue

        pat = patient_cache[pid].copy()
        pat["rel_t"] = pat["t_window_start_s"] - t0
        pat = pat[(pat["rel_t"] >= BL_MIN - MATCH_TOL) &
                  (pat["rel_t"] <= HALF_WINDOW + MATCH_TOL)]

        if len(pat) == 0:
            log.warning(f"  Evento {pid}@{t0:.0f}s: sin datos en ±5min")
            continue

        rel_vals = pat["rel_t"].values
        for j, g in enumerate(GRID):
            diffs = np.abs(rel_vals - g)
            best  = diffs.argmin()
            if diffs[best] <= MATCH_TOL:
                idx = pat.index[best]
                for f in available_feats:
                    grids[f][i, j] = pat.loc[idx, f]

    return grids, meta


log.info("Construyendo rejilla de trayectorias …")
grids, meta = build_grids()
meta_df = pd.DataFrame(meta)
log.info(f"  Rejilla construida: {len(events)} eventos × {len(GRID)} timepoints")


# ── Helpers de plot ───────────────────────────────────────────────────────────
def plot_trajectory_panel(ax, traj: np.ndarray, feat: str,
                          title: str = "", color: str = "#2166ac",
                          color2: str = None, label2: str = None,
                          traj2: np.ndarray = None):
    """
    Dibuja trayectorias en `ax`.
    traj shape: (n_events, n_grid).
    Si se pasa traj2, dibuja dos grupos comparativos.
    """
    grid_min = GRID / 60.0  # en minutos para el eje x

    # Baseline índices (−5 a −2 min)
    bl_idx = np.where((GRID >= BL_MIN) & (GRID <= BL_MAX))[0]

    def _draw(data, col, lbl=None):
        # Líneas individuales
        for k in range(data.shape[0]):
            row = data[k]
            if np.sum(~np.isnan(row)) >= 3:
                ax.plot(grid_min, row, color=col, alpha=0.15,
                        linewidth=0.7, zorder=1)
        # Mediana e IQR
        med = np.nanmedian(data, axis=0)
        q25 = np.nanpercentile(data, 25, axis=0)
        q75 = np.nanpercentile(data, 75, axis=0)
        ax.fill_between(grid_min, q25, q75, color=col, alpha=0.20, zorder=2)
        ax.plot(grid_min, med, color=col, linewidth=2.0, zorder=3, label=lbl)
        return med, q25, q75

    if traj2 is not None:
        med1, _, _ = _draw(traj,  color,  label2 if label2 else "Grupo 1")
        med2, _, _ = _draw(traj2, color2, "Grupo 2")
    else:
        med, q25, q75 = _draw(traj, color)
        # Baseline horizontal
        bl_vals = traj[:, bl_idx].flatten()
        bl_med  = np.nanmedian(bl_vals)
        ax.axhline(bl_med, color="gray", linewidth=1.0,
                   linestyle=":", alpha=0.7, zorder=2)

    # Línea t=0
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=4)

    # Decor
    n_valid = int(np.sum(np.sum(~np.isnan(traj), axis=1) >= 3))
    ax.set_xlabel("Tiempo relativo al estímulo (min)", fontsize=7)
    ax.set_ylabel(FEATURE_LABELS.get(feat, feat), fontsize=7)
    ax.set_title(f"{title}\n(n={n_valid})", fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    ax.set_xlim(-5, 5)


# ── Figura A: overall 3×3 ────────────────────────────────────────────────────
log.info("Generando figura A: trajectory_overall.png …")
fig, axes = plt.subplots(3, 3, figsize=(13, 11))
fig.suptitle("Trayectorias event-locked (todos los estímulos pooled, n=52)",
             fontsize=11, fontweight="bold")
colors_all = ["#2166ac"] * 5 + ["#d73027"] * 4

for ax, feat, col in zip(axes.flat, ALL_FEATURES, colors_all):
    if feat in grids:
        plot_trajectory_panel(ax, grids[feat], feat,
                              title=FEATURE_LABELS.get(feat, feat), color=col)
    else:
        ax.text(0.5, 0.5, f"{feat}\nNO DISPONIBLE",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(feat, fontsize=8)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "trajectory_overall.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log.info(f"  Guardado: trajectory_overall.png")


# ── Figura B: por subcategoría (5 features × subcat) ─────────────────────────
log.info("Generando figura B: trajectory_by_stimulus_type.png …")
subcats_all = ["anclaje", "piel", "trocar", "sutura", "fresa"]
MIN_EVENTS_SUBCAT = 3
subcats_valid = []
for sc in subcats_all:
    n = int((meta_df["subcategory"] == sc).sum())
    if n < MIN_EVENTS_SUBCAT:
        log.warning(f"  Subcategoría '{sc}' tiene solo {n} eventos — OMITIDA (umbral={MIN_EVENTS_SUBCAT})")
    else:
        subcats_valid.append(sc)
log.info(f"  Subcategorías válidas (≥{MIN_EVENTS_SUBCAT}): {subcats_valid}")

n_rows = len(subcats_valid)
n_cols = len(PRIMARY_FEATURES)
palette_sub = ["#1b7837", "#762a83", "#e08214", "#2166ac", "#d73027"]

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 3.0, n_rows * 2.8),
                         squeeze=False)
fig.suptitle("Trayectorias event-locked por tipo de estímulo",
             fontsize=11, fontweight="bold")

for r, sc in enumerate(subcats_valid):
    sc_idx = meta_df[meta_df["subcategory"] == sc].index.tolist()
    for c, feat in enumerate(PRIMARY_FEATURES):
        ax = axes[r][c]
        if feat not in grids:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        traj_sc = grids[feat][sc_idx, :]
        col     = palette_sub[c % len(palette_sub)]
        lbl     = f"{sc} (n={len(sc_idx)})" if c == 0 else sc
        plot_trajectory_panel(ax, traj_sc, feat,
                              title=FEATURE_LABELS.get(feat, feat) if r == 0 else "",
                              color=col)
        if c == 0:
            ax.set_ylabel(f"{sc}\n{FEATURE_LABELS.get(feat, feat)}", fontsize=7)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "trajectory_by_stimulus_type.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("  Guardado: trajectory_by_stimulus_type.png")


# ── Figura C: por grupo (interescalénico vs supra+axilar) ────────────────────
log.info("Generando figura C: trajectory_by_group.png …")
groups = sorted(meta_df["group"].unique())
log.info(f"  Grupos encontrados: {groups}")
group_colors = {"interescalenico": "#d73027", "supra_axilar": "#2166ac"}
default_colors = ["#d73027", "#2166ac", "#1a9850", "#984ea3"]

fig, axes = plt.subplots(1, len(PRIMARY_FEATURES),
                         figsize=(len(PRIMARY_FEATURES) * 3.2, 3.8),
                         squeeze=False)
fig.suptitle("Trayectorias event-locked por grupo de bloqueo",
             fontsize=11, fontweight="bold")

for c, feat in enumerate(PRIMARY_FEATURES):
    ax = axes[0][c]
    if feat not in grids:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                transform=ax.transAxes)
        continue

    grid_min = GRID / 60.0
    bl_idx   = np.where((GRID >= BL_MIN) & (GRID <= BL_MAX))[0]

    for gi, grp in enumerate(groups):
        grp_idx = meta_df[meta_df["group"] == grp].index.tolist()
        data    = grids[feat][grp_idx, :]
        col     = group_colors.get(grp, default_colors[gi % len(default_colors)])
        n_valid = int(np.sum(np.sum(~np.isnan(data), axis=1) >= 3))
        lbl     = f"{grp} (n={n_valid})"

        med = np.nanmedian(data, axis=0)
        q25 = np.nanpercentile(data, 25, axis=0)
        q75 = np.nanpercentile(data, 75, axis=0)

        ax.fill_between(grid_min, q25, q75, color=col, alpha=0.18, zorder=2)
        ax.plot(grid_min, med, color=col, linewidth=2.0, zorder=3, label=lbl)

        for k in range(data.shape[0]):
            row = data[k]
            if np.sum(~np.isnan(row)) >= 3:
                ax.plot(grid_min, row, color=col, alpha=0.12,
                        linewidth=0.6, zorder=1)

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", zorder=4)
    ax.set_xlim(-5, 5)
    ax.set_xlabel("Tiempo relativo (min)", fontsize=7)
    ax.set_ylabel(FEATURE_LABELS.get(feat, feat), fontsize=7)
    ax.set_title(FEATURE_LABELS.get(feat, feat), fontsize=8, pad=3)
    ax.tick_params(labelsize=6)
    if c == len(PRIMARY_FEATURES) - 1:
        ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
fig.savefig(FIGURES_DIR / "trajectory_by_group.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
log.info("  Guardado: trajectory_by_group.png")


# ── trajectory_summary.csv ────────────────────────────────────────────────────
log.info("Computando trajectory_summary.csv …")
bl_idx   = np.where((GRID >= BL_MIN) & (GRID <= BL_MAX))[0]
post_idx = np.where(GRID > 0)[0]

def find_return_to_baseline(med_traj, bl_med, bl_iqr_half, peak_j, peak_dir):
    """
    Primer timepoint tras peak_j en que la mediana vuelve dentro del
    IQR del baseline.
    bl_iqr_half = (q75_baseline - q25_baseline) / 2
    """
    search_range = np.arange(peak_j + 1, len(GRID))
    for j in search_range:
        if abs(med_traj[j] - bl_med) <= bl_iqr_half:
            return GRID[j]
    return np.nan

rows = []
for feat in PRIMARY_FEATURES:
    if feat not in grids:
        log.warning(f"  {feat}: no disponible, omitida del summary")
        continue

    data    = grids[feat]
    n_valid = int(np.sum(np.sum(~np.isnan(data), axis=1) >= 3))

    if n_valid < 10:
        log.warning(f"  {feat}: solo {n_valid} eventos válidos (< 10) — trayectoria muy ruidosa")

    # Baseline
    bl_data   = data[:, bl_idx]
    bl_med    = float(np.nanmedian(bl_data))
    bl_q25    = float(np.nanpercentile(bl_data, 25))
    bl_q75    = float(np.nanpercentile(bl_data, 75))
    bl_iqr_h  = (bl_q75 - bl_q25) / 2.0

    # Mediana por timepoint
    med_traj = np.nanmedian(data, axis=0)

    # Peak en [0, +300]
    post_vals = med_traj[post_idx]
    deviations = post_vals - bl_med
    if np.all(np.isnan(deviations)):
        peak_val    = np.nan
        peak_t      = np.nan
        peak_dir    = "?"
        return_t    = np.nan
    else:
        abs_dev    = np.abs(deviations)
        best_local = int(np.nanargmax(abs_dev))
        peak_j_global = post_idx[best_local]
        peak_val   = float(med_traj[peak_j_global])
        peak_t     = float(GRID[peak_j_global])
        peak_dir   = "+" if deviations[best_local] > 0 else "-"
        return_t   = find_return_to_baseline(
            med_traj, bl_med, bl_iqr_h, peak_j_global, peak_dir)

    # Magnitud del peak en unidades de IQR del baseline
    peak_magnitude_iqr = abs(peak_val - bl_med) / (bl_iqr_h + 1e-12) if not np.isnan(peak_val) else np.nan

    rows.append({
        "feature":              feat,
        "baseline_median":      round(bl_med, 4),
        "baseline_iqr":         round(bl_q75 - bl_q25, 4),
        "peak_value":           round(peak_val, 4) if not np.isnan(peak_val) else np.nan,
        "peak_timepoint_s":     int(peak_t) if not np.isnan(peak_t) else np.nan,
        "direction_of_peak":    peak_dir,
        "peak_magnitude_iqr":   round(peak_magnitude_iqr, 3) if not np.isnan(peak_magnitude_iqr) else np.nan,
        "return_to_baseline_s": int(return_t) if not np.isnan(return_t) else np.nan,
        "n_eventos":            n_valid,
        "expected_dir":         EXPECTED_DIR.get(feat, "?"),
        "dir_match":            peak_dir == EXPECTED_DIR.get(feat, "?"),
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_SUMMARY, index=False)
log.info(f"  Guardado: trajectory_summary.csv ({len(summary_df)} features)")
print("\n" + summary_df.to_string(index=False) + "\n")


# ── trajectory_diagnostic.md ─────────────────────────────────────────────────
log.info("Generando trajectory_diagnostic.md …")

# Contar cuántas features primarias tienen respuesta aguda visible (>0.5·IQR)
# y dirección consistente con BeatLabile
n_with_response = int(summary_df[
    (summary_df["peak_magnitude_iqr"] > 0.5) &
    (summary_df["dir_match"] == True)
].shape[0])
n_wrong_dir = int(summary_df[
    (summary_df["peak_magnitude_iqr"] > 0.5) &
    (summary_df["dir_match"] == False)
].shape[0])
n_flat = int(summary_df[
    summary_df["peak_magnitude_iqr"] <= 0.5
].shape[0])

# Veredicto
if n_with_response >= 3:
    verdict = "PROCEDER CON A+B"
    verdict_reason = (f"{n_with_response}/5 features primarias muestran respuesta "
                      f"aguda visible (>0.5·IQR) con dirección consistente con BeatLabile.")
elif n_with_response + n_wrong_dir >= 3:
    verdict = "PROCEDER CON A+B PERO REDISEÑAR"
    verdict_reason = (f"Hay respuestas ({n_with_response + n_wrong_dir} features), "
                      f"pero {n_wrong_dir} tienen dirección contraria a lo predicho. "
                      f"Revisar surrogate mapping.")
else:
    verdict = "ABANDONAR Q1, IR A Q2"
    verdict_reason = (f"Solo {n_with_response} features muestran respuesta aguda "
                      f"discernible y con dirección correcta. La anestesia bloquea "
                      f"la respuesta autonómica — hallazgo negativo legítimo.")

# Recomendación de ventana
peak_times = summary_df["peak_timepoint_s"].dropna()
return_times = summary_df["return_to_baseline_s"].dropna()

if len(peak_times) > 0:
    median_peak_t = int(np.nanmedian(peak_times))
    median_return_t = int(np.nanmedian(return_times)) if len(return_times) > 0 else "NaN"
    # Ventana sugerida
    post_end = min(int(median_return_t) + 30, 300) if not isinstance(median_return_t, str) else 300
    window_rec = f"Pre: [{BL_MIN//60}, {BL_MAX//60}] min  →  Post: [0, {post_end}s] ({post_end//60:.1f} min)"
else:
    median_peak_t   = "N/A"
    median_return_t = "N/A"
    window_rec = "No determinable (sin picos claros)"

# Tabla de features
feat_lines = []
for _, r in summary_df.iterrows():
    match_sym = "✓" if r["dir_match"] else "✗"
    mag_str   = f"{r['peak_magnitude_iqr']:.2f}×IQR" if not pd.isna(r['peak_magnitude_iqr']) else "N/A"
    pk_t      = f"{int(r['peak_timepoint_s'])}s" if not pd.isna(r['peak_timepoint_s']) else "N/A"
    ret_t     = f"{int(r['return_to_baseline_s'])}s" if not pd.isna(r['return_to_baseline_s']) else ">5min"

    # Interpretación automática
    if pd.isna(r['peak_magnitude_iqr']) or r['peak_magnitude_iqr'] <= 0.2:
        interp = "SIN respuesta detectable."
    elif r['peak_magnitude_iqr'] <= 0.5:
        interp = f"Cambio mínimo ({mag_str}) — por debajo del umbral de relevancia."
    else:
        dir_word = "aumento" if r['direction_of_peak'] == "+" else "descenso"
        match_word = "CONSISTENTE con BeatLabile" if r['dir_match'] else "CONTRARIO a BeatLabile"
        interp = (f"Respuesta visible: {dir_word} de {mag_str}, pico a {pk_t}, "
                  f"retorno en {ret_t}. Dirección {match_sym} {match_word}.")

    feat_lines.append(
        f"- **{r['feature']}** (esperado `{r['expected_dir']}`, observado `{r['direction_of_peak']}`): "
        + interp
    )

feat_text = "\n".join(feat_lines)

# Resumen de subcategorías omitidas
omitted_subcats = [sc for sc in subcats_all if sc not in subcats_valid]
omit_text = (f"Subcategorías omitidas por <{MIN_EVENTS_SUBCAT} eventos: "
             + (", ".join(omitted_subcats) if omitted_subcats else "ninguna"))

# Resumen de señales con pocos eventos
low_n_feats = summary_df[summary_df["n_eventos"] < 30][["feature", "n_eventos"]]
low_n_text = (
    "Ninguna" if len(low_n_feats) == 0
    else "; ".join(f"{r['feature']} (n={r['n_eventos']})" for _, r in low_n_feats.iterrows())
)

report_text = f"""# Trajectory Diagnostic — Q1
Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Objetivo
Determinar si la respuesta autonómica al estímulo doloroso bajo
propofol+remi+bloqueo es transitoria y si la ventana de 5 min la está diluyendo.

## Datos utilizados
- `features_long.parquet`: 83 692 ventanas 30s (paso 1s), 18 pacientes
- `event_windows.csv`: 52 eventos limpios (17 pacientes)
- Rejilla temporal: −5 a +5 min, paso 30s ({len(GRID)} timepoints/evento)
- Baseline definido: [−5, −2] min (índices {bl_idx[0]}…{bl_idx[-1]})

## Figura A — Trayectorias pooled
![trajectory_overall.png](figures/trajectory_overall.png)

*9 features (3×3). Líneas tenues = estímulos individuales. Línea gruesa = mediana.
Banda = IQR 25–75%. Línea punteada horizontal = mediana baseline. Discontinua vertical = t=0.*

## Figura B — Por tipo de estímulo
![trajectory_by_stimulus_type.png](figures/trajectory_by_stimulus_type.png)

*{omit_text}.*

## Figura C — Por grupo de bloqueo
![trajectory_by_group.png](figures/trajectory_by_group.png)

*Interescalénico (n=35 eventos) vs Supra+Axilar (n=17 eventos).*

## Interpretación por feature primaria

{feat_text}

### Notas de calidad
- Features con ≤30 eventos válidos: {low_n_text}
- brs_alpha_lf enmascarado cuando brs_valid=False
- hr_mean calculada como n_rr/30×60 (aprox., sin PLETH_HR disponible)

## Cuantificación del peak
| Feature | Baseline med | Peak | Peak t (s) | Magnitud (×IQR) | Retorno (s) | Dir obs | Dir exp | Match |
|---------|-------------|------|-----------|-----------------|-------------|---------|---------|-------|
""" + "\n".join(
    f"| {r['feature']} | {r['baseline_median']:.3f} | {r['peak_value']:.3f} | "
    f"{int(r['peak_timepoint_s']) if not pd.isna(r['peak_timepoint_s']) else 'N/A'} | "
    f"{r['peak_magnitude_iqr']:.2f} | "
    f"{int(r['return_to_baseline_s']) if not pd.isna(r['return_to_baseline_s']) else '>300'} | "
    f"{r['direction_of_peak']} | {r['expected_dir']} | {'✓' if r['dir_match'] else '✗'} |"
    for _, r in summary_df.iterrows()
) + f"""

## Recomendación de ventana
- Timepoint de peak mediano observado: **{median_peak_t}s**
- Retorno a baseline mediano: **{median_return_t}s**
- **Ventana sugerida**: {window_rec}
  > Comparado con la ventana actual de 5 min completos, esta ventana más estrecha
  > podría aumentar la relación señal/ruido en análisis A+B.

## Veredicto sobre A+B

> ### {verdict}
> {verdict_reason}

### Razonamiento:
- Features con respuesta visible (>0.5×IQR) y dirección correcta: **{n_with_response}/5**
- Features con respuesta visible pero dirección incorrecta: **{n_wrong_dir}/5**
- Features sin respuesta discernible: **{n_flat}/5**

---
*Fin del diagnóstico de trayectorias Q1*
"""

OUT_REPORT.write_text(report_text, encoding="utf-8")
log.info(f"  Guardado: trajectory_diagnostic.md")

log.info("=== ANÁLISIS COMPLETADO ===")
log.info(f"Veredicto: {verdict}")
log.info(f"  → {verdict_reason}")
