"""
q1_events.py
============
PASO 4 — Definición de ventanas de evento y control.

Para cada estímulo "limpio" (sin confounders ±5 min):
  - VENTANA EVENTO: [t_estimulo, t_estimulo + 5 min]
  - VENTANA CONTROL: quiescente de 5 min, muestreada aleatoriamente (seed=42)

Output: event_windows.csv con todas las features agregadas.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from q1_config import (
    DRUG_DELTA_EXCL,
    EVENT_WINDOW_S,
    EXCLUSION_BUFFER_S,
    GLOBAL_SEED,
    PARCIAL,
    PARCIAL_MAX_WINDOW_MIN,
    PPG_QUALITY_MIN_CONTROL,
    PRIMARY_COL_NAMES,
    PRIMARY_FEATURES,
    RESULTS_DIR,
    STIMULUS_SUBCATEGORIES,
    WINDOW_S,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aggregation functions over a set of 30s feature values
# ---------------------------------------------------------------------------

_AGG_FUNCS = {
    "mean":  np.nanmean,
    "std":   np.nanstd,
    "min":   np.nanmin,
    "max":   np.nanmax,
    "slope": lambda x: _slope(x),
}


def _slope(x: np.ndarray) -> float:
    """Pendiente de regresión lineal (no-NaN)."""
    x_c = x[np.isfinite(x)]
    if len(x_c) < 2:
        return np.nan
    t = np.arange(len(x_c), dtype=float)
    slope, *_ = np.polyfit(t, x_c, 1)
    return float(slope)


def aggregate_window(features_df: pd.DataFrame,
                     t_start: float, t_end: float,
                     patient_id: str) -> dict:
    """
    Agrega features de ventanas 30s que caen en [t_start, t_end].
    Retorna dict con {feature__agg: valor, ...}.
    """
    mask = (
        (features_df["patient_id"] == patient_id) &
        (features_df["t_window_start_s"] >= t_start) &
        (features_df["t_window_start_s"] < t_end)
    )
    sub = features_df[mask]
    n_sub = len(sub)

    result: dict = {
        "n_30s_subwindows": n_sub,
        "ppg_valid_pct":    float(sub["ppg_valid_pct"].mean()) if n_sub > 0 else np.nan,
    }

    if n_sub == 0:
        return result

    # Todas las combinaciones feature × agregación definidas en PRIMARY + EXPLORATORY
    all_features = set(
        feat for feat, _, _ in
        __import__("q1_config").PRIMARY_FEATURES +
        __import__("q1_config").EXPLORATORY_FEATURES
    )
    # Añadir BRS y PAI
    all_features.update(["brs_alpha_lf", "brs_coherence_max", "pai_mean", "pai_std"])

    for feat in all_features:
        if feat not in sub.columns:
            continue
        vals = sub[feat].values.astype(float)
        for agg_name, agg_fn in _AGG_FUNCS.items():
            col = f"{feat}__{agg_name}"
            try:
                result[col] = float(agg_fn(vals))
            except Exception:
                result[col] = np.nan

    return result


# ---------------------------------------------------------------------------
# Detección de confounders en ventana
# ---------------------------------------------------------------------------

def _has_stimuli_in_buffer(t_stim: float, stimulus_times: list[float],
                            buffer_s: float = EXCLUSION_BUFFER_S) -> bool:
    """¿Hay otro estímulo en [t_stim + 0, t_stim + buffer_s]?"""
    for ts in stimulus_times:
        if ts != t_stim and t_stim < ts < t_stim + buffer_s:
            return True
    return False


def _has_vasopresor_in_buffer(t_stim: float, drug_df: pd.DataFrame,
                               patient_id: str,
                               buffer_s: float = EXCLUSION_BUFFER_S) -> bool:
    """¿Hay bolus de vasopresor ±buffer alrededor de t_stim?"""
    if drug_df is None or drug_df.empty:
        return False
    sub = drug_df[drug_df["patient_id"] == patient_id]
    if sub.empty:
        return False
    lo, hi = t_stim - buffer_s, t_stim + buffer_s
    win = sub[(sub["t_s"] >= lo) & (sub["t_s"] <= hi)]
    # Detectar saltos en cumulativas de efedrina/fenilefrina
    for col in ["cumulative_efedrina_mg", "cumulative_fenilefrina_mcg"]:
        if col in win.columns:
            vals = win[col].values
            if len(vals) >= 2:
                delta = float(vals[-1]) - float(vals[0])
                if delta > 0:
                    return True
    return False


def _has_infusion_change_in_buffer(t_stim: float, ann_df: pd.DataFrame,
                                    patient_id: str,
                                    buffer_s: float = EXCLUSION_BUFFER_S,
                                    delta_thresh: float = DRUG_DELTA_EXCL) -> bool:
    """¿Hay cambio de infusión propofol/remi >0.5 en ±buffer?"""
    if ann_df is None or ann_df.empty:
        return False
    sub = ann_df[
        (ann_df["patient_id"] == patient_id) &
        (ann_df["category"] == "infusion_change")
    ]
    lo, hi = t_stim - buffer_s, t_stim + buffer_s
    win = sub[(sub["t_seconds"] >= lo) & (sub["t_seconds"] <= hi)]
    if win.empty:
        return False
    # Verificar si hay un cambio significativo
    for _, row in win.iterrows():
        if pd.notna(row.get("propo_target")) or pd.notna(row.get("remi_target")):
            # Comparar con el cambio anterior (si hay)
            prev = sub[sub["t_seconds"] < row["t_seconds"]].tail(1)
            if prev.empty:
                return True   # primer cambio → no sabemos baseline → excluir
            for col in ["propo_target", "remi_target"]:
                if pd.notna(row.get(col)) and pd.notna(prev.iloc[0].get(col)):
                    if abs(float(row[col]) - float(prev.iloc[0][col])) > delta_thresh:
                        return True
    return False


def _get_ag_window(ann_df: pd.DataFrame, patient_id: str,
                   recording_end_s: float = float("inf")) -> tuple[float, float]:
    """
    Retorna (t_AG, t_fin_AG) para el paciente.
    - Sin marcador AG   → t_ag  = 0.0  (inicio del archivo)
    - Sin fin_AG/fin_IQ → t_fin = recording_end_s (final del archivo)
    """
    sub = ann_df[ann_df["patient_id"] == patient_id]
    ag_rows  = sub[sub["subcategory"] == "AG"]["t_seconds"]
    fin_rows = sub[sub["subcategory"].isin(["fin_AG", "fin_IQ"])]["t_seconds"]
    t_ag  = float(ag_rows.min())  if len(ag_rows)  > 0 else 0.0
    t_fin = float(fin_rows.max()) if len(fin_rows) > 0 else recording_end_s
    return t_ag, t_fin


# ---------------------------------------------------------------------------
# Construcción de event_windows
# ---------------------------------------------------------------------------

def build_event_windows(
    ann_df:      pd.DataFrame,
    drug_df:     pd.DataFrame,
    features_df: pd.DataFrame,
    groups:      dict[str, str],   # {patient_id: group_label}
    patients:    list[str],
) -> pd.DataFrame:
    """
    Construye event_windows.csv con todas las ventanas de evento y control.
    """
    rng = random.Random(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    all_rows: list[dict] = []
    window_id = 0

    for pid in patients:
        log.info("PASO 4 — %s: definiendo ventanas", pid)
        group = groups.get(pid, "desconocido")
        is_partial = pid in PARCIAL

        # Fin de grabación: último t_window_start conocido + 30s
        feat_sub = features_df[features_df["patient_id"] == pid]["t_window_start_s"]
        recording_end_s = float(feat_sub.max()) + WINDOW_S if not feat_sub.empty else float("inf")

        # Timestamps AG (usa fin de grabación si no hay fin_AG explícito)
        t_ag, t_fin = _get_ag_window(ann_df, pid, recording_end_s=recording_end_s)
        log.info("%s: ventana AG = [%.0f s, %.0f s] (%.1f min)",
                 pid, t_ag, t_fin, (t_fin - t_ag) / 60)

        # Estímulos de este paciente
        stim_rows = ann_df[
            (ann_df["patient_id"] == pid) &
            (ann_df["category"] == "estimulo") &
            (ann_df["subcategory"].isin(STIMULUS_SUBCATEGORIES))
        ]
        if stim_rows.empty:
            log.warning("%s: sin estímulos dolorosos en anotaciones", pid)
            continue

        stimulus_times = list(stim_rows["t_seconds"])

        # ──────────────────────────────────────────────────────────────
        # VENTANAS EVENTO
        # ──────────────────────────────────────────────────────────────
        clean_events: list[dict] = []
        n_raw_events = len(stim_rows)

        for _, stim_row in stim_rows.iterrows():
            t_s = float(stim_row["t_seconds"])
            t_end = t_s + EVENT_WINDOW_S

            # Excluir si parcial y ventana ≥ PARCIAL_MAX_WINDOW_MIN
            if is_partial and (t_s + EVENT_WINDOW_S) > PARCIAL_MAX_WINDOW_MIN * 60:
                log.debug("%s (parcial): estímulo en t=%.0f excluido (ventana fuera de rango)", pid, t_s)
                continue

            # Excluir fuera de AG
            if t_s < t_ag or t_s > t_fin:
                log.debug("%s: estímulo en t=%.0f fuera de AG [%.0f, %.0f]", pid, t_s, t_ag, t_fin)
                continue

            # Excluir si hay otro estímulo en los 5 min posteriores
            if _has_stimuli_in_buffer(t_s, stimulus_times):
                log.debug("%s: estímulo en t=%.0f excluido por solape con otro estímulo", pid, t_s)
                continue

            # Excluir si hay vasopresor ±5 min
            if _has_vasopresor_in_buffer(t_s, drug_df, pid):
                log.debug("%s: estímulo en t=%.0f excluido por vasopresor ±5 min", pid, t_s)
                continue

            # Excluir si hay cambio infusión >0.5 ±5 min
            if _has_infusion_change_in_buffer(t_s, ann_df, pid):
                log.debug("%s: estímulo en t=%.0f excluido por cambio infusión ±5 min", pid, t_s)
                continue

            # Agregar features
            agg = aggregate_window(features_df, t_s, t_end, pid)
            if agg["n_30s_subwindows"] == 0:
                log.debug("%s: estímulo en t=%.0f sin ventanas 30s disponibles", pid, t_s)
                continue

            row = {
                "patient_id":       pid,
                "group":            group,
                "window_id":        window_id,
                "window_type":      "event",
                "event_subcategory": stim_row["subcategory"],
                "t_start_s":        t_s,
                "t_end_s":          t_end,
            }
            row.update(agg)
            clean_events.append(row)
            window_id += 1

        log.info("%s: %d estímulos raw → %d limpios", pid, n_raw_events, len(clean_events))
        all_rows.extend(clean_events)

        # ──────────────────────────────────────────────────────────────
        # VENTANAS CONTROL
        # ──────────────────────────────────────────────────────────────
        # Generar todas las ventanas quiescentes posibles de 5 min
        n_controls_needed = len(clean_events)
        if n_controls_needed == 0:
            continue

        # Candidatos: ventanas de 5 min con paso de 1 min entre AG y fin_AG
        candidate_controls: list[dict] = []
        t_iter = t_ag
        while t_iter + EVENT_WINDOW_S <= t_fin:
            t_c  = t_iter
            t_ce = t_c + EVENT_WINDOW_S

            # Verificar que NO solape con estímulos ±5 min
            near_stim = any(
                abs(t_c - ts) < EXCLUSION_BUFFER_S
                for ts in stimulus_times
            )
            if near_stim:
                t_iter += 60.0
                continue

            # Sin vasopresor ±5 min
            if _has_vasopresor_in_buffer(t_c, drug_df, pid):
                t_iter += 60.0
                continue

            # Sin cambio infusión >0.5 ±5 min
            if _has_infusion_change_in_buffer(t_c, ann_df, pid):
                t_iter += 60.0
                continue

            # PPG válido ≥ 80%
            agg_c = aggregate_window(features_df, t_c, t_ce, pid)
            if agg_c["n_30s_subwindows"] == 0:
                t_iter += 60.0
                continue
            ppg_pct = agg_c.get("ppg_valid_pct", 0.0)
            if pd.isna(ppg_pct) or ppg_pct < PPG_QUALITY_MIN_CONTROL:
                t_iter += 60.0
                continue

            candidate_controls.append({
                "patient_id":       pid,
                "group":            group,
                "window_id":        -1,   # asignar luego
                "window_type":      "control",
                "event_subcategory": "none",
                "t_start_s":        t_c,
                "t_end_s":          t_ce,
                **agg_c,
            })
            t_iter += 60.0   # paso de 1 min para candidatos

        # Muestrear n_controls_needed controles (o máximo disponible)
        n_avail = len(candidate_controls)
        if n_avail < n_controls_needed:
            log.warning(
                "%s: solo %d ventanas control disponibles para %d eventos",
                pid, n_avail, n_controls_needed
            )

        n_sample = min(n_avail, n_controls_needed)
        if n_sample > 0:
            sampled = rng.sample(candidate_controls, n_sample)
            for row in sampled:
                row["window_id"] = window_id
                window_id += 1
            all_rows.extend(sampled)

    if not all_rows:
        log.error("PASO 4: no se generó ninguna ventana")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Estadísticas de filtrado
    n_events   = (df["window_type"] == "event").sum()
    n_controls = (df["window_type"] == "control").sum()
    log.info("PASO 4: %d ventanas evento, %d ventanas control, %d pacientes",
             n_events, n_controls, df["patient_id"].nunique())

    return df


def save_paso4(event_windows_df: pd.DataFrame) -> None:
    """Guarda event_windows.csv en RESULTS_DIR."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "event_windows.csv"
    if not event_windows_df.empty:
        event_windows_df.to_csv(path, index=False)
        log.info("Guardado: %s (%d filas)", path, len(event_windows_df))
    else:
        log.warning("event_windows_df vacío; no se guardó")
