"""
q1_features.py
==============
PASO 3 — Computación de features autonómicas en ventanas deslizantes de 30 s.

Features por ventana:
  RR-derived:  hrv_sdnn, hrv_rmssd, hrv_pnn50
  PTT-derived: ptt_mean, ptt_std, ptt_cv, ptt_arv
  PPG-PAI:     pai_mean, pai_std
  BRS:         brs_alpha_lf, brs_coherence_max

Retorna DataFrame 'features_long.parquet' con columnas:
  patient_id, t_window_start_s, [feature_name], n_rr, n_ptt, n_pai,
  ppg_valid_pct, brs_valid
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from q1_config import (
    BRS_MIN_COHERENCE,
    GLOBAL_SEED,
    LF_BAND,
    RESULTS_DIR,
    WINDOW_S,
    WINDOW_STEP,
)
from q1_signals import PPGSeries, PTTSeries, RRSeries

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HRV features en ventana 30s
# ---------------------------------------------------------------------------

def _hrv_features(rr_ms: np.ndarray) -> dict:
    """Calcula SDNN, RMSSD, pNN50 sobre un array de RR en ms."""
    out = {"hrv_sdnn": np.nan, "hrv_rmssd": np.nan, "hrv_pnn50": np.nan, "n_rr": 0}
    rr = rr_ms[np.isfinite(rr_ms)]
    n  = len(rr)
    out["n_rr"] = n
    if n < 3:
        return out
    out["hrv_sdnn"]  = float(np.std(rr, ddof=1))
    diffs = np.diff(rr)
    out["hrv_rmssd"] = float(np.sqrt(np.mean(diffs ** 2)))
    out["hrv_pnn50"] = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100)
    return out


# ---------------------------------------------------------------------------
# PTT features en ventana 30s
# ---------------------------------------------------------------------------

def _ptt_features(ptt_ms: np.ndarray) -> dict:
    """Calcula ptt_mean, ptt_std, ptt_cv, ptt_arv."""
    out = {
        "ptt_mean": np.nan, "ptt_std": np.nan,
        "ptt_cv":   np.nan, "ptt_arv": np.nan,
        "n_ptt":    0,
    }
    p = ptt_ms[np.isfinite(ptt_ms)]
    n = len(p)
    out["n_ptt"] = n
    if n < 3:
        return out
    mu = float(np.mean(p))
    sd = float(np.std(p, ddof=1))
    out["ptt_mean"] = mu
    out["ptt_std"]  = sd
    out["ptt_cv"]   = sd / mu if mu > 0 else np.nan
    out["ptt_arv"]  = float(np.mean(np.abs(np.diff(p))))
    return out


# ---------------------------------------------------------------------------
# PPG-PAI features en ventana 30s
# ---------------------------------------------------------------------------

def _pai_features(pai: np.ndarray) -> dict:
    """Calcula pai_mean, pai_std."""
    out = {"pai_mean": np.nan, "pai_std": np.nan, "n_pai": 0}
    p = pai[np.isfinite(pai)]
    n = len(p)
    out["n_pai"] = n
    if n < 3:
        return out
    out["pai_mean"] = float(np.mean(p))
    out["pai_std"]  = float(np.std(p, ddof=1))
    return out


# ---------------------------------------------------------------------------
# BRS via cross-spectral PTT-RR
# ---------------------------------------------------------------------------

def _brs_features(rr_t: np.ndarray, rr_ms: np.ndarray,
                  ptt_t: np.ndarray, ptt_ms: np.ndarray,
                  srate_interp: float = 4.0) -> dict:
    """
    Estima BRS-alpha via coherencia PTT-RR en banda LF (0.04–0.15 Hz).

    brs_alpha_lf   = sqrt(PSD_RR / PSD_PTT) en LF, ponderado por coherencia ≥ 0.5
    brs_coherence_max = máxima coherencia en LF

    Si coherencia < 0.5 en toda la banda → NaN.
    Nota: se usa valor absoluto del coeficiente (ganancia de coupling).
    """
    out = {"brs_alpha_lf": np.nan, "brs_coherence_max": np.nan, "brs_valid": False}

    # Necesitamos ≥ 2 series alineadas
    if len(rr_ms) < 8 or len(ptt_ms) < 8:
        return out

    # Interpolar ambas series a frecuencia regular (4 Hz)
    t0 = max(rr_t[0], ptt_t[0])
    t1 = min(rr_t[-1], ptt_t[-1])
    if t1 - t0 < 10:
        return out

    t_interp = np.arange(t0, t1, 1.0 / srate_interp)
    if len(t_interp) < 16:
        return out

    rr_i  = np.interp(t_interp, rr_t,  rr_ms)
    ptt_i = np.interp(t_interp, ptt_t, ptt_ms)

    # Remover tendencia lineal
    rr_d  = rr_i  - np.polyval(np.polyfit(t_interp, rr_i,  1), t_interp)
    ptt_d = ptt_i - np.polyval(np.polyfit(t_interp, ptt_i, 1), t_interp)

    # Welch PSD
    nfft   = max(64, _next_power_of_2(len(t_interp) // 2))
    nperseg = min(nfft, len(t_interp))

    try:
        f, psd_rr  = sp_signal.welch(rr_d,  fs=srate_interp, nperseg=nperseg, nfft=nfft)
        _, psd_ptt = sp_signal.welch(ptt_d, fs=srate_interp, nperseg=nperseg, nfft=nfft)
        _, csd_rr_ptt = sp_signal.csd(rr_d, ptt_d, fs=srate_interp, nperseg=nperseg, nfft=nfft)
        coherence = np.abs(csd_rr_ptt) ** 2 / (psd_rr * psd_ptt + 1e-10)
    except Exception as exc:
        log.debug("BRS Welch falló: %s", exc)
        return out

    # Banda LF
    lf_mask = (f >= LF_BAND[0]) & (f <= LF_BAND[1])
    if not lf_mask.any():
        return out

    coh_lf   = coherence[lf_mask]
    psd_rr_lf  = psd_rr[lf_mask]
    psd_ptt_lf = psd_ptt[lf_mask]

    coh_max = float(np.max(coh_lf))
    out["brs_coherence_max"] = coh_max

    if coh_max < BRS_MIN_COHERENCE:
        return out   # brs_alpha_lf permanece NaN

    # Calcular alpha ponderado por coherencia ≥ 0.5
    coh_mask = coh_lf >= BRS_MIN_COHERENCE
    if coh_mask.sum() == 0:
        return out

    alpha_per_freq = np.sqrt(
        psd_rr_lf[coh_mask] / (psd_ptt_lf[coh_mask] + 1e-10)
    )
    weights = coh_lf[coh_mask]
    alpha_lf = float(np.average(np.abs(alpha_per_freq), weights=weights))

    out["brs_alpha_lf"] = alpha_lf
    out["brs_valid"]    = True
    return out


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Ventanas deslizantes: un DataFrame por paciente
# ---------------------------------------------------------------------------

def compute_windowed_features(
    patient_id: str,
    rr:  RRSeries,
    ppg: PPGSeries,
    ptt: PTTSeries,
    duration_s: float,
) -> pd.DataFrame:
    """
    Desliza ventanas de WINDOW_S con paso WINDOW_STEP sobre toda la grabación.
    Retorna DataFrame con una fila por ventana y todas las features.
    """
    rows: list[dict] = []

    t_max = duration_s - WINDOW_S
    if t_max <= 0:
        log.warning("%s: grabación muy corta (%.1f s) para ventanas de %d s",
                    patient_id, duration_s, WINDOW_S)
        return pd.DataFrame()

    # Precomputar arrays para indexación eficiente
    t_rr_arr   = rr.t_rr_s    if len(rr.t_rr_s)   > 0 else np.array([])
    t_ptt_arr  = ptt.t_s      if len(ptt.t_s)      > 0 else np.array([])
    t_pai_arr  = ppg.t_foot_s if len(ppg.t_foot_s) > 0 else np.array([])
    t_peak_arr = rr.t_peak_s  if len(rr.t_peak_s)  > 0 else np.array([])

    t_windows = np.arange(0, t_max, WINDOW_STEP, dtype=np.float64)

    for t_start in t_windows:
        t_end = t_start + WINDOW_S

        # ── RR en ventana ──
        mask_rr = (t_rr_arr >= t_start) & (t_rr_arr < t_end)
        rr_win  = rr.rr_ms[mask_rr] if len(rr.rr_ms) > 0 else np.array([])
        hrv = _hrv_features(rr_win)

        # Timestamps de peaks para BRS
        mask_pk = (t_peak_arr >= t_start) & (t_peak_arr < t_end)
        rr_t_win = t_rr_arr[mask_rr] if len(t_rr_arr) > 0 else np.array([])

        # ── PTT en ventana ──
        mask_ptt = (t_ptt_arr >= t_start) & (t_ptt_arr < t_end)
        ptt_win  = ptt.ptt_ms[mask_ptt] if len(ptt.ptt_ms) > 0 else np.array([])
        ptt_t_win = t_ptt_arr[mask_ptt]
        ptt_f = _ptt_features(ptt_win)

        # ── PPG-PAI en ventana ──
        mask_pai = (t_pai_arr >= t_start) & (t_pai_arr < t_end)
        pai_win  = ppg.pai[mask_pai] if len(ppg.pai) > 0 else np.array([])
        pai_f    = _pai_features(pai_win)

        # ── BRS en ventana ──
        brs_f = _brs_features(rr_t_win, rr_win, ptt_t_win, ptt_win)

        # ── PPG valid pct en ventana ──
        lo_idx = int(t_start)
        hi_idx = int(t_end) + 1
        ppg_vm = ppg.ppg_valid_mask
        if len(ppg_vm) > 0:
            vm_slice = ppg_vm[lo_idx:min(hi_idx, len(ppg_vm))]
            ppg_valid_pct = float(vm_slice.mean()) if len(vm_slice) > 0 else np.nan
        else:
            ppg_valid_pct = np.nan

        row = {
            "patient_id":       patient_id,
            "t_window_start_s": float(t_start),
            "ppg_valid_pct":    ppg_valid_pct,
        }
        row.update(hrv)
        row.update(ptt_f)
        row.update(pai_f)
        row.update(brs_f)
        rows.append(row)

    df = pd.DataFrame(rows)
    log.debug("%s: %d ventanas de 30s generadas", patient_id, len(df))
    return df


# ---------------------------------------------------------------------------
# Pipeline completo por paciente y guardado global
# ---------------------------------------------------------------------------

def run_paso3(signals_by_patient: dict) -> pd.DataFrame:
    """
    signals_by_patient: {pid: (RRSeries, PPGSeries, PTTSeries, duration_s)}
    Retorna DataFrame concatenado de features de todos los pacientes.
    """
    dfs: list[pd.DataFrame] = []
    for pid, (rr, ppg, ptt, dur) in signals_by_patient.items():
        log.info("PASO 3 — %s: computando features en ventanas deslizantes", pid)
        df = compute_windowed_features(pid, rr, ppg, ptt, dur)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        log.error("PASO 3: no se generó ninguna feature")
        return pd.DataFrame()

    features_df = pd.concat(dfs, ignore_index=True)
    log.info("PASO 3: %d filas totales (ventanas), %d columnas",
             len(features_df), len(features_df.columns))
    return features_df


def save_paso3(features_df: pd.DataFrame) -> None:
    """Guarda features_long.parquet en RESULTS_DIR."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "features_long.parquet"
    if not features_df.empty:
        features_df.to_parquet(path, index=False)
        log.info("Guardado: %s (%d filas)", path, len(features_df))
    else:
        log.warning("features_df vacío; no se guardó")
