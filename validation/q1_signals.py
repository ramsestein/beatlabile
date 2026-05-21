"""
q1_signals.py
=============
PASO 2 — Preprocesado de señales ECG, PPG y cálculo de PTT.

Provee:
  - process_ecg(vf, patient_id) → RRSeries  (namedtuple)
  - process_ppg(vf, patient_id) → PPGSeries (namedtuple)
  - compute_ptt(rr, ppg)        → PTTSeries (namedtuple)
  - sanity_check_plot(pid, vf)  → Path a PNG

Convenciones:
  - ECG a 500 Hz, Intellivue/ECG_II primario → ECG_V fallback
  - PPG a 125 Hz, Intellivue/PLETH
  - RR en ms, timestamps en segundos desde inicio
  - PTT en ms, etiquetado por timestamp del R-peak de referencia
"""

from __future__ import annotations

import logging
import warnings as pywarnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from q1_config import (
    ECG_SRATE,
    ECG_TRACKS,
    FIGURES_DIR,
    PPG_SRATE,
    PPG_TRACKS,
    PPG_REJECT_HIGH_PCT,
    PPG_REJECT_LOW_PCT,
    PPG_MOVING_MEDIAN_S,
    PTT_MAX_MS,
    PTT_MIN_MS,
    RR_MAX_MS,
    RR_MIN_MS,
)
from q1_load import get_continuous_signal, get_recording_duration_s, pick_track

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses de resultado
# ---------------------------------------------------------------------------

@dataclass
class RRSeries:
    patient_id:   str
    t_peak_s:     np.ndarray      # tiempo de cada R-peak (s desde inicio)
    rr_ms:        np.ndarray      # intervalo RR en ms (len = n_peaks - 1)
    t_rr_s:       np.ndarray      # timestamp del RR (inicio del intervalo)
    n_peaks_raw:  int = 0
    n_peaks_kept: int = 0
    ecg_track:    str = ""
    ecg_srate:    float = 0.0
    duration_s:   float = 0.0
    error:        str = ""


@dataclass
class PPGSeries:
    patient_id:     str
    t_foot_s:       np.ndarray    # tiempo de cada foot PPG (s)
    pai:            np.ndarray    # amplitud pico-valle por pulso (PPG-PAI)
    ppg_valid_mask: np.ndarray    # bool mask sobre señal 1Hz: PPG válido
    n_feet_raw:     int = 0
    n_feet_kept:    int = 0
    pct_valid:      float = 0.0
    ppg_track:      str = ""
    ppg_srate:      float = 0.0
    duration_s:     float = 0.0
    error:          str = ""


@dataclass
class PTTSeries:
    patient_id: str
    t_s:        np.ndarray   # timestamp del R-peak (s)
    ptt_ms:     np.ndarray   # PTT en ms
    n_raw:      int = 0
    n_kept:     int = 0
    error:      str = ""


# ---------------------------------------------------------------------------
# Procesado ECG → RR
# ---------------------------------------------------------------------------

def _ecg_detect_rpeaks(sig_f: np.ndarray, srate: float) -> np.ndarray:
    """Detecta R-peaks usando neurokit2 (neurokit → pantompkins1985 fallback).
    Retorna array de índices (vacío si falla)."""
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise ImportError("neurokit2 no instalado") from exc

    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        try:
            _, info = nk.ecg_peaks(sig_f, sampling_rate=int(srate), method="neurokit")
            return info["ECG_R_Peaks"]
        except Exception:
            pass
        try:
            _, info = nk.ecg_peaks(sig_f, sampling_rate=int(srate), method="pantompkins1985")
            return info["ECG_R_Peaks"]
        except Exception:
            return np.array([], dtype=int)


def _count_valid_rr(r_idx: np.ndarray, srate: float) -> int:
    """Cuenta RR intervals en rango fisiológico [RR_MIN_MS, RR_MAX_MS]."""
    if len(r_idx) < 2:
        return 0
    rr_ms = np.diff(r_idx.astype(np.float64)) / srate * 1000.0
    return int(((rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)).sum())


def _is_ecg_signal_valid(sig: np.ndarray) -> bool:
    """Devuelve True si la señal tiene variabilidad suficiente (no es flat/ruido DC)."""
    if len(sig) == 0:
        return False
    valid = sig[~np.isnan(sig)]
    if len(valid) < 100:
        return False
    return float(np.ptp(valid)) > 1e-4   # al menos 0.1 µV de rango


def process_ecg(vf, patient_id: str) -> RRSeries:
    """Detecta R-peaks con neurokit2 y construye serie RR filtrada.

    Selecciona automáticamente la derivación ECG con más intervalos RR
    fisiológicamente válidos, no simplemente la primera de la lista.
    """
    try:
        import neurokit2 as nk   # noqa: F401 (verificar disponibilidad)
    except ImportError as exc:
        raise ImportError("neurokit2 no instalado") from exc

    tracks = list(vf.trks.keys())

    # ── Selección de derivación: probar todas las candidatas, elegir la mejor ──
    candidate_leads = [t for t in ECG_TRACKS if t in tracks]
    if not candidate_leads:
        return RRSeries(patient_id=patient_id, t_peak_s=np.array([]),
                        rr_ms=np.array([]), t_rr_s=np.array([]),
                        error="No se encontró track ECG")

    best_track   = None
    best_r_idx   = np.array([], dtype=int)
    best_n_valid = -1
    best_sig     = None
    best_srate   = ECG_SRATE

    for lead in candidate_leads:
        sig, srate = get_continuous_signal(vf, lead)
        if sig is None or len(sig) == 0:
            continue
        if not _is_ecg_signal_valid(sig):
            log.debug("%s: %s — señal flat/sin variabilidad, omitiendo", patient_id, lead)
            continue
        sig_f = sig.astype(np.float64)
        nan_mask = np.isnan(sig_f)
        if nan_mask.any():
            idx = np.arange(len(sig_f))
            if (~nan_mask).sum() > 1:
                sig_f[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], sig_f[~nan_mask])
        r_idx = _ecg_detect_rpeaks(sig_f, srate)
        n_valid = _count_valid_rr(r_idx, srate)
        log.debug("%s: %s → %d peaks, %d RR válidos", patient_id, lead, len(r_idx), n_valid)
        if n_valid > best_n_valid:
            best_n_valid = n_valid
            best_track   = lead
            best_r_idx   = r_idx
            best_sig     = sig
            best_srate   = srate

    if best_track is None:
        return RRSeries(patient_id=patient_id, t_peak_s=np.array([]),
                        rr_ms=np.array([]), t_rr_s=np.array([]),
                        error="Todas las derivaciones ECG están flat o vacías")

    if best_track != candidate_leads[0]:
        log.warning("%s: derivación ECG seleccionada automáticamente: %s (lead por defecto %s tenía menos RR válidos)",
                    patient_id, best_track, candidate_leads[0])

    ecg_track  = best_track
    r_idx      = best_r_idx
    srate      = best_srate
    sig        = best_sig
    duration_s = len(sig) / srate

    n_raw = len(r_idx)
    if n_raw < 2:
        return RRSeries(patient_id=patient_id, t_peak_s=np.array([]),
                        rr_ms=np.array([]), t_rr_s=np.array([]),
                        ecg_track=ecg_track, n_peaks_raw=n_raw,
                        duration_s=duration_s, error="Muy pocos R-peaks")

    t_peaks = r_idx / srate   # segundos

    # Calcular RR
    rr_all = np.diff(t_peaks) * 1000.0   # ms

    # Filtrar artefactos/ectópicos
    valid = (rr_all >= RR_MIN_MS) & (rr_all <= RR_MAX_MS)
    rr_kept  = rr_all[valid]
    t_rr_s   = t_peaks[:-1][valid]
    t_kept   = t_peaks[np.concatenate([[True], valid]) & np.concatenate([valid, [True]])]
    # Reconstruir t_peaks válidos (conservadores: solo los peaks que forman RRs válidos)
    valid_peak_idx = np.unique(np.concatenate([
        np.where(np.concatenate([[True], valid]))[0],
        np.where(np.concatenate([valid, [True]]))[0],
    ]))
    t_peaks_kept = t_peaks[valid_peak_idx]

    log.debug("%s ECG: %d peaks raw → %d RR válidos (%.1f%%)",
              patient_id, n_raw, len(rr_kept),
              100 * len(rr_kept) / max(1, n_raw - 1))

    return RRSeries(
        patient_id=patient_id,
        t_peak_s=t_peaks,           # todos los peaks (para PTT)
        rr_ms=rr_kept,
        t_rr_s=t_rr_s,
        n_peaks_raw=n_raw,
        n_peaks_kept=len(rr_kept) + 1,
        ecg_track=ecg_track,
        ecg_srate=srate,
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# Procesado PPG → feet + PAI
# ---------------------------------------------------------------------------

def process_ppg(vf, patient_id: str) -> PPGSeries:
    """Detecta feet PPG y calcula PPG-PAI (amplitud pico-valle)."""
    try:
        import neurokit2 as nk
    except ImportError as exc:
        raise ImportError("neurokit2 no instalado") from exc

    tracks = list(vf.trks.keys())
    ppg_track = pick_track(tracks, PPG_TRACKS)
    if ppg_track is None:
        return PPGSeries(patient_id=patient_id, t_foot_s=np.array([]),
                         pai=np.array([]), ppg_valid_mask=np.array([], dtype=bool),
                         error="No se encontró track PPG/PLETH")

    sig, srate = get_continuous_signal(vf, ppg_track)
    if sig is None or len(sig) == 0:
        return PPGSeries(patient_id=patient_id, t_foot_s=np.array([]),
                         pai=np.array([]), ppg_valid_mask=np.array([], dtype=bool),
                         ppg_track=ppg_track, error="Señal PPG vacía")

    duration_s = len(sig) / srate
    sig_f = sig.astype(np.float64)

    # Máscara de validez (no-NaN, no clipping)
    nan_mask    = np.isnan(sig_f)
    valid_range = ~nan_mask
    # Detección de clipping
    if valid_range.any():
        vmax = float(np.nanmax(sig_f))
        vmin = float(np.nanmin(sig_f))
        clip_thresh = vmax - 0.001 * abs(vmax - vmin + 1e-6)
        clip_mask = sig_f >= clip_thresh
        valid_range &= ~clip_mask

    # Máscara 1 Hz sobre duración completa
    n_1hz = int(np.ceil(duration_s)) + 1
    valid_1hz = np.zeros(n_1hz, dtype=bool)
    valid_indices = np.where(valid_range)[0]
    if len(valid_indices):
        # Convertir índices de muestra a segundos y marcar en 1Hz
        t_valid_s = valid_indices / srate
        for tv in t_valid_s:
            idx = int(tv)
            if 0 <= idx < n_1hz:
                valid_1hz[idx] = True

    pct_valid = float(valid_range.mean()) * 100
    log.debug("%s PPG: %.1f%% válido", patient_id, pct_valid)

    # Rellenar NaN para detección (interpolación lineal)
    sig_filled = sig_f.copy()
    if nan_mask.any():
        idx = np.arange(len(sig_filled))
        if (~nan_mask).sum() > 1:
            sig_filled[nan_mask] = np.interp(
                idx[nan_mask], idx[~nan_mask], sig_filled[~nan_mask]
            )
        else:
            return PPGSeries(patient_id=patient_id, t_foot_s=np.array([]),
                             pai=np.array([]), ppg_valid_mask=valid_1hz,
                             ppg_track=ppg_track, pct_valid=pct_valid,
                             ppg_srate=srate, duration_s=duration_s,
                             error="PPG >99% NaN")

    # Detección de feet (valleys) con neurokit2
    with pywarnings.catch_warnings():
        pywarnings.simplefilter("ignore")
        try:
            ppg_signals, info = nk.ppg_process(sig_filled, sampling_rate=int(srate))
            # PPG_Peaks en neurokit2 son los picos sistólicos
            peaks_idx = info.get("PPG_Peaks", np.array([]))
            # Feet = mínimos entre picos consecutivos
            feet_idx = _find_feet_between_peaks(sig_filled, peaks_idx)
        except Exception as exc:
            log.warning("%s: nk.ppg_process falló (%s); usando detección por derivada", patient_id, exc)
            feet_idx = _find_feet_derivative(sig_filled, srate)
            peaks_idx = _find_peaks_derivative(sig_filled, srate)

    if len(feet_idx) < 2:
        return PPGSeries(patient_id=patient_id, t_foot_s=np.array([]),
                         pai=np.array([]), ppg_valid_mask=valid_1hz,
                         ppg_track=ppg_track, pct_valid=pct_valid,
                         ppg_srate=srate, duration_s=duration_s,
                         n_feet_raw=len(feet_idx),
                         error="Muy pocos feet PPG detectados")

    n_feet_raw = len(feet_idx)

    # Calcular PAI (amplitud pico-valle) para cada pulso
    pai_arr, t_feet_idx, valid_pulse = _compute_pai(sig_filled, feet_idx, peaks_idx)
    # Nota: NO convertir a segundos aquí — _filter_pai_outliers recibe índices
    # y devuelve t_feet / srate (segundos) en su return statement.

    # Filtrar pulsos con amplitud anómala (< 30% o > 300% mediana móvil)
    pai_arr, t_feet, valid_pulse = _filter_pai_outliers(
        pai_arr, t_feet_idx, valid_pulse, srate
    )

    n_kept = int(np.sum(valid_pulse))
    log.debug("%s PPG: %d feet raw → %d kept (%.1f%%)",
              patient_id, n_feet_raw, n_kept, 100 * n_kept / max(1, n_feet_raw))

    # Solo devolver pulsos válidos
    t_feet_kept = t_feet[valid_pulse]
    pai_kept    = pai_arr[valid_pulse]

    return PPGSeries(
        patient_id=patient_id,
        t_foot_s=t_feet_kept,
        pai=pai_kept,
        ppg_valid_mask=valid_1hz,
        n_feet_raw=n_feet_raw,
        n_feet_kept=n_kept,
        pct_valid=pct_valid,
        ppg_track=ppg_track,
        ppg_srate=srate,
        duration_s=duration_s,
    )


def _find_feet_between_peaks(sig: np.ndarray, peaks_idx: np.ndarray) -> np.ndarray:
    """Encuentra el mínimo entre picos consecutivos (feet PPG)."""
    if len(peaks_idx) < 2:
        return np.array([], dtype=int)
    feet = []
    for i in range(len(peaks_idx) - 1):
        seg = sig[peaks_idx[i]:peaks_idx[i + 1]]
        if len(seg) > 0:
            feet.append(int(peaks_idx[i]) + int(np.argmin(seg)))
    return np.array(feet, dtype=int)


def _find_feet_derivative(sig: np.ndarray, srate: float) -> np.ndarray:
    """Detección de feet por mínimos locales en señal PPG filtrada (fallback)."""
    from scipy import signal as sp
    # Filtro paso-bajo para suavizar
    b, a = sp.butter(2, 8.0 / (srate / 2), btype="low")
    sig_lp = sp.filtfilt(b, a, sig)
    # Mínimos locales con distancia mínima ~0.4s (40 bpm máximo)
    min_dist = max(1, int(0.4 * srate))
    peaks_neg, _ = sp.find_peaks(-sig_lp, distance=min_dist)
    return peaks_neg


def _find_peaks_derivative(sig: np.ndarray, srate: float) -> np.ndarray:
    """Picos sistólicos por máximos locales (fallback)."""
    from scipy import signal as sp
    b, a = sp.butter(2, 8.0 / (srate / 2), btype="low")
    sig_lp = sp.filtfilt(b, a, sig)
    min_dist = max(1, int(0.4 * srate))
    peaks, _ = sp.find_peaks(sig_lp, distance=min_dist)
    return peaks


def _compute_pai(sig: np.ndarray, feet_idx: np.ndarray,
                 peaks_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula amplitud pico-valle para cada pulso.
    Retorna (pai_arr, t_feet, valid_mask).
    """
    pai_arr   = np.full(len(feet_idx), np.nan)
    valid_arr = np.zeros(len(feet_idx), dtype=bool)

    for i, foot in enumerate(feet_idx):
        # Buscar el siguiente pico sistólico
        candidates = peaks_idx[peaks_idx > foot]
        if len(candidates) == 0:
            continue
        # Solo usar el primer pico dentro de 1s
        peak = candidates[0]
        dt_samples = peak - foot
        # Si es demasiado lejos (>1s a 125 Hz = 125 muestras), omitir
        if dt_samples > 250:
            continue
        amplitude = float(sig[peak]) - float(sig[foot])
        if amplitude > 0:
            pai_arr[i]   = amplitude
            valid_arr[i] = True

    t_feet = feet_idx.astype(np.float64)   # realmente son índices; se convierten con srate externamente
    return pai_arr, t_feet, valid_arr


def _filter_pai_outliers(pai: np.ndarray, t_feet: np.ndarray,
                          valid: np.ndarray, srate: float) -> tuple:
    """Filtra pulsos con PAI anómala (< 30% o > 300% mediana móvil de 30s)."""
    if valid.sum() < 5:
        return pai, t_feet, valid

    window_samples = int(PPG_MOVING_MEDIAN_S * srate)
    n = len(pai)
    new_valid = valid.copy()

    # Mediana móvil sobre valores válidos solamente
    valid_idx   = np.where(valid)[0]
    valid_pais  = pai[valid_idx]

    half = window_samples // 2
    for k, (orig_idx, p) in enumerate(zip(valid_idx, valid_pais)):
        # Ventana de vecinos válidos
        lo = max(0, k - 15)
        hi = min(len(valid_pais), k + 15)
        window_pais = valid_pais[lo:hi]
        med = float(np.nanmedian(window_pais))
        if med <= 0:
            continue
        ratio = p / med
        if ratio < PPG_REJECT_LOW_PCT or ratio > PPG_REJECT_HIGH_PCT:
            new_valid[orig_idx] = False

    pct_rejected = 1 - new_valid.sum() / max(1, valid.sum())
    if pct_rejected > 0:
        log.debug("PPG PAI filter: %.1f%% de pulsos rechazados por outlier", pct_rejected * 100)

    return pai, t_feet / srate, new_valid


# ---------------------------------------------------------------------------
# PTT = t(PPG_foot) − t(R_peak)
# ---------------------------------------------------------------------------

def compute_ptt(rr: RRSeries, ppg: PPGSeries) -> PTTSeries:
    """
    Para cada R-peak, busca el siguiente PPG foot en [PTT_MIN_MS, PTT_MAX_MS].
    Devuelve serie PTT en ms, etiquetada por timestamp del R-peak.
    """
    if len(rr.t_peak_s) == 0 or len(ppg.t_foot_s) == 0:
        reason = "sin R-peaks" if len(rr.t_peak_s) == 0 else "sin PPG feet"
        return PTTSeries(patient_id=rr.patient_id, t_s=np.array([]),
                         ptt_ms=np.array([]), error=f"PTT no computable: {reason}")

    t_peaks = rr.t_peak_s
    t_feet  = ppg.t_foot_s

    ptt_list: list[float] = []
    t_list:   list[float] = []

    # Para eficiencia con arreglos grandes, usar búsqueda binaria
    j_start = 0
    for t_r in t_peaks:
        lo_s = t_r + PTT_MIN_MS / 1000.0
        hi_s = t_r + PTT_MAX_MS / 1000.0
        # Buscar feet en ventana [lo_s, hi_s]
        candidates = t_feet[(t_feet >= lo_s) & (t_feet <= hi_s)]
        if len(candidates) == 0:
            continue
        # Tomar el más cercano al lower bound (onset del pulso)
        best_foot = candidates[0]
        ptt_ms    = (best_foot - t_r) * 1000.0
        if PTT_MIN_MS <= ptt_ms <= PTT_MAX_MS:
            ptt_list.append(ptt_ms)
            t_list.append(t_r)

    n_raw  = len(t_peaks)
    n_kept = len(ptt_list)
    log.debug("%s PTT: %d R-peaks → %d PTT válidos (%.1f%%)",
              rr.patient_id, n_raw, n_kept, 100 * n_kept / max(1, n_raw))

    return PTTSeries(
        patient_id=rr.patient_id,
        t_s=np.array(t_list),
        ptt_ms=np.array(ptt_list),
        n_raw=n_raw,
        n_kept=n_kept,
    )


# ---------------------------------------------------------------------------
# Sanity check visual (3 pacientes aleatorios)
# ---------------------------------------------------------------------------

def sanity_check_plot(patient_id: str, vf,
                      rr: RRSeries, ppg: PPGSeries, ptt: PTTSeries,
                      segment_start_s: float = 60.0,
                      segment_dur_s:   float = 60.0) -> Path:
    """
    Genera figura de sanity check: 60s de ECG + PPG con peaks/feet marcados + PTT.
    Guarda en FIGURES_DIR/sanity_check_<pid>.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"sanity_check_{patient_id}.png"

    tracks = list(vf.trks.keys())
    ecg_track = pick_track(tracks, ECG_TRACKS) or ""
    ppg_track = pick_track(tracks, PPG_TRACKS) or ""

    ecg_sig, ecg_sr = get_continuous_signal(vf, ecg_track) if ecg_track else (None, 0)
    ppg_sig, ppg_sr = get_continuous_signal(vf, ppg_track) if ppg_track else (None, 0)

    seg_start_ecg = int(segment_start_s * ecg_sr) if ecg_sr > 0 else 0
    seg_end_ecg   = seg_start_ecg + int(segment_dur_s * ecg_sr) if ecg_sr > 0 else 0
    seg_start_ppg = int(segment_start_s * ppg_sr) if ppg_sr > 0 else 0
    seg_end_ppg   = seg_start_ppg + int(segment_dur_s * ppg_sr) if ppg_sr > 0 else 0

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Sanity check — {patient_id}", fontsize=13)
    gs = GridSpec(3, 1, hspace=0.4)

    # ── ECG + R-peaks ──
    ax1 = fig.add_subplot(gs[0])
    if ecg_sig is not None and len(ecg_sig) > seg_end_ecg > seg_start_ecg:
        t_ecg = np.arange(seg_start_ecg, seg_end_ecg) / ecg_sr
        ax1.plot(t_ecg, ecg_sig[seg_start_ecg:seg_end_ecg],
                 lw=0.8, color="steelblue", label="ECG_II")
        # Marcar R-peaks en el segmento
        mask = (rr.t_peak_s >= segment_start_s) & (rr.t_peak_s <= segment_start_s + segment_dur_s)
        ax1.scatter(rr.t_peak_s[mask],
                    np.interp(rr.t_peak_s[mask],
                              np.arange(len(ecg_sig)) / ecg_sr,
                              ecg_sig.astype(float)),
                    color="red", s=30, zorder=5, label="R-peaks")
    ax1.set_ylabel("ECG (mV)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlim(segment_start_s, segment_start_s + segment_dur_s)

    # ── PPG + feet ──
    ax2 = fig.add_subplot(gs[1])
    if ppg_sig is not None and len(ppg_sig) > seg_end_ppg > seg_start_ppg:
        t_ppg = np.arange(seg_start_ppg, seg_end_ppg) / ppg_sr
        ax2.plot(t_ppg, ppg_sig[seg_start_ppg:seg_end_ppg],
                 lw=0.8, color="darkorange", label="PPG/PLETH")
        mask_feet = (ppg.t_foot_s >= segment_start_s) & (ppg.t_foot_s <= segment_start_s + segment_dur_s)
        if mask_feet.any():
            ax2.scatter(ppg.t_foot_s[mask_feet],
                        np.interp(ppg.t_foot_s[mask_feet],
                                  np.arange(len(ppg_sig)) / ppg_sr,
                                  ppg_sig.astype(float)),
                        color="green", s=30, zorder=5, label="PPG feet")
    ax2.set_ylabel("PPG (a.u.)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(segment_start_s, segment_start_s + segment_dur_s)

    # ── PTT ──
    ax3 = fig.add_subplot(gs[2])
    mask_ptt = (ptt.t_s >= segment_start_s) & (ptt.t_s <= segment_start_s + segment_dur_s)
    if mask_ptt.any():
        ax3.plot(ptt.t_s[mask_ptt], ptt.ptt_ms[mask_ptt],
                 "o-", ms=4, lw=1, color="purple", label="PTT (ms)")
    ax3.set_ylabel("PTT (ms)")
    ax3.set_xlabel("Tiempo (s desde inicio)")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xlim(segment_start_s, segment_start_s + segment_dur_s)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Sanity check guardado: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Evaluación de la calidad de detección
# ---------------------------------------------------------------------------

def check_detection_quality(ppg: PPGSeries, threshold: float = 0.20) -> bool:
    """
    Retorna True si la detección es aceptable (< 20% pulsos rechazados).
    threshold: fracción máxima de pulsos rechazados.
    """
    if ppg.n_feet_raw == 0:
        return False
    pct_rejected = 1 - ppg.n_feet_kept / ppg.n_feet_raw
    return pct_rejected <= threshold
