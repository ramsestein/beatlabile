#!/usr/bin/env python3
"""
audit_cohorte_hombro.py
=======================
Auditoría completa de la cohorte cirugía hombro / bloqueo interescalénico.
Genera un informe Markdown, dos CSV y un PNG de calidad de PPG.

Uso:
    python scripts/audit_cohorte_hombro.py

Salida (results/validation/):
    informe_auditoria_hombro.md
    inventario_pacientes.csv
    anotaciones_unicas.csv
    ppg_quality_histogram.png
"""

from __future__ import annotations

import os
import re
import sys
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "pacientes vital recorder"
OUT_DIR = REPO_ROOT / "results" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DEL INVESTIGADOR
# Asignación de grupos: paciente → True=bloqueo interescalénico, False=bloqueo supra+axilar
# Fuente: database general pacientes estudio hombro.xlsx (columna 'B. interescalenico')
# ---------------------------------------------------------------------------
GRUPO_BLOQUEO: dict[str, bool | None] = {
    "230393":   False,  # n=19 supra+axilar
    "397651":   False,  # n=05 supra+axilar
    "4214722":  True,   # n=09 interescalénico
    "4234018":  True,   # n=07 interescalénico (sin vital - VR apagado)
    "4247699":  True,   # n=03 interescalénico
    "4912692":  False,  # n=18 supra+axilar
    "5020549":  True,   # n=15 interescalénico
    "5362391":  True,   # n=02 interescalénico
    "5431482":  True,   # n=17 interescalénico
    "5582912":  True,   # n=12 interescalénico
    "5589679":  True,   # n=11 interescalénico
    "5684023":  True,   # n=01 interescalénico
    "70078466": False,  # n=13 supra+axilar (sin vital - archivo erróneo)
    "70288016": False,  # n=20 supra+axilar
    "70297385": True,   # n=04 interescalénico
    "70431992": False,  # n=21 supra+axilar
    "70436283": True,   # n=14 interescalénico
    "70551555": False,  # n=06 supra+axilar
    "70628874": False,  # n=16 supra+axilar
    "70767707": True,   # n=08 interescalénico (VR apagado a los 23 min)
    "720142":   True,   # n=10 interescalénico
}

# Señales esperadas (nombre parcial, sin prefijo de dispositivo)
PPG_CANDIDATES   = ["PLETH", "PPG", "SpO2_PLETH", "PLETHYSMOGRAPHY"]
NIBP_SBP_CANDS   = ["NIBP_SBP", "NIBP_SYS", "NBP_SBP", "NBP_SYS", "SBP"]
NIBP_DBP_CANDS   = ["NIBP_DBP", "NIBP_DIA", "NBP_DBP", "NBP_DIA", "DBP"]
NIBP_MBP_CANDS   = ["NIBP_MBP", "NIBP_MAP", "NIBP_MEAN", "NBP_MBP", "NBP_MAP", "MBP", "MAP"]
HR_CANDS         = ["ECG_HR", "PLETH_HR", "NIBP_HR", "HR", "HEART_RATE", "PULSE", "HEARTRATE"]
SPO2_CANDS       = ["PLETH_SAT_O2", "SpO2", "SPO2", "SAO2", "SaO2"]
BIS_CANDS        = ["EEG_BIS", "BIS", "BIS_EEG", "BIS_INDEX"]
RESP_CANDS       = ["RESP", "RR", "RESPIRATORY_RATE", "RESPIRATION"]
ETCO2_CANDS      = ["EtCO2", "ETCO2", "CO2", "EXPCO2"]

# Umbrales de calidad PPG
PPG_QUALITY_THRESHOLD = 0.70  # < 70% PPG válido → candidato a exclusión
NIBP_MAP_HYPOTENSION  = 55    # mmHg — umbral MAP para hipotensión

# Palabras clave de anotaciones de estímulos esperadas
STIMULI_KEYWORDS = [
    r"piel", r"trocar", r"ancl", r"sutura", r"fresa",
    r"incisi", r"corte", r"portal", r"cann", r"introduc",
    r"maniobr", r"inicio\s*cirug", r"fin\s*cirug", r"posici",
]
DRUG_KEYWORDS = [
    r"efedrin", r"fenilefrin", r"noradrenal", r"atropin",
    r"propofol", r"remifentanil", r"fentanil", r"sevoflur",
    r"ketamin", r"dexmedetomidin", r"neostigmin", r"sugamm",
    r"morfin", r"tramadol", r"bloque", r"interescal",
    r"ropivacain", r"bupivacain", r"lidocain",
]

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _norm_name(n: str) -> str:
    """Normalizar nombre de track: mayúsculas, quitar prefijo dispositivo."""
    return n.split("/")[-1].upper()


def _pick_track(track_names: list[str], candidates: list[str]) -> str | None:
    """Buscar candidato en lista de tracks (sin prefijo de dispositivo)."""
    upper_map = {_norm_name(t): t for t in track_names}
    for c in candidates:
        if c.upper() in upper_map:
            return upper_map[c.upper()]
    return None


def _is_stimulus(text: str) -> bool:
    t = text.lower()
    return any(re.search(kw, t) for kw in STIMULI_KEYWORDS)


def _is_drug(text: str) -> bool:
    t = text.lower()
    return any(re.search(kw, t) for kw in DRUG_KEYWORDS)


def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Extracción de pistas de un VitalFile
# ---------------------------------------------------------------------------

def _get_track_list(vf) -> list[dict]:
    """Retorna lista de dicts con metadatos de cada track."""
    result = []
    if not hasattr(vf, "trks"):
        return result
    for name, trk in vf.trks.items():
        srate   = float(getattr(trk, "srate", 0) or 0)
        ttype   = int(getattr(trk, "type",  0) or 0)
        unit    = str(getattr(trk, "unit",  "") or "")
        minval  = getattr(trk, "minval", None)
        maxval  = getattr(trk, "maxval", None)
        result.append({
            "name":   name,
            "srate":  srate,
            "type":   ttype,   # 1=wave, 2=numeric, 5=string/event
            "unit":   unit,
            "minval": minval,
            "maxval": maxval,
        })
    return result


def _recording_duration_s(vf) -> float:
    """Duración total de la grabación en segundos."""
    try:
        dtstart = getattr(vf, "dtstart", None)
        dtend   = getattr(vf, "dtend",   None)
        if dtstart and dtend:
            # dtstart/dtend son floats Unix timestamp en esta versión de vitaldb
            diff = float(dtend) - float(dtstart)
            if diff > 0:
                return diff
    except Exception:
        pass
    return 0.0


def _get_events(vf) -> list[dict]:
    """Extrae anotaciones de tipo string/evento del VitalFile.

    En vitaldb (version usada), trk.recs es una lista de dicts con claves 'dt' y 'val'.
    'dt' es Unix timestamp float; tiempo relativo = dt - vf.dtstart.
    """
    events = []
    if not hasattr(vf, "trks"):
        return events

    dtstart = float(getattr(vf, "dtstart", 0) or 0)

    # Buscar tracks de tipo string (type==5)
    for name, trk in vf.trks.items():
        ttype = int(getattr(trk, "type", 0) or 0)
        srate = float(getattr(trk, "srate", 0) or 0)
        if ttype != 5 and not (srate == 0 and any(
            kw in name.upper() for kw in ["EVENT", "NOTE", "CMD", "ANNOT", "MARK"]
        )):
            continue
        try:
            recs = getattr(trk, "recs", []) or []
            for rec in recs:
                # Los recs son dicts con claves 'dt' y 'val'
                if isinstance(rec, dict):
                    dt_abs = float(rec.get("dt", dtstart) or dtstart)
                    val    = rec.get("val", "")
                else:
                    dt_abs = float(getattr(rec, "dt", dtstart) or dtstart)
                    val    = getattr(rec, "val", "")
                if val is not None and str(val).strip():
                    events.append({
                        "time_s": dt_abs - dtstart,   # segundos desde inicio
                        "time_abs": dt_abs,
                        "text":   str(val).strip(),
                        "track":  name,
                    })
        except Exception:
            pass
    return events


def _get_numeric_events(vf, track_name: str) -> list[tuple[float, float]]:
    """
    Retorna lista de (tiempo_relativo_s, valor) para tracks numéricos de NIBP.

    Para Intellivue/NIBP_SYS y similares, la señal está almacenada a 1 Hz
    con forward-fill entre mediciones. Detectamos mediciones reales como
    los puntos donde el valor cambia respecto al anterior.
    """
    pairs = []
    if not hasattr(vf, "trks") or track_name not in vf.trks:
        return pairs

    dtstart = float(getattr(vf, "dtstart", 0) or 0)
    trk     = vf.trks[track_name]
    srate   = float(getattr(trk, "srate", 0) or 0)

    try:
        if srate > 0:
            # Track continuo (e.g. 1 Hz trend forward-filled)
            # Usar to_numpy y detectar cambios de valor = mediciones reales
            arr = vf.to_numpy(track_name, interval=1.0 / srate)
            if arr is None:
                return pairs
            arr = np.asarray(arr, dtype=np.float32).ravel()
            nan_mask = np.isnan(arr)
            if nan_mask.all():
                return pairs
            # Detectar cambios de valor (nueva medición NIBP)
            prev_val = np.nan
            for i, v in enumerate(arr):
                if not np.isnan(v) and v != prev_val:
                    pairs.append((float(i) / srate, float(v)))
                    prev_val = v
        else:
            # Track de evento puro (srate=0) — recs son dicts con 'dt' y 'val'
            recs = getattr(trk, "recs", []) or []
            for rec in recs:
                if isinstance(rec, dict):
                    t = rec.get("dt")
                    v = rec.get("val")
                else:
                    t = getattr(rec, "dt", None)
                    v = getattr(rec, "val", None)
                if t is not None and v is not None:
                    try:
                        pairs.append((float(t) - dtstart, float(v)))
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return pairs


def _get_continuous_signal(vf, track_name: str) -> tuple[np.ndarray | None, float]:
    """
    Carga señal continua como array numpy y retorna (signal, srate_efectivo).
    Retorna (None, 0) si falla.
    """
    if not hasattr(vf, "trks") or track_name not in vf.trks:
        return None, 0.0
    trk = vf.trks[track_name]
    srate = float(getattr(trk, "srate", 0) or 0)
    if srate == 0:
        return None, 0.0
    try:
        arr = vf.to_numpy(track_name, interval=1.0 / srate)
        if arr is None:
            return None, srate
        return np.asarray(arr, dtype=np.float32).ravel(), srate
    except Exception:
        pass
    # Alternativa: construir desde recs
    try:
        recs = getattr(trk, "recs", []) or []
        if recs:
            times  = np.array([float(r.dt) for r in recs], dtype=np.float64)
            values = np.array([float(r.val) if r.val is not None else np.nan
                               for r in recs], dtype=np.float32)
            return values, srate
    except Exception:
        pass
    return None, srate


# ---------------------------------------------------------------------------
# Análisis de calidad de PPG
# ---------------------------------------------------------------------------

@dataclass
class PPGQuality:
    patient_id: str = ""
    duration_total_s: float = 0.0
    duration_valid_s: float = 0.0
    pct_valid: float = 0.0
    pct_clipping: float = 0.0
    pct_flat: float = 0.0
    n_gaps_gt5s: int = 0
    total_gap_s: float = 0.0
    srate_nominal: float = 0.0
    srate_effective: float = 0.0
    snr_median_db: float = float("nan")
    snr_p10_db: float = float("nan")
    n_windows_30s: int = 0
    notes: list[str] = field(default_factory=list)
    error: str = ""


def _analyze_ppg(signal: np.ndarray, srate: float, patient_id: str) -> PPGQuality:
    """Análisis de calidad de señal PPG."""
    q = PPGQuality(patient_id=patient_id, srate_nominal=srate)
    n = len(signal)
    if n == 0:
        q.error = "señal vacía"
        return q

    q.duration_total_s = n / srate

    # --- Identificar NaN como ausencia de señal ---
    nan_mask = np.isnan(signal)
    valid_mask = ~nan_mask
    q.pct_valid = float(valid_mask.mean()) * 100
    q.duration_valid_s = q.duration_total_s * q.pct_valid / 100

    # --- Gaps > 5 s ---
    gap_min_samples = int(5 * srate)
    in_gap = False
    gap_start = 0
    gaps = []
    for i, is_nan in enumerate(nan_mask):
        if is_nan and not in_gap:
            in_gap = True
            gap_start = i
        elif not is_nan and in_gap:
            in_gap = False
            gap_len = i - gap_start
            if gap_len >= gap_min_samples:
                gaps.append(gap_len / srate)
    if in_gap:
        gap_len = n - gap_start
        if gap_len >= gap_min_samples:
            gaps.append(gap_len / srate)
    q.n_gaps_gt5s = len(gaps)
    q.total_gap_s = float(sum(gaps))

    # Trabajar solo con valores válidos
    valid_signal = signal[valid_mask].copy()
    if len(valid_signal) == 0:
        q.error = "100% NaN"
        return q

    # --- Clipping: samples en el percentil 99.9 o igual al máximo ---
    p999 = float(np.percentile(valid_signal, 99.9))
    vmax = float(np.max(valid_signal))
    vmin = float(np.min(valid_signal))
    clip_thresh = vmax - 0.001 * abs(vmax - vmin + 1e-6)
    pct_clip = float(np.sum(valid_signal >= clip_thresh)) / len(valid_signal) * 100
    q.pct_clipping = pct_clip

    # --- Segmentos planos (varianza local nula → desconexión) ---
    WIN_FLAT = int(2 * srate)  # ventana 2 s
    n_valid = len(valid_signal)
    n_flat = 0
    for start in range(0, n_valid - WIN_FLAT + 1, WIN_FLAT):
        seg = valid_signal[start:start + WIN_FLAT]
        if np.var(seg) < 1e-6:
            n_flat += WIN_FLAT
    q.pct_flat = (n_flat / n_valid * 100) if n_valid > 0 else 0.0

    # --- Frecuencia de muestreo efectiva ---
    # Estimada como n_valid_samples / duration_valid
    if q.duration_valid_s > 0:
        q.srate_effective = len(valid_signal) / q.duration_valid_s
    else:
        q.srate_effective = srate

    # --- SNR estimado en ventanas de 30 s ---
    # Método: potencia en banda cardíaca (0.5-4 Hz) / potencia residual
    WIN_SNR = int(30 * srate)
    snrs = []
    if len(valid_signal) >= WIN_SNR:
        try:
            # Diseño de filtros
            sos_band = sp_signal.butter(4,
                                        [0.5 / (srate / 2), 4.0 / (srate / 2)],
                                        btype="bandpass", output="sos")
            n_wins = len(valid_signal) // WIN_SNR
            q.n_windows_30s = n_wins
            for i in range(min(n_wins, 200)):  # máximo 200 ventanas
                seg = valid_signal[i * WIN_SNR:(i + 1) * WIN_SNR]
                if np.var(seg) < 1e-6:
                    continue
                # Normalizar
                seg = seg - np.mean(seg)
                cardiac = sp_signal.sosfiltfilt(sos_band, seg)
                residual = seg - cardiac
                p_cardiac  = float(np.var(cardiac))
                p_residual = float(np.var(residual))
                if p_residual > 0:
                    snrs.append(10 * np.log10((p_cardiac + 1e-10) / (p_residual + 1e-10)))
        except Exception as e:
            q.notes.append(f"SNR no calculable: {e}")

    if snrs:
        q.snr_median_db = float(np.median(snrs))
        q.snr_p10_db    = float(np.percentile(snrs, 10))

    return q


# ---------------------------------------------------------------------------
# Análisis de NIBP
# ---------------------------------------------------------------------------

@dataclass
class NIBPStats:
    patient_id: str = ""
    n_readings: int = 0
    mean_cycle_min: float = float("nan")
    median_cycle_min: float = float("nan")
    sbp_values: list[float] = field(default_factory=list)
    dbp_values: list[float] = field(default_factory=list)
    map_values: list[float] = field(default_factory=list)
    n_outliers_sbp: int = 0
    n_outliers_dbp: int = 0
    n_hypo_maps: int = 0  # lecturas MAP < 55
    timestamps_s: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""


def _analyze_nibp(vf, track_names: list[str], patient_id: str, dur_s: float) -> NIBPStats:
    """Caracterizar NIBP de un paciente."""
    stats = NIBPStats(patient_id=patient_id)

    sbp_trk = _pick_track(track_names, NIBP_SBP_CANDS)
    dbp_trk = _pick_track(track_names, NIBP_DBP_CANDS)
    map_trk = _pick_track(track_names, NIBP_MBP_CANDS)

    sbp_pairs = _get_numeric_events(vf, sbp_trk) if sbp_trk else []
    dbp_pairs = _get_numeric_events(vf, dbp_trk) if dbp_trk else []
    map_pairs = _get_numeric_events(vf, map_trk) if map_trk else []

    # Intentar también señal continua si no hay eventos
    if not sbp_pairs and sbp_trk:
        arr, sr = _get_continuous_signal(vf, sbp_trk)
        if arr is not None:
            valid = arr[~np.isnan(arr)]
            sbp_pairs = [(i / max(sr, 1), float(v)) for i, v in enumerate(valid)]

    # La referencia temporal para ciclos es SBP
    if sbp_pairs:
        times = [p[0] for p in sbp_pairs]
        stats.timestamps_s = times
        stats.n_readings = len(sbp_pairs)
        stats.sbp_values = [p[1] for p in sbp_pairs]
        if len(times) > 1:
            diffs = np.diff(sorted(times)) / 60.0
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                stats.mean_cycle_min   = float(np.mean(diffs))
                stats.median_cycle_min = float(np.median(diffs))
        # Outliers SBP
        sbp_arr = np.array(stats.sbp_values)
        stats.n_outliers_sbp = int(np.sum((sbp_arr < 60) | (sbp_arr > 220)))

    if dbp_pairs:
        stats.dbp_values = [p[1] for p in dbp_pairs]
        dbp_arr = np.array(stats.dbp_values)
        stats.n_outliers_dbp = int(np.sum((dbp_arr < 30) | (dbp_arr > 130)))

    if map_pairs:
        stats.map_values = [p[1] for p in map_pairs]
        map_arr = np.array(stats.map_values)
        stats.n_hypo_maps = int(np.sum(map_arr < NIBP_MAP_HYPOTENSION))
    elif stats.sbp_values and stats.dbp_values:
        # Estimar MAP si no existe directamente
        n = min(len(stats.sbp_values), len(stats.dbp_values))
        sbp = np.array(stats.sbp_values[:n])
        dbp = np.array(stats.dbp_values[:n])
        map_est = dbp + (sbp - dbp) / 3.0
        stats.map_values = map_est.tolist()
        stats.n_hypo_maps = int(np.sum(map_est < NIBP_MAP_HYPOTENSION))
        stats.notes.append("MAP estimado como DBP + (SBP-DBP)/3 (no disponible directamente)")

    if stats.n_readings == 0:
        stats.error = "Sin lecturas NIBP detectadas"

    return stats


# ---------------------------------------------------------------------------
# Inventario de señales para un track continuo
# ---------------------------------------------------------------------------

def _track_signal_inventory(vf, track_name: str, srate: float) -> dict:
    """Estadísticas básicas de un track continuo."""
    arr, sr = _get_continuous_signal(vf, track_name)
    if arr is None:
        return {"pct_valid": None, "vmin": None, "vmax": None,
                "p1": None, "p99": None, "dur_min": None}
    valid = arr[~np.isnan(arr)]
    pct_valid = 100.0 * len(valid) / len(arr) if len(arr) > 0 else 0.0
    dur_min   = len(arr) / max(sr, 1) / 60.0
    if len(valid) == 0:
        return {"pct_valid": 0, "vmin": None, "vmax": None,
                "p1": None, "p99": None, "dur_min": dur_min}
    return {
        "pct_valid": round(pct_valid, 1),
        "vmin":  round(float(np.min(valid)), 2),
        "vmax":  round(float(np.max(valid)), 2),
        "p1":    round(float(np.percentile(valid, 1)), 2),
        "p99":   round(float(np.percentile(valid, 99)), 2),
        "dur_min": round(dur_min, 1),
    }


# ---------------------------------------------------------------------------
# ASCII Gantt
# ---------------------------------------------------------------------------

def _gantt_ascii(dur_s: float, nibp_times: list[float],
                 annot_times: list[float], gap_periods: list[tuple[float, float]],
                 width: int = 60) -> str:
    """Genera barra Gantt ASCII para un paciente."""
    if dur_s <= 0:
        return "  [duración desconocida]"

    def _pos(t: float) -> int:
        return min(int(t / dur_s * width), width - 1)

    bar = ["-"] * width
    # Gaps de PPG → 'G'
    for g_start, g_end in gap_periods:
        for i in range(_pos(g_start), min(_pos(g_end) + 1, width)):
            bar[i] = "G"
    # NIBP → 'N'
    for t in nibp_times:
        p = _pos(t)
        if 0 <= p < width:
            bar[p] = "N"
    # Anotaciones → 'A'
    for t in annot_times:
        p = _pos(t)
        if 0 <= p < width:
            bar[p] = "A"

    bar_str = "".join(bar)
    dur_min = dur_s / 60.0
    t_marks = " " * width
    # Marcas de tiempo a 0%, 25%, 50%, 75%, 100%
    header = ""
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        pos = int(frac * (width - 1))
        label = f"{frac * dur_min:.0f}m"
        header = header.ljust(pos) + label
    header = header[:width]

    return (
        f"  |{bar_str}|\n"
        f"   {header}\n"
        f"  Leyenda: G=gap PPG  N=NIBP  A=anotación  -=sin dato"
    )


# ---------------------------------------------------------------------------
# Procesamiento principal de un paciente
# ---------------------------------------------------------------------------

@dataclass
class PatientRecord:
    patient_id: str
    folder: Path
    vital_files: list[Path] = field(default_factory=list)
    file_sizes_bytes: list[int] = field(default_factory=list)
    ambiguous_name: bool = False
    parse_error: str = ""

    # Señales
    track_list: list[dict] = field(default_factory=list)
    dur_s: float = 0.0

    # Calidad PPG
    ppg_quality: PPGQuality | None = None

    # NIBP
    nibp: NIBPStats | None = None

    # Anotaciones
    events: list[dict] = field(default_factory=list)

    # Grupo
    grupo_bloqueo: bool | None = None

    # Gaps PPG para Gantt
    ppg_gaps: list[tuple[float, float]] = field(default_factory=list)

    # Señales encontradas
    found_signals: dict[str, str] = field(default_factory=dict)  # tipo → track_name

    # Timestamp absoluto de inicio (para detección de duplicados)
    dtstart_unix: float = 0.0


def _process_patient(folder: Path) -> PatientRecord:
    """Procesa una carpeta de paciente y retorna su registro de auditoría."""
    patient_id = folder.name

    # Detectar si la carpeta tiene sufijo ' mal'
    rec = PatientRecord(
        patient_id=patient_id,
        folder=folder,
        grupo_bloqueo=GRUPO_BLOQUEO.get(patient_id.replace(" mal", "").strip()),
    )

    # 1. Inventario de archivos
    vital_files = list(folder.glob("*.vital"))
    rec.vital_files = sorted(vital_files)
    rec.file_sizes_bytes = [f.stat().st_size for f in rec.vital_files]

    if not vital_files:
        rec.parse_error = "Sin archivos .vital"
        return rec

    # Detectar nombre ambiguo (nombre de archivo ≠ ID de carpeta)
    folder_stem = patient_id.replace(" mal", "").strip()
    for vf_path in vital_files:
        stem = vf_path.stem
        if folder_stem not in stem and stem not in folder_stem:
            rec.ambiguous_name = True
            break

    # 2. Cargar VitalFile (usar el primer archivo si hay varios)
    vf = None
    for vf_path in vital_files:
        try:
            import vitaldb as _vdb
            vf = _vdb.VitalFile(str(vf_path))
            break
        except Exception as e:
            rec.parse_error = f"Error al abrir {vf_path.name}: {e}"

    if vf is None:
        return rec

    # 3. Inventario de tracks
    rec.track_list = _get_track_list(vf)
    rec.dur_s = _recording_duration_s(vf)
    rec.dtstart_unix = float(getattr(vf, "dtstart", 0) or 0)
    track_names = [t["name"] for t in rec.track_list]

    # Identificar señales esperadas
    for tipo, cands in [
        ("ppg",     PPG_CANDIDATES),
        ("nibp_sbp", NIBP_SBP_CANDS),
        ("nibp_dbp", NIBP_DBP_CANDS),
        ("nibp_map", NIBP_MBP_CANDS),
        ("hr",      HR_CANDS),
        ("spo2",    SPO2_CANDS),
        ("bis",     BIS_CANDS),
        ("resp",    RESP_CANDS),
        ("etco2",   ETCO2_CANDS),
    ]:
        found = _pick_track(track_names, cands)
        if found:
            rec.found_signals[tipo] = found

    # 4. Calidad PPG
    ppg_trk = rec.found_signals.get("ppg")
    if ppg_trk:
        ppg_signal, ppg_sr = _get_continuous_signal(vf, ppg_trk)
        if ppg_signal is not None and ppg_sr > 0:
            rec.ppg_quality = _analyze_ppg(ppg_signal, ppg_sr, patient_id)
            # Calcular gaps para Gantt
            if rec.ppg_quality:
                _extract_ppg_gaps(ppg_signal, ppg_sr, rec)
        else:
            rec.ppg_quality = PPGQuality(patient_id=patient_id, error="No se pudo cargar señal PPG")
    else:
        rec.ppg_quality = PPGQuality(patient_id=patient_id, error="Track PPG no encontrado")

    # 5. NIBP
    rec.nibp = _analyze_nibp(vf, track_names, patient_id, rec.dur_s)

    # 6. Anotaciones
    rec.events = _get_events(vf)

    return rec


def _extract_ppg_gaps(signal: np.ndarray, srate: float, rec: PatientRecord):
    """Extrae períodos de gap (NaN) en el PPG para el Gantt."""
    nan_mask = np.isnan(signal)
    in_gap = False
    gap_start = 0
    gap_min = int(5 * srate)
    for i, is_nan in enumerate(nan_mask):
        if is_nan and not in_gap:
            in_gap = True
            gap_start = i
        elif not is_nan and in_gap:
            in_gap = False
            if i - gap_start >= gap_min:
                rec.ppg_gaps.append((gap_start / srate, i / srate))
    if in_gap and len(signal) - gap_start >= gap_min:
        rec.ppg_gaps.append((gap_start / srate, len(signal) / srate))


# ---------------------------------------------------------------------------
# Tabla de factibilidad de features BeatLabile-like
# ---------------------------------------------------------------------------

FEATURE_FEASIBILITY = [
    {
        "Feature BeatLabile": "cv-PA-std",
        "Descripción": "Coef. variación de PA latido-a-latido",
        "Disponible directamente": "NO — requiere arteria invasiva",
        "Surrogate propuesto": "Variabilidad de amplitud de pulso PPG (PPG-PAV); o variabilidad inter-ciclo de NIBP (escala temporal diferente: ~5 min vs latido)",
        "Calidad del surrogate": "Baja-Moderada — proxy indirecto, escala temporal incomparable",
    },
    {
        "Feature BeatLabile": "brs-min",
        "Descripción": "Baroreflex sensitivity (método secuencial mínimo)",
        "Disponible directamente": "NO — requiere PA sistólica latido-a-latido",
        "Surrogate propuesto": "Cross-spectral coherence PPG-RR (PPG-BRS); o PTT variability vs RR si hay ECG + PPG",
        "Calidad del surrogate": "Moderada — validado en literatura pero con limitaciones en perioperatorio",
    },
    {
        "Feature BeatLabile": "HRV-SDNN",
        "Descripción": "Desviación estándar intervalos NN",
        "Disponible directamente": "COMPUTABLE vía PRV (pulse rate variability) desde peaks PPG",
        "Surrogate propuesto": "PRV-SDNN: SDNN de intervalos inter-pulso PPG",
        "Calidad del surrogate": "Moderada-Alta — PRV ≈ HRV en condiciones estables; sesgo en arritmias o artefactos PPG",
    },
    {
        "Feature BeatLabile": "HRV-RMSSD",
        "Descripción": "Raíz cuadrática media de diferencias sucesivas NN",
        "Disponible directamente": "COMPUTABLE vía PRV",
        "Surrogate propuesto": "PRV-RMSSD",
        "Calidad del surrogate": "Moderada-Alta — misma caveat que PRV-SDNN",
    },
    {
        "Feature BeatLabile": "HRV-pNN50",
        "Descripción": "% intervalos con diferencia > 50 ms",
        "Disponible directamente": "COMPUTABLE vía PRV",
        "Surrogate propuesto": "PRV-pNN50",
        "Calidad del surrogate": "Moderada — sensible a calidad del detector de peaks PPG",
    },
    {
        "Feature BeatLabile": "RSA",
        "Descripción": "Arritmia sinusal respiratoria (potencia HF de HRV)",
        "Disponible directamente": "COMPUTABLE desde PRV si banda respiratoria identificable",
        "Surrogate propuesto": "Potencia HF (0.15-0.4 Hz) de PRV; o coherencia PPG-respiro si hay capnografía/resp",
        "Calidad del surrogate": "Moderada — anestesia y bloqueantes reducen RSA; útil como marcador de efecto",
    },
    {
        "Feature BeatLabile": "ARV de PA",
        "Descripción": "Average real variability de presión arterial",
        "Disponible directamente": "COMPUTABLE macroscópicamente desde NIBP (escala 5 min, no latido-a-latido)",
        "Surrogate propuesto": "ARV-NIBP: suma |ΔMAP| entre lecturas consecutivas de NIBP / duración",
        "Calidad del surrogate": "Baja — ARV original es latido-a-latido; versión NIBP captura solo variabilidad lenta",
    },
    {
        "Feature BeatLabile": "PPG-PAI (Perfusion Amplitude Index)",
        "Descripción": "No en BeatLabile original — específica de PPG",
        "Disponible directamente": "COMPUTABLE directamente — amplitud pico-valle PPG",
        "Surrogate propuesto": "Nativo PPG — no requiere surrogate",
        "Calidad del surrogate": "Alta para señal PPG de calidad — marcador de vasoconstricción/vasodilatación",
    },
    {
        "Feature BeatLabile": "PRV-LF/HF ratio",
        "Descripción": "No en BeatLabile original — balance simpático-vagal",
        "Disponible directamente": "COMPUTABLE desde PRV",
        "Surrogate propuesto": "Ratio potencia LF (0.04-0.15 Hz) / HF (0.15-0.4 Hz) de PRV",
        "Calidad del surrogate": "Baja-Moderada — interpretación LF/HF muy debatida; anestesia confunde",
    },
]


# ---------------------------------------------------------------------------
# Generación del informe Markdown
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list]) -> str:
    """Genera tabla Markdown."""
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
               for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    header = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |")
    return "\n".join(lines)


def generate_report(patient_records: list[PatientRecord]) -> str:
    """Genera el informe Markdown completo."""
    lines = []
    L = lines.append

    L("# Informe de auditoría — Cohorte hombro/bloqueo")
    L(f"\n_Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    # =====================================================================
    # 1. Resumen ejecutivo
    # =====================================================================
    L("## 1. Resumen ejecutivo\n")

    n_total = len(patient_records)
    n_con_vital = sum(1 for p in patient_records if p.vital_files)
    n_sin_vital = n_total - n_con_vital
    n_ppg_ok    = sum(1 for p in patient_records
                      if p.ppg_quality and not p.ppg_quality.error
                      and p.ppg_quality.pct_valid >= PPG_QUALITY_THRESHOLD * 100)
    n_ppg_low   = sum(1 for p in patient_records
                      if p.ppg_quality and not p.ppg_quality.error
                      and p.ppg_quality.pct_valid < PPG_QUALITY_THRESHOLD * 100)
    n_ppg_err   = sum(1 for p in patient_records
                      if p.ppg_quality and p.ppg_quality.error)
    n_nibp_ok   = sum(1 for p in patient_records if p.nibp and p.nibp.n_readings > 10)
    n_con_annot = sum(1 for p in patient_records if p.events)
    n_grupo_asig = sum(1 for p in patient_records if p.grupo_bloqueo is not None)
    n_hypo_pat  = sum(1 for p in patient_records
                      if p.nibp and p.nibp.n_hypo_maps > 0)
    n_ambiguous = sum(1 for p in patient_records if p.ambiguous_name)

    # Detección de duplicados para resumen
    from collections import defaultdict as _dd
    _dt_groups: dict[float, list[str]] = _dd(list)
    for p in patient_records:
        if p.dtstart_unix > 0:
            _dt_groups[p.dtstart_unix].append(p.patient_id)
    n_dup_groups = sum(1 for g in _dt_groups.values() if len(g) > 1)
    n_dup_pats   = sum(len(g) for g in _dt_groups.values() if len(g) > 1)

    # Veredicto Q1/Q2/Q3
    verdict_q1 = "INCIERTO" if n_ppg_ok < 15 else ("POSIBLE" if n_nibp_ok > 15 else "LIMITADO")
    verdict_q2 = "POSIBLE" if n_con_annot > 15 else "LIMITADO"
    verdict_q3 = "POSIBLE" if n_ppg_ok >= 12 and n_nibp_ok >= 12 and n_grupo_asig == n_total else "INCIERTO"

    L(f"""**Cohorte total analizada**: {n_total} carpetas de paciente.
**Archivos .vital encontrados**: {n_con_vital} pacientes con ≥1 archivo; {n_sin_vital} sin ningún .vital.
**Calidad PPG**: {n_ppg_ok} pacientes con ≥70% señal válida; {n_ppg_low} con calidad baja; {n_ppg_err} con error de carga.
**NIBP disponible**: {n_nibp_ok} pacientes con >10 lecturas.
**Anotaciones**: {n_con_annot} pacientes con eventos detectados en el .vital.
**Asignación de grupo**: {n_grupo_asig}/{n_total} pacientes asignados ({sum(1 for p in patient_records if p.grupo_bloqueo is True)} interescalénico / {sum(1 for p in patient_records if p.grupo_bloqueo is False)} supra+axilar).
**Pacientes con nombre de archivo ambiguo**: {n_ambiguous}.
**Pacientes con ≥1 lectura MAP <55 mmHg**: {n_hypo_pat}.
**Grupos de grabaciones duplicadas (mismo dtstart)**: {n_dup_groups} ({n_dup_pats} carpetas afectadas).

### Veredicto por pregunta pre-especificada

| Pregunta | Veredicto preliminar | Caveat principal |
|----------|---------------------|-----------------|
| **Q1** — Features autonómicas PPG-derived discriminan pre-hipotensión | **{verdict_q1}** | NIBP intermitente limita definición de "pre-hipotensión"; n=20 da potencia muy baja |
| **Q2** — Firma autonómica alrededor de estímulos dolorosos | **{verdict_q2}** | Depende de calidad de anotaciones en .vital; verificar timing exacto |
| **Q3** — Perfil hemodinámico interescalénico vs supra+axilar | **{verdict_q3}** | n pequeña; ambos grupos tienen bloqueo regional — diferencia de efecto simpaticolítico puede ser menor que bloqueo vs sin-bloqueo |

> ⚠️ **Caveats críticos**: (1) Sin línea arterial, las features de BeatLabile original no son computables — solo surrogates PPG. (2) NIBP intermitente impide definir "hipotensión sostenida" con el mismo criterio que BeatLabile; hay que redefinir operativamente. (3) n=21 ofrece potencia estadística muy limitada para análisis multivariante. (4) Todos los pacientes tienen algún tipo de bloqueo regional; la comparación es **interescalénico vs supra+axilar**, no bloqueo vs sin-bloqueo.
""")

    # =====================================================================
    # 2. Inventario de archivos
    # =====================================================================
    L("## 2. Inventario de archivos\n")

    inv_rows = []
    for p in patient_records:
        n_vitals   = len(p.vital_files)
        total_size = sum(p.file_sizes_bytes)
        filenames  = ", ".join(f.name for f in p.vital_files) if p.vital_files else "—"
        ambig      = "⚠️ sí" if p.ambiguous_name else "no"
        is_mal     = "⚠️ carpeta marcada 'mal'" if "mal" in p.patient_id.lower() else ""
        grupo      = ("interescalenico" if p.grupo_bloqueo is True
                      else ("supra+axilar" if p.grupo_bloqueo is False else "DESCONOCIDO"))
        notes      = p.parse_error or is_mal or ""
        inv_rows.append([
            p.patient_id,
            n_vitals,
            _human_bytes(total_size),
            filenames,
            ambig,
            grupo,
            notes,
        ])

    L(_md_table(
        ["Paciente", "#vitals", "Tamaño total", "Archivo(s)", "Nombre ambiguo", "Grupo", "Notas"],
        inv_rows,
    ))

    # Totales
    total_size_all = sum(sum(p.file_sizes_bytes) for p in patient_records)
    total_vitals   = sum(len(p.vital_files) for p in patient_records)
    L(f"\n**Total archivos .vital**: {total_vitals} | **Tamaño total**: {_human_bytes(total_size_all)}")
    if any(len(p.vital_files) > 1 for p in patient_records):
        multi = [p.patient_id for p in patient_records if len(p.vital_files) > 1]
        L(f"\n**Pacientes con múltiples .vital**: {', '.join(multi)}")
    if n_sin_vital > 0:
        no_vital = [p.patient_id for p in patient_records if not p.vital_files]
        L(f"\n**Pacientes SIN archivo .vital**: {', '.join(no_vital)}")
    if n_ambiguous > 0:
        ambig_pats = [p.patient_id for p in patient_records if p.ambiguous_name]
        L(f"\n**Pacientes con nombre de archivo ambiguo**: {', '.join(ambig_pats)}")

    # Detección de grabaciones duplicadas (mismo dtstart_unix)
    from collections import defaultdict
    dt_groups: dict[float, list[str]] = defaultdict(list)
    for p in patient_records:
        if p.dtstart_unix > 0:
            dt_groups[p.dtstart_unix].append(p.patient_id)
    dup_groups = {dt: pids for dt, pids in dt_groups.items() if len(pids) > 1}
    if dup_groups:
        L("\n> 🚨 **ALERTA — GRABACIONES DUPLICADAS DETECTADAS**:")
        for dt, pids in dup_groups.items():
            import datetime
            dt_str = datetime.datetime.utcfromtimestamp(dt).strftime("%Y-%m-%d %H:%M UTC")
            L(f"> - Mismo `dtstart` ({dt_str}): carpetas **{', '.join(pids)}** → el archivo .vital es idéntico.")
        L("> Antes de cualquier análisis, verificar cuál es el identificador de paciente correcto y eliminar los duplicados.")

    # =====================================================================
    # 3. Inventario de señales
    # =====================================================================
    L("\n## 3. Inventario de señales\n")

    # Resumen por tipo de señal
    sig_summary = {}
    for p in patient_records:
        for tipo in ["ppg", "nibp_sbp", "nibp_dbp", "nibp_map", "hr", "spo2", "bis", "resp", "etco2"]:
            sig_summary.setdefault(tipo, 0)
            if tipo in p.found_signals:
                sig_summary[tipo] += 1

    L("### 3.1 Señales esperadas — disponibilidad por paciente\n")
    sig_rows = []
    for tipo, label in [
        ("ppg", "PLETH/PPG (500 Hz)"),
        ("nibp_sbp", "NIBP_SBP"),
        ("nibp_dbp", "NIBP_DBP"),
        ("nibp_map", "NIBP_MAP/MBP"),
        ("hr", "HR continua"),
        ("spo2", "SpO2"),
        ("bis", "BIS"),
        ("resp", "Resp/RR"),
        ("etco2", "EtCO2"),
    ]:
        n_found = sig_summary.get(tipo, 0)
        pct     = 100 * n_found / n_total if n_total > 0 else 0
        status  = "✅" if pct == 100 else ("⚠️" if pct >= 50 else "❌")
        sig_rows.append([label, f"{n_found}/{n_total}", f"{pct:.0f}%", status])

    L(_md_table(["Señal esperada", "Pacientes con señal", "% cohorte", "Estado"], sig_rows))

    L("\n### 3.2 Tracks disponibles por paciente\n")
    for p in patient_records:
        if not p.track_list:
            L(f"**{p.patient_id}**: sin tracks (error de carga o sin .vital)\n")
            continue
        dur_min = p.dur_s / 60.0
        L(f"**{p.patient_id}** — duración total: {dur_min:.1f} min | tracks: {len(p.track_list)}\n")
        trk_rows = []
        for trk in sorted(p.track_list, key=lambda x: x["name"]):
            tipo_str = {1: "onda continua", 2: "numérico", 5: "evento/string"}.get(trk["type"], f"tipo {trk['type']}")
            sr_str = f"{trk['srate']:.0f} Hz" if trk["srate"] > 0 else "evento"
            mn = f"{trk['minval']:.1f}" if trk["minval"] is not None else "—"
            mx = f"{trk['maxval']:.1f}" if trk["maxval"] is not None else "—"
            trk_rows.append([trk["name"], tipo_str, sr_str, trk["unit"], mn, mx])
        L(_md_table(["Track", "Tipo", "Srate", "Unidad", "Min", "Max"], trk_rows))
        L("")

    # =====================================================================
    # 4. Calidad de PPG
    # =====================================================================
    L("## 4. Calidad de PPG\n")

    ppg_rows = []
    for p in patient_records:
        q = p.ppg_quality
        if q is None:
            ppg_rows.append([p.patient_id, "—", "—", "—", "—", "—", "—", "—", "Sin análisis"])
            continue
        if q.error:
            ppg_rows.append([p.patient_id, "—", "—", "—", "—", "—", "—", "—", f"ERROR: {q.error}"])
            continue
        flag = "⚠️ EXCLUIR" if q.pct_valid < PPG_QUALITY_THRESHOLD * 100 else ""
        ppg_rows.append([
            p.patient_id,
            f"{q.duration_total_s/60:.1f}",
            f"{q.pct_valid:.1f}%",
            f"{q.pct_clipping:.2f}%",
            f"{q.pct_flat:.2f}%",
            str(q.n_gaps_gt5s),
            f"{q.total_gap_s/60:.1f}",
            f"{q.snr_median_db:.1f}" if not np.isnan(q.snr_median_db) else "—",
            flag,
        ])

    L(_md_table(
        ["Paciente", "Dur total (min)", "% válido", "% clipping", "% plano",
         "#gaps>5s", "Total gap (min)", "SNR mediana (dB)", "Flag"],
        ppg_rows,
    ))

    L(f"\n> Umbral de exclusión: <{PPG_QUALITY_THRESHOLD*100:.0f}% PPG válido.")
    L("> **Método SNR**: potencia en banda cardíaca (0.5–4 Hz, Butterworth 4º orden) / potencia residual, en ventanas de 30 s. Mediana sobre todas las ventanas válidas.")
    excluded_ppg = [p.patient_id for p in patient_records
                    if p.ppg_quality and not p.ppg_quality.error
                    and p.ppg_quality.pct_valid < PPG_QUALITY_THRESHOLD * 100]
    if excluded_ppg:
        L(f"\n**Candidatos a exclusión por calidad PPG**: {', '.join(excluded_ppg)}")
    L("\n> Ver figura `ppg_quality_histogram.png` en el directorio de resultados.")

    # =====================================================================
    # 5. Caracterización de NIBP
    # =====================================================================
    L("\n## 5. Caracterización de NIBP\n")

    L("### 5.1 Estadísticas por paciente\n")
    nibp_rows = []
    for p in patient_records:
        n = p.nibp
        if n is None:
            nibp_rows.append([p.patient_id] + ["—"] * 10)
            continue
        sbp_str = f"{np.mean(n.sbp_values):.0f}±{np.std(n.sbp_values):.0f}" if n.sbp_values else "—"
        dbp_str = f"{np.mean(n.dbp_values):.0f}±{np.std(n.dbp_values):.0f}" if n.dbp_values else "—"
        map_str = f"{np.mean(n.map_values):.0f}±{np.std(n.map_values):.0f}" if n.map_values else "—"
        sbp_range = (f"[{min(n.sbp_values):.0f}-{max(n.sbp_values):.0f}]"
                     if n.sbp_values else "—")
        nibp_rows.append([
            p.patient_id,
            str(n.n_readings),
            f"{n.mean_cycle_min:.1f}" if not np.isnan(n.mean_cycle_min) else "—",
            f"{n.median_cycle_min:.1f}" if not np.isnan(n.median_cycle_min) else "—",
            sbp_str, sbp_range,
            dbp_str, map_str,
            str(n.n_outliers_sbp), str(n.n_outliers_dbp),
            str(n.n_hypo_maps),
        ])

    L(_md_table(
        ["Paciente", "#lecturas", "Ciclo medio (min)", "Ciclo mediana (min)",
         "SBP media±SD", "SBP rango",
         "DBP media±SD", "MAP media±SD",
         "#outliers SBP", "#outliers DBP", "#MAP<55"],
        nibp_rows,
    ))

    L("""
### 5.2 Criterio operativo para "hipotensión" con NIBP intermitente

Con NIBP cada ~5 min (estimado) no se puede verificar "sostenido ≥3 min" como en BeatLabile.
**Propuesta de criterio operativo**:
- **Caso A (laxo)**: ≥1 lectura MAP < 65 mmHg en un intervalo de 30 min.
- **Caso B (estricto)**: ≥2 lecturas consecutivas MAP < 55 mmHg (implica ≥~5-10 min si ciclo ~5 min).
- Reportar ambos criterios y analizar sensibilidad.
- Si el ciclo NIBP es >10 min, la definición de "episodio hipotensivo" es poco fiable y debe indicarse explícitamente como limitación.
""")

    # =====================================================================
    # 6. Anotaciones
    # =====================================================================
    L("## 6. Anotaciones (estímulos y medicación)\n")

    all_annot_texts = []
    for p in patient_records:
        all_annot_texts.extend([ev["text"] for ev in p.events if ev["text"].strip()])

    if not all_annot_texts:
        L("> **⚠️ No se encontraron anotaciones en ningún archivo .vital.**")
        L("> Verificar si están almacenadas en los archivos .csv asociados o en otro formato.")
    else:
        counter = Counter(all_annot_texts)
        L(f"**Total de anotaciones encontradas**: {len(all_annot_texts)} | "
          f"**Únicas**: {len(counter)}\n")

        # Clasificar
        stimuli  = {t: n for t, n in counter.items() if _is_stimulus(t)}
        drugs    = {t: n for t, n in counter.items() if _is_drug(t)}
        others   = {t: n for t, n in counter.items() if not _is_stimulus(t) and not _is_drug(t)}

        if stimuli:
            L("### 6.1 Anotaciones de estímulos quirúrgicos\n")
            rows = sorted(stimuli.items(), key=lambda x: -x[1])
            L(_md_table(["Texto", "Frecuencia"], [[t, n] for t, n in rows]))
        else:
            L("### 6.1 Anotaciones de estímulos quirúrgicos\n")
            L("> No identificadas con palabras clave esperadas (piel, trocar, ancle, sutura, fresa...).")

        if drugs:
            L("\n### 6.2 Anotaciones de medicación\n")
            rows = sorted(drugs.items(), key=lambda x: -x[1])
            L(_md_table(["Texto", "Frecuencia"], [[t, n] for t, n in rows]))
        else:
            L("\n### 6.2 Anotaciones de medicación\n")
            L("> No identificadas con palabras clave de fármacos esperados.")

        if others:
            L("\n### 6.3 Otras anotaciones (top 30)\n")
            rows = sorted(others.items(), key=lambda x: -x[1])[:30]
            L(_md_table(["Texto", "Frecuencia"], [[t, n] for t, n in rows]))

    L("\n### 6.4 Resumen de anotaciones por paciente\n")
    pa_rows = []
    for p in patient_records:
        n_stim = sum(1 for ev in p.events if _is_stimulus(ev["text"]))
        n_drug = sum(1 for ev in p.events if _is_drug(ev["text"]))
        n_other = len(p.events) - n_stim - n_drug
        ts_out = []
        dur_s  = p.dur_s
        for ev in p.events:
            t = ev["time_s"]
            if dur_s > 0 and (t < 0 or t > dur_s + 60):
                ts_out.append(f"{t:.0f}s")
        pa_rows.append([
            p.patient_id,
            str(len(p.events)),
            str(n_stim),
            str(n_drug),
            str(n_other),
            ", ".join(ts_out[:3]) or "—",
        ])
    L(_md_table(
        ["Paciente", "#total", "#estímulos", "#medicación", "#otros", "Timestamps fuera de rango"],
        pa_rows,
    ))

    # =====================================================================
    # 7. Reconciliación con análisis previo
    # =====================================================================
    L("\n## 7. Reconciliación con análisis previo\n")
    L("""El enunciado menciona "46.539 registros con eventos en 20 pacientes" de un
RESUMEN_RESULTADOS_ESTADISTICOS. **Este fichero no se ha encontrado en el repositorio.**
La reconciliación completa no es posible hasta que se aporte ese documento.

### Estimación de implicaciones del número 46.539

Si 46.539 representa el número de *ventanas de análisis* (no de lecturas NIBP):

| Supuesto | Implicación |
|----------|-------------|
| Ventanas de 30 s, paso 1 latido (BRS-like, ~1 s paso) | 46.539 ventanas / 20 pac = ~2.327 ventanas/pac → ~2.327 s = ~38.8 min de datos válidos/pac (razonable para cirugía 2-4 h) |
| Ventanas de 30 s, paso 30 s (sin solapamiento) | 46.539 × 30 s / 20 pac = ~23.270 s = ~388 min/pac → **muy alto para cirugía** |
| Lecturas NIBP directas | 46.539 / 20 = ~2.327 lecturas NIBP/pac → a 5 min/ciclo = ~194 h/pac → **imposible** |

**Hipótesis más probable**: los 46.539 registros son filas de un DataFrame con ventanas
deslizantes (paso ~1 latido ≈ 1 s) sobre señal PPG-derivada. Esto implica señal continua
reindexada, no lecturas NIBP brutas. Verificar cuando esté disponible el análisis previo.

### Inconsistencias a verificar
- Este script ha encontrado {n_total} carpetas de paciente. Si el análisis previo usó 20 pacientes
  exactamente, hay que identificar cuáles fueron incluidos/excluidos.
- Las carpetas marcadas "mal" sugieren exclusiones previas: **4234018 mal** (sin .vital) y
  **70767707 mal** (tiene .vital — verificar si fue excluido).
""".format(n_total=n_total))

    # =====================================================================
    # 8. Factibilidad de features BeatLabile-like
    # =====================================================================
    L("## 8. Factibilidad de features BeatLabile-like\n")
    L(_md_table(
        ["Feature BeatLabile", "Disponible directamente", "Surrogate propuesto", "Calidad surrogate"],
        [[f["Feature BeatLabile"], f["Disponible directamente"],
          f["Surrogate propuesto"], f["Calidad del surrogate"]]
         for f in FEATURE_FEASIBILITY],
    ))

    L("""
### Conclusión de factibilidad

Las features de BeatLabile original (cv-PA-std, brs-min, ARV) requieren arteria invasiva
y **no son computables en esta cohorte**. Los surrogates PPG/PRV son técnicamente posibles
pero introducen limitaciones conocidas:

1. **PRV ≠ HRV** durante anestesia, especialmente con cambios de volumen o vasopresores.
2. **La escala temporal del NIBP** (~5 min) es incomparable con la escala latido-a-latido
   de BeatLabile — las features basadas en NIBP miden variabilidad macroscópica, no
   la micro-variabilidad que caracteriza la firma autonómica.
3. El **PPG-PAI** (perfusion amplitude index) puede ser el mejor surrogate disponible:
   refleja tono vasomotor y puede detectar cambios autonómicos perioperatorios.
4. Sin ECG simultáneo, no es posible estimar PTT para BRS surrogate.
""")

    # =====================================================================
    # 9. Sanity checks y Gantts
    # =====================================================================
    L("## 9. Sanity checks y Gantts por paciente\n")

    L("### 9.1 Verificación de sincronización temporal\n")
    L("""Los tracks dentro de un mismo .vital comparten la línea de tiempo del VitalRecorder.
En principio están sincronizados. Los desfases observados en datos reales suelen deberse a:
- Reinicios de dispositivo durante la cirugía
- Segmentos con timestamps no monotónicos (se reportarían como gaps)
- El dispositivo VitalRecorder puede tener deriva del reloj < 1 s/h en grabaciones largas

Resultado de verificación: no se ha ejecutado análisis de monotonía de timestamps en
este script (requeriría iterar sobre todos los recs). Se recomienda verificar con
`sorted(rec.dt for rec in trk.recs)` vs `[rec.dt for rec in trk.recs]` antes del análisis.
""")

    L("### 9.2 Solapamiento anotaciones / gaps PPG\n")
    L("Verificación por paciente de si hay anotaciones en períodos donde el PPG está caído:\n")
    overlap_rows = []
    for p in patient_records:
        annot_in_gap = 0
        for ev in p.events:
            t = ev["time_s"]
            for g_start, g_end in p.ppg_gaps:
                if g_start <= t <= g_end:
                    annot_in_gap += 1
                    break
        overlap_rows.append([p.patient_id, str(len(p.events)), str(len(p.ppg_gaps)), str(annot_in_gap)])
    L(_md_table(
        ["Paciente", "#anotaciones", "#gaps PPG >5s", "#anotaciones en gap PPG"],
        overlap_rows,
    ))

    L("\n### 9.3 Diagramas Gantt por paciente\n")
    L("```\nLeyenda: G=gap PPG  N=NIBP  A=anotación  -=datos continuos\n```\n")
    for p in patient_records:
        dur_s      = p.dur_s
        nibp_times = p.nibp.timestamps_s if p.nibp else []
        annot_t    = [ev["time_s"] for ev in p.events]
        gap_pds    = p.ppg_gaps
        grupo_str  = ("interescalenico" if p.grupo_bloqueo is True
                      else ("supra+axilar" if p.grupo_bloqueo is False else "grupo?"))
        L(f"**{p.patient_id}** ({grupo_str}) — duración: {dur_s/60:.1f} min | "
          f"NIBP: {len(nibp_times)} | Anotaciones: {len(annot_t)}\n")
        L("```")
        L(_gantt_ascii(dur_s, nibp_times, annot_t, gap_pds))
        L("```\n")

    # =====================================================================
    # 10. Recomendaciones operativas
    # =====================================================================
    L("## 10. Recomendaciones operativas\n")

    excluded_list = []
    if n_sin_vital > 0:
        for p in patient_records:
            if not p.vital_files:
                excluded_list.append(f"**{p.patient_id}**: sin archivo .vital")
    for p in patient_records:
        if p.ppg_quality and not p.ppg_quality.error and p.ppg_quality.pct_valid < PPG_QUALITY_THRESHOLD * 100:
            excluded_list.append(f"**{p.patient_id}**: PPG válido {p.ppg_quality.pct_valid:.1f}% < {PPG_QUALITY_THRESHOLD*100:.0f}%")

    L("### 10.1 Pacientes candidatos a exclusión\n")
    if excluded_list:
        for e in excluded_list:
            L(f"- {e}")
    else:
        L("> Ningún paciente alcanza el umbral de exclusión automática. Verificar manualmente.")

    L("""
### 10.2 Definiciones operativas a fijar antes del análisis

| Parámetro | Propuesta | Justificación |
|-----------|-----------|---------------|
| **Umbral hipotensión** | MAP < 65 mmHg (1 lectura) o MAP < 55 mmHg (2 lecturas consecutivas) | Sin arteria invasiva, no se puede exigir "3 min continuos"; proponer dos criterios y analizar sensibilidad |
| **Ventana pre-evento** | 30 min antes de primera lectura hipotensora | Consistente con BeatLabile; verificar que haya ≥10 lecturas NIBP en esa ventana |
| **Criterio de calidad PPG mínima** | ≥70% válido en la ventana de análisis | Umbral estándar; considerar elevar a 80% dado que SNR PPG perioperatorio es variable |
| **Definición "baja PA estable" (Q3)** | MAP ∈ [50-65] mmHg durante ≥2 lecturas consecutivas SIN tendencia a caída | Distinguir del "pre-hipotensión": la tendencia temporal (slope de NIBP) es clave |
| **Ventana de bloqueo activo** | Desde anotación de colocación + 20 min (onset típico) hasta fin de cirugía | Ajustar si hay anotación explícita de "inicio de bloqueo confirmado" |
| **Estimación PRV** | Peaks PPG con neurokit2, rechazo de peaks con Δt <200 ms o >2000 ms | Consistente con criterios de HRV estándar (RMSSD/SDNN) |

### 10.3 Riesgos identificados

1. **Potencia estadística (n=20)**: Con 12+8 pacientes, incluso un AUC de 0.80 con SD=0.15 tiene potencia <50% para detectarlo frente a 0.50. Se recomienda presentar intervalos de confianza amplios y no concluir sobre ausencia de efecto.
2. **Confounders no controlados**: BMI, HTA previa, medicación habitual, tipo de anestesia (propofol vs sevoflurano), duración de cirugía. Verificar disponibilidad en archivos .csv de cada paciente.
3. **Bloqueo interescalénico y PPG**: El bloqueo causa vasodilatación ipsilateral del brazo. Si el pulsioxímetro está en el brazo bloqueado, el PPG tendrá morfología alterada (mayor amplitud, menor variabilidad). **Verificar en qué brazo está el sensor SpO2**.
4. **Posición en silla de playa**: La posición semi-sentada puede causar hipotensión posicional y reducir la señal PPG. Considerar ajuste de valores de referencia.
5. **Desfibrilación/artefactos electroquirúrgicos**: La electrocirugía (fresa artroscópica) genera artefactos en todas las señales. Verificar correlación entre anotaciones de fresa y gaps en PPG.
6. **Sesgo de selección**: Si los "pacientes con bloqueo" tienen cirugías más largas (mayor complejidad) o diferente perfil de riesgo, el grupo no es comparable por duración.
7. **Transportabilidad**: BeatLabile demostró baja concordancia entre cohortes para hipotensión (ρ=0.188). Esta cohorte es aún más específica (hombro + bloqueo), lo que puede mejorar la homogeneidad intra-cohorte pero dificultar la generalización.

### 10.4 Próximos pasos recomendados

1. Confirmar asignación de grupos (13 bloqueo interescalénico / 8 bloqueo supra+axilar) — **cargada desde Excel**.
2. Verificar los archivos .csv de cada paciente (datos clínicos: BMI, HTA, ASA, etc.).
3. Localizar los datos de Medasense (NOL — nociception level) si aplican al subestudio.
4. Revisar manualmente las anotaciones de las carpetas con alertas de nombre ambiguo.
5. Definir si la carpeta "4234018 mal" (sin .vital) y "70767707 mal" (tiene .vital) son exclusiones previas o datos pendientes de cargar.
""")

    # =====================================================================
    # Cuestiones abiertas
    # =====================================================================
    L("## Cuestiones abiertas\n")
    L("""1. **Asignación de grupos**: Cargada desde `database general pacientes estudio hombro.xlsx` — 13 interescalénico / 8 supra+axilar. ✅ Resuelta.
2. **Archivos .vital con nombre de dispositivo** (patrón `scjrke68a_YYMMDD_HHMMSS.vital`): No contienen el ID de paciente en el nombre. La correspondencia se ha verificado manualmente por fecha y por `dtstart`. Marcados como "nombre ambiguo" solo como aviso informativo, no como error.
3. **Datos Medasense (NOL)**: Todas las carpetas tienen subcarpeta `Medasense_Data/`. ¿Se va a integrar el índice de nocicepción como covariable?
4. **Sensor PPG ipsilateral vs contralateral**: ¿El pulsioxímetro está en el brazo del bloqueo o en el contralateral? Crítico para interpretar cambios en PPG-PAI.
5. **Archivos .csv por paciente**: Contienen datos clínicos o exportaciones de VitalRecorder. ¿Hay información adicional de señal o solo metadatos?
6. **Sincronización Medasense-VitalRecorder**: Si se integran los datos NOL, verificar que los timestamps estén en la misma zona horaria y reference time.
7. **Paciente 4234018** (carpeta `4234018 mal`): Sin .vital — VitalRecorder se apagó antes de grabar. Confirmar si se excluye del análisis.
8. **Paciente 70767707** (carpeta `70767707 mal`): Grabación de solo 23 min — VitalRecorder se apagó. Insuficiente para features de ventana larga. Revisar si se incluye como parcial o se excluye.
9. **Paciente 70078466**: Sin .vital (archivo erróneo eliminado). Sin datos de señal recuperables. Excluir.
10. **Anotaciones en ficheros .csv**: Si las anotaciones no aparecen en el .vital (ver sección 6), pueden estar en los .csv exportados. Incluir parser de CSV en la próxima iteración.
""")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def plot_ppg_histogram(patient_records: list[PatientRecord], out_path: Path):
    """Histograma de % PPG válido por paciente."""
    pcts  = []
    ids   = []
    flags = []
    for p in patient_records:
        q = p.ppg_quality
        if q and not q.error:
            pcts.append(q.pct_valid)
            ids.append(p.patient_id)
            flags.append(q.pct_valid < PPG_QUALITY_THRESHOLD * 100)

    if not pcts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sin datos de PPG disponibles", ha="center", va="center")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Calidad de PPG — Cohorte hombro/bloqueo", fontsize=13, fontweight="bold")

    # Histograma
    colors = ["#e74c3c" if f else "#2ecc71" for f in flags]
    ax1.bar(range(len(pcts)), pcts, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(PPG_QUALITY_THRESHOLD * 100, color="navy", linestyle="--",
                linewidth=1.5, label=f"Umbral {PPG_QUALITY_THRESHOLD*100:.0f}%")
    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids, rotation=60, ha="right", fontsize=8)
    ax1.set_ylabel("% PPG válido (no-NaN)")
    ax1.set_ylim(0, 105)
    ax1.set_title("% PPG válido por paciente")
    ax1.legend()
    red_patch   = mpatches.Patch(color="#e74c3c", label=f"< {PPG_QUALITY_THRESHOLD*100:.0f}% (excluir)")
    green_patch = mpatches.Patch(color="#2ecc71", label=f"≥ {PPG_QUALITY_THRESHOLD*100:.0f}% (incluir)")
    ax1.legend(handles=[red_patch, green_patch,
                        mpatches.Patch(color="navy", label=f"Umbral {PPG_QUALITY_THRESHOLD*100:.0f}%")])

    # Histograma de distribución
    ax2.hist(pcts, bins=10, range=(0, 100), color="#3498db", edgecolor="white", linewidth=0.7)
    ax2.axvline(PPG_QUALITY_THRESHOLD * 100, color="#e74c3c", linestyle="--",
                linewidth=2, label=f"Umbral {PPG_QUALITY_THRESHOLD*100:.0f}%")
    ax2.set_xlabel("% PPG válido")
    ax2.set_ylabel("# Pacientes")
    ax2.set_title("Distribución de % PPG válido")
    ax2.legend()

    # Anotaciones de resumen
    n_ok  = sum(1 for f in flags if not f)
    n_bad = sum(1 for f in flags if f)
    ax2.text(0.05, 0.95, f"OK: {n_ok}  |  Bajo: {n_bad}",
             transform=ax2.transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Figura guardada: {out_path}")


# ---------------------------------------------------------------------------
# CSVs de artefactos
# ---------------------------------------------------------------------------

def save_patient_csv(patient_records: list[PatientRecord], out_path: Path):
    """CSV de inventario por paciente."""
    rows = []
    for p in patient_records:
        q = p.ppg_quality
        n = p.nibp
        dur_s = p.dur_s
        n_stim = sum(1 for ev in p.events if _is_stimulus(ev["text"]))
        n_drug = sum(1 for ev in p.events if _is_drug(ev["text"]))
        rows.append({
            "paciente": p.patient_id,
            "archivo_vital": "; ".join(f.name for f in p.vital_files) or "—",
            "n_vitals": len(p.vital_files),
            "tamano_total_mb": round(sum(p.file_sizes_bytes) / 1e6, 2),
            "duracion_min": round(dur_s / 60, 1) if dur_s > 0 else None,
            "nombre_ambiguo": p.ambiguous_name,
            "tipo_bloqueo": ("interescalenico" if p.grupo_bloqueo is True
                             else ("supra_axilar" if p.grupo_bloqueo is False else None)),
            "pct_ppg_valido": round(q.pct_valid, 1) if q and not q.error else None,
            "ppg_dur_valida_min": round(q.duration_valid_s / 60, 1) if q and not q.error else None,
            "ppg_snr_median_db": round(q.snr_median_db, 1) if q and not q.error and not np.isnan(q.snr_median_db) else None,
            "ppg_n_gaps_gt5s": q.n_gaps_gt5s if q and not q.error else None,
            "ppg_flag_excluir": (q.pct_valid < PPG_QUALITY_THRESHOLD * 100) if q and not q.error else None,
            "n_nibp": n.n_readings if n else 0,
            "nibp_ciclo_medio_min": round(n.mean_cycle_min, 1) if n and not np.isnan(n.mean_cycle_min) else None,
            "nibp_map_media": round(float(np.mean(n.map_values)), 1) if n and n.map_values else None,
            "nibp_n_map_lt55": n.n_hypo_maps if n else 0,
            "n_anotaciones_estimulos": n_stim,
            "n_anotaciones_medicacion": n_drug,
            "n_anotaciones_total": len(p.events),
            "error_carga": p.parse_error,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  -> CSV inventario pacientes: {out_path}")


def save_annotations_csv(patient_records: list[PatientRecord], out_path: Path):
    """CSV con todas las anotaciones únicas y su frecuencia."""
    all_texts = []
    for p in patient_records:
        for ev in p.events:
            if ev["text"].strip():
                all_texts.append({
                    "texto": ev["text"].strip(),
                    "tipo": ("estimulo" if _is_stimulus(ev["text"])
                             else ("medicacion" if _is_drug(ev["text"]) else "otro")),
                })
    if not all_texts:
        pd.DataFrame(columns=["texto", "tipo", "frecuencia"]).to_csv(out_path, index=False)
        print(f"  -> CSV anotaciones (vacio): {out_path}")
        return
    df = pd.DataFrame(all_texts)
    freq = df.groupby(["texto", "tipo"]).size().reset_index(name="frecuencia")
    freq = freq.sort_values("frecuencia", ascending=False)
    freq.to_csv(out_path, index=False)
    print(f"  -> CSV anotaciones unicas: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  AUDITORÍA COHORTE HOMBRO/BLOQUEO")
    print(f"  Datos: {DATA_DIR}")
    print(f"  Salida: {OUT_DIR}")
    print("=" * 60)

    # Verificar dependencias
    try:
        import vitaldb
        print(f"  vitaldb version: {getattr(vitaldb, '__version__', 'desconocida')}")
    except ImportError:
        print("ERROR: vitaldb no instalado. Ejecutar: pip install vitaldb")
        sys.exit(1)

    # Listar carpetas de paciente
    patient_folders = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    print(f"\n  Carpetas encontradas: {len(patient_folders)}")

    patient_records: list[PatientRecord] = []

    for folder in patient_folders:
        pid = folder.name
        print(f"\n  Procesando: {pid} ...", end=" ", flush=True)
        try:
            rec = _process_patient(folder)
            patient_records.append(rec)
            ppg_pct = (f"PPG={rec.ppg_quality.pct_valid:.0f}%"
                       if rec.ppg_quality and not rec.ppg_quality.error
                       else f"PPG=err({rec.ppg_quality.error if rec.ppg_quality else 'N/A'})")
            nibp_n  = f"NIBP={rec.nibp.n_readings if rec.nibp else 0}"
            annot_n = f"Annot={len(rec.events)}"
            dur_str = f"Dur={rec.dur_s/60:.0f}min" if rec.dur_s > 0 else "Dur=?"
            print(f"OK | {dur_str} | {ppg_pct} | {nibp_n} | {annot_n}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            # Crear registro vacío para que aparezca en el informe
            rec = PatientRecord(patient_id=pid, folder=folder,
                                parse_error=f"Excepción: {e}")
            patient_records.append(rec)

    print(f"\n  Pacientes procesados: {len(patient_records)}")

    # Generar informe
    print("\n  Generando informe Markdown...")
    report_text = generate_report(patient_records)
    report_path = OUT_DIR / "informe_auditoria_hombro.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  -> Informe: {report_path}")

    # CSV inventario
    save_patient_csv(patient_records, OUT_DIR / "inventario_pacientes.csv")

    # CSV anotaciones
    save_annotations_csv(patient_records, OUT_DIR / "anotaciones_unicas.csv")

    # Figura PPG
    print("  Generando figura histograma PPG...")
    plot_ppg_histogram(patient_records, OUT_DIR / "ppg_quality_histogram.png")

    print()
    print("=" * 60)
    print("  AUDITORIA COMPLETADA")
    print(f"  Resultados en: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
