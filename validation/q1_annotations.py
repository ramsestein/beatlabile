"""
q1_annotations.py
=================
PASO 1 — Parser y normalización de anotaciones quirúrgicas.

Produce tres outputs:
  A) annotations_normalized.csv  — evento por fila, categorizado
  B) drug_timeseries.parquet     — series temporales de fármacos a 1 Hz
  C) Log de inconsistencias y resumen estadístico

Categorías:
  estimulo     : piel, trocar, anclaje, sutura, fresa
  medicacion_bolus : efedrina, fenilefrina, fentanilo, metadona, atropina
  infusion_change  : cambio de propofol/remifentanilo (regex)
  fase             : AG, fin AG, limpian, fin IQ, DLI, DLD, silla playa
  otro             : resto
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from q1_config import (
    EVENT_WINDOW_S,
    EXCLUIDOS,
    INCLUIDOS_CON_PARCIAL,
    RESULTS_DIR,
    STIMULUS_SUBCATEGORIES,
)
from q1_load import get_events_raw, load_vital, list_patients

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patrones de categorización
# ---------------------------------------------------------------------------

# ── Estímulos quirúrgicos ──
_STIM_PATTERNS: list[tuple[str, str]] = [
    (r"\bfresa[n]?\b",                      "fresa"),
    (r"\btrocar\b",                          "trocar"),
    (r"\bancl[ae]\b",                        "anclaje"),
    (r"\banclaje\b",                         "anclaje"),
    (r"\bsutura\b",                          "sutura"),
    (r"\btiran\s+la\s+sutura\b",             "sutura"),
    (r"\bsutura\s+piel\b",                   "sutura"),
    (r"\bpiel\b",                            "piel"),
]

# ── Medicación bolus ──
_MED_PATTERNS: list[tuple[str, str]] = [
    (r"\befedrina?\b",    "efedrina"),
    (r"\bfenilefrina?\b", "fenilefrina"),
    (r"\bfentanilo?\b",   "fentanilo"),
    (r"\bfenta\b",        "fentanilo"),
    (r"\bmetadona\b",     "metadona"),
    (r"\batropina?\b",    "atropina"),
]

# ── Infusión TCI (propofol / remifentanil) ──
# Captura patrones como "propo 2.8, remi 2" o "propo 3, remi 3.5"
# Número flotante: uno o más dígitos, opcionalmente punto decimal seguido de dígitos
_NUM = r"(\d+(?:\.\d+)?)"
_INFUSION_RE = re.compile(
    r"propo\s*" + _NUM + r".*?remi\s*" + _NUM,
    re.IGNORECASE | re.DOTALL,
)
# Solo remi (sin propo en la misma anotación)
_REMI_ONLY_RE  = re.compile(r"remi\s*" + _NUM, re.IGNORECASE)
_PROPO_ONLY_RE = re.compile(r"propo\s*" + _NUM, re.IGNORECASE)

# ── Dosis numérica en bolus ──
_DOSE_RE = re.compile(r"([\d.]+)\s*(mg|mcg|µg|ug|ml|mg/kg)?", re.IGNORECASE)

# ── Fases quirúrgicas ──
_FASE_PATTERNS: list[tuple[str, str]] = [
    # IMPORTANTE: los patrones más específicos primero
    (r"\bfin\s*ag\b",     "fin_AG"),   # "fin AG" antes que "AG"
    (r"\bfin\s*iq\b",     "fin_IQ"),
    (r"\bag\b",           "AG"),
    (r"\blimpian\b",      "limpian"),
    (r"\bdli\b",          "DLI"),
    (r"\bdld\b",          "DLD"),
    (r"\bml\b",           "ML"),
    (r"\bdl\b",           "DL"),
    (r"\bsilla\s+playa\b", "silla_playa"),
]


# ---------------------------------------------------------------------------
# Dataclass de anotación normalizada
# ---------------------------------------------------------------------------

@dataclass
class NormalizedAnnotation:
    patient_id:   str
    t_s:          float          # segundos desde inicio de grabación
    raw_text:     str
    category:     str            # estimulo | medicacion_bolus | infusion_change | fase | otro
    subcategory:  str            # nombre específico
    dose:         Optional[float] = None
    dose_unit:    Optional[str]  = None
    propo_target: Optional[float] = None   # solo para infusion_change
    remi_target:  Optional[float] = None   # solo para infusion_change
    seq_num:      int = 0        # número secuencial para anclaje_2, trocar_2, etc.


# ---------------------------------------------------------------------------
# Funciones de categorización
# ---------------------------------------------------------------------------

def _categorize(text: str) -> list[NormalizedAnnotation]:
    """
    Parsea un texto de anotación y retorna LISTA de anotaciones normalizadas
    (una anotación combinada puede producir múltiples eventos).
    """
    results: list[NormalizedAnnotation] = []
    t_low = text.lower()

    # 1. Separar líneas/partes (anotaciones combinadas con \n)
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]

    for part in parts:
        part_low = part.lower()
        ann = _categorize_single(part, part_low)
        if ann is not None:
            results.append(ann)

    if not results:
        # Ninguna parte reconocida → un solo 'otro'
        results.append(NormalizedAnnotation(
            patient_id="", t_s=0.0,
            raw_text=text, category="otro", subcategory="otro"
        ))
    return results


def _categorize_single(part: str, part_low: str) -> Optional[NormalizedAnnotation]:
    """Categoriza una sola línea de anotación."""

    # ── Infusion change (propo + remi) ──
    m = _INFUSION_RE.search(part)
    if m:
        return NormalizedAnnotation(
            patient_id="", t_s=0.0,
            raw_text=part, category="infusion_change", subcategory="propofol_remi",
            propo_target=float(m.group(1)),
            remi_target=float(m.group(2)),
        )

    # ── Infusion solo remi ──
    m_r = _REMI_ONLY_RE.search(part)
    m_p = _PROPO_ONLY_RE.search(part)
    if m_r and not m_p:
        return NormalizedAnnotation(
            patient_id="", t_s=0.0,
            raw_text=part, category="infusion_change", subcategory="remi_only",
            remi_target=float(m_r.group(1)),
        )
    if m_p and not m_r:
        return NormalizedAnnotation(
            patient_id="", t_s=0.0,
            raw_text=part, category="infusion_change", subcategory="propo_only",
            propo_target=float(m_p.group(1)),
        )

    # ── Estímulo ──
    for pat, subcat in _STIM_PATTERNS:
        if re.search(pat, part_low):
            return NormalizedAnnotation(
                patient_id="", t_s=0.0,
                raw_text=part, category="estimulo", subcategory=subcat,
            )

    # ── Medicación bolus ──
    for pat, subcat in _MED_PATTERNS:
        if re.search(pat, part_low):
            dose, unit = _extract_dose(part)
            return NormalizedAnnotation(
                patient_id="", t_s=0.0,
                raw_text=part, category="medicacion_bolus", subcategory=subcat,
                dose=dose, dose_unit=unit,
            )

    # ── Fase ──
    for pat, subcat in _FASE_PATTERNS:
        if re.search(pat, part_low):
            return NormalizedAnnotation(
                patient_id="", t_s=0.0,
                raw_text=part, category="fase", subcategory=subcat,
            )

    # ── Otro (incluye ".", ML, etc.) ──
    return NormalizedAnnotation(
        patient_id="", t_s=0.0,
        raw_text=part, category="otro", subcategory="otro",
    )


def _extract_dose(text: str) -> tuple[Optional[float], Optional[str]]:
    """Extrae dosis numérica y unidad de un texto de medicación."""
    m = re.search(r"([\d.]+)\s*(mg|mcg|µg|ug|ml|mg/kg)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), m.group(2).lower()
        except ValueError:
            pass
    # Buscar solo número
    m2 = re.search(r"([\d.]+)", text)
    if m2:
        try:
            return float(m2.group(1)), None
        except ValueError:
            pass
    return None, None


# ---------------------------------------------------------------------------
# Numeración secuencial de anclajes / trocares
# ---------------------------------------------------------------------------

def _assign_seq_numbers(anns: list[NormalizedAnnotation]) -> None:
    """Asigna seq_num secuencial para cada subcategoría de estímulo
    y normaliza "anclaje_2", "trocar_3", etc. → subcategory = "anclaje",
    seq_num = 2.
    """
    # Extraer número del raw_text si existe (e.g. "anclaje 2" → seq=2)
    _NUM_RE = re.compile(r"\b(\d+)\s*$")
    counts: dict[str, int] = {}

    for ann in anns:
        if ann.category != "estimulo":
            continue
        subcat = ann.subcategory
        m = _NUM_RE.search(ann.raw_text.strip())
        if m:
            ann.seq_num = int(m.group(1))
        else:
            # Asignar secuencialmente
            counts[subcat] = counts.get(subcat, 0) + 1
            ann.seq_num = counts[subcat]


# ---------------------------------------------------------------------------
# Parser principal por paciente
# ---------------------------------------------------------------------------

def parse_annotations(patient_id: str, vf) -> list[NormalizedAnnotation]:
    """Parsea y normaliza todas las anotaciones de un VitalFile."""
    raw_events = get_events_raw(vf)
    anns: list[NormalizedAnnotation] = []

    for ev in raw_events:
        new_anns = _categorize(ev["text"])
        for ann in new_anns:
            ann.patient_id = patient_id
            ann.t_s        = ev["time_s"]
            ann.raw_text   = ev["text"]  # mantener raw original siempre
        anns.extend(new_anns)

    _assign_seq_numbers(anns)
    return anns


# ---------------------------------------------------------------------------
# Validación de consistencia
# ---------------------------------------------------------------------------

def validate_annotations(patient_id: str, anns: list[NormalizedAnnotation],
                          duration_s: float, logger) -> list[str]:
    """Verifica consistencias y devuelve lista de advertencias."""
    warnings: list[str] = []

    ag_times   = [a.t_s for a in anns if a.category == "fase" and a.subcategory == "AG"]
    fin_times  = [a.t_s for a in anns if a.category == "fase" and a.subcategory in ("fin_AG", "fin_IQ")]
    stim_times = [a.t_s for a in anns if a.category == "estimulo"]
    drug_times = [a.t_s for a in anns if a.category == "medicacion_bolus"]
    infusion_t = [a.t_s for a in anns if a.category == "infusion_change"]

    if not ag_times:
        warnings.append(f"{patient_id}: SIN anotación 'AG' — se usará t=0 como inicio")
    if not fin_times:
        warnings.append(f"{patient_id}: SIN 'fin AG'/'fin IQ' — se usará fin de grabación ({duration_s:.0f} s) como límite")
    if not stim_times:
        warnings.append(f"{patient_id}: SIN estímulos dolorosos")

    t_ag  = min(ag_times)  if ag_times  else 0.0
    t_fin = max(fin_times) if fin_times else duration_s

    # Estímulos antes de AG
    pre_ag = [t for t in stim_times if t < t_ag]
    if pre_ag:
        warnings.append(
            f"{patient_id}: {len(pre_ag)} estímulo(s) antes de AG (t={pre_ag})"
        )

    # Cambios de infusión después de fin AG
    post_fin_inf = [t for t in infusion_t if t > t_fin]
    if post_fin_inf:
        warnings.append(
            f"{patient_id}: {len(post_fin_inf)} infusion_change después de fin AG"
        )

    # Timestamps negativos o mayores que la duración
    out_of_range = [
        a.t_s for a in anns if a.t_s < 0 or (duration_s > 0 and a.t_s > duration_s + 60)
    ]
    if out_of_range:
        warnings.append(
            f"{patient_id}: {len(out_of_range)} anotaciones fuera de rango temporal"
        )

    for w in warnings:
        logger.warning(w)

    return warnings


# ---------------------------------------------------------------------------
# Construcción de drug_timeseries (1 Hz, por paciente)
# ---------------------------------------------------------------------------

def build_drug_timeseries(patient_id: str, anns: list[NormalizedAnnotation],
                           duration_s: float) -> pd.DataFrame:
    """
    Reconstruye series temporales de fármacos a 1 Hz.
    Asume: concentración constante desde cada cambio hasta el siguiente.
    Retorna DataFrame con columnas:
      t_s, propofol_target, remifentanil_target,
      cumulative_efedrina_mg, cumulative_fenilefrina_mcg, cumulative_fentanilo_mcg
    """
    if duration_s <= 0:
        return pd.DataFrame()

    n = int(np.ceil(duration_s)) + 1
    t_arr = np.arange(n, dtype=np.float64)

    propo_arr  = np.zeros(n, dtype=np.float32)
    remi_arr   = np.zeros(n, dtype=np.float32)
    efed_cum   = np.zeros(n, dtype=np.float32)
    feni_cum   = np.zeros(n, dtype=np.float32)
    fenta_cum  = np.zeros(n, dtype=np.float32)

    # Obtener tiempos AG/fin_AG para limitar
    ag_times  = [a.t_s for a in anns if a.category == "fase" and a.subcategory == "AG"]
    fin_times = [a.t_s for a in anns if a.category == "fase"
                 and a.subcategory in ("fin_AG", "fin_IQ")]

    t_ag_start = int(min(ag_times)) if ag_times else 0
    t_ag_end   = int(max(fin_times)) if fin_times else n

    # --- Infusiones TCI (step function) ---
    infusions = sorted(
        [a for a in anns if a.category == "infusion_change"],
        key=lambda a: a.t_s,
    )

    cur_propo = 0.0
    cur_remi  = 0.0
    prev_t    = t_ag_start

    for ann in infusions:
        t_idx = int(ann.t_s)
        if t_idx > t_ag_end:
            break
        if t_idx > prev_t:
            lo = max(0, prev_t)
            hi = min(n, t_idx)
            propo_arr[lo:hi] = cur_propo
            remi_arr[lo:hi]  = cur_remi
        if ann.propo_target is not None:
            cur_propo = ann.propo_target
        if ann.remi_target is not None:
            cur_remi  = ann.remi_target
        prev_t = t_idx

    # Rellenar desde el último cambio hasta fin_AG
    if prev_t < t_ag_end:
        propo_arr[prev_t:t_ag_end] = cur_propo
        remi_arr[prev_t:t_ag_end]  = cur_remi

    # --- Bolus acumulados (step function) ---
    _bolus_cumsum(anns, "efedrina",     "mg",  efed_cum,  n)
    _bolus_cumsum(anns, "fenilefrina",  "mcg", feni_cum,  n)
    _bolus_cumsum(anns, "fentanilo",    "mcg", fenta_cum, n)

    df = pd.DataFrame({
        "t_s":                   t_arr,
        "patient_id":            patient_id,
        "propofol_target":       propo_arr,
        "remifentanil_target":   remi_arr,
        "cumulative_efedrina_mg":    efed_cum,
        "cumulative_fenilefrina_mcg": feni_cum,
        "cumulative_fentanilo_mcg":  fenta_cum,
    })
    return df


def _bolus_cumsum(anns: list[NormalizedAnnotation], drug: str, unit: str,
                   arr: np.ndarray, n: int) -> None:
    """Acumula bolus de un fármaco en el array arr (in-place, 1 Hz)."""
    events = sorted(
        [(int(a.t_s), a.dose or 0.0)
         for a in anns
         if a.category == "medicacion_bolus" and a.subcategory == drug],
        key=lambda x: x[0],
    )
    cumsum = 0.0
    ev_idx = 0
    for i in range(n):
        while ev_idx < len(events) and events[ev_idx][0] <= i:
            cumsum += events[ev_idx][1]
            ev_idx += 1
        arr[i] = cumsum


# ---------------------------------------------------------------------------
# Función principal: procesar toda la cohorte
# ---------------------------------------------------------------------------

def run_paso1(patients: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Ejecuta el PASO 1 completo.

    Returns:
      annotations_df : DataFrame de anotaciones normalizadas
      drug_ts_df     : DataFrame de drug_timeseries (concatenado, todos los pacientes)
      all_warnings   : Lista de advertencias/inconsistencias
    """
    if patients is None:
        patients = list_patients(include_partial=True)

    all_anns: list[dict] = []
    all_drug_dfs: list[pd.DataFrame] = []
    all_warnings: list[str] = []

    # Contadores para log
    total_raw   = 0
    by_category: dict[str, int] = {}
    by_subcat:   dict[str, int] = {}

    for pid in patients:
        log.info("PASO 1 — %s: cargando anotaciones...", pid)
        vf = load_vital(pid)
        if vf is None:
            log.warning("PASO 1 — %s: no hay VitalFile", pid)
            continue

        from q1_load import get_recording_duration_s
        dur = get_recording_duration_s(vf)

        anns = parse_annotations(pid, vf)
        total_raw += len(anns)

        warnings = validate_annotations(pid, anns, dur, log)
        all_warnings.extend(warnings)

        drug_df = build_drug_timeseries(pid, anns, dur)
        all_drug_dfs.append(drug_df)

        for ann in anns:
            by_category[ann.category] = by_category.get(ann.category, 0) + 1
            key = f"{ann.category}/{ann.subcategory}"
            by_subcat[key] = by_subcat.get(key, 0) + 1
            all_anns.append({
                "patient_id":    ann.patient_id,
                "t_seconds":     ann.t_s,
                "raw_text":      ann.raw_text,
                "category":      ann.category,
                "subcategory":   ann.subcategory,
                "seq_num":       ann.seq_num,
                "dose":          ann.dose,
                "dose_unit":     ann.dose_unit,
                "propo_target":  ann.propo_target,
                "remi_target":   ann.remi_target,
            })

    annotations_df = pd.DataFrame(all_anns) if all_anns else pd.DataFrame()
    drug_ts_df     = pd.concat(all_drug_dfs, ignore_index=True) if all_drug_dfs else pd.DataFrame()

    # Resumen en log
    log.info("── PASO 1 RESUMEN ──────────────────────────────")
    log.info("Total anotaciones procesadas: %d", total_raw)
    for cat, cnt in sorted(by_category.items(), key=lambda x: -x[1]):
        log.info("  %-25s %4d", cat, cnt)
    log.info("Subcategorías:")
    for k, cnt in sorted(by_subcat.items(), key=lambda x: -x[1]):
        log.info("  %-40s %4d", k, cnt)
    log.info("Advertencias: %d", len(all_warnings))

    # Pacientes sin estímulos
    if not annotations_df.empty:
        stim_patients = set(
            annotations_df.loc[annotations_df["category"] == "estimulo", "patient_id"]
        )
        for pid in patients:
            if pid not in stim_patients:
                msg = f"{pid}: SIN estímulos en annotations_df"
                log.warning(msg)
                all_warnings.append(msg)

    return annotations_df, drug_ts_df, all_warnings


# ---------------------------------------------------------------------------
# Guardar outputs
# ---------------------------------------------------------------------------

def save_paso1(annotations_df: pd.DataFrame, drug_ts_df: pd.DataFrame) -> None:
    """Guarda los outputs del PASO 1 en RESULTS_DIR."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ann_path  = RESULTS_DIR / "annotations_normalized.csv"
    drug_path = RESULTS_DIR / "drug_timeseries.parquet"

    if not annotations_df.empty:
        annotations_df.to_csv(ann_path, index=False)
        log.info("Guardado: %s (%d filas)", ann_path, len(annotations_df))
    else:
        log.warning("annotations_df vacío; no se guardó CSV")

    if not drug_ts_df.empty:
        drug_ts_df.to_parquet(drug_path, index=False)
        log.info("Guardado: %s (%d filas)", drug_path, len(drug_ts_df))
    else:
        log.warning("drug_ts_df vacío; no se guardó parquet")
