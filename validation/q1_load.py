"""
q1_load.py
==========
Carga de archivos .vital y datos clínicos (Excel) para la cohorte Q1.

Provee:
  - find_vital_file(patient_id) → Path | None
  - load_vital(patient_id)      → vitaldb.VitalFile | None
  - load_clinical_db()          → pd.DataFrame
  - get_patient_group(patient_id) → str  ('interescalenico' | 'supra_axilar')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from q1_config import (
    DATA_DIR,
    EXCEL_DB,
    EXCLUIDOS,
    GRUPO_BLOQUEO,
    GROUP_LABEL,
    INCLUIDOS_CON_PARCIAL,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Localización de archivos .vital
# ---------------------------------------------------------------------------

def find_vital_file(patient_id: str) -> Optional[Path]:
    """Busca el archivo .vital de un paciente en DATA_DIR.

    Estrategia:
      1. Buscar carpeta cuyo nombre empieza por patient_id (strip 'mal', espacios)
      2. Dentro de esa carpeta, devolver el primer .vital encontrado
    Returns None si no existe o el paciente está excluido.
    """
    if patient_id in EXCLUIDOS:
        log.debug("Paciente %s excluido → omitiendo", patient_id)
        return None

    for folder in DATA_DIR.iterdir():
        if not folder.is_dir():
            continue
        # Normalizar nombre de carpeta: quitar " mal", espacios y comparar inicio
        folder_norm = folder.name.replace(" mal", "").strip()
        if folder_norm == patient_id or folder_norm.startswith(patient_id):
            vitals = list(folder.glob("*.vital"))
            if not vitals:
                log.warning("Carpeta %s encontrada pero sin .vital", folder.name)
                return None
            if len(vitals) > 1:
                log.warning(
                    "Paciente %s: múltiples .vital %s → usando el primero",
                    patient_id, [v.name for v in vitals],
                )
            return vitals[0]

    log.warning("No se encontró carpeta para paciente %s en %s", patient_id, DATA_DIR)
    return None


def load_vital(patient_id: str):
    """Carga y devuelve un VitalFile para el paciente dado.
    Retorna None si no hay archivo o hay error de carga.
    """
    try:
        import vitaldb
    except ImportError as exc:
        raise ImportError("vitaldb no instalado: pip install vitaldb") from exc

    path = find_vital_file(patient_id)
    if path is None:
        return None
    try:
        vf = vitaldb.VitalFile(str(path))
        log.debug("Cargado %s (%s)", patient_id, path.name)
        return vf
    except Exception as exc:
        log.error("Error cargando %s (%s): %s", patient_id, path.name, exc)
        return None


# ---------------------------------------------------------------------------
# Datos clínicos desde Excel
# ---------------------------------------------------------------------------

def load_clinical_db() -> pd.DataFrame:
    """Carga el Excel de base de datos clínica y normaliza columnas clave.

    Columnas relevantes esperadas (pueden variar; se buscan por aproximación):
      - ID / N° paciente
      - Bloqueo interescalénico (bool/texto)
      - Edad, Peso, Talla, BMI
      - HTA (antecedente)
      - Posición quirúrgica (silla playa, DLI, DLD)
    """
    if not EXCEL_DB.exists():
        log.warning("Excel clínico no encontrado: %s", EXCEL_DB)
        return pd.DataFrame()

    try:
        df = pd.read_excel(EXCEL_DB, engine="openpyxl", header=0)
    except Exception as exc:
        log.error("Error leyendo Excel: %s", exc)
        return pd.DataFrame()

    # Normalizar nombres de columna
    df.columns = [str(c).strip() for c in df.columns]

    # Intentar identificar columna de ID
    id_candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ["id", "paciente", "nro", "n°", "num", "código", "codigo"]
    )]
    if id_candidates:
        df = df.rename(columns={id_candidates[0]: "patient_id_raw"})
        df["patient_id"] = df["patient_id_raw"].astype(str).str.strip().str.split(".").str[0]
    else:
        log.warning("No se encontró columna de ID en Excel; añadir manualmente")

    # Identificar columna de grupo
    interesc_cands = [c for c in df.columns if any(
        kw in c.lower() for kw in ["interescal", "bloqueo", "tipo"]
    )]
    if interesc_cands:
        col = interesc_cands[0]
        df["grupo"] = df[col].apply(
            lambda x: "interescalenico" if _is_truthy(x) else "supra_axilar"
        )
    else:
        # Construir desde diccionario de config
        df["grupo"] = df.get("patient_id", pd.Series(dtype=str)).map(
            lambda pid: GROUP_LABEL.get(GRUPO_BLOQUEO.get(str(pid)), "desconocido")
        )

    log.info("Excel cargado: %d filas, %d columnas", len(df), len(df.columns))
    return df


def _is_truthy(val) -> bool:
    """Interpreta celdas Excel como booleano."""
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in {"1", "sí", "si", "yes", "true", "x", "✓", "v"}


# ---------------------------------------------------------------------------
# Utilidades de grupo
# ---------------------------------------------------------------------------

def get_patient_group(patient_id: str) -> str:
    """Retorna 'interescalenico' o 'supra_axilar'."""
    val = GRUPO_BLOQUEO.get(patient_id)
    if val is None:
        log.warning("Paciente %s no en GRUPO_BLOQUEO", patient_id)
        return "desconocido"
    return GROUP_LABEL[val]


def list_patients(include_partial: bool = True) -> list[str]:
    """Lista pacientes incluidos en el estudio."""
    if include_partial:
        return sorted(INCLUIDOS_CON_PARCIAL)
    from q1_config import INCLUIDOS_PLENOS
    return sorted(INCLUIDOS_PLENOS)


# ---------------------------------------------------------------------------
# Utilidades de VitalFile
# ---------------------------------------------------------------------------

def get_track_names(vf) -> list[str]:
    """Devuelve lista de nombres de tracks en el VitalFile."""
    if not hasattr(vf, "trks"):
        return []
    return list(vf.trks.keys())


def get_recording_duration_s(vf) -> float:
    """Duración total de grabación en segundos."""
    try:
        dtstart = float(getattr(vf, "dtstart", 0) or 0)
        dtend   = float(getattr(vf, "dtend",   0) or 0)
        diff = dtend - dtstart
        if diff > 0:
            return diff
    except Exception:
        pass
    return 0.0


def get_continuous_signal(vf, track_name: str) -> tuple[Optional["np.ndarray"], float]:
    """Carga señal continua como ndarray float32 y devuelve (signal, srate).
    Retorna (None, 0) si falla.
    """
    import numpy as np
    if not hasattr(vf, "trks") or track_name not in vf.trks:
        return None, 0.0
    trk   = vf.trks[track_name]
    srate = float(getattr(trk, "srate", 0) or 0)
    if srate == 0:
        return None, 0.0
    try:
        arr = vf.to_numpy(track_name, interval=1.0 / srate)
        if arr is None:
            return None, srate
        return np.asarray(arr, dtype=np.float32).ravel(), srate
    except Exception as exc:
        log.debug("to_numpy falló para %s: %s; intentando desde recs", track_name, exc)
    # Fallback: construir desde recs
    try:
        recs = getattr(trk, "recs", []) or []
        if recs:
            vals = [
                float(r["val"]) if isinstance(r, dict) else float(getattr(r, "val", float("nan")))
                for r in recs
            ]
            import numpy as np
            return np.array(vals, dtype=np.float32), srate
    except Exception:
        pass
    return None, srate


def pick_track(track_names: list[str], candidates: list[str]) -> Optional[str]:
    """Retorna el primer candidato presente en track_names."""
    for c in candidates:
        if c in track_names:
            return c
    return None


def get_events_raw(vf) -> list[dict]:
    """Extrae anotaciones (tipo string) del VitalFile como lista de dicts.
    Campos: time_s (desde inicio), text, track.
    """
    events: list[dict] = []
    if not hasattr(vf, "trks"):
        return events
    dtstart = float(getattr(vf, "dtstart", 0) or 0)
    for name, trk in vf.trks.items():
        ttype = int(getattr(trk, "type", 0) or 0)
        srate = float(getattr(trk, "srate", 0) or 0)
        is_event_track = (
            ttype == 5
            or (srate == 0 and any(
                kw in name.upper()
                for kw in ["EVENT", "NOTE", "CMD", "ANNOT", "MARK"]
            ))
        )
        if not is_event_track:
            continue
        try:
            recs = getattr(trk, "recs", []) or []
            for rec in recs:
                if isinstance(rec, dict):
                    dt_abs = float(rec.get("dt", dtstart) or dtstart)
                    val    = rec.get("val", "")
                else:
                    dt_abs = float(getattr(rec, "dt", dtstart) or dtstart)
                    val    = getattr(rec, "val", "")
                val_str = str(val).strip() if val is not None else ""
                if val_str:
                    events.append({
                        "time_s": dt_abs - dtstart,
                        "text":   val_str,
                        "track":  name,
                    })
        except Exception as exc:
            log.debug("Error leyendo recs de %s: %s", name, exc)
    return sorted(events, key=lambda e: e["time_s"])
