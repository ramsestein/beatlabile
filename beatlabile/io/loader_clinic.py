"""Loader for Clínic .vital files (VitalRecorder format, 500 Hz).

The Clínic dataset is organised as:
  <clinic_root>/box<N>/<boxdir>/<patid>/<date>/<filename>.vital

Returns a standardised dict with keys:
  ecg     : np.ndarray, shape (N,), µV or mV (raw units kept)
  art     : np.ndarray, shape (N,), mmHg
  fs_ecg  : int   — sampling frequency for ECG
  fs_art  : int   — sampling frequency for ART
  patient : str   — patient identifier
  source  : str   — 'clinic'
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Generator

import numpy as np

try:
    import vitaldb
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install vitaldb: pip install vitaldb") from exc


# Channel names used in VitalRecorder files at Hospital Clínic
_ECG_CANDIDATES = ["ECG_II", "ECG_I", "ECG_III", "ECGII", "ECGI"]
_ART_CANDIDATES = ["ART", "ABP", "IBP1", "ART_MBP", "ARTERIAL"]


def _pick_channel(track_names: list[str], candidates: list[str]) -> str | None:
    # Exact match first
    upper = {t.upper(): t for t in track_names}
    for c in candidates:
        if c.upper() in upper:
            return upper[c.upper()]
    # Suffix match: handles 'Intellivue/ECG_II' → matches 'ECG_II'
    suffix_map = {t.split("/")[-1].upper(): t for t in track_names}
    for c in candidates:
        if c.upper() in suffix_map:
            return suffix_map[c.upper()]
    return None


def load_vital_file(path: str | Path) -> dict | None:
    """Load a single .vital file.  Returns None if required channels are missing."""
    path = Path(path)
    try:
        vf = vitaldb.VitalFile(str(path))
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not open {path}: {exc}")
        return None

    track_names = list(vf.trks.keys()) if hasattr(vf, "trks") else []

    ecg_ch = _pick_channel(track_names, _ECG_CANDIDATES)
    art_ch = _pick_channel(track_names, _ART_CANDIDATES)

    if ecg_ch is None or art_ch is None:
        return None

    fs_ecg = int(vf.trks[ecg_ch].srate) if hasattr(vf.trks[ecg_ch], "srate") else 500
    fs_art = int(vf.trks[art_ch].srate) if hasattr(vf.trks[art_ch], "srate") else 500

    ecg = vf.to_numpy(ecg_ch, interval=1.0 / fs_ecg)
    art = vf.to_numpy(art_ch, interval=1.0 / fs_art)

    if ecg is None or art is None:
        return None

    ecg = np.asarray(ecg, dtype=np.float32).ravel()
    art = np.asarray(art, dtype=np.float32).ravel()

    patient = path.parent.name  # date directory parent is patient id
    return {
        "ecg": ecg,
        "art": art,
        "fs_ecg": fs_ecg,
        "fs_art": fs_art,
        "patient": patient,
        "filepath": str(path),
        "source": "clinic",
    }


def iter_clinic_files(
    clinic_root: str | Path,
) -> Generator[dict, None, None]:
    """Yield loaded record dicts for all valid .vital files under *clinic_root*."""
    clinic_root = Path(clinic_root)
    vital_paths = sorted(clinic_root.rglob("*.vital"))
    for p in vital_paths:
        rec = load_vital_file(p)
        if rec is not None:
            yield rec
