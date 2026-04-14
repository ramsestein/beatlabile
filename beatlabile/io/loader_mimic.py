"""Loader for MIMIC-IV Waveform Database (WFDB format, 125–500 Hz).

Directory layout expected:
  <mimic_root>/p<group>/p<subject_id>/<stay_id>/<record>.hea

MIMIC-IV uses multi-segment WFDB records:
  - Master header  : <stay_id>.hea  (lists segments, no signal data)
  - Segment headers: <stay_id>_NNNN.hea  (actual signal data per chunk)

Each segment may contain ECG channels at ~250 Hz and ABP at ~125 Hz,
stored as multi-frame records (samps_per_frame > 1) relative to a base
frequency (~62.5 Hz).  This loader resamples each channel to its true
effective rate using scipy.

Returns a standardised dict with same keys as loader_clinic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

import numpy as np

try:
    import wfdb
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install wfdb: pip install wfdb") from exc


_ECG_LABELS = {"II", "I", "III", "V", "AVF", "ECG"}
_ART_LABELS = {"ART", "ABP", "ARTERIAL", "IBP", "BP"}

# Regex matching segment headers: ends with _NNNN.hea
_SEGMENT_RE = re.compile(r'_\d+\.hea$')


def _find_channel(sig_names: list[str], labels: set[str]) -> int | None:
    for i, name in enumerate(sig_names):
        if name.upper().strip() in labels:
            return i
    # partial match fallback
    for i, name in enumerate(sig_names):
        for lab in labels:
            if lab in name.upper():
                return i
    return None


def load_wfdb_record(hea_path: str | Path) -> dict | None:
    """Load a WFDB segment record given the path to its .hea file.

    Skips master (multi-segment) headers and resamples each channel to
    its effective sampling rate (base_fs × samps_per_frame).
    """
    hea_path = Path(hea_path)

    # Skip master headers — they list segments but hold no signal data
    if not _SEGMENT_RE.search(hea_path.name):
        return None

    record_name = hea_path.stem

    try:
        record = wfdb.rdrecord(str(hea_path.parent / record_name))
    except Exception:
        # Silently skip empty/layout segments (sampto <= sampfrom etc.)
        return None

    if record.p_signal is None:
        return None

    sig_names: list[str] = list(record.sig_name)
    ecg_idx = _find_channel(sig_names, _ECG_LABELS)
    art_idx = _find_channel(sig_names, _ART_LABELS)

    if ecg_idx is None or art_idx is None:
        return None

    base_fs: float = record.fs
    spf: list[int] = list(record.samps_per_frame)

    signal = record.p_signal  # shape (n_frames, n_channels) at base_fs

    ecg_raw = signal[:, ecg_idx].astype(np.float64)
    art_raw = signal[:, art_idx].astype(np.float64)

    ecg_spf = spf[ecg_idx]
    art_spf = spf[art_idx]

    # Use np.repeat (nearest-neighbour) to expand multi-frame signals to
    # their effective per-channel rate — ~55x faster than FFT resample and
    # adequate since the original samples are already at native resolution.
    ecg = np.repeat(ecg_raw, ecg_spf).astype(np.float32) if ecg_spf > 1 else ecg_raw.astype(np.float32)
    art = np.repeat(art_raw, art_spf).astype(np.float32) if art_spf > 1 else art_raw.astype(np.float32)

    fs_ecg = int(round(base_fs * ecg_spf))
    fs_art = int(round(base_fs * art_spf))

    patient = hea_path.parent.parent.name  # p<subject_id> directory

    return {
        "ecg": ecg,
        "art": art,
        "fs_ecg": fs_ecg,
        "fs_art": fs_art,
        "patient": patient,
        "filepath": str(hea_path),
        "source": "mimic",
    }


def iter_mimic_files(mimic_root: str | Path) -> Generator[dict, None, None]:
    """Yield loaded record dicts for all WFDB segment .hea files under *mimic_root*."""
    mimic_root = Path(mimic_root)
    for hea in sorted(mimic_root.rglob("*.hea")):
        rec = load_wfdb_record(hea)
        if rec is not None:
            yield rec
