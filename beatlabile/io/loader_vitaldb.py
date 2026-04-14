"""Loader for VitalDB .vital files (500 Hz, same format as Clínic).

VitalDB files are located in:
  <vitaldb_root>/<caseid>.vital

Clinical metadata is saved to:
  datasets/data/vitaldb/clinical_data/cases.csv
  datasets/data/vitaldb/clinical_data/labs.csv
  datasets/data/vitaldb/clinical_data/caseids_art_ecg.csv

Run scripts/download_vitaldb_clinical.py once to populate these files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd

from beatlabile.io.loader_clinic import load_vital_file

# Default location of downloaded clinical data (relative to repo root)
_CLINICAL_DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "data" / "vitaldb" / "clinical_data"


def iter_vitaldb_files(vitaldb_root: str | Path) -> Generator[dict, None, None]:
    """Yield loaded record dicts for all .vital files in *vitaldb_root*."""
    vitaldb_root = Path(vitaldb_root)
    for p in sorted(vitaldb_root.glob("*.vital")):
        rec = load_vital_file(p)
        if rec is not None:
            rec["source"] = "vitaldb"
            rec["caseid"] = p.stem
            rec["patient"] = p.stem
            yield rec


def load_clinical_info(
    vitaldb_root: str | Path | None = None,
    clinical_data_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """Load VitalDB clinical metadata (cases.csv).

    Searches in this order:
      1. ``clinical_data_dir`` if provided
      2. ``<vitaldb_root>/clinical_data/`` (legacy)
      3. ``datasets/data/vitaldb/clinical_data/`` (default download location)
    """
    candidates = []
    if clinical_data_dir:
        candidates.append(Path(clinical_data_dir) / "cases.csv")
    if vitaldb_root:
        candidates.append(Path(vitaldb_root) / "clinical_data" / "cases.csv")
        candidates.append(Path(vitaldb_root) / "clinical_information.csv")
    candidates.append(_CLINICAL_DATA_DIR / "cases.csv")

    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


def load_labs(
    vitaldb_root: str | Path | None = None,
    clinical_data_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    """Load VitalDB pre-operative laboratory results (labs.csv)."""
    candidates = []
    if clinical_data_dir:
        candidates.append(Path(clinical_data_dir) / "labs.csv")
    if vitaldb_root:
        candidates.append(Path(vitaldb_root) / "clinical_data" / "labs.csv")
        candidates.append(Path(vitaldb_root) / "labs.csv")
    candidates.append(_CLINICAL_DATA_DIR / "labs.csv")

    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


def load_caseids_art_ecg(
    clinical_data_dir: str | Path | None = None,
) -> list[int] | None:
    """Return list of caseIDs that have both ART and ECG waveforms."""
    path = Path(clinical_data_dir) / "caseids_art_ecg.csv" if clinical_data_dir else None
    default = _CLINICAL_DATA_DIR / "caseids_art_ecg.csv"
    for p in [path, default]:
        if p is not None and p.exists():
            return pd.read_csv(p)["caseid"].tolist()
    return None

