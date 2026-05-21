#!/usr/bin/env python3
"""Download VitalDB public clinical metadata and lab data.

Saves to datasets/data/vitaldb/clinical_data/:
  - cases.csv       — demographics, ASA, surgery type, comorbidities, outcomes
  - labs.csv        — pre-operative laboratory results (34 analytes)
  - tracks.csv      — track list (used to identify cases with ART+ECG)
  - caseids_art_ecg.csv — caseIDs that have both ART and ECG waveforms

Usage:
    python scripts/download_vitaldb_clinical.py
"""

import sys
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.vitaldb.net"
OUT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "data" / "vitaldb" / "clinical_data"

# Tracks required for Act 3 signal processing
REQUIRED_TRACKS = ["ART", "ECG_II"]


def _download_csv(endpoint: str, label: str) -> pd.DataFrame:
    url = f"{API_URL}/{endpoint}"
    print(f"  Downloading {label} from {url} ...", flush=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    print(f"    → {len(df):,} rows, {len(df.columns)} columns")
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    # ------------------------------------------------------------------ #
    # 1. Clinical information (cases)
    # ------------------------------------------------------------------ #
    cases_path = OUT_DIR / "cases.csv"
    if cases_path.exists():
        print(f"  cases.csv already exists ({cases_path.stat().st_size // 1024} KB) — skipping download.")
        cases = pd.read_csv(cases_path)
    else:
        cases = _download_csv("cases", "clinical cases")
        cases.to_csv(cases_path, index=False)
        print(f"  Saved → {cases_path}")

    # ------------------------------------------------------------------ #
    # 2. Laboratory results
    # ------------------------------------------------------------------ #
    labs_path = OUT_DIR / "labs.csv"
    if labs_path.exists():
        print(f"  labs.csv already exists ({labs_path.stat().st_size // 1024} KB) — skipping download.")
        labs = pd.read_csv(labs_path)
    else:
        labs = _download_csv("labs", "laboratory results")
        labs.to_csv(labs_path, index=False)
        print(f"  Saved → {labs_path}")

    # ------------------------------------------------------------------ #
    # 3. Track list (to identify cases with ART + ECG)
    # ------------------------------------------------------------------ #
    tracks_path = OUT_DIR / "tracks.csv"
    if tracks_path.exists():
        print(f"  tracks.csv already exists ({tracks_path.stat().st_size // 1024} KB) — skipping download.")
        trks = pd.read_csv(tracks_path)
    else:
        trks = _download_csv("trks", "track list")
        trks.to_csv(tracks_path, index=False)
        print(f"  Saved → {tracks_path}")

    # ------------------------------------------------------------------ #
    # 4. Derive caseids with ART + ECG
    # ------------------------------------------------------------------ #
    caseids_path = OUT_DIR / "caseids_art_ecg.csv"
    art_labels = ["ART", "ABP", "IBP1", "ARTERIAL"]
    ecg_labels = ["ECG_II", "ECG_I", "ECG_III", "ECGII"]

    tname_col = "tname" if "tname" in trks.columns else trks.columns[1]
    caseid_col = "caseid" if "caseid" in trks.columns else trks.columns[0]

    has_art = trks[trks[tname_col].str.contains("|".join(art_labels), case=False, na=False)][caseid_col]
    has_ecg = trks[trks[tname_col].str.contains("|".join(ecg_labels), case=False, na=False)][caseid_col]
    both = sorted(set(has_art) & set(has_ecg))

    caseids_df = pd.DataFrame({"caseid": both})
    caseids_df.to_csv(caseids_path, index=False)
    print(f"\n  Cases with ART+ECG: {len(both):,} → {caseids_path}")

    # ------------------------------------------------------------------ #
    # 5. Summary
    # ------------------------------------------------------------------ #
    print("\n=== Download complete ===")
    print(f"  cases.csv  : {len(cases):,} patients")
    print(f"  labs.csv   : {len(labs):,} lab records")
    print(f"  tracks.csv : {len(trks):,} track entries")
    print(f"  ART+ECG caseids : {len(both):,}")

    # Show available clinical columns relevant to Act 3 sufficiency test
    act3_cols = [
        "caseid", "age", "sex", "weight", "height", "bmi",
        "asa", "optype", "opname", "department", "emop",
        "preop_htn", "preop_dm", "death_inhosp", "icu_days",
    ]
    available = [c for c in act3_cols if c in cases.columns]
    missing = [c for c in act3_cols if c not in cases.columns]
    print(f"\n  Act 3 clinical cols available : {available}")
    if missing:
        print(f"  Act 3 clinical cols MISSING  : {missing}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
