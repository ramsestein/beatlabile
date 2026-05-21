"""Global configuration loader for BeatLabile.

Reads config.yaml from the repository root and exposes a typed `CFG` dict.
Override any value by setting the env var BEATLABILE_CONFIG to an alternate path.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path or os.environ.get("BEATLABILE_CONFIG", _DEFAULT_CONFIG))
    with config_path.open() as fh:
        return yaml.safe_load(fh)


CFG: dict = load_config()

# Convenience shortcuts
DATA_CLINIC: Path = Path(CFG["data"]["clinic"])
DATA_MIMIC: Path = Path(CFG["data"]["mimic"])
DATA_VITALDB: Path = Path(CFG["data"]["vitaldb"])
RESULTS_DIR: Path = Path(CFG["results"]["dir"])

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
