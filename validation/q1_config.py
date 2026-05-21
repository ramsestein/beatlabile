"""
q1_config.py
============
Constantes globales, listas de pacientes y definición de features pre-especificadas
para el sub-estudio Q1 (validación firma autonómica pre-hipertensión alrededor de
estímulos dolorosos quirúrgicos) — cohorte cirugía de hombro.

IMPORTANTE: Este archivo define el protocolo. No modificar después de pre-registro.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR    = REPO_ROOT / "pacientes vital recorder"
EXCEL_DB    = REPO_ROOT / "database general pacientes estudio hombro.xlsx"
RESULTS_DIR = REPO_ROOT / "results" / "validation" / "Q1"
FIGURES_DIR = RESULTS_DIR / "figures"
VALID_DIR   = REPO_ROOT / "validation"

# ---------------------------------------------------------------------------
# Semilla global
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42

# ---------------------------------------------------------------------------
# Cohorte — grupos (confirmado desde Excel auditado)
# True = bloqueo interescalénico, False = bloqueo supra+axilar
# ---------------------------------------------------------------------------
GRUPO_BLOQUEO: dict[str, bool] = {
    "230393":   False,   # supra+axilar
    "397651":   False,   # supra+axilar
    "4214722":  True,    # interescalenico
    "4247699":  True,    # interescalenico
    "4912692":  False,   # supra+axilar
    "5020549":  True,    # interescalenico
    "5362391":  True,    # interescalenico
    "5431482":  True,    # interescalenico
    "5582912":  True,    # interescalenico
    "5589679":  True,    # interescalenico
    "5684023":  True,    # interescalenico — EXCLUIDO PPG <70%
    "70078466": False,   # supra+axilar    — EXCLUIDO sin .vital
    "70288016": False,   # supra+axilar
    "70297385": True,    # interescalenico
    "70431992": False,   # supra+axilar
    "70436283": True,    # interescalenico
    "70551555": False,   # supra+axilar
    "70628874": False,   # supra+axilar
    "70767707": True,    # interescalenico — PARCIAL 23 min
    "720142":   True,    # interescalenico
    "4234018":  True,    # interescalenico — EXCLUIDO sin .vital
}

# Pacientes excluidos completamente
EXCLUIDOS = {
    "4234018",   # sin .vital
    "70078466",  # sin .vital
    "5684023",   # PPG <70%
}

# Paciente parcial (solo análisis intra-evento si ≥1 estímulo; excluir ventanas ≥30 min)
PARCIAL = {"70767707"}
PARCIAL_MAX_WINDOW_MIN = 30.0   # excluir de cualquier modelo que requiera ≥30 min contexto

# Pacientes finales incluidos (plenos + parcial)
INCLUIDOS_PLENOS = {
    pid for pid, grupo in GRUPO_BLOQUEO.items()
    if pid not in EXCLUIDOS and pid not in PARCIAL
}
INCLUIDOS_CON_PARCIAL = INCLUIDOS_PLENOS | PARCIAL

GROUP_LABEL = {True: "interescalenico", False: "supra_axilar"}

# ---------------------------------------------------------------------------
# Nombres de tracks en .vital (prioridad: primero → fallback)
# ---------------------------------------------------------------------------
ECG_TRACKS    = ["Intellivue/ECG_II", "Intellivue/ECG_I", "Intellivue/ECG_III", "Intellivue/ECG_V"]
PPG_TRACKS    = ["Intellivue/PLETH"]
NIBP_SBP      = ["Intellivue/NIBP_SYS", "Intellivue/NIBP_SBP"]
NIBP_DBP      = ["Intellivue/NIBP_DIA", "Intellivue/NIBP_DBP"]
NIBP_MAP      = ["Intellivue/NIBP_MEAN", "Intellivue/NIBP_MAP", "Intellivue/NIBP_MBP"]
EVENT_TRACK   = "EVENT"

ECG_SRATE     = 500.0   # Hz
PPG_SRATE     = 125.0   # Hz

# ---------------------------------------------------------------------------
# Preprocesado de señales
# ---------------------------------------------------------------------------
RR_MIN_MS   = 300.0    # < 300 ms → artefacto/ectópico
RR_MAX_MS   = 2000.0   # > 2000 ms → artefacto/bradicardia extrema
PTT_MIN_MS  = 100.0    # límites fisiológicos PTT (anestesia general: puede llegar a 600ms)
PTT_MAX_MS  = 600.0
PPG_REJECT_LOW_PCT  = 0.30   # amplitud < 30% mediana móvil → rechazar pulso
PPG_REJECT_HIGH_PCT = 3.00   # amplitud > 300% mediana móvil → rechazar pulso
PPG_MOVING_MEDIAN_S = 30.0   # ventana mediana móvil para PPG-PAI

# Umbral de calidad PPG para ventanas de control (≥80%)
PPG_QUALITY_MIN_CONTROL = 0.80

# ---------------------------------------------------------------------------
# Parámetros de features autonómicas
# ---------------------------------------------------------------------------
WINDOW_S    = 30    # segundos por ventana
WINDOW_STEP = 1     # paso en segundos

# Banda LF para BRS (Hz)
LF_BAND = (0.04, 0.15)
BRS_MIN_COHERENCE = 0.5   # coherencia mínima para BRS válido

# ---------------------------------------------------------------------------
# Definición de features primarias pre-especificadas
# ---------------------------------------------------------------------------
# Formato: (nombre_feature, agregacion, direccion_esperada_evento_vs_control)
# direccion: +1 → feature aumenta en evento, -1 → feature disminuye en evento
PRIMARY_FEATURES: list[tuple[str, str, int]] = [
    ("ptt_std",   "std",   +1),   # std-PA-std surrogate       → sube con simpático
    ("ptt_cv",    "std",   -1),   # cv-PA-std surrogate        → baja
    ("brs_alpha_lf", "min", -1),  # brs-min surrogate          → baja
    ("ptt_cv",    "mean",  -1),   # cv-PA-mean surrogate       → baja
    ("ptt_arv",   "std",   +1),   # arv-std surrogate          → sube
    ("ptt_std",   "slope", +1),   # std-PA-slope surrogate     → sube
    ("ptt_std",   "max",   -1),   # std-PA-max surrogate       → baja
]

# Nombre de columna resultante para cada feature primaria
def primary_col_name(feat: str, agg: str) -> str:
    return f"{feat}__{agg}"

PRIMARY_COL_NAMES = [primary_col_name(f, a) for f, a, _ in PRIMARY_FEATURES]

# ---------------------------------------------------------------------------
# Features exploratorias (no entran en Bonferroni)
# ---------------------------------------------------------------------------
EXPLORATORY_FEATURES: list[tuple[str, str, int]] = [
    ("hrv_sdnn",  "mean", -1),
    ("hrv_rmssd", "mean", -1),
    ("pai_mean",  "mean", -1),
]

# ---------------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------------
N_PRIMARY_TESTS   = len(PRIMARY_FEATURES)   # 7
ALPHA_NOMINAL     = 0.05
ALPHA_BONFERRONI  = ALPHA_NOMINAL / N_PRIMARY_TESTS   # ≈ 0.00714

# ---------------------------------------------------------------------------
# Definición de ventanas de evento y control
# ---------------------------------------------------------------------------
EVENT_WINDOW_S      = 5 * 60     # 5 minutos = 300 s
EXCLUSION_BUFFER_S  = 5 * 60     # buffer ±5 min para exclusión por vasopresor/infusión
DRUG_DELTA_EXCL     = 0.5        # cambio mínimo en propofol/remi que activa exclusión

# Subcategorías de estímulos dolorosos (definidas en q1_annotations.py)
STIMULUS_SUBCATEGORIES = {"piel", "trocar", "anclaje", "sutura", "fresa"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
LOG_DATE   = "%H:%M:%S"
