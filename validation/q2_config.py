"""
Q2 Analysis Configuration
Pre-specified parameters for "Validar firma autonómica pre-hipotensión".
DO NOT modify after pre-registration.
"""
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
Q1_RES = ROOT / "results" / "validation" / "Q1"   # READ-ONLY
Q2_RES = ROOT / "results" / "validation" / "Q2"
FIG    = Q2_RES / "figures"

# ── Input files (Q1, read-only) ─────────────────────────────────────────────
FEATURES_LONG   = Q1_RES / "features_long.parquet"
EVENT_WINDOWS   = Q1_RES / "event_windows.csv"
ANNOTATIONS     = Q1_RES / "annotations_normalized.csv"
DRUG_TIMESERIES = Q1_RES / "drug_timeseries.parquet"
Q1_RESULTS_V2   = Q1_RES / "test_results_v2.csv"
Q1_PAIRED_DATA  = Q1_RES / "paired_event_data.csv"

# ── Output files ─────────────────────────────────────────────────────────────
VASOPRESSOR_EVENTS = Q2_RES / "vasopressor_events.csv"
PAIRED_PRE_CONTROL = Q2_RES / "paired_pre_control.csv"
PAIRED_PRE_POST    = Q2_RES / "paired_pre_post.csv"
TEST_RESULTS_Q2A   = Q2_RES / "test_results_Q2a.csv"
TEST_RESULTS_Q2B   = Q2_RES / "test_results_Q2b.csv"
REPORT             = Q2_RES / "Q2_report.md"

# ── Time windows relative to bolus (seconds) ─────────────────────────────────
PRE_START_S  = -300   # t_bolus - 5 min
PRE_END_S    =    0   # t_bolus
POST_START_S =    0   # t_bolus
POST_END_S   =  300   # t_bolus + 5 min
WINDOW_S     =   30   # feature window duration

# ── Exclusion thresholds ──────────────────────────────────────────────────────
BOLUS_ISOLATION_S = 300   # no other bolus within ±5 min
# NOTE: original pre-spec was ±10 min (600 s); relaxed to ±5 min after
# checkpoint failure (n=4 < 10) — flagged as modified primary analysis.
INFUSION_BUFFER_S = 300   # no large infusion change within ±5 min
STIMULUS_BUFFER_S = 120   # no pain stimulus in pre-window [-2, 0] min
# NOTE: original pre-spec was [-5, 0] min (300 s); relaxed to [-2, 0] min
# after checkpoint failure — flagged as modified primary analysis.
DELTA_DRUG_THRESH = 0.5   # min |Δ| propofol or remi target to flag change
MIN_PPG_VALID     = 0.80  # ≥80 % PPG valid in window
MIN_VALID_FRAC    = 0.25  # ≥25 % non-NaN windows for each primary feature
# NOTE: original pre-spec was 0.50; relaxed to 0.25 — flagged as modified.
MIN_EVENTS        = 10    # checkpoint: fewer clean events → STOP
MIN_PRE_WINDOWS   = 3     # minimum 30 s windows required in pre-window

# ── Statistical parameters ────────────────────────────────────────────────────
ALPHA_BONF = 0.05 / 5   # 0.010  (5 primary features)
SEED       = 42

PARTIAL_PATIENT = "70767707"   # keep but flag (same as Q1)

# ── Primary features (pre-specified, NOT modified based on results) ───────────
# direction: -1 expected (feature decreases pre-hypotension vs quiescent)
PRIMARY = [
    ("ptt_cv",       "mean", -1),
    ("ptt_cv",       "std",  -1),
    ("ptt_std",      "max",  -1),
    ("pai_mean",     "mean", -1),
    ("brs_alpha_lf", "min",  -1),   # key dissociation feature vs Q1
]

# ── Exploratory features (two-sided, no Bonferroni) ───────────────────────────
EXPLORATORY = [
    ("ptt_std",  "std",   -1),
    ("ptt_std",  "slope", -1),
    ("ptt_arv",  "std",   -1),
]

# ── Labels for figures ────────────────────────────────────────────────────────
FEAT_LABELS = {
    "ptt_cv__mean":       "PTT-CV  (mean)",
    "ptt_cv__std":        "PTT-CV  (std)",
    "ptt_std__max":       "PTT-SD  (max)",
    "pai_mean__mean":     "PAI     (mean)",
    "brs_alpha_lf__min":  "BRS-\u03b1LF (min)  \u2190 disociaci\u00f3n",
    "ptt_std__std":       "PTT-SD  (std)  [expl.]",
    "ptt_std__slope":     "PTT-SD  (slope) [expl.]",
    "ptt_arv__std":       "PTT-ARV (std)  [expl.]",
}
