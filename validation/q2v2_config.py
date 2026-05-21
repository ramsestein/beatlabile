"""
Q2 v2 Analysis Configuration
──────────────────────────────
Pre-specified parameters for Q2 v2 (relaxed filters, BRS_seq primary feature).
Changes vs Q2 v1 — pre-specified BEFORE seeing v2 results:
  1. Remove infusion-change exclusion criterion for events
     (infusion change → covariable, not exclusion)
  2. Control window duration relaxed to 3 min (was 5 min)
  3. BRS_seq (sequence method) replaces brs_alpha_lf as primary feature
  4. BRS_seq also computed for Q1 events for dissociation figure
  5. Sensitivity C: adjust by delta_infusion covariables
"""
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
Q1_RES  = ROOT / "results" / "validation" / "Q1"   # READ-ONLY (except brs_seq additions)
Q2V2_RES = ROOT / "results" / "validation" / "Q2v2"
FIG      = Q2V2_RES / "figures"

# ── Input files (Q1, read-only) ──────────────────────────────────────────────
FEATURES_LONG    = Q1_RES / "features_long.parquet"
EVENT_WINDOWS    = Q1_RES / "event_windows.csv"
ANNOTATIONS      = Q1_RES / "annotations_normalized.csv"
DRUG_TIMESERIES  = Q1_RES / "drug_timeseries.parquet"
SIGNALS_CACHE    = Q1_RES / "signals_cache.pkl"
Q1_RESULTS_V2    = Q1_RES / "test_results_v2.csv"
Q1_PAIRED_DATA   = Q1_RES / "paired_event_data.csv"

# ── New Q1 output (append-only) ───────────────────────────────────────────────
BRS_SEQ_Q1_EVENTS = Q1_RES / "brs_seq_per_event.csv"   # new file, Q1 BRS_seq results

# ── Q2v2 output files ─────────────────────────────────────────────────────────
FEATURES_BRS_SEQ     = Q2V2_RES / "features_brs_seq.parquet"
VASOPRESSOR_EVENTS   = Q2V2_RES / "vasopressor_events_v2.csv"
PAIRED_PRE_CONTROL   = Q2V2_RES / "paired_pre_control_v2.csv"
PAIRED_PRE_POST      = Q2V2_RES / "paired_pre_post_v2.csv"
TEST_RESULTS_Q2A     = Q2V2_RES / "test_results_Q2a_v2.csv"
TEST_RESULTS_Q2B     = Q2V2_RES / "test_results_Q2b_v2.csv"
REPORT               = Q2V2_RES / "Q2_report_v2.md"

# ── Time windows relative to bolus (seconds) ──────────────────────────────────
PRE_START_S  = -300   # t_bolus - 5 min
PRE_END_S    =    0   # t_bolus
POST_START_S =    0   # t_bolus
POST_END_S   =  300   # t_bolus + 5 min
WINDOW_S     =   30   # feature window duration

# ── Event exclusion thresholds (v2) ───────────────────────────────────────────
BOLUS_ISOLATION_S  = 300   # no other vasopresor bolus within ±5 min (kept from v1)
# NOTE: original pre-spec ±10 min (600s); v1 relaxed to ±5 min (300s); kept for v2
STIMULUS_BUFFER_S  = 120   # no pain stimulus in pre-window [-2, 0] min (kept from v1)
MIN_PPG_VALID_PRE  = 0.70  # ≥70% valid PPG in pre-window (relaxed from 0.80 in v1)
MIN_VALID_FRAC     = 0.25  # ≥25% non-NaN windows per primary feature (kept from v1)
MIN_PRE_WINDOWS    = 3     # minimum 30s windows required in pre-window
# KEY CHANGE v2: infusion criterion REMOVED for events;
# delta_propofol_pre and delta_remi_pre added as covariables instead.
DELTA_DRUG_THRESH  = 0.5   # used for computing covariables (not as exclusion)

# ── Control window thresholds (v2) ────────────────────────────────────────────
CONTROL_DURATION_S        = 180   # 3 min (relaxed from 5 min in v1)
CONTROL_INFUSION_BUFFER_S = 180   # no infusion change within ±3 min of control (kept)
CONTROL_BOLUS_BUFFER_S    = 300   # no bolus within ±5 min of control start

# ── Statistical parameters ────────────────────────────────────────────────────
ALPHA_BONF = 0.05 / 5   # 0.010  (5 primary features)
SEED       = 42

PARTIAL_PATIENT = "70767707"   # kept from Q1 / Q2v1

# ── BRS_seq algorithm parameters ─────────────────────────────────────────────
BRS_SEQ_MIN_BEATS  = 3     # min consecutive beats per sequence (≥3 → ≥2 diffs)
BRS_SEQ_R_THRESH   = 0.6   # min |Pearson r| between RR and PTT in sequence
BRS_SEQ_SANITY_MIN = 0.05  # min fraction of windows with valid brs_seq to pass sanity

# ── Primary features v2 (pre-specified, BRS_seq replaces brs_alpha_lf) ────────
# direction: -1 expected (feature decreases pre-hypotension vs quiescent)
PRIMARY = [
    ("ptt_cv",   "mean", -1),
    ("ptt_cv",   "std",  -1),
    ("ptt_std",  "max",  -1),
    ("pai_mean", "mean", -1),
    ("brs_seq",  "min",  -1),   # KEY v2 change: sequence-method BRS (direct analogue BeatLabile)
]

# ── Exploratory features (two-sided, no Bonferroni) ───────────────────────────
EXPLORATORY = [
    ("brs_alpha_lf", "min",   -1),   # kept from v1 primary, now exploratory for comparison
    ("ptt_std",      "std",   -1),
    ("ptt_std",      "slope", -1),
    ("ptt_arv",      "std",   -1),
]

# ── Checkpoint ───────────────────────────────────────────────────────────────
MIN_EVENTS_CHECKPOINT = 12   # <12 → continue but document power limitation

# ── Labels for figures ────────────────────────────────────────────────────────
FEAT_LABELS = {
    "ptt_cv__mean":       "PTT-CV  (mean)",
    "ptt_cv__std":        "PTT-CV  (std)",
    "ptt_std__max":       "PTT-SD  (max)",
    "pai_mean__mean":     "PAI     (mean)",
    "brs_seq__min":       "BRS-seq (min)  ← clave",
    "brs_alpha_lf__min":  "BRS-\u03b1LF (min)  [expl.]",
    "ptt_std__std":       "PTT-SD  (std)  [expl.]",
    "ptt_std__slope":     "PTT-SD  (slope) [expl.]",
    "ptt_arv__std":       "PTT-ARV (std)  [expl.]",
}
