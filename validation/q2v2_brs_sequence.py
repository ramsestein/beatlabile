"""
Q2 v2 – BRS Sequence Feature (PASO 1)
───────────────────────────────────────
Computes BRS_seq (baroreflex sensitivity by the PTT-RR sequence method) as the
direct PTT analogue of the BeatLabile original SBP-RR sequence method.

Physiological rationale:
  PTT ∝ 1/BP  →  PTT up means BP down (and vice versa)
  Baroreflex: BP ↑ (PTT ↓) → RR ↑  (parasympathetic slowing)
              BP ↓ (PTT ↑) → RR ↓  (sympathetic acceleration)
  ∴ in a valid baroreflex sequence: sign(ΔPTT) = -sign(ΔRR)

Algorithm (per 30 s sliding window, step 1 s):
  1. Identify candidate sequences: ≥3 consecutive beats where:
       • all ΔPTT[i] have the same sign, AND
       • all ΔRR[i] have the OPPOSITE sign to ΔPTT[i]
  2. For each candidate:
       • Compute linear slope (ΔRR/ΔPTT) via polyfit(ptt, rr, 1)
       • Compute Pearson r between RR and PTT values in the sequence
       • Keep if |r| ≥ BRS_SEQ_R_THRESH (0.6)
  3. brs_seq = median |slope| of kept sequences in the window
              (NaN if no valid sequence)

Output: results/validation/Q2v2/features_brs_seq.parquet
Columns: patient_id, t_window_start_s, brs_seq, n_sequences_window, n_sequences_valid

Sanity-check plots: figures/sanity_brs_sequences_<pid>.png for 3 random patients.
"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2v2_config import (
    SIGNALS_CACHE, FEATURES_LONG, FEATURES_BRS_SEQ, FIG,
    WINDOW_S, SEED,
    BRS_SEQ_MIN_BEATS, BRS_SEQ_R_THRESH, BRS_SEQ_SANITY_MIN,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Core sequence-detection algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_brs_seq_window(
    rr_vals: np.ndarray,
    ptt_vals: np.ndarray,
    min_beats: int = 3,
    r_thresh: float = 0.6,
) -> tuple[float, int, int]:
    """
    Compute BRS_seq for a single window.
    Returns (brs_seq, n_seq_candidates, n_seq_valid).
    brs_seq = median |slope| (ms_RR / ms_PTT) from valid sequences.
    """
    n = len(rr_vals)
    if n < min_beats:
        return np.nan, 0, 0

    drr  = np.diff(rr_vals.astype(float))
    dptt = np.diff(ptt_vals.astype(float))

    # Mask out zero-change steps (undefined sign)
    nonzero = (drr != 0.0) & (dptt != 0.0)

    slopes_valid = []
    n_cand = 0

    i = 0
    n_diffs = len(drr)
    while i < n_diffs:
        # Skip if not nonzero or same sign (not opposite)
        if not nonzero[i] or (np.sign(drr[i]) * np.sign(dptt[i]) != -1):
            i += 1
            continue

        # Start of a potential sequence — record expected PTT direction
        ptt_dir = np.sign(dptt[i])
        j = i  # j = last diff index included in sequence

        while (j + 1 < n_diffs
               and nonzero[j + 1]
               and np.sign(drr[j + 1]) * np.sign(dptt[j + 1]) == -1
               and np.sign(dptt[j + 1]) == ptt_dir):
            j += 1

        # beats in sequence: indices i through j+1 inclusive (j-i+2 beats)
        n_beats_seq = j - i + 2
        if n_beats_seq >= min_beats:
            n_cand += 1
            seq_rr  = rr_vals[i: j + 2]
            seq_ptt = ptt_vals[i: j + 2]
            if len(seq_rr) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r_val, _ = pearsonr(seq_ptt, seq_rr)
                if np.isfinite(r_val) and abs(r_val) >= r_thresh:
                    try:
                        slope, _ = np.polyfit(seq_ptt, seq_rr, 1)
                        if np.isfinite(slope):
                            slopes_valid.append(abs(slope))
                    except Exception:
                        pass

        # Advance: allow next sequence to start from beat j+1
        i = j + 1

    if not slopes_valid:
        return np.nan, n_cand, 0
    return float(np.median(slopes_valid)), n_cand, len(slopes_valid)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-patient computation
# ═══════════════════════════════════════════════════════════════════════════════

def _align_rr_ptt(rr_series, ptt_series, tol: float = 0.15) -> pd.DataFrame:
    """
    Align RR and PTT beat-by-beat data by R-peak timestamp.
    Both are indexed by the same R-peaks so a nearest merge with tolerance works.
    Returns DataFrame with columns [t, rr, ptt].
    """
    df_rr  = pd.DataFrame({"t": rr_series.t_rr_s,  "rr":  rr_series.rr_ms})
    df_ptt = pd.DataFrame({"t": ptt_series.t_s,     "ptt": ptt_series.ptt_ms})

    df_rr  = df_rr[np.isfinite(df_rr["rr"])].sort_values("t").reset_index(drop=True)
    df_ptt = df_ptt[np.isfinite(df_ptt["ptt"])].sort_values("t").reset_index(drop=True)

    merged = pd.merge_asof(
        df_rr, df_ptt, on="t", tolerance=tol, direction="nearest"
    )
    return merged.dropna(subset=["rr", "ptt"]).reset_index(drop=True)


def compute_brs_seq_patient(
    rr_series,
    ptt_series,
    t_win_starts: np.ndarray,
    window_s: float = 30.0,
    min_beats: int = BRS_SEQ_MIN_BEATS,
    r_thresh: float = BRS_SEQ_R_THRESH,
) -> pd.DataFrame:
    """
    Compute BRS_seq for all windows of a single patient.
    t_win_starts: array of window start times (seconds) — taken from features_long.
    """
    beats = _align_rr_ptt(rr_series, ptt_series)
    if beats.empty:
        rows = [{"t_window_start_s": t, "brs_seq": np.nan,
                 "n_sequences_window": 0, "n_sequences_valid": 0}
                for t in t_win_starts]
        return pd.DataFrame(rows)

    t_arr  = beats["t"].values
    rr_arr  = beats["rr"].values
    ptt_arr = beats["ptt"].values

    rows = []
    for t_start in t_win_starts:
        t_end = t_start + window_s
        idx   = np.searchsorted(t_arr, [t_start, t_end])
        lo, hi = idx[0], idx[1]

        if hi - lo < min_beats:
            rows.append({
                "t_window_start_s": float(t_start),
                "brs_seq": np.nan,
                "n_sequences_window": 0,
                "n_sequences_valid": 0,
            })
            continue

        brs, n_c, n_v = _compute_brs_seq_window(
            rr_arr[lo:hi], ptt_arr[lo:hi], min_beats, r_thresh
        )
        rows.append({
            "t_window_start_s": float(t_start),
            "brs_seq":          brs,
            "n_sequences_window": n_c,
            "n_sequences_valid":  n_v,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Sanity-check plots
# ═══════════════════════════════════════════════════════════════════════════════

def _sanity_plot(pid: str, rr_series, ptt_series, out_dir: Path):
    """
    For a single patient, plot a 60 s segment with:
      - RR tachogram (beat-by-beat)
      - PTT (beat-by-beat)
      - Detected sequences highlighted
      - BRS_seq value for that window
    """
    beats = _align_rr_ptt(rr_series, ptt_series)
    if beats.empty or len(beats) < 10:
        print(f"  [SANITY] {pid}: no aligned beats, skipping plot")
        return

    # Pick a 60 s window from the middle of the recording
    t_mid = beats["t"].median()
    t_lo  = t_mid - 30.0
    t_hi  = t_mid + 30.0
    seg   = beats[(beats["t"] >= t_lo) & (beats["t"] <= t_hi)].copy()

    if len(seg) < BRS_SEQ_MIN_BEATS:
        print(f"  [SANITY] {pid}: segment too short, skipping plot")
        return

    rr_v  = seg["rr"].values
    ptt_v = seg["ptt"].values
    t_v   = seg["t"].values

    # Detect sequences in this segment
    drr   = np.diff(rr_v)
    dptt  = np.diff(ptt_v)

    def _seq_label(rr_a, ptt_a, mb, rt):
        """Return list of (start_beat_idx, end_beat_idx) for valid sequences."""
        seqs = []
        n_d  = len(rr_a) - 1
        if n_d < mb - 1:
            return seqs
        drr_  = np.diff(rr_a.astype(float))
        dptt_ = np.diff(ptt_a.astype(float))
        nz    = (drr_ != 0.0) & (dptt_ != 0.0)
        i = 0
        while i < n_d:
            if not nz[i] or np.sign(drr_[i]) * np.sign(dptt_[i]) != -1:
                i += 1; continue
            ptt_dir = np.sign(dptt_[i])
            j = i
            while (j + 1 < n_d and nz[j+1]
                   and np.sign(drr_[j+1]) * np.sign(dptt_[j+1]) == -1
                   and np.sign(dptt_[j+1]) == ptt_dir):
                j += 1
            if j - i + 2 >= mb:
                seq_rr_  = rr_a[i:j+2]
                seq_ptt_ = ptt_a[i:j+2]
                if len(seq_rr_) >= 3:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r_v_, _ = pearsonr(seq_ptt_, seq_rr_)
                    if np.isfinite(r_v_) and abs(r_v_) >= rt:
                        seqs.append((i, j + 1))
            i = j + 1
        return seqs

    seqs = _seq_label(rr_v, ptt_v, BRS_SEQ_MIN_BEATS, BRS_SEQ_R_THRESH)
    brs_val, n_c, n_v = _compute_brs_seq_window(rr_v, ptt_v, BRS_SEQ_MIN_BEATS, BRS_SEQ_R_THRESH)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1.plot(t_v, rr_v, color="#4e79a7", lw=1.2, label="RR (ms)")
    ax2.plot(t_v, ptt_v, color="#f28e2b", lw=1.2, label="PTT (ms)")

    for s_lo, s_hi in seqs:
        ax1.axvspan(t_v[s_lo], t_v[s_hi], alpha=0.25, color="green", zorder=0)
        ax2.axvspan(t_v[s_lo], t_v[s_hi], alpha=0.25, color="green", zorder=0)

    brs_str = f"{brs_val:.3f}" if np.isfinite(brs_val) else "NaN"
    ax1.set_title(
        f"Patient {pid}  –  BRS_seq sanity check\n"
        f"60 s window  |  brs_seq = {brs_str} ms/ms  "
        f"|  {n_v}/{n_c} valid sequences",
        fontsize=9,
    )
    ax1.set_ylabel("RR (ms)", fontsize=8)
    ax2.set_ylabel("PTT (ms)", fontsize=8)
    ax2.set_xlabel("Time (s)", fontsize=8)

    for ax in (ax1, ax2):
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out_path = out_dir / f"sanity_brs_sequences_{pid}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SANITY] saved {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_brs_seq() -> pd.DataFrame:
    """
    Compute BRS_seq for every patient and merge with features_long.
    Returns merged DataFrame with brs_seq added.

    Saves:
      FEATURES_BRS_SEQ    (parquet, same rows as features_long + brs_seq cols)
      figures/sanity_brs_sequences_<pid>.png  (3 random patients)
    """
    print("\n[PASO 1] Computing BRS_seq (sequence method) …")

    # Load signals cache
    if not SIGNALS_CACHE.exists():
        raise FileNotFoundError(f"signals_cache.pkl not found at {SIGNALS_CACHE}")
    with open(SIGNALS_CACHE, "rb") as fh:
        cache = pickle.load(fh)

    # Load features_long (for window start times and merge target)
    feat = pd.read_parquet(FEATURES_LONG)
    feat["patient_id"] = feat["patient_id"].astype(str)

    # Choose 3 random patients for sanity plots
    rng     = np.random.RandomState(SEED)
    all_pids = sorted(cache.keys())
    sanity_pids = set(rng.choice(all_pids, size=min(3, len(all_pids)), replace=False).tolist())

    all_rows = []
    stats_per_pat = []

    for pid_raw in all_pids:
        pid = str(pid_raw)
        tup = cache[pid_raw]
        if len(tup) < 3:
            print(f"  [WARN] {pid}: cache entry has only {len(tup)} elements, skipping")
            continue
        rr_series, ppg_series, ptt_series = tup[0], tup[1], tup[2]

        # Get window starts for this patient from features_long
        pat_feat = feat[feat["patient_id"] == pid]
        if pat_feat.empty:
            print(f"  [WARN] {pid}: no windows in features_long, skipping")
            continue

        t_wins = pat_feat["t_window_start_s"].values

        # Compute BRS_seq
        df_brs = compute_brs_seq_patient(rr_series, ptt_series, t_wins)
        df_brs["patient_id"] = pid
        all_rows.append(df_brs)

        # Statistics
        valid_frac = df_brs["brs_seq"].notna().mean()
        n_valid = df_brs["brs_seq"].notna().sum()
        median_val = df_brs["brs_seq"].median()
        stats_per_pat.append({
            "patient_id": pid,
            "n_windows": len(df_brs),
            "n_windows_valid": n_valid,
            "valid_frac": valid_frac,
            "median_brs_seq": median_val,
        })
        print(f"  {pid:12s}: {n_valid:5d}/{len(df_brs)} windows valid "
              f"({100*valid_frac:.1f}%)  median={median_val:.3f}")

        # Sanity plot
        if pid in sanity_pids:
            _sanity_plot(pid, rr_series, ptt_series, FIG)

    # Concatenate all BRS_seq rows
    brs_df = pd.concat(all_rows, ignore_index=True)

    # Merge with features_long
    merged = feat.merge(
        brs_df[["patient_id", "t_window_start_s", "brs_seq",
                "n_sequences_window", "n_sequences_valid"]],
        on=["patient_id", "t_window_start_s"],
        how="left",
    )

    # ── Sanity check ──────────────────────────────────────────────────────────
    overall_valid = merged["brs_seq"].notna().mean()
    print(f"\n[PASO 1] Overall windows with valid BRS_seq: "
          f"{merged['brs_seq'].notna().sum():,}/{len(merged):,} "
          f"({100*overall_valid:.1f}%)")

    if overall_valid < BRS_SEQ_SANITY_MIN:
        raise RuntimeError(
            f"[PASO 1] SANITY CHECK FAILED: only {100*overall_valid:.1f}% of windows "
            f"have valid BRS_seq (threshold {100*BRS_SEQ_SANITY_MIN:.0f}%). "
            "Algorithm may be miscalibrated — check sanity plots."
        )
    print(f"[PASO 1] ✓ Sanity check PASSED ({100*overall_valid:.1f}% ≥ {100*BRS_SEQ_SANITY_MIN:.0f}%)")

    # Save
    FEATURES_BRS_SEQ.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(FEATURES_BRS_SEQ, index=False)
    print(f"[SAVE] {FEATURES_BRS_SEQ.name}")

    # Save per-patient stats
    stats_df = pd.DataFrame(stats_per_pat)
    stats_df.to_csv(FEATURES_BRS_SEQ.parent / "brs_seq_stats_per_patient.csv", index=False)

    return merged


if __name__ == "__main__":
    compute_all_brs_seq()
