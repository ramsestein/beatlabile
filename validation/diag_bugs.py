"""
diag_bugs.py — Diagnóstico de BUG 2 y BUG 3
Ejecutar desde validation/
"""
import sys, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, ".")
from q1_config import RESULTS_DIR, FIGURES_DIR, DATA_DIR, ECG_TRACKS, PPG_TRACKS, PTT_MIN_MS, PTT_MAX_MS
from q1_load import get_continuous_signal, pick_track

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Cargar caché ───────────────────────────────────────────────────────────
with open(RESULTS_DIR / "signals_cache.pkl", "rb") as f:
    signals = pickle.load(f)

# ════════════════════════════════════════════════════════════════════════════
# PARTE 1 — Distribución de latencias R→foot por paciente (BUG 3 — H2)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== PARTE 1: Distribución latencias R→foot (ms) ===")
print(f"{'pid':12} | {'med':>6} | {'p5':>5} | {'p25':>5} | {'p75':>5} | {'p95':>5} | {'n':>5}")
lat_data = {}
for pid, (rr, ppg, ptt, dur) in sorted(signals.items()):
    t_r = rr.t_peak_s
    t_f = ppg.t_foot_s
    if len(t_r) == 0 or len(t_f) == 0:
        print(f"{pid:12} | NODATA")
        continue
    lats = []
    for tr in t_r:
        mask = (t_f >= tr) & (t_f <= tr + 1.0)
        if mask.any():
            lats.append((t_f[mask][0] - tr) * 1000)
    if lats:
        lats = np.array(lats)
        lat_data[pid] = lats
        print(f"{pid:12} | {np.median(lats):6.0f} | {np.percentile(lats,5):5.0f} | "
              f"{np.percentile(lats,25):5.0f} | {np.percentile(lats,75):5.0f} | "
              f"{np.percentile(lats,95):5.0f} | {len(lats):5d}")
    else:
        print(f"{pid:12} | NOLAT")

# Figura — histograma de latencias por paciente
fig, axes = plt.subplots(5, 4, figsize=(18, 14))
axes = axes.flat
pids_sorted = sorted(lat_data.keys())
for ax, pid in zip(axes, pids_sorted):
    lats = lat_data[pid]
    ax.hist(lats, bins=50, range=(0, 1000), color="steelblue", alpha=0.7)
    ax.axvline(PTT_MIN_MS, color="red", ls="--", lw=1, label=f"min={PTT_MIN_MS}")
    ax.axvline(PTT_MAX_MS, color="orange", ls="--", lw=1, label=f"max={PTT_MAX_MS}")
    in_win = ((lats >= PTT_MIN_MS) & (lats <= PTT_MAX_MS)).mean() * 100
    ax.set_title(f"{pid}\n{in_win:.0f}% en [{PTT_MIN_MS},{PTT_MAX_MS}]ms", fontsize=8)
    ax.set_xlabel("Latencia R→foot (ms)", fontsize=7)
    ax.set_ylabel("N", fontsize=7)
for ax in list(axes)[len(pids_sorted):]:
    ax.set_visible(False)
fig.suptitle("Distribución latencias R→foot por paciente (ventana actual [150,400]ms)", fontsize=12)
plt.tight_layout()
out = FIGURES_DIR / "ptt_latency_distribution_per_patient.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada: {out}")

# ════════════════════════════════════════════════════════════════════════════
# PARTE 2 — BUG 2: 70551555, inspeccionar todas las derivaciones ECG
# ════════════════════════════════════════════════════════════════════════════
print("\n=== PARTE 2: ECG 70551555 — todas las derivaciones ===")
import vitaldb

pid_bug = "70551555"
vital_path = None
for p in (DATA_DIR / pid_bug).iterdir():
    if p.suffix.lower() == ".vital":
        vital_path = p
        break

if vital_path is None:
    print("ERROR: no se encontró .vital para 70551555")
else:
    vf = vitaldb.VitalFile(str(vital_path))
    all_tracks = list(vf.trks.keys())
    print(f"Tracks disponibles: {all_tracks}")

    # Derivaciones ECG candidatas
    ecg_cands = [t for t in all_tracks if "ECG" in t.upper() or "EKG" in t.upper()]
    print(f"Tracks ECG encontrados: {ecg_cands}")

    try:
        import neurokit2 as nk
    except ImportError:
        print("neurokit2 no instalado — instalar primero")
        sys.exit(1)

    fig, axes = plt.subplots(len(ecg_cands), 3, figsize=(18, 3 * max(len(ecg_cands), 2)))
    if len(ecg_cands) == 1:
        axes = [axes]
    else:
        axes = list(axes)

    results_ecg = {}
    for i, lead in enumerate(ecg_cands):
        sig, srate = get_continuous_signal(vf, lead)
        if sig is None or len(sig) == 0:
            print(f"  {lead}: vacío")
            continue
        dur = len(sig) / srate

        # Segment 30 min mark (mid-surgery)
        t_start = min(30 * 60, dur * 0.4)
        idx_start = int(t_start * srate)
        idx_end   = idx_start + int(60 * srate)  # 60s
        idx_end   = min(idx_end, len(sig))

        seg = sig[idx_start:idx_end].astype(np.float64)
        # Fill NaN
        nan_mask = np.isnan(seg)
        pct_nan = nan_mask.mean() * 100
        if (~nan_mask).sum() > 2:
            idx_arr = np.arange(len(seg))
            seg[nan_mask] = np.interp(idx_arr[nan_mask], idx_arr[~nan_mask], seg[~nan_mask])

        # Detect R-peaks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, info = nk.ecg_peaks(seg, sampling_rate=int(srate), method="neurokit")
                r_idx = info["ECG_R_Peaks"]
            except Exception as e:
                print(f"  {lead}: neurokit falló ({e})")
                r_idx = np.array([])

        # Also try full signal for overall count
        sig_full = sig.astype(np.float64)
        nm = np.isnan(sig_full)
        if (~nm).sum() > 2:
            idx_a = np.arange(len(sig_full))
            sig_full[nm] = np.interp(idx_a[nm], idx_a[~nm], sig_full[~nm])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, info_full = nk.ecg_peaks(sig_full, sampling_rate=int(srate), method="neurokit")
                r_idx_full = info_full["ECG_R_Peaks"]
            except Exception:
                r_idx_full = np.array([])

        if len(r_idx) > 1:
            rr_ms = np.diff(r_idx) / srate * 1000
            valid = (rr_ms >= 300) & (rr_ms <= 2000)
            hr_mean = 60000 / np.mean(rr_ms[valid]) if valid.sum() > 0 else 0
        else:
            hr_mean = 0
            valid = np.array([], dtype=bool)

        results_ecg[lead] = {
            "n_peaks_full": len(r_idx_full),
            "n_peaks_60s": len(r_idx),
            "hr_60s": hr_mean,
            "pct_nan_60s": pct_nan,
            "valid_rr": int(valid.sum()),
        }
        print(f"  {lead}: dur={dur/60:.1f}min, NaN_60s={pct_nan:.1f}%, "
              f"peaks_60s={len(r_idx)}, HR={hr_mean:.0f}bpm, peaks_full={len(r_idx_full)}")

        # Plot
        ax = axes[i] if len(ecg_cands) == 1 else axes[i]
        if isinstance(ax, np.ndarray):
            ax0, ax1, ax2 = ax[0], ax[1], ax[2]
        else:
            ax0, ax1, ax2 = ax, ax, ax
        t = np.arange(len(seg)) / srate + t_start
        ax0.plot(t, seg, lw=0.5, color="steelblue")
        if len(r_idx) > 0:
            ax0.scatter(r_idx / srate + t_start,
                        seg[np.clip(r_idx, 0, len(seg)-1)],
                        color="red", s=20, zorder=5)
        ax0.set_title(f"{lead} | NaN={pct_nan:.1f}% | peaks={len(r_idx)} | HR={hr_mean:.0f}bpm", fontsize=8)
        ax0.set_xlabel("t (s)")

        # Full signal overview (downsample)
        ds = max(1, int(srate // 10))
        t_full = np.arange(0, len(sig), ds) / srate
        ax1.plot(t_full, sig[::ds], lw=0.4, color="gray")
        ax1.set_title(f"{lead} — señal completa (ds×{ds})", fontsize=8)
        ax1.set_xlabel("t (s)")

        # RR distribution
        if len(r_idx_full) > 1:
            rr_all = np.diff(r_idx_full) / srate * 1000
            ax2.hist(rr_all, bins=50, range=(0, 3000), color="green", alpha=0.7)
            ax2.axvline(300, color="red", ls="--")
            ax2.axvline(2000, color="red", ls="--")
            ax2.set_title(f"{lead} — RR dist (n_full={len(r_idx_full)})", fontsize=8)
            ax2.set_xlabel("RR (ms)")

    fig.suptitle(f"Paciente 70551555 — todas las derivaciones ECG", fontsize=13)
    plt.tight_layout()
    out2 = FIGURES_DIR / "sanity_70551555_all_leads.png"
    plt.savefig(out2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada: {out2}")

    # Determinar mejor derivación
    best = max(results_ecg, key=lambda k: results_ecg[k]["n_peaks_full"])
    print(f"\nMejor derivación: {best} ({results_ecg[best]['n_peaks_full']} peaks totales)")

# ════════════════════════════════════════════════════════════════════════════
# PARTE 3 — Sanity PPG: 4214722 (bueno) vs 397651 (malo)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== PARTE 3: Sanity PTT detection 4214722 vs 397651 ===")

for compare_pid in ["4214722", "397651"]:
    rr, ppg, ptt, dur = signals[compare_pid]
    vital_p = None
    for p in (DATA_DIR / compare_pid).iterdir():
        if p.suffix.lower() == ".vital":
            vital_p = p
            break
    if vital_p is None:
        print(f"  No .vital para {compare_pid}")
        continue
    vf2 = vitaldb.VitalFile(str(vital_p))
    tracks2 = list(vf2.trks.keys())

    ecg_t = pick_track(tracks2, ECG_TRACKS)
    ppg_t = pick_track(tracks2, PPG_TRACKS)
    ecg_sig, ecg_sr = get_continuous_signal(vf2, ecg_t) if ecg_t else (None, 0)
    ppg_sig, ppg_sr = get_continuous_signal(vf2, ppg_t) if ppg_t else (None, 0)

    t0, t1 = 30 * 60, 30 * 60 + 30  # 30s in minute 30

    fig2, (ax_e, ax_p) = plt.subplots(2, 1, figsize=(14, 6))
    fig2.suptitle(f"Sanity PTT — {compare_pid} | 30s at min 30", fontsize=12)

    if ecg_sig is not None:
        i0, i1 = int(t0 * ecg_sr), int(t1 * ecg_sr)
        t_e = np.arange(i0, i1) / ecg_sr
        ax_e.plot(t_e, ecg_sig[i0:i1], lw=0.8, color="steelblue", label="ECG")
        mask_r = (rr.t_peak_s >= t0) & (rr.t_peak_s <= t1)
        if mask_r.any():
            yv = np.interp(rr.t_peak_s[mask_r],
                           np.arange(len(ecg_sig)) / ecg_sr,
                           ecg_sig.astype(float))
            ax_e.scatter(rr.t_peak_s[mask_r], yv, color="red", s=40, zorder=5, label="R-peaks")
        ax_e.set_ylabel("ECG"); ax_e.legend(fontsize=8)

    if ppg_sig is not None:
        i0p, i1p = int(t0 * ppg_sr), int(t1 * ppg_sr)
        t_p = np.arange(i0p, i1p) / ppg_sr
        ax_p.plot(t_p, ppg_sig[i0p:i1p], lw=0.8, color="darkorange", label="PPG")
        mask_f = (ppg.t_foot_s >= t0) & (ppg.t_foot_s <= t1)
        if mask_f.any():
            yf = np.interp(ppg.t_foot_s[mask_f],
                           np.arange(len(ppg_sig)) / ppg_sr,
                           ppg_sig.astype(float))
            ax_p.scatter(ppg.t_foot_s[mask_f], yf, color="green", s=40, zorder=5, label="PPG feet")
        # También marcar timestamps de PTT (latencias que sí matchearon)
        mask_ptt = (ptt.t_s >= t0) & (ptt.t_s <= t1)
        if mask_ptt.any():
            # Marca los feet de PTT
            t_feet_ptt = ptt.t_s[mask_ptt] + ptt.ptt_ms[mask_ptt] / 1000
            yp2 = np.interp(t_feet_ptt,
                            np.arange(len(ppg_sig)) / ppg_sr,
                            ppg_sig.astype(float))
            ax_p.scatter(t_feet_ptt, yp2, color="red", s=25, marker="x", zorder=6,
                         label=f"PTT-matched feet (n={mask_ptt.sum()})")
        ax_p.set_ylabel("PPG"); ax_p.set_xlabel("t (s)"); ax_p.legend(fontsize=8)

    plt.tight_layout()
    out3 = FIGURES_DIR / f"sanity_ptt_partial_{compare_pid}.png"
    plt.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  {compare_pid}: guardado {out3}")

# Combined figure
print("Generando sanity_ptt_detection_4214722_vs_397651.png ...")
fig3, axes3 = plt.subplots(4, 1, figsize=(16, 14))
fig3.suptitle("Sanity PTT detection — 4214722 (bueno, 61%) vs 397651 (malo, 2.3%)", fontsize=12)
pairs = [("4214722", 0, 1), ("397651", 2, 3)]
for compare_pid, row_e, row_p in pairs:
    rr, ppg, ptt, dur = signals[compare_pid]
    vital_p = None
    for p in (DATA_DIR / compare_pid).iterdir():
        if p.suffix.lower() == ".vital":
            vital_p = p
            break
    if vital_p is None:
        continue
    vf2 = vitaldb.VitalFile(str(vital_p))
    tracks2 = list(vf2.trks.keys())
    ecg_t = pick_track(tracks2, ECG_TRACKS)
    ppg_t = pick_track(tracks2, PPG_TRACKS)
    ecg_sig, ecg_sr = get_continuous_signal(vf2, ecg_t) if ecg_t else (None, 0)
    ppg_sig, ppg_sr = get_continuous_signal(vf2, ppg_t) if ppg_t else (None, 0)
    t0, t1 = 30 * 60, 30 * 60 + 30

    ax_e = axes3[row_e]; ax_p = axes3[row_p]
    ratio = ptt.n_kept / max(1, rr.n_peaks_kept) * 100
    if ecg_sig is not None:
        i0, i1 = int(t0 * ecg_sr), int(t1 * ecg_sr)
        t_e = np.arange(i0, i1) / ecg_sr
        ax_e.plot(t_e, ecg_sig[i0:i1], lw=0.8, color="steelblue")
        mask_r = (rr.t_peak_s >= t0) & (rr.t_peak_s <= t1)
        if mask_r.any():
            yv = np.interp(rr.t_peak_s[mask_r], np.arange(len(ecg_sig)) / ecg_sr, ecg_sig.astype(float))
            ax_e.scatter(rr.t_peak_s[mask_r], yv, color="red", s=40, zorder=5)
        ax_e.set_ylabel(f"ECG {compare_pid}\nptt_ratio={ratio:.1f}%", fontsize=8)
    if ppg_sig is not None:
        i0p, i1p = int(t0 * ppg_sr), int(t1 * ppg_sr)
        t_p = np.arange(i0p, i1p) / ppg_sr
        ax_p.plot(t_p, ppg_sig[i0p:i1p], lw=0.8, color="darkorange", label="PPG")
        mask_f = (ppg.t_foot_s >= t0) & (ppg.t_foot_s <= t1)
        if mask_f.any():
            yf = np.interp(ppg.t_foot_s[mask_f], np.arange(len(ppg_sig)) / ppg_sr, ppg_sig.astype(float))
            ax_p.scatter(ppg.t_foot_s[mask_f], yf, color="green", s=40, zorder=5, label="feet")
        mask_ptt = (ptt.t_s >= t0) & (ptt.t_s <= t1)
        if mask_ptt.any():
            t_fp = ptt.t_s[mask_ptt] + ptt.ptt_ms[mask_ptt] / 1000
            yp2 = np.interp(t_fp, np.arange(len(ppg_sig)) / ppg_sr, ppg_sig.astype(float))
            ax_p.scatter(t_fp, yp2, color="red", s=25, marker="x", zorder=6, label=f"PTT-matched (n={mask_ptt.sum()})")
        ax_p.set_ylabel(f"PPG {compare_pid}", fontsize=8)
        ax_p.set_xlabel("t (s)")
        ax_p.legend(fontsize=8)

plt.tight_layout()
out4 = FIGURES_DIR / "sanity_ptt_detection_4214722_vs_397651.png"
plt.savefig(out4, dpi=120, bbox_inches="tight")
plt.close(fig3)
print(f"Figura guardada: {out4}")

print("\n=== DIAGNÓSTICO COMPLETO ===")
