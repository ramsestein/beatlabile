"""
q1_main.py
==========
Orquestador principal para el sub-estudio Q1.

Ejecución idempotente: los resultados ya generados no se recalculan
si los archivos de salida existen (usar --force para regenerar).

USO:
  cd validation
  python q1_main.py [--force] [--step {1,2,3,4,5,6}] [--no-partial]

Pasos:
  1 — Anotaciones y farmacología  → annotations_normalized.csv, drug_timeseries.parquet
  2 — Señales (ECG/PPG/PTT)       → [sanity check; detiene si >20% rechazo]
  3 — Features 30s                → features_long.parquet
  4 — Ventanas evento/control     → event_windows.csv
  5 — Tests confirmatorios        → test_results.csv
  6 — Figuras                     → figures/*.png
  7 — Informe final               → Q1_report.md
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap path — permite ejecutar desde validation/ o desde repo root
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ---------------------------------------------------------------------------
# Imports propios
# ---------------------------------------------------------------------------
from q1_config import (
    ALPHA_BONFERRONI,
    EXCLUIDOS,
    GRUPO_BLOQUEO,
    GROUP_LABEL,
    INCLUIDOS_CON_PARCIAL,
    INCLUIDOS_PLENOS,
    N_PRIMARY_TESTS,
    PARCIAL,
    RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(run_ts: str, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"run_{run_ts}.log"
    fmt = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    # StreamHandler con UTF-8 explícito para evitar UnicodeEncodeError en Windows (cp1252)
    stream_handler = logging.StreamHandler(sys.stdout)
    try:
        stream_handler.stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, TypeError):
        pass  # Python <3.7 o stream no soporta reconfigure
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        stream_handler,
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    logging.getLogger(__name__).info("Log iniciado: %s", log_path)


# ---------------------------------------------------------------------------
# Carga/caché de artefactos
# ---------------------------------------------------------------------------

def _load_cached_or_run(path: Path, run_fn, force: bool):
    """Si path existe y no force, carga. Si no, ejecuta run_fn y devuelve resultado."""
    import pandas as pd
    if not force and path.exists():
        logging.getLogger(__name__).info("Cargando caché: %s", path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)
    return run_fn()


# ---------------------------------------------------------------------------
# PASO 1
# ---------------------------------------------------------------------------

def paso1(patients: list[str], force: bool):
    log = logging.getLogger("paso1")
    from q1_annotations import run_paso1, save_paso1
    from q1_load import load_vital

    ann_path  = RESULTS_DIR / "annotations_normalized.csv"
    drug_path = RESULTS_DIR / "drug_timeseries.parquet"

    if not force and ann_path.exists() and drug_path.exists():
        log.info("PASO 1: cargando caché")
        import pandas as pd
        ann_df  = pd.read_csv(ann_path, dtype={"patient_id": str})
        drug_df = pd.read_parquet(drug_path)
        drug_df["patient_id"] = drug_df["patient_id"].astype(str)
        return ann_df, drug_df

    log.info("PASO 1: parseando anotaciones para %d pacientes", len(patients))
    ann_df, drug_df, warnings = run_paso1(patients)
    save_paso1(ann_df, drug_df)

    # Reporte breve
    if not ann_df.empty:
        log.info("─── PASO 1 RESUMEN ───────────────────────────────")
        log.info("Total anotaciones normalizadas: %d", len(ann_df))
        log.info("Pacientes con anotaciones: %d", ann_df["patient_id"].nunique())
        for cat, grp in ann_df.groupby("category"):
            subcats = grp["subcategory"].value_counts().to_dict()
            log.info("  %-22s n=%-4d  subcategorías: %s", cat, len(grp), subcats)
        stim_pids = set(ann_df[ann_df["category"] == "estimulo"]["patient_id"])
        ag_pids   = set(ann_df[ann_df["subcategory"] == "AG"]["patient_id"])
        no_stim   = set(patients) - stim_pids
        no_ag     = set(patients) - ag_pids
        if no_stim:
            log.warning("Sin estímulos dolorosos: %s", sorted(no_stim))
        if no_ag:
            log.warning("Sin marcador AG: %s", sorted(no_ag))
    for w in warnings:
        log.warning("PASO 1 advertencia: %s", w)

    return ann_df, drug_df


# ---------------------------------------------------------------------------
# PASO 2
# ---------------------------------------------------------------------------

def paso2(patients: list[str], force: bool):
    """
    Procesa ECG/PPG/PTT para todos los pacientes.
    Detiene el pipeline si >20% de rechazo en ≥3 pacientes muestreados.
    Retorna: dict {pid: (RRSeries, PPGSeries, PTTSeries, duration_s)}
    """
    log = logging.getLogger("paso2")
    import pickle
    cache_path = RESULTS_DIR / "signals_cache.pkl"

    if not force and cache_path.exists():
        log.info("PASO 2: cargando caché de señales")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    from q1_load import load_vital, get_recording_duration_s
    from q1_signals import (
        process_ecg, process_ppg, compute_ptt,
        check_detection_quality, sanity_check_plot,
    )

    signals: dict = {}
    ppg_rejection_flags: list[str] = []

    for pid in patients:
        log.info("PASO 2 — %s: cargando archivo .vital", pid)
        vf = load_vital(pid)
        if vf is None:
            log.warning("%s: no se pudo cargar .vital, omitiendo", pid)
            continue

        rr  = process_ecg(vf, pid)
        ppg = process_ppg(vf, pid)
        ptt = compute_ptt(rr, ppg)
        dur = get_recording_duration_s(vf)

        quality_ok = check_detection_quality(ppg, threshold=0.20)
        if not quality_ok:
            log.warning("%s: calidad PPG <80%% (ppg_valid_pct=%.1f%%)",
                        pid, ppg.pct_valid if ppg else 0.0)
            ppg_rejection_flags.append(pid)

        signals[pid] = (rr, ppg, ptt, dur)
        log.info("%s: RR=%d peaks, PPG=%d feet, PTT=%d valid, dur=%.1f min",
                 pid,
                 rr.n_peaks_kept if rr else 0,
                 ppg.n_feet_kept if ppg else 0,
                 ptt.n_kept if ptt else 0,
                 dur / 60)

    # ── Generar detection_quality_per_patient.csv ──
    import pandas as pd
    quality_rows = []
    for pid, (rr, ppg, ptt, dur) in signals.items():
        n_rpeaks = rr.n_peaks_kept if rr else 0
        ptt_ratio = ptt.n_kept / max(1, n_rpeaks)
        mean_ptt = float(np.mean(ptt.ptt_ms)) if ptt and len(ptt.ptt_ms) > 0 else np.nan
        med_ptt  = float(np.median(ptt.ptt_ms)) if ptt and len(ptt.ptt_ms) > 0 else np.nan
        quality_rows.append({
            "patient_id":         pid,
            "n_rpeaks":           n_rpeaks,
            "n_ppg_events":       ppg.n_feet_kept if ppg else 0,
            "n_ptt_pairs":        ptt.n_kept if ptt else 0,
            "ratio_ptt_to_rpeaks": round(ptt_ratio, 4),
            "mean_ptt_ms":        round(mean_ptt, 1) if not np.isnan(mean_ptt) else np.nan,
            "median_ptt_ms":      round(med_ptt, 1)  if not np.isnan(med_ptt)  else np.nan,
            "ppg_valid_pct":      round(ppg.pct_valid, 1) if ppg else np.nan,
            "ecg_lead_used":      rr.ecg_track if rr else "",
        })
    dq_df = pd.DataFrame(quality_rows)
    dq_path = RESULTS_DIR / "detection_quality_per_patient.csv"
    dq_df.to_csv(dq_path, index=False)
    log.info("Calidad de detección guardada: %s", dq_path)

    # ── Sanity check estricto: PTT matching rate en TODOS los pacientes ──
    # Umbral: ratio PTT/R-peaks ≥ 0.20 (20%) para al menos el 80% de los pacientes.
    # Si más de 3 pacientes tienen ratio < 20%, el pipeline se detiene.
    PTT_RATIO_MIN = 0.20
    ptt_failure_flags = [
        pid for pid, row in zip(dq_df["patient_id"], quality_rows)
        if row["ratio_ptt_to_rpeaks"] < PTT_RATIO_MIN
    ]
    if len(ptt_failure_flags) > 3:
        log.error(
            "PASO 2 SANITY CHECK FALLIDO: %d/%d pacientes con ratio PTT/R-peaks < %.0f%%: %s\n"
            "  → Verificar ventana PTT_MIN/MAX_MS en q1_config.py y detección de feet PPG.\n"
            "  → PIPELINE DETENIDO.",
            len(ptt_failure_flags), len(signals),
            PTT_RATIO_MIN * 100, ptt_failure_flags,
        )
        raise SystemExit(2)
    elif ptt_failure_flags:
        log.warning(
            "PASO 2: %d pacientes con ratio PTT < %.0f%% (aceptable): %s",
            len(ptt_failure_flags), PTT_RATIO_MIN * 100, ptt_failure_flags,
        )

    # Sanity check original PPG (secundario, ahora no detiene el pipeline)
    if len(ppg_rejection_flags) >= 3:
        log.warning(
            "PASO 2: >20%% de rechazo PPG en %d pacientes: %s",
            len(ppg_rejection_flags), ppg_rejection_flags
        )

    # Generar plots de sanity check para 3 pacientes aleatorios
    from q1_load import load_vital
    rng = random.Random(42)
    sample_pids = rng.sample(list(signals.keys()), min(3, len(signals)))
    for pid in sample_pids:
        vf = load_vital(pid)
        if vf and pid in signals:
            rr, ppg, ptt, _ = signals[pid]
            try:
                sanity_check_plot(pid, vf, rr, ppg, ptt)
            except Exception as exc:
                log.warning("Sanity plot %s falló: %s", pid, exc)

    log.info("PASO 2: %d/%d pacientes procesados, %d con advertencia PPG",
             len(signals), len(patients), len(ppg_rejection_flags))

    with open(cache_path, "wb") as fh:
        import pickle
        pickle.dump(signals, fh)
    log.info("Señales cacheadas en %s", cache_path)

    return signals


# ---------------------------------------------------------------------------
# PASO 3
# ---------------------------------------------------------------------------

def paso3(signals: dict, force: bool):
    log = logging.getLogger("paso3")
    import pandas as pd
    from q1_features import run_paso3, save_paso3

    path = RESULTS_DIR / "features_long.parquet"
    if not force and path.exists():
        log.info("PASO 3: cargando caché")
        return pd.read_parquet(path)

    log.info("PASO 3: computando features para %d pacientes", len(signals))
    features_df = run_paso3(signals)
    save_paso3(features_df)
    log.info("PASO 3: %d ventanas generadas", len(features_df))
    return features_df


# ---------------------------------------------------------------------------
# PASO 4
# ---------------------------------------------------------------------------

def paso4(ann_df, drug_df, features_df, groups, patients, force: bool):
    log = logging.getLogger("paso4")
    import pandas as pd
    from q1_events import build_event_windows, save_paso4

    path = RESULTS_DIR / "event_windows.csv"
    if not force and path.exists():
        log.info("PASO 4: cargando caché")
        return pd.read_csv(path, dtype={"patient_id": str})

    log.info("PASO 4: construyendo ventanas de evento y control")
    ew = build_event_windows(ann_df, drug_df, features_df, groups, patients)
    save_paso4(ew)
    return ew


# ---------------------------------------------------------------------------
# PASO 5
# ---------------------------------------------------------------------------

def paso5(event_windows, clinical_df, force: bool):
    log = logging.getLogger("paso5")
    import pandas as pd
    from q1_stats import run_paso5, save_paso5

    path = RESULTS_DIR / "test_results.csv"
    if not force and path.exists():
        log.info("PASO 5: cargando caché")
        return pd.read_csv(path)

    log.info("PASO 5: ejecutando tests confirmatorios")
    tr = run_paso5(event_windows, clinical_df)
    save_paso5(tr)
    return tr


# ---------------------------------------------------------------------------
# PASO 6
# ---------------------------------------------------------------------------

def paso6(event_windows, test_results_df, force: bool):
    log = logging.getLogger("paso6")
    from q1_figures import run_paso6

    fig_dir = RESULTS_DIR / "figures"
    figs_exist = (
        (fig_dir / "forest_plot_directional_tests.png").exists() and
        (fig_dir / "feature_distributions_by_event.png").exists()
    )
    if not force and figs_exist:
        log.info("PASO 6: figuras ya existen, omitiendo")
        return

    log.info("PASO 6: generando figuras")
    run_paso6(event_windows, test_results_df)


# ---------------------------------------------------------------------------
# PASO 7 — Informe Q1_report.md
# ---------------------------------------------------------------------------

def paso7(ann_df, signals, features_df, event_windows, test_results_df,
          patients: list[str], force: bool) -> None:
    log = logging.getLogger("paso7")
    out = RESULTS_DIR / "Q1_report.md"
    if not force and out.exists():
        log.info("PASO 7: Q1_report.md ya existe, omitiendo")
        return

    import pandas as pd

    n_plenos  = len(INCLUIDOS_PLENOS)
    n_parcial = len(PARCIAL)
    n_total   = len(patients)

    lines: list[str] = [
        "# Informe Q1 — Validación Firma Autonómica Pre-Hipertensión",
        f"\nGenerado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Cohorte",
        f"- Pacientes incluidos plenos: **{n_plenos}**",
        f"- Pacientes parciales: **{n_parcial}** ({', '.join(sorted(PARCIAL))})",
        f"- Total analizados: **{n_total}**",
        f"- Excluidos: {', '.join(sorted(EXCLUIDOS))} (sin .vital o PPG <70%)",
        "",
        "## 2. Anotaciones (PASO 1)",
    ]

    if not ann_df.empty:
        n_ann = len(ann_df)
        n_pid = ann_df["patient_id"].nunique()
        lines += [
            f"- Total anotaciones normalizadas: **{n_ann}**",
            f"- Pacientes con anotaciones: **{n_pid}**",
        ]
        for cat, grp in ann_df.groupby("category"):
            lines.append(f"  - {cat}: n={len(grp)}")
        stim_pids = ann_df[ann_df["category"] == "estimulo"]["patient_id"].unique()
        lines.append(f"- Pacientes con estímulos: {len(stim_pids)}")
    else:
        lines.append("- No se generaron anotaciones.")

    lines += [
        "",
        "## 3. Señales (PASO 2)",
    ]
    if signals:
        for pid, (rr, ppg, ptt, dur) in signals.items():
            pct = ppg.pct_valid if ppg else 0.0
            lines.append(
                f"- {pid}: dur={dur/60:.1f} min  |  "
                f"RR peaks={rr.n_peaks_kept if rr else 'N/A'}  |  "
                f"PPG valid={pct:.1f}%  |  "
                f"PTT pairs={ptt.n_kept if ptt else 0}"
            )
    else:
        lines.append("- Sin señales procesadas.")

    lines += [
        "",
        "## 4. Features (PASO 3)",
    ]
    if features_df is not None and not features_df.empty:
        lines.append(f"- Total ventanas 30s: **{len(features_df)}**")
        lines.append(f"- Pacientes: {features_df['patient_id'].nunique()}")
    else:
        lines.append("- Sin features calculadas.")

    lines += [
        "",
        "## 5. Ventanas Evento/Control (PASO 4)",
    ]
    if event_windows is not None and not event_windows.empty:
        n_ev  = (event_windows["window_type"] == "event").sum()
        n_ct  = (event_windows["window_type"] == "control").sum()
        lines += [
            f"- Ventanas evento: **{n_ev}**",
            f"- Ventanas control: **{n_ct}**",
            f"- Pacientes con ventanas: {event_windows['patient_id'].nunique()}",
        ]
    else:
        lines.append("- Sin ventanas generadas.")

    lines += [
        "",
        "## 6. Tests Confirmatorios (PASO 5)",
        f"- 7 tests pre-especificados, Bonferroni α={ALPHA_BONFERRONI:.5f}",
        "",
        "| Feature | Agg | Dir_exp | β | IC95% lo | IC95% hi | p(1-sided) | p(Bonf) | Veredicto |",
        "|---------|-----|---------|---|----------|----------|-----------|---------|-----------|",
    ]

    if test_results_df is not None and not test_results_df.empty:
        primary = test_results_df[test_results_df["analysis"] == "primary"]
        for _, row in primary.iterrows():
            def fmt(v):
                if hasattr(v, '__float__'):
                    import math
                    if math.isnan(float(v)):
                        return "N/A"
                    return f"{float(v):.4f}"
                return str(v)
            lines.append(
                f"| {row['feature']} | {row['aggregation']} | "
                f"{row.get('direction_expected','?')} | {fmt(row['beta'])} | "
                f"{fmt(row['ci_lo'])} | {fmt(row['ci_hi'])} | "
                f"{fmt(row['p_onesided'])} | {fmt(row['p_bonferroni'])} | "
                f"**{row['verdict']}** |"
            )
        validados = primary[primary["verdict"] == "validado"]["col_name"].tolist()
        lines += [
            "",
            f"**Features validadas (p_Bonf < {ALPHA_BONFERRONI:.5f}):** "
            f"{', '.join(validados) if validados else 'ninguna'}",
        ]

    lines += [
        "",
        "## 7. Figuras (PASO 6)",
        "- feature_distributions_by_event.png",
        "- per_patient_event_response.png",
        "- group_gradient_plot.png (si hay features validadas)",
        "- forest_plot_directional_tests.png",
        "",
        "---",
        "*Fin del informe Q1*",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Informe guardado: %s", out)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline Q1 — validación autonómica")
    p.add_argument("--force",  action="store_true", help="Regenerar todos los artefactos")
    p.add_argument("--step",   type=int, default=7,  help="Ejecutar hasta este paso (1-7)")
    p.add_argument("--no-partial", action="store_true", help="Excluir paciente parcial")
    return p.parse_args()


def main():
    args = parse_args()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(run_ts, RESULTS_DIR)
    log = logging.getLogger("main")

    log.info("===========================================")
    log.info("  BeatLabile Q1 - Pipeline de validacion")
    log.info("  Timestamp: %s", run_ts)
    log.info("===========================================")

    # Selección de pacientes
    if args.no_partial:
        patients = sorted(INCLUIDOS_PLENOS)
        log.info("Modo: solo plenos (%d pacientes)", len(patients))
    else:
        patients = sorted(INCLUIDOS_CON_PARCIAL)
        log.info("Modo: plenos + parcial (%d pacientes)", len(patients))

    groups = {
        pid: GROUP_LABEL[GRUPO_BLOQUEO[pid]]
        for pid in patients
    }

    # ── PASO 1 ──────────────────────────────────────────────────────
    ann_df, drug_df = paso1(patients, args.force)
    if args.step < 2:
        return

    # ── PASO 2 ──────────────────────────────────────────────────────
    signals = paso2(patients, args.force)
    if args.step < 3:
        return

    # ── PASO 3 ──────────────────────────────────────────────────────
    features_df = paso3(signals, args.force)
    if args.step < 4:
        return

    # ── PASO 4 ──────────────────────────────────────────────────────
    event_windows = paso4(ann_df, drug_df, features_df, groups, patients, args.force)
    if args.step < 5:
        return

    # ── PASO 5 ──────────────────────────────────────────────────────
    # Intentar cargar clinical_df para covariables (no crítico)
    try:
        from q1_load import load_clinical_db
        clinical_df = load_clinical_db()
    except Exception as exc:
        log.warning("No se pudo cargar BD clínica: %s", exc)
        clinical_df = None

    test_results_df = paso5(event_windows, clinical_df, args.force)
    if args.step < 6:
        return

    # ── PASO 6 ──────────────────────────────────────────────────────
    paso6(event_windows, test_results_df, args.force)
    if args.step < 7:
        return

    # ── PASO 7 ──────────────────────────────────────────────────────
    paso7(ann_df, signals, features_df, event_windows, test_results_df, patients, args.force)

    log.info("== Pipeline Q1 completado ==")


if __name__ == "__main__":
    main()
