#!/usr/bin/env python3
"""BeatLabile — Run the full experimental suite end-to-end.

Usage
-----
    python run_all.py                  # all acts + sensitivity + figures
    python run_all.py --skip-figures   # skip figure generation
    python run_all.py --act 1          # run only Act 1
    python run_all.py --act 2          # run only Act 2
    python run_all.py --act 3          # run only Act 3
    python run_all.py --sensitivity    # run only sensitivity analyses
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("beatlabile")


def _banner(text: str) -> None:
    bar = "=" * 60
    log.info(bar)
    log.info("  %s", text)
    log.info(bar)


def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    m, s = divmod(int(secs), 60)
    return f"{m}m {s}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="BeatLabile experimental suite")
    parser.add_argument("--act", type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Run only the specified Act (1–5)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run only sensitivity analyses")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    t_total = time.time()
    errors: list[str] = []

    run_all_acts = args.act is None and not args.sensitivity

    # ------------------------------------------------------------------ #
    # Act 1 — Development cohort (Clínic Barcelona)
    # ------------------------------------------------------------------ #
    if run_all_acts or args.act == 1:
        _banner("Act 1 — Development (Clínic Barcelona)")
        t0 = time.time()
        try:
            from experiments.act1_clinic import run_act1
            run_act1()
            log.info("Act 1 completed in %s", _elapsed(t0))
        except Exception as exc:
            log.error("Act 1 FAILED: %s", exc, exc_info=True)
            errors.append(f"Act 1: {exc}")

    # ------------------------------------------------------------------ #
    # Act 2 — Blind external validation (MIMIC-IV)
    # ------------------------------------------------------------------ #
    if run_all_acts or args.act == 2:
        _banner("Act 2 — Blind Validation (MIMIC-IV)")
        t0 = time.time()
        try:
            from experiments.act2_mimic import run_act2
            run_act2()
            log.info("Act 2 completed in %s", _elapsed(t0))
        except Exception as exc:
            log.error("Act 2 FAILED: %s", exc, exc_info=True)
            errors.append(f"Act 2: {exc}")

    # ------------------------------------------------------------------ #
    # Act 3 — Revalidation + sufficiency test (VitalDB)
    # ------------------------------------------------------------------ #
    if run_all_acts or args.act == 3:
        _banner("Act 3 — Revalidation + Sufficiency (VitalDB)")
        t0 = time.time()
        try:
            from experiments.act3_vitaldb import run_act3
            run_act3()
            log.info("Act 3 completed in %s", _elapsed(t0))
        except Exception as exc:
            log.error("Act 3 FAILED: %s", exc, exc_info=True)
            errors.append(f"Act 3: {exc}")

    # ------------------------------------------------------------------ #
    # Act 4 — Univariate validity & domain-shift robustness
    # ------------------------------------------------------------------ #
    if run_all_acts or args.act == 4:
        _banner("Act 4 — Univariate Predictor Validity & Domain-Shift Robustness")
        t0 = time.time()
        try:
            from experiments.act4_univariate import run_act4
            run_act4()
            log.info("Act 4 completed in %s", _elapsed(t0))
        except Exception as exc:
            log.error("Act 4 FAILED: %s", exc, exc_info=True)
            errors.append(f"Act 4: {exc}")

    # ------------------------------------------------------------------ #
    # Act 5 — TRIPOD checklist & Table 1
    # ------------------------------------------------------------------ #
    if run_all_acts or args.act == 5:
        _banner("Act 5 — TRIPOD Checklist & Table 1 Demographics")
        t0 = time.time()
        try:
            from experiments.act5_tripod import run_act5
            run_act5()
            log.info("Act 5 completed in %s", _elapsed(t0))
        except Exception as exc:
            log.error("Act 5 FAILED: %s", exc, exc_info=True)
            errors.append(f"Act 5: {exc}")

    # ------------------------------------------------------------------ #
    # Sensitivity analyses
    # ------------------------------------------------------------------ #
    if run_all_acts or args.sensitivity:
        _banner("Sensitivity Analyses")
        t0 = time.time()
        try:
            from experiments.sensitivity import run_sensitivity
            run_sensitivity()
            log.info("Sensitivity analyses completed in %s", _elapsed(t0))
        except Exception as exc:
            log.error("Sensitivity FAILED: %s", exc, exc_info=True)
            errors.append(f"Sensitivity: {exc}")

    # ------------------------------------------------------------------ #
    # Figure generation
    # ------------------------------------------------------------------ #
    if not args.skip_figures and (run_all_acts or args.sensitivity):
        _banner("Generating Figures (Fig 1–10)")
        t0 = time.time()
        try:
            from beatlabile.config import RESULTS_DIR
            from figures.plotting import generate_all_figures
            generate_all_figures(RESULTS_DIR)
            log.info("Figures generated in %s", _elapsed(t0))
        except Exception as exc:
            log.error("Figure generation FAILED: %s", exc, exc_info=True)
            errors.append(f"Figures: {exc}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    _banner("Run Complete")
    log.info("Total wall-clock time: %s", _elapsed(t_total))

    if errors:
        log.error("The following steps encountered errors:")
        for e in errors:
            log.error("  • %s", e)
        log.error("Check logs above for details.")
        return 1

    log.info("All steps finished successfully.")
    try:
        from beatlabile.config import RESULTS_DIR
        log.info("Results saved to: %s", Path(RESULTS_DIR).resolve())
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
