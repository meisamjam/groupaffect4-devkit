"""Run the full task-aware physiology feature pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("run_feature_pipeline")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run physio + pupil + dynamics + semantic extraction pipeline.")
    p.add_argument("--data-root", type=Path, required=True, help="Dataset root.")
    p.add_argument("--out-dir", type=Path, default=Path("data") / "derived_features", help="Output directory.")
    p.add_argument("--sessions", nargs="*", default=None, help="Optional session IDs filter.")
    p.add_argument("--window-s", type=float, default=30.0, help="Rolling window length.")
    p.add_argument("--step-s", type=float, default=15.0, help="Rolling window step.")
    p.add_argument("--verbose", action="store_true", help="Verbose logs.")
    return p


def _run(cmd: list[str]) -> None:
    LOG.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    py = sys.executable
    base = [
        "--data-root",
        str(args.data_root),
        "--out-dir",
        str(args.out_dir),
        "--window-s",
        str(args.window_s),
        "--step-s",
        str(args.step_s),
    ]
    if args.sessions:
        base.extend(["--sessions", *args.sessions])
    if args.verbose:
        base.append("--verbose")

    _run([py, "tools/features/extract_physio_features.py", *base])
    _run([py, "tools/features/extract_pupil_features.py", *base])
    dyn = [py, "tools/features/compute_group_dynamics.py", "--features-dir", str(args.out_dir)]
    if args.verbose:
        dyn.append("--verbose")
    _run(dyn)
    sem = [py, "tools/features/build_semantic_biomarkers.py", "--features-dir", str(args.out_dir)]
    if args.verbose:
        sem.append("--verbose")
    _run(sem)
    comp = [
        py,
        "tools/features/build_participant_group_comparisons.py",
        "--data-root",
        str(args.data_root),
        "--features-dir",
        str(args.out_dir),
    ]
    if args.sessions:
        comp.extend(["--sessions", *args.sessions])
    if args.verbose:
        comp.append("--verbose")
    _run(comp)
    LOG.info("Feature pipeline complete. Outputs in %s", args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
