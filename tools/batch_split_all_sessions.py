#!/usr/bin/env python3
"""
Batch runner: split all AV sessions into BIDS task-window clips in parallel.

Discovers every sub-*/ses-* directory under AV_ROOT, matches each to the
corresponding recording session and stimuli directory, then runs
multisource_to_bids_runs.py --split-media for all sessions concurrently.

If a session already has a task_run_windows.tsv (e.g. from a prior run or
manual annotation), that file is passed via --task-windows-tsv so sessions
with incomplete stimuli logs (grp-08, grp-11) still get media splits.

Usage:
    py tools/batch_split_all_sessions.py [--workers N] [--ffmpeg-threads N]
                                          [--dry-run] [--sessions grp-12 grp-15 ...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Root directories — edit these if data has moved
# ---------------------------------------------------------------------------
AV_ROOT = Path(r"F:\affectai-capture-av\sessions\final")
REC_ROOT = Path(r"F:\affectai-capture-recording\sessions\final")
STIM_ROOT = Path(r"F:\affectai-capture-recording\stimuli\data")
OUT_ROOT = Path(r"F:\processed_data")
SUB_ID = "sub-01"  # currently only one subject

PIPELINE = Path(__file__).parent / "multisource_to_bids_runs.py"

DEFAULT_FLAGS = [
    "--split-media",
    "--no-source-ingest",
    "--keep-raw",
    "--allow-missing-xdf",
    "--no-write-session-events",
]

_RE_SES = re.compile(r"ses-(?P<date>\d{8})_(?P<grp>grp-\d+)_(?P<run>run\d+)")
_RE_STIM = re.compile(r"(?P<date>\d{8})_(?P<grp>grp-\d+)_(?P<run>run\d+)_\d{8}_\d{6}$")
_RE_WINDOWS_TSV = re.compile(r"_task-T\w+_task_run_windows\.tsv$")


# ---------------------------------------------------------------------------


@dataclass
class SessionSpec:
    ses_id: str
    av_dir: Path
    rec_dir: Path
    stim_dir: Path
    out_dir: Path
    extra_flags: list[str] = field(default_factory=list)


def _parse_ses(name: str) -> tuple[str, str] | None:
    """Return (grp, run) or None."""
    m = _RE_SES.search(name)
    return (m.group("grp"), m.group("run")) if m else None


def _find_existing_windows_tsv(out_dir: Path) -> Path | None:
    """Return existing task_run_windows.tsv in out_dir/annot/, or None."""
    annot = out_dir / "annot"
    if not annot.exists():
        return None
    for p in sorted(annot.glob("*_task_run_windows.tsv")):
        if _RE_WINDOWS_TSV.search(p.name):
            return p
    return None


def discover_sessions(sub: str = SUB_ID) -> list[SessionSpec]:
    av_sub = AV_ROOT / sub
    rec_sub = REC_ROOT / sub

    if not av_sub.exists():
        sys.exit(f"AV subject dir not found: {av_sub}")
    if not rec_sub.exists():
        sys.exit(f"Recording subject dir not found: {rec_sub}")

    # Index recording sessions by (grp, run)
    rec_index: dict[tuple[str, str], Path] = {}
    for d in rec_sub.iterdir():
        if not d.is_dir():
            continue
        key = _parse_ses(d.name)
        if key:
            rec_index[key] = d

    # Index stimuli dirs by (grp, run) → list sorted by timestamp (latest last)
    stim_index: dict[tuple[str, str], list[Path]] = {}
    for d in STIM_ROOT.iterdir():
        if not d.is_dir():
            continue
        m = _RE_STIM.match(d.name)
        if not m:
            continue
        key = (m.group("grp"), m.group("run"))
        stim_index.setdefault(key, []).append(d)
    for lst in stim_index.values():
        lst.sort(key=lambda p: p.name)

    specs: list[SessionSpec] = []
    for av_ses in sorted(av_sub.iterdir()):
        if not av_ses.is_dir():
            continue
        key = _parse_ses(av_ses.name)
        if not key:
            continue
        grp, run = key

        rec_dir = rec_index.get(key)
        if rec_dir is None:
            print(f"[SKIP] {av_ses.name}: no matching recording session for ({grp}, {run})")
            continue

        stim_candidates = stim_index.get(key, [])
        if not stim_candidates:
            print(f"[SKIP] {av_ses.name}: no stimuli dir found for ({grp}, {run})")
            continue
        stim_dir = stim_candidates[-1]  # latest timestamp wins

        out_dir = OUT_ROOT / sub / av_ses.name

        # If an existing windows TSV is present, pass it as override so sessions
        # with incomplete stimuli logs (grp-08, grp-11) still get media splits.
        extra: list[str] = []
        existing_windows = _find_existing_windows_tsv(out_dir)
        if existing_windows is not None:
            extra += ["--task-windows-tsv", str(existing_windows)]

        specs.append(SessionSpec(
            ses_id=av_ses.name,
            av_dir=av_ses,
            rec_dir=rec_dir,
            stim_dir=stim_dir,
            out_dir=out_dir,
            extra_flags=extra,
        ))

    return specs


def build_cmd(spec: SessionSpec, ffmpeg_threads: int) -> list[str]:
    cmd = [
        sys.executable, str(PIPELINE),
        "--av-session-dir", str(spec.av_dir),
        "--recording-session-dir", str(spec.rec_dir),
        "--stimuli-dir", str(spec.stim_dir),
        "--output-session-dir", str(spec.out_dir),
        *DEFAULT_FLAGS,
        *spec.extra_flags,
    ]
    if ffmpeg_threads > 0:
        cmd += ["--ffmpeg-threads", str(ffmpeg_threads)]
    return cmd


def run_session(spec: SessionSpec, dry_run: bool, ffmpeg_threads: int) -> tuple[str, int, str]:
    cmd = build_cmd(spec, ffmpeg_threads)
    label = spec.ses_id
    if dry_run:
        print(f"[DRY-RUN] {label}")
        print("  " + " ".join(cmd))
        return label, 0, ""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return label, result.returncode, result.stdout + result.stderr
    except Exception as exc:
        return label, 1, str(exc)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=3,
                   help="Number of parallel sessions (default: 3)")
    p.add_argument("--ffmpeg-threads", type=int, default=4,
                   help="Threads per ffmpeg call (default: 4). "
                        "total CPU ~= workers * ffmpeg-threads for video encode")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing them")
    p.add_argument("--sessions", nargs="+", metavar="GRP",
                   help="Limit to specific groups, e.g. --sessions grp-12 grp-15")
    args = p.parse_args()

    specs = discover_sessions()

    if args.sessions:
        filter_set = {s.lower() for s in args.sessions}
        specs = [s for s in specs if any(f in s.ses_id.lower() for f in filter_set)]
        if not specs:
            sys.exit("No sessions matched the --sessions filter.")

    if not specs:
        print("No sessions found.")
        return 0

    print(f"Found {len(specs)} session(s) — {args.workers} workers x {args.ffmpeg_threads} ffmpeg threads:\n")
    for s in specs:
        note = ""
        if any("task-windows-tsv" in f for f in s.extra_flags):
            note = " [using existing windows TSV]"
        print(f"  {s.ses_id}{note}")
        print(f"    stim: {s.stim_dir.name}")
    print()

    if args.dry_run:
        for s in specs:
            run_session(s, dry_run=True, ffmpeg_threads=args.ffmpeg_threads)
        return 0

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_session, s, False, args.ffmpeg_threads): s for s in specs}
        for fut in as_completed(futures):
            label, rc, output = fut.result()
            if rc == 0:
                print(f"[OK]   {label}")
            else:
                print(f"[FAIL] {label} (exit {rc})")
                lines = output.strip().splitlines()
                for ln in lines[-40:]:
                    print(f"       {ln}")
                failed.append(label)

    print()
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
        return 1

    print(f"All {len(specs)} sessions completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
