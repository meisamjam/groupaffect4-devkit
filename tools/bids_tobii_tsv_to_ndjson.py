#!/usr/bin/env python3
"""Convert BIDS-style Tobii gaze TSV.gz files to NDJSON for tobii_multi_glasses_world_align.py.

The BIDS TSV.gz files produced by ``multisource_to_bids_runs.py`` contain one
row per LSL sample with columns::

    lsl_time  stream_name  stream_type  value_0 … value_N

Channel layout for ``Tobii_P*_stream`` (default, no ``--with-3d``):
  value_0  gaze_x        (normalised 0–1)
  value_1  gaze_y        (normalised 0–1)
  value_2  pupil_left    (mm diameter)
  value_3  pupil_right   (mm diameter)
  value_4  gaze_valid    (1.0 = valid, 0.0 = invalid)

With ``--with-3d`` the bridge appends additional channels; this converter
handles that gracefully by only reading the first five.

Output NDJSON format (one JSON object per line)::

    {"packet": {"timestamp_ticks": <lsl_time>, "gaze2d": [gaze_x, gaze_y],
                "pupil_left": ..., "pupil_right": ..., "gaze_valid": ...}}

Pass ``--ticks-per-second 1`` to ``tobii_multi_glasses_world_align.py`` because
``timestamp_ticks`` here is already in seconds (LSL clock).

Usage::

    python tools/bids_tobii_tsv_to_ndjson.py \\
        --input  "F:/processed_data/.../et/*_task-T1_run-01_acq-P2_tobii.tsv.gz" \\
        --output "F:/processed_data/.../et/P2_task-T1_gaze.ndjson"

    # Or convert all per-participant files in a directory:
    python tools/bids_tobii_tsv_to_ndjson.py \\
        --session-et-dir "F:/processed_data/sub-01/ses-XXXX/et" \\
        --task T1
"""

import argparse
import csv
import gzip
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Column indices within the value_* block.
_IDX_GAZE_X = 0
_IDX_GAZE_Y = 1
_IDX_PUPIL_LEFT = 2
_IDX_PUPIL_RIGHT = 3
_IDX_GAZE_VALID = 4


def convert_tsv_to_ndjson(src: Path, dst: Path) -> int:
    """Convert a single TSV.gz gaze file to NDJSON.

    Returns number of samples written.

    # Privacy: participant IDs in filenames are P1–P4 only (anonymised).
    """
    n_written = 0
    n_invalid = 0

    with gzip.open(src, "rt", encoding="utf-8") as fh_in, dst.open("w", encoding="utf-8") as fh_out:
        reader = csv.DictReader(fh_in, delimiter="\t")
        for row in reader:
            try:
                lsl_time = float(row["lsl_time"])
            except (KeyError, ValueError):
                continue

            # value_* columns
            vals: list[float] = []
            for i in range(5):
                raw = row.get(f"value_{i}", "nan")
                try:
                    vals.append(float(raw))
                except ValueError:
                    vals.append(float("nan"))

            import math

            gaze_x = vals[_IDX_GAZE_X]
            gaze_y = vals[_IDX_GAZE_Y]

            # Skip rows where gaze is explicitly invalid or NaN
            gaze_valid = vals[_IDX_GAZE_VALID]
            if math.isnan(gaze_x) or math.isnan(gaze_y):
                n_invalid += 1
                continue

            packet: dict = {
                "timestamp_ticks": lsl_time,  # seconds; use --ticks-per-second 1
                "gaze2d": [gaze_x, gaze_y],
            }
            if not math.isnan(vals[_IDX_PUPIL_LEFT]):
                packet["pupil_left"] = vals[_IDX_PUPIL_LEFT]
            if not math.isnan(vals[_IDX_PUPIL_RIGHT]):
                packet["pupil_right"] = vals[_IDX_PUPIL_RIGHT]
            if not math.isnan(gaze_valid):
                packet["gaze_valid"] = gaze_valid

            fh_out.write(json.dumps({"packet": packet}) + "\n")
            n_written += 1

    if n_invalid:
        logger.debug("%s: skipped %d NaN gaze rows", src.name, n_invalid)
    return n_written


def convert_session(et_dir: Path, task: str, output_dir: Path | None = None) -> dict[str, Path]:
    """Convert all per-participant Tobii TSV.gz for a given task.

    Returns mapping {participant_id: output_ndjson_path}.
    """
    out_dir = output_dir or et_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    # Match pattern like *_task-T1_run-01_acq-P2_tobii.tsv.gz
    for src in sorted(et_dir.glob(f"*_task-{task}_run-*_acq-P*_tobii.tsv.gz")):
        # Extract participant ID (P1, P2, P3, P4)
        stem = src.stem.replace(".tsv", "")
        parts = stem.split("_acq-")
        if len(parts) < 2:
            continue
        acq = parts[-1]
        participant_id = acq.split("_")[0]  # e.g., "P2"

        dst = out_dir / f"{participant_id}_task-{task}_gaze.ndjson"
        if dst.exists():
            logger.info("Skipping %s (already exists)", dst.name)
            results[participant_id] = dst
            continue

        n = convert_tsv_to_ndjson(src, dst)
        logger.info("%-50s → %s  (%d samples)", src.name, dst.name, n)
        results[participant_id] = dst

    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert BIDS Tobii TSV.gz → NDJSON for tobii_multi_glasses_world_align.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, metavar="PATH",
                      help="Single TSV.gz input file")
    mode.add_argument("--session-et-dir", type=Path, metavar="DIR",
                      help="Session et/ directory; converts all per-participant files for --task")

    p.add_argument("--output", type=Path, metavar="PATH",
                   help="Output NDJSON path (required with --input)")
    p.add_argument("--task", default="T1",
                   help="Task label to filter (e.g. T1, T2); used with --session-et-dir (default: T1)")
    p.add_argument("--output-dir", type=Path, metavar="DIR",
                   help="Output directory for --session-et-dir mode (default: same as et-dir)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.input:
        if not args.output:
            print("ERROR: --output is required when using --input", file=sys.stderr)
            return 1
        src = args.input.resolve()
        if not src.exists():
            print(f"ERROR: input not found: {src}", file=sys.stderr)
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        n = convert_tsv_to_ndjson(src, args.output)
        logger.info("Wrote %d samples → %s", n, args.output)
    else:
        et_dir = args.session_et_dir.resolve()
        if not et_dir.exists():
            print(f"ERROR: et-dir not found: {et_dir}", file=sys.stderr)
            return 1
        results = convert_session(et_dir, args.task, args.output_dir)
        if not results:
            print(f"WARNING: no files found for task {args.task} in {et_dir}", file=sys.stderr)
            return 1
        print(f"\nConverted {len(results)} participant(s): {', '.join(sorted(results))}")
        print("Output files:")
        for pid, path in sorted(results.items()):
            print(f"  {pid}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
