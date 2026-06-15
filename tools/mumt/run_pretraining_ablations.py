"""run_pretraining_ablations.py

Run Phase-0 objective ablations for MuMTAffect and collect downstream metrics.

Each ablation launches tools/mumt/train_affectai.py with a different set of
pretraining objective weights, then reads the produced results.csv and writes
an aggregate ablation summary table.

Example:
  python tools/mumt/run_pretraining_ablations.py \
      --data-path data/mumt/dataset.pkl \
      --pretrain-data data/mumt/pretrain_dataset.pkl \
      --participants-tsv data/zenodo/participants.tsv \
      --output-root data/mumt/ablations_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AblationSpec:
    name: str
    w_task: float
    w_subject: float
    w_session: float
    w_personality: float
    w_sex: float
    w_age: float
    w_next: float
    nsp_warmup_epochs: int = 0


ABLATIONS: list[AblationSpec] = [
    AblationSpec("full", 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.5, 0),
    AblationSpec("full_plus_nsp_warmup", 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.5, 10),
    AblationSpec("no_personality", 1.0, 1.0, 0.5, 0.0, 0.5, 0.3, 0.5, 0),
    AblationSpec("no_next", 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.0, 0),
    AblationSpec("next_only", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 10),
    AblationSpec("no_demographics", 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0),
    AblationSpec("no_identity", 1.0, 0.0, 0.0, 0.5, 0.5, 0.3, 0.5, 0),
    AblationSpec("task_personality_next", 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0),
]


def run_one_ablation(args: argparse.Namespace, spec: AblationSpec) -> dict[str, str | float]:
    out_dir = Path(args.output_root) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "tools/mumt/train_affectai.py",
        "--data-path",
        args.data_path,
        "--output-dir",
        str(out_dir),
        "--class-weights",
        "auto",
        "--data-driven-bins",
        "--pretrain-data",
        args.pretrain_data,
        "--pretrain-epochs",
        str(args.pretrain_epochs),
        "--participants-tsv",
        args.participants_tsv,
        "--freeze-transformers",
        "--freeze-transformers-from-phase",
        str(args.freeze_transformers_from_phase),
        "--pretrain-w-task",
        str(spec.w_task),
        "--pretrain-w-subject",
        str(spec.w_subject),
        "--pretrain-w-session",
        str(spec.w_session),
        "--pretrain-w-personality",
        str(spec.w_personality),
        "--pretrain-w-sex",
        str(spec.w_sex),
        "--pretrain-w-age",
        str(spec.w_age),
        "--pretrain-w-next",
        str(spec.w_next),
        "--pretrain-nsp-warmup-epochs",
        str(spec.nsp_warmup_epochs),
    ]

    print(f"[ablation] running: {spec.name}", flush=True)
    proc = subprocess.run(cmd, check=False)

    row: dict[str, str | float] = {
        "name": spec.name,
        "status": "ok" if proc.returncode == 0 else f"failed:{proc.returncode}",
        "w_task": spec.w_task,
        "w_subject": spec.w_subject,
        "w_session": spec.w_session,
        "w_personality": spec.w_personality,
        "w_sex": spec.w_sex,
        "w_age": spec.w_age,
        "w_next": spec.w_next,
        "nsp_warmup_epochs": spec.nsp_warmup_epochs,
        "output_dir": str(out_dir),
    }

    results_csv = out_dir / "results.csv"
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        if not df.empty:
            m = df.iloc[0].to_dict()
            row["valence_f1"] = float(m.get("valence_f1", float("nan")))
            row["arousal_f1"] = float(m.get("arousal_f1", float("nan")))
            row["dominance_f1"] = float(m.get("dominance_f1", float("nan")))
            row["personality_r2_mean"] = float(m.get("personality_r2_mean", float("nan")))

    meta_json = out_dir / "ablation_config.json"
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)

    return row


def main(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    specs = ABLATIONS
    if args.only:
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in ABLATIONS if s.name in allow]

    rows: list[dict[str, str | float]] = []
    for spec in specs:
        rows.append(run_one_ablation(args, spec))

    summary_path = output_root / "ablation_summary.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[ablation] summary written to: {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pretraining objective ablations.")
    parser.add_argument("--data-path", required=True, help="Path to data/mumt/dataset.pkl")
    parser.add_argument("--pretrain-data", required=True, help="Path to data/mumt/pretrain_dataset.pkl")
    parser.add_argument("--participants-tsv", required=True, help="Path to participants.tsv")
    parser.add_argument("--output-root", default="data/mumt/ablations", help="Root output dir")
    parser.add_argument("--pretrain-epochs", type=int, default=80, help="Phase-0 epochs")
    parser.add_argument(
        "--freeze-transformers-from-phase",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Freeze transformer blocks from this fine-tuning phase onward.",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated subset of ablation names to run.",
    )
    main(parser.parse_args())
