"""Build participant/group pooled tables and join with answers + annotation context."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tools.features.common import discover_session_dirs
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.features.common import discover_session_dirs  # type: ignore[no-redef]

LOG = logging.getLogger("build_participant_group_comparisons")

_RE_GROUP = re.compile(r"(grp-[A-Za-z0-9]+)")
_RE_DESC_PARTICIPANT = re.compile(r"participant=([A-Za-z0-9_/.-]+)")


def _group_id_from_session(session_id: str) -> str:
    m = _RE_GROUP.search(session_id or "")
    return m.group(1) if m else ""


def _safe_mode_join(values: pd.Series) -> str:
    vals = sorted({str(v) for v in values.dropna().tolist() if str(v).strip()})
    return ";".join(vals)


def _label_from_vad_rating(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value <= 3.0:
        return "Low"
    if value >= 7.0:
        return "High"
    return "Moderate"


def _label_from_biomarker_score(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value >= 1.0:
        return "High"
    if value <= -1.0:
        return "Low"
    return "Moderate"


def _participant_from_description(description: str) -> str:
    m = _RE_DESC_PARTICIPANT.search(str(description))
    if not m:
        return ""
    raw = m.group(1).strip()
    if not raw or raw.lower() in {"n/a", "na", "moderator"}:
        return ""
    if raw.lower().startswith("p") and raw[1:].isdigit():
        return f"P{int(raw[1:])}"
    if raw.isdigit():
        return f"P{int(raw)}"
    return ""


def _quantile_labels(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series([""] * len(x), index=x.index, dtype=object)
    valid = x.dropna()
    if valid.empty:
        return out
    if valid.nunique() == 1:
        out.loc[valid.index] = "Moderate"
        return out
    q1 = float(valid.quantile(0.33))
    q2 = float(valid.quantile(0.66))
    out.loc[(x.notna()) & (x <= q1)] = "Low"
    out.loc[(x.notna()) & (x >= q2)] = "High"
    out.loc[(x.notna()) & (out == "")] = "Moderate"
    return out


def _load_answers_by_participant_task(session_dir: Path) -> pd.DataFrame:
    beh_dir = session_dir / "beh"
    answers_files = sorted(beh_dir.glob("*_stimuli_answers.tsv"))
    if not answers_files:
        return pd.DataFrame()
    df = pd.read_csv(answers_files[-1], sep="\t")
    if df.empty:
        return pd.DataFrame()
    for col in ["task", "participant", "item_key", "item_value", "response_type"]:
        if col not in df.columns:
            return pd.DataFrame()
    df = df.rename(columns={"participant": "participant_id"})
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["task"] = df["task"].astype(str).str.strip()
    df["item_key"] = df["item_key"].astype(str).str.strip()
    df["response_type"] = df["response_type"].astype(str).str.strip()

    numeric = df.copy()
    numeric["item_value_num"] = pd.to_numeric(numeric["item_value"], errors="coerce")
    num = (
        numeric.dropna(subset=["item_value_num"])
        .groupby(["task", "participant_id", "item_key"], as_index=False)["item_value_num"]
        .mean()
    )
    if not num.empty:
        num = num.pivot(index=["task", "participant_id"], columns="item_key", values="item_value_num").reset_index()
        num.columns = [f"ans_{c}" if c not in {"task", "participant_id"} else c for c in num.columns]

    text = (
        df.groupby(["task", "participant_id", "item_key"], as_index=False)["item_value"]
        .agg(_safe_mode_join)
    )
    if not text.empty:
        text = text.pivot(index=["task", "participant_id"], columns="item_key", values="item_value").reset_index()
        text.columns = [f"ans_text_{c}" if c not in {"task", "participant_id"} else c for c in text.columns]

    counts = df.groupby(["task", "participant_id"], as_index=False).agg(
        answers_n=("item_key", "count"),
        response_types=("response_type", _safe_mode_join),
    )

    out = counts.copy()
    if not num.empty:
        out = out.merge(num, on=["task", "participant_id"], how="left")
    if not text.empty:
        out = out.merge(text, on=["task", "participant_id"], how="left")
    return out


def _load_annotation_context(session_dir: Path) -> pd.DataFrame:
    annot_dir = session_dir / "annot"
    beh_dir = session_dir / "beh"
    rows: list[dict] = []
    for task in ["T0", "T1", "T2", "T3", "T4"]:
        ann_path = next(iter(sorted(annot_dir.glob(f"*_task-{task}_run-01_annotations.json"))), None)
        entries_count = np.nan
        tiers_count = np.nan
        if ann_path and ann_path.exists():
            try:
                payload = json.loads(ann_path.read_text(encoding="utf-8"))
                entries_count = float(len(payload.get("entries", [])))
                tiers_count = float(len(payload.get("tiers", [])))
            except Exception:
                entries_count = np.nan
                tiers_count = np.nan

        events_path = next(iter(sorted(beh_dir.glob(f"*_task-{task}_run-01_events.tsv"))), None)
        events_count = np.nan
        if events_path and events_path.exists():
            try:
                events_count = float(max(0, len(pd.read_csv(events_path, sep="\t"))))
            except Exception:
                events_count = np.nan

        rows.append(
            {
                "task": task,
                "annotation_entries_count": entries_count,
                "annotation_tiers_count": tiers_count,
                "task_events_count": events_count,
            }
        )
    return pd.DataFrame(rows)


def _load_annotation_participant_context(session_dir: Path) -> pd.DataFrame:
    beh_dir = session_dir / "beh"
    rows: list[dict] = []
    for events_path in sorted(beh_dir.glob("*_task-T*_run-01_events.tsv")):
        try:
            df = pd.read_csv(events_path, sep="\t")
        except Exception:
            continue
        if df.empty or "description" not in df.columns:
            continue
        task_m = re.search(r"task-(T\d)", events_path.name)
        task = task_m.group(1) if task_m else ""
        if not task:
            continue
        trial = df.get("trial_type", pd.Series([""] * len(df))).astype(str).str.strip()
        pids = df["description"].astype(str).map(_participant_from_description)
        onsets = pd.to_numeric(df.get("onset", pd.Series([np.nan] * len(df))), errors="coerce")
        tmp = pd.DataFrame({"task": task, "participant_id": pids, "trial_type": trial, "onset": onsets})
        tmp = tmp[tmp["participant_id"] != ""]
        if tmp.empty:
            continue
        for pid, pdf in tmp.groupby("participant_id"):
            onset = pd.to_numeric(pdf["onset"], errors="coerce")
            duration = float(onset.max() - onset.min()) if onset.notna().sum() >= 2 else np.nan
            rows.append(
                {
                    "task": task,
                    "participant_id": pid,
                    "ann_total_events_n": float(len(pdf)),
                    "ann_response_vad_n": float((pdf["trial_type"] == "response_vad").sum()),
                    "ann_response_postblock_n": float((pdf["trial_type"] == "response_postblock").sum()),
                    "ann_response_form_n": float((pdf["trial_type"] == "response_form").sum()),
                    "ann_push_vad_n": float((pdf["trial_type"] == "push_vad").sum()),
                    "ann_event_span_s": duration,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.groupby(["task", "participant_id"], as_index=False).mean(numeric_only=True)
    return out


def _enrich_vad_labels(participant: pd.DataFrame) -> pd.DataFrame:
    out = participant.copy()
    valence_num = pd.to_numeric(out.get("ans_valence"), errors="coerce")
    if valence_num.isna().all():
        valence_num = pd.to_numeric(out.get("ans_overall_valence"), errors="coerce")
    arousal_num = pd.to_numeric(out.get("ans_arousal"), errors="coerce")
    dominance_num = pd.to_numeric(out.get("ans_dominance"), errors="coerce")
    if dominance_num.isna().all():
        dominance_num = pd.to_numeric(out.get("ans_perceived_control"), errors="coerce")

    out["vad_valence_self_num"] = valence_num
    out["vad_arousal_self_num"] = arousal_num
    out["vad_dominance_self_num"] = dominance_num
    out["vad_valence_self_label"] = valence_num.map(_label_from_vad_rating)
    out["vad_arousal_self_label"] = arousal_num.map(_label_from_vad_rating)
    out["vad_dominance_self_label"] = dominance_num.map(_label_from_vad_rating)

    b_arousal = pd.to_numeric(out.get("biomarker_arousal_stress"), errors="coerce")
    b_recovery = pd.to_numeric(out.get("biomarker_recovery_capacity"), errors="coerce")
    b_fatigue = pd.to_numeric(out.get("biomarker_fatigue_depletion"), errors="coerce")
    b_decision = pd.to_numeric(out.get("biomarker_decision_pressure"), errors="coerce")
    b_attention = pd.to_numeric(out.get("biomarker_attention"), errors="coerce")

    valence_pred_score = b_recovery - b_fatigue
    valence_pred_score = valence_pred_score.where(valence_pred_score.notna(), b_recovery)
    valence_pred_score = valence_pred_score.where(valence_pred_score.notna(), -b_fatigue)
    dominance_pred_score = b_decision.where(b_decision.notna(), b_attention)

    out["vad_arousal_pred_score"] = b_arousal
    out["vad_valence_pred_score"] = valence_pred_score
    out["vad_dominance_pred_score"] = dominance_pred_score
    out["vad_arousal_pred_label"] = b_arousal.map(_label_from_biomarker_score)
    out["vad_valence_pred_label"] = valence_pred_score.map(_label_from_biomarker_score)
    out["vad_dominance_pred_label"] = dominance_pred_score.map(_label_from_biomarker_score)

    for dim in ["valence", "arousal", "dominance"]:
        self_col = f"vad_{dim}_self_label"
        pred_col = f"vad_{dim}_pred_label"
        match_col = f"vad_{dim}_label_match"
        valid = (out[self_col] != "") & (out[pred_col] != "")
        out[match_col] = np.where(valid, out[self_col] == out[pred_col], np.nan)

    if "ann_response_vad_n" in out.columns:
        out["ann_vad_activity_label"] = (
            out.groupby(["session_id", "task"])["ann_response_vad_n"]
            .transform(_quantile_labels)
            .astype(str)
        )
        valid = (out["ann_vad_activity_label"] != "") & (out["vad_arousal_pred_label"] != "")
        out["vad_arousal_vs_annotation_match"] = np.where(
            valid,
            out["ann_vad_activity_label"] == out["vad_arousal_pred_label"],
            np.nan,
        )
    return out


def _performance_long(
    participant: pd.DataFrame,
    dims: Iterable[str],
    self_prefix: str,
    pred_prefix: str,
    match_prefix: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for (sid, pid), pdf in participant.groupby(["session_id", "participant_id"]):
        for dim in dims:
            self_col = f"{self_prefix}{dim}_self_label"
            pred_col = f"{pred_prefix}{dim}_pred_label"
            match_col = f"{match_prefix}{dim}_label_match"
            if self_col not in pdf.columns or pred_col not in pdf.columns or match_col not in pdf.columns:
                continue
            valid = pdf[(pdf[self_col] != "") & (pdf[pred_col] != "")]
            n = int(len(valid))
            acc = float(valid[match_col].astype(float).mean()) if n else np.nan
            rows.append(
                {
                    "session_id": sid,
                    "participant_id": pid,
                    "dimension": dim,
                    "n_compared": n,
                    "label_agreement": acc,
                }
            )
    return pd.DataFrame(rows)


def _group_pool(df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["session_id", "group_id", "task"]
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"session_id", "group_id", "task", "participant_id", "source_file_physio", "source_file_pupil"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    pooled = df[group_keys + ["participant_id"] + numeric_cols].copy()
    out = pooled.groupby(group_keys, as_index=False).agg(participant_n=("participant_id", "nunique"))
    if not numeric_cols:
        return out
    gb = pooled.groupby(group_keys)
    mean_df = gb[numeric_cols].mean().reset_index()
    std_df = gb[numeric_cols].std(ddof=0).reset_index()
    mean_df = mean_df.rename(columns={c: f"{c}_group_mean" for c in numeric_cols})
    std_df = std_df.rename(columns={c: f"{c}_group_std" for c in numeric_cols})
    out = out.merge(mean_df, on=group_keys, how="left")
    out = out.merge(std_df, on=group_keys, how="left")
    return out


def _participant_vs_group(df: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    keys = ["session_id", "group_id", "task"]
    merged = df.merge(pooled, on=keys, how="left")
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"session_id", "group_id", "task", "participant_id", "source_file_physio", "source_file_pupil"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    additions: dict[str, pd.Series] = {}
    for col in numeric_cols:
        mean_col = f"{col}_group_mean"
        std_col = f"{col}_group_std"
        if col not in merged.columns or mean_col not in merged.columns:
            continue
        additions[f"{col}_delta_vs_group"] = merged[col] - merged[mean_col]
        if std_col in merged.columns:
            additions[f"{col}_z_vs_group"] = np.where(
                (merged[std_col].notna()) & (merged[std_col] > 0),
                (merged[col] - merged[mean_col]) / merged[std_col],
                np.nan,
            )
    if additions:
        merged = pd.concat([merged, pd.DataFrame(additions, index=merged.index)], axis=1)
    return merged


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Join participant features with answers and group pooled comparisons.")
    p.add_argument("--data-root", type=Path, required=True, help="Dataset root (e.g. .../seed/data).")
    p.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory containing semantic_biomarkers_participant_task.tsv.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: features-dir).",
    )
    p.add_argument("--sessions", nargs="*", default=None, help="Optional session ids filter.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    features_dir = args.features_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else features_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_path = features_dir / "semantic_biomarkers_participant_task.tsv"
    if not sem_path.exists():
        raise FileNotFoundError(f"Missing input: {sem_path}")
    sem = pd.read_csv(sem_path, sep="\t")
    if sem.empty:
        LOG.warning("Semantic participant-task table is empty; writing empty outputs.")
        pd.DataFrame().to_csv(out_dir / "participant_features_answers_annotations.tsv", sep="\t", index=False)
        pd.DataFrame().to_csv(out_dir / "group_pool_task_summary.tsv", sep="\t", index=False)
        pd.DataFrame().to_csv(out_dir / "participant_vs_group_comparison.tsv", sep="\t", index=False)
        return 0

    sem["group_id"] = sem["session_id"].astype(str).map(_group_id_from_session)

    session_dirs = discover_session_dirs(args.data_root, args.sessions)
    answer_rows: list[pd.DataFrame] = []
    annot_rows: list[pd.DataFrame] = []
    annot_participant_rows: list[pd.DataFrame] = []
    for session_dir in session_dirs:
        sid = session_dir.name
        a = _load_answers_by_participant_task(session_dir)
        if not a.empty:
            a["session_id"] = sid
            a["group_id"] = _group_id_from_session(sid)
            answer_rows.append(a)
        c = _load_annotation_context(session_dir)
        if not c.empty:
            c["session_id"] = sid
            c["group_id"] = _group_id_from_session(sid)
            annot_rows.append(c)
        cp = _load_annotation_participant_context(session_dir)
        if not cp.empty:
            cp["session_id"] = sid
            cp["group_id"] = _group_id_from_session(sid)
            annot_participant_rows.append(cp)

    answers = pd.concat(answer_rows, ignore_index=True) if answer_rows else pd.DataFrame()
    annot = pd.concat(annot_rows, ignore_index=True) if annot_rows else pd.DataFrame()
    annot_participant = pd.concat(annot_participant_rows, ignore_index=True) if annot_participant_rows else pd.DataFrame()

    participant = sem.copy()
    if not answers.empty:
        participant = participant.merge(
            answers,
            on=["session_id", "group_id", "task", "participant_id"],
            how="left",
        )
    if not annot.empty:
        participant = participant.merge(
            annot,
            on=["session_id", "group_id", "task"],
            how="left",
        )
    if not annot_participant.empty:
        participant = participant.merge(
            annot_participant,
            on=["session_id", "group_id", "task", "participant_id"],
            how="left",
        )
    participant = _enrich_vad_labels(participant)

    pooled = _group_pool(participant)
    participant_vs_group = _participant_vs_group(participant, pooled)
    if not annot.empty:
        pooled = pooled.merge(annot, on=["session_id", "group_id", "task"], how="left")

    participant_path = out_dir / "participant_features_answers_annotations.tsv"
    pooled_path = out_dir / "group_pool_task_summary.tsv"
    compare_path = out_dir / "participant_vs_group_comparison.tsv"
    vad_compare_path = out_dir / "biomarker_vad_label_comparison.tsv"
    vad_perf_path = out_dir / "biomarker_vad_performance_by_participant.tsv"
    ann_perf_path = out_dir / "biomarker_annotation_performance_by_participant.tsv"

    participant.sort_values(["session_id", "task", "participant_id"]).to_csv(
        participant_path, sep="\t", index=False
    )
    pooled.sort_values(["session_id", "task"]).to_csv(
        pooled_path, sep="\t", index=False
    )
    participant_vs_group.sort_values(["session_id", "task", "participant_id"]).to_csv(
        compare_path, sep="\t", index=False
    )

    vad_cols = [
        "session_id",
        "group_id",
        "task",
        "participant_id",
        "vad_valence_self_num",
        "vad_arousal_self_num",
        "vad_dominance_self_num",
        "vad_valence_self_label",
        "vad_arousal_self_label",
        "vad_dominance_self_label",
        "vad_valence_pred_score",
        "vad_arousal_pred_score",
        "vad_dominance_pred_score",
        "vad_valence_pred_label",
        "vad_arousal_pred_label",
        "vad_dominance_pred_label",
        "vad_valence_label_match",
        "vad_arousal_label_match",
        "vad_dominance_label_match",
        "ann_response_vad_n",
        "ann_vad_activity_label",
        "vad_arousal_vs_annotation_match",
    ]
    vad_cols = [c for c in vad_cols if c in participant.columns]
    participant[vad_cols].sort_values(["session_id", "task", "participant_id"]).to_csv(
        vad_compare_path, sep="\t", index=False
    )
    perf = _performance_long(
        participant,
        dims=["valence", "arousal", "dominance"],
        self_prefix="vad_",
        pred_prefix="vad_",
        match_prefix="vad_",
    )
    perf.sort_values(["session_id", "participant_id", "dimension"]).to_csv(
        vad_perf_path, sep="\t", index=False
    )
    ann_perf_rows = []
    if "vad_arousal_vs_annotation_match" in participant.columns:
        for (sid, pid), pdf in participant.groupby(["session_id", "participant_id"]):
            valid = pdf[
                (pdf.get("ann_vad_activity_label", "") != "")
                & (pdf.get("vad_arousal_pred_label", "") != "")
            ]
            ann_perf_rows.append(
                {
                    "session_id": sid,
                    "participant_id": pid,
                    "n_compared": int(len(valid)),
                    "label_agreement": float(valid["vad_arousal_vs_annotation_match"].astype(float).mean())
                    if len(valid)
                    else np.nan,
                }
            )
    pd.DataFrame(ann_perf_rows).to_csv(ann_perf_path, sep="\t", index=False)

    LOG.info("Wrote %s (%d rows)", participant_path, len(participant))
    LOG.info("Wrote %s (%d rows)", pooled_path, len(pooled))
    LOG.info("Wrote %s (%d rows)", compare_path, len(participant_vs_group))
    LOG.info("Wrote %s", vad_compare_path)
    LOG.info("Wrote %s", vad_perf_path)
    LOG.info("Wrote %s", ann_perf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
