"""Compute and visualise RMS energy levels across all DPA recordings.

For every DPA close-talk WAV file found under --audio-root the script
computes a time-series of RMS energy (default: 500 ms windows, 250 ms hop)
and produces four figures:

  rms_timeseries_<session>.png  — per-session figure: one row per task,
                                   one line per participant (P1–P4)
  rms_heatmap_task_participant.png — mean RMS per task × participant,
                                      one panel per session
  rms_boxplot_by_task.png        — distribution of frame RMS per task,
                                   all sessions pooled, split by participant
  rms_summary.tsv                — mean/median/SD/max RMS per
                                   session × task × participant

Mic → participant mapping (fixed across all sessions):
  mic9 → P1,  mic10 → P2,  mic11 → P3,  mic12 → P4

# Privacy: participant IDs are P1–P4 only; no real names are stored.

Usage:
    python tools/features/plot_rms_levels.py \\
        --audio-root "C:/path/to/audio data" \\
        --out-dir figures/rms

    # Restrict to specific tasks or groups:
    python tools/features/plot_rms_levels.py \\
        --audio-root "..." --tasks T0 T2 --groups grp-07 grp-08

    # Larger windows for smoother traces:
    python tools/features/plot_rms_levels.py \\
        --audio-root "..." --window-s 1.0 --hop-s 0.5
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import soundfile as sf

LOG = logging.getLogger("plot_rms_levels")

# ── Constants ──────────────────────────────────────────────────────────────────

MIC_TO_PARTICIPANT: dict[str, str] = {
    "mic9": "P1",
    "mic10": "P2",
    "mic11": "P3",
    "mic12": "P4",
}

PARTICIPANT_COLORS: dict[str, str] = {
    "P1": "#1f77b4",
    "P2": "#ff7f0e",
    "P3": "#2ca02c",
    "P4": "#d62728",
}

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]

_WAV_RE = re.compile(
    r"(?:sub-(?P<sub>[^_]+)_)?"
    r"ses-(?P<ses_date>\d{8})_"
    r"(?P<grp>grp-\d+)_"
    r"run(?P<ses_run>\d+)"
    r".*?"
    r"_task-(?P<task>T\d+)_"
    r".*?"
    r"_acq-dpa_(?P<mic>mic\d+)_aud\.wav$",
    re.IGNORECASE,
)

# ── Signal helpers ─────────────────────────────────────────────────────────────


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _rms_timeseries(
    audio: np.ndarray,
    sr: int,
    window_s: float,
    hop_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, rms_values) arrays for a sliding RMS window."""
    win = max(1, int(window_s * sr))
    hop = max(1, int(hop_s * sr))
    n = len(audio)
    starts = np.arange(0, n - win + 1, hop)
    rms = np.array([
        float(np.sqrt(np.mean(audio[s: s + win].astype(np.float64) ** 2)))
        for s in starts
    ])
    times = (starts + win / 2) / sr
    return times, rms


def _db(rms: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    """Convert linear RMS to dBFS; clamp to floor_db."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return np.maximum(db, floor_db)


# ── Discovery ──────────────────────────────────────────────────────────────────


def _parse_session_id(session_dir: Path) -> tuple[str, str]:
    name = session_dir.name
    grp_m = re.search(r"(grp-\d+)", name)
    date_m = re.search(r"(\d{8})", name)
    run_m = re.search(r"run(\d+)", name, re.IGNORECASE)
    group_id = grp_m.group(1) if grp_m else "unknown"
    date_s = date_m.group(1) if date_m else "unknown"
    run_s = run_m.group(1) if run_m else "01"
    return f"ses-{date_s}_{group_id}_run{run_s}", group_id


def _discover_sessions(audio_root: Path, groups: list[str] | None) -> list[Path]:
    sessions = []
    for d in sorted(audio_root.iterdir()):
        if not d.is_dir():
            continue
        if groups and not any(g.lower() in d.name.lower() for g in groups):
            continue
        # Accept either session/audio/ layout or session/ with WAVs at top level
        if not (d / "audio").is_dir() and not list(d.glob("*_acq-dpa_mic*_aud.wav")):
            continue
        sessions.append(d)
    return sessions


def _discover_wavs(
    audio_dir: Path, tasks: list[str] | None
) -> list[tuple[str, str, Path]]:
    """Return list of (task_id, mic_id, wav_path) tuples."""
    result = []
    for wav in sorted(audio_dir.glob("*_acq-dpa_mic*_aud.wav")):
        name = wav.name.lower()
        if re.search(r"_aud_hp", name) or "peaks" in wav.parts:
            continue
        m = _WAV_RE.search(wav.name)
        if m is None:
            continue
        task, mic = m.group("task"), m.group("mic")
        if tasks and task not in tasks:
            continue
        result.append((task, mic, wav))
    return result


# ── Core computation ───────────────────────────────────────────────────────────


def compute_session_rms(
    session_dir: Path,
    tasks: list[str] | None,
    window_s: float,
    hop_s: float,
) -> tuple[list[dict], list[dict]]:
    """Return (summary_rows, timeseries_rows) for one session.

    summary_rows  — one row per session × task × participant with aggregate stats.
    timeseries_rows — one row per time window per mic per task.
    # Privacy: participant IDs are P1–P4 only; no real names are stored.
    """
    session_id, group_id = _parse_session_id(session_dir)
    audio_dir = session_dir / "audio" if (session_dir / "audio").is_dir() else session_dir
    wavs = _discover_wavs(audio_dir, tasks)

    summary_rows: list[dict] = []
    ts_rows: list[dict] = []

    for task_id, mic, wav_path in wavs:
        participant_id = MIC_TO_PARTICIPANT.get(mic)
        if participant_id is None:
            LOG.debug("Unknown mic %s — skipping", mic)
            continue

        try:
            audio, sr = _load_mono(wav_path)
        except Exception as exc:
            LOG.warning("[%s | %s | %s] load failed: %s", session_id, task_id, mic, exc)
            continue

        times, rms = _rms_timeseries(audio, sr, window_s, hop_s)
        db_vals = _db(rms)

        summary_rows.append({
            "session_id": session_id,
            "group_id": group_id,
            "task_id": task_id,
            "participant_id": participant_id,
            "duration_s": round(len(audio) / sr, 2),
            "rms_mean": float(np.mean(rms)),
            "rms_median": float(np.median(rms)),
            "rms_sd": float(np.std(rms)),
            "rms_max": float(np.max(rms)),
            "rms_db_mean": float(np.mean(db_vals)),
            "rms_db_median": float(np.median(db_vals)),
            "rms_db_sd": float(np.std(db_vals)),
            "rms_db_max": float(np.max(db_vals)),
        })

        for t, r, d in zip(times, rms, db_vals):
            ts_rows.append({
                "session_id": session_id,
                "group_id": group_id,
                "task_id": task_id,
                "participant_id": participant_id,
                "time_s": round(float(t), 4),
                "rms": round(float(r), 8),
                "rms_db": round(float(d), 4),
            })

        LOG.info(
            "[%s | %s | %s] %.1fs  mean=%.4f (%.1f dBFS)",
            session_id, task_id, participant_id,
            len(audio) / sr, float(np.mean(rms)), float(np.mean(db_vals)),
        )

    return summary_rows, ts_rows


# ── Plotting ───────────────────────────────────────────────────────────────────


def _tasks_present(ts_df: pd.DataFrame, session_id: str) -> list[str]:
    sess = ts_df[ts_df["session_id"] == session_id]
    found = [t for t in TASK_ORDER if t in sess["task_id"].values]
    return found or sorted(sess["task_id"].unique())


def plot_timeseries(ts_df: pd.DataFrame, out_dir: Path) -> None:
    """One figure per session: rows = tasks, lines = participants."""
    for session_id in sorted(ts_df["session_id"].unique()):
        sess_df = ts_df[ts_df["session_id"] == session_id]
        tasks = _tasks_present(ts_df, session_id)
        n_tasks = len(tasks)
        if n_tasks == 0:
            continue

        fig, axes = plt.subplots(
            n_tasks, 1,
            figsize=(14, 2.8 * n_tasks),
            sharex=False,
            squeeze=False,
        )
        fig.suptitle(f"RMS energy over time — {session_id}", fontsize=13, fontweight="bold", y=1.01)

        for row_idx, task_id in enumerate(tasks):
            ax = axes[row_idx][0]
            task_df = sess_df[sess_df["task_id"] == task_id]

            for pid in ["P1", "P2", "P3", "P4"]:
                p_df = task_df[task_df["participant_id"] == pid].sort_values("time_s")
                if p_df.empty:
                    continue
                ax.plot(
                    p_df["time_s"].values,
                    p_df["rms_db"].values,
                    color=PARTICIPANT_COLORS[pid],
                    linewidth=0.9,
                    alpha=0.85,
                    label=pid,
                )

            ax.set_ylabel("RMS (dBFS)", fontsize=9)
            ax.set_title(f"Task {task_id}", fontsize=10, loc="left", pad=4)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.set_xlim(left=0)
            if row_idx == 0:
                ax.legend(title="Participant", fontsize=8, title_fontsize=8,
                          loc="upper right", framealpha=0.7)

        axes[-1][0].set_xlabel("Time (s)", fontsize=9)
        fig.tight_layout()
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
        out_path = out_dir / f"rms_timeseries_{safe_name}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        LOG.info("Saved %s", out_path)


def plot_heatmap(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Mean RMS (dBFS) heatmap: tasks × participants, one panel per session."""
    sessions = sorted(summary_df["session_id"].unique())
    n = len(sessions)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle("Mean RMS (dBFS) — task × participant per session",
                 fontsize=13, fontweight="bold")

    for idx, session_id in enumerate(sessions):
        ax = axes[idx // ncols][idx % ncols]
        sess = summary_df[summary_df["session_id"] == session_id]

        tasks = [t for t in TASK_ORDER if t in sess["task_id"].values]
        participants = ["P1", "P2", "P3", "P4"]
        matrix = np.full((len(tasks), len(participants)), np.nan)
        for r, task in enumerate(tasks):
            for c, pid in enumerate(participants):
                val = sess[(sess["task_id"] == task) & (sess["participant_id"] == pid)]["rms_db_mean"]
                if not val.empty:
                    matrix[r, c] = val.iloc[0]

        vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else -60
        vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 0
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(participants)))
        ax.set_xticklabels(participants, fontsize=9)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, fontsize=9)
        ax.set_title(session_id, fontsize=8, pad=6)

        # Annotate cells
        for r in range(len(tasks)):
            for c in range(len(participants)):
                val = matrix[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color="white" if val < (vmin + vmax) / 2 else "black")
                else:
                    ax.text(c, r, "N/A", ha="center", va="center", fontsize=6, color="grey")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dBFS")

    # Hide unused panels
    for idx in range(len(sessions), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / "rms_heatmap_task_participant.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", out_path)


def plot_boxplot(ts_df: pd.DataFrame, out_dir: Path) -> None:
    """Distribution of frame RMS per task × participant, all sessions pooled."""
    tasks = [t for t in TASK_ORDER if t in ts_df["task_id"].values]
    participants = ["P1", "P2", "P3", "P4"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.5 * len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    fig.suptitle("RMS distribution per task × participant (all sessions pooled)",
                 fontsize=12, fontweight="bold")

    for ax, task_id in zip(axes, tasks):
        data = [
            ts_df[(ts_df["task_id"] == task_id) & (ts_df["participant_id"] == pid)]["rms_db"].dropna().values
            for pid in participants
        ]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
            widths=0.55,
        )
        for patch, pid in zip(bp["boxes"], participants):
            patch.set_facecolor(PARTICIPANT_COLORS[pid])
            patch.set_alpha(0.8)

        ax.set_xticks(range(1, len(participants) + 1))
        ax.set_xticklabels(participants, fontsize=9)
        ax.set_title(f"Task {task_id}", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if ax is axes[0]:
            ax.set_ylabel("RMS (dBFS)", fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "rms_boxplot_by_task.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Saved %s", out_path)


# ── CLI ────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--audio-root", type=Path, required=True,
        help=(
            "Root directory containing per-session subdirectories, each with "
            "an audio/ folder. Example: 'C:/data/ses-20260319_grp-15_run01-...'"
        ),
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("figures/rms"),
        help="Output directory for figures and TSV. Created if absent. Default: figures/rms",
    )
    ap.add_argument(
        "--tasks", nargs="+", default=None, metavar="TASK",
        help="Restrict to specific tasks, e.g. --tasks T0 T1 T2. Default: all.",
    )
    ap.add_argument(
        "--groups", nargs="+", default=None, metavar="GRP",
        help="Restrict to specific group IDs, e.g. --groups grp-07 grp-15.",
    )
    ap.add_argument(
        "--window-s", type=float, default=0.5,
        help="RMS window length in seconds (default: 0.5).",
    )
    ap.add_argument(
        "--hop-s", type=float, default=0.25,
        help="RMS window hop in seconds (default: 0.25).",
    )
    ap.add_argument(
        "--no-timeseries", action="store_true",
        help="Skip per-session time-series figures (faster for many sessions).",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.audio_root.exists():
        LOG.error("Audio root not found: %s", args.audio_root)
        return 1

    sessions = _discover_sessions(args.audio_root, args.groups)
    if not sessions:
        LOG.error("No session directories with audio/ found under %s", args.audio_root)
        return 1
    LOG.info("Found %d session(s)", len(sessions))

    all_summary: list[dict] = []
    all_ts: list[dict] = []

    for session_dir in sessions:
        LOG.info("=== %s ===", session_dir.name)
        summary, ts = compute_session_rms(session_dir, args.tasks, args.window_s, args.hop_s)
        all_summary.extend(summary)
        all_ts.extend(ts)

    if not all_summary:
        LOG.error("No data found — check --audio-root and file naming.")
        return 1

    summary_df = pd.DataFrame(all_summary).sort_values(
        ["session_id", "task_id", "participant_id"]
    ).reset_index(drop=True)

    ts_df = pd.DataFrame(all_ts)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary TSV
    tsv_path = args.out_dir / "rms_summary.tsv"
    summary_df.to_csv(tsv_path, sep="\t", index=False)
    LOG.info("Wrote %s (%d rows)", tsv_path, len(summary_df))

    # Figures
    if not args.no_timeseries:
        plot_timeseries(ts_df, args.out_dir)

    if not summary_df.empty:
        plot_heatmap(summary_df, args.out_dir)

    if not ts_df.empty:
        plot_boxplot(ts_df, args.out_dir)

    LOG.info("Done. Figures written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
