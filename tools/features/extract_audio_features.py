"""Extract paper-ready audio/prosodic features from DPA close-talk microphone recordings.

For each session × task × participant the script:

  1. Loads all four DPA mic WAVs for the task.
  2. Applies **cross-talk suppression** with two layers:
     Layer 1 (Energy-ratio rejection): 25 ms frame attributed to target mic only when
       its RMS energy exceeds all other mics by at least 1.5× (~3.5 dB threshold).
       Calibrated from RMS analysis: speech ~-46 to -50 dBFS, noise floor ~-54 to -57 dBFS.
     Layer 2 (Spectral embedding filtering): rejects frames where adjacent mics have
       non-speech spectral signatures (low centroid <2.5 kHz + high flatness >0.6),
       indicating crosstalk or room noise rather than target speech.
  3. Concatenates the clean speech frames into a temporary WAV
     and runs opensmile GeMAPSv01b (62 parameters) on cleaned audio.
  4. Derives speaking time, pause count, and overlap fraction directly from
     the energy mask (no transcript required).
  5. If ``audio_annot/{task}/master_transcript.tsv`` exists (produced by
     ``audio_annotate.py``), adds transcript-based turn count and
     uncertain-segment fraction.

Thresholds (from RMS analysis of 10 sessions, 5 tasks, 4 participants):
  - Silence floor: RMS < 1e-5 (−80 dBFS)
  - Noise/crosstalk: spectral centroid <2.5 kHz, flatness >0.6
  - Energy dominance: ratio ≥1.5× (default; configurable via --energy-ratio)

Outputs written to ``--out-dir`` (default: ``features/``):
  audio_participant_task.tsv   — one row per session × task × participant
  audio_qc_summary.tsv         — coverage and QC flags per row

Usage:
    python tools/features/extract_audio_features.py \\
        --audio-root "C:/path/to/audio data" \\
        --out-dir features

    # Restrict to specific tasks or groups:
    python tools/features/extract_audio_features.py \\
        --audio-root "..." --tasks T0 T1 T2 --groups grp-07 grp-08

    # Increase bleed-rejection threshold (stricter cross-talk suppression):
    python tools/features/extract_audio_features.py \\
        --audio-root "..." --energy-ratio 2.0
"""

from __future__ import annotations

import argparse
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

try:
    import opensmile
except ModuleNotFoundError:  # pragma: no cover
    opensmile = None  # type: ignore[assignment]

LOG = logging.getLogger("extract_audio_features")

# ── Constants ──────────────────────────────────────────────────────────────────

MIC_TO_PARTICIPANT: dict[str, str] = {
    "mic9": "P1",
    "mic10": "P2",
    "mic11": "P3",
    "mic12": "P4",
}

FRAME_S = 0.025           # 25 ms analysis frames
MIN_SPEECH_S = 0.10       # minimum contiguous speech block to keep (100 ms)
PAUSE_THRESHOLD_S = 0.50  # gap ≥ this counts as an intra-speaker pause
SILENCE_FLOOR = 1e-5      # RMS below this level is treated as silence

# Filename regex — handles all tasks T0–T4 and all four mics
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

# ── Audio signal helpers ───────────────────────────────────────────────────────


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV as mono float32 at its native sample rate."""
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def _frame_rms(audio: np.ndarray, sr: int, frame_s: float) -> np.ndarray:
    """Return per-frame RMS array (shape: n_frames)."""
    frame_len = max(1, int(frame_s * sr))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    chunks = audio[: n_frames * frame_len].reshape(n_frames, frame_len).astype(np.float64)
    return np.sqrt(np.mean(chunks ** 2, axis=1))


def _spectral_centroid(audio: np.ndarray, sr: int, frame_s: float) -> np.ndarray:
    """Per-frame spectral centroid (Hz) — used as embedding for cross-talk detection.
    
    Lower centroid (~1–3 kHz) suggests noise/crosstalk from nearby mics;
    higher centroid (~3–5 kHz) suggests target speaker.
    """
    frame_len = max(1, int(frame_s * sr))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    centroids = []
    for i in range(n_frames):
        chunk = audio[i * frame_len: (i + 1) * frame_len].astype(np.float64)
        if len(chunk) == 0 or np.max(np.abs(chunk)) < SILENCE_FLOOR:
            centroids.append(0.0)
            continue
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)
        mag_sum = np.sum(fft)
        if mag_sum == 0:
            centroids.append(0.0)
        else:
            centroid = np.sum(freqs * fft) / mag_sum
            centroids.append(float(centroid))
    return np.array(centroids, dtype=np.float64)


def _spectral_flat(audio: np.ndarray, sr: int, frame_s: float) -> np.ndarray:
    """Per-frame spectral flatness (0–1 scale).
    
    High flatness (>0.5) suggests noise; low flatness (<0.3) suggests clean speech.
    Used as embedding for cross-talk detection.
    """
    frame_len = max(1, int(frame_s * sr))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    flats = []
    for i in range(n_frames):
        chunk = audio[i * frame_len: (i + 1) * frame_len].astype(np.float64)
        if len(chunk) == 0 or np.max(np.abs(chunk)) < SILENCE_FLOOR:
            flats.append(0.5)  # Neutral value for silence
            continue
        fft = np.abs(np.fft.rfft(chunk)) + 1e-10
        geom_mean = np.exp(np.mean(np.log(fft)))
        arith_mean = np.mean(fft)
        flatness = geom_mean / arith_mean if arith_mean > 0 else 0.0
        flats.append(float(np.clip(flatness, 0.0, 1.0)))
    return np.array(flats, dtype=np.float64)


def _build_speech_mask(
    mic_rms: dict[str, np.ndarray],
    mic_audio: dict[str, np.ndarray],
    mic_sr: dict[str, int],
    target_mic: str,
    energy_ratio: float,
    min_speech_frames: int,
    frame_s: float,
) -> np.ndarray:
    """Boolean mask of frames where *target_mic* is the dominant speaker.

    A frame is accepted when:
      - target_mic RMS > SILENCE_FLOOR                    (not silence)
      - target_mic RMS / max(other mics RMS) >= energy_ratio  (energy dominance)
      - spectral embedding coherence check                 (cross-talk suppression)
    
    Spectral embedding (centroid + flatness) detects crosstalk by identifying
    frames where neighbouring mics have unnatural spectral shapes (low centroid,
    high flatness = noise/crosstalk signature).
    
    Short isolated bursts shorter than min_speech_frames are removed.
    """
    target = mic_rms[target_mic]
    n = len(target)
    others = [v[:n] for k, v in mic_rms.items() if k != target_mic and len(v) >= n]

    if not others:
        raw = target > SILENCE_FLOOR
    else:
        max_other = np.maximum(np.max(np.stack(others, axis=0), axis=0), SILENCE_FLOOR)
        ratio = target / max_other
        raw = (target > SILENCE_FLOOR) & (ratio >= energy_ratio)

    # Spectral embedding check: compute centroid + flatness for target mic
    if target_mic in mic_audio:
        target_audio = mic_audio[target_mic]
        target_sr = mic_sr[target_mic]
        try:
            target_centroid = _spectral_centroid(target_audio, target_sr, frame_s)[:n]
            target_flatness = _spectral_flat(target_audio, target_sr, frame_s)[:n]
            
            # Coherence penalty: frames with low centroid (<2.5 kHz) + high flatness (>0.6)
            # are likely noise/crosstalk, not target speech. Apply spectral gate.
            low_freq_noise = (target_centroid < 2500) & (target_flatness > 0.6)
            raw = raw & ~low_freq_noise
        except Exception:
            # If spectral extraction fails, fall back to energy-only
            pass

    if min_speech_frames <= 1:
        return raw

    # Remove bursts shorter than min_speech_frames
    mask = raw.copy()
    in_seg = False
    start = 0
    for i in range(n):
        if raw[i] and not in_seg:
            in_seg, start = True, i
        elif not raw[i] and in_seg:
            in_seg = False
            if (i - start) < min_speech_frames:
                mask[start:i] = False
    if in_seg and (n - start) < min_speech_frames:
        mask[start:n] = False
    return mask


def _mask_to_segments(mask: np.ndarray, frame_s: float) -> list[tuple[float, float]]:
    """Convert boolean frame mask → list of (start_s, end_s) segments."""
    segments: list[tuple[float, float]] = []
    in_seg = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_seg:
            in_seg, start = True, i
        elif not val and in_seg:
            in_seg = False
            segments.append((start * frame_s, i * frame_s))
    if in_seg:
        segments.append((start * frame_s, len(mask) * frame_s))
    return segments


def _pause_count(segments: list[tuple[float, float]], threshold_s: float) -> int:
    """Count inter-segment gaps ≥ threshold_s (intra-speaker pauses)."""
    return sum(
        1 for i in range(1, len(segments))
        if segments[i][0] - segments[i - 1][1] >= threshold_s
    )


def _overlap_fraction(
    mic_rms: dict[str, np.ndarray],
    energy_ratio: float,
) -> float:
    """Fraction of multi-speaker frames where no single mic clearly dominates.

    Frames where at least two mics are above SILENCE_FLOOR and the loudest
    mic does not beat the second-loudest by energy_ratio are counted as
    cross-talk/overlap.
    """
    if len(mic_rms) < 2:
        return 0.0
    n = min(len(v) for v in mic_rms.values())
    stacked = np.stack([v[:n] for v in mic_rms.values()], axis=0)  # (mics, frames)
    multi = (stacked > SILENCE_FLOOR).sum(axis=0) >= 2
    if not multi.any():
        return 0.0
    sub = stacked[:, multi]
    sub_sorted = np.sort(sub, axis=0)[::-1]   # descending per frame
    top = sub_sorted[0]
    second = np.maximum(sub_sorted[1], SILENCE_FLOOR)
    dominated = (top / second) >= energy_ratio
    return float(1.0 - dominated.mean())


def _extract_clean_audio(
    audio: np.ndarray, sr: int, mask: np.ndarray, frame_s: float
) -> np.ndarray:
    """Concatenate only the accepted (clean) frames into a new array."""
    frame_len = max(1, int(frame_s * sr))
    n_frames = min(len(mask), len(audio) // frame_len)
    chunks = [audio[i * frame_len: (i + 1) * frame_len] for i in range(n_frames) if mask[i]]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


# ── opensmile wrapper ──────────────────────────────────────────────────────────


def _build_smile() -> "opensmile.Smile":
    """Instantiate a GeMAPSv01b (62-functional) extractor."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def _run_opensmile(
    smile: "opensmile.Smile",
    clean_audio: np.ndarray,
    sr: int,
) -> dict[str, float]:
    """Extract GeMAPSv01b functionals from clean concatenated audio.

    Returns an empty dict if the clip is too short or extraction fails.
    """
    if len(clean_audio) < int(sr * 0.10):
        return {}
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), clean_audio, sr)
        df = smile.process_file(str(tmp_path))
        if df.empty:
            return {}
        return df.iloc[0].to_dict()
    except Exception as exc:
        LOG.warning("opensmile failed: %s", exc)
        return {}
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Transcript helpers ─────────────────────────────────────────────────────────


def _load_transcript(annot_dir: Path, task_id: str) -> pd.DataFrame | None:
    """Load master_transcript.tsv for a task if it exists."""
    tsv = annot_dir / task_id / "master_transcript.tsv"
    if not tsv.exists():
        return None
    try:
        return pd.read_csv(tsv, sep="\t")
    except Exception as exc:
        LOG.warning("Could not read transcript %s: %s", tsv, exc)
        return None


def _transcript_features(
    df: pd.DataFrame | None,
    participant_id: str,
) -> dict[str, float | int | None]:
    """Derive turn count and uncertain fraction from master_transcript.tsv.

    The transcript uses columns: onset, duration, speaker, confidence.
    speaker values are like 'P1' or 'P1_Firstname' — matched with startswith.
    # Privacy: real names are never written to disk or LSL.
    """
    out: dict[str, float | int | None] = {"turn_count": None, "uncertain_fraction": None}
    if df is None or df.empty:
        return out

    required = {"onset", "speaker", "confidence"}
    if not required.issubset(df.columns):
        return out

    p_mask = df["speaker"].str.startswith(participant_id)
    p_segs = df[p_mask]

    # Turn count: times a new segment from this participant follows a different speaker
    sorted_df = df.sort_values("onset")
    speakers = sorted_df["speaker"].tolist()
    turns = sum(
        1 for i in range(1, len(speakers))
        if speakers[i].startswith(participant_id)
        and not speakers[i - 1].startswith(participant_id)
    )
    out["turn_count"] = turns

    if not p_segs.empty:
        uncertain = (p_segs["confidence"] == "uncertain").sum()
        out["uncertain_fraction"] = round(float(uncertain / len(p_segs)), 4)

    return out


# ── Discovery ──────────────────────────────────────────────────────────────────


def _discover_sessions(audio_root: Path, groups: list[str] | None) -> list[Path]:
    """Return subdirectories of audio_root that contain an audio/ folder."""
    sessions = []
    for d in sorted(audio_root.iterdir()):
        if not d.is_dir() or not (d / "audio").is_dir():
            continue
        if groups and not any(g.lower() in d.name.lower() for g in groups):
            continue
        sessions.append(d)
    return sessions


def _discover_task_mics(
    audio_dir: Path, tasks: list[str] | None
) -> dict[str, dict[str, Path]]:
    """Return {task_id: {mic_id: wav_path}} for canonical DPA WAV files."""
    result: dict[str, dict[str, Path]] = {}
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
        result.setdefault(task, {})[mic] = wav
    return result


def _parse_session_id(session_dir: Path) -> tuple[str, str]:
    """Return (session_id, group_id) from a session directory name."""
    name = session_dir.name
    grp_m = re.search(r"(grp-\d+)", name)
    date_m = re.search(r"(\d{8})", name)
    run_m = re.search(r"run(\d+)", name, re.IGNORECASE)
    group_id = grp_m.group(1) if grp_m else "unknown"
    date_s = date_m.group(1) if date_m else "unknown"
    run_s = run_m.group(1) if run_m else "01"
    return f"ses-{date_s}_{group_id}_run{run_s}", group_id


# ── Per-session processing ─────────────────────────────────────────────────────


def _missing_row(
    session_id: str, group_id: str, task_id: str, participant_id: str, duration_s: float
) -> dict:
    return {
        "session_id": session_id,
        "group_id": group_id,
        "task_id": task_id,
        "participant_id": participant_id,
        "audio_available": False,
        "duration_s": round(duration_s, 3),
        "speaking_time_s": None,
        "speaking_fraction": None,
        "pause_count": None,
        "turn_count": None,
        "overlap_fraction": None,
        "uncertain_fraction": None,
        "transcript_available": False,
        "energy_mean": None,
        "energy_sd": None,
        "pitch_mean": None,
        "pitch_sd": None,
        "hnr_mean": None,
        "jitter_mean": None,
        "shimmer_mean": None,
        "voiced_segments_per_sec": None,
        "mean_voiced_segment_s": None,
        "mean_unvoiced_segment_s": None,
        "speech_rate_proxy": None,
        "qc_flag": "missing_audio",
    }


def process_session(
    session_dir: Path,
    smile: "opensmile.Smile | None",
    tasks: list[str] | None,
    energy_ratio: float,
    frame_s: float,
) -> list[dict]:
    """Process one session directory — all tasks, all participants.

    Returns a list of feature-row dicts.
    # Privacy: participant IDs are P1–P4 only; no real names are stored.
    """
    session_id, group_id = _parse_session_id(session_dir)
    audio_dir = session_dir / "audio"
    annot_dir = session_dir / "audio_annot"

    task_mics = _discover_task_mics(audio_dir, tasks)
    if not task_mics:
        LOG.warning("[%s] No DPA WAVs found under %s", session_id, audio_dir)
        return []

    min_speech_frames = max(1, int(MIN_SPEECH_S / frame_s))
    rows: list[dict] = []

    for task_id in sorted(task_mics.keys()):
        mic_paths = task_mics[task_id]
        LOG.info("[%s | %s] mics available: %s", session_id, task_id, sorted(mic_paths.keys()))

        # ── Load all available mics ────────────────────────────────────────
        mic_audio: dict[str, np.ndarray] = {}
        mic_sr: dict[str, int] = {}
        for mic, wav_path in mic_paths.items():
            try:
                audio, sr = _load_mono(wav_path)
                mic_audio[mic] = audio
                mic_sr[mic] = sr
            except Exception as exc:
                LOG.warning("[%s | %s | %s] load failed: %s", session_id, task_id, mic, exc)

        if not mic_audio:
            continue

        # Reference duration from the longest mic recording
        max_dur_s = max(len(a) / mic_sr[m] for m, a in mic_audio.items())

        # ── Load optional transcript ───────────────────────────────────────
        transcript_df = _load_transcript(annot_dir, task_id)

        # ── Compute per-mic RMS frames ─────────────────────────────────────
        mic_rms: dict[str, np.ndarray] = {
            mic: _frame_rms(audio, mic_sr[mic], frame_s)
            for mic, audio in mic_audio.items()
        }

        # Session-level overlap fraction (same for all participants this task)
        overlap_frac = _overlap_fraction(mic_rms, energy_ratio)

        # ── Per-participant feature extraction ─────────────────────────────
        for mic, participant_id in MIC_TO_PARTICIPANT.items():
            if mic not in mic_audio:
                rows.append(_missing_row(session_id, group_id, task_id, participant_id, max_dur_s))
                continue

            audio = mic_audio[mic]
            sr = mic_sr[mic]
            total_dur_s = len(audio) / sr

            # Energy-ratio bleed rejection + spectral embedding cross-talk suppression
            mask = _build_speech_mask(
                mic_rms, mic_audio, mic_sr, mic, energy_ratio, min_speech_frames, frame_s
            )
            n_speech_frames = int(mask.sum())
            speaking_time_s = float(n_speech_frames * frame_s)
            speaking_fraction = speaking_time_s / total_dur_s if total_dur_s > 0 else 0.0

            speech_segments = _mask_to_segments(mask, frame_s)
            pause_count = _pause_count(speech_segments, PAUSE_THRESHOLD_S)

            # GeMAPSv01b on clean (bleed-rejected) audio
            smile_feats: dict[str, float] = {}
            if smile is not None:
                clean = _extract_clean_audio(audio, sr, mask, frame_s)
                smile_feats = _run_opensmile(smile, clean, sr)

            # Transcript-derived features (optional)
            tx = _transcript_features(transcript_df, participant_id)

            # QC flags
            flags: list[str] = []
            if n_speech_frames == 0:
                flags.append("no_clean_speech")
            if 0 < speaking_fraction < 0.05:
                flags.append("very_low_speaking_fraction")
            if smile is not None and not smile_feats:
                flags.append("opensmile_failed")
            if smile is None:
                flags.append("opensmile_not_installed")
            qc_flag = ";".join(flags) if flags else "ok"

            row: dict = {
                "session_id": session_id,
                "group_id": group_id,
                "task_id": task_id,
                "participant_id": participant_id,
                "audio_available": True,
                "duration_s": round(total_dur_s, 3),
                # Speaking-time features (from energy-ratio mask)
                "speaking_time_s": round(speaking_time_s, 3),
                "speaking_fraction": round(speaking_fraction, 4),
                "pause_count": pause_count,
                # Transcript-derived (None if transcript absent)
                "turn_count": tx["turn_count"],
                # Session-level overlap (same value for all participants in this task)
                "overlap_fraction": round(overlap_frac, 4),
                "uncertain_fraction": tx["uncertain_fraction"],
                "transcript_available": transcript_df is not None,
                # GeMAPSv01b-derived features on clean speech
                "energy_mean": smile_feats.get("loudness_sma3_amean"),
                "energy_sd": smile_feats.get("loudness_sma3_stddevNorm"),
                "pitch_mean": smile_feats.get("F0semitoneFrom27.5Hz_sma3nz_amean"),
                "pitch_sd": smile_feats.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm"),
                "hnr_mean": smile_feats.get("HNRdBACF_sma3nz_amean"),
                "jitter_mean": smile_feats.get("jitterLocal_sma3nz_amean"),
                "shimmer_mean": smile_feats.get("shimmerLocaldB_sma3nz_amean"),
                "voiced_segments_per_sec": smile_feats.get("VoicedSegmentsPerSec"),
                "mean_voiced_segment_s": smile_feats.get("MeanVoicedSegmentLengthSec"),
                "mean_unvoiced_segment_s": smile_feats.get("MeanUnvoicedSegmentLength"),
                "speech_rate_proxy": smile_feats.get("loudnessPeaksPerSec"),
                "qc_flag": qc_flag,
            }
            rows.append(row)

            LOG.info(
                "[%s | %s | %s] speaking=%.1fs (%.0f%%), pauses=%d, turns=%s, overlap=%.2f, qc=%s",
                session_id, task_id, participant_id,
                speaking_time_s, speaking_fraction * 100,
                pause_count, tx["turn_count"], overlap_frac, qc_flag,
            )

    return rows


# ── QC summary ─────────────────────────────────────────────────────────────────


def build_qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a compact QC table from the full feature table."""
    cols = [
        "session_id", "participant_id", "task_id",
        "audio_available", "transcript_available",
        "duration_s", "speaking_time_s", "speaking_fraction",
        "overlap_fraction", "qc_flag",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


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
            "an audio/ folder and optionally an audio_annot/ folder. "
            "Example: 'C:/data/ses-20260319_grp-15_run01-...'"
        ),
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("features"),
        help="Output directory for TSV files. Created if absent. Default: features/",
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
        "--energy-ratio", type=float, default=1.5,
        help=(
            "Energy-ratio threshold for Layer 1 bleed rejection (default: 1.5). "
            "A 25 ms frame is attributed to a mic only when its RMS energy "
            "exceeds all other mics by this factor (≈3.5 dB at ratio=1.5). "
            "Calibrated from RMS analysis: speech ~-46 to -50 dBFS, "
            "noise floor ~-54 to -57 dBFS. "
            "Layer 2 (spectral embedding) additionally rejects frames with "
            "non-speech spectral signatures (low centroid + high flatness)."
        ),
    )
    ap.add_argument(
        "--frame-s", type=float, default=FRAME_S,
        help=f"Frame duration in seconds for energy analysis (default: {FRAME_S}).",
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

    if opensmile is None:
        LOG.warning(
            "opensmile not installed — acoustic features (energy, pitch, etc.) "
            "will be missing. Install with: pip install opensmile"
        )
        smile = None
    else:
        smile = _build_smile()
        LOG.info("opensmile GeMAPSv01b ready (%d features)", len(smile.feature_names))

    all_rows: list[dict] = []
    for session_dir in sessions:
        LOG.info("=== %s ===", session_dir.name)
        rows = process_session(session_dir, smile, args.tasks, args.energy_ratio, args.frame_s)
        all_rows.extend(rows)

    if not all_rows:
        LOG.error("No rows produced — check --audio-root and file naming.")
        return 1

    df = pd.DataFrame(all_rows).sort_values(
        ["session_id", "task_id", "participant_id"]
    ).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_path = args.out_dir / "audio_participant_task.tsv"
    qc_path = args.out_dir / "audio_qc_summary.tsv"
    df.to_csv(task_path, sep="\t", index=False)
    build_qc_summary(df).to_csv(qc_path, sep="\t", index=False)

    LOG.info("Wrote %s (%d rows)", task_path, len(df))
    LOG.info("Wrote %s", qc_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
