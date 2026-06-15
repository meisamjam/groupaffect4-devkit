#!/usr/bin/env python3
"""Audio-based sync correction for Tobii video and microphone recordings.

Compares audio tracks from Tobii scene videos with microphone recordings
to detect and correct any sync drift using cross-correlation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile


def extract_audio_from_video(video_path: Path, audio_output: Path) -> bool:
    """Extract audio from video file using ffmpeg."""
    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-q:a", "9",
            "-n",  # Don't overwrite
            str(audio_output),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return audio_output.exists()
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False


def load_audio(audio_path: Path, sr: int = 16000) -> tuple[np.ndarray, int] | None:
    """Load audio file and resample to target sample rate."""
    try:
        # Try with scipy first (WAV files)
        if audio_path.suffix.lower() == ".wav":
            sample_rate, audio_data = wavfile.read(audio_path)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Stereo to mono
            if sample_rate != sr:
                # Resample
                num_samples = int(len(audio_data) * sr / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
            return audio_data.astype(np.float32), sr
        else:
            # Try ffmpeg for other formats
            cmd = [
                "ffmpeg",
                "-i", str(audio_path),
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", str(sr),
                "-ac", "1",
                "-",
            ]
            result = subprocess.run(cmd, capture_output=True, check=True)
            audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
            return audio_data, sr
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None


def compute_sync_offset(
    video_audio: np.ndarray,
    mic_audio: np.ndarray,
    sr: int = 16000,
    max_offset_s: float = 2.0,
) -> dict[str, Any]:
    """Compute sync offset between video and microphone audio using cross-correlation.

    Returns:
        Dict with:
          - offset_s: Time offset in seconds (positive = video lags behind mic)
          - confidence: Cross-correlation peak value (0-1)
          - video_duration: Duration of video audio
          - mic_duration: Duration of microphone audio
    """
    result = {
        "offset_s": 0.0,
        "confidence": 0.0,
        "video_duration": len(video_audio) / sr,
        "mic_duration": len(mic_audio) / sr,
        "error": None,
    }

    if len(video_audio) == 0 or len(mic_audio) == 0:
        result["error"] = "Empty audio data"
        return result

    # Normalize audio
    video_audio = (video_audio - np.mean(video_audio)) / (np.std(video_audio) + 1e-8)
    mic_audio = (mic_audio - np.mean(mic_audio)) / (np.std(mic_audio) + 1e-8)

    # Compute cross-correlation
    max_offset_samples = int(max_offset_s * sr)
    corr = signal.correlate(video_audio, mic_audio, mode="valid")

    if len(corr) == 0:
        result["error"] = "Cross-correlation failed"
        return result

    # Find peak
    peak_idx = np.argmax(np.abs(corr))
    peak_value = corr[peak_idx] / len(video_audio)  # Normalize

    # Convert to time offset
    offset_samples = peak_idx - min(len(video_audio), len(mic_audio))
    offset_s = offset_samples / sr

    result["offset_s"] = float(offset_s)
    result["confidence"] = float(np.abs(peak_value))

    return result


def process_session(
    session_dir: Path,
    tobii_root: Path,
    tobii_mapping: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Process one session for audio-based sync correction."""
    session_id = session_dir.name
    results = {
        "session": session_id,
        "status": "ok",
        "participant_sync_offsets": {},
        "errors": [],
    }

    # Extract group from session ID
    group_id = None
    for part in session_id.split("_"):
        if part.startswith("grp-"):
            group_id = part
            break

    if not group_id or not tobii_mapping:
        results["status"] = "skip"
        results["errors"].append("No group found or tobii_mapping missing")
        return results

    # Find group in mapping
    group_data = None
    for g in tobii_mapping.get("groups", []):
        if g.get("group_id") == group_id:
            group_data = g
            break

    if not group_data:
        results["status"] = "skip"
        results["errors"].append(f"Group {group_id} not found in mapping")
        return results

    # Process each participant
    # Try multiple possible audio directory names
    audio_dir = None
    for dirname in ["audio", "aud", "audio_annot"]:
        potential_dir = session_dir / dirname
        if potential_dir.exists():
            audio_dir = potential_dir
            break

    if not audio_dir:
        results["status"] = "skip"
        results["errors"].append("No audio directory found (tried: audio, aud, audio_annot)")
        return results

    mic_map = {"P1": "mic9", "P2": "mic10", "P3": "mic11", "P4": "mic12"}

    for recording in group_data.get("recordings", []):
        participant = recording.get("participant")
        folder_names = recording.get("folder_names", [])
        if isinstance(folder_names, str):
            folder_names = [folder_names]

        if not folder_names or not participant:
            continue

        mic_id = mic_map.get(participant)
        if not mic_id:
            continue

        # Find microphone audio file
        mic_files = list(audio_dir.glob(f"*{mic_id}*.wav")) + \
                   list(audio_dir.glob(f"*{mic_id}*.mp3"))

        if not mic_files:
            results["errors"].append(f"{participant}: No microphone audio found for {mic_id}")
            continue

        # Use first Tobii folder for this participant
        tobii_folder = folder_names[0]
        video_path = tobii_root / tobii_folder / "scenevideo.mp4"

        if not video_path.exists():
            results["errors"].append(f"{participant}: Video not found at {video_path}")
            continue

        if dry_run:
            results["participant_sync_offsets"][participant] = {
                "status": "dry_run",
                "video": str(video_path),
                "mic_audio": str(mic_files[0]),
            }
            continue

        # Extract audio from video
        temp_video_audio = session_dir / f".temp_video_audio_{participant}.wav"
        if not extract_audio_from_video(video_path, temp_video_audio):
            results["errors"].append(f"{participant}: Failed to extract video audio")
            continue

        # Load both audio files
        video_audio_data = load_audio(temp_video_audio)
        mic_audio_data = load_audio(mic_files[0])

        if video_audio_data is None or mic_audio_data is None:
            results["errors"].append(f"{participant}: Failed to load audio data")
            temp_video_audio.unlink(missing_ok=True)
            continue

        video_audio, sr = video_audio_data
        mic_audio, _ = mic_audio_data

        # Compute sync offset
        sync_result = compute_sync_offset(video_audio, mic_audio, sr)
        sync_result["tobii_folder"] = tobii_folder
        sync_result["mic_file"] = str(mic_files[0])

        results["participant_sync_offsets"][participant] = sync_result

        # Cleanup
        temp_video_audio.unlink(missing_ok=True)

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute audio-based sync corrections for Tobii videos"
    )
    p.add_argument(
        "--sessions-root",
        type=Path,
        default=Path("affectai-data-processing-seed/data/sub-01"),
        help="Root containing ses-* folders",
    )
    p.add_argument(
        "--tobii-root",
        type=Path,
        required=True,
        help="Root containing Tobii recording folders",
    )
    p.add_argument(
        "--mapping-file",
        type=Path,
        default=Path("configs/tobii_recordings_mapping.json"),
        help="Path to tobii_recordings_mapping.json",
    )
    p.add_argument(
        "--session-glob",
        default="ses-*",
        help="Session folder glob (default: ses-*)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without processing audio",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    sessions_root = args.sessions_root.resolve()
    tobii_root = args.tobii_root.resolve()
    mapping_file = args.mapping_file.resolve()

    if not sessions_root.exists():
        raise FileNotFoundError(f"Sessions root not found: {sessions_root}")
    if not tobii_root.exists():
        raise FileNotFoundError(f"Tobii root not found: {tobii_root}")
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    # Load tobii mapping
    try:
        tobii_mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to load mapping file: {e}")

    session_dirs = sorted(p for p in sessions_root.glob(args.session_glob) if p.is_dir())
    if not session_dirs:
        print("No session folders matched.")
        return 0

    results = []
    for session_dir in session_dirs:
        res = process_session(
            session_dir=session_dir,
            tobii_root=tobii_root,
            tobii_mapping=tobii_mapping,
            dry_run=args.dry_run,
        )
        results.append(res)
        print(json.dumps(res, ensure_ascii=True, indent=2))

    # Save summary
    out = sessions_root / "_audio_sync_correction_summary.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nWrote summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
