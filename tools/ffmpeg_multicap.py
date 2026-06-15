"""FFmpeg multi-device capture with shared LSL clock and progress anchors.

Supports macOS (AVFoundation) and Windows (DirectShow). Captures multiple video
devices, optional camera-audio muxed into MKV, and audio-only devices (including
optional WDM-KS direct input on Windows for compatible hardware).

The tool publishes a shared LSL clock stream plus per-device progress streams
(`ffmpeg_progress_<label>`), writes per-run progress TSV logs to
`<session>/sourcedata/sync/*_ffmpeg_progress.tsv`, and records capture lifecycle
events to `<session>/video/ffmpeg_multicap_events.jsonl`.

Common usage:
  python tools/ffmpeg_multicap.py --list-devices
  python tools/ffmpeg_multicap.py --config configs/ffmpeg_multicap.json
  python tools/ffmpeg_multicap.py --config configs/ffmpeg_multicap.json --frame-log --record-lsl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Force UTF-8 output on Windows so non-ASCII characters never crash the process
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import scipy.io.wavfile as wav
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    sd = None  # type: ignore
    wav = None  # type: ignore

from pylsl import (
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    cf_double64,
    cf_string,
    local_clock,
    resolve_streams,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_event_timestamps() -> dict[str, float | int | str]:
    """Return standardized, merge-friendly timestamps for event logs."""
    unix_time_ns = time.time_ns()
    unix_time_s = unix_time_ns / 1_000_000_000
    return {
        "unix_time_s": unix_time_s,
        "unix_time_ns": unix_time_ns,
        "lsl_time": local_clock(),
        "monotonic_ns": time.monotonic_ns(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_showinfo_line(line: str) -> tuple[int, float] | None:
    """Parse a single showinfo line and return (frame, pts_time)."""
    showinfo_re = re.compile(r"n:\s*(\d+).*?pts_time:\s*([0-9.-]+)")
    match = showinfo_re.search(line)
    if not match:
        return None
    try:
        frame = int(match.group(1))
        pts_time = float(match.group(2))
    except ValueError:
        return None
    return frame, pts_time


def parse_ffmpeg_progress_lines(lines: list[str]) -> list[dict[str, str]]:
    """Parse ffmpeg -progress output into records (one per progress block)."""
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key] = value
        if key == "progress":
            records.append(current)
            current = {}
    return records


def parse_ffmpeg_progress_line(
    line: str, current: dict[str, str]
) -> tuple[dict[str, str] | None, dict[str, str]]:
    """Incrementally parse ffmpeg -progress output.

    Returns (record, current). record is only set when a progress block ends.
    """
    raw = line.strip()
    if not raw or "=" not in raw:
        return None, current
    key, value = raw.split("=", 1)
    current[key] = value
    if key == "progress":
        record = current
        return record, {}
    return None, current


def parse_media_time_us(progress: dict[str, str]) -> int | None:
    """Extract media time (microseconds) from a progress record."""
    if "out_time_us" in progress:
        try:
            return int(progress["out_time_us"])
        except ValueError:
            return None
    if "out_time_ms" in progress:
        try:
            return int(progress["out_time_ms"]) * 1000
        except ValueError:
            return None
    if "out_time" in progress:
        try:
            h, m, s = progress["out_time"].split(":", 2)
            total_s = (int(h) * 3600) + (int(m) * 60) + float(s)
            return int(total_s * 1_000_000)
        except (ValueError, TypeError):
            return None
    return None


def disable_jabra_intelligent_zoom(
    subnets: list[str] | None = None,
    timeout: float = 2.0,
) -> dict[str, bool]:
    """Try to disable intelligent zoom on Jabra PanaCast devices via REST API.

    Scans common network subnets for Jabra PanaCast devices and attempts
    to disable auto-zoom, auto-frame, and intelligent framing features
    to maintain consistent video framing.

    Args:
        subnets: List of subnet patterns to scan (e.g., ["192.168.1", "10.0.0"]).
                 If None, uses common defaults.
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict mapping IP addresses to success status.
    """
    import socket
    import urllib.error
    import urllib.request

    if subnets is None:
        # Common lab/office subnets
        subnets = ["192.168.1", "192.168.0", "10.0.0", "10.0.1", "172.16.0"]

    results: dict[str, bool] = {}
    found_devices: list[str] = []

    # Quick scan for Jabra devices on common subnets
    logger.info("Scanning for Jabra PanaCast devices to disable intelligent zoom...")
    print("\n[>>] Scanning for Jabra PanaCast devices...")

    for subnet in subnets:
        # Check a range of IPs quickly
        for i in [1, 50, 100, 150, 200, 254]:  # Quick sample
            ip = f"{subnet}.{i}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.3)
                result = sock.connect_ex((ip, 8080))
                sock.close()
                if result == 0:
                    # Port 8080 open, check if it's a Jabra device
                    try:
                        url = f"http://{ip}:8080/api/rest/v1/health"
                        req = urllib.request.Request(url, method="GET")
                        with urllib.request.urlopen(req, timeout=timeout) as resp:
                            if resp.status == 200:
                                found_devices.append(ip)
                                logger.debug(f"Found Jabra device at {ip}")
                    except Exception:
                        pass
            except Exception:
                pass

    if not found_devices:
        print("   [!]  No Jabra devices found on network")
        print("   [i] Tip: Disable intelligent zoom manually in Jabra Direct app or via web UI")
        print("          See docs/panacast_usb_disable_zoom.md for instructions")
        logger.warning("No Jabra devices found for zoom control; may need manual configuration")
        return results

    # Try to disable zoom on found devices
    for ip in found_devices:
        success = False

        # Try multiple API endpoint variants
        endpoints_payloads = [
            (f"http://{ip}:8080/api/rest/v1/camera/zoom/disable", {"enabled": False}),
            (f"http://{ip}:8080/api/rest/v1/camera/auto_zoom", {"auto_zoom": False, "auto_frame": False}),
            (f"http://{ip}:8080/api/rest/v1/video/settings",
             {"auto_zoom": False, "auto_focus": False, "intelligent_framing": False}),
            (f"http://{ip}:8080/api/rest/v1/settings/video",
             {"intelligentZoom": False, "autoFrame": False}),
        ]

        for endpoint, payload in endpoints_payloads:
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    endpoint,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status in [200, 201, 204]:
                        success = True
                        break
            except Exception:
                continue

        results[ip] = success
        if success:
            print(f"   [ok] Disabled intelligent zoom on {ip}")
            logger.info(f"Disabled intelligent zoom on Jabra device at {ip}")
        else:
            print(f"   [!]  Could not disable zoom on {ip} (may need Jabra Direct app)")
            logger.warning(f"Failed to disable zoom on {ip}; may need manual configuration")

    return results


def detect_ffmpeg_backend() -> str | None:
    """Choose ffmpeg input backend based on platform."""
    if sys.platform == "darwin":
        return "avfoundation"
    if sys.platform.startswith("win") or sys.platform == "cygwin":
        return "dshow"
    # Extend here if we add v4l2/alsa support for Linux.
    return None


@dataclass
class DeviceConfig:
    """Configuration for one device (video+audio or audio-only)."""
    label: str
    video_index: int | None = None
    audio_index: int | None = None
    video_name: str | None = None
    audio_name: str | None = None
    video_alt_name: str | None = None
    audio_alt_name: str | None = None
    width: int = 1920
    height: int = 1080
    fps: int = 30
    video_bitrate: int = 5000
    audio_bitrate: int = 24
    audio_only: bool = False
    audio_channels: int = 1
    audio_pan: str | None = None  # e.g. 'mono|c0=c0' to extract left channel only
    format: str = "mkv"
    subdir: str = "video"
    use_video_device_timestamps: bool = False
    force_wallclock_timestamps: bool = False
    audio_sample_rate: int = 48000
    # For audio-only: "dshow" (default) or "wdmks" (direct kernel streaming)
    audio_backend: str | None = None
    # For wdmks: sounddevice device index and channel (0=left, 1=right)
    wdmks_device_id: int | None = None
    wdmks_channel: int = 0  # 0=left, 1=right for stereo WDM-KS devices
    # Mux audio into video file (True) or capture video-only (False)
    mux_audio: bool = True
    # Input video codec to request from camera (e.g. "mjpeg"). Forces the
    # DirectShow capture pin to use this codec instead of raw YUY2/NV12.
    # Set to None to let the camera choose its default.
    input_video_codec: str | None = None
    # Pixel format to request from camera (e.g. "yuyv422"). Useful when camera
    # defaults to lower-quality formats like nv12. Set to None for camera default.
    pixel_format: str | None = None
    # ── Camera controls (exposure / white-balance / gain) ──
    # If True, ffmpeg opens the native DirectShow property dialog before capture
    # so the operator can manually lock exposure, WB, etc.
    show_camera_dialog: bool = False
    # Arbitrary extra DirectShow video-input options appended before -i.
    # Passed verbatim as individual ffmpeg args, e.g. ["-rtbufsize","512M"].
    dshow_extra_args: list[str] | None = None
    # Pre-capture script (path) to run for camera control (e.g. UVC exposure
    # lock tool).  Receives the video device name / alt_name as the first arg.
    camera_setup_script: str | None = None
    # Extra arguments passed to camera_setup_script after the device identifier.
    # e.g. ["--exposure", "-6", "--wb", "5000", "--gain", "0"]
    camera_setup_args: list[str] | None = None
    # Camera orientation metadata shared across tools. ffmpeg_multicap does not
    # transform frames with this flag; OpenCV-based tools can consume it.
    rotate_180: bool = False

    def output_path(self, session_dir: Path) -> Path:
        """Get primary output path (for audio-only devices or legacy use)."""
        root = Path(session_dir)
        target_dir = root / self.subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = self.format or ("wav" if self.audio_only else "mkv")
        return target_dir / f"{self.label}.{suffix}"

    def video_output_path(self, session_dir: Path) -> Path:
        """Get video output path for video+audio devices."""
        root = Path(session_dir)
        target_dir = root / self.subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{self.label}_video.mkv"

    def audio_output_path(self, session_dir: Path) -> Path:
        """Get audio output path for video+audio devices."""
        root = Path(session_dir)
        # Store audio separately in audio/ subdirectory
        target_dir = root / "audio"
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{self.label}_audio.wav"


class EventLogger:
    """Write capture events to JSONL file."""

    def __init__(self, session_dir: Path):
        self.path = session_dir / "video" / "ffmpeg_multicap_events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, device_id: str, event_type: str, details: dict) -> None:
        entry = {
            "device_id": device_id,
            "event_type": event_type,
            **build_event_timestamps(),
            **details,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def find_wdmks_devices() -> dict[str, dict]:
    """Find WDM-KS input devices (RME Fireface, etc.) for direct audio capture.

    Returns dict mapping channel-pair name to device info:
        {"9+10": {"device_id": 57, "name": "Analog (9+10) (RME Fireface 802)"}, ...}
    """
    if not HAS_SOUNDDEVICE:
        return {}

    wdm_ks = {}
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        name = d['name']
        # Check for WDM-KS host API
        hostapi = sd.query_hostapis(d['hostapi'])['name']
        if 'WDM-KS' not in hostapi:
            continue
        if d['max_input_channels'] == 0:
            continue

        # Store device info with channel pair key if it looks like RME Fireface
        device_info = {
            "device_id": i,
            "name": name,
            "max_input_channels": d['max_input_channels'],
            "default_samplerate": d.get('default_samplerate', 48000),
        }

        # Extract channel pair from name (e.g., "9+10", "11+12", "1+2")
        match = re.search(r'\((\d+\+\d+)\)', name)
        if match:
            pair = match.group(1)
            wdm_ks[pair] = device_info
        else:
            # Store by device index for non-standard names
            wdm_ks[f"device_{i}"] = device_info

    return wdm_ks


class WDMKSAudioCapture:
    """Capture audio from a single WDM-KS device channel using sounddevice.

    Bypasses TotalMix routing by accessing hardware inputs directly via WDM-KS
    kernel streaming. Each instance captures one mono channel from a stereo pair.
    """

    def __init__(
        self,
        device: DeviceConfig,
        output_path: Path,
        event_logger: EventLogger,
        sample_rate: int = 48000,
    ):
        if not HAS_SOUNDDEVICE:
            raise ImportError("sounddevice and scipy required for WDM-KS capture")

        self.device = device
        self.output_path = output_path
        self.event_logger = event_logger
        self.sample_rate = sample_rate
        self.running = False
        self._stream: Any = None  # sd.InputStream when sounddevice is available
        self._buffer: list[np.ndarray] = []
        self._sample_count: int = 0
        self._start_time: float = 0
        self._start_lsl_time: float = 0
        self._lock = threading.Lock()

    def start(self, target_time: float | None = None) -> bool:
        """Start audio capture.

        Args:
            target_time: Optional target unix timestamp to start at (for sync).
        """
        if self.running:
            return False

        device_id = self.device.wdmks_device_id
        if device_id is None:
            logger.error(f"WDM-KS device ID not set for {self.device.label}")
            return False

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Wait until target_time if specified (for parallel sync)
            if target_time is not None:
                wait_time = target_time - time.time()
                if wait_time > 0:
                    time.sleep(wait_time)

            self._buffer = []
            self._sample_count = 0

            # Create input stream (always capture stereo, extract channel later)
            self._stream = sd.InputStream(
                device=device_id,
                channels=2,
                samplerate=self.sample_rate,
                dtype='int16',
                callback=self._audio_callback,
            )

            self._start_time = time.time()
            self._start_lsl_time = local_clock()
            self._stream.start()
            self.running = True

            self.event_logger.log_event(
                self.device.label,
                "capture_started",
                {
                    "output": str(self.output_path),
                    "backend": "wdmks",
                    "device_id": device_id,
                    "channel": self.device.wdmks_channel,
                    "sample_rate": self.sample_rate,
                },
            )

            logger.info(
                f"WDM-KS capture started: {self.device.label} "
                f"(device {device_id}, ch {self.device.wdmks_channel})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to start WDM-KS capture for {self.device.label}: {e}")
            self.event_logger.log_event(
                self.device.label,
                "capture_error",
                {"error": str(e), "stage": "start", "backend": "wdmks"},
            )
            return False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice input stream."""
        if status:
            logger.warning(f"WDM-KS audio status ({self.device.label}): {status}")

        # Extract the appropriate channel (0=left, 1=right)
        channel = self.device.wdmks_channel
        with self._lock:
            self._buffer.append(indata[:, channel].copy())
            self._sample_count += frames

    def stop(self) -> bool:
        """Stop capture and save WAV file."""
        if not self.running:
            return False

        self.running = False
        stop_lsl_time = local_clock()

        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception as e:
            logger.warning(f"Error stopping WDM-KS stream ({self.device.label}): {e}")

        # Save recording
        with self._lock:
            if self._buffer:
                audio_data = np.concatenate(self._buffer)
                wav.write(str(self.output_path), self.sample_rate, audio_data)
                duration_s = len(audio_data) / self.sample_rate
                logger.info(
                    f"WDM-KS capture saved: {self.device.label} -> {self.output_path} "
                    f"({duration_s:.1f}s, {self._sample_count} samples)"
                )

                self.event_logger.log_event(
                    self.device.label,
                    "capture_completed",
                    {
                        "output": str(self.output_path),
                        "backend": "wdmks",
                        "duration_s": duration_s,
                        "sample_count": self._sample_count,
                        "start_lsl_time": self._start_lsl_time,
                        "stop_lsl_time": stop_lsl_time,
                    },
                )
            else:
                logger.warning(f"No audio data captured for {self.device.label}")

            self._buffer.clear()

        return True


class FFmpegCapture:
    """Manage one ffmpeg capture subprocess."""

    def __init__(
        self,
        device: DeviceConfig,
        output_path: Path,
        logger: EventLogger,
        backend: str,
        device_catalog: dict[str, list[dict]] | None = None,
        enable_frame_log: bool = False,
        emit_progress: bool = True,
        progress_label: str | None = None,
        essential: bool = True,
        max_duration: float | None = None,
    ):
        self.device = device
        self.output_path = output_path
        self.event_logger = logger
        self.enable_frame_log = enable_frame_log
        self.backend = backend
        self.device_catalog = device_catalog or {}
        self.emit_progress = emit_progress
        self.progress_label = progress_label or device.label
        self.essential = essential
        self.max_duration = max_duration
        self.process: subprocess.Popen | None = None
        self.running = False
        self.lsl_outlet: StreamOutlet | None = None
        self.lsl_thread: threading.Thread | None = None
        self.progress_thread: threading.Thread | None = None
        # For video+audio devices, store separate paths
        self.video_path: Path | None = None
        self.audio_path: Path | None = None
        # Frame log file handle/path (video only)
        self._frame_log_fp = None
        self._frame_log_path: Path | None = None
        self._progress_log_fp = None
        self._progress_log_path: Path | None = None
        self._last_media_time_us: int | None = None
        self._last_media_time_lsl: float | None = None
        self._progress_last_emit: float | None = None
        self._completion_logged = False
        self._stopped = False
        if self.device.audio_only:
            self.audio_path = self.output_path
        else:
            self.video_path = self.output_path

    def run_camera_setup(self) -> bool:
        """Run pre-capture camera setup script (e.g. exposure / WB lock).

        Should be called before ``start()`` — ideally sequentially for all
        cameras so that each lock_exposure process gets exclusive DirectShow
        access without contention.

        Returns True if setup succeeded or was not needed.
        """
        if not self.device.camera_setup_script or self.device.audio_only:
            return True

        video_id = (
            self.device.video_alt_name
            or self.device.video_name
            or str(self.device.video_index or "")
        )
        script = self.device.camera_setup_script
        # On Windows, .py scripts are not directly executable – prepend
        # the current Python interpreter so the subprocess works reliably.
        if script.endswith(".py"):
            setup_cmd = [sys.executable, script, video_id]
        else:
            setup_cmd = [script, video_id]
        if self.device.camera_setup_args:
            setup_cmd.extend(self.device.camera_setup_args)
        logger.info(f"Running camera setup for {self.device.label}: {setup_cmd}")
        try:
            result = subprocess.run(
                setup_cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    f"Camera setup script returned {result.returncode} "
                    f"for {self.device.label}: {result.stderr.strip()}"
                )
                return False
            else:
                logger.info(f"Camera setup OK for {self.device.label}")
                return True
        except Exception as e:
            logger.warning(f"Camera setup script failed for {self.device.label}: {e}")
            return False

    def start(self, target_time: float | None = None) -> bool:
        """Start ffmpeg capture process.

        Args:
            target_time: Optional target unix timestamp (time.time()) to start at.
                        If provided, will sleep until this time before launching process.
        """
        if self.running:
            logger.warning(f"Capture already running for {self.device.label}")
            return False
        if self.process and self.process.poll() is None:
            logger.warning(f"Capture process already exists for {self.device.label}")
            return False

        cmd = self._build_command()

        logger.info(f"Starting {self.device.label}: {' '.join(cmd)}")

        try:
            self._stopped = False
            self._completion_logged = False
            # Synchronization barrier: wait until target_time if specified
            if target_time is not None:
                wait_time = target_time - time.time()
                if wait_time > 0:
                    time.sleep(wait_time)

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.running = True
            # Prepare frame log file if enabled and this is a video capture
            if self.enable_frame_log and not self.device.audio_only:
                session_dir = self.output_path.parent.parent
                frame_dir = session_dir / "frame_logs"
                frame_dir.mkdir(parents=True, exist_ok=True)
                self._frame_log_path = frame_dir / f"{self.device.label}_frames.jsonl"
                self._frame_log_fp = self._frame_log_path.open("w", encoding="utf-8")

            # Prepare progress log file for ffmpeg -progress output
            session_dir = self.output_path.parent.parent
            progress_dir = session_dir / "sourcedata" / "sync"
            progress_dir.mkdir(parents=True, exist_ok=True)
            self._progress_log_path = progress_dir / f"{self.progress_label}_ffmpeg_progress.tsv"
            # Start a fresh per-run progress log to prevent mixed-run sync anchors.
            self._progress_log_fp = self._progress_log_path.open("w", encoding="utf-8")
            self._progress_log_fp.write(
                "host_time_sec\tout_time_sec\tframe\tdrop_frames\tdup_frames\n"
            )
            self._progress_log_fp.flush()

            # Start progress monitor thread to capture -progress output
            self.progress_thread = threading.Thread(target=self._monitor_output, daemon=True)
            self.progress_thread.start()
            if self.emit_progress:
                self._start_progress_stream()

            event_data = {
                "output": str(self.output_path),
                "audio_only": self.device.audio_only,
                "format": self.device.format,
            }
            if self.video_path:
                event_data["video_output"] = str(self.video_path)
            if self.audio_path:
                event_data["audio_output"] = str(self.audio_path)
            self.event_logger.log_event(
                self.device.label,
                "capture_started",
                event_data,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start capture for {self.device.label}: {e}")
            self.running = False
            if self._frame_log_fp:
                try:
                    self._frame_log_fp.close()
                except Exception:
                    pass
                finally:
                    self._frame_log_fp = None
            if self._progress_log_fp:
                try:
                    self._progress_log_fp.close()
                except Exception:
                    pass
                finally:
                    self._progress_log_fp = None
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                except Exception:
                    try:
                        self.process.kill()
                        self.process.wait(timeout=3)
                    except Exception:
                        pass
            self.process = None
            self.event_logger.log_event(
                self.device.label,
                "capture_error",
                {"error": str(e), "stage": "start"},
            )
            return False

    def _build_command(self) -> list[str]:
        """Build ffmpeg command for this device."""
        if self.backend == "avfoundation":
            return self._build_avfoundation_command()
        if self.backend == "dshow":
            return self._build_dshow_command()
        raise ValueError(f"Unsupported ffmpeg backend: {self.backend}")

    def _build_avfoundation_command(self) -> list[str]:
        """Build avfoundation command (macOS)."""
        if self.device.audio_only:
            cmd = [
                "ffmpeg",
                "-f",
                "avfoundation",
                *(
                    ["-use_wallclock_as_timestamps", "1"]
                    if self.device.force_wallclock_timestamps or self.enable_frame_log
                    else []
                ),
                "-i",
                f":{self.device.audio_index}",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(self.device.audio_sample_rate),
                "-ac",
                str(self.device.audio_channels),
            ]
            if self.max_duration is not None and self.max_duration > 0:
                cmd.extend(["-t", str(int(self.max_duration))])
            if self.device.audio_pan:
                cmd.extend(["-af", f"pan={self.device.audio_pan}"])
            cmd.extend(["-stats_period", "0.1", "-progress", "pipe:1", "-nostats"])
            cmd.extend(["-y", str(self.output_path)])
            return cmd

        if self.device.video_index is not None:
            input_spec = str(self.device.video_index)
        else:
            input_spec = ""
        cmd = [
            "ffmpeg",
            "-f",
            "avfoundation",
            "-framerate",
            str(self.device.fps),
            *(
                ["-use_wallclock_as_timestamps", "1"]
                if self.device.force_wallclock_timestamps or self.enable_frame_log
                else []
            ),
            "-i",
            input_spec,
            "-map",
            "0:v",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "28",
            "-s",
            f"{self.device.width}x{self.device.height}",
        ]
        if self.max_duration is not None and self.max_duration > 0:
            cmd.extend(["-t", str(int(self.max_duration))])
        if self.enable_frame_log and not self.device.audio_only:
            cmd.extend(["-vf", "showinfo"])
        cmd.extend(["-stats_period", "0.1", "-progress", "pipe:1", "-nostats"])
        cmd.extend(["-y", str(self.output_path)])
        return cmd

    def _build_dshow_command(self) -> list[str]:
        """Build DirectShow command (Windows)."""
        if self.device.audio_only:
            # Prefer alt_name (unique device path) to avoid ambiguity with duplicate display names
            audio_name = (
                self.device.audio_alt_name
                or self.device.audio_name
                or self._resolve_device_name("audio", self.device.audio_index)
            )
            if not audio_name:
                raise ValueError("No audio device resolved for DirectShow input")

            input_spec = f"audio={self._quote_device_name(audio_name)}"
            cmd = [
                "ffmpeg",
                "-f",
                "dshow",
                "-rtbufsize",
                "256M",
                *(
                    ["-use_wallclock_as_timestamps", "1"]
                    if self.device.force_wallclock_timestamps or self.enable_frame_log
                    else []
                ),
                "-i",
                input_spec,
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(self.device.audio_sample_rate),
                "-ac",
                str(self.device.audio_channels),
            ]
            if self.max_duration is not None and self.max_duration > 0:
                cmd.extend(["-t", str(int(self.max_duration))])
            if self.device.audio_pan:
                cmd.extend(["-af", f"pan={self.device.audio_pan}"])
            cmd.extend(["-stats_period", "0.1", "-progress", "pipe:1", "-nostats"])
            cmd.extend(["-y", str(self.output_path)])
            return cmd

        # Prefer alt_name (unique device path) to avoid ambiguity with duplicate display names
        video_name = (
            self.device.video_alt_name
            or self.device.video_name
            or self._resolve_device_name("video", self.device.video_index)
        )

        if not video_name:
            raise ValueError("No DirectShow video device resolved")

        # Resolve audio device if mux_audio is enabled
        audio_name = None
        if self.device.mux_audio and self.device.audio_index is not None:
            audio_name = (
                self.device.audio_alt_name
                or self.device.audio_name
                or self._resolve_device_name("audio", self.device.audio_index)
            )

        # Build input specification (video only, or video:audio combined)
        if audio_name:
            input_spec = f"video={self._quote_device_name(video_name)}:audio={self._quote_device_name(audio_name)}"
        else:
            input_spec = f"video={self._quote_device_name(video_name)}"

        cmd = [
            "ffmpeg",
            "-f",
            "dshow",
            "-rtbufsize",
            "256M",
            "-video_size",
            f"{self.device.width}x{self.device.height}",
            "-framerate",
            str(self.device.fps),
            *(
                ["-vcodec", self.device.input_video_codec]
                if self.device.input_video_codec else []
            ),
            *(
                ["-pixel_format", self.device.pixel_format]
                if self.device.pixel_format else []
            ),
            *(
                ["-show_video_device_dialog", "true"]
                if self.device.show_camera_dialog else []
            ),
            *(self.device.dshow_extra_args or []),
            *(
                ["-use_wallclock_as_timestamps", "1"]
                if self.device.force_wallclock_timestamps or self.enable_frame_log
                else [
                    "-use_video_device_timestamps", "0"
                ] if not self.device.use_video_device_timestamps else []
            ),
            "-i",
            input_spec,
        ]

        # Video output: stream-copy native camera output (usually MJPEG).
        # Avoids libx264 (GPL) which is not available in all ffmpeg builds.
        # showinfo filter requires a decode/filter pipeline, so use mjpeg re-encode
        # when frame logging is enabled; otherwise stream-copy for efficiency.
        # Note: quality 2 is higher than quality 3 (lower number = better quality).
        if self.enable_frame_log and not self.device.audio_only:
            video_codec_args = ["-c:v", "mjpeg", "-q:v", "2"]
        else:
            video_codec_args = ["-c:v", "copy"]

        cmd.extend([
            "-map",
            "0:v",
            *video_codec_args,
        ])

        # Add audio stream if muxing
        if audio_name:
            cmd.extend([
                "-map",
                "0:a",
                "-c:a",
                "aac",
                "-b:a",
                f"{self.device.audio_bitrate}k",
                # Resample audio to compensate for wallclock timestamp jitter on
                # the video stream; prevents A/V drift in the muxed MKV file.
                "-af", "aresample=async=1",
            ])

        # max_muxing_queue_size prevents "Too many packets buffered" errors
        # when A/V streams have slightly different arrival rates
        cmd.extend(["-max_muxing_queue_size", "1024"])
        if self.max_duration is not None and self.max_duration > 0:
            cmd.extend(["-t", str(int(self.max_duration))])
        if self.enable_frame_log and not self.device.audio_only:
            cmd.extend(["-vf", "showinfo"])
        cmd.extend(["-stats_period", "0.1", "-progress", "pipe:1", "-nostats"])
        cmd.extend(["-y", str(self.output_path)])
        return cmd

    def _resolve_device_name(self, kind: str, index: int | None) -> str | None:
        if index is None:
            return None
        devices = self.device_catalog.get(kind, [])
        for device in devices:
            if device.get("index") == index:
                return device.get("name")
        return None

    @staticmethod
    def _quote_device_name(name: str | None) -> str:
        """Return device name without extra quotes (subprocess handles argument boundaries)."""
        if not name:
            return ""
        # Don't add quotes - subprocess.Popen with list args handles boundaries automatically
        return name

    def _start_progress_stream(self) -> None:
        """Start ffmpeg progress LSL stream for this device."""
        info = StreamInfo(
            f"ffmpeg_progress_{self.progress_label}",
            "ffmpeg_progress",
            5,  # [out_time_sec, media_time_us, frame, drop_frames, dup_frames]
            10.0,
            cf_double64,
            f"ffmpeg_progress_{self.progress_label}_uuid",
        )
        self.lsl_outlet = StreamOutlet(info)
        logger.info("LSL progress stream '%s' started at 10 Hz (stats_period=0.1s)", self.progress_label)

    def _monitor_output(self) -> None:
        """Parse ffmpeg combined output for progress records and frame logs."""
        if not self.process:
            return

        output_lines: list[str] = []
        current: dict[str, str] = {}
        try:
            for line in self.process.stdout:
                s = line.strip()
                output_lines.append(s)
                # Parse per-frame info when enabled
                if self.enable_frame_log and not self.device.audio_only and self._frame_log_fp:
                    parsed = parse_showinfo_line(s)
                    if parsed:
                        frame, pts_time = parsed
                        record = {
                            "frame": frame,
                            "pts_time": pts_time,
                            "lsl_time": local_clock(),
                            "unix_time_s": time.time(),
                            "unix_time_ns": time.time_ns(),
                            "monotonic_ns": time.monotonic_ns(),
                            "unix_time": time.time(),
                        }
                        try:
                            self._frame_log_fp.write(json.dumps(record) + "\n")
                            self._frame_log_fp.flush()
                        except Exception:
                            pass
                record, current = parse_ffmpeg_progress_line(s, current)
                if record:
                    self._handle_progress_record(record)
                # Look for fatal errors
                if "error" in s.lower() or "failed" in s.lower():
                    logger.error(f"{self.device.label} ffmpeg error: {s}")
        except Exception as e:
            logger.warning(f"Output monitor error for {self.device.label}: {e}")
        finally:
            if self.process and self.process.poll() is not None:
                was_running = self.running
                self.running = False
                returncode = self.process.returncode
                if returncode != 0 and not self._stopped:
                    logger.error(
                        f"{self.device.label} ffmpeg exited with code {returncode}. "
                        f"Last output lines:\n" + "\n".join(output_lines[-20:])
                    )
                    self.event_logger.log_event(
                        self.device.label,
                        "capture_error",
                        {
                            "error": f"ffmpeg exited with code {returncode}",
                            "stderr": output_lines[-20:],
                        },
                    )
                elif was_running:
                    logger.info("%s ffmpeg exited normally", self.device.label)

    def _handle_progress_record(self, record: dict[str, str]) -> None:
        media_time_us = parse_media_time_us(record)
        if media_time_us is None:
            return
        self._last_media_time_us = media_time_us
        self._last_media_time_lsl = local_clock()
        out_time_sec = media_time_us / 1_000_000
        try:
            frame = float(record.get("frame", "0"))
        except ValueError:
            frame = 0.0
        try:
            drop_frames = float(record.get("drop_frames", "0"))
        except ValueError:
            drop_frames = 0.0
        try:
            dup_frames = float(record.get("dup_frames", "0"))
        except ValueError:
            dup_frames = 0.0

        if self._progress_log_fp:
            try:
                host_time = local_clock()
                self._progress_log_fp.write(
                    f"{host_time:.6f}\t{out_time_sec:.6f}\t{frame:.0f}\t{drop_frames:.0f}\t{dup_frames:.0f}\n"
                )
                self._progress_log_fp.flush()
            except Exception:
                pass

        if not self.lsl_outlet:
            return
        try:
            # 5-channel: [out_time_sec, media_time_us, frame, drop_frames, dup_frames]
            # media_time_us is the raw microsecond value so readers can compute
            # start_anchor = stream_time - media_time_us / 1e6 without conversion error.
            self.lsl_outlet.push_sample(
                [out_time_sec, float(media_time_us), frame, drop_frames, dup_frames]
            )
            self._progress_last_emit = time.monotonic()
        except Exception as e:
            logger.warning(f"Progress LSL error for {self.device.label}: {e}")

    def stop(self) -> bool:
        """Stop ffmpeg process gracefully."""
        if not self.process or self._stopped:
            return False

        try:
            logger.info(f"Stopping {self.device.label}...")
            self._stopped = True
            self.running = False  # Signal LSL thread to stop

            if self.progress_thread:
                self.progress_thread.join(timeout=2)
            # Close frame log file if open
            if self._frame_log_fp:
                try:
                    self._frame_log_fp.flush()
                    self._frame_log_fp.close()
                except Exception:
                    pass
                finally:
                    self._frame_log_fp = None

            # Close progress log file if open
            if self._progress_log_fp:
                try:
                    self._progress_log_fp.flush()
                    self._progress_log_fp.close()
                except Exception:
                    pass
                finally:
                    self._progress_log_fp = None

            returncode = self.process.poll()
            if returncode is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force-killing {self.device.label}")
                    self.process.kill()
                    self.process.wait()
                returncode = self.process.returncode

            if returncode == 0 and not self._completion_logged:
                self.event_logger.log_event(
                    self.device.label,
                    "capture_completed",
                    {"output": str(self.output_path)},
                )
                self._completion_logged = True
            return True
        except Exception as e:
            logger.error(f"Error stopping {self.device.label}: {e}")
            return False
        finally:
            self.process = None


class LSLClockPublisher:
    """Publish system clock to LSL for physiological data alignment."""

    def __init__(self, stream_name: str, rate_hz: float):
        self.stream_name = stream_name
        self.rate_hz = rate_hz
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Start publishing clock."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"LSL clock '{self.stream_name}' started at {self.rate_hz} Hz")

    def stop(self) -> None:
        """Stop publishing clock."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _run(self) -> None:
        """Publish clock samples."""
        info = StreamInfo(
            self.stream_name,
            "clock",
            1,
            self.rate_hz,
            cf_double64,
            f"{self.stream_name}-uuid",
        )
        outlet = StreamOutlet(info)
        period = 1.0 / self.rate_hz

        while self.running:
            try:
                now_sec = local_clock()
                outlet.push_sample([now_sec])
                time.sleep(period)
            except Exception as e:
                logger.warning(f"LSL clock error: {e}")


class MarkerOutlet:
    """Publish marker labels to LSL for session annotation."""

    def __init__(self, stream_name: str = "affectai_markers"):
        self.stream_name = stream_name
        self._outlet: StreamOutlet | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        info = StreamInfo(
            self.stream_name,
            "Markers",
            1,
            0.0,
            cf_string,
            f"{self.stream_name}-uuid",
        )
        self._outlet = StreamOutlet(info)
        self._running = True
        self._thread = threading.Thread(target=self._listen_for_keys, daemon=True)
        self._thread.start()
        logger.info("LSL marker outlet '%s' started (press 's' or 'e')", self.stream_name)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def emit(self, label: str) -> None:
        if not self._outlet:
            return
        try:
            self._outlet.push_sample([label])
        except Exception as e:
            logger.warning("Marker emit failed: %s", e)

    def _listen_for_keys(self) -> None:
        while self._running:
            try:
                char = sys.stdin.read(1)
            except Exception:
                break
            if not char:
                continue
            if char.lower() == "s":
                self.emit("sync_anchor_start")
            elif char.lower() == "e":
                self.emit("sync_anchor_end")


class _LSLStreamWorker:
    """Background writer for one LSL inlet."""

    def __init__(self, stream, output_dir: Path):
        self.stream = stream
        # Disable liblsl auto-recovery so shutdown does not spam reconnect errors.
        try:
            self.inlet = StreamInlet(stream, max_buflen=120, recover=False)  # ~2 minutes buffer
        except TypeError:
            # Backward compatibility with older pylsl signatures.
            self.inlet = StreamInlet(stream, max_buflen=120)
        safe_name = stream.name().replace("/", "_").replace("\\", "_")
        self.path = output_dir / f"{safe_name}.jsonl"
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            # Close inlet first so pull_sample unblocks quickly during shutdown.
            self.inlet.close_stream()
        except Exception:
            pass
        self.thread.join(timeout=5)

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            while not self._stop.is_set():
                try:
                    sample, ts = self.inlet.pull_sample(timeout=1.0)
                except Exception as exc:
                    # Expected when stream closes during normal shutdown.
                    if self._stop.is_set():
                        break
                    logger.debug("LSL recorder inlet read ended for %s: %s", self.stream.name(), exc)
                    time.sleep(0.1)
                    continue
                if sample is None:
                    continue
                record = {
                    "stream_time": ts,
                    "received_time": datetime.utcnow().isoformat(),
                    "values": sample,
                }
                f.write(json.dumps(record) + "\n")


class LSLStreamRecorder:
    """Discover and record LSL streams to JSONL for offline sync."""

    def __init__(self, session_dir: Path, prefixes: list[str]):
        self.session_dir = session_dir
        self.prefixes = prefixes
        self.output_dir = session_dir / "lsl"
        self._workers: list[_LSLStreamWorker] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for w in self._workers:
            w.stop()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        # Retry discovery a few times to catch streams after capture starts
        attempts = 0
        streams = []
        while not streams and attempts < 5 and not self._stop.is_set():
            streams = self._discover_streams()
            if streams:
                break
            attempts += 1
            time.sleep(1)

        if not streams:
            logger.warning("LSL recorder: no matching streams discovered")
            return

        logger.info(
            "LSL recorder: recording %d stream(s) to %s",
            len(streams),
            self.output_dir,
        )

        self._workers = [_LSLStreamWorker(s, self.output_dir) for s in streams]
        for w in self._workers:
            w.start()

        while not self._stop.wait(1):
            continue

    def _discover_streams(self):
        streams = resolve_streams(wait_time=1.0)
        if self.prefixes:
            streams = [
                s for s in streams if any(s.name().startswith(p) for p in self.prefixes)
            ]
        return streams


class FFmpegMulticap:
    """Orchestrate multiple ffmpeg captures with shared LSL clock."""

    def __init__(
        self,
        session_dir: Path,
        devices: list[DeviceConfig],
        backend: str | None = None,
        enable_frame_log: bool = False,
        enable_markers: bool = False,
        max_duration: float | None = None,
    ):
        self.session_dir = session_dir
        self.devices = devices
        self.enable_frame_log = enable_frame_log
        self.enable_markers = enable_markers
        self.max_duration = max_duration
        # Captures can be FFmpegCapture or WDMKSAudioCapture
        self.captures: list[FFmpegCapture | WDMKSAudioCapture] = []
        self.event_logger = EventLogger(session_dir)
        self.clock_publisher: LSLClockPublisher | None = None
        self.running = False
        self.backend = backend or detect_ffmpeg_backend()
        video_devices, audio_devices = parse_available_devices(self.backend, quiet=True)
        self.device_catalog = {"video": video_devices, "audio": audio_devices} if video_devices or audio_devices else {}
        self.marker_outlet: MarkerOutlet | None = None

    def start_all(
        self,
        lsl_stream_name: str = "ffmpeg_clock",
        lsl_rate: float = 100.0,
        stabilization_delay: float = 0.0,
        sequential_start_delay: float = 0.0,
    ) -> bool:
        """Start all captures and LSL clock.

        Args:
            lsl_stream_name: Name for the shared LSL clock stream
            lsl_rate: Sample rate for LSL clock (Hz)
            stabilization_delay: Seconds to wait after processes start before recording.
                                 This gives cameras time to initialize their streams.
            sequential_start_delay: Seconds to wait between launching each camera thread.
                                    Use 0.2-0.5s to avoid DirectShow conflicts with many cameras.
                                    Only applies to video captures; audio always starts in parallel.
        """
        if self.running:
            logger.warning("Captures already running")
            return False

        if not self.backend:
            logger.error("Unsupported platform: no ffmpeg input backend detected")
            return False

        self.running = True

        # Start LSL clock
        self.clock_publisher = LSLClockPublisher(lsl_stream_name, lsl_rate)
        self.clock_publisher.start()

        # Prepare all capture objects
        for device in self.devices:
            if device.audio_only:
                output_path = device.output_path(self.session_dir)

                # Use WDM-KS backend if specified (bypasses TotalMix on Windows)
                if device.audio_backend == "wdmks":
                    if not HAS_SOUNDDEVICE:
                        logger.error(f"sounddevice not available for WDM-KS capture ({device.label})")
                        continue
                    if device.wdmks_device_id is None:
                        logger.error(f"wdmks_device_id not set for {device.label}")
                        continue
                    capture = WDMKSAudioCapture(
                        device,
                        output_path,
                        self.event_logger,
                        sample_rate=device.audio_sample_rate,
                    )
                else:
                    # Default: use FFmpeg with DirectShow/AVFoundation
                    capture = FFmpegCapture(
                        device,
                        output_path,
                        self.event_logger,
                        backend=self.backend,
                        device_catalog=self.device_catalog,
                        enable_frame_log=self.enable_frame_log,
                        max_duration=self.max_duration,
                    )
                self.captures.append(capture)
                continue

            video_device = DeviceConfig(**{**device.__dict__, "audio_only": False})
            video_output = video_device.video_output_path(self.session_dir)
            video_capture = FFmpegCapture(
                video_device,
                video_output,
                self.event_logger,
                backend=self.backend,
                device_catalog=self.device_catalog,
                enable_frame_log=self.enable_frame_log,
                emit_progress=True,
                progress_label=video_device.label,
                max_duration=self.max_duration,
            )
            self.captures.append(video_capture)

            # Only create separate audio capture if mux_audio is False
            # (when mux_audio is True, audio is already included in vid file)
            has_audio_source = (
                device.audio_alt_name
                or device.audio_name
                or device.audio_index is not None
            )
            if has_audio_source and not device.mux_audio:
                audio_device = DeviceConfig(**{**device.__dict__, "audio_only": True})
                audio_output = device.audio_output_path(self.session_dir)
                audio_capture = FFmpegCapture(
                    audio_device,
                    audio_output,
                    self.event_logger,
                    backend=self.backend,
                    device_catalog=self.device_catalog,
                    enable_frame_log=False,
                    emit_progress=False,
                    progress_label=f"{audio_device.label}_audio",
                    essential=False,  # camera mic failure shouldn't crash session
                    max_duration=self.max_duration,
                )
                self.captures.append(audio_capture)

        if not self.captures:
            logger.error("No valid capture devices were configured; aborting start.")
            if self.clock_publisher:
                self.clock_publisher.stop()
            self.running = False
            return False

        if self.enable_markers:
            self.marker_outlet = MarkerOutlet()
            self.marker_outlet.start()

        # ── Pre-capture camera setup (sequential) ─────────────────────
        # Run camera_setup_script (e.g. lock_exposure.py) for each video
        # capture *sequentially*.  This avoids:
        #   1) Multiple lock_exposure instances fighting for exclusive
        #      DirectShow device handles simultaneously.
        #   2) The race between OpenCV releasing the handle and ffmpeg
        #      immediately trying to grab it.
        # A 0.5 s cooldown after the last setup gives DirectShow time to
        # fully release device handles before ffmpeg opens them.
        any_setup_ran = False
        for cap in self.captures:
            if isinstance(cap, FFmpegCapture) and not cap.device.audio_only:
                if cap.run_camera_setup():
                    any_setup_ran = True
        if any_setup_ran:
            time.sleep(0.5)  # DirectShow handle release cooldown

        # Start all captures in parallel with microsecond-precision synchronization barrier
        # Set target start time 500ms in future to allow all threads to reach barrier
        # Add stabilization delay to give cameras time to fully initialize
        if stabilization_delay > 0:
            logger.info(f"Stabilization delay: waiting {stabilization_delay:.1f}s for cameras to initialize...")
            print(f"\n[..] Stabilization delay: {stabilization_delay:.1f}s for camera streams to initialize...")
            for i in range(int(stabilization_delay), 0, -1):
                print(f"   Starting in {i}...")
                time.sleep(1)
            remaining = stabilization_delay - int(stabilization_delay)
            if remaining > 0:
                time.sleep(remaining)
            print("   [>>] Starting recording NOW!\n")

        target_start = time.time() + 0.5

        results: list[bool] = [False] * len(self.captures)
        threads: list[threading.Thread] = []

        def _start_capture(
            idx: int, cap: FFmpegCapture | WDMKSAudioCapture, target_time: float
        ) -> None:
            try:
                results[idx] = cap.start(target_time=target_time)
            except Exception as exc:
                essential = getattr(cap, "essential", True)
                logger.error(
                    "Capture %s failed to start: %s (essential=%s)",
                    getattr(cap, "progress_label", idx),
                    exc,
                    essential,
                )
                results[idx] = False

        for i, cap in enumerate(self.captures):
            t = threading.Thread(target=_start_capture, args=(i, cap, target_start))
            threads.append(t)

        # Launch all threads as close together as possible
        # They will all wait at the barrier until target_start
        # If sequential_start_delay > 0, stagger video device launches to avoid DirectShow conflicts
        if sequential_start_delay > 0:
            logger.info(
                f"Sequential start enabled: {sequential_start_delay:.2f}s delay between video devices"
            )
            for i, (t, cap) in enumerate(zip(threads, self.captures, strict=True)):
                is_video = isinstance(cap, FFmpegCapture) and not cap.device.audio_only
                t.start()
                # Only delay between video devices; audio can start immediately
                if is_video and i < len(threads) - 1:
                    # Check if next capture is also video
                    next_cap = self.captures[i + 1] if i + 1 < len(self.captures) else None
                    next_is_video = (
                        isinstance(next_cap, FFmpegCapture) and not next_cap.device.audio_only
                    ) if next_cap else False
                    if next_is_video:
                        time.sleep(sequential_start_delay)
        else:
            for t in threads:
                t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # Check if any essential captures failed
        essential_ok = all(
            ok for cap, ok in zip(self.captures, results, strict=True)
            if getattr(cap, "essential", True)
        )
        non_essential_failed = [
            cap for cap, ok in zip(self.captures, results, strict=True)
            if not getattr(cap, "essential", True) and not ok
        ]
        if non_essential_failed:
            labels = [getattr(c, "progress_label", "?") for c in non_essential_failed]
            logger.warning(
                "Non-essential captures failed (continuing without them): %s",
                ", ".join(labels),
            )
            # Remove failed non-essential captures from the active list
            self.captures = [
                c for c, ok in zip(self.captures, results, strict=True) if ok
            ]

        if not essential_ok:
            logger.error("Essential captures failed to start — aborting session")
            started = [c for c, ok in zip(self.captures, results, strict=True) if ok]
            for capture in started:
                try:
                    capture.stop()
                except Exception as e:
                    logger.warning("Failed cleanup stop for %s: %s", capture, e)
            if self.clock_publisher:
                self.clock_publisher.stop()
            if self.marker_outlet:
                self.marker_outlet.stop()
            self.running = False
            self.captures = []
            return False

        return True

    def stop_all(self) -> None:
        """Stop all captures and LSL clock."""
        if not self.running and not any(getattr(c, "running", False) for c in self.captures):
            return

        logger.info("Stopping all captures...")
        for capture in self.captures:
            capture.stop()

        if self.clock_publisher:
            self.clock_publisher.stop()
        if self.marker_outlet:
            self.marker_outlet.stop()

        self.running = False
        logger.info("All captures stopped")

    def monitor(self) -> None:
        """Monitor captures and report status periodically."""
        while self.running:
            active = 0
            for capture in self.captures:
                if isinstance(capture, FFmpegCapture):
                    if capture.process and capture.process.poll() is not None:
                        capture.running = False
                if capture.running:
                    active += 1
            logger.info(f"Status: {active}/{len(self.captures)} captures active")
            if active == 0:
                logger.warning("No active captures remain; stopping session state.")
                self.stop_all()
                break
            time.sleep(10)


def load_config(config_path: Path) -> tuple[Path, list[DeviceConfig]]:
    """Load device config from JSON file."""
    data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    session_dir = Path(data.get("session_dir", "./data/session"))
    devices = [DeviceConfig(**d) for d in data.get("devices", [])]
    return session_dir, devices


def list_available_devices() -> int:
    """List all available audio/video devices using the platform backend."""
    backend = detect_ffmpeg_backend()
    if not backend:
        logger.error("Unsupported platform: no ffmpeg input backend detected")
        return 1

    video_devices, audio_devices = parse_available_devices(backend)

    print("\n" + "=" * 80)
    print(f"Available devices via {backend}")
    print("=" * 80)

    if video_devices:
        print("\nVIDEO DEVICES:")
        for dev in video_devices:
            print(f"  [{dev['index']}] {dev['name']}")
            if dev.get("alt_name"):
                print(f"       alt: {dev['alt_name']}")

    if audio_devices:
        print("\nAUDIO DEVICES:")
        rme_detected = False
        for dev in audio_devices:
            if "Fireface 802" in dev['name']:
                print(f"  * [{dev['index']}] {dev['name']} <- RME Fireface 802 (4x DPA-4066 auto-mapped)")
                rme_detected = True
            else:
                print(f"  [{dev['index']}] {dev['name']}")
            if dev.get("alt_name"):
                print(f"       alt: {dev['alt_name']}")
        if rme_detected:
            print("\n  NOTE: RME Fireface 802 inputs 9-12 will be mapped to dpa_1-4 with:")
            print("      - Mono (1 channel) - 48 kHz - PCM 16-bit")


    if not video_devices and not audio_devices:
        print("(no devices detected)\n")

    print("\n" + "=" * 80)
    if backend == "avfoundation":
        print("\nUsage: Update your config JSON with the device indices above.")
        print("Example: video_index: 0, audio_index: 0 for the first video/audio device")
    elif backend == "dshow":
        print("\nUsage (Windows / DirectShow): use device names in config (video_name/audio_name).")
        print("Example: video_name: Integrated Camera, audio_name: Microphone (Realtek)")
    else:
        print("\nUsage: supply device identifiers supported by your platform backend.")
    print("=" * 80 + "\n")

    return 0


def list_wdmks_devices() -> int:
    """List WDM-KS audio input devices (bypasses TotalMix on Windows)."""
    if not HAS_SOUNDDEVICE:
        logger.error("sounddevice not available. Install with: pip install sounddevice scipy")
        return 1

    print("\n" + "=" * 80)
    print("WDM-KS Audio Input Devices (direct kernel streaming)")
    print("=" * 80)

    wdmks_devices = find_wdmks_devices()

    if not wdmks_devices:
        print("\n(no WDM-KS input devices found)")
        print("\nNote: WDM-KS devices are typically available on Windows with")
        print("      professional audio interfaces like RME Fireface 802.")
        print("=" * 80 + "\n")
        return 0

    print("\n🎤 WDM-KS INPUT DEVICES:")
    for pair, info in sorted(wdmks_devices.items(), key=lambda x: x[1]['device_id']):
        dev_id = info['device_id']
        name = info['name']
        channels = info.get('max_input_channels', 2)
        samplerate = int(info.get('default_samplerate', 48000))

        # Highlight RME Fireface devices
        if 'Fireface' in name:
            print(f"  📊 [{dev_id}] {name} (ch {pair})")
            print(f"       └─ {channels} ch @ {samplerate} Hz  ← DPA mic candidate")
        else:
            print(f"  [{dev_id}] {name}")
            print(f"       └─ {channels} ch @ {samplerate} Hz")

    print("\n" + "-" * 80)
    print("CONFIG USAGE:")
    print("-" * 80)
    print("Add to device config with audio_backend='wdmks':")
    print("""
  {
    "label": "dpa_1",
    "audio_only": true,
    "audio_backend": "wdmks",
    "wdmks_device_id": 57,    // device ID from list above
    "wdmks_channel": 0,       // 0=left, 1=right
    "audio_sample_rate": 48000,
    "format": "wav",
    "subdir": "audio"
  }
""")
    print("This bypasses TotalMix routing and captures directly from hardware inputs.")
    print("=" * 80 + "\n")

    return 0


def parse_available_devices(backend: str | None = None, quiet: bool = False) -> tuple[list[dict], list[dict]]:
    """Parse available devices using ffmpeg device listing."""
    backend = backend or detect_ffmpeg_backend()
    if not backend:
        if not quiet:
            logger.error("Unsupported platform: cannot list devices")
        return [], []

    cmd = ["ffmpeg", "-f", backend, "-list_devices", "true"]
    cmd.extend(["-i", "dummy" if backend == "dshow" else ""])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stderr
    except Exception as e:
        if not quiet:
            logger.error(f"Failed to list devices ({backend}): {e}")
        return [], []

    if backend == "avfoundation":
        return _parse_avfoundation_devices(output)
    if backend == "dshow":
        return _parse_dshow_devices(output)

    if not quiet:
        logger.error(f"Device parsing for backend '{backend}' is not implemented")
    return [], []


def _parse_avfoundation_devices(output: str) -> tuple[list[dict], list[dict]]:
    video_devices: list[dict] = []
    audio_devices: list[dict] = []
    in_video = False
    in_audio = False

    for line in output.split("\n"):
        if "AVFoundation video devices:" in line:
            in_video = True
            in_audio = False
        elif "AVFoundation audio devices:" in line:
            in_video = False
            in_audio = True
        elif in_video or in_audio:
            if line.strip().startswith("[") and "] [" in line:
                parts = line.split("] [", 1)
                if len(parts) > 1:
                    rest = parts[1]
                    idx_end = rest.find("]")
                    if idx_end > 0:
                        idx = int(rest[:idx_end])
                        name = rest[idx_end + 1 :].strip()
                        if in_video:
                            video_devices.append({"index": idx, "name": name})
                        elif in_audio:
                            audio_devices.append({"index": idx, "name": name})
    return video_devices, audio_devices


def _parse_dshow_devices(output: str) -> tuple[list[dict], list[dict]]:
    video_devices: list[dict] = []
    audio_devices: list[dict] = []

    last_device: dict | None = None

    for line in output.split("\n"):
        # Match lines like: [dshow @ ...] "Device Name" (video)
        # or: [dshow @ ...] "Device Name" (audio)
        if "[dshow @" not in line:
            continue

        # Extract quoted strings
        matches = re.findall(r'"([^\"]+)"', line)
        if not matches:
            continue

        # Capture alternative name for the last device
        if "Alternative name" in line:
            if last_device is not None:
                last_device["alt_name"] = matches[0]
            continue

        device_name = matches[0]  # First quoted string is the device name

        # Determine if it's video or audio based on the (video) or (audio) marker
        if "(video)" in line:
            last_device = {
                "index": len(video_devices),
                "name": device_name,
                "alt_name": None,
            }
            video_devices.append(last_device)
        elif "(audio)" in line:
            last_device = {
                "index": len(audio_devices),
                "name": device_name,
                "alt_name": None,
            }
            audio_devices.append(last_device)
        else:
            last_device = None

    return video_devices, audio_devices


def _parse_dshow_device_kinds(output: str) -> dict[str, dict[str, str]]:
    """Parse dshow device roles from ffmpeg output, including '(none)' entries.

    Returns mapping alt_name -> {"name": <friendly>, "kind": "video|audio|none"}.
    """
    by_alt: dict[str, dict[str, str]] = {}
    last_name: str | None = None
    last_kind: str | None = None

    for line in output.split("\n"):
        if "[dshow @" not in line:
            continue

        role_match = re.search(r'"([^\"]+)" \((video|audio|none)\)', line)
        if role_match:
            last_name = role_match.group(1)
            last_kind = role_match.group(2)
            continue

        if "Alternative name" in line and last_name and last_kind:
            alt_matches = re.findall(r'"([^\"]+)"', line)
            if alt_matches:
                by_alt[alt_matches[0]] = {"name": last_name, "kind": last_kind}

    return by_alt


def get_dshow_device_kinds(quiet: bool = False) -> dict[str, dict[str, str]]:
    """Get DirectShow device roles (video/audio/none) keyed by alt_name."""
    cmd = ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        if not quiet:
            logger.warning("Failed to enumerate DirectShow device kinds: %s", exc)
        return {}

    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return _parse_dshow_device_kinds(output)


def update_config_devices(config_path: Path) -> bool:
    """Update config file with currently available devices."""
    logger.info("Detecting available devices...")
    backend = detect_ffmpeg_backend()
    video_devices, audio_devices = parse_available_devices(backend)

    if not backend:
        logger.error("Unsupported platform: cannot update config")
        return False

    if not video_devices and not audio_devices:
        logger.error("No devices detected")
        return False

    logger.info(f"Found {len(video_devices)} video and {len(audio_devices)} audio devices")

    try:
        data = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except Exception:
        data = {"session_dir": "./data/sub-001/ses-001"}

    devices: list[dict] = []

    def _audio_name_from_index(idx: int | None) -> str | None:
        if idx is None:
            return None
        for aud in audio_devices:
            if aud.get("index") == idx:
                return aud.get("name")
        return None

    for vid in video_devices:
        if "Capture screen" in vid["name"]:
            continue

        label = vid["name"].lower().replace(" ", "_").replace("-", "_")
        label = "".join(c for c in label if c.isalnum() or c == "_")
        label = label[:32]

        audio_idx = None
        audio_name = None
        for aud in audio_devices:
            if vid["name"] in aud["name"] or aud["name"] in vid["name"]:
                audio_idx = aud["index"]
                audio_name = aud["name"]
                break
        if audio_idx is None and audio_devices:
            audio_idx = audio_devices[0]["index"]
            audio_name = audio_devices[0]["name"]

        device = {
            "label": f"{label}_vid",
            "video_index": vid["index"],
            "audio_index": audio_idx,
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "video_bitrate": 4000,
            "audio_bitrate": 128,
            "format": "mkv",
            "subdir": "video",
        }

        if backend == "dshow":
            device["video_name"] = vid.get("name")
            device["audio_name"] = audio_name or _audio_name_from_index(audio_idx)
            # Store alt names (unique device paths) to disambiguate duplicate display names
            if vid.get("alt_name"):
                device["video_alt_name"] = vid["alt_name"]
            audio_alt = next(
                (a.get("alt_name") for a in audio_devices if a.get("index") == audio_idx and a.get("alt_name")),
                None,
            )
            if audio_alt:
                device["audio_alt_name"] = audio_alt

        devices.append(device)

    for aud in audio_devices:
        label = aud["name"].lower().replace(" ", "_").replace("-", "_")
        label = "".join(c for c in label if c.isalnum() or c == "_")
        label = label[:32]

        device = {
            "label": f"{label}_aud",
            "audio_index": aud["index"],
            "audio_only": True,
            "format": "wav",
            "subdir": "audio",
        }

        if backend == "dshow":
            device["audio_name"] = aud.get("name")
            if aud.get("alt_name"):
                device["audio_alt_name"] = aud["alt_name"]

        devices.append(device)

    # Add RME Fireface 802 support with DPA microphone channels
    rme_fireface = None
    for aud in audio_devices:
        if "Fireface 802" in aud.get("name", ""):
            rme_fireface = aud
            break

    if rme_fireface:
        logger.info(f"Detected RME Fireface 802 at audio index {rme_fireface['index']}")
        print("\n[ok] Detected RME Fireface 802 - Adding 4x DPA-4066 microphone entries (inputs 9-12)\n")
        # Map 4 DPA microphones to Fireface input channels 9-12 (indices 8-11)
        for i in range(4):
            dpa_device = {
                "label": f"dpa_{i+1}",
                "audio_index": 8 + i,  # Inputs 9-12 map to indices 8-11
                "audio_only": True,
                "audio_channels": 1,
                "audio_sample_rate": 48000,
                "audio_bitrate": 24,
                "format": "wav",
                "subdir": "audio",
            }
            if backend == "dshow":
                dpa_device["audio_name"] = rme_fireface.get("name")
                if rme_fireface.get("alt_name"):
                    dpa_device["audio_alt_name"] = rme_fireface["alt_name"]
            devices.append(dpa_device)

    data["devices"] = devices
    config_path.write_text(json.dumps(data, indent=2))
    logger.info(f"Updated {config_path} with {len(devices)} devices")

    print("\n" + "=" * 80)
    print("Updated Config Summary")
    print("=" * 80)
    for dev in devices:
        if dev.get("audio_only"):
            target = dev.get("audio_name") or f"audio[{dev['audio_index']}]"
            print(f"  [aud] {dev['label']}: {target}")
        else:
            v_target = dev.get("video_name") or f"video[{dev['video_index']}]"
            a_target = dev.get("audio_name") or f"audio[{dev['audio_index']}]"
            print(f"  [vid] {dev['label']}: {v_target} + {a_target}")
    print("=" * 80 + "\n")

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FFmpeg multi-device capture with LSL clock for AffectAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument(
        "--group-id",
        help="Group ID / name for session folder naming (format: {group_id}_{YYYYMMDD}_{HHMMSS})",
    )
    parser.add_argument(
        "--lsl-stream-name",
        default="ffmpeg_clock",
        help="LSL stream name for shared clock",
    )
    parser.add_argument(
        "--lsl-rate",
        type=float,
        default=100.0,
        help="LSL clock sample rate (Hz)",
    )
    parser.add_argument(
        "--enable-markers",
        action="store_true",
        help="Enable LSL marker outlet (affectai_markers)",
    )
    parser.add_argument(
        "--emit-marker",
        action="append",
        help="Emit a marker label immediately (can be specified multiple times)",
    )
    parser.add_argument(
        "--record-lsl",
        action="store_true",
        help="Record LSL streams (default: names starting with ffmpeg_) to session/lsl/*.jsonl",
    )
    parser.add_argument(
        "--frame-log",
        action="store_true",
        help="Log per-frame PTS via ffmpeg showinfo to <session>/frame_logs/*.jsonl (video devices only)",
    )
    parser.add_argument(
        "--lsl-prefixes",
        default="ffmpeg_",
        help="Comma-separated stream name prefixes to record (default: ffmpeg_)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all available capture audio/video devices for the active platform backend and exit",
    )
    parser.add_argument(
        "--list-wdmks",
        action="store_true",
        help="List WDM-KS audio input devices (Windows, for direct RME Fireface capture)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Auto-detect and update config file with available devices before running",
    )
    parser.add_argument(
        "--stabilization-delay",
        type=float,
        default=0.0,
        help="Seconds to wait after ffmpeg processes start before actual recording begins (default: 0). "
             "Use 2-3 seconds to ensure all camera streams are fully initialized.",
    )
    parser.add_argument(
        "--sequential-start-delay",
        type=float,
        default=0.0,
        help="Seconds to wait between launching each video capture (default: 0 = parallel start). "
             "Use 0.2-0.5s to avoid DirectShow conflicts when capturing many cameras. "
             "Only delays video devices; audio captures always start in parallel.",
    )
    parser.add_argument(
        "--keep-zoom",
        action="store_true",
        help="Keep intelligent zoom/auto-framing enabled on Jabra PanaCast cameras (default: disabled)",
    )
    parser.add_argument(
        "--mux-audio",
        action="store_true",
        help="Force-enable camera audio muxing into video files for video devices with audio inputs",
    )
    parser.add_argument(
        "--show-camera-dialog",
        action="store_true",
        help="Open native DirectShow property dialog for each video device before "
             "capture starts (lets you lock exposure, WB, gain interactively)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=7200,
        help="Maximum recording duration in seconds (default: 7200 = 2 hours). "
             "Use 0 or negative value for unlimited recording. Example: 10800 for 3 hours.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.emit_marker:
        args.enable_markers = True

    if args.emit_marker and not args.config:
        marker_outlet = MarkerOutlet()
        marker_outlet.start()
        for label in args.emit_marker:
            marker_outlet.emit(label)
        marker_outlet.stop()
        return 0

    # Handle --list-devices flag
    if args.list_devices:
        return list_available_devices()

    # Handle --list-wdmks flag
    if args.list_wdmks:
        return list_wdmks_devices()

    # Config is required for normal operation
    if not args.config:
        logger.error("--config is required (or use --list-devices to see available devices)")
        return 1

    # Handle --update flag
    if args.update:
        if not update_config_devices(Path(args.config)):
            logger.error("Failed to update config with available devices")
            return 1

    session_dir, devices = load_config(Path(args.config))

    if detect_ffmpeg_backend() == "dshow":
        dshow_kinds = get_dshow_device_kinds(quiet=True)
        none_video: list[tuple[str, str]] = []
        missing_video: list[str] = []

        for device in devices:
            if device.audio_only or not device.video_alt_name:
                continue
            info = dshow_kinds.get(device.video_alt_name)
            if info is None:
                missing_video.append(device.label)
            elif info.get("kind") == "none":
                none_video.append((device.label, info.get("name", "?")))

        if none_video or missing_video:
            if none_video:
                logger.error(
                    "Configured cameras are present but not video-capable in DirectShow '(none)': %s",
                    ", ".join(f"{label}={name}" for label, name in none_video),
                )
            if missing_video:
                logger.error(
                    "Configured cameras are missing from DirectShow device listing: %s",
                    ", ".join(missing_video),
                )
            logger.error(
                "Preflight failed. Run '--list-devices' and ensure each target camera appears as '(video)' "
                "before recording (replug or power-cycle cameras/hubs if shown as '(none)')."
            )
            return 1

    # Create session directory name with group_id (if provided) + date + time
    # Format: {group_id}_{YYYYMMDD}_{HHMMSS} or <original_name>_<HHMMSS> if no group_id
    if args.group_id:
        timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dirname = f"{args.group_id}_{timestamp_suffix}"
    else:
        timestamp_suffix = datetime.now().strftime("_%H%M%S")
        session_dirname = session_dir.name + timestamp_suffix
    session_dir = session_dir.parent / session_dirname
    logger.info("Session directory (timestamped): %s", session_dir)

    if not devices:
        logger.error("No devices defined in config")
        return 1

    if args.mux_audio:
        mux_enabled = 0
        for device in devices:
            if device.audio_only:
                continue
            if device.audio_index is None and not device.audio_name and not device.audio_alt_name:
                continue
            device.mux_audio = True
            mux_enabled += 1
        logger.warning(
            "--mux-audio enabled: forcing muxed camera audio for %d video device(s). "
            "This may reduce sync stability under USB load.",
            mux_enabled,
        )

    # Apply --show-camera-dialog to all video devices
    if args.show_camera_dialog:
        for device in devices:
            if not device.audio_only:
                device.show_camera_dialog = True
        logger.info(
            "--show-camera-dialog: native DirectShow property dialog will open "
            "for each video device before capture starts"
        )

    session_dir.mkdir(parents=True, exist_ok=True)

    # Disable intelligent zoom on Jabra cameras by default (use --keep-zoom to skip)
    if not args.keep_zoom:
        disable_jabra_intelligent_zoom()

    lsl_prefixes = [p.strip() for p in args.lsl_prefixes.split(",") if p.strip()]
    lsl_recorder = LSLStreamRecorder(session_dir, lsl_prefixes) if args.record_lsl else None

    multicap = FFmpegMulticap(
        session_dir,
        devices,
        enable_frame_log=args.frame_log,
        enable_markers=args.enable_markers,
        max_duration=args.max_duration,
    )

    # Signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping captures...")
        if lsl_recorder:
            lsl_recorder.stop()
        multicap.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start all captures
    if not multicap.start_all(
        args.lsl_stream_name,
        args.lsl_rate,
        stabilization_delay=args.stabilization_delay,
        sequential_start_delay=args.sequential_start_delay,
    ):
        logger.error("Failed to start captures")
        return 1

    if lsl_recorder:
        lsl_recorder.start()

    logger.info(f"All {len(multicap.captures)} captures started. Press Ctrl+C to stop.")

    if multicap.marker_outlet and args.emit_marker:
        for label in args.emit_marker:
            multicap.marker_outlet.emit(label)

    # Monitor in main thread
    try:
        multicap.monitor()
    except KeyboardInterrupt:
        if lsl_recorder:
            lsl_recorder.stop()
        multicap.stop_all()
    finally:
        if lsl_recorder:
            lsl_recorder.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
