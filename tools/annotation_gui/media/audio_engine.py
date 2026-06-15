"""Multichannel audio engine for the annotation GUI.

Loads N mono/stereo WAVs (typically 7 channels from DPA + room mics), resamples
them to a common sample rate, and plays them in lockstep via sounddevice.
The master clock drives playback position; per-channel gain is applied.
Each channel plays independently (no mixing, no normalization) for raw quality.

Heavy deps (sounddevice, soundfile, numpy) are imported lazily so the module
can be imported in environments without them (e.g. CI). The GUI will surface
import errors through the MainWindow rather than crashing at startup.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .clock import MasterClock

logger = logging.getLogger(__name__)


@dataclass
class Channel:
    name: str
    path: Path
    gain: float = 1.0
    mute: bool = False
    solo: bool = False
    time_offset: float = 0.0  # seconds offset for this channel
    # Loaded lazily: mono float32 PCM at engine sample_rate.
    samples: object | None = None  # numpy.ndarray when loaded
    sample_rate: int = 0
    duration: float = 0.0
    source_rate: int = 0
    source_bits: int = 0
    source_subtype: str = ""


@dataclass
class AudioEngine:
    clock: MasterClock
    # `sample_rate=0` means "use the default output device's native rate".
    sample_rate: int = 0
    block_size: int = 0  # 0 → driver picks
    output_channels: int = 2  # stereo output
    master_gain: float = 1.0
    channels: list[Channel] = field(default_factory=list)
    _stream: object | None = None
    _device_info: dict | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _ensure_sample_rate(self) -> int:
        if self.sample_rate > 0:
            return self.sample_rate
        import sounddevice as sd

        self._device_info = sd.query_devices(kind="output")
        self.sample_rate = int(self._device_info["default_samplerate"])
        return self.sample_rate

    def set_master_gain(self, gain: float) -> None:
        self.master_gain = max(0.0, float(gain))

    def add_wav(self, path: Path, name: str | None = None) -> Channel:
        self._ensure_sample_rate()
        ch = Channel(name=name or path.stem, path=Path(path))
        self._load(ch)
        self.channels.append(ch)
        dur = max((c.duration for c in self.channels), default=0.0)
        if dur > self.clock.duration:
            self.clock.set_duration(dur)
        return ch

    def _load(self, ch: Channel) -> None:
        import numpy as np
        import soundfile as sf

        info = sf.info(str(ch.path))
        data, sr = sf.read(str(ch.path), dtype="float32", always_2d=True)
        # Mix multi-channel source files down to mono for per-channel control.
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        if sr != self.sample_rate:
            mono = _hq_resample(mono, sr, self.sample_rate)
        ch.samples = mono.astype(np.float32, copy=False)
        ch.sample_rate = self.sample_rate
        ch.duration = len(ch.samples) / float(self.sample_rate)
        ch.source_rate = int(sr)
        ch.source_subtype = str(info.subtype or "")
        ch.source_bits = _parse_bits_per_sample(ch.source_subtype)

    def set_gain(self, idx: int, gain: float) -> None:
        with self._lock:
            self.channels[idx].gain = max(0.0, float(gain))

    def set_mute(self, idx: int, mute: bool) -> None:
        with self._lock:
            self.channels[idx].mute = bool(mute)

    def set_solo(self, idx: int, solo: bool) -> None:
        with self._lock:
            self.channels[idx].solo = bool(solo)

    def set_channel_offset(self, idx: int, offset_seconds: float) -> None:
        with self._lock:
            self.channels[idx].time_offset = float(offset_seconds)

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        self._ensure_sample_rate()
        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.output_channels,
            dtype="float32",
            callback=self._callback,
        )
        stream.start()
        self._stream = stream
        dev = self._device_info or sd.query_devices(kind="output")
        logger.info(
            f"[audio_engine] output={dev.get('name', '?')} "
            f"rate={self.sample_rate} channels={self.output_channels} "
            f"blocksize={self.block_size or 'auto'} sources={len(self.channels)}"
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):  # noqa: ARG002
        import numpy as np

        outdata.fill(0.0)
        if not self.clock.is_playing() or not self.channels:
            return

        with self._lock:
            any_solo = any(c.solo for c in self.channels)
            active = [
                c for c in self.channels
                if c.samples is not None and not c.mute and (c.solo or not any_solo)
            ]
            start_sample = int(self.clock.position() * self.sample_rate)

        mix = np.zeros(frames, dtype=np.float64)
        for c in active:
            # Apply time offset for this channel
            offset_samples = int(c.time_offset * self.sample_rate)
            ch_start_sample = start_sample + offset_samples
            ch_end_sample = ch_start_sample + frames
            # Check bounds
            if ch_start_sample >= len(c.samples):
                continue
            # Clamp to valid range
            seg_start = max(0, ch_start_sample)
            seg_end = min(len(c.samples), ch_end_sample)
            if seg_start >= seg_end:
                continue
            seg = c.samples[seg_start:seg_end]
            # Calculate offset into the output buffer
            out_offset = max(0, -ch_start_sample)
            if len(seg) > 0:
                mix[out_offset : out_offset + len(seg)] += seg * c.gain

        if not np.any(mix):
            return

        mix *= float(self.master_gain)
        out = mix.astype(np.float32, copy=False)
        # Fan out mono mix to all output channels.
        for ch in range(self.output_channels):
            outdata[:, ch] = out


def _hq_resample(samples, src_rate: int, dst_rate: int):
    """Polyphase (sinc-windowed) resample when scipy is available.

    Falls back to linear interpolation if scipy is missing — audibly worse,
    but keeps the engine usable in minimal environments.
    """
    import numpy as np

    if src_rate == dst_rate:
        return samples

    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src_rate, dst_rate)
        up, down = dst_rate // g, src_rate // g
        return resample_poly(samples, up, down).astype(np.float32, copy=False)
    except ImportError:
        pass

    try:
        import av

        src = samples.astype(np.float32, copy=False)
        frame = av.AudioFrame.from_ndarray(src.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = int(src_rate)
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=int(dst_rate))

        out_chunks: list[np.ndarray] = []
        for out_frame in resampler.resample(frame) or []:
            out_chunks.append(out_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False))
        for out_frame in resampler.resample(None) or []:
            out_chunks.append(out_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False))
        if out_chunks:
            return np.concatenate(out_chunks, axis=0)
    except Exception:
        pass

    # Last-resort fallback.
    logger.warning("Falling back to linear audio resampling (%s -> %s Hz).", src_rate, dst_rate)
    try:
        ratio = dst_rate / src_rate
        n_out = int(round(len(samples) * ratio))
        x_src = np.arange(len(samples), dtype=np.float64)
        x_dst = np.linspace(0, len(samples) - 1, n_out, dtype=np.float64)
        return np.interp(x_dst, x_src, samples).astype(np.float32)
    except Exception:
        return samples.astype(np.float32, copy=False)


def _parse_bits_per_sample(subtype: str) -> int:
    digits = "".join(ch for ch in subtype if ch.isdigit())
    if not digits:
        return 0
    try:
        return int(digits)
    except ValueError:
        return 0
