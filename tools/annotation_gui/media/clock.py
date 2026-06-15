"""Master timeline clock for synchronised audio+video playback.

Exposes a single monotonic "media position" in seconds that every engine
(audio, video, timeline cursor) reads from. Using a single source of truth
avoids drift between independently-timed players.

The clock is stateful but purely numeric — it has no Qt dependency, so it
can be unit-tested without a display.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class MasterClock:
    """Wall-clock-driven media position.

    position(t_wall) = offset + (t_wall - anchor_wall) * rate   while playing
                     = offset                                  while paused

    `duration` is the task's total length; `position()` is clamped to [0, duration].
    """

    duration: float = 0.0
    rate: float = 1.0
    _offset: float = 0.0
    _anchor_wall: float = 0.0
    _playing: bool = False

    def _now(self) -> float:
        return time.monotonic()

    def position(self) -> float:
        if not self._playing:
            return max(0.0, min(self._offset, self.duration))
        pos = self._offset + (self._now() - self._anchor_wall) * self.rate
        if pos >= self.duration:
            self._offset = self.duration
            self._playing = False
            return self.duration
        if pos < 0.0:
            self._offset = 0.0
            self._playing = False
            return 0.0
        return pos

    def is_playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        if self._playing:
            return
        self._anchor_wall = self._now()
        self._playing = True

    def pause(self) -> None:
        if not self._playing:
            return
        self._offset = self.position()
        self._playing = False

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def seek(self, seconds: float) -> None:
        seconds = max(0.0, min(seconds, self.duration))
        self._offset = seconds
        if self._playing:
            self._anchor_wall = self._now()

    def set_rate(self, rate: float) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self._offset = self.position()
        self._anchor_wall = self._now()
        self.rate = rate

    def set_duration(self, duration: float) -> None:
        self.duration = max(0.0, duration)
        if self._offset > self.duration:
            self.seek(self.duration)
