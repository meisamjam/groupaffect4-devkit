"""Waveform visualization widget for annotation playback."""

from __future__ import annotations

from math import ceil

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget


class WaveformWidget(QWidget):
	"""Display stacked waveforms for currently loaded audio channels."""

	seekRequested = Signal(float)

	def __init__(self, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self._duration: float = 0.0
		self._channels: list[tuple[str, object]] = []
		self._render_guard = False

		self._plot = pg.PlotWidget(background="#12171f")
		self._plot.setMouseEnabled(x=True, y=False)
		self._plot.setMenuEnabled(False)
		self._plot.hideButtons()
		self._plot.setLabel("bottom", "time (ms)")
		self._plot.getPlotItem().getAxis("left").setStyle(tickTextOffset=8)
		self._plot.showGrid(x=True, y=True, alpha=0.2)
		self._plot.setAntialiasing(True)
		self._plot.getPlotItem().vb.sigXRangeChanged.connect(self._on_xrange_changed)

		self._playhead = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#ffffff", width=2))
		self._playhead.setZValue(20)
		self._plot.addItem(self._playhead)
		self._plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self._plot)

	def clear(self) -> None:
		self._channels.clear()
		self._duration = 0.0
		self._plot.getPlotItem().clear()
		self._plot.addItem(self._playhead)
		self._plot.setXRange(0, 1000.0, padding=0)
		self._plot.setYRange(-1.0, 1.0, padding=0)
		self._plot.getAxis("left").setTicks([[]])

	def set_channels(self, channels: list[object], duration: float) -> None:
		self._channels = []
		self._duration = max(0.0, float(duration))
		for channel in channels:
			name = str(getattr(channel, "name", "audio"))
			samples = getattr(channel, "samples", None)
			if samples is None:
				continue
			self._channels.append((name, samples))
		self._render(preserve_x=False)

	def set_playhead(self, t: float) -> None:
		self._playhead.setPos(float(t) * 1000.0)

	def zoom_in_x(self) -> None:
		self._scale_x(0.5)

	def zoom_out_x(self) -> None:
		self._scale_x(2.0)

	def reset_zoom_x(self) -> None:
		end_ms = max(1000.0, self._duration * 1000.0)
		self._plot.setXRange(0.0, end_ms, padding=0)
		self._render(preserve_x=True)

	def _scale_x(self, factor: float) -> None:
		if self._duration <= 0:
			return
		x0, x1 = self._plot.getPlotItem().vb.viewRange()[0]
		center = float(self._playhead.value())
		if center < x0 or center > x1:
			center = (x0 + x1) / 2.0

		half = max(25.0, ((x1 - x0) * factor) / 2.0)
		max_ms = self._duration * 1000.0
		new_x0 = max(0.0, center - half)
		new_x1 = min(max_ms, center + half)
		if new_x1 - new_x0 < 50.0:
			return
		self._plot.setXRange(new_x0, new_x1, padding=0)
		self._render(preserve_x=True)

	def _on_xrange_changed(self, *_args) -> None:
		if self._render_guard:
			return
		self._render(preserve_x=True)

	def _render(self, preserve_x: bool) -> None:
		import numpy as np

		self._render_guard = True
		try:
			x_range = self._plot.getPlotItem().vb.viewRange()[0] if preserve_x else None

			self._plot.getPlotItem().clear()
			self._plot.addItem(self._playhead)

			if not self._channels or self._duration <= 0:
				self._plot.setXRange(0, max(1000.0, self._duration * 1000.0), padding=0)
				self._plot.setYRange(-1.0, 1.0, padding=0)
				self._plot.getAxis("left").setTicks([[]])
				return

			y_ticks: list[tuple[float, str]] = []
			spacing = 2.4
			max_points = 4000
			palette = ["#4C9AFF", "#F78166", "#7EE787", "#B392F0", "#FFD596", "#79C0FF"]

			if x_range is None:
				x0_ms, x1_ms = (0.0, self._duration * 1000.0)
			else:
				x0_ms, x1_ms = x_range
			x0_ms = max(0.0, float(x0_ms))
			x1_ms = min(float(self._duration * 1000.0), float(x1_ms))

			for i, (name, samples_obj) in enumerate(self._channels):
				samples = np.asarray(samples_obj, dtype=np.float32)
				if samples.size == 0:
					continue

				n = int(samples.size)
				i0 = int((x0_ms / (self._duration * 1000.0)) * n) if self._duration > 0 else 0
				i1 = int((x1_ms / (self._duration * 1000.0)) * n) if self._duration > 0 else n
				i0 = max(0, min(n - 1, i0)) if n > 0 else 0
				i1 = max(i0 + 2, min(n, i1)) if n > 1 else 1
				visible = samples[i0:i1]
				if visible.size <= 1:
					continue

				stride = max(1, int(ceil(visible.size / max_points)))
				y = visible[::stride]
				x_ms = np.linspace(x0_ms, x1_ms, num=y.size, endpoint=False, dtype=np.float64)

				local_min = float(np.min(y))
				local_max = float(np.max(y))
				center = (local_min + local_max) * 0.5
				span = max(1e-6, local_max - local_min)
				y = (y - center) / (0.5 * span)

				y0 = float(i) * spacing
				color = palette[i % len(palette)]
				self._plot.plot(x_ms, y + y0, pen=pg.mkPen(color, width=1.2))
				self._plot.addLine(y=y0, pen=pg.mkPen("#3a4554", width=0.8))
				y_ticks.append((y0, name))

			y_max = max(1.0, (len(y_ticks) - 1) * spacing + 1.4)
			if preserve_x:
				self._plot.setXRange(x0_ms, x1_ms, padding=0)
			else:
				self._plot.setXRange(0.0, self._duration * 1000.0, padding=0)
			self._plot.setYRange(-1.3, y_max, padding=0)
			self._plot.getAxis("left").setTicks([y_ticks])
		finally:
			self._render_guard = False

	def _on_scene_clicked(self, ev) -> None:
		if ev.button() != Qt.MouseButton.LeftButton:
			return
		pos = ev.scenePos()
		vb = self._plot.getPlotItem().vb
		if not vb.sceneBoundingRect().contains(pos):
			return
		mouse_point = vb.mapSceneToView(pos)
		t_seconds = float(mouse_point.x()) / 1000.0
		self.seekRequested.emit(max(0.0, min(t_seconds, self._duration)))

