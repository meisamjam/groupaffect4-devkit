"""Multi-tier timeline widget (pyqtgraph).

Each tier is one horizontal row. Spans are drawn as filled rectangles,
points as thin vertical ticks. A playhead follows the master clock, and
clicking the timeline seeks. Clicking inside a span selects that entry
(for edit/delete).
"""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..io.annotations import AnnotationDoc, Entry

_PALETTE = [
    "#4C9AFF", "#F78166", "#B392F0", "#79C0FF", "#F69D50",
    "#A5D6FF", "#FFD596", "#D2A8FF", "#FFA198", "#7EE787",
]


@dataclass
class _TierRow:
    tier_id: str
    y: int
    color: str
    readonly: bool


class TimelineWidget(QWidget):
    """Tiered timeline. Emits signals for seeking and entry selection."""

    seekRequested = Signal(float)
    entrySelected = Signal(object)  # Entry or None
    entryDoubleClicked = Signal(object)  # Entry

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._doc: AnnotationDoc | None = None
        self._duration: float = 0.0
        self._tier_rows: dict[str, _TierRow] = {}
        self._span_items: list[tuple[Entry, pg.LinearRegionItem]] = []
        self._point_items: list[tuple[Entry, pg.InfiniteLine]] = []
        self._selected: Entry | None = None

        self._plot = pg.PlotWidget(background="#1e1e1e")
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.hideButtons()
        self._plot.setLabel("bottom", "time", units="ms")
        self._plot.getPlotItem().getAxis("left").setStyle(tickTextOffset=8)
        self._plot.getPlotItem().getViewBox().setLimits(yMin=-0.5)

        self._playhead = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#ffffff", width=2))
        self._playhead.setZValue(100)
        self._plot.addItem(self._playhead)

        self._plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

    # ------------------------------------------------------------------- state
    def set_document(self, doc: AnnotationDoc | None, duration: float) -> None:
        self._doc = doc
        self._duration = max(0.0, duration)
        self._selected = None
        self.refresh()

    def set_playhead(self, t: float) -> None:
        self._playhead.setPos(t * 1000.0)

    def set_selected(self, entry: Entry | None) -> None:
        self._selected = entry
        self._apply_selection_style()

    # ----------------------------------------------------------------- drawing
    def refresh(self) -> None:
        plot = self._plot.getPlotItem()
        for _, item in self._span_items:
            plot.removeItem(item)
        for _, item in self._point_items:
            plot.removeItem(item)
        self._span_items.clear()
        self._point_items.clear()
        self._tier_rows.clear()

        if self._doc is None or not self._doc.tiers:
            plot.setXRange(0, max(self._duration * 1000.0, 1000.0), padding=0)
            plot.setYRange(-0.5, 0.5, padding=0)
            plot.getAxis("left").setTicks([[]])
            return

        # Assign row index per tier (bottom → top in playback order).
        ticks: list[tuple[float, str]] = []
        for idx, tier in enumerate(self._doc.tiers):
            color = tier.color or _PALETTE[idx % len(_PALETTE)]
            self._tier_rows[tier.id] = _TierRow(
                tier_id=tier.id, y=idx, color=color, readonly=tier.readonly
            )
            ticks.append((float(idx), tier.id))
        plot.getAxis("left").setTicks([ticks])

        for entry in self._doc.entries:
            row = self._tier_rows.get(entry.tier)
            if row is None:
                continue
            if entry.start == entry.end:
                self._add_point(entry, row)
            else:
                self._add_span(entry, row)

        plot.setXRange(0, max(self._duration * 1000.0, 1000.0), padding=0)
        plot.setYRange(-0.5, max(0.5, len(self._doc.tiers) - 0.5), padding=0)

    def _add_span(self, entry: Entry, row: _TierRow) -> None:
        color = QColor(row.color)
        color.setAlphaF(0.55)
        region = pg.LinearRegionItem(
            values=(entry.start * 1000.0, entry.end * 1000.0),
            orientation="vertical",
            brush=color,
            pen=pg.mkPen(row.color, width=1),
            movable=not row.readonly,
        )
        # Clamp region to tier row vertically by wrapping in a ViewBox-level
        # item is impossible without a custom item. Instead, we rely on the
        # region being full-height and use label offsets to keep rows legible.
        region.setZValue(10 + row.y)
        region.setRegion((entry.start * 1000.0, entry.end * 1000.0))
        region.sigRegionChangeFinished.connect(
            lambda r, e=entry: self._on_region_changed(e, r)
        )
        region.mouseDragEvent = self._guard_region_drag(region, row.readonly, region.mouseDragEvent)
        region.mouseClickEvent = self._wrap_click(region, entry)
        self._plot.addItem(region)
        self._span_items.append((entry, region))

    def _add_point(self, entry: Entry, row: _TierRow) -> None:
        line = pg.InfiniteLine(
            pos=entry.start * 1000.0,
            angle=90,
            pen=pg.mkPen(row.color, width=2),
            movable=not row.readonly,
        )
        line.setZValue(20 + row.y)
        line.sigPositionChangeFinished.connect(
            lambda ln, e=entry: self._on_point_moved(e, ln)
        )
        self._plot.addItem(line)
        self._point_items.append((entry, line))

    # ------------------------------------------------------------------ events
    def _on_scene_clicked(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        pos = ev.scenePos()
        vb = self._plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        t = float(mouse_point.x()) / 1000.0
        self.seekRequested.emit(max(0.0, min(t, self._duration)))

    def _wrap_click(self, region: pg.LinearRegionItem, entry: Entry):
        original = region.mouseClickEvent

        def handler(ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                self._selected = entry
                self._apply_selection_style()
                self.entrySelected.emit(entry)
                if ev.double():
                    self.entryDoubleClicked.emit(entry)
            original(ev)

        return handler

    def _guard_region_drag(self, region, readonly: bool, original):
        if not readonly:
            return original

        def handler(ev):
            ev.ignore()

        return handler

    def _on_region_changed(self, entry: Entry, region: pg.LinearRegionItem) -> None:
        lo, hi = region.getRegion()
        entry.start = float(min(lo, hi)) / 1000.0
        entry.end = float(max(lo, hi)) / 1000.0

    def _on_point_moved(self, entry: Entry, line: pg.InfiniteLine) -> None:
        t = float(line.value()) / 1000.0
        entry.start = entry.end = t

    def _apply_selection_style(self) -> None:
        for entry, region in self._span_items:
            row = self._tier_rows.get(entry.tier)
            if row is None:
                continue
            color = QColor(row.color)
            color.setAlphaF(0.85 if entry is self._selected else 0.55)
            region.setBrush(color)
        for entry, line in self._point_items:
            row = self._tier_rows.get(entry.tier)
            if row is None:
                continue
            width = 4 if entry is self._selected else 2
            line.setPen(pg.mkPen(row.color, width=width))
