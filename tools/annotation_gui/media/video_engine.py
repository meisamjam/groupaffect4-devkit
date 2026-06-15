"""PyAV-based video decoder that renders frames matching the master clock.

Each VideoPanel owns one file, one decoder thread, and one QLabel displaying
the most recent frame. A Qt timer polls the master clock at ~30 Hz and asks
each panel to advance to the corresponding frame. Because every panel reads
from the same clock, the panels stay in sync without inter-panel messaging.

Design notes:
- Decoding happens on the Qt thread for simplicity; fine for 5–7 streams at
  720p/1080p on a workstation. If this becomes a bottleneck we can move each
  panel to a QThread with a frame queue.
- We decode *forward only* with a seek when the clock jumps backwards or far
  ahead. The seek keyframe precision is "any" so small scrubs are cheap.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

try:
    import av  # type: ignore
    from PySide6.QtCore import QPoint, QRect, Qt, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QRubberBand, QVBoxLayout, QWidget
except Exception:  # pragma: no cover - import guard for headless CI
    av = None  # type: ignore
    Qt = QPoint = QRect = Signal = QImage = QPixmap = QHBoxLayout = QLabel = QRubberBand = QVBoxLayout = QWidget = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

from .clock import MasterClock


_CAM_FEED_FLIP_RE = re.compile(r"(?:^|[^a-z0-9])(?:cam|camera)[_-]?([1-4])(?:[^a-z0-9]|$)")
_CAM5_FEED_RE = re.compile(r"(?:^|[^a-z0-9])(?:cam|camera)[_-]?5(?:[^a-z0-9]|$)")
_TOBII_FEED_RE = re.compile(r"(?:^|[^a-z0-9])tobii(?:[^a-z0-9]|$)")
_P1_P4_FEED_RE = re.compile(r"(?:^|[^a-z0-9])p([1-4])(?:[^a-z0-9]|$)")


def _should_flip_180(label: str) -> bool:
    """Return True when a video label refers to cam1-cam4 feeds."""
    return bool(_CAM_FEED_FLIP_RE.search((label or "").strip().lower()))


def _is_cam5_feed(label: str) -> bool:
    """Return True when a video label refers to cam5 feed."""
    return bool(_CAM5_FEED_RE.search((label or "").strip().lower()))


def _is_tobii_feed(label: str) -> bool:
    """Return True when a video label refers to a Tobii feed."""
    return bool(_TOBII_FEED_RE.search((label or "").strip().lower()))


def _is_p1_p4_feed(label: str) -> bool:
    """Return True when a video label refers to participant feeds P1-P4."""
    return bool(_P1_P4_FEED_RE.search((label or "").strip().lower()))


def _flip_frame_180(frame: np.ndarray) -> np.ndarray:
    """Rotate an RGB frame 180 degrees."""
    return np.ascontiguousarray(np.flip(frame, axis=(0, 1)))


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two [x, y, w, h] boxes."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    union = float(aw * ah + bw * bh - inter)
    if union <= 0:
        return 0.0
    return inter / union


def _stabilize_face_boxes(
    previous: list[tuple[int, int, int, int]],
    current: list[tuple[int, int, int, int]],
    smooth_alpha: float = 0.65,
    min_iou: float = 0.1,
) -> list[tuple[int, int, int, int]]:
    """Smooth per-face boxes frame-to-frame to reduce jitter."""
    if not current:
        return []
    if not previous:
        return current[:2]

    stabilized: list[tuple[int, int, int, int]] = []
    remaining = current[:2]
    for prev_box in previous[:2]:
        best_idx = -1
        best_iou = 0.0
        for idx, cur_box in enumerate(remaining):
            iou = _box_iou(prev_box, cur_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx < 0 or best_iou < min_iou:
            continue

        cur = remaining.pop(best_idx)
        px, py, pw, ph = prev_box
        cx, cy, cw, ch = cur
        a = max(0.0, min(1.0, float(smooth_alpha)))
        sx = int(round(a * px + (1.0 - a) * cx))
        sy = int(round(a * py + (1.0 - a) * cy))
        sw = int(round(a * pw + (1.0 - a) * cw))
        sh = int(round(a * ph + (1.0 - a) * ch))
        stabilized.append((sx, sy, max(1, sw), max(1, sh)))

    stabilized.extend(remaining)
    stabilized.sort(key=lambda b: b[2] * b[3], reverse=True)
    return stabilized[:2]


def _decode_res10_dnn_detections(
    detections: np.ndarray,
    width: int,
    height: int,
    confidence_threshold: float,
) -> list[tuple[int, int, int, int]]:
    """Decode OpenCV Res10 SSD detections into [x, y, w, h] boxes."""
    boxes: list[tuple[int, int, int, int]] = []
    if detections.ndim != 4 or detections.shape[-1] < 7:
        return boxes
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue
        x0 = int(detections[0, 0, i, 3] * width)
        y0 = int(detections[0, 0, i, 4] * height)
        x1 = int(detections[0, 0, i, 5] * width)
        y1 = int(detections[0, 0, i, 6] * height)
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        boxes.append((x0, y0, bw, bh))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:2]


def _select_face_detection_roi(
    image_rgb: np.ndarray,
    restrict_to_upper_quadrants: bool,
    manual_roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Return ROI image and (x, y) offset in full-frame coordinates."""
    if manual_roi is not None:
        h, w = image_rgb.shape[:2]
        x, y, rw, rh = manual_roi
        x0 = max(0, min(w - 1, int(x)))
        y0 = max(0, min(h - 1, int(y)))
        x1 = max(x0 + 1, min(w, int(x + rw)))
        y1 = max(y0 + 1, min(h, int(y + rh)))
        return image_rgb[y0:y1, x0:x1], x0, y0
    if not restrict_to_upper_quadrants:
        return image_rgb, 0, 0
    h = image_rgb.shape[0]
    top_half = max(1, h // 2)
    return image_rgb[:top_half, :], 0, 0


def _offset_boxes(
    boxes: list[tuple[int, int, int, int]],
    offset_x: int,
    offset_y: int,
) -> list[tuple[int, int, int, int]]:
    """Map ROI-local boxes back to full-frame coordinates."""
    if offset_x == 0 and offset_y == 0:
        return boxes
    return [(x + offset_x, y + offset_y, w, h) for x, y, w, h in boxes]


class VideoFrameLabel(QLabel):  # type: ignore[misc]
    """Label with optional drag-to-select ROI support (up to 2 ROIs)."""

    roiSelected = Signal(int, int, int, int, int)  # index, x, y, w, h

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._roi_edit_enabled = False
        self._rois: list[QRect] = []  # up to 2 ROI rectangles
        self._dragging_roi_idx: int = -1
        self._drag_origin = QPoint()
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)

    def set_roi_edit_enabled(self, enabled: bool) -> None:
        self._roi_edit_enabled = bool(enabled)
        self.setCursor(Qt.CursorShape.CrossCursor if self._roi_edit_enabled else Qt.CursorShape.ArrowCursor)
        if not self._roi_edit_enabled:
            self._rubber_band.hide()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        if self._roi_edit_enabled and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            # Check if clicking inside an existing ROI
            clicked_idx = -1
            for idx, roi in enumerate(self._rois):
                if roi.contains(pos):
                    clicked_idx = idx
                    break
            # If clicked inside an ROI, replace it; otherwise start new ROI (max 2)
            if clicked_idx >= 0:
                self._dragging_roi_idx = clicked_idx
            else:
                self._dragging_roi_idx = len(self._rois) if len(self._rois) < 2 else 0
            self._drag_origin = pos
            self._rubber_band.setGeometry(QRect(self._drag_origin, self._drag_origin))
            self._rubber_band.show()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        if self._roi_edit_enabled and not self._drag_origin.isNull():
            current = event.position().toPoint()
            self._rubber_band.setGeometry(QRect(self._drag_origin, current).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        if self._roi_edit_enabled and event.button() == Qt.MouseButton.LeftButton:
            rect = self._rubber_band.geometry().normalized()
            self._rubber_band.hide()
            if rect.width() >= 4 and rect.height() >= 4:
                idx = self._dragging_roi_idx
                self.roiSelected.emit(idx, rect.x(), rect.y(), rect.width(), rect.height())
                # Update _rois list
                while len(self._rois) <= idx:
                    self._rois.append(QRect())
                self._rois[idx] = rect
            self._drag_origin = QPoint()
            self._dragging_roi_idx = -1
            event.accept()
            return
        super().mouseReleaseEvent(event)


class VideoPanel(QWidget):  # type: ignore[misc]
    """A single video file rendered into a QLabel, driven by MasterClock."""

    def __init__(
        self,
        path: Path,
        clock: MasterClock,
        title: str = "",
        flip_180: bool | None = None,
        enable_face_detection: bool = True,
        show_face_section: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.path = Path(path)
        self.clock = clock
        self._flip_180 = _should_flip_180(title or self.path.stem) if flip_180 is None else bool(flip_180)

        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._time_base = float(self._stream.time_base or 0) or 1.0 / 1000.0
        self._duration = float(self._stream.duration or 0) * self._time_base
        if self._duration <= 0 and self._container.duration:
            self._duration = self._container.duration / 1_000_000.0

        self._last_frame_t = -1.0
        self._iter = None  # lazy decoder iterator
        self._face_boxes: list[tuple[int, int, int, int]] = []
        self._face_frame_counter = 0
        self._face_detector = None
        self._face_mode = "auto"  # auto | haar | dnn | off
        self._face_overlay = False
        self._face_interval = 2
        self._face_min_neighbors = 3
        self._face_min_size = 24
        self._face_conf_threshold = 0.65
        self._face_hold_frames = 4
        self._face_miss_streak = 0
        self._face_dnn_res10 = None
        self._face_yunet = None
        self._active_face_backend = "Off"
        self._manual_face_rois: list[tuple[int, int, int, int]] = []  # up to 2 ROIs as (x, y, w, h)
        self._roi_edit_enabled = False
        self._last_frame_wh: tuple[int, int] = (0, 0)
        self._last_draw_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._time_offset: float = 0.0  # seconds offset for this video
        self._show_face_section = bool(show_face_section)
        self._face_detection_allowed = bool(enable_face_detection and self._show_face_section)
        self._face_preview_scale = 1.0

        self._label = VideoFrameLabel(title or self.path.name)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(240, 135)
        self._label.setStyleSheet("background-color: #111; color: #aaa;")
        self._label.roiSelected.connect(self._on_label_roi_selected)  # (index, x, y, w, h)

        self._face_labels: list[QLabel] = []
        face_row = QHBoxLayout()
        face_row.setContentsMargins(0, 0, 0, 0)
        face_row.setSpacing(4)
        if self._show_face_section:
            for i in range(2):
                face_label = QLabel(f"Face {i + 1}")
                face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                face_label.setMinimumSize(110, 70)
                face_label.setStyleSheet(
                    "background-color: #0d1117; color: #666; border: 1px solid #2b3137;"
                )
                self._face_labels.append(face_label)
                face_row.addWidget(face_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._label)
        if self._show_face_section:
            layout.addLayout(face_row)

        if cv2 is not None:
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            detector = cv2.CascadeClassifier(cascade_path)
            if detector is not None and not detector.empty():
                self._face_detector = detector
            self._face_dnn_res10 = self._create_res10_detector()
            self._face_yunet = self._create_yunet_detector()

        # Prime the clock's duration if no audio has set it yet.
        if self._duration > clock.duration:
            clock.set_duration(self._duration)

        self._apply_face_preview_scale()
        self._seek(0.0)

    def _apply_face_preview_scale(self) -> None:
        base_w, base_h = (110, 70)
        w = max(60, int(base_w * self._face_preview_scale))
        h = max(40, int(base_h * self._face_preview_scale))
        for label in self._face_labels:
            label.setMinimumSize(w, h)

    def set_face_preview_scale(self, scale: float) -> None:
        self._face_preview_scale = max(0.6, min(2.0, float(scale)))
        self._apply_face_preview_scale()

    def set_face_enabled(self, enabled: bool) -> None:
        if not self._face_detection_allowed:
            self._face_mode = "off"
            self._active_face_backend = "Off"
            return
        self._face_mode = "auto" if enabled else "off"
        if not enabled:
            self._active_face_backend = "Off"

    def set_face_mode(self, mode: str) -> None:
        if not self._face_detection_allowed:
            self._face_mode = "off"
            self._active_face_backend = "Off"
            return
        mode_norm = (mode or "auto").strip().lower()
        if mode_norm not in {"auto", "haar", "dnn", "off"}:
            mode_norm = "auto"
        self._face_mode = mode_norm
        if mode_norm == "off":
            self._active_face_backend = "Off"

    def set_face_overlay(self, enabled: bool) -> None:
        self._face_overlay = bool(enabled)

    def set_face_sensitivity(self, value: int) -> None:
        # value expected in [1..100]: higher -> more sensitive
        v = max(1, min(100, int(value)))
        self._face_min_neighbors = max(2, 8 - int(v / 16))
        self._face_min_size = max(14, 54 - int(v * 0.4))
        self._face_interval = 1 if v >= 70 else (2 if v >= 40 else 3)
        # Higher sensitivity should accept lower detector confidence.
        self._face_conf_threshold = 0.75 - (0.35 * ((v - 1) / 99.0))

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def active_face_backend(self) -> str:
        return self._active_face_backend

    def set_roi_edit_enabled(self, enabled: bool) -> None:
        self._roi_edit_enabled = bool(enabled)
        self._label.set_roi_edit_enabled(enabled)

    def set_time_offset(self, offset_seconds: float) -> None:
        """Set time offset for this video feed (in seconds)."""
        self._time_offset = float(offset_seconds)

    def clear_manual_face_roi(self) -> None:
        self._manual_face_rois.clear()

    @property
    def has_manual_face_roi(self) -> bool:
        return len(self._manual_face_rois) > 0

    def _on_label_roi_selected(self, idx: int, x: int, y: int, w: int, h: int) -> None:
        frame_w, frame_h = self._last_frame_wh
        draw_x, draw_y, draw_w, draw_h = self._last_draw_rect
        if frame_w <= 0 or frame_h <= 0 or draw_w <= 0 or draw_h <= 0:
            return

        sel_x0 = max(draw_x, x)
        sel_y0 = max(draw_y, y)
        sel_x1 = min(draw_x + draw_w, x + w)
        sel_y1 = min(draw_y + draw_h, y + h)
        if sel_x1 - sel_x0 < 2 or sel_y1 - sel_y0 < 2:
            return

        rel_x0 = (sel_x0 - draw_x) / float(draw_w)
        rel_y0 = (sel_y0 - draw_y) / float(draw_h)
        rel_x1 = (sel_x1 - draw_x) / float(draw_w)
        rel_y1 = (sel_y1 - draw_y) / float(draw_h)

        fx0 = int(round(rel_x0 * frame_w))
        fy0 = int(round(rel_y0 * frame_h))
        fx1 = int(round(rel_x1 * frame_w))
        fy1 = int(round(rel_y1 * frame_h))
        fx0 = max(0, min(frame_w - 1, fx0))
        fy0 = max(0, min(frame_h - 1, fy0))
        fx1 = max(fx0 + 1, min(frame_w, fx1))
        fy1 = max(fy0 + 1, min(frame_h, fy1))
        roi = (fx0, fy0, fx1 - fx0, fy1 - fy0)
        # Expand list if needed and set ROI at index
        while len(self._manual_face_rois) <= idx:
            self._manual_face_rois.append((0, 0, 0, 0))
        self._manual_face_rois[idx] = roi

    def _seek(self, seconds: float) -> None:
        """Seek to <= seconds and rebuild the decoder iterator."""
        try:
            pts = max(0, int(seconds / self._time_base))
            self._container.seek(pts, any_frame=False, backward=True, stream=self._stream)
        except av.AVError:
            return
        self._iter = self._container.decode(self._stream)
        self._last_frame_t = -1.0

    def render_at(self, t_seconds: float) -> None:
        """Advance decode until a frame at or after t_seconds, then display it."""
        # Apply time offset for this feed
        adjusted_t = max(0.0, t_seconds + self._time_offset)
        if self._iter is None:
            self._seek(adjusted_t)

        # Large backward scrub or skip >1s forward → seek first.
        if self._last_frame_t >= 0 and (
            adjusted_t < self._last_frame_t - 0.05 or adjusted_t > self._last_frame_t + 1.0
        ):
            self._seek(adjusted_t)

        frame = None
        assert self._iter is not None
        try:
            for f in self._iter:
                if f.pts is None:
                    continue
                ft = float(f.pts * self._time_base)
                if ft >= adjusted_t:
                    frame = f
                    self._last_frame_t = ft
                    break
                self._last_frame_t = ft
        except (StopIteration, av.AVError):
            return

        if frame is None:
            return

        img = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
        if self._flip_180:
            img = _flip_frame_180(img)
        h, w, _ = img.shape
        self._last_frame_wh = (w, h)

        label_w = max(1, self._label.width())
        label_h = max(1, self._label.height())
        scale = min(label_w / float(w), label_h / float(h))
        draw_w = max(1, int(w * scale))
        draw_h = max(1, int(h * scale))
        draw_x = (label_w - draw_w) // 2
        draw_y = (label_h - draw_h) // 2
        self._last_draw_rect = (draw_x, draw_y, draw_w, draw_h)

        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_RGB888).copy()
        draw_img = img
        need_redraw = False
        # Draw manual ROI rectangles
        if self._manual_face_rois and cv2 is not None:
            draw_img = img.copy()
            need_redraw = True
            colors = [(120, 180, 255), (255, 165, 0)]  # blue, orange
            for i, (rx, ry, rw, rh) in enumerate(self._manual_face_rois):
                color = colors[i % len(colors)]
                cv2.rectangle(draw_img, (rx, ry), (rx + rw, ry + rh), color, 2)
        # Draw face overlay
        if self._face_overlay and self._face_boxes:
            if not need_redraw:
                draw_img = img.copy()
                need_redraw = True
            self._draw_face_overlay(draw_img)
        # Update qimg if anything was drawn
        if need_redraw:
            qimg = QImage(
                draw_img.data,
                w,
                h,
                draw_img.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()

        pix = QPixmap.fromImage(qimg).scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pix)
        if self._show_face_section:
            try:
                self._update_face_views(img)
            except Exception:
                # Never break main playback if face crop rendering fails.
                pass

    def _update_face_views(self, img_rgb) -> None:
        if not self._face_detection_allowed:
            self._face_mode = "off"
            self._face_boxes = []
            self._face_miss_streak = 0
            self._active_face_backend = "Off"
            for idx, label in enumerate(self._face_labels):
                label.clear()
                label.setText(f"Face {idx + 1}")
            return
        self._face_frame_counter += 1
        if self._face_mode != "off":
            if self._face_frame_counter % self._face_interval == 0 or not self._face_boxes:
                detected = self._detect_faces(img_rgb)
                if detected:
                    self._face_boxes = _stabilize_face_boxes(self._face_boxes, detected)
                    self._face_miss_streak = 0
                else:
                    self._face_miss_streak += 1
                    if self._face_miss_streak > self._face_hold_frames:
                        self._face_boxes = []
        else:
            self._face_boxes = []
            self._face_miss_streak = 0

        for idx, label in enumerate(self._face_labels):
            # Check if we have a manual ROI for this face slot
            has_roi = idx < len(self._manual_face_rois)
            roi_box = self._manual_face_rois[idx] if has_roi else None

            # Check if we have a detected face at this slot
            has_detected = idx < len(self._face_boxes)
            detected_box = self._face_boxes[idx] if has_detected else None

            # Decide what crop to show
            crop_box = None
            if has_detected and detected_box:
                # Check if detected face is inside the ROI (if ROI exists)
                if roi_box:
                    roi_x, roi_y, roi_w, roi_h = roi_box
                    det_cx = detected_box[0] + detected_box[2] // 2
                    det_cy = detected_box[1] + detected_box[3] // 2
                    if roi_x <= det_cx < roi_x + roi_w and roi_y <= det_cy < roi_y + roi_h:
                        crop_box = detected_box
                else:
                    crop_box = detected_box
            elif roi_box:
                # No detected face, but ROI exists: show ROI crop
                crop_box = roi_box

            if crop_box is None:
                label.clear()
                label.setText(f"Face {idx + 1}")
                continue

            x, y, w, h = crop_box
            x_pad = int(w * 0.15)
            y_pad = int(h * 0.2)
            x0 = max(0, x - x_pad)
            y0 = max(0, y - y_pad)
            x1 = min(img_rgb.shape[1], x + w + x_pad)
            y1 = min(img_rgb.shape[0], y + h + y_pad)
            crop = img_rgb[y0:y1, x0:x1]
            if crop.size == 0:
                label.clear()
                label.setText(f"Face {idx + 1}")
                continue

            crop = np.ascontiguousarray(crop)
            ch, cw, _ = crop.shape
            qimg = QImage(crop.data, cw, ch, crop.strides[0], QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(pix)

    def _detect_faces(self, img_rgb) -> list[tuple[int, int, int, int]]:
        if cv2 is None:
            self._active_face_backend = "Unavailable"
            return []

        mode = self._face_mode
        if mode == "off":
            self._active_face_backend = "Off"
            return []

        # If 2 ROIs defined, detect separately in each and collect results.
        if len(self._manual_face_rois) == 2:
            all_boxes: list[tuple[int, int, int, int]] = []
            for roi_tuple in self._manual_face_rois:
                roi_img, roi_x, roi_y = _select_face_detection_roi(
                    img_rgb,
                    restrict_to_upper_quadrants=self._flip_180,
                    manual_roi=roi_tuple,
                )
                boxes = self._detect_in_roi(roi_img, mode)
                all_boxes.extend(_offset_boxes(boxes, roi_x, roi_y))
            return all_boxes[:2]

        # Single ROI or no ROI: use existing logic
        manual_roi = self._manual_face_rois[0] if len(self._manual_face_rois) == 1 else None
        roi_img, roi_x, roi_y = _select_face_detection_roi(
            img_rgb,
            restrict_to_upper_quadrants=self._flip_180,
            manual_roi=manual_roi,
        )

        boxes = self._detect_in_roi(roi_img, mode)
        return _offset_boxes(boxes, roi_x, roi_y)

    def _detect_in_roi(self, roi_img, mode: str) -> list[tuple[int, int, int, int]]:
        """Detect faces in a ROI image. Returns boxes in ROI-local coordinates."""
        if mode in {"auto", "dnn"} and self._face_dnn_res10 is not None:
            self._active_face_backend = "Res10"
            strong_boxes = self._detect_faces_res10(roi_img)
            if strong_boxes:
                return strong_boxes

        if mode in {"auto", "dnn"} and self._face_yunet is not None:
            self._active_face_backend = "YuNet"
            dnn_boxes = self._detect_faces_yunet(roi_img)
            if dnn_boxes:
                return dnn_boxes

        # Resilient fallback: if DNN model is unavailable or transiently misses,
        # fall back to Haar so face views stay usable during annotation.
        if mode == "dnn" and self._face_detector is None:
            self._active_face_backend = "Unavailable"
            return []

        if self._face_detector is None:
            self._active_face_backend = "Unavailable"
            return []

        self._active_face_backend = "Haar"

        h, w, _ = roi_img.shape
        gray = cv2.cvtColor(roi_img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        scale = 1.0
        target_w = 640
        if w > target_w:
            scale = target_w / float(w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=self._face_min_neighbors,
            minSize=(self._face_min_size, self._face_min_size),
        )
        if faces is None or len(faces) == 0:
            return []

        boxes: list[tuple[int, int, int, int]] = []
        inv = 1.0 / scale
        for x, y, fw, fh in faces:
            boxes.append((int(x * inv), int(y * inv), int(fw * inv), int(fh * inv)))

        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        return boxes[:2]

    def _create_res10_detector(self):
        if cv2 is None:
            return None
        candidates = [
            (
                Path("configs") / "models" / "face" / "deploy.prototxt",
                Path("configs") / "models" / "face" / "res10_300x300_ssd_iter_140000_fp16.caffemodel",
            ),
            (
                Path("models") / "face" / "deploy.prototxt",
                Path("models") / "face" / "res10_300x300_ssd_iter_140000_fp16.caffemodel",
            ),
        ]
        root = Path.cwd()
        for proto_rel, model_rel in candidates:
            proto = (root / proto_rel).resolve()
            model = (root / model_rel).resolve()
            if not proto.is_file() or not model.is_file():
                continue
            try:
                return cv2.dnn.readNetFromCaffe(str(proto), str(model))
            except Exception:
                continue
        return None

    def _detect_faces_res10(self, img_rgb) -> list[tuple[int, int, int, int]]:
        if cv2 is None or self._face_dnn_res10 is None:
            return []
        try:
            h, w, _ = img_rgb.shape
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            blob = cv2.dnn.blobFromImage(
                cv2.resize(bgr, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0),
            )
            self._face_dnn_res10.setInput(blob)
            detections = self._face_dnn_res10.forward()
            return _decode_res10_dnn_detections(detections, w, h, self._face_conf_threshold)
        except Exception:
            return []

    def _create_yunet_detector(self):
        if cv2 is None:
            return None
        if not hasattr(cv2, "FaceDetectorYN"):
            return None

        candidates = [
            Path(cv2.data.haarcascades).parent / "face_detection_yunet_2023mar.onnx",
            Path(cv2.data.haarcascades).parent / "face_detection_yunet_2022mar.onnx",
        ]
        model_path = next((p for p in candidates if p.is_file()), None)
        if model_path is None:
            return None
        try:
            return cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.75, 0.3, 5000)
        except Exception:
            return None

    def _detect_faces_yunet(self, img_rgb) -> list[tuple[int, int, int, int]]:
        if cv2 is None or self._face_yunet is None:
            return []
        try:
            h, w, _ = img_rgb.shape
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            self._face_yunet.setInputSize((w, h))
            _, faces = self._face_yunet.detect(bgr)
            if faces is None or len(faces) == 0:
                return []

            boxes: list[tuple[int, int, int, int]] = []
            for row in faces:
                x, y, fw, fh = row[:4]
                score = float(row[14]) if len(row) > 14 else 1.0
                if score < self._face_conf_threshold:
                    continue
                boxes.append((int(x), int(y), int(fw), int(fh)))
            boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            return boxes[:2]
        except Exception:
            return []

    def _draw_face_overlay(self, img_rgb) -> None:
        if cv2 is None:
            return
        for i, (x, y, w, h) in enumerate(self._face_boxes[:2]):
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (80, 255, 120), 2)
            cv2.putText(
                img_rgb,
                f"Face {i + 1}",
                (x, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 255, 120),
                1,
                cv2.LINE_AA,
            )

    def close(self) -> None:
        try:
            self._container.close()
        except Exception:
            pass
