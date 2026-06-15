"""Main window for the annotation GUI (M1 + M2: playback, tiers, annotations)."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .io import annotations as anns
from .io.sync import (
    SyncOffsets,
    infer_sync_offsets_from_lsl_sync,
    infer_sync_offsets_from_source_ffmpeg_lsl,
    load_sync,
    load_task_start_lsl,
    save_sync,
)
from .media.audio_engine import AudioEngine
from .media.clock import MasterClock
from .media.video_engine import VideoPanel, _is_cam5_feed, _is_p1_p4_feed, _is_tobii_feed
from .session_loader import TaskRun, discover_sessions, discover_task_runs, load_participant_map
from .widgets.tier_panel import TierPanel
from .widgets.timeline import TimelineWidget
from .widgets.waveform import WaveformWidget


class MainWindow(QMainWindow):
    def __init__(self, bids_root: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("AffectAI Annotation GUI")
        self.resize(1600, 1000)

        self.bids_root: Path | None = bids_root
        self.clock = MasterClock()
        self.audio = AudioEngine(clock=self.clock)
        self.video_panels: list[VideoPanel] = []
        self.video_checkboxes: list[QCheckBox] = []
        self._video_cols = 3
        self.channel_controls: list[QCheckBox] = []
        self._task_runs: list[TaskRun] = []
        self._current_task: TaskRun | None = None
        self._doc: anns.AnnotationDoc | None = None
        self._doc_path: Path | None = None
        self._transcript_path: Path | None = None
        self._pending_start: float | None = None  # set by "I" key
        self._splitter_h: QSplitter | None = None
        self._splitter_v: QSplitter | None = None
        self._sync_offsets: SyncOffsets | None = None
        self._sync_spinboxes: dict[str, QDoubleSpinBox] = {}
        self._sync_transcript_spinbox: QDoubleSpinBox | None = None
        self._sync_row_host: QWidget | None = None
        self._sync_video_lock_box: QCheckBox | None = None
        self._sync_audio_lock_box: QCheckBox | None = None
        self._sync_updating: bool = False
        self._segment_rows: list[dict[str, object]] = []
        self._word_rows: list[dict[str, object]] = []
        self._audio_annot_master_transcript_path: Path | None = None
        self._audio_annot_master_words_path: Path | None = None
        self._transcript_table_updating: bool = False
        self._active_word_rows: set[int] = set()
        self._active_segment_rows: set[int] = set()

        self._build_ui()
        self._build_menu()
        self._build_shortcuts()

        self._tick = QTimer(self)
        self._tick.setInterval(33)  # ~30 Hz
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

        if self.bids_root:
            self._populate_sessions()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top row — pickers
        top = QHBoxLayout()
        self.root_label = QLabel("No BIDS root")
        self.root_label.setStyleSheet("color: #888;")
        self.session_box = QComboBox()
        self.session_box.currentIndexChanged.connect(self._on_session_changed)
        self.task_box = QComboBox()
        self.load_btn = QPushButton("Load task")
        self.load_btn.clicked.connect(self._load_current_task)

        self.layout_target_box = QComboBox()
        self.layout_target_box.addItems(["Video", "Side", "Timeline", "Waveform"])
        self.layout_grow_btn = QPushButton("Section +")
        self.layout_grow_btn.clicked.connect(self._grow_selected_section)
        self.layout_shrink_btn = QPushButton("Section -")
        self.layout_shrink_btn.clicked.connect(self._shrink_selected_section)
        self.layout_reset_btn = QPushButton("Reset Layout")
        self.layout_reset_btn.clicked.connect(self._reset_layout_sizes)

        top.addWidget(QLabel("Session:"))
        top.addWidget(self.session_box, 2)
        top.addWidget(QLabel("Task:"))
        top.addWidget(self.task_box, 1)
        top.addWidget(self.load_btn)
        top.addWidget(QLabel("Resize:"))
        top.addWidget(self.layout_target_box)
        top.addWidget(self.layout_grow_btn)
        top.addWidget(self.layout_shrink_btn)
        top.addWidget(self.layout_reset_btn)
        top.addStretch(1)
        top.addWidget(self.root_label)
        root.addLayout(top)

        # Camera show/hide row
        self.camera_row = QHBoxLayout()
        self.camera_row.addWidget(QLabel("Cameras:"))
        self.face_enable_box = QCheckBox("Faces")
        self.face_enable_box.setChecked(True)
        self.face_enable_box.toggled.connect(self._apply_face_controls)
        self.camera_row.addWidget(self.face_enable_box)

        self.face_mode_box = QComboBox()
        self.face_mode_box.addItems(["Auto", "Haar", "DNN", "Off"])
        self.face_mode_box.currentIndexChanged.connect(self._apply_face_controls)
        self.camera_row.addWidget(QLabel("Mode:"))
        self.camera_row.addWidget(self.face_mode_box)

        self.face_sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.face_sensitivity.setRange(1, 100)
        self.face_sensitivity.setValue(70)
        self.face_sensitivity.setFixedWidth(130)
        self.face_sensitivity.valueChanged.connect(self._apply_face_controls)
        self.camera_row.addWidget(QLabel("Sensitivity:"))
        self.camera_row.addWidget(self.face_sensitivity)

        self.face_overlay_box = QCheckBox("Overlay")
        self.face_overlay_box.setChecked(False)
        self.face_overlay_box.toggled.connect(self._apply_face_controls)
        self.camera_row.addWidget(self.face_overlay_box)

        self.face_backend_label = QLabel("Detector: Off")
        self.face_backend_label.setStyleSheet("color: #888;")
        self.camera_row.addWidget(self.face_backend_label)

        self.face_roi_edit_box = QCheckBox("ROI Edit")
        self.face_roi_edit_box.setChecked(False)
        self.face_roi_edit_box.toggled.connect(self._apply_face_controls)
        self.camera_row.addWidget(self.face_roi_edit_box)

        self.face_roi_clear_btn = QPushButton("Clear ROI")
        self.face_roi_clear_btn.clicked.connect(self._clear_face_rois)
        self.camera_row.addWidget(self.face_roi_clear_btn)

        self.face_roi_hint = QLabel("Drag on each feed")
        self.face_roi_hint.setStyleSheet("color: #888;")
        self.camera_row.addWidget(self.face_roi_hint)

        self.face_preview_size = QSlider(Qt.Orientation.Horizontal)
        self.face_preview_size.setRange(70, 180)
        self.face_preview_size.setValue(100)
        self.face_preview_size.setFixedWidth(110)
        self.face_preview_size.valueChanged.connect(self._apply_face_controls)
        self.camera_row.addWidget(QLabel("Face Size:"))
        self.camera_row.addWidget(self.face_preview_size)

        self.feed_move_box = QComboBox()
        self.feed_move_box.setMinimumWidth(180)
        self.move_feed_left_btn = QPushButton("←")
        self.move_feed_left_btn.clicked.connect(lambda: self._move_selected_feed(-1))
        self.move_feed_right_btn = QPushButton("→")
        self.move_feed_right_btn.clicked.connect(lambda: self._move_selected_feed(1))
        self.move_feed_up_btn = QPushButton("↑")
        self.move_feed_up_btn.clicked.connect(lambda: self._move_selected_feed(-self._video_cols))
        self.move_feed_down_btn = QPushButton("↓")
        self.move_feed_down_btn.clicked.connect(lambda: self._move_selected_feed(self._video_cols))
        self.camera_row.addWidget(QLabel("Move feed:"))
        self.camera_row.addWidget(self.feed_move_box)
        self.camera_row.addWidget(self.move_feed_left_btn)
        self.camera_row.addWidget(self.move_feed_right_btn)
        self.camera_row.addWidget(self.move_feed_up_btn)
        self.camera_row.addWidget(self.move_feed_down_btn)

        self.camera_row_host = QWidget()
        self.camera_row_host.setLayout(self.camera_row)
        root.addWidget(self.camera_row_host)

        # Sync offsets row (will be populated on task load)
        self.sync_row = QHBoxLayout()
        self.sync_row.addWidget(QLabel("Sync (s):"))
        self.sync_save_btn = QPushButton("Save Sync")
        self.sync_save_btn.clicked.connect(self._save_sync_offsets)
        self.sync_row_host = QWidget()
        self.sync_row_host.setLayout(self.sync_row)
        root.addWidget(self.sync_row_host)
        # Requested: hide re-sync controls for now.
        self.sync_row_host.setVisible(False)

        # Main workspace — videos | tier panel, split vertically with timeline below.
        splitter_h = QSplitter(Qt.Orientation.Horizontal)
        self._splitter_h = splitter_h
        self.video_grid = QGridLayout()
        self.video_grid.setContentsMargins(0, 0, 0, 0)
        video_host = QWidget()
        video_layout = QVBoxLayout(video_host)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_grid_host = QWidget()
        video_grid_host.setLayout(self.video_grid)
        video_layout.addWidget(video_grid_host, 1)

        self.speaker_strip_label = QLabel("Now speaking: -")
        self.speaker_strip_label.setStyleSheet("padding:4px 8px; background:#1f2937; color:#e5e7eb;")
        self.word_strip_label = QLabel("Word: -")
        self.word_strip_label.setStyleSheet("padding:4px 8px; background:#111827; color:#cbd5e1;")
        speaker_strip = QHBoxLayout()
        speaker_strip.setContentsMargins(0, 0, 0, 0)
        speaker_strip.addWidget(self.speaker_strip_label, 1)
        speaker_strip.addWidget(self.word_strip_label, 2)
        speaker_strip_host = QWidget()
        speaker_strip_host.setLayout(speaker_strip)
        video_layout.addWidget(speaker_strip_host)

        transcript_header = QHBoxLayout()
        transcript_header.addWidget(QLabel("Transcript"))
        self.load_transcript_btn = QPushButton("Load")
        self.load_transcript_btn.clicked.connect(self._load_transcript_dialog)
        self.save_transcript_btn = QPushButton("Save")
        self.save_transcript_btn.clicked.connect(self._save_transcript)
        transcript_header.addWidget(self.load_transcript_btn)
        transcript_header.addWidget(self.save_transcript_btn)

        transcript_events = QHBoxLayout()
        transcript_events.addWidget(QLabel("Add after current:"))
        self.transcript_event_category = QComboBox()
        self.transcript_event_category.addItems(
            [
                "Pause",
                "Scilence",
                "Breath",
                "Filled Pause",
                "Laughter",
                "Backchannel",
                "Overlap",
                "Noise",
            ]
        )
        transcript_events.addWidget(self.transcript_event_category)
        self.transcript_event_speaker = QComboBox()
        self.transcript_event_speaker.addItems(["P1", "P2", "P3", "P4", "Moderator"])
        transcript_events.addWidget(self.transcript_event_speaker)
        self.transcript_event_text = QLineEdit()
        self.transcript_event_text.setPlaceholderText("text / note")
        transcript_events.addWidget(self.transcript_event_text, 1)
        self.transcript_event_add_btn = QPushButton("Insert")
        self.transcript_event_add_btn.clicked.connect(self._insert_transcript_line_after_current)
        transcript_events.addWidget(self.transcript_event_add_btn)

        self.transcript_path_label = QLabel("No transcript loaded")
        self.transcript_path_label.setStyleSheet("color: #888;")

        self.transcript_editor = QTextEdit()
        self.transcript_editor.setPlaceholderText("Live transcript view. Segment/word tables below are editable.")
        self.transcript_editor.setVisible(False)

        transcript_tables_header = QHBoxLayout()
        transcript_tables_header.addWidget(QLabel("Transcript Rows (editable speaker/text/category)"))
        self.save_audio_annot_btn = QPushButton("Save Rows")
        self.save_audio_annot_btn.clicked.connect(self._save_audio_annot_edits)
        transcript_tables_header.addWidget(self.save_audio_annot_btn)

        self.transcript_segments_table = QTableWidget(0, 5)
        self.transcript_segments_table.setHorizontalHeaderLabels(["Time", "Speaker", "Mic", "Category", "Text"])
        self.transcript_segments_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.transcript_segments_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.transcript_segments_table.itemChanged.connect(self._on_segment_table_item_changed)

        self.transcript_words_table = QTableWidget(0, 4)
        self.transcript_words_table.setHorizontalHeaderLabels(["Time", "Speaker", "Mic", "Word"])
        self.transcript_words_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.transcript_words_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.transcript_words_table.itemChanged.connect(self._on_word_table_item_changed)
        # Requested: hide master words table for now.
        self.transcript_words_table.setVisible(False)

        transcript_host = QWidget()
        transcript_layout = QVBoxLayout(transcript_host)
        transcript_layout.setContentsMargins(0, 0, 0, 0)
        transcript_layout.addLayout(transcript_header)
        transcript_layout.addLayout(transcript_events)
        transcript_layout.addWidget(self.transcript_path_label)
        transcript_layout.addWidget(self.transcript_editor, 1)
        transcript_layout.addLayout(transcript_tables_header)
        transcript_layout.addWidget(self.transcript_segments_table, 1)
        transcript_layout.addWidget(self.transcript_words_table, 1)
        video_layout.addWidget(transcript_host, 2)
        splitter_h.addWidget(video_host)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.tier_panel = TierPanel()
        self.tier_panel.tiersMutated.connect(self._on_doc_mutated)
        # Requested: transcript panel is under videos; hide tier panel area for now.
        self.tier_panel.setVisible(False)
        right_layout.addWidget(QLabel("Annotation tier panel hidden"), 1)

        splitter_h.addWidget(right_panel)
        splitter_h.setStretchFactor(0, 4)
        splitter_h.setStretchFactor(1, 1)

        splitter_v = QSplitter(Qt.Orientation.Vertical)
        self._splitter_v = splitter_v
        splitter_v.addWidget(splitter_h)

        self.timeline = TimelineWidget()
        self.timeline.seekRequested.connect(self.clock.seek)
        self.timeline.entryDoubleClicked.connect(self._edit_entry)
        splitter_v.addWidget(self.timeline)

        self.waveform = WaveformWidget()
        self.waveform.seekRequested.connect(self.clock.seek)
        splitter_v.addWidget(self.waveform)
        splitter_v.setStretchFactor(0, 3)
        splitter_v.setStretchFactor(1, 1)
        splitter_v.setStretchFactor(2, 1)
        root.addWidget(splitter_v, 1)
        self._reset_layout_sizes()

        # Audio channel row
        self.channel_row = QHBoxLayout()
        self.channel_row.addWidget(QLabel("Mics:"))
        self.wave_zoom_out_btn = QPushButton("Wave -")
        self.wave_zoom_out_btn.clicked.connect(self.waveform.zoom_out_x)
        self.wave_zoom_in_btn = QPushButton("Wave +")
        self.wave_zoom_in_btn.clicked.connect(self.waveform.zoom_in_x)
        self.wave_zoom_reset_btn = QPushButton("Wave 1:1")
        self.wave_zoom_reset_btn.clicked.connect(self.waveform.reset_zoom_x)
        self.channel_row.addWidget(self.wave_zoom_out_btn)
        self.channel_row.addWidget(self.wave_zoom_in_btn)
        self.channel_row.addWidget(self.wave_zoom_reset_btn)
        channel_host = QWidget()
        channel_host.setLayout(self.channel_row)
        root.addWidget(channel_host)

        # Transport
        transport = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.play_btn.clicked.connect(self._toggle_play)
        self.pos_label = QLabel("00:00 / 00:00")
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.sliderMoved.connect(self._on_scrub)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 200)
        self.volume.setValue(100)
        self.volume.setFixedWidth(120)
        self.volume.valueChanged.connect(lambda v: self.audio.set_master_gain(v / 100.0))
        transport.addWidget(self.play_btn)
        transport.addWidget(self.scrub, 1)
        transport.addWidget(self.pos_label)
        transport.addWidget(QLabel("Vol"))
        transport.addWidget(self.volume)
        root.addLayout(transport)

        # Hint strip
        hint = QLabel(
            "Annotate:  I = mark start · O = mark end (prompts for label) · "
            "P = add point on active tier · Del = delete selected · Ctrl+S = save"
        )
        hint.setStyleSheet("color:#888; padding:2px 4px;")
        root.addWidget(hint)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("Open &BIDS root…", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._pick_root)
        file_menu.addAction(open_action)
        save_action = QAction("&Save annotations", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save_annotations)
        file_menu.addAction(save_action)
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_shortcuts(self) -> None:
        # I: mark span start at playhead.
        QShortcut(QKeySequence(Qt.Key.Key_I), self, activated=self._mark_in)
        # O: mark span end at playhead (prompts for label, commits entry).
        QShortcut(QKeySequence(Qt.Key.Key_O), self, activated=self._mark_out)
        # P: add instantaneous "point" entry at playhead.
        QShortcut(QKeySequence(Qt.Key.Key_P), self, activated=self._add_point)
        # Del: delete selected entry.
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self._delete_selected)
        # Left/Right arrows: nudge playhead.
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self._nudge(-0.1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self._nudge(0.1))
        # F: quick toggle for face detection while playback is running.
        QShortcut(QKeySequence(Qt.Key.Key_F), self, activated=self._toggle_face_detection)
        # Layout section resize shortcuts.
        QShortcut(QKeySequence("Ctrl++"), self, activated=self._grow_selected_section)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self._shrink_selected_section)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self._reset_layout_sizes)

    def _toggle_face_detection(self) -> None:
        self.face_enable_box.setChecked(not self.face_enable_box.isChecked())

    def _grow_selected_section(self) -> None:
        target = self.layout_target_box.currentText().lower()
        self._resize_section(target, delta=120)

    def _shrink_selected_section(self) -> None:
        target = self.layout_target_box.currentText().lower()
        self._resize_section(target, delta=-120)

    def _resize_section(self, target: str, delta: int) -> None:
        if self._splitter_h is None or self._splitter_v is None:
            return
        if target == "video":
            self._adjust_splitter_slot(self._splitter_h, 0, delta)
            self._adjust_splitter_slot(self._splitter_v, 0, delta)
            return
        if target == "side":
            self._adjust_splitter_slot(self._splitter_h, 1, delta)
            return
        if target == "timeline":
            self._adjust_splitter_slot(self._splitter_v, 1, delta)
            return
        if target == "waveform":
            self._adjust_splitter_slot(self._splitter_v, 2, delta)

    def _adjust_splitter_slot(self, splitter: QSplitter, index: int, delta: int) -> None:
        sizes = splitter.sizes()
        if index < 0 or index >= len(sizes):
            return
        new_sizes = sizes[:]
        new_sizes[index] = max(80, new_sizes[index] + delta)

        for i, _ in enumerate(new_sizes):
            if i == index:
                continue
            # Distribute inverse delta across other panes to keep total stable.
            adjust = int(delta / max(1, len(new_sizes) - 1))
            new_sizes[i] = max(80, new_sizes[i] - adjust)
        splitter.setSizes(new_sizes)

    def _reset_layout_sizes(self) -> None:
        if self._splitter_h is not None:
            self._splitter_h.setSizes([1200, 420])
        if self._splitter_v is not None:
            self._splitter_v.setSizes([760, 220, 220])

    # --------------------------------------------------------------- Actions
    def _pick_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select BIDS root")
        if path:
            self.bids_root = Path(path)
            self._populate_sessions()

    def _populate_sessions(self) -> None:
        if self.bids_root is None:
            return
        self.root_label.setText(str(self.bids_root))
        self.session_box.blockSignals(True)
        self.session_box.clear()
        for ses in discover_sessions(self.bids_root):
            self.session_box.addItem(f"{ses.parent.name}/{ses.name}", ses)
        self.session_box.blockSignals(False)
        self._on_session_changed()

    def _on_session_changed(self) -> None:
        self.task_box.clear()
        ses = self.session_box.currentData()
        if ses is None:
            return
        self._task_runs = discover_task_runs(ses)
        for tr in self._task_runs:
            self.task_box.addItem(
                f"task-{tr.task}_run-{tr.run}  ({len(tr.audio)}A / {len(tr.video)}V)",
                tr,
            )

    def _load_current_task(self) -> None:
        tr = self.task_box.currentData()
        if tr is None:
            QMessageBox.information(self, "No task", "Select a session with at least one task run.")
            return
        try:
            self._teardown_media()
            self._current_task = tr
            self._setup_videos(tr)
            self._setup_audio(tr)
            self._setup_annotations(tr)
            self._auto_load_transcript(tr)
            self._setup_sync_offsets(tr)
            self.scrub.setValue(0)
            self.clock.seek(0.0)
        except Exception as exc:  # surface load errors instead of crashing
            QMessageBox.critical(self, "Load failed", f"{type(exc).__name__}: {exc}")
            self._teardown_media()

    def _setup_videos(self, tr: TaskRun) -> None:
        labels = [mf.acq or mf.path.stem for mf in tr.video]
        has_cam0 = any(
            bool(re.search(r"(?:^|[^a-z0-9])(?:cam|camera)[_-]?0(?:[^a-z0-9]|$)", lbl.lower()))
            for lbl in labels
        )
        if has_cam0:
            default_mask = [
                bool(re.search(r"(?:^|[^a-z0-9])(?:cam|camera)[_-]?[01](?:[^a-z0-9]|$)", lbl.lower()))
                for lbl in labels
            ]
        else:
            # Common case in this project is 1-indexed camera naming (cam1..cam6).
            default_mask = [
                bool(re.search(r"(?:^|[^a-z0-9])(?:cam|camera)[_-]?[12](?:[^a-z0-9]|$)", lbl.lower()))
                for lbl in labels
            ]
        has_requested_defaults = any(default_mask)
        for i, mf in enumerate(tr.video):
            label = mf.acq or mf.path.stem
            is_tobii = _is_tobii_feed(label) or _is_tobii_feed(mf.path.stem)
            is_participant_feed = _is_p1_p4_feed(label) or _is_p1_p4_feed(mf.path.stem)
            panel = VideoPanel(
                mf.path,
                self.clock,
                title=label,
                enable_face_detection=not is_tobii,
                show_face_section=not is_participant_feed,
            )
            panel.set_face_sensitivity(self.face_sensitivity.value())
            self.video_panels.append(panel)
            self.video_grid.addWidget(panel, i // self._video_cols, i % self._video_cols)

            cb = QCheckBox(label)
            default_on = default_mask[i] if has_requested_defaults else i < 2
            cb.setChecked(default_on)
            panel.setVisible(default_on)
            cb.toggled.connect(lambda on, p=panel: p.setVisible(on))
            self.video_checkboxes.append(cb)
            self.camera_row.addWidget(cb)
        self._refresh_video_move_controls()
        self.camera_row.addStretch(1)
        self._apply_face_controls()

    def _refresh_video_move_controls(self) -> None:
        current = self.feed_move_box.currentIndex()
        self.feed_move_box.blockSignals(True)
        self.feed_move_box.clear()
        for idx, cb in enumerate(self.video_checkboxes):
            self.feed_move_box.addItem(f"{idx + 1}: {cb.text()}")
        if self.video_checkboxes:
            self.feed_move_box.setCurrentIndex(max(0, min(current, len(self.video_checkboxes) - 1)))
        self.feed_move_box.blockSignals(False)

    def _rebuild_video_grid(self) -> None:
        for i, panel in enumerate(self.video_panels):
            self.video_grid.addWidget(panel, i // self._video_cols, i % self._video_cols)

    def _move_selected_feed(self, delta: int) -> None:
        if not self.video_panels:
            return
        src = self.feed_move_box.currentIndex()
        if src < 0:
            return
        dst = src + delta
        if dst < 0 or dst >= len(self.video_panels):
            return

        self.video_panels[src], self.video_panels[dst] = self.video_panels[dst], self.video_panels[src]
        self.video_checkboxes[src], self.video_checkboxes[dst] = self.video_checkboxes[dst], self.video_checkboxes[src]
        self._rebuild_video_grid()
        self._refresh_video_move_controls()
        self.feed_move_box.setCurrentIndex(dst)

    def _apply_face_controls(self) -> None:
        enabled = self.face_enable_box.isChecked()
        mode = self.face_mode_box.currentText().strip().lower()
        sensitivity = self.face_sensitivity.value()
        overlay = self.face_overlay_box.isChecked()
        roi_edit = self.face_roi_edit_box.isChecked()
        self.face_mode_box.setEnabled(enabled)
        self.face_sensitivity.setEnabled(enabled)
        self.face_overlay_box.setEnabled(enabled)
        self.face_roi_clear_btn.setEnabled(bool(self.video_panels))
        self.face_roi_hint.setEnabled(roi_edit)
        preview_scale = self.face_preview_size.value() / 100.0
        for panel in self.video_panels:
            panel.set_face_enabled(enabled)
            panel.set_face_mode(mode)
            panel.set_face_sensitivity(sensitivity)
            panel.set_face_overlay(overlay)
            panel.set_roi_edit_enabled(roi_edit)
            panel.set_face_preview_scale(preview_scale)
        self._refresh_face_backend_label()

    def _clear_face_rois(self) -> None:
        for panel in self.video_panels:
            panel.clear_manual_face_roi()
        self._refresh_face_backend_label()

    def _refresh_face_backend_label(self) -> None:
        if not self.face_enable_box.isChecked():
            self.face_backend_label.setText("Detector: Off")
            return
        active = {
            panel.active_face_backend
            for panel in self.video_panels
            if panel.isVisible() and panel.active_face_backend
        }
        roi_count = sum(1 for panel in self.video_panels if panel.has_manual_face_roi)
        if not active:
            self.face_backend_label.setText(f"Detector: Waiting  ROI:{roi_count}")
            return
        self.face_backend_label.setText(f"Detector: {'/'.join(sorted(active))}  ROI:{roi_count}")

    def _setup_audio(self, tr: TaskRun) -> None:
        while self.channel_row.count() > 4:  # keep "Mics:" + wave zoom controls
            item = self.channel_row.takeAt(4)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.channel_controls.clear()
        self.audio.channels.clear()

        for mf in tr.audio:
            display_name = _display_audio_name(mf.acq or mf.path.stem)
            ch = self.audio.add_wav(mf.path, name=display_name)
            cb = QCheckBox(_format_mic_label(ch))
            cb.setChecked(True)
            idx = len(self.channel_controls)

            def _on_toggle(on: bool, i: int = idx) -> None:
                self.audio.set_mute(i, not on)
                self._refresh_waveform()

            cb.toggled.connect(_on_toggle)
            self.channel_controls.append(cb)
            self.channel_row.addWidget(cb)
        self.channel_row.addStretch(1)
        self._refresh_waveform()

        if self.audio.channels:
            try:
                self.audio.start()
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Audio device error",
                    f"Could not open audio output: {type(exc).__name__}: {exc}\n"
                    "Video will still play. Check the default Windows output device.",
                )

    def _refresh_waveform(self) -> None:
        selected_channels: list[object] = []
        for channel, control in zip(self.audio.channels, self.channel_controls):
            if control.isChecked():
                selected_channels.append(channel)
        self.waveform.set_channels(selected_channels, self.clock.duration)

    def _setup_annotations(self, tr: TaskRun) -> None:
        self._doc_path = tr.default_annotations_path()
        self._doc = anns.load(self._doc_path)
        self._doc.sub, self._doc.ses = tr.sub, tr.ses
        self._doc.task, self._doc.run = tr.task, tr.run
        if not self._doc.participants:
            self._doc.participants = _participants_from_map(tr)
        self.tier_panel.set_document(self._doc)
        self.timeline.set_document(self._doc, self.clock.duration)

    def _setup_sync_offsets(self, tr: TaskRun) -> None:
        sync_path = tr.default_sync_offsets_path()
        self._sync_offsets = load_sync(sync_path)
        lsl_sync_tsv = tr.root / "annot" / f"{tr.sub}_{tr.ses}_task-{tr.task}_run-{tr.run}_acq-lsl_sync.tsv"
        video_labels = [mf.acq or mf.path.stem for mf in tr.video]
        audio_raw_labels = [mf.acq or mf.path.stem for mf in tr.audio]
        lsl_ref_time = load_task_start_lsl(tr.task_windows_tsv, tr.task, tr.run)
        inferred = infer_sync_offsets_from_lsl_sync(
            lsl_sync_tsv,
            video_labels,
            audio_raw_labels,
            lsl_ref_time=lsl_ref_time,
        )
        source_session_dir = _resolve_source_session_dir(tr)
        source_inferred = (
            infer_sync_offsets_from_source_ffmpeg_lsl(
                source_session_dir,
                video_labels,
                audio_raw_labels,
                lsl_ref_time=lsl_ref_time,
            )
            if source_session_dir is not None
            else SyncOffsets()
        )
        inferred_count = 0
        if self._sync_offsets:
            for label, value in inferred.video.items():
                if label not in self._sync_offsets.video:
                    self._sync_offsets.video[label] = value
                    inferred_count += 1
            for label, value in source_inferred.video.items():
                if label not in self._sync_offsets.video:
                    self._sync_offsets.video[label] = value
                    inferred_count += 1
            # Audio offsets are keyed by display channel name in GUI sync files.
            for idx, mf in enumerate(tr.audio):
                if idx >= len(self.audio.channels):
                    continue
                raw_label = mf.acq or mf.path.stem
                channel_name = self.audio.channels[idx].name
                chosen = None
                if raw_label in inferred.audio:
                    chosen = inferred.audio[raw_label]
                elif raw_label in source_inferred.audio:
                    chosen = source_inferred.audio[raw_label]
                if chosen is None:
                    continue
                if channel_name not in self._sync_offsets.audio:
                    self._sync_offsets.audio[channel_name] = chosen
                    inferred_count += 1
        if inferred_count > 0:
            self.statusBar().showMessage(
                f"Loaded {inferred_count} auto sync offset(s) from {lsl_sync_tsv.name}",
                3500,
            )
        self._populate_sync_row(tr)

    def _populate_sync_row(self, tr: TaskRun) -> None:
        """Build spinboxes for video, audio, and transcript sync offsets."""
        lock_video = self._sync_video_lock_box.isChecked() if self._sync_video_lock_box is not None else True
        lock_audio = self._sync_audio_lock_box.isChecked() if self._sync_audio_lock_box is not None else True
        # Clear old spinboxes
        while self.sync_row.count() > 1:  # keep label
            item = self.sync_row.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._sync_video_lock_box = None
        self._sync_audio_lock_box = None
        self._sync_spinboxes.clear()
        self._sync_transcript_spinbox = None

        if self._sync_video_lock_box is None:
            self._sync_video_lock_box = QCheckBox("Lock V")
            self._sync_video_lock_box.setChecked(lock_video)
        if self._sync_audio_lock_box is None:
            self._sync_audio_lock_box = QCheckBox("Lock A")
            self._sync_audio_lock_box.setChecked(lock_audio)
        self.sync_row.addWidget(self._sync_video_lock_box)
        self.sync_row.addWidget(self._sync_audio_lock_box)

        # Video offsets
        for i, mf in enumerate(tr.video):
            label = mf.acq or mf.path.stem
            spin = QDoubleSpinBox()
            spin.setRange(-30.0, 30.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setPrefix(f"V{i} ")
            spin.setSuffix(" ")
            if self._sync_offsets and label in self._sync_offsets.video:
                spin.setValue(self._sync_offsets.video[label])
            idx = i
            spin.valueChanged.connect(lambda v, panel_idx=idx: self._on_video_sync_changed(panel_idx, v))
            self._sync_spinboxes[f"video_{i}"] = spin
            self.sync_row.addWidget(spin)
            # Apply to panel
            if idx < len(self.video_panels):
                self._apply_video_offset(idx, spin.value())

        # Audio offsets
        for i, ch in enumerate(self.audio.channels):
            spin = QDoubleSpinBox()
            spin.setRange(-30.0, 30.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setPrefix(f"A{i} ")
            spin.setSuffix(" ")
            if self._sync_offsets and ch.name in self._sync_offsets.audio:
                spin.setValue(self._sync_offsets.audio[ch.name])
            spin.valueChanged.connect(lambda v, ch_idx=i: self._on_audio_sync_changed(ch_idx, v))
            self._sync_spinboxes[f"audio_{i}"] = spin
            self.sync_row.addWidget(spin)
            # Apply to channel
            self._apply_audio_offset(i, spin.value())

        # Transcript offset
        trans_spin = QDoubleSpinBox()
        trans_spin.setRange(-30.0, 30.0)
        trans_spin.setSingleStep(0.05)
        trans_spin.setDecimals(2)
        trans_spin.setPrefix("Trans ")
        trans_spin.setSuffix(" ")
        if self._sync_offsets:
            trans_spin.setValue(self._sync_offsets.transcript)
        trans_spin.valueChanged.connect(self._on_transcript_sync_changed)
        self._sync_transcript_spinbox = trans_spin
        self.sync_row.addWidget(trans_spin)

        self.sync_row.addWidget(self.sync_save_btn)
        self.sync_row.addStretch(1)

    def _on_video_sync_changed(self, panel_idx: int, offset: float) -> None:
        if self._sync_updating:
            return
        old = self._video_offset_at(panel_idx)
        self._apply_video_offset(panel_idx, offset)
        if not (self._sync_video_lock_box and self._sync_video_lock_box.isChecked()):
            return
        delta = float(offset) - old
        if abs(delta) < 1e-9:
            return
        self._sync_updating = True
        try:
            for i, _ in enumerate(self.video_panels):
                if i == panel_idx:
                    continue
                cur = self._video_offset_at(i)
                new_val = _clamp_offset(cur + delta)
                self._set_sync_spin_value(f"video_{i}", new_val)
                self._apply_video_offset(i, new_val)
        finally:
            self._sync_updating = False

    def _on_audio_sync_changed(self, ch_idx: int, offset: float) -> None:
        if self._sync_updating:
            return
        old = self._audio_offset_at(ch_idx)
        self._apply_audio_offset(ch_idx, offset)
        if not (self._sync_audio_lock_box and self._sync_audio_lock_box.isChecked()):
            return
        delta = float(offset) - old
        if abs(delta) < 1e-9:
            return
        self._sync_updating = True
        try:
            for i, _ in enumerate(self.audio.channels):
                if i == ch_idx:
                    continue
                cur = self._audio_offset_at(i)
                new_val = _clamp_offset(cur + delta)
                self._set_sync_spin_value(f"audio_{i}", new_val)
                self._apply_audio_offset(i, new_val)
        finally:
            self._sync_updating = False

    def _on_transcript_sync_changed(self, offset: float) -> None:
        if self._sync_offsets:
            self._sync_offsets.transcript = offset

    def _set_sync_spin_value(self, key: str, value: float) -> None:
        spin = self._sync_spinboxes.get(key)
        if spin is None:
            return
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _video_offset_at(self, panel_idx: int) -> float:
        if not self._sync_offsets or not self._current_task or panel_idx >= len(self._current_task.video):
            return 0.0
        label = self._current_task.video[panel_idx].acq or self._current_task.video[panel_idx].path.stem
        return float(self._sync_offsets.video.get(label, 0.0))

    def _audio_offset_at(self, ch_idx: int) -> float:
        if not self._sync_offsets or ch_idx >= len(self.audio.channels):
            return 0.0
        return float(self._sync_offsets.audio.get(self.audio.channels[ch_idx].name, 0.0))

    def _apply_video_offset(self, panel_idx: int, offset: float) -> None:
        if panel_idx < len(self.video_panels):
            self.video_panels[panel_idx].set_time_offset(offset)
        if self._sync_offsets and self._current_task and panel_idx < len(self._current_task.video):
            label = self._current_task.video[panel_idx].acq or self._current_task.video[panel_idx].path.stem
            self._sync_offsets.video[label] = float(offset)

    def _apply_audio_offset(self, ch_idx: int, offset: float) -> None:
        if ch_idx < len(self.audio.channels):
            self.audio.set_channel_offset(ch_idx, offset)
        if self._sync_offsets and ch_idx < len(self.audio.channels):
            self._sync_offsets.audio[self.audio.channels[ch_idx].name] = float(offset)

    def _save_sync_offsets(self) -> None:
        if self._current_task is None or self._sync_offsets is None:
            return
        sync_path = self._current_task.default_sync_offsets_path()
        save_sync(sync_path, self._sync_offsets)
        self.statusBar().showMessage(f"Sync offsets saved → {sync_path.name}", 2500)

    def _teardown_media(self) -> None:
        self.clock.pause()
        self.audio.stop()
        for p in self.video_panels:
            self.video_grid.removeWidget(p)
            p.close()
            p.deleteLater()
        self.video_panels.clear()

        for checkbox in self.video_checkboxes:
            self.camera_row.removeWidget(checkbox)
            checkbox.deleteLater()

        while self.camera_row.count() > 0:
            item = self.camera_row.itemAt(self.camera_row.count() - 1)
            if item is None or item.widget() is not None:
                break
            self.camera_row.takeAt(self.camera_row.count() - 1)

        self.video_checkboxes.clear()
        self.feed_move_box.clear()
        self._pending_start = None
        self._doc = None
        self._doc_path = None
        self._transcript_path = None
        self._sync_offsets = None
        # Clear sync spinboxes
        while self.sync_row.count() > 1:
            item = self.sync_row.takeAt(1)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._sync_spinboxes.clear()
        self._sync_transcript_spinbox = None
        self._sync_video_lock_box = None
        self._sync_audio_lock_box = None
        self.tier_panel.set_document(None)
        self.timeline.set_document(None, 0.0)
        self.waveform.clear()
        self.transcript_editor.clear()
        self.transcript_path_label.setText("No transcript loaded")
        self.speaker_strip_label.setText("Now speaking: -")
        self.word_strip_label.setText("Word: -")
        self._segment_rows.clear()
        self._word_rows.clear()
        self._audio_annot_master_transcript_path = None
        self._audio_annot_master_words_path = None
        self._active_word_rows.clear()
        self._active_segment_rows.clear()
        self._transcript_table_updating = True
        self.transcript_segments_table.setRowCount(0)
        self.transcript_words_table.setRowCount(0)
        self._transcript_table_updating = False
        self._refresh_face_backend_label()

    def _auto_load_transcript(self, tr: TaskRun) -> None:
        structured = _load_audio_annot_transcript(
            tr,
            audio_raw_labels=[mf.acq or mf.path.stem for mf in tr.audio],
            audio_channel_labels=[ch.name for ch in self.audio.channels],
        )
        if structured is not None:
            text, source_label, segments, words, transcript_path, words_path = structured
            self._transcript_path = None
            self.transcript_editor.setPlainText(text)
            self.transcript_path_label.setText(source_label)
            self._segment_rows = segments
            self._word_rows = words
            self._audio_annot_master_transcript_path = transcript_path
            self._audio_annot_master_words_path = words_path
            self._populate_transcript_tables()
            return

        self._segment_rows = []
        self._word_rows = []
        self._audio_annot_master_transcript_path = None
        self._audio_annot_master_words_path = None
        self._populate_transcript_tables()

        for candidate in _transcript_candidates(tr):
            if candidate.is_file():
                self._load_transcript_file(candidate)
                return
        self._transcript_path = None
        self.transcript_editor.clear()
        self.transcript_path_label.setText("No transcript loaded")

    def _load_transcript_dialog(self) -> None:
        start_dir = str(self._current_task.root if self._current_task else Path.cwd())
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open transcript",
            start_dir,
            "Text/Subtitle (*.txt *.srt *.vtt *.tsv *.json *.ndjson);;All files (*.*)",
        )
        if not path_str:
            return
        self._load_transcript_file(Path(path_str))

    def _load_transcript_file(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except Exception as exc:
                QMessageBox.warning(self, "Transcript load failed", f"{type(exc).__name__}: {exc}")
                return
        self._transcript_path = path
        self.transcript_editor.setPlainText(text)
        self.transcript_path_label.setText(str(path))
        self._segment_rows = []
        self._word_rows = []
        self._audio_annot_master_transcript_path = None
        self._audio_annot_master_words_path = None
        self._populate_transcript_tables()

    def _save_transcript(self) -> None:
        if self._transcript_path is None:
            start_dir = str(self._current_task.root if self._current_task else Path.cwd())
            path_str, _ = QFileDialog.getSaveFileName(
                self,
                "Save transcript",
                start_dir,
                "Text/Subtitle (*.txt *.srt *.vtt *.tsv *.json *.ndjson);;All files (*.*)",
            )
            if not path_str:
                return
            self._transcript_path = Path(path_str)

        try:
            self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self._transcript_path.write_text(self.transcript_editor.toPlainText(), encoding="utf-8")
            self.transcript_path_label.setText(str(self._transcript_path))
            self.statusBar().showMessage(f"Transcript saved -> {self._transcript_path.name}", 2500)
        except Exception as exc:
            QMessageBox.warning(self, "Transcript save failed", f"{type(exc).__name__}: {exc}")

    # --------------------------------------------------------------- Transport
    def _toggle_play(self) -> None:
        self.clock.toggle()
        self.play_btn.setText("Pause" if self.clock.is_playing() else "Play")

    def _on_scrub(self, value: int) -> None:
        if self.clock.duration <= 0:
            return
        target = value / 1000.0 * self.clock.duration
        self.clock.seek(target)

    def _nudge(self, delta: float) -> None:
        self.clock.seek(self.clock.position() + delta)

    def _on_tick(self) -> None:
        pos = self.clock.position()
        dur = self.clock.duration
        for panel in self.video_panels:
            if panel.isVisible():
                panel.render_at(pos)
        self._refresh_face_backend_label()
        if dur > 0 and not self.scrub.isSliderDown():
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(pos / dur * 1000))
            self.scrub.blockSignals(False)
        self.pos_label.setText(f"{_fmt(pos)} / {_fmt(dur)}")
        self.timeline.set_playhead(pos)
        self.waveform.set_playhead(pos)
        self._update_transcript_playhead(pos)
        if not self.clock.is_playing():
            self.play_btn.setText("Play")

    def _populate_transcript_tables(self) -> None:
        self._transcript_table_updating = True
        try:
            self.transcript_segments_table.setRowCount(len(self._segment_rows))
            for row_idx, row in enumerate(self._segment_rows):
                self.transcript_segments_table.setItem(
                    row_idx,
                    0,
                    QTableWidgetItem(_fmt(float(row.get("_start", 0.0)))),
                )
                self.transcript_segments_table.setItem(
                    row_idx,
                    1,
                    QTableWidgetItem(str(row.get("speaker", ""))),
                )
                self.transcript_segments_table.setItem(
                    row_idx,
                    2,
                    QTableWidgetItem(str(row.get("mic", ""))),
                )
                self.transcript_segments_table.setItem(
                    row_idx,
                    3,
                    QTableWidgetItem(str(row.get("category", ""))),
                )
                self.transcript_segments_table.setItem(
                    row_idx,
                    4,
                    QTableWidgetItem(str(row.get("text", ""))),
                )

            self.transcript_words_table.setRowCount(len(self._word_rows))
            for row_idx, row in enumerate(self._word_rows):
                self.transcript_words_table.setItem(
                    row_idx,
                    0,
                    QTableWidgetItem(_fmt(float(row.get("_start", 0.0)))),
                )
                self.transcript_words_table.setItem(
                    row_idx,
                    1,
                    QTableWidgetItem(str(row.get("speaker", ""))),
                )
                self.transcript_words_table.setItem(
                    row_idx,
                    2,
                    QTableWidgetItem(str(row.get("mic", ""))),
                )
                self.transcript_words_table.setItem(
                    row_idx,
                    3,
                    QTableWidgetItem(str(row.get("word", ""))),
                )
        finally:
            self._transcript_table_updating = False

    def _update_transcript_playhead(self, playhead_seconds: float) -> None:
        if not self._word_rows and not self._segment_rows:
            self.speaker_strip_label.setText("Now speaking: -")
            self.word_strip_label.setText("Word: -")
            self.word_strip_label.setStyleSheet("padding:4px 8px; background:#111827; color:#cbd5e1;")
            return

        trans_offset = self._sync_offsets.transcript if self._sync_offsets is not None else 0.0
        transcript_t = playhead_seconds + trans_offset

        active_word_idxs = _find_active_transcript_rows(self._word_rows, transcript_t)
        if active_word_idxs:
            active_speakers: list[str] = []
            active_word_chunks: list[str] = []
            for idx in active_word_idxs:
                row = self._word_rows[idx]
                speaker = str(row.get("speaker", "-") or "-")
                word = str(row.get("word", "") or "")
                mic = str(row.get("mic", "") or "")
                if speaker and speaker not in active_speakers:
                    active_speakers.append(speaker)
                if word:
                    label = f"{speaker}:{word}" if len(active_word_idxs) > 1 else word
                    if mic:
                        label = f"{label}[{mic}]"
                    active_word_chunks.append(label)

            speaking = ", ".join(active_speakers) if active_speakers else "-"
            words = " | ".join(active_word_chunks) if active_word_chunks else "-"
            self.speaker_strip_label.setText(f"Now speaking: {speaking}")
            self.word_strip_label.setText(f"Word: {words}")
            self.word_strip_label.setStyleSheet(
                "padding:4px 8px; background:#7f1d1d; color:#ffffff; font-weight:700;"
            )

            self._set_table_row_highlight(
                self.transcript_words_table,
                self._active_word_rows,
                set(active_word_idxs),
                bg="#b91c1c",
                fg="#ffffff",
            )
        else:
            self.speaker_strip_label.setText("Now speaking: -")
            self.word_strip_label.setText("Word: -")
            self.word_strip_label.setStyleSheet("padding:4px 8px; background:#111827; color:#cbd5e1;")
            self._set_table_row_highlight(
                self.transcript_words_table,
                self._active_word_rows,
                set(),
                bg="#b91c1c",
                fg="#ffffff",
            )

        active_segment_idxs = _find_active_transcript_rows(self._segment_rows, transcript_t)
        self._set_table_row_highlight(
            self.transcript_segments_table,
            self._active_segment_rows,
            set(active_segment_idxs),
            bg="#f59e0b",
            fg="#111827",
        )
        if active_segment_idxs and active_segment_idxs[0] < self.transcript_segments_table.rowCount():
            self.transcript_segments_table.scrollToItem(
                self.transcript_segments_table.item(active_segment_idxs[0], 0),
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )

    def _set_table_row_highlight(
        self,
        table: QTableWidget,
        prev_active: set[int],
        now_active: set[int],
        bg: str,
        fg: str,
    ) -> None:
        if prev_active == now_active:
            return
        cols = table.columnCount()
        for row_idx in prev_active:
            if row_idx < 0 or row_idx >= table.rowCount():
                continue
            for col in range(cols):
                item = table.item(row_idx, col)
                if item is None:
                    continue
                item.setBackground(QColor("#00000000"))
                item.setForeground(QColor("#e5e7eb"))
        for row_idx in now_active:
            if row_idx < 0 or row_idx >= table.rowCount():
                continue
            for col in range(cols):
                item = table.item(row_idx, col)
                if item is None:
                    continue
                item.setBackground(QColor(bg))
                item.setForeground(QColor(fg))
        prev_active.clear()
        prev_active.update(now_active)

    def _on_segment_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._transcript_table_updating:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self._segment_rows):
            return
        if col == 1:
            self._segment_rows[row]["speaker"] = item.text().strip()
        elif col == 3:
            self._segment_rows[row]["category"] = item.text().strip()
        elif col == 4:
            self._segment_rows[row]["text"] = item.text().strip()
        self._refresh_transcript_editor_from_segments()
        self._save_audio_annot_edits()

    def _on_word_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._transcript_table_updating:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self._word_rows):
            return
        if col == 1:
            self._word_rows[row]["speaker"] = item.text().strip()
        elif col == 3:
            self._word_rows[row]["word"] = item.text().strip()
        self._save_audio_annot_edits()

    def _insert_transcript_line_after_current(self) -> None:
        if not self._segment_rows:
            self.statusBar().showMessage("No transcript segment rows loaded", 2500)
            return

        active_idxs = sorted(self._active_segment_rows)
        current_idx = active_idxs[-1] if active_idxs else None
        if current_idx is None:
            t = self.clock.position()
            trans_offset = self._sync_offsets.transcript if self._sync_offsets is not None else 0.0
            transcript_t = t + trans_offset
            now_rows = _find_active_transcript_rows(self._segment_rows, transcript_t)
            current_idx = now_rows[-1] if now_rows else None
            if current_idx is None:
                current_idx = len(self._segment_rows) - 1

        insert_idx = max(0, min(len(self._segment_rows), current_idx + 1))
        t = self.clock.position()
        trans_offset = self._sync_offsets.transcript if self._sync_offsets is not None else 0.0
        start = max(0.0, t + trans_offset)
        duration = 0.300
        speaker = self.transcript_event_speaker.currentText().strip() or "P1"
        category = self.transcript_event_category.currentText().strip() or "backchannel"
        text = self.transcript_event_text.text().strip() or category

        new_row: dict[str, object] = {
            "onset": f"{start:.3f}",
            "duration": f"{duration:.3f}",
            "speaker": speaker,
            "text": text,
            "category": category,
            "mic": "shared",
            "confidence": "manual",
            "_start": start,
            "_end": start + duration,
        }
        self._segment_rows.insert(insert_idx, new_row)

        self._populate_transcript_tables()
        if insert_idx < self.transcript_segments_table.rowCount():
            self.transcript_segments_table.selectRow(insert_idx)
            self.transcript_segments_table.scrollToItem(
                self.transcript_segments_table.item(insert_idx, 0),
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )
        self._refresh_transcript_editor_from_segments()
        self._save_audio_annot_edits()
        self.statusBar().showMessage("Inserted transcript annotation row", 2500)

    def _refresh_transcript_editor_from_segments(self) -> None:
        if not self._segment_rows:
            return
        lines: list[str] = []
        for row in self._segment_rows:
            start = _safe_float(row.get("_start"), default=_safe_float(row.get("onset"), default=0.0)) or 0.0
            speaker = str(row.get("speaker", "UNK") or "UNK")
            text = str(row.get("text", "") or "")
            category = str(row.get("category", "") or "")
            cat_part = f" {{{category}}}" if category else ""
            lines.append(f"[{_fmt(float(start))}] {speaker}{cat_part}: {text}")
        self.transcript_editor.setPlainText("\n".join(lines))

    def _save_audio_annot_edits(self) -> None:
        if self._audio_annot_master_transcript_path is None and self._audio_annot_master_words_path is None:
            self.statusBar().showMessage("No structured audio_annot table loaded", 2500)
            return

        try:
            if self._audio_annot_master_transcript_path is not None and self._segment_rows:
                _write_tsv_rows(self._audio_annot_master_transcript_path, self._segment_rows)
            if self._audio_annot_master_words_path is not None and self._word_rows:
                _write_tsv_rows(self._audio_annot_master_words_path, self._word_rows)
        except (OSError, csv.Error) as exc:
            QMessageBox.warning(self, "Save Rows failed", f"{type(exc).__name__}: {exc}")
            return

        self.statusBar().showMessage("Saved audio_annot transcript rows", 2500)

    # -------------------------------------------------------------- Annotate
    def _mark_in(self) -> None:
        if self._doc is None:
            return
        self._pending_start = self.clock.position()
        self.statusBar().showMessage(f"Mark start: {_fmt(self._pending_start)}", 2000)

    def _mark_out(self) -> None:
        if self._doc is None:
            return
        tier_id = self.tier_panel.active_tier_id()
        if not tier_id:
            QMessageBox.information(self, "No active tier", "Add a tier first (+ Tier).")
            return
        if self._pending_start is None:
            QMessageBox.information(self, "No start marked", "Press I first to mark the start.")
            return
        start = self._pending_start
        end = self.clock.position()
        if end < start:
            start, end = end, start
        label, ok = QInputDialog.getText(self, "Label", "Annotation label (optional):")
        if not ok:
            return
        try:
            self._doc.add_entry(anns.Entry(tier=tier_id, start=start, end=end, label=label))
        except ValueError as exc:
            QMessageBox.warning(self, "Add failed", str(exc))
            return
        self._pending_start = None
        self._on_doc_mutated()

    def _add_point(self) -> None:
        if self._doc is None:
            return
        tier_id = self.tier_panel.active_tier_id()
        if not tier_id:
            QMessageBox.information(self, "No active tier", "Add a tier first (+ Tier).")
            return
        t = self.clock.position()
        label, ok = QInputDialog.getText(self, "Label", "Point label (optional):")
        if not ok:
            return
        try:
            self._doc.add_entry(anns.Entry(tier=tier_id, start=t, end=t, label=label))
        except ValueError as exc:
            QMessageBox.warning(self, "Add failed", str(exc))
            return
        self._on_doc_mutated()

    def _delete_selected(self) -> None:
        # Stored on the timeline; we re-select by searching for the current
        # selection on the timeline widget.
        entry = getattr(self.timeline, "_selected", None)
        if entry is None or self._doc is None:
            return
        self._doc.remove_entry(entry)
        self._on_doc_mutated()

    def _edit_entry(self, entry: anns.Entry) -> None:
        new_label, ok = QInputDialog.getText(
            self, "Edit annotation", "Label:", text=entry.label
        )
        if not ok:
            return
        entry.label = new_label
        self._on_doc_mutated()

    def _on_doc_mutated(self) -> None:
        self.timeline.refresh()
        self.tier_panel.refresh()
        self._save_annotations(silent=True)

    def _save_annotations(self, silent: bool = False) -> None:
        if self._doc is None or self._doc_path is None:
            return
        try:
            anns.save(self._doc_path, self._doc)
            if not silent:
                self.statusBar().showMessage(f"Saved → {self._doc_path.name}", 3000)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", f"{type(exc).__name__}: {exc}")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self._save_annotations(silent=True)
        # Auto-save sync offsets if any are non-zero
        if (
            self._current_task is not None
            and self._sync_offsets is not None
            and (self._sync_offsets.video or self._sync_offsets.audio or self._sync_offsets.transcript != 0.0)
        ):
            self._save_sync_offsets()
        self._teardown_media()
        super().closeEvent(event)


def _fmt(t: float) -> str:
    t = max(0.0, t)
    total_ms = int(round(t * 1000.0))
    total_s, ms = divmod(total_ms, 1000)
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}" if h else f"{m:02d}:{s:02d}.{ms:03d}"


def _clamp_offset(value: float) -> float:
    return max(-30.0, min(30.0, float(value)))


def _resolve_source_session_dir(tr: TaskRun) -> Path | None:
    """Resolve matching source-session directory for raw capture logs.

    Search order:
    1) `AFFECTAI_SOURCE_SESSIONS_ROOT` (expects `.../sub-XX/<ses-id>`)
    2) default local path used in this project setup
    """
    roots: list[Path] = []
    env_root = os.getenv("AFFECTAI_SOURCE_SESSIONS_ROOT", "").strip()
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path(r"C:\Users\meisa\Documents\Code\affectai-capture-small\sessions\final"))

    for root in roots:
        candidate = root / tr.sub / tr.ses
        if candidate.is_dir():
            return candidate
    return None


def _participants_from_map(tr: TaskRun) -> dict[str, str]:
    """Best-effort {P1:sub-01, ...} map from participant_signal_map.tsv."""
    df = load_participant_map(tr.participant_map_tsv)
    if df.empty:
        return {}
    out: dict[str, str] = {}
    part_col = next((c for c in df.columns if c.lower() in ("participant", "sub", "subject")), None)
    role_col = next((c for c in df.columns if c.lower() in ("role", "slot", "position")), None)
    if part_col is None:
        return {}
    for _, row in df.iterrows():
        key = str(row[role_col]) if role_col else str(row[part_col])
        val = str(row[part_col])
        if key and val and key.lower() != "nan":
            out[key] = val
    return out


def _display_audio_name(raw_name: str) -> str:
    name = raw_name.strip()
    token = name.lower().replace("-", "_")
    match = re.search(r"dpa[_ ]*mic[_ ]*(\d+)", token)
    if match:
        return f"dpa {int(match.group(1))}"

    mic_match = re.search(r"mic[_ ]*(\d+)", token)
    if mic_match:
        return f"dpa {int(mic_match.group(1))}"

    return name.replace("_", " ")


def _format_mic_label(channel: object) -> str:
    name = str(getattr(channel, "name", "audio"))
    source_rate = int(getattr(channel, "source_rate", 0))
    source_bits = int(getattr(channel, "source_bits", 0))
    if source_rate <= 0 and source_bits <= 0:
        return name
    if source_bits > 0:
        return f"{name} ({source_rate // 1000} kHz / {source_bits}-bit)"
    return f"{name} ({source_rate // 1000} kHz)"


def _transcript_candidates(tr: TaskRun) -> list[Path]:
    base = f"{tr.sub}_{tr.ses}_task-{tr.task}_run-{tr.run}"
    return [
        tr.root / "annot" / f"{base}_transcript.txt",
        tr.root / "annot" / f"{base}_transcript.srt",
        tr.root / "annot" / f"{base}_transcript.vtt",
        tr.root / "beh" / f"{base}_transcript.txt",
        tr.root / "beh" / f"{base}_transcript.srt",
        tr.root / "beh" / f"{base}_transcript.vtt",
    ]


def _load_audio_annot_transcript(
    tr: TaskRun,
    audio_raw_labels: list[str],
    audio_channel_labels: list[str],
) -> tuple[str, str, list[dict[str, object]], list[dict[str, object]], Path | None, Path | None] | None:
    """Load transcript text from audio_annot task folder when present."""
    task = (tr.task or "").strip().upper()
    if not task:
        return None

    task_dir = tr.root / "audio_annot" / task
    if not task_dir.is_dir():
        return None

    master_transcript = task_dir / "master_transcript.tsv"
    master_words = task_dir / "master_words.tsv"
    mic_jsons = sorted(task_dir.glob("mic*_transcript.json"))

    if not master_transcript.is_file() and not master_words.is_file() and not mic_jsons:
        return None

    lines: list[str] = []
    source_paths: list[str] = []
    segment_rows: list[dict[str, object]] = []
    word_rows: list[dict[str, object]] = []

    if master_transcript.is_file():
        source_paths.append(str(master_transcript))
        lines.append("# master_transcript")
        try:
            with master_transcript.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    onset = _safe_float(row.get("onset"), default=None)
                    duration = _safe_float(row.get("duration"), default=0.0) or 0.0
                    speaker = str(row.get("speaker") or "UNK").strip()
                    mic = str(row.get("mic") or "").strip()
                    category = str(row.get("category") or "").strip()
                    text = str(row.get("text") or "").strip()
                    if not text:
                        continue
                    ts = _fmt(onset) if onset is not None else "--:--"
                    mic_part = f" [{mic}]" if mic else ""
                    cat_part = f" {{{category}}}" if category else ""
                    lines.append(f"[{ts}] {speaker}{mic_part}{cat_part}: {text}")
                    start = float(onset) if onset is not None else 0.0
                    end = max(start, start + float(duration))
                    segment_rows.append(
                        {
                            **dict(row),
                            "speaker": speaker,
                            "mic": mic,
                            "category": category,
                            "text": text,
                            "_start": start,
                            "_end": end,
                        }
                    )
        except (OSError, csv.Error, UnicodeDecodeError, ValueError) as exc:
            lines.append(f"[load error] {master_transcript.name}: {type(exc).__name__}: {exc}")

    if master_words.is_file():
        source_paths.append(str(master_words))
        try:
            with master_words.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    word = str(row.get("word") or "").strip()
                    if not word:
                        continue
                    mic = str(row.get("mic") or "unknown").strip() or "unknown"
                    speaker = str(row.get("speaker") or "UNK").strip()
                    onset = _safe_float(row.get("onset"), default=0.0) or 0.0
                    duration = _safe_float(row.get("duration"), default=0.0) or 0.0
                    end = max(float(onset), float(onset) + float(duration))
                    word_rows.append(
                        {
                            **dict(row),
                            "speaker": speaker,
                            "mic": mic,
                            "word": word,
                            "_start": float(onset),
                            "_end": end,
                        }
                    )
        except (OSError, csv.Error, UnicodeDecodeError, ValueError) as exc:
            # Keep master_words hidden from transcript text UI for now.
            _ = exc

    if mic_jsons:
        dpa_mics = _collect_dpa_mic_ids(audio_raw_labels, audio_channel_labels)
        matched: list[str] = []
        unmatched: list[str] = []
        for path in mic_jsons:
            source_paths.append(str(path))
            mic_id = _extract_mic_id(path.stem)
            seg_count = 0
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    segments = payload.get("segments")
                    if isinstance(segments, list):
                        seg_count = len(segments)
            except (OSError, UnicodeDecodeError, ValueError):
                pass

            if mic_id is not None and mic_id in dpa_mics:
                matched.append(f"dpa mic{mic_id} -> {path.name} ({seg_count} segments)")
            else:
                unmatched.append(f"{path.name} ({seg_count} segments)")

        lines.append("")
        lines.append("# mic transcript matching")
        if matched:
            lines.extend(f"- {item}" for item in matched)
        else:
            lines.append("- no mic transcript file matched loaded DPA channels")
        if unmatched:
            lines.append("# unmatched mic transcripts")
            lines.extend(f"- {item}" for item in unmatched)

    text = "\n".join(lines).strip()
    if not text:
        return None
    source_label = "; ".join(source_paths)
    return (
        text,
        f"audio_annot loaded: {source_label}",
        segment_rows,
        word_rows,
        master_transcript if master_transcript.is_file() else None,
        master_words if master_words.is_file() else None,
    )


def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _extract_mic_id(text: str) -> int | None:
    match = re.search(r"mic[_\- ]*(\d+)", (text or "").lower())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _collect_dpa_mic_ids(audio_raw_labels: list[str], audio_channel_labels: list[str]) -> set[int]:
    ids: set[int] = set()
    for label in list(audio_raw_labels) + list(audio_channel_labels):
        mic_id = _extract_mic_id(label)
        if mic_id is not None:
            ids.add(mic_id)
    return ids


def _find_active_transcript_rows(rows: list[dict[str, object]], t_seconds: float) -> list[int]:
    active: list[int] = []
    for idx, row in enumerate(rows):
        start = _safe_float(row.get("_start"), default=None)
        end = _safe_float(row.get("_end"), default=None)
        if start is None or end is None:
            continue
        if float(start) <= t_seconds <= float(end):
            active.append(idx)
    return active


def _write_tsv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    preferred = [
        "onset",
        "duration",
        "speaker",
        "text",
        "word",
        "category",
        "mic",
        "confidence",
        "score",
        "energy",
        "energy_ratio",
    ]
    present: list[str] = []
    for row in rows:
        for k in row.keys():
            key = str(k)
            if key.startswith("_"):
                continue
            if key not in present:
                present.append(key)
    headers = [k for k in preferred if k in present] + [k for k in present if k not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def run(bids_root: Path | None = None) -> int:
    # pandas import kept local-scope-free; pd referenced implicitly via session_loader.
    _ = pd  # silence unused import warning in some linters
    app = QApplication(sys.argv)
    win = MainWindow(bids_root=bids_root)
    win.show()
    return app.exec()
