"""Tier management panel: list tiers, add/remove, select the active tier.

The "active tier" is where new spans created with the I/O shortcuts go. The
panel also shows a count of entries per tier and marks read-only tiers.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..io.annotations import AnnotationDoc, Tier


class NewTierDialog(QDialog):
    def __init__(self, participants: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New tier")
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("e.g. backchannel")
        self.participant_input = QComboBox()
        self.participant_input.addItem("(session)", "")
        for p in participants:
            self.participant_input.addItem(p, p)
        self.kind_input = QComboBox()
        self.kind_input.addItems(["span", "point"])

        form = QFormLayout()
        form.addRow("Tier name:", self.id_input)
        form.addRow("Participant:", self.participant_input)
        form.addRow("Kind:", self.kind_input)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def result_tier(self) -> Tier | None:
        name = self.id_input.text().strip()
        if not name:
            return None
        part = self.participant_input.currentData() or ""
        full_id = f"{part}.{name}" if part else name
        return Tier(id=full_id, participant=part, kind=self.kind_input.currentText())


class TierPanel(QWidget):
    """List of tiers on the side; emits when the active tier changes."""

    activeTierChanged = Signal(str)  # tier_id, or "" if none
    tiersMutated = Signal()  # fire after add/remove so host can save+refresh

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._doc: AnnotationDoc | None = None

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.currentItemChanged.connect(self._on_current_changed)

        self._add_btn = QPushButton("+ Tier")
        self._add_btn.clicked.connect(self._on_add)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove)

        btns = QHBoxLayout()
        btns.addWidget(self._add_btn)
        btns.addWidget(self._remove_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Tiers"))
        layout.addWidget(self._list, 1)
        layout.addLayout(btns)

    # ------------------------------------------------------------------- state
    def set_document(self, doc: AnnotationDoc | None) -> None:
        self._doc = doc
        self.refresh()

    def active_tier_id(self) -> str:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def refresh(self) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        if self._doc is not None:
            for tier in self._doc.tiers:
                n = len(self._doc.entries_in_tier(tier.id))
                tag = " [ro]" if tier.readonly else ""
                item = QListWidgetItem(f"{tier.id}  ({n}){tag}")
                item.setData(Qt.ItemDataRole.UserRole, tier.id)
                self._list.addItem(item)
        self._list.blockSignals(False)
        if self._list.count() > 0:
            self._list.setCurrentRow(0)
        self._on_current_changed(self._list.currentItem(), None)

    # ----------------------------------------------------------------- actions
    def _on_add(self) -> None:
        if self._doc is None:
            return
        participants = sorted(self._doc.participants.keys()) or ["P1", "P2", "P3", "P4"]
        dlg = NewTierDialog(participants, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        tier = dlg.result_tier()
        if tier is None:
            return
        try:
            self._doc.add_tier(tier)
        except ValueError:
            return  # duplicate id; silently ignore
        self.refresh()
        self.tiersMutated.emit()

    def _on_remove(self) -> None:
        tier_id = self.active_tier_id()
        if not tier_id or self._doc is None:
            return
        self._doc.remove_tier(tier_id)
        self.refresh()
        self.tiersMutated.emit()

    def _on_current_changed(self, current, _previous) -> None:
        tier_id = current.data(Qt.ItemDataRole.UserRole) if current else ""
        self.activeTierChanged.emit(tier_id or "")
