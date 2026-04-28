"""
The FILES section in the left sidebar.

Lists files in the user's library. Each row has a checkbox (selects
file for next Run) and a delete button. A "+ Add file" button at the
bottom triggers an upload to all configured providers.

Persists across app sessions because state lives in SQLite via the
shared FileLibrary instance.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QRunnable, QObject, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from attachments.file_library import FileLibrary, LibraryFile, UploadOutcome, DeleteOutcome


# ---------- Worker signals ----------

class _LibraryWorkerSignals(QObject):
    upload_finished = Signal(object)   # UploadOutcome
    delete_finished = Signal(object)   # DeleteOutcome
    failed = Signal(str)               # error message


# ---------- Background workers ----------

class _UploadWorker(QRunnable):
    """Runs FileLibrary.add_file on a pool thread to keep UI responsive."""

    def __init__(self, library: FileLibrary, path: Path, signals: _LibraryWorkerSignals) -> None:
        super().__init__()
        self._library = library
        self._path = path
        self._signals = signals

    def run(self) -> None:
        try:
            outcome = self._library.add_file(self._path)
            self._signals.upload_finished.emit(outcome)
        except Exception as exc:
            self._signals.failed.emit(f"Upload failed: {exc}")


class _DeleteWorker(QRunnable):
    """Runs FileLibrary.delete_file on a pool thread."""

    def __init__(self, library: FileLibrary, file_id: int, signals: _LibraryWorkerSignals) -> None:
        super().__init__()
        self._library = library
        self._file_id = file_id
        self._signals = signals

    def run(self) -> None:
        try:
            outcome = self._library.delete_file(self._file_id)
            self._signals.delete_finished.emit(outcome)
        except Exception as exc:
            self._signals.failed.emit(f"Delete failed: {exc}")


# ---------- Main panel widget ----------

class FileLibraryPanel(QWidget):
    """Sidebar section for managing the file library."""

    # Emitted when the set of checked files changes. Payload = list of file_ids.
    selection_changed = Signal(list)

    def __init__(self, library: FileLibrary) -> None:
        super().__init__()
        self._library = library
        self._rows: dict[int, _FileRow] = {}    # file_id -> row widget
        self._busy = False                       # blocks concurrent ops

        self._signals = _LibraryWorkerSignals()
        self._signals.upload_finished.connect(self._on_upload_finished)
        self._signals.delete_finished.connect(self._on_delete_finished)
        self._signals.failed.connect(self._on_worker_failed)

        self._pool = QThreadPool.globalInstance()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        section_label = QLabel("FILES")
        section_label.setObjectName("sectionLabel")
        layout.addWidget(section_label)
        layout.addSpacing(2)

        # Container for file rows
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        layout.addWidget(self._rows_container)

        # Empty-state label, shown when no files
        self._empty_label = QLabel("No files yet.")
        self._empty_label.setStyleSheet("color: #5E5E5E; font-size: 11px; padding: 4px 0;")
        layout.addWidget(self._empty_label)

        # "+ Add file" button
        self._add_button = QPushButton("+ Add file")
        self._add_button.setObjectName("secondary")
        self._add_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_button.clicked.connect(self._on_add_clicked)
        layout.addWidget(self._add_button)

        self._refresh_from_library()

    # ---------- Public API ----------

    def selected_file_ids(self) -> list[int]:
        """Return file_ids of currently-checked rows."""
        return [fid for fid, row in self._rows.items() if row.is_checked()]

    # ---------- Refresh / reload from registry ----------

    def _refresh_from_library(self) -> None:
        """Rebuild the row list from the FileLibrary."""
        # Clear existing rows
        for row in self._rows.values():
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()

        files = self._library.list_files()
        for f in files:
            self._add_row(f, checked=True)

        self._update_empty_state()

    def _add_row(self, file: LibraryFile, checked: bool) -> None:
        row = _FileRow(file, checked=checked)
        row.checkbox_changed.connect(self._on_checkbox_changed)
        row.delete_requested.connect(self._on_delete_clicked)
        self._rows[file.file_id] = row
        self._rows_layout.addWidget(row)

    def _update_empty_state(self) -> None:
        has_files = len(self._rows) > 0
        self._empty_label.setVisible(not has_files)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._add_button.setEnabled(not busy)
        for row in self._rows.values():
            row.set_enabled(not busy)
        if busy:
            QCursor.setPos(QCursor.pos())  # nudge to ensure repaint
            from PySide6.QtWidgets import QApplication
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            from PySide6.QtWidgets import QApplication
            QApplication.restoreOverrideCursor()

    # ---------- User actions ----------

    def _on_add_clicked(self) -> None:
        if self._busy:
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add files to library",
            "",
            "PDF documents (*.pdf);;All files (*)",
        )
        if not paths:
            return

        # Upload each picked file in turn. We launch one worker per file
        # to avoid blocking the UI during multi-file selection.
        self._set_busy(True)
        self._pending_uploads = len(paths)
        for path_str in paths:
            worker = _UploadWorker(self._library, Path(path_str), self._signals)
            self._pool.start(worker)

    def _on_delete_clicked(self, file_id: int) -> None:
        if self._busy:
            return

        # Confirm with the user — deletion is destructive across all providers.
        row = self._rows.get(file_id)
        if row is None:
            return
        filename = row.filename()

        reply = QMessageBox.question(
            self,
            "Delete file",
            f"Delete '{filename}' from all providers? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._set_busy(True)
        worker = _DeleteWorker(self._library, file_id, self._signals)
        self._pool.start(worker)

    def _on_checkbox_changed(self, file_id: int, checked: bool) -> None:
        self.selection_changed.emit(self.selected_file_ids())

    # ---------- Worker callbacks ----------

    def _on_upload_finished(self, outcome: UploadOutcome) -> None:
        self._pending_uploads -= 1
        if self._pending_uploads <= 0:
            self._set_busy(False)
            self._pending_uploads = 0

        if outcome.failed_providers and not outcome.successful_providers:
            QMessageBox.warning(
                self,
                "Upload failed",
                f"Could not upload {outcome.file_record.filename} to any provider.\n\n"
                + "\n".join(f"- {p}: {msg}" for p, msg in outcome.failed_providers.items()),
            )
            return

        # Refresh the list from the registry to pick up the new row.
        self._refresh_from_library()
        # Newly-added file is checked by default (refresh checks all).
        self.selection_changed.emit(self.selected_file_ids())

        if outcome.failed_providers:
            QMessageBox.information(
                self,
                "Partial upload",
                f"Uploaded {outcome.file_record.filename} to "
                f"{len(outcome.successful_providers)} of "
                f"{len(outcome.successful_providers) + len(outcome.failed_providers)} "
                "providers.\n\nFailed: "
                + ", ".join(outcome.failed_providers.keys()),
            )

    def _on_delete_finished(self, outcome: DeleteOutcome) -> None:
        self._set_busy(False)

        if outcome.failed_providers:
            QMessageBox.warning(
                self,
                "Partial delete",
                f"Some providers failed to delete the file.\n\n"
                + "\n".join(f"- {p}: {msg}" for p, msg in outcome.failed_providers.items())
                + "\n\nThe file remains in the library so you can retry.",
            )
        # Refresh whether or not it fully deleted — successful providers
        # are gone from the registry.
        self._refresh_from_library()
        self.selection_changed.emit(self.selected_file_ids())

    def _on_worker_failed(self, message: str) -> None:
        self._set_busy(False)
        QMessageBox.critical(self, "Operation failed", message)


# ---------- One row per file ----------

class _FileRow(QFrame):
    """A single row in the FILES section: checkbox + name + delete."""

    checkbox_changed = Signal(int, bool)     # file_id, checked
    delete_requested = Signal(int)            # file_id

    def __init__(self, file: LibraryFile, checked: bool) -> None:
        super().__init__()
        self._file_id = file.file_id
        self._filename = file.filename

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        self._checkbox = QCheckBox()
        self._checkbox.setChecked(checked)
        self._checkbox.stateChanged.connect(self._on_checkbox_state_changed)
        layout.addWidget(self._checkbox)

        # Truncate long filenames to fit the narrow sidebar.
        display = file.filename
        if len(display) > 18:
            display = display[:15] + "..."
        self._name_label = QLabel(display)
        self._name_label.setToolTip(file.filename)
        self._name_label.setStyleSheet("color: #D4D4D4; font-size: 12px;")
        self._name_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self._name_label)

        self._delete_button = QPushButton("×")
        self._delete_button.setObjectName("fileChipRemove")
        self._delete_button.setFixedSize(18, 18)
        self._delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._delete_button.setToolTip("Delete from all providers")
        self._delete_button.clicked.connect(lambda: self.delete_requested.emit(self._file_id))
        layout.addWidget(self._delete_button)

    def is_checked(self) -> bool:
        return self._checkbox.isChecked()

    def filename(self) -> str:
        return self._filename

    def set_enabled(self, enabled: bool) -> None:
        self._checkbox.setEnabled(enabled)
        self._delete_button.setEnabled(enabled)

    def _on_checkbox_state_changed(self, _state: int) -> None:
        self.checkbox_changed.emit(self._file_id, self._checkbox.isChecked())