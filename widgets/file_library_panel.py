"""
The FILES section in the left sidebar.

Lists files in the user's library. Each row has a checkbox (selects
file for next Run) and a delete button. A "+ Add file" button at the
bottom triggers an upload to the providers that natively support
the file's type.

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
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from attachments.file_library import FileLibrary, LibraryFile, UploadOutcome, DeleteOutcome


# Phase 1 native-supported file types. Update alongside the routing layer's
# capability matrix as new types come in (docx, pptx, gif, webp, etc.).
_FILE_DIALOG_FILTER = (
    "Supported files (*.pdf *.xlsx *.csv *.png *.jpg *.jpeg);;"
    "PDF documents (*.pdf);;"
    "Spreadsheets (*.xlsx *.csv);;"
    "Images (*.png *.jpg *.jpeg);;"
    "All files (*)"
)


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

        # Upload queue. We process one file at a time to avoid SQLite
        # race conditions in FileRegistry — its sqlite3.Connection is
        # not thread-safe, and parallel writes corrupt cursor state
        # producing random "tuple index out of range" /
        # "NoneType has no attribute 'endswith'" errors. Serial uploads
        # trade speed for reliability. Verified Apr 2026 with a 10-file
        # repro that lost ~20% of uploads under parallel execution.
        self._upload_queue: list[Path] = []


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

        # Container for file rows. Wrapped in a QScrollArea so the
        # FILES list scrolls when there are more files than fit in
        # the sidebar's vertical space. Without this the rows stack
        # off the bottom and either push the "+ Add file" button
        # off-screen or get silently clipped — a real bug for users
        # who upload many files. v0.1.11.
        self._rows_container = QWidget()
        self._rows_container.setObjectName("filesRowsContainer")
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        # AlignTop keeps rows packed at the top instead of stretching
        # vertically when there are few files (avoids ugly gaps).
        self._rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._rows_scroll = QScrollArea()
        self._rows_scroll.setObjectName("filesRowsScroll")
        self._rows_scroll.setWidgetResizable(True)
        self._rows_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._rows_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # Explicit ScrollBarAsNeeded — on macOS, leaving this default
        # can result in the OS-managed floating overlay scroll bar that
        # ignores our QSS styling. Forcing AsNeeded gives us Qt's
        # styled scroll bar consistently across Windows and macOS.
        self._rows_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._rows_scroll.setWidget(self._rows_container)
        # stretch=1 lets the scroll area absorb whatever vertical
        # space the sidebar has, so the empty-state label and the
        # "+ Add file" button below stay anchored at the bottom.
        layout.addWidget(self._rows_scroll, stretch=1)

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
            _FILE_DIALOG_FILTER,
        )
        if not paths:
            return
        self.queue_uploads([Path(p) for p in paths])

    def queue_uploads(self, paths: list[Path]) -> bool:
        """Public entry for queuing uploads from any source.

        Used by the "+ Add file" button (via _on_add_clicked) and by
        the main window's drag-and-drop handler. Returns True if the
        upload batch started, False if the panel is busy (the caller
        can decide how to surface that — a status message, a beep,
        or silently ignore).

        Serial upload via _kick_next_upload — same constraint as the
        button path: one file at a time to avoid SQLite races in
        FileRegistry. The UI is disabled for the duration of the
        batch via _set_busy.
        """
        if self._busy or not paths:
            return False
        self._set_busy(True)
        self._pending_uploads = len(paths)
        self._upload_queue = list(paths)
        self._kick_next_upload()
        return True

    def _kick_next_upload(self) -> None:
        """Start the next queued upload, or finish the batch if empty.

        Called from _on_add_clicked to start the first upload, and
        from _on_upload_finished / _on_worker_failed to chain the
        next one. Only one _UploadWorker runs at a time.
        """
        if not self._upload_queue:
            # Batch complete. Re-enable the UI.
            self._set_busy(False)
            self._pending_uploads = 0
            return
        next_path = self._upload_queue.pop(0)
        worker = _UploadWorker(self._library, next_path, self._signals)
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
        # Wrap the whole body in try/finally so _kick_next_upload always
        # runs regardless of which exit path the method takes (hard
        # failure return, type-unsupported return, or normal completion).
        # Without this, any early return leaves the queue stalled and
        # the UI permanently busy.
        try:
            self._pending_uploads -= 1
            # Don't clear busy state here — _kick_next_upload does it
            # when the queue is finally empty.

            filename = outcome.file_record.filename
            n_success = len(outcome.successful_providers)
            n_failed = len(outcome.failed_providers)
            n_skipped = len(outcome.skipped_providers)

            # Hard failure: nothing succeeded AND nothing was skipped
            # (so every provider tried and every provider errored).
            if n_success == 0 and n_failed > 0 and n_skipped == 0:
                QMessageBox.warning(
                    self,
                    "Upload failed",
                    f"Could not upload {filename} to any provider.\n\n"
                    + "\n".join(f"- {p}: {msg}" for p, msg in outcome.failed_providers.items()),
                )
                return

            # Hard wall: zero providers natively support this file type.
            # Different message — retry won't fix it; user needs a
            # different file or provider.
            if n_success == 0 and n_failed == 0 and n_skipped > 0:
                QMessageBox.warning(
                    self,
                    "File type not supported",
                    f"None of your configured providers natively support {filename}.\n\n"
                    "The file is in your library but cannot be used in chat. "
                    "Either configure a provider that supports this type, or use a "
                    "different file.",
                )
                self._refresh_from_library()
                self.selection_changed.emit(self.selected_file_ids())
                return

            # Refresh the list from the registry to pick up the new row.
            self._refresh_from_library()
            # Newly-added file is checked by default (refresh checks all).
            self.selection_changed.emit(self.selected_file_ids())

            # Partial success — surface skipped providers and any upload
            # failures in a single combined dialog so the user gets the
            # full picture at a glance instead of two consecutive popups.
            notes: list[str] = []
            if n_skipped > 0:
                skipped_names = ", ".join(sorted(outcome.skipped_providers.keys()))
                notes.append(
                    f"Not natively supported by: {skipped_names}. "
                    f"These models will be unavailable for this file at Run time."
                )
            if n_failed > 0:
                notes.append(
                    "Upload errors:\n"
                    + "\n".join(f"  - {p}: {msg}" for p, msg in outcome.failed_providers.items())
                )

            if notes:
                QMessageBox.information(
                    self,
                    f"Added {filename}",
                    f"Uploaded to {n_success} provider(s).\n\n" + "\n\n".join(notes),
                )
        finally:
            # Always chain to the next queued upload, regardless of how
            # this one ended (success, hard failure, type-unsupported).
            # If the queue is empty, _kick_next_upload re-enables the UI.
            self._kick_next_upload()

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
        # Don't clear busy or stop the batch — _kick_next_upload
        # handles that when the queue is empty. Show the error and
        # continue with the next file. This way one bad PDF doesn't
        # abort the whole batch.
        QMessageBox.critical(self, "Operation failed", message)
        self._pending_uploads = max(0, self._pending_uploads - 1)
        self._kick_next_upload()


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
