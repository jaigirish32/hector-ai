"""
The main application window for HECTOR-AI.

Composes the sidebar (left) with a stacked content area (right).
Owns the shared FileLibrary so the sidebar and ComparisonView share state.

Drag and drop:
    The window accepts file drops anywhere on its surface. Dropped PDFs
    are queued through the existing FileLibraryPanel.queue_uploads()
    path, which handles the per-provider fan-out, partial-coverage
    messages, and SQLite-safe serialization. Non-PDF files and folders
    are rejected with a brief message — Phase 1 supports PDFs only.

    A teal-bordered overlay covers the central widget while a valid
    drag is in progress, giving the user clear feedback that the drop
    will be accepted. The overlay is hidden on drop or leave.

    Cross-platform: Qt's drag/drop translates Finder URLs (macOS) and
    Windows Explorer paths into QUrl uniformly. QUrl.toLocalFile()
    returns a usable path either way. No platform-specific code needed.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from attachments.file_library import FileLibrary
from views.comparison_view import ComparisonView
from views.settings_view import SettingsView
from windows.sidebar import Sidebar


# Files accepted via drag and drop in Phase 1. Keep in sync with
# _FILE_DIALOG_FILTER in widgets/file_library_panel.py — both gate the
# same set of supported types. Phase 2 widens this when docx/xlsx land.
_DRAG_DROP_ALLOWED_EXTENSIONS = frozenset({".pdf"})


class MainWindow(QMainWindow):
    """The top-level window. Holds a sidebar and a stack of views."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("HECTOR-AI")
        self.setMinimumSize(1200, 780)
        self.resize(1360, 860)

        # Enable drops on the window itself. Qt routes drag events
        # through the top-level widget; child widgets without their
        # own drop handler bubble them up to here.
        self.setAcceptDrops(True)

        # One FileLibrary shared across the app — sidebar manages it,
        # ComparisonView reads selected file_ids from it.
        self._file_library = FileLibrary()

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = Sidebar(file_library=self._file_library)
        self.sidebar.view_changed.connect(self._on_view_changed)
        root_layout.addWidget(self.sidebar)

        self.content_stack = QStackedWidget()
        root_layout.addWidget(self.content_stack, stretch=1)

        # Index 0 — Compare models. Pass the file library so it can
        # read selected file_ids when Run is clicked.
        self.comparison_view = ComparisonView(file_library=self._file_library)
        self.comparison_view.set_file_panel(self.sidebar.file_panel)
        self.content_stack.addWidget(self.comparison_view)

        # Index 1 — Settings
        self.settings_view = SettingsView()
        self.content_stack.addWidget(self.settings_view)

        # Drop overlay — created once, shown/hidden as needed. Parented
        # to the central widget so it sits over the sidebar + content.
        # Hidden by default; raised to the top of its z-order when shown
        # so it draws over child widgets without interfering with their
        # event handling when not visible.
        self._drop_overlay = self._build_drop_overlay(central)
        self._drop_overlay.hide()

    # ---------- Drop overlay ----------

    def _build_drop_overlay(self, parent: QWidget) -> QWidget:
        """Translucent teal-bordered hint that appears during a valid drag.

        Inline style rather than a theme rule because this widget only
        exists in this one place. Pointer-event-transparent isn't a Qt
        thing — instead we just don't install event handlers on it, and
        it gets raised above siblings only while visible.
        """
        overlay = QWidget(parent)
        overlay.setObjectName("dropOverlay")
        overlay.setStyleSheet(
            "QWidget#dropOverlay {"
            "  background-color: rgba(0, 212, 196, 30);"
            "  border: 2px dashed #00D4C4;"
            "  border-radius: 12px;"
            "}"
        )

        layout = QVBoxLayout(overlay)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        msg = QLabel("Drop PDF to add to library")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet(
            "color: #00D4C4;"
            "font-size: 18px;"
            "font-weight: 600;"
            "background: transparent;"
            "border: 0;"
        )
        layout.addWidget(msg)

        return overlay

    def resizeEvent(self, event) -> None:
        """Keep the drop overlay sized to the central widget on resize."""
        super().resizeEvent(event)
        central = self.centralWidget()
        if central is not None and self._drop_overlay is not None:
            self._drop_overlay.setGeometry(central.rect())

    # ---------- Drag and drop ----------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept the drag if it has at least one valid PDF URL.

        Validation is conservative: every URL in the drag must be a
        local file with a .pdf extension. Mixed drops (one PDF + one
        non-PDF) are rejected as a whole so the user gets clear all-or-
        nothing behavior. Folders are rejected (toLocalFile returns a
        directory path; we check is_file()).
        """
        mime = event.mimeData()
        if not mime.hasUrls():
            event.ignore()
            return

        paths = self._extract_drop_paths(mime.urls())
        if not paths:
            event.ignore()
            return

        all_valid = all(
            p.is_file() and p.suffix.lower() in _DRAG_DROP_ALLOWED_EXTENSIONS
            for p in paths
        )
        if not all_valid:
            event.ignore()
            return

        # Accept the drag and show the overlay. raise_() makes sure it
        # paints above any child widget that happens to be at the same
        # z-level.
        event.acceptProposedAction()
        central = self.centralWidget()
        if central is not None:
            self._drop_overlay.setGeometry(central.rect())
        self._drop_overlay.raise_()
        self._drop_overlay.show()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        """Hide the overlay when the drag leaves the window."""
        self._drop_overlay.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        """Hand accepted PDF paths to the file panel's upload queue.

        At this point dragEnterEvent has already validated that every
        URL is a local PDF file — we re-extract and trust the result.
        If the panel is busy (another batch in flight), we surface a
        small message rather than silently dropping the request.
        """
        self._drop_overlay.hide()

        mime = event.mimeData()
        if not mime.hasUrls():
            event.ignore()
            return

        paths = self._extract_drop_paths(mime.urls())
        if not paths:
            event.ignore()
            return

        # Hand to the FileLibraryPanel's public upload queue. It manages
        # the busy flag, the SQLite-serialised one-at-a-time upload, the
        # per-provider fan-out, and the user-facing success/skip dialogs.
        # We do not duplicate any of that here.
        started = self.sidebar.file_panel.queue_uploads(paths)
        if not started:
            QMessageBox.information(
                self,
                "Upload in progress",
                "Another upload is already running. "
                "Wait for it to finish, then drop again.",
            )

        event.acceptProposedAction()

    @staticmethod
    def _extract_drop_paths(urls) -> list[Path]:
        """Convert dropped QUrls into local Paths.

        Filters out non-file URLs (http, ftp, etc.) which Finder /
        Explorer don't normally produce on file drops but which we
        should be defensive about. Empty list means "no usable paths."
        """
        paths: list[Path] = []
        for url in urls:
            local = url.toLocalFile()
            if not local:
                continue
            paths.append(Path(local))
        return paths

    # ---------- View switching ----------

    def _on_view_changed(self, index: int) -> None:
        if 0 <= index < self.content_stack.count():
            self.content_stack.setCurrentIndex(index)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cancel any in-flight LLM workers before the window closes.

        Without this, workers can emit signals to a destroyed dispatcher
        during teardown, producing 'Signal source has been deleted'
        errors and potentially hanging the Python process. The
        dispatcher's shutdown() cancels workers, waits up to 3s for
        clean exit, then disconnects signals.

        We delegate to ComparisonView.shutdown() rather than reaching
        into its private _dispatcher directly — keeps encapsulation
        clean, lets ComparisonView add other shutdown work later if it
        ever owns more state.
        """
        self.comparison_view.shutdown()
        super().closeEvent(event)