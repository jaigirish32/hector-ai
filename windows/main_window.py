"""
The main application window for HECTOR-AI.

Composes the sidebar (left) with a stacked content area (right).
Owns the shared FileLibrary so the sidebar and ComparisonView share state.
"""
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from attachments.file_library import FileLibrary
from views.comparison_view import ComparisonView
from views.settings_view import SettingsView
from windows.sidebar import Sidebar
from PySide6.QtGui import QCloseEvent

class MainWindow(QMainWindow):
    """The top-level window. Holds a sidebar and a stack of views."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("HECTOR-AI")
        self.setMinimumSize(1200, 780)
        self.resize(1360, 860)

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