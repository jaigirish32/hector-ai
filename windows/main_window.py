"""
The main application window for HECTOR-AI.

Composes the sidebar (left) with a stacked content area (right) that
swaps between the Compare and Settings views.
"""
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from views.comparison_view import ComparisonView
from views.settings_view import SettingsView
from windows.sidebar import Sidebar


class MainWindow(QMainWindow):
    """The top-level window. Holds a sidebar and a stack of views."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("HECTOR-AI")
        self.setMinimumSize(1200, 780)
        self.resize(1360, 860)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.view_changed.connect(self._on_view_changed)
        root_layout.addWidget(self.sidebar)

        # Content stack — the order here MUST match the sidebar's NAV_ITEMS order.
        self.content_stack = QStackedWidget()
        root_layout.addWidget(self.content_stack, stretch=1)

        # Index 0 — Compare models
        self.comparison_view = ComparisonView()
        self.content_stack.addWidget(self.comparison_view)

        # Index 1 — Settings
        self.settings_view = SettingsView()
        self.content_stack.addWidget(self.settings_view)

    def _on_view_changed(self, index: int) -> None:
        """Switch the content stack to show the view at the given index."""
        if 0 <= index < self.content_stack.count():
            self.content_stack.setCurrentIndex(index)