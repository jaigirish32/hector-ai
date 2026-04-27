"""
The left sidebar for HECTOR-AI.

Shows the expert365 logo, product name, nav buttons, and branding footer.
Emits a signal when the user switches views so the main window can react.
"""
from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from widgets.nav_button import NavButton

# Absolute path to the logo so it resolves regardless of where Python is run from.
LOGO_PATH = Path(__file__).parent.parent / "assets" / "logo.png"


class Sidebar(QWidget):
    """Left-hand navigation panel."""

    # Fires when the user selects a different view. Payload = view index.
    view_changed = Signal(int)

    # Nav items in display order. Tuple of (label, index).
    NAV_ITEMS = [
    ("Compare models", 0),
    ("Settings", 1),
    ]

    def __init__(self) -> None:
        super().__init__()

        self.setObjectName("sidebar")
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 20, 14, 16)
        layout.setSpacing(6)

        # ---------- Top: header block ----------
        self._build_header(layout)

        # ---------- Middle: nav section ----------
        layout.addSpacing(12)
        section_label = QLabel("WORKSPACE")
        section_label.setObjectName("sectionLabel")
        layout.addWidget(section_label)
        layout.addSpacing(4)

        self._nav_buttons: list[NavButton] = []
        for label, index in self.NAV_ITEMS:
            button = NavButton(label, index)
            button.clicked_with_index.connect(self._on_nav_clicked)
            self._nav_buttons.append(button)
            layout.addWidget(button)

        # Push the footer to the bottom of the sidebar.
        layout.addStretch()

        # ---------- Bottom: branding footer ----------
        self._build_footer(layout)

        # Start with "Compare models" selected.
        self._on_nav_clicked(0)

    def _build_header(self, layout: QVBoxLayout) -> None:
        """Add the expert365 logo, product name, and tagline."""
        # Logo inside a rounded off-white container so its light background
        # looks intentional against the dark sidebar.
        logo_container = QFrame()
        logo_container.setObjectName("logoContainer")
        container_layout = QVBoxLayout(logo_container)
        container_layout.setContentsMargins(10, 8, 10, 8)

        logo_pixmap = QPixmap(str(LOGO_PATH))
        if not logo_pixmap.isNull():
            scaled = logo_pixmap.scaledToWidth(
                170, Qt.TransformationMode.SmoothTransformation
            )
            logo_image = QLabel()
            logo_image.setPixmap(scaled)
            logo_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(logo_image)
        else:
            # Fallback if the logo file is missing or unreadable.
            fallback = QLabel("expert365")
            fallback.setStyleSheet(
                "color: #C89932; font-weight: 600; font-size: 18px;"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(fallback)

        layout.addWidget(logo_container)
        layout.addSpacing(14)

        # Product name beneath the company logo.
        product = QLabel("HECTOR-AI")
        product.setObjectName("logo")
        layout.addWidget(product)

        tagline = QLabel("LLM comparison studio")
        tagline.setObjectName("tagline")
        layout.addWidget(tagline)

    def _build_footer(self, layout: QVBoxLayout) -> None:
        """Add a thin separator and the branding footer at the bottom."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #242424; background-color: #242424;")
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        layout.addSpacing(8)

        brand = QLabel("Proprietary Karri product")
        brand.setObjectName("brandFooter")
        brand.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(brand)

        version = QLabel("v0.1.0 · Desktop")
        version.setObjectName("brandFooter")
        version.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(version)

    def _on_nav_clicked(self, index: int) -> None:
        """Update active state of nav buttons and broadcast the change."""
        for button in self._nav_buttons:
            button.set_active(button._index == index)
        self.view_changed.emit(index)