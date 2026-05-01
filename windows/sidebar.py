"""
The left sidebar for HECTOR-AI.

Shows the expert365 logo, product name, nav buttons, the file library,
and the branding footer. Emits a signal when the user switches views.
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

from attachments.file_library import FileLibrary
from widgets.file_library_panel import FileLibraryPanel
from widgets.nav_button import NavButton
from paths import resource_path


LOGO_PATH = resource_path("assets/logo.jpeg")


class Sidebar(QWidget):
    """Left-hand navigation panel."""

    view_changed = Signal(int)

    NAV_ITEMS = [
        ("Compare models", 0),
        ("Settings", 1),
    ]

    def __init__(self, file_library: FileLibrary) -> None:
        super().__init__()

        self.setObjectName("sidebar")
        self.setFixedWidth(240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 20, 14, 16)
        layout.setSpacing(6)

        self._build_header(layout)

        # ---------- WORKSPACE nav ----------
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

        # ---------- FILES section ----------
        layout.addSpacing(14)
        self.file_panel = FileLibraryPanel(library=file_library)
        layout.addWidget(self.file_panel)

        layout.addStretch()

        self._build_footer(layout)

        self._on_nav_clicked(0)

    def _build_header(self, layout: QVBoxLayout) -> None:
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
            fallback = QLabel("expert365")
            fallback.setStyleSheet(
                "color: #C89932; font-weight: 600; font-size: 18px;"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(fallback)

        layout.addWidget(logo_container)
        layout.addSpacing(14)

        product = QLabel("HECTOR-AI")
        product.setObjectName("logo")
        layout.addWidget(product)

        tagline = QLabel("LLM comparison studio")
        tagline.setObjectName("tagline")
        layout.addWidget(tagline)

    def _build_footer(self, layout: QVBoxLayout) -> None:
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color: #242424; background-color: #242424;")
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        layout.addSpacing(8)

        brand = QLabel("Proprietary Karri product")
        brand.setObjectName("brandFooter")
        layout.addWidget(brand)

        version = QLabel("v0.1.7  · Desktop")
        version.setObjectName("brandFooter")
        layout.addWidget(version)

    def _on_nav_clicked(self, index: int) -> None:
        for button in self._nav_buttons:
            button.set_active(button._index == index)
        self.view_changed.emit(index)