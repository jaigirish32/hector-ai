"""
A reusable row widget for one secret field in Settings.

Layout:
    [Label]                              [status indicator]
    [••••••••••••] [Show] [Save]
    [optional helper text]
"""
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SettingsRow(QFrame):
    """One field for entering a secret — label + password input + save button."""

    save_requested = Signal(str, str)

    def __init__(
        self,
        secret_key: str,
        label_text: str,
        placeholder: str = "Paste your API key here",
        helper_text: str = "",
    ) -> None:
        super().__init__()

        self._secret_key = secret_key
        self._is_visible = False

        self.setObjectName("settingsRow")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(8)

        # ---------- Top row: label + status indicator ----------
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)

        label = QLabel(label_text)
        label.setObjectName("fieldLabel")
        header_layout.addWidget(label)
        header_layout.addStretch()

        self._status = QLabel("")
        self._status.setObjectName("fieldStatus")
        header_layout.addWidget(self._status)

        root.addWidget(header_widget)

        # ---------- Input row: field + show/hide + save ----------
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        self._input = QLineEdit()
        self._input.setObjectName("secretInput")
        self._input.setPlaceholderText(placeholder)
        self._input.setEchoMode(QLineEdit.EchoMode.Password)
        self._input.returnPressed.connect(self._on_save_clicked)
        input_layout.addWidget(self._input, stretch=1)

        self._toggle_button = QPushButton("Show")
        self._toggle_button.setObjectName("secondary")
        self._toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_button.setFixedWidth(70)
        self._toggle_button.clicked.connect(self._toggle_visibility)
        input_layout.addWidget(self._toggle_button)

        self._save_button = QPushButton("Save")
        self._save_button.setObjectName("primary")
        self._save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_button.setFixedWidth(80)
        self._save_button.clicked.connect(self._on_save_clicked)
        input_layout.addWidget(self._save_button)

        root.addWidget(input_widget)

        # ---------- Helper text (optional) ----------
        if helper_text:
            helper = QLabel(helper_text)
            helper.setObjectName("fieldHelper")
            helper.setWordWrap(True)
            root.addWidget(helper)

    # ---------- Public API ----------

    @property
    def secret_key(self) -> str:
        return self._secret_key

    def set_value(self, value: str) -> None:
        self._input.setText(value)
        if value:
            self._set_status("Saved", good=True)
        else:
            self._set_status("", good=False)

    def show_save_success(self) -> None:
        self._set_status("Saved ✓", good=True)
        QTimer.singleShot(3000, lambda: self._set_status("Saved", good=True))

    def show_save_error(self, message: str) -> None:
        self._set_status(f"Error: {message[:40]}", good=False, is_error=True)

    def show_cleared(self) -> None:
        self._set_status("", good=False)

    # ---------- Internal ----------

    def _toggle_visibility(self) -> None:
        self._is_visible = not self._is_visible
        if self._is_visible:
            self._input.setEchoMode(QLineEdit.EchoMode.Normal)
            self._toggle_button.setText("Hide")
        else:
            self._input.setEchoMode(QLineEdit.EchoMode.Password)
            self._toggle_button.setText("Show")

    def _on_save_clicked(self) -> None:
        """Trim and emit the save signal."""
        value = self._input.text().strip()
        self.save_requested.emit(self._secret_key, value)

    def _set_status(self, text: str, good: bool, is_error: bool = False) -> None:
        self._status.setText(text)
        if is_error:
            object_name = "fieldStatusError"
        elif good:
            object_name = "fieldStatusGood"
        else:
            object_name = "fieldStatus"
        self._status.setObjectName(object_name)
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)