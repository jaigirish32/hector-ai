"""
The prompt area — top of the Compare view.

Contains: a multi-line prompt text box, model selection chips
(loaded from the central registry), run parameters, and the
"Run comparison" button.

Files are managed in the sidebar's FILES section, not here.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from models import DEFAULT_MODELS
from widgets.model_chip import ModelChip


class PromptArea(QFrame):
    """Top panel where the user composes a prompt and picks models."""

    # User clicked Run — payload = (prompt_text, [selected_model_ids])
    run_requested = Signal(str, list)

    def __init__(self) -> None:
        super().__init__()

        self.setObjectName("card")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        self._chips: list[ModelChip] = []

        # ---------- Section 1: prompt header + text input ----------
        header_row = QHBoxLayout()
        prompt_label = QLabel("PROMPT")
        prompt_label.setObjectName("sectionLabel")
        header_row.addWidget(prompt_label)
        header_row.addStretch()

        self._char_counter = QLabel("0 / 8000")
        self._char_counter.setObjectName("hintText")
        header_row.addWidget(self._char_counter)
        root.addLayout(header_row)

        self._prompt_input = QTextEdit()
        self._prompt_input.setObjectName("promptInput")
        self._prompt_input.setPlaceholderText(
            "Ask HECTOR anything. Check files in the sidebar to include them."
        )
        self._prompt_input.setMinimumHeight(80)
        self._prompt_input.setMaximumHeight(160)
        self._prompt_input.textChanged.connect(self._update_char_counter)
        root.addWidget(self._prompt_input)

        # ---------- Section 2: model chips row ----------
        chips_row = QHBoxLayout()
        chips_row.setSpacing(6)
        chips_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        models_label = QLabel("MODELS")
        models_label.setObjectName("sectionLabel")
        chips_row.addWidget(models_label)
        chips_row.addSpacing(4)

        for model in DEFAULT_MODELS:
            if not model.enabled:
                continue
            chip = ModelChip(model.label, model.id, selected=True)
            chip.toggled_changed.connect(self._on_chip_toggled)
            self._chips.append(chip)
            chips_row.addWidget(chip)

        chips_row.addStretch()
        root.addLayout(chips_row)

        # ---------- Section 3: bottom control row ----------
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #242424;")
        separator.setFixedHeight(1)
        root.addWidget(separator)

        bottom_row = QHBoxLayout()
        params_label = QLabel(
            "Temp <b>0.7</b>    ·    Max tokens <b>2048</b>    ·    Stream <b>ON</b>"
        )
        params_label.setObjectName("hintText")
        params_label.setTextFormat(Qt.TextFormat.RichText)
        bottom_row.addWidget(params_label)

        bottom_row.addStretch()

        self._run_button = QPushButton("Run comparison  →")
        self._run_button.setObjectName("primary")
        self._run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_button.clicked.connect(self._on_run_clicked)
        bottom_row.addWidget(self._run_button)
        root.addLayout(bottom_row)
        self._update_run_button_state()

    # ---------- Public API ----------

    def selected_models(self) -> list[str]:
        return [chip.model_id for chip in self._chips if chip.is_selected()]

    def prompt_text(self) -> str:
        return self._prompt_input.toPlainText().strip()

    # ---------- Internal handlers ----------

    def _update_char_counter(self) -> None:
        count = len(self._prompt_input.toPlainText())
        self._char_counter.setText(f"{count} / 8000")

    def _on_chip_toggled(self, model_id: str, is_selected: bool) -> None:
        self._update_run_button_state()

    def _update_run_button_state(self) -> None:
        selected_count = sum(1 for chip in self._chips if chip.is_selected())
        if selected_count >= 2:
            self._run_button.setText("Run comparison  →")
        else:
            self._run_button.setText("Run  →")
        self._run_button.setEnabled(selected_count >= 1)

    def _on_run_clicked(self) -> None:
        prompt = self.prompt_text()
        models = self.selected_models()
        if not prompt or not models:
            return
        self.run_requested.emit(prompt, models)