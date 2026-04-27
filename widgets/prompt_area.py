"""
The prompt area — top of the Compare view.

Contains: a multi-line prompt text box, attached-file chips, model
selection chips (loaded from the central registry), run parameters,
and the "Run comparison" button.
"""
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models import DEFAULT_MODELS
from widgets.model_chip import ModelChip


class PromptArea(QFrame):
    """Top panel where the user composes a prompt and picks models."""

    # User clicked Run — payload = (prompt_text, [selected_model_ids], [file_paths_as_strings])
    run_requested = Signal(str, list, list)

    def __init__(self) -> None:
        super().__init__()

        self.setObjectName("card")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        self._attached_files: list[Path] = []
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
            "Ask HECTOR anything. Attach a PDF to compare how each model reads it."
        )
        self._prompt_input.setMinimumHeight(80)
        self._prompt_input.setMaximumHeight(160)
        self._prompt_input.textChanged.connect(self._update_char_counter)
        root.addWidget(self._prompt_input)

        # ---------- Section 2: attached files row ----------
        self._files_row = QHBoxLayout()
        self._files_row.setSpacing(6)
        self._files_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._attach_button = QPushButton("+ Attach file")
        self._attach_button.setObjectName("secondary")
        self._attach_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._attach_button.clicked.connect(self._open_file_picker)
        self._files_row.addWidget(self._attach_button)
        self._files_row.addStretch()
        root.addLayout(self._files_row)

        # ---------- Section 3: model chips row (loaded from registry) ----------
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

        # ---------- Section 4: bottom control row (params + run button) ----------
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
        """Return model IDs of currently-checked chips."""
        return [chip.model_id for chip in self._chips if chip.is_selected()]

    def prompt_text(self) -> str:
        """Return the current prompt, stripped of leading/trailing whitespace."""
        return self._prompt_input.toPlainText().strip()

    def attached_file_paths(self) -> list[Path]:
        """Return the list of currently-attached file paths."""
        return list(self._attached_files)

    # ---------- Internal handlers ----------

    def _update_char_counter(self) -> None:
        count = len(self._prompt_input.toPlainText())
        self._char_counter.setText(f"{count} / 8000")

    def _open_file_picker(self) -> None:
        # Filter narrowed to PDF for now. Excel and image support
        # arrive in Phase 2e; we'll widen the filter at that point.
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach files",
            "",
            "PDF documents (*.pdf);;All files (*)",
        )
        for path_str in paths:
            self._add_attached_file(Path(path_str))

    def _add_attached_file(self, path: Path) -> None:
        if path in self._attached_files:
            return
        self._attached_files.append(path)

        chip = _AttachedFileChip(path)
        chip.remove_requested.connect(self._remove_attached_file)
        insert_index = self._files_row.count() - 1
        self._files_row.insertWidget(insert_index, chip)

    def _remove_attached_file(self, path: Path, widget: QWidget) -> None:
        if path in self._attached_files:
            self._attached_files.remove(path)
        widget.setParent(None)
        widget.deleteLater()

    def _on_chip_toggled(self, model_id: str, is_selected: bool) -> None:
        """A chip was toggled — re-evaluate the run button's state."""
        self._update_run_button_state()

    def _update_run_button_state(self) -> None:
        """Update the Run button label and enabled state based on chip count."""
        selected_count = sum(1 for chip in self._chips if chip.is_selected())

        if selected_count >= 2:
            self._run_button.setText("Run comparison  →")
        else:
            self._run_button.setText("Run  →")

        # Disable the button if nothing is selected.
        self._run_button.setEnabled(selected_count >= 1)

    def _on_run_clicked(self) -> None:
        prompt = self.prompt_text()
        models = self.selected_models()
        files = self.attached_file_paths()
        if not prompt or not models:
            return
        self.run_requested.emit(prompt, models, [str(p) for p in files])


class _AttachedFileChip(QFrame):
    """A small pill showing an attached filename with a remove button."""

    remove_requested = Signal(Path, QWidget)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self.setObjectName("fileChip")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 6, 4)
        layout.setSpacing(6)

        display_name = path.name
        if len(display_name) > 28:
            display_name = display_name[:25] + "..."

        name_label = QLabel(display_name)
        name_label.setObjectName("fileChipName")
        layout.addWidget(name_label)

        size_kb = max(1, path.stat().st_size // 1024) if path.exists() else 0
        size_label = QLabel(f"{size_kb} KB")
        size_label.setObjectName("fileChipSize")
        layout.addWidget(size_label)

        remove_btn = QPushButton("×")
        remove_btn.setObjectName("fileChipRemove")
        remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_btn.setFixedSize(18, 18)
        remove_btn.clicked.connect(self._emit_remove)
        layout.addWidget(remove_btn)

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _emit_remove(self) -> None:
        self.remove_requested.emit(self._path, self)