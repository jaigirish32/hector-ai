"""
A toggleable chip for selecting which LLM models participate in a comparison.

Click to include/exclude the model. Visual state flips between dim
(not selected) and gold-tinted (selected). Emits a signal so the parent
can track how many models are active.
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton


class ModelChip(QPushButton):
    """A toggle chip representing one LLM model (e.g., 'GPT-4o')."""

    # Fires whenever the chip is toggled. Payload = (model_id, is_selected).
    toggled_changed = Signal(str, bool)

    def __init__(
        self,
        label: str,
        model_id: str,
        selected: bool = True,
    ) -> None:
        super().__init__(label)

        self._model_id = model_id

        self.setObjectName("chip")
        self.setCheckable(True)
        self.setChecked(selected)
        self.setProperty("selected", selected)
        self._sync_style()

        from PySide6.QtCore import Qt
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Built-in QPushButton.toggled signal fires when the checked state
        # changes. We translate it into our signal that carries the model_id.
        self.toggled.connect(self._on_toggled)

    @property
    def model_id(self) -> str:
        """Stable identifier for this model (e.g., 'gpt-4o')."""
        return self._model_id

    def is_selected(self) -> bool:
        """Return True if the chip is currently toggled on."""
        return self.isChecked()

    def _on_toggled(self, checked: bool) -> None:
        """Respond to the button's toggled signal."""
        self.setProperty("selected", checked)
        self._sync_style()
        self.toggled_changed.emit(self._model_id, checked)

    def _sync_style(self) -> None:
        """Force Qt to re-read the stylesheet after a property change."""
        self.style().unpolish(self)
        self.style().polish(self)