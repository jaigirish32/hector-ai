"""
A response card — shows one LLM's output with metrics and actions.

State machine:
    EMPTY    -> no request yet, card shows a dim placeholder
    LOADING  -> request in flight, card shows a spinner line
    COMPLETE -> response received, card shows content + metrics
    ERROR    -> request failed, card shows the error message

Supports 'up' / 'down' voting which a future analytics layer can log.

Caveats: when set_response is called with caveats (e.g. "this provider
only saw 1 of 2 attached files"), they appear in italic grey text
between the answer body and the metrics footer. Hidden when no caveats
are present and in non-COMPLETE states.
"""
from enum import Enum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models import ModelInfo, Provider


class CardState(str, Enum):
    """Which phase of the request/response cycle the card is in."""

    EMPTY = "empty"
    LOADING = "loading"
    COMPLETE = "complete"
    ERROR = "error"


# Provider accent colors — used for the initial letter badge.
PROVIDER_COLORS = {
    Provider.OPENAI:       ("#0F2E20", "#10A37F"),  # (bg, fg)
    Provider.AZURE_OPENAI: ("#0F1E35", "#4A90E2"),
    Provider.ANTHROPIC:    ("#2A1A10", "#D97757"),
    Provider.GOOGLE:       ("#0F1E35", "#4A90E2"),
    Provider.XAI:          ("#1A1A1A", "#EDEDED"),
    Provider.LOCAL:        ("#1A1A1A", "#8A8A8A"),
}


class ResponseCard(QFrame):
    """Visual card for one provider's response in a comparison run."""

    # Fires when the user votes — payload = (model_id, is_positive).
    voted = Signal(str, bool)

    # Fires when the user clicks the regenerate button — payload = model_id.
    regenerate_requested = Signal(str)

    def __init__(self, model: ModelInfo) -> None:
        super().__init__()

        self._model = model
        self._state = CardState.EMPTY

        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---------- Header ----------
        self._header = self._build_header()
        root.addWidget(self._header)

        # ---------- Body ----------
        self._body = QTextEdit()
        self._body.setObjectName("responseBody")
        self._body.setReadOnly(True)
        self._body.setMinimumHeight(140)
        self._body.setPlaceholderText("Waiting for prompt...")
        root.addWidget(self._body, stretch=1)

        # ---------- Caveats (hidden by default) ----------
        # Sits between body and footer. Italic grey text. Wraps. Visible
        # only when caveats are present and the card is in COMPLETE state.
        self._caveats_label = QLabel("")
        self._caveats_label.setObjectName("cardCaveat")
        self._caveats_label.setWordWrap(True)
        self._caveats_label.setStyleSheet(
            "color: #8A8A8A; font-size: 11px; font-style: italic; "
            "padding: 6px 12px 0 12px;"
        )
        self._caveats_label.setVisible(False)
        root.addWidget(self._caveats_label)

        # ---------- Footer (metrics + vote buttons) ----------
        self._footer = self._build_footer()
        root.addWidget(self._footer)

        self._apply_state()

    # ---------- Building blocks ----------

    def _build_header(self) -> QWidget:
        """Header strip: provider initial + model label + status badge."""
        header = QFrame()
        header.setObjectName("cardHeader")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Provider initial badge — colored square with the first letter
        initial = self._model.label[0].upper()
        bg, fg = PROVIDER_COLORS.get(self._model.provider, ("#1A1A1A", "#EDEDED"))

        badge = QLabel(initial)
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background-color: {bg}; color: {fg}; "
            f"border-radius: 6px; font-weight: 600; font-size: 13px;"
        )
        layout.addWidget(badge)

        # Label + provider name stacked vertically
        labels_col = QVBoxLayout()
        labels_col.setSpacing(1)
        labels_col.setContentsMargins(0, 0, 0, 0)

        name = QLabel(self._model.label)
        name.setObjectName("providerName")
        labels_col.addWidget(name)

        org = QLabel(self._model.provider.value.replace("_", " ").title())
        org.setObjectName("providerOrg")
        labels_col.addWidget(org)

        layout.addLayout(labels_col)
        layout.addStretch()

        # Status badge — visible only in specific states
        self._status_badge = QLabel("")
        self._status_badge.setObjectName("badge")
        self._status_badge.setVisible(False)
        layout.addWidget(self._status_badge)

        return header

    def _build_footer(self) -> QWidget:
        """Footer strip: latency / tokens / cost / vote buttons."""
        footer = QFrame()
        footer.setObjectName("cardFooter")

        layout = QHBoxLayout(footer)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(16)

        self._latency_metric = self._make_metric("LATENCY", "—")
        self._tokens_metric = self._make_metric("TOKENS", "—")
        self._cost_metric = self._make_metric("COST", "—")

        layout.addWidget(self._latency_metric)
        layout.addWidget(self._tokens_metric)
        layout.addWidget(self._cost_metric)
        layout.addStretch()

        # Vote buttons
        self._vote_up = QPushButton("↑")
        self._vote_up.setObjectName("vote")
        self._vote_up.setFixedSize(26, 22)
        self._vote_up.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vote_up.clicked.connect(lambda: self._on_vote(True))

        self._vote_down = QPushButton("↓")
        self._vote_down.setObjectName("vote")
        self._vote_down.setFixedSize(26, 22)
        self._vote_down.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vote_down.clicked.connect(lambda: self._on_vote(False))

        layout.addWidget(self._vote_up)
        layout.addWidget(self._vote_down)

        return footer

    def _make_metric(self, label_text: str, value_text: str) -> QWidget:
        """A small label-over-value column, used for latency / tokens / cost."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label = QLabel(label_text)
        label.setObjectName("metricLabel")
        layout.addWidget(label)

        value = QLabel(value_text)
        value.setObjectName("metricValue")
        layout.addWidget(value)

        # Stash the value label on the container so we can update it later.
        container.value_label = value  # type: ignore[attr-defined]
        return container

    # ---------- Public API — called from the Compare view ----------

    def set_loading(self) -> None:
        """Mark the card as waiting on a response."""
        self._state = CardState.LOADING
        self._body.setPlaceholderText("Generating response...")
        self._body.clear()
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("GENERATING", accent=False)
        self._apply_state()

    def set_response(
        self,
        text: str,
        latency_seconds: float,
        tokens: int,
        cost_usd: float,
        caveats: tuple[str, ...] = (),
    ) -> None:
        """Populate the card with a successful response.

        caveats appear under the answer in italic grey text. Pass an
        empty tuple (default) to omit. Each caveat becomes its own line.
        """
        self._state = CardState.COMPLETE
        self._body.setPlainText(text)
        self._update_metrics(
            f"{latency_seconds:.1f} s",
            f"{tokens}",
            f"${cost_usd:.4f}",
        )
        self._set_caveats(caveats)
        self._set_status("", accent=False)
        self._apply_state()

    def set_error(self, message: str) -> None:
        """Show an error state on the card."""
        self._state = CardState.ERROR
        self._body.setPlainText(f"Error: {message}")
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("FAILED", accent=False)
        self._apply_state()

    def set_badge(self, text: str, accent: bool = True) -> None:
        """Show a winner badge like 'FASTEST' or 'CHEAPEST'."""
        self._set_status(text, accent=accent)

    def reset(self) -> None:
        """Return the card to its empty state."""
        self._state = CardState.EMPTY
        self._body.clear()
        self._body.setPlaceholderText("Waiting for prompt...")
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("", accent=False)
        self._apply_state()

    @property
    def model(self) -> ModelInfo:
        """Which model this card represents."""
        return self._model

    # ---------- Internal helpers ----------

    def _update_metrics(self, latency: str, tokens: str, cost: str) -> None:
        """Update all three metric values at once."""
        self._latency_metric.value_label.setText(latency)
        self._tokens_metric.value_label.setText(tokens)
        self._cost_metric.value_label.setText(cost)

    def _set_caveats(self, caveats: tuple[str, ...]) -> None:
        """Show or hide the caveats label. Multiple caveats are joined
        with newlines so each appears on its own line."""
        if not caveats:
            self._caveats_label.setVisible(False)
            self._caveats_label.setText("")
            return
        # Each caveat on its own line, prefixed with a soft bullet so
        # multi-caveat cases read clearly.
        text = "\n".join(f"• {c}" for c in caveats)
        self._caveats_label.setText(text)
        self._caveats_label.setVisible(True)

    def _set_status(self, text: str, accent: bool) -> None:
        """Set the small status/winner badge in the header."""
        if not text:
            self._status_badge.setVisible(False)
            return
        self._status_badge.setText(text)
        self._status_badge.setObjectName("badgeGold" if accent else "badge")
        self._status_badge.style().unpolish(self._status_badge)
        self._status_badge.style().polish(self._status_badge)
        self._status_badge.setVisible(True)

    def _apply_state(self) -> None:
        """Enable/disable vote buttons based on whether a response exists."""
        is_complete = self._state == CardState.COMPLETE
        self._vote_up.setEnabled(is_complete)
        self._vote_down.setEnabled(is_complete)

    def _on_vote(self, is_positive: bool) -> None:
        """Emit the voted signal with model id and direction."""
        self.voted.emit(self._model.id, is_positive)