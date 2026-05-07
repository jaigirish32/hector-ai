"""Response card widget — one LLM's output with metrics and actions."""
from enum import Enum

from PySide6.QtCore import QByteArray, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QIcon, QPixmap, QTextCursor
from PySide6.QtSvg import QSvgRenderer
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

import markdown

from models import ModelInfo, Provider


# Inline SVG icons for the copy button.
_COPY_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="none" stroke="#A0A0A0" stroke-width="2" stroke-linecap="round"
stroke-linejoin="round">
<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
</svg>"""

_CHECK_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="none" stroke="#00D4C4" stroke-width="2.5" stroke-linecap="round"
stroke-linejoin="round">
<polyline points="20 6 9 17 4 12"/>
</svg>"""

_STOP_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="#A0A0A0" stroke="none">
<rect x="6" y="6" width="12" height="12" rx="2"/>
</svg>"""

def _svg_to_icon(svg_text: str, size: int = 20) -> QIcon:
    """Render an SVG string into a QIcon."""
    renderer = QSvgRenderer(QByteArray(svg_text.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    from PySide6.QtGui import QPainter
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


# CSS for response body — applied via setHtml in streaming and completion.
_RESPONSE_CSS = """
body {
    font-family: 'Segoe UI', 'SF Pro Display', 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 13px;
    line-height: 1.5;
    color: #EDEDED;
    background-color: #161616;
    padding: 16px 20px;
    margin: 0;
}
h1 {
    font-size: 22px;
    font-weight: 600;
    line-height: 1.3;
    color: #EDEDED;
    margin: 18px 0 6px 0;
    padding: 0;
}
h2 {
    font-size: 18px;
    font-weight: 600;
    line-height: 1.3;
    color: #EDEDED;
    margin: 4px 0 4px 0;
    padding: 0;
}
h3 {
    font-size: 15px;
    font-weight: 600;
    line-height: 1.3;
    color: #EDEDED;
    margin: 14px 0 6px 0;
}
h4, h5, h6 {
    font-size: 13px;
    font-weight: 600;
    line-height: 1.3;
    color: #EDEDED;
    margin: 12px 0 4px 0;
}
p {
    margin: 0 0 12px 0;
    color: #EDEDED;
}
ul, ol {
    margin: 4px 0 12px 0;
    padding-left: 20px;
}
li {
    margin-bottom: 4px;
    color: #EDEDED;
}
strong, b {
    color: #EDEDED;
    font-weight: 600;
}
em, i {
    color: #EDEDED;
    font-style: italic;
}
a {
    color: #00D4C4;
    text-decoration: none;
}
hr {
    border: 0;
    border-top: 1px solid #242424;
    margin: 16px 0;
    background-color: transparent;
}
table {
    border-collapse: collapse;
    margin: 12px 0;
    background-color: #1A1A1A;
}
th {
    background-color: #1D1D1D;
    color: #EDEDED;
    font-weight: 600;
    padding: 8px 14px;
    border: 1px solid #242424;
    text-align: left;
}
td {
    color: #EDEDED;
    padding: 8px 14px;
    border: 1px solid #242424;
}
code {
    font-family: 'Cascadia Code', 'SF Mono', 'Consolas', 'Menlo', monospace;
    font-size: 12px;
    background-color: #1A1A1A;
    color: #00D4C4;
    padding: 2px 6px;
    border-radius: 3px;
}
pre {
    font-family: 'Cascadia Code', 'SF Mono', 'Consolas', 'Menlo', monospace;
    font-size: 12px;
    background-color: #1A1A1A;
    color: #EDEDED;
    padding: 12px 14px;
    border: 1px solid #242424;
    border-radius: 6px;
    margin: 12px 0;
    white-space: pre-wrap;
}
pre code {
    background-color: transparent;
    color: #EDEDED;
    padding: 0;
    border-radius: 0;
}
blockquote {
    border-left: 3px solid #00D4C4;
    padding-left: 12px;
    margin: 12px 0;
    color: #9A9A9A;
}
"""


def _wrap_with_css(body_content: str) -> str:
    """Wrap HTML in a styled document for QTextEdit.setHtml."""
    return f"<html><head><style>{_RESPONSE_CSS}</style></head><body>{body_content}</body></html>"


# Debounced markdown re-render interval during streaming.
_STREAM_RENDER_MS = 250


class CardState(str, Enum):
    """Card lifecycle states.

    Typical paths:
      EMPTY → LOADING → STREAMING → COMPLETE
      EMPTY → LOADING → THINKING → STREAMING → COMPLETE  (Gemini, Anthropic)
      Any active state → CANCELLING → CANCELLED  (user clicked Stop)
      Any → ERROR
    """

    EMPTY = "empty"
    LOADING = "loading"
    THINKING = "thinking"
    STREAMING = "streaming"
    CANCELLING = "cancelling"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


# Provider accent colors for the initial-letter badge.
PROVIDER_COLORS = {
    Provider.OPENAI:       ("#0F2E20", "#10A37F"),
    Provider.AZURE_OPENAI: ("#0F1E35", "#4A90E2"),
    Provider.ANTHROPIC:    ("#2A1A10", "#D97757"),
    Provider.GOOGLE:       ("#0F1E35", "#4A90E2"),
    Provider.XAI:          ("#1A1A1A", "#EDEDED"),
    Provider.LOCAL:        ("#1A1A1A", "#8A8A8A"),
}


class ResponseCard(QFrame):
    """One provider's response card in a comparison run."""

    regenerate_requested = Signal(str)
    cancel_requested = Signal(str)

    def __init__(self, model: ModelInfo) -> None:
        super().__init__()

        self._model = model
        self._state = CardState.EMPTY

        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        self._header = self._build_header()
        root.addWidget(self._header)

        # Body
        self._body = QTextEdit()
        self._body.setObjectName("responseBody")
        self._body.setReadOnly(True)
        self._body.setMinimumHeight(140)
        self._body.setPlaceholderText("Waiting for prompt...")
        root.addWidget(self._body, stretch=1)

        # Caveats label (italic grey, hidden by default)
        self._caveats_label = QLabel("")
        self._caveats_label.setObjectName("cardCaveat")
        self._caveats_label.setWordWrap(True)
        self._caveats_label.setStyleSheet(
            "color: #8A8A8A; font-size: 11px; font-style: italic; "
            "padding: 6px 12px 0 12px;"
        )
        self._caveats_label.setVisible(False)
        root.addWidget(self._caveats_label)

        # Footer (metrics)
        self._footer = self._build_footer()
        root.addWidget(self._footer)

        # Streaming render state — debounced re-render of accumulating text.
        self._stream_buffer = ""
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_stream_buffer)

        self._apply_state()

    # ---------- Building blocks ----------

    def _build_header(self) -> QWidget:
        """Provider initial + model label + status badge + copy button."""
        header = QFrame()
        header.setObjectName("cardHeader")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

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

        self._status_badge = QLabel("")
        self._status_badge.setObjectName("badge")
        self._status_badge.setVisible(False)
        layout.addWidget(self._status_badge)

        self._copy_icon_default = _svg_to_icon(_COPY_ICON_SVG, size=16)
        self._copy_icon_done = _svg_to_icon(_CHECK_ICON_SVG, size=16)
        self._stop_icon = _svg_to_icon(_STOP_ICON_SVG, size=14)

        # Stop and Copy share the same slot — only one is visible at a time.
        # Stop appears during LOADING/THINKING/STREAMING; Copy after COMPLETE.
        self._stop_button = QPushButton()
        self._stop_button.setIcon(self._stop_icon)
        self._stop_button.setObjectName("stopButton")
        self._stop_button.setFixedSize(32, 28)
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setToolTip("Stop generating")
        self._stop_button.clicked.connect(self._on_stop)
        self._stop_button.setVisible(False)
        layout.addWidget(self._stop_button)

        self._copy_button = QPushButton()
        self._copy_button.setIcon(self._copy_icon_default)
        self._copy_button.setObjectName("copyButton")
        self._copy_button.setFixedSize(32, 28)
        self._copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._copy_button.setToolTip("Copy response to clipboard")
        self._copy_button.clicked.connect(self._on_copy)
        layout.addWidget(self._copy_button)

        return header

    def _build_footer(self) -> QWidget:
        """Latency / tokens / cost metrics."""
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

        return footer

    def _make_metric(self, label_text: str, value_text: str) -> QWidget:
        """Label-over-value column for footer metrics."""
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

        container.value_label = value  # type: ignore[attr-defined]
        return container

    # ---------- Public API ----------

    def set_loading(self) -> None:
        """Card is queued for a Run, no events arrived yet."""
        self._render_timer.stop()
        self._stream_buffer = ""
        self._state = CardState.LOADING
        self._body.setPlaceholderText("Generating response...")
        self._body.clear()
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("GENERATING", accent=False)
        self._apply_state()

    def start_thinking(self) -> None:
        """Provider is reasoning internally before producing visible text.

        Currently used by Gemini 2.5 Flash. Transitions LOADING → THINKING.
        Body stays empty; badge shows THINKING. When the first text chunk
        arrives via start_streaming, card transitions THINKING → STREAMING.
        """
        # No-op if we've moved past LOADING/THINKING (defensive against
        # late events arriving after a state change).
        if self._state not in (CardState.LOADING, CardState.THINKING):
            return
        self._state = CardState.THINKING
        self._body.setHtml(_wrap_with_css(""))
        self._body.setPlaceholderText("")
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("THINKING", accent=False)
        self._apply_state()

    def start_streaming(self, model_name: str) -> None:
        """Stream is emitting visible text. Resets buffer + render timer."""
        self._state = CardState.STREAMING
        self._stream_buffer = ""
        self._render_timer.stop()
        self._body.setHtml(_wrap_with_css(""))
        self._body.setPlaceholderText("")
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("STREAMING", accent=False)
        self._apply_state()

    def append_stream_text(self, chunk: str) -> None:
        """Buffer a chunk and start the debounced render timer."""
        if self._state != CardState.STREAMING:
            return
        if not chunk:
            return

        self._stream_buffer += chunk

        if not self._render_timer.isActive():
            self._render_timer.start(_STREAM_RENDER_MS)

    def _render_stream_buffer(self) -> None:
        """Render the streaming buffer as markdown→HTML, preserving scroll."""
        if self._state != CardState.STREAMING:
            return
        if not self._stream_buffer:
            return

        scrollbar = self._body.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 5
        prior_position = scrollbar.value()

        html = markdown.markdown(
            self._stream_buffer,
            extensions=["tables", "fenced_code", "nl2br"],
        )
        self._body.setHtml(_wrap_with_css(html))

        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(prior_position)

    def update_stream_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update token counter mid-stream."""
        if self._state != CardState.STREAMING:
            return
        total = input_tokens + output_tokens
        self._tokens_metric.value_label.setText(f"{total}")

    def set_response(
        self,
        text: str,
        latency_seconds: float,
        tokens: int,
        cost_usd: float,
        caveats: tuple[str, ...] = (),
    ) -> None:
        """Final authoritative render at completion.

        Race guard: if the user clicked Stop and we're CANCELLING/CANCELLED,
        a late-arriving completion event must not flip the card back to
        COMPLETE. The cancel wins.
        """
        if self._state in (CardState.CANCELLING, CardState.CANCELLED):
            return
        self._render_timer.stop()
        self._state = CardState.COMPLETE

        html = markdown.markdown(
            text,
            extensions=["tables", "fenced_code", "nl2br"],
        )
        self._body.setHtml(_wrap_with_css(html))

        self._update_metrics(
            f"{latency_seconds:.1f} s",
            f"{tokens}",
            f"${cost_usd:.4f}",
        )
        self._set_caveats(caveats)
        self._set_status("", accent=False)
        self._apply_state()

    def set_error(self, message: str) -> None:
        """Show error state. Overwrites any partial streamed text."""
        self._render_timer.stop()
        self._state = CardState.ERROR
        error_html = f"<p>Error: {message}</p>"
        self._body.setHtml(_wrap_with_css(error_html))
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("FAILED", accent=False)
        self._apply_state()

    def set_cancelled(self) -> None:
        """Dispatcher confirmed cancellation. Preserves whatever was rendered."""
        self._render_timer.stop()
        self._state = CardState.CANCELLED
        self._set_status("STOPPED", accent=False)
        self._apply_state()

    def set_badge(self, text: str, accent: bool = True) -> None:
        """Show 'FASTEST' / 'CHEAPEST' winner badge."""
        self._set_status(text, accent=accent)

    def reset(self) -> None:
        """Return card to empty state."""
        self._render_timer.stop()
        self._stream_buffer = ""
        self._state = CardState.EMPTY
        self._body.clear()
        self._body.setPlaceholderText("Waiting for prompt...")
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("", accent=False)
        self._apply_state()

    @property
    def model(self) -> ModelInfo:
        return self._model

    # ---------- Internal helpers ----------

    def _update_metrics(self, latency: str, tokens: str, cost: str) -> None:
        self._latency_metric.value_label.setText(latency)
        self._tokens_metric.value_label.setText(tokens)
        self._cost_metric.value_label.setText(cost)

    def _set_caveats(self, caveats: tuple[str, ...]) -> None:
        if not caveats:
            self._caveats_label.setVisible(False)
            self._caveats_label.setText("")
            return
        text = "\n".join(f"• {c}" for c in caveats)
        self._caveats_label.setText(text)
        self._caveats_label.setVisible(True)

    def _set_status(self, text: str, accent: bool) -> None:
        if not text:
            self._status_badge.setVisible(False)
            return
        self._status_badge.setText(text)
        self._status_badge.setObjectName("badgeGold" if accent else "badge")
        self._status_badge.style().unpolish(self._status_badge)
        self._status_badge.style().polish(self._status_badge)
        self._status_badge.setVisible(True)

    def _apply_state(self) -> None:
        """Toggle Stop and Copy buttons based on current state.

        Stop is visible during active states (LOADING/THINKING/STREAMING).
        It is disabled once clicked (CANCELLING) to prevent double-clicks.
        Copy is visible only after successful completion.
        """
        active_states = (
            CardState.LOADING,
            CardState.THINKING,
            CardState.STREAMING,
        )
        is_active = self._state in active_states
        is_cancelling = self._state == CardState.CANCELLING
        is_complete = self._state == CardState.COMPLETE

        self._stop_button.setVisible(is_active or is_cancelling)
        self._stop_button.setEnabled(is_active)

        self._copy_button.setVisible(not (is_active or is_cancelling))
        self._copy_button.setEnabled(is_complete)

    def _on_copy(self) -> None:
        """Rich-text copy — preserves HTML format for Word/Outlook paste."""
        if not self._body.toPlainText():
            return
        self._body.selectAll()
        self._body.copy()
        cursor = self._body.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._body.setTextCursor(cursor)
        self._copy_button.setIcon(self._copy_icon_done)
        QTimer.singleShot(1500, self._revert_copy_icon)
    
    def _on_stop(self) -> None:
        """User clicked Stop. Move to CANCELLING (button greys out),
        emit signal so ComparisonView can call dispatcher.cancel.
        Card will move to CANCELLED when StreamCancelled arrives."""
        if self._state not in (
            CardState.LOADING,
            CardState.THINKING,
            CardState.STREAMING,
        ):
            return
        self._state = CardState.CANCELLING
        self._set_status("STOPPING", accent=False)
        self._apply_state()
        self.cancel_requested.emit(self._model.id)

    def _revert_copy_icon(self) -> None:
        self._copy_button.setIcon(self._copy_icon_default)