"""Response card widget — one LLM's output with metrics and actions."""
import html as html_module
from enum import Enum

from PySide6.QtCore import QByteArray, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap, QTextCursor
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


# Inline SVG icons.
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

_TRASH_ICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
fill="none" stroke="#A0A0A0" stroke-width="2" stroke-linecap="round"
stroke-linejoin="round">
<polyline points="3 6 5 6 21 6"/>
<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
<path d="M10 11v6"/>
<path d="M14 11v6"/>
<path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
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


# CSS for response body.
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

# Teal divider rendered as a table row.
_TURN_DIVIDER_HTML = (
    '<table width="100%" cellspacing="0" cellpadding="0"'
    ' style="margin: 16px 0; background-color: transparent;">'
    '<tr><td style="border-top: 2px solid #00D4C4; padding: 0;"></td></tr>'
    '</table>'
)


def _wrap_with_css(body_content: str) -> str:
    """Wrap HTML in a styled document for QTextEdit.setHtml."""
    return (
        f"<html><head><style>{_RESPONSE_CSS}</style></head>"
        f"<body>{body_content}</body></html>"
    )


# Debounced markdown re-render interval during streaming.
_STREAM_RENDER_MS = 250


class CardState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    THINKING = "thinking"
    STREAMING = "streaming"
    CANCELLING = "cancelling"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


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
    # Emitted when user clicks the per-card Clear History button.
    # Carries model_id so ComparisonView can route to the right store entry.
    clear_history_requested = Signal(str)

    def __init__(self, model: ModelInfo) -> None:
        super().__init__()

        self._model = model
        self._state = CardState.EMPTY
        self._history_html: str = ""
        self._current_prompt_html: str = ""

        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header = self._build_header()
        root.addWidget(self._header)

        self._body = QTextEdit()
        self._body.setObjectName("responseBody")
        self._body.setReadOnly(True)
        self._body.setMinimumHeight(140)
        self._body.setPlaceholderText("Waiting for prompt...")
        root.addWidget(self._body, stretch=1)

        self._caveats_label = QLabel("")
        self._caveats_label.setObjectName("cardCaveat")
        self._caveats_label.setWordWrap(True)
        self._caveats_label.setStyleSheet(
            "color: #8A8A8A; font-size: 11px; font-style: italic; "
            "padding: 6px 12px 0 12px;"
        )
        self._caveats_label.setVisible(False)
        root.addWidget(self._caveats_label)

        self._footer = self._build_footer()
        root.addWidget(self._footer)

        self._stream_buffer = ""
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_stream_buffer)

        self._apply_state()

    # ---------- Building blocks ----------

    def _build_header(self) -> QWidget:
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
        self._trash_icon = _svg_to_icon(_TRASH_ICON_SVG, size=14)

        self._stop_button = QPushButton()
        self._stop_button.setIcon(self._stop_icon)
        self._stop_button.setObjectName("stopButton")
        self._stop_button.setFixedSize(32, 28)
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setToolTip("Stop generating")
        self._stop_button.clicked.connect(self._on_stop)
        self._stop_button.setVisible(False)
        layout.addWidget(self._stop_button)

        # Clear History button — trash icon, same size as copy/stop.
        # Visible when not actively generating, hidden during active states.
        self._clear_history_button = QPushButton()
        self._clear_history_button.setIcon(self._trash_icon)
        self._clear_history_button.setObjectName("copyButton")
        self._clear_history_button.setFixedSize(32, 28)
        self._clear_history_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_history_button.setToolTip("Clear conversation history for this model")
        self._clear_history_button.clicked.connect(self._on_clear_history)
        layout.addWidget(self._clear_history_button)

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

    # ---------- History rendering ----------

    def _build_history_html(self, history: list[tuple[str, str]]) -> str:
        if not history:
            return ""

        parts: list[str] = []
        for user_content, assistant_content in history:
            user_escaped = html_module.escape(user_content)
            assistant_html = markdown.markdown(
                assistant_content,
                extensions=["tables", "fenced_code", "nl2br"],
            )
            parts.append(
                '<div style="opacity: 0.55; margin-bottom: 16px;'
                ' padding-bottom: 16px;'
                ' border-bottom: 1px solid #1E1E1E;">'
                '<div style="color: #5E5E5E; font-size: 10px;'
                ' font-weight: 600; letter-spacing: 0.08em;'
                ' margin-bottom: 4px;">YOU</div>'
                '<div style="background-color: #1A1A1A; border-radius: 6px;'
                ' padding: 8px 12px; color: #9A9A9A; font-size: 12px;'
                f' white-space: pre-wrap; word-break: break-word;">'
                f'{user_escaped}</div>'
                '<div style="color: #5E5E5E; font-size: 10px;'
                ' font-weight: 600; letter-spacing: 0.08em;'
                ' margin-top: 10px; margin-bottom: 4px;">ASSISTANT</div>'
                f'{assistant_html}'
                '</div>'
            )

        parts.append(_TURN_DIVIDER_HTML)
        return "".join(parts)

    def _build_prompt_html(self, prompt: str) -> str:
        if not prompt:
            return ""

        user_escaped = html_module.escape(prompt)
        return (
            '<div style="color: #8A8A8A; font-size: 10px;'
            ' font-weight: 600; letter-spacing: 0.08em;'
            ' margin-bottom: 4px;">YOU</div>'
            '<div style="background-color: #1A1A1A; border-radius: 6px;'
            ' padding: 8px 12px; color: #EDEDED; font-size: 12px;'
            f' white-space: pre-wrap; word-break: break-word;">'
            f'{user_escaped}</div>'
            '<div style="color: #8A8A8A; font-size: 10px;'
            ' font-weight: 600; letter-spacing: 0.08em;'
            ' margin-top: 10px; margin-bottom: 4px;">ASSISTANT</div>'
        )

    # ---------- Public API ----------

    def set_loading(
        self,
        history: list[tuple[str, str]] | None = None,
        current_prompt: str = "",
    ) -> None:
        self._render_timer.stop()
        self._stream_buffer = ""
        self._state = CardState.LOADING

        self._history_html = self._build_history_html(history or [])
        self._current_prompt_html = self._build_prompt_html(current_prompt)

        if self._history_html or self._current_prompt_html:
            body_html = (
                self._history_html
                + self._current_prompt_html
                + '<p style="color: #5E5E5E; font-size: 12px;'
                ' font-style: italic; margin-top: 8px;">'
                'Generating response...</p>'
            )
            self._body.setPlaceholderText("")
            self._body.setHtml(_wrap_with_css(body_html))
            self._body.verticalScrollBar().setValue(
                self._body.verticalScrollBar().maximum()
            )
        else:
            self._body.clear()
            self._body.setPlaceholderText("Generating response...")

        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("GENERATING", accent=False)
        self._apply_state()

    def start_thinking(self) -> None:
        if self._state not in (CardState.LOADING, CardState.THINKING):
            return
        self._state = CardState.THINKING
        self._body.setHtml(
            _wrap_with_css(self._history_html + self._current_prompt_html)
        )
        self._body.verticalScrollBar().setValue(
            self._body.verticalScrollBar().maximum()
        )
        self._body.setPlaceholderText("")
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("THINKING", accent=False)
        self._apply_state()

    def start_streaming(self, model_name: str) -> None:
        self._state = CardState.STREAMING
        self._stream_buffer = ""
        self._render_timer.stop()
        self._body.setHtml(
            _wrap_with_css(self._history_html + self._current_prompt_html)
        )
        self._body.verticalScrollBar().setValue(
            self._body.verticalScrollBar().maximum()
        )
        self._body.setPlaceholderText("")
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("STREAMING", accent=False)
        self._apply_state()

    def append_stream_text(self, chunk: str) -> None:
        if self._state != CardState.STREAMING:
            return
        if not chunk:
            return
        self._stream_buffer += chunk
        if not self._render_timer.isActive():
            self._render_timer.start(_STREAM_RENDER_MS)

    def _render_stream_buffer(self) -> None:
        if self._state != CardState.STREAMING:
            return
        if not self._stream_buffer:
            return

        scrollbar = self._body.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 5
        prior_position = scrollbar.value()

        current_html = markdown.markdown(
            self._stream_buffer,
            extensions=["tables", "fenced_code", "nl2br"],
        )
        self._body.setHtml(
            _wrap_with_css(
                self._history_html + self._current_prompt_html + current_html
            )
        )

        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(prior_position)

    def update_stream_usage(self, input_tokens: int, output_tokens: int) -> None:
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
        if self._state in (CardState.CANCELLING, CardState.CANCELLED):
            return
        self._render_timer.stop()
        self._state = CardState.COMPLETE

        current_html = markdown.markdown(
            text,
            extensions=["tables", "fenced_code", "nl2br"],
        )
        self._body.setHtml(
            _wrap_with_css(
                self._history_html + self._current_prompt_html + current_html
            )
        )
        self._body.verticalScrollBar().setValue(
            self._body.verticalScrollBar().maximum()
        )
        self._update_metrics(
            f"{latency_seconds:.1f} s",
            f"{tokens}",
            f"${cost_usd:.4f}",
        )
        self._set_caveats(caveats)
        self._set_status("", accent=False)
        self._apply_state()

    def set_error(self, message: str) -> None:
        self._render_timer.stop()
        self._state = CardState.ERROR
        error_html = f"<p>Error: {message}</p>"
        self._body.setHtml(
            _wrap_with_css(
                self._history_html + self._current_prompt_html + error_html
            )
        )
        self._body.verticalScrollBar().setValue(
            self._body.verticalScrollBar().maximum()
        )
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("FAILED", accent=False)
        self._apply_state()

    def set_cancelled(self) -> None:
        self._render_timer.stop()
        self._state = CardState.CANCELLED
        self._set_status("STOPPED", accent=False)
        self._apply_state()

    def set_badge(self, text: str, accent: bool = True) -> None:
        self._set_status(text, accent=accent)

    def reset(self) -> None:
        self._render_timer.stop()
        self._stream_buffer = ""
        self._history_html = ""
        self._current_prompt_html = ""
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
        """Toggle buttons based on current state.

        Stop   — visible during active generation.
        Trash  — visible when not actively generating (idle, complete, error).
        Copy   — visible when not actively generating; enabled only on COMPLETE.
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

        # Trash and Copy share the same visibility rule — hidden during
        # active generation, visible at all other times.
        not_generating = not (is_active or is_cancelling)
        self._clear_history_button.setVisible(not_generating)
        self._clear_history_button.setEnabled(not_generating)

        self._copy_button.setVisible(not_generating)
        self._copy_button.setEnabled(is_complete)

    def _on_copy(self) -> None:
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

    def _on_clear_history(self) -> None:
        """User clicked the trash icon — emit signal with model_id.
        ComparisonView handles the actual store deletion and UI reset.
        """
        self.clear_history_requested.emit(self._model.id)

    def _revert_copy_icon(self) -> None:
        self._copy_button.setIcon(self._copy_icon_default)