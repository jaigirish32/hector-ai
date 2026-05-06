"""
A response card — shows one LLM's output with metrics and actions.

State machine:
    EMPTY     -> no request yet, card shows a dim placeholder
    LOADING   -> request is queued or in flight, no events yet, spinner-y badge
    STREAMING -> events are arriving from the provider, body fills in live
                 with debounced markdown→HTML rendering (v0.2.0)
    COMPLETE  -> stream finished cleanly, body shows full answer + metrics
    ERROR     -> stream errored out, card shows the error message
    CANCELLED -> user clicked Stop mid-stream; partial text preserved,
                 badge shows STOPPED. Not an error.

A copy-to-clipboard button in the header copies the response body.
The button is enabled only in the COMPLETE state and shows a brief
checkmark confirmation when clicked.

v0.2.0 incremental rendering
----------------------------
During streaming, chunks accumulate in _stream_buffer. A debounced
QTimer fires every _STREAM_RENDER_MS (default 250ms) and re-renders
the entire buffer as markdown→HTML via setHtml. Tables, headings,
code blocks, and bold/italic appear formatted as the response streams
in — much closer to claude.ai's experience than waiting until
completion to render.

Trade-offs:
- Re-render is full-document each time (markdown parsing is global —
  you can't append HTML for "the new bit" without context). Fine in
  practice; markdown is fast for typical response sizes.
- Mild flicker every 250ms during the re-render. Acceptable.
- First 250ms of a stream may briefly show as raw markdown before
  the first render kicks in. Minor.
- Scroll position preservation: if user is near the bottom (within
  5px of max), they stay at the bottom (auto-follow stream). If
  they've scrolled up to read earlier content, their position is
  preserved so they don't get yanked back.
"""
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


# SVG icons for the copy button. Defined inline so we don't bundle
# extra asset files and the icons render identically on every OS via
# Qt's SVG renderer.
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


def _svg_to_icon(svg_text: str, size: int = 20) -> QIcon:
    """Render an SVG string into a QIcon at the requested pixel size."""
    renderer = QSvgRenderer(QByteArray(svg_text.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    from PySide6.QtGui import QPainter
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)


# ---------- CSS for rendered response body ----------
#
# Applied to the QTextEdit body in both streaming and completion states.
# QTextEdit supports a subset of CSS — enough for typography, colors,
# tables, code blocks, and basic spacing. Background color matches
# BG_CARD so the body blends with the card; teal accent matches the
# GOLD brand color.

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
    """Wrap an HTML/text body in a full styled HTML document.

    Used by both streaming (rendered markdown HTML inside styled body)
    and completion (final markdown-rendered HTML inside styled body)
    paths. QTextEdit's setHtml accepts a full document or a fragment;
    we use a full document so the <style> block is recognized and
    applied.
    """
    return f"<html><head><style>{_RESPONSE_CSS}</style></head><body>{body_content}</body></html>"


# How often (in milliseconds) to re-render the streaming buffer as
# markdown→HTML. Lower = smoother incremental rendering but more
# CPU and visible flicker. Higher = less work but slower visual
# updates. 250ms is a good middle ground — feels live without
# being janky.
_STREAM_RENDER_MS = 250


class CardState(str, Enum):
    """Which phase of the request/response cycle the card is in."""

    EMPTY = "empty"
    LOADING = "loading"
    STREAMING = "streaming"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


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

        # ---------- Streaming render state ----------
        # Chunks arrive on stream_text_delta; we accumulate them in
        # _stream_buffer and re-render markdown to HTML via _render_timer
        # at most every _STREAM_RENDER_MS milliseconds. This gives
        # incremental claude.ai-style rendering during the stream
        # without thrashing setHtml on every chunk (which would flicker
        # and lose scroll position).
        self._stream_buffer = ""
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_stream_buffer)

        self._apply_state()

    # ---------- Building blocks ----------

    def _build_header(self) -> QWidget:
        """Header strip: provider initial + model label + status badge."""
        header = QFrame()
        header.setObjectName("cardHeader")

        layout = QHBoxLayout(header)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Provider initial badge
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

        # Copy button
        self._copy_icon_default = _svg_to_icon(_COPY_ICON_SVG, size=16)
        self._copy_icon_done = _svg_to_icon(_CHECK_ICON_SVG, size=16)

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
        """Footer strip: latency / tokens / cost metrics."""
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
        """A small label-over-value column."""
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
        self._render_timer.stop()
        self._stream_buffer = ""
        self._state = CardState.LOADING
        self._body.setPlaceholderText("Generating response...")
        self._body.clear()
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("GENERATING", accent=False)
        self._apply_state()

    def start_streaming(self, model_name: str) -> None:
        """Transition into the STREAMING state.

        Resets the streaming buffer and the styled body. Subsequent
        append_stream_text calls accumulate chunks in _stream_buffer;
        the debounced _render_timer converts the buffer to rendered
        markdown every _STREAM_RENDER_MS so the user sees incremental
        updates (tables forming, headings rendered) rather than raw
        markdown source. set_response at completion still applies the
        final authoritative render.

        Stops any pending render timer from a prior stream so we don't
        get a stale render arriving after a new stream has started.
        """
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
        """Append a streamed text chunk to the buffer.

        Called once per dispatcher stream_text_delta signal. Instead
        of inserting the chunk directly into the body (which would
        show raw markdown source like '|' and '#' until completion),
        we append to _stream_buffer and start a debounced timer.
        Every _STREAM_RENDER_MS, _render_stream_buffer converts the
        accumulated buffer to rendered HTML — so the user sees
        tables form, headings render, code blocks appear with proper
        styling as the response streams.

        No-op if the card is not in STREAMING state. This guards
        against late deltas arriving after the card has already
        transitioned to COMPLETE / ERROR / CANCELLED.
        """
        if self._state != CardState.STREAMING:
            return
        if not chunk:
            return

        self._stream_buffer += chunk

        # Start the timer if not already running. Once started, more
        # chunks arriving during the wait window are simply added to
        # the buffer; the timer fires once and renders everything
        # accumulated. This is the debounce — bursts of chunks
        # produce one render, not one per chunk.
        if not self._render_timer.isActive():
            self._render_timer.start(_STREAM_RENDER_MS)

    def _render_stream_buffer(self) -> None:
        """Render the accumulated streaming buffer as markdown→HTML.

        Called by _render_timer every _STREAM_RENDER_MS during streaming.
        Converts the entire accumulated buffer (not just new chunks) to
        HTML and replaces the body. This is full re-render, not
        incremental, because markdown's parsing is inherently global —
        you can't append HTML for "the new bit" without context.

        Scroll position handling: if the user is at the bottom of the
        view (auto-scroll mode), we keep them there after the re-render.
        If they've scrolled up to read earlier text, we preserve their
        scroll position so the re-render doesn't yank them back to the
        bottom mid-read.

        No-op if the card has transitioned out of STREAMING state — a
        late timer firing after stream completion would corrupt the
        final rendered state set by set_response.
        """
        if self._state != CardState.STREAMING:
            return
        if not self._stream_buffer:
            return

        # Save scroll position. We check whether the user is "near the
        # bottom" (within 5px) — if so, they're following the stream
        # and we should keep them at the bottom after re-render.
        # Otherwise they're reading earlier text and we preserve their
        # scroll position.
        scrollbar = self._body.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 5
        prior_position = scrollbar.value()

        # Render the accumulated buffer.
        html = markdown.markdown(
            self._stream_buffer,
            extensions=["tables", "fenced_code", "nl2br"],
        )
        self._body.setHtml(_wrap_with_css(html))

        # Restore scroll position after setHtml (which resets to top).
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(prior_position)

    def update_stream_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update the token metric mid-stream."""
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
        """Populate the card with a successful response.

        Stops any pending streaming-render timer (so a stale render
        doesn't overwrite our final state), then sets the authoritative
        final HTML. In normal operation the final render matches the
        last streaming render closely — the user sees a smooth
        transition rather than a "snap."
        """
        self._render_timer.stop()
        self._state = CardState.COMPLETE

        # Render markdown to HTML so tables, headers, lists, and code
        # blocks display properly. The 'tables' extension handles
        # markdown table syntax; 'fenced_code' handles ```code blocks```;
        # 'nl2br' converts single newlines to <br> so the rendered
        # output matches claude.ai's line-break behavior.
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
        """Show an error state on the card.

        Stops any pending render timer, then replaces the body with
        the error message. Any partial streamed content is overwritten
        — for an error case, showing partial text alongside the error
        would be confusing.
        """
        self._render_timer.stop()
        self._state = CardState.ERROR
        # Use styled body so error text picks up the same typography
        # as success responses.
        error_html = f"<p>Error: {message}</p>"
        self._body.setHtml(_wrap_with_css(error_html))
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("FAILED", accent=False)
        self._apply_state()

    def set_cancelled(self) -> None:
        """Mark the card as cancelled by the user.

        Called from ComparisonView when the dispatcher's stream_cancelled
        signal fires. Stops the render timer but PRESERVES whatever
        partial rendered text was last shown — the user explicitly
        chose to stop and presumably wants to see what they got. The
        STOPPED badge in the header signals the cancellation. Metrics
        stay as-is (token count from update_stream_usage if it arrived,
        otherwise dashes; latency and cost stay dashes since the stream
        didn't complete).
        """
        self._render_timer.stop()
        self._state = CardState.CANCELLED
        # Body content is preserved as-is — whatever was last rendered
        # before the user clicked Stop.
        self._set_status("STOPPED", accent=False)
        self._apply_state()

    def set_badge(self, text: str, accent: bool = True) -> None:
        """Show a winner badge like 'FASTEST' or 'CHEAPEST'."""
        self._set_status(text, accent=accent)

    def reset(self) -> None:
        """Return the card to its empty state."""
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
        """Which model this card represents."""
        return self._model

    # ---------- Internal helpers ----------

    def _update_metrics(self, latency: str, tokens: str, cost: str) -> None:
        """Update all three metric values at once."""
        self._latency_metric.value_label.setText(latency)
        self._tokens_metric.value_label.setText(tokens)
        self._cost_metric.value_label.setText(cost)

    def _set_caveats(self, caveats: tuple[str, ...]) -> None:
        """Show or hide the caveats label."""
        if not caveats:
            self._caveats_label.setVisible(False)
            self._caveats_label.setText("")
            return
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
        """Enable/disable copy button based on whether a response exists."""
        is_complete = self._state == CardState.COMPLETE
        self._copy_button.setEnabled(is_complete)

    def _on_copy(self) -> None:
        """Copy the response body to the system clipboard, preserving
        rich-text formatting.

        Uses QTextEdit.selectAll() + .copy() rather than reading
        toPlainText() and calling clipboard.setText(). The difference
        matters for tables, headers, and lists: QTextEdit.copy() puts
        BOTH HTML and plain-text formats on the clipboard, so:
          - Pasting into Word/Outlook → HTML format → tables render
          - Pasting into Notepad/terminal → plain-text fallback
        """
        if not self._body.toPlainText():
            return
        self._body.selectAll()
        self._body.copy()
        cursor = self._body.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._body.setTextCursor(cursor)
        self._copy_button.setIcon(self._copy_icon_done)
        QTimer.singleShot(1500, self._revert_copy_icon)

    def _revert_copy_icon(self) -> None:
        """Restore the copy button to its default icon."""
        self._copy_button.setIcon(self._copy_icon_default)