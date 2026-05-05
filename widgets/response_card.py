"""
A response card — shows one LLM's output with metrics and actions.

State machine:
    EMPTY     -> no request yet, card shows a dim placeholder
    LOADING   -> request is queued or in flight, no events yet, spinner-y badge
    STREAMING -> events are arriving from the provider, body fills in live
    COMPLETE  -> stream finished cleanly, body shows full answer + metrics
    ERROR     -> stream errored out, card shows the error message
    CANCELLED -> user clicked Stop mid-stream; partial text preserved,
                 badge shows STOPPED. Not an error.

A copy-to-clipboard button in the header copies the response body.
The button is enabled only in the COMPLETE state and shows a brief
checkmark confirmation when clicked. (Streaming and cancelled states
leave the copy button disabled — partial text is incomplete; copying
it would be confusing. The user can still select-and-Ctrl-C from the
text widget directly if they really want to.)

Caveats: when set_response is called with caveats (e.g. "this provider
only saw 1 of 2 attached files"), they appear in italic grey text
between the answer body and the metrics footer. Hidden when no caveats
are present and in non-COMPLETE states.

v0.2.0 streaming
----------------
The card supports live text streaming via four new methods that the
ComparisonView calls in response to dispatcher streaming signals:

    start_streaming(model_name)         — STREAMING state, body cleared
    append_stream_text(chunk)           — append a text chunk live
    update_stream_usage(input, output)  — update token metric mid-stream
    set_cancelled()                     — CANCELLED state, body preserved

The existing set_response / set_error / set_loading / reset / set_badge
methods are unchanged in shape and behaviour. Streaming is additive:
during a normal Run the sequence is set_loading -> start_streaming ->
many append_stream_text -> update_stream_usage -> set_response (which
writes the authoritative final text from ChatResponse, replacing the
streamed buffer). The "snap" from streamed text to authoritative text
at completion should be invisible in practice — the SDK assembles the
final text from the same deltas we received.
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
# Qt's SVG renderer. The currentColor pattern means the icon picks
# up its color from the surrounding stylesheet.
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


class CardState(str, Enum):
    """Which phase of the request/response cycle the card is in.

    Six states, ordered by typical lifecycle:
        EMPTY     - card just created, no request yet
        LOADING   - request dispatched, waiting for first event
        STREAMING - events arriving, body filling in live (v0.2.0)
        COMPLETE  - terminal happy path
        ERROR     - terminal error path
        CANCELLED - terminal cancelled path (user clicked Stop) (v0.2.0)
    """

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


import markdown


# ---------- CSS for rendered response body ----------
#
# Applied to the QTextEdit body in both streaming and completion states.
# QTextEdit supports a subset of CSS — enough for typography, colors,
# tables, code blocks, and basic spacing. Background color matches
# BG_CARD so the body blends with the card; teal accent matches the
# GOLD brand color.
#
# QTextEdit does NOT support: flexbox, transitions, hover states,
# box-shadow, or advanced selectors. For pixel-perfect claude.ai-style
# rendering we would need QWebEngineView (Chromium) — not worth the
# Mac packaging complexity for v0.2.0.

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
    margin: 14px 0 4px 0;
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

    Used by both streaming (plain text inside styled body) and
    completion (markdown-rendered HTML inside styled body) paths.
    QTextEdit's setHtml accepts a full document or a fragment; we
    use a full document so the <style> block is recognized and
    applied. The body content is inserted as-is — for streaming
    paths it's plain text; for completion paths it's rendered
    markdown HTML.
    """
    return f"<html><head><style>{_RESPONSE_CSS}</style></head><body>{body_content}</body></html>"


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

       # Copy button — copies the response body to clipboard.
        # Enabled only in COMPLETE state. Shows a checkmark icon for
        # ~1.5s after click as confirmation, then reverts. Uses an
        # inline SVG icon so it renders identically on Windows and
        # macOS (avoids font-glyph fallback issues with Unicode
        # symbols on macOS).
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
        """Mark the card as waiting on a response.

        Called from ComparisonView right after the Run button is
        clicked, before any provider events have arrived. The brief
        moment between dispatch and the first stream_started event
        lives here. As soon as start_streaming() is called, the card
        leaves LOADING for STREAMING.
        """
        self._state = CardState.LOADING
        self._body.setPlaceholderText("Generating response...")
        self._body.clear()
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("GENERATING", accent=False)
        self._apply_state()

    def start_streaming(self, model_name: str) -> None:
        """..."""
        self._state = CardState.STREAMING
        # Establish the styled empty body. Subsequent append_stream_text
        # calls insert plain text inside this styled document, so the
        # streaming text picks up the body's font, color, and line-height
        # from the CSS. Markdown characters (|, #, **) appear literally
        # during streaming — they'll be rendered properly when set_response
        # finalizes the card with markdown→HTML conversion.
        self._body.setHtml(_wrap_with_css(""))
        self._body.setPlaceholderText("")
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("STREAMING", accent=False)
        self._apply_state()

    def append_stream_text(self, chunk: str) -> None:
        """Append a streamed text chunk to the body.

        Called once per dispatcher stream_text_delta signal. Uses
        QTextCursor positioned at the end of the document so we
        insert without re-rendering the whole text — efficient even
        for hundreds of chunks per second. After insertion, scrolls
        the view to the bottom so the latest text is always visible.

        No-op if the card is not in STREAMING state. This guards
        against late deltas arriving after the card has already
        transitioned to COMPLETE / ERROR / CANCELLED — the dispatcher
        layer should not deliver these in correct operation, but a
        defensive check keeps the UI stable if it ever does.
        """
        if self._state != CardState.STREAMING:
            return
        if not chunk:
            return

        # Insert at the end of the document. Using a cursor positioned
        # at End is the canonical Qt pattern for streaming append; it
        # avoids re-flowing the entire document the way setPlainText
        # would, and preserves any selection the user has elsewhere
        # in the text.
        cursor = self._body.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._body.setTextCursor(cursor)
        self._body.insertPlainText(chunk)

        # Always auto-scroll to the bottom for v0.2.0. A future polish
        # step will detect "user has scrolled up to read" and skip the
        # auto-scroll in that case (claude.ai-style behaviour). For
        # now: simple, predictable, latest text always visible.
        scroll_bar = self._body.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def update_stream_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Update the token metric mid-stream.

        Called once per dispatcher stream_usage signal. Most providers
        emit usage near or at the end of a stream, so this typically
        fires shortly before set_response is called. The token counter
        flips from "—" to the real numbers a moment before completion.

        No-op if the card is not in STREAMING state. As with
        append_stream_text, this is defensive — late events shouldn't
        arrive but if they do, ignore them rather than corrupt a
        completed/cancelled card's display.
        """
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
        
        self._state = CardState.COMPLETE

        # Render markdown to HTML so tables, headers, lists, and code
        # blocks display properly. The 'tables' extension handles
        # markdown table syntax; 'fenced_code' handles ```code blocks```;
        # 'nl2br' converts single newlines to <br> so the rendered
        # output matches claude.ai's line-break behavior. We use
        # setHtml() instead of setPlainText() — QTextEdit's rich-text
        # mode handles the HTML and the clipboard automatically gets
        # both HTML and plain-text formats.
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
        """..."""
        self._state = CardState.ERROR
        # Use styled body so error text picks up the same typography
        # as success responses. Wrapping in a <p> tag makes the error
        # message a proper paragraph rather than raw text appended to
        # the body root.
        error_html = f"<p>Error: {message}</p>"
        self._body.setHtml(_wrap_with_css(error_html))
        self._update_metrics("—", "—", "—")
        self._set_caveats(())
        self._set_status("FAILED", accent=False)
        self._apply_state()

    def set_cancelled(self) -> None:
        """Mark the card as cancelled by the user (Stop pressed mid-stream).

        Called from ComparisonView when the dispatcher's stream_cancelled
        signal fires. UNLIKE set_error, this preserves whatever partial
        text was already streamed into the body — the user explicitly
        chose to stop and presumably wants to see what they got. The
        STOPPED badge in the header signals the cancellation. Metrics
        stay as-is (token count from update_stream_usage if it arrived,
        otherwise dashes; latency and cost stay dashes since the stream
        didn't complete).
        """
        self._state = CardState.CANCELLED
        # Body text is preserved as-is — whatever streamed in before
        # the user clicked Stop. No call to setPlainText or clear().
        self._set_status("STOPPED", accent=False)
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
        """Enable/disable copy button based on whether a response exists.

        Copy is enabled ONLY in COMPLETE state. STREAMING and CANCELLED
        leave it disabled because the buffer is incomplete (streaming)
        or partial (cancelled) — copying it would set the user up for
        confusion. Power users can still select+Ctrl-C from the text
        widget directly if they want partial text.
        """
        is_complete = self._state == CardState.COMPLETE
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

    def _revert_copy_icon(self) -> None:
        """Restore the copy button to its default icon."""
        self._copy_button.setIcon(self._copy_icon_default)