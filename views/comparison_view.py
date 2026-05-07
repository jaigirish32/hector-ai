"""
The Compare view — workspace where users run prompts against multiple LLMs
side by side.

Reads selected files from the sidebar's FileLibraryPanel and dispatches
real parallel API calls via the Dispatcher.

v0.2.0 streaming
----------------
Connects to the dispatcher's streaming signals (stream_started,
stream_thinking, stream_text_delta, stream_usage, stream_cancelled)
in addition to the existing terminal signals (response_received,
response_failed, all_complete). Each streaming signal routes to the
matching ResponseCard method so the user sees text appear live,
word-by-word, the way claude.ai and chatgpt.com do.

For thinking providers (Gemini 2.5 Flash) stream_thinking arrives
before any text, prompting the card to show a THINKING badge during
the model's internal reasoning phase. When the first text chunk
arrives, stream_started transitions the card to STREAMING.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from attachments.file_library import FileLibrary
from models import ModelInfo, Provider, get_model
from providers.base import ChatResponse, FileRef
from providers.dispatcher import Dispatcher
from widgets.file_library_panel import FileLibraryPanel
from widgets.prompt_area import PromptArea
from widgets.response_card import ResponseCard


# Same provider key mapping used by the dispatcher's orchestrator path.
_PROVIDER_TO_LIBRARY_KEY: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.AZURE_OPENAI: "azure_openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE: "gemini",
}


# ---------- Hardcoded chat defaults (temporary) ----------

SYSTEM_PROMPT = """You are a thoughtful, senior technical advisor.

## Response style
- Lead with the most important takeaway. Never open with 
  acknowledgement or framing — get to the point immediately.
- Match depth to complexity. One-sentence questions get 
  one-paragraph answers. Deep technical content gets thorough analysis.
- Explain your reasoning, including trade-offs and alternatives 
  you considered but rejected and why.
- Flag risks, missing information, and ambiguities honestly 
  rather than hedging with qualifications.

## When analyzing content (code, documents, prompts)
- Reference specific sections, terms, lines, or values by name.
- Never describe what something does in general if you can 
  point to the exact place it does it.
- Explicitly distinguish between claims that are evidenced 
  and claims that are merely asserted without proof.
- If the content contradicts the user's assumptions, 
  flag that conflict before answering.

## When uncertain
- State your interpretation of an ambiguous question 
  explicitly before answering.
- If you lack enough context to answer confidently, 
  say precisely what's missing rather than hedging.

## Avoid
- Filler openers: "Great question", "I'd be happy to help", 
  "Certainly", "Of course"
- Vague generalities when specifics are available
- Restating the question before answering it
"""

MAX_OUTPUT_TOKENS = 32768


class ComparisonView(QWidget):
    """Multi-LLM comparison workspace."""

    def __init__(self, file_library: FileLibrary) -> None:
        super().__init__()

        self._cards: dict[str, ResponseCard] = {}
        self._file_library = file_library
        self._file_panel: FileLibraryPanel | None = None
        self._dispatcher = Dispatcher()

        # Existing terminal-signal connections — unchanged.
        self._dispatcher.response_received.connect(self._on_response_received)
        self._dispatcher.response_failed.connect(self._on_response_failed)
        self._dispatcher.all_complete.connect(self._on_all_complete)

        # v0.2.0 streaming-signal connections.
        self._dispatcher.stream_started.connect(self._on_stream_started)
        self._dispatcher.stream_thinking.connect(self._on_stream_thinking)
        self._dispatcher.stream_text_delta.connect(self._on_stream_text_delta)
        self._dispatcher.stream_usage.connect(self._on_stream_usage)
        self._dispatcher.stream_cancelled.connect(self._on_stream_cancelled)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        self._prompt_area = PromptArea()
        self._prompt_area.run_requested.connect(self._on_run_requested)
        root.addWidget(self._prompt_area)

        self._cards_scroll = QScrollArea()
        self._cards_scroll.setWidgetResizable(True)
        self._cards_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._cards_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._cards_container = QWidget()
        self._cards_grid = QGridLayout(self._cards_container)
        self._cards_grid.setContentsMargins(0, 0, 0, 0)
        self._cards_grid.setSpacing(12)

        self._cards_scroll.setWidget(self._cards_container)
        root.addWidget(self._cards_scroll, stretch=1)

        self._empty_state = self._build_empty_state()
        self._cards_grid.addWidget(self._empty_state, 0, 0)

    # ---------- Setup wiring ----------

    def set_file_panel(self, panel: FileLibraryPanel) -> None:
        """Inject the sidebar's file panel so we can read its selection at Run time."""
        self._file_panel = panel

    # ---------- UI helpers ----------

    def _build_empty_state(self) -> QWidget:
        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        heading = QLabel("Ready when you are.")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("color: #9A9A9A; font-size: 16px; font-weight: 500;")
        layout.addWidget(heading)

        hint = QLabel(
            "Type a prompt above, pick your models, and click "
            "Run to fan out across providers."
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color: #5E5E5E; font-size: 12px;")
        hint.setWordWrap(True)
        hint.setMaximumWidth(420)
        layout.addWidget(hint)

        return container

    def _clear_cards_grid(self) -> None:
        while self._cards_grid.count() > 0:
            item = self._cards_grid.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cards.clear()

    def _lay_out_cards(self, models: list[ModelInfo]) -> None:
        self._clear_cards_grid()

        for index, model in enumerate(models):
            card = ResponseCard(model)
            card.cancel_requested.connect(self._on_cancel_requested)
            self._cards[model.id] = card

            row = index // 2
            col = index % 2
            self._cards_grid.addWidget(card, row, col)

        self._cards_grid.setColumnStretch(0, 1)
        self._cards_grid.setColumnStretch(1, 1)

    # ---------- Run dispatch ----------

    def _on_run_requested(
        self,
        prompt: str,
        model_ids: list,
    ) -> None:
        """User clicked Run — fan out to selected models with checked files."""
        models = [m for m_id in model_ids if (m := get_model(m_id)) is not None]
        if not models:
            return

        self._lay_out_cards(models)
        for card in self._cards.values():
            card.set_loading()

        # Per-model: build pre-resolved file_refs from the library, since
        # the files were already uploaded at attach time.
        selected_ids: list[int] = (
            self._file_panel.selected_file_ids() if self._file_panel else []
        )

        per_model_refs: dict[str, tuple[FileRef, ...]] = {}
        for model in models:
            provider_key = _PROVIDER_TO_LIBRARY_KEY.get(model.provider)
            if provider_key is None or not selected_ids:
                per_model_refs[model.id] = ()
                continue
            refs = self._file_library.get_refs_for_provider(selected_ids, provider_key)
            per_model_refs[model.id] = tuple(refs)

        self._dispatcher.dispatch_with_resolved_refs(
            prompt=prompt,
            model_ids=model_ids,
            per_model_refs=per_model_refs,
            attached_file_ids=selected_ids,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    # ---------- Dispatcher signal handlers ----------

    def _on_response_received(self, model_id: str, response: ChatResponse) -> None:
        """Terminal happy path."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.set_response(
            text=response.text,
            latency_seconds=response.latency_seconds,
            tokens=response.input_tokens + response.output_tokens,
            cost_usd=response.cost_usd,
            caveats=response.caveats,
        )

    def _on_response_failed(self, model_id: str, message: str) -> None:
        """Terminal error path."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.set_error(message)

    def _on_all_complete(self) -> None:
        self._maybe_award_badges()

    # ---------- v0.2.0 streaming signal handlers ----------

    def _on_stream_started(self, model_id: str, served_model: str) -> None:
        """First text-producing event of a stream. Card transitions
        LOADING (or THINKING for thinking providers) -> STREAMING."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.start_streaming(served_model)

    def _on_stream_thinking(self, model_id: str) -> None:
        """Provider is reasoning internally before producing visible text.
        Currently emitted only by Gemini 2.5 Flash. Card transitions
        LOADING -> THINKING; will transition THINKING -> STREAMING when
        first text chunk arrives via stream_started."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.start_thinking()

    def _on_stream_text_delta(self, model_id: str, chunk: str) -> None:
        """A text chunk from the provider."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.append_stream_text(chunk)

    def _on_stream_usage(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        """Token usage report."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.update_stream_usage(input_tokens, output_tokens)

    def _on_stream_cancelled(self, model_id: str) -> None:
        """Terminal cancellation path."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.set_cancelled()

    def _on_cancel_requested(self, model_id: str) -> None:
        """User clicked Stop on a card. Tell the dispatcher to cancel
        that worker. The dispatcher will eventually emit stream_cancelled
        which routes back to the card's set_cancelled()."""
        self._dispatcher.cancel(model_id)

    # ---------- Shutdown ----------

    def shutdown(self) -> None:
        """Tell the dispatcher to cancel workers and clean up."""
        self._dispatcher.shutdown()

    # ---------- Badges ----------

    def _maybe_award_badges(self) -> None:
        complete_cards = [
            card for card in self._cards.values()
            if card._state.value == "complete"
        ]
        if len(complete_cards) < 2:
            return

        for card in complete_cards:
            card.set_badge("", accent=False)

        try:
            fastest = min(
                complete_cards,
                key=lambda c: float(
                    c._latency_metric.value_label.text().split()[0]
                ),
            )
            fastest.set_badge("FASTEST", accent=True)

            cheapest = min(
                complete_cards,
                key=lambda c: float(
                    c._cost_metric.value_label.text().replace("$", "")
                ),
            )
            if cheapest is not fastest:
                cheapest.set_badge("CHEAPEST", accent=True)
        except (ValueError, IndexError):
            pass