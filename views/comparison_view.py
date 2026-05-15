"""
The Compare view — workspace where users run prompts against multiple LLMs
side by side.

Feature 1.6 — Conversation history
-----------------------------------
Each model maintains an independent conversation thread stored in
SQLite via ConversationStore. History is loaded per model before each
dispatch and passed through ChatRequest.history to the provider client.
After a successful response, the turn (user prompt + assistant text)
is saved back to the store.

Clear History is per-model — each ResponseCard has a trash icon button
that emits clear_history_requested(model_id).

Startup / chip toggle behaviour
--------------------------------
All enabled model cards are rendered immediately on startup in EMPTY
state. When a chip is toggled off, its card is removed from the grid
but NOT destroyed — its state (history, response text) is preserved
in memory. Toggle it back on and the card reappears exactly as it was.
On each Run, only the currently-selected chips receive API calls.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from attachments.file_library import FileLibrary
from conversation_store import ConversationStore
from models import DEFAULT_MODELS, ModelInfo, Provider, get_model
from providers.base import ChatResponse, FileRef, HistoryMessage
from providers.dispatcher import Dispatcher
from widgets.file_library_panel import FileLibraryPanel
from widgets.prompt_area import PromptArea
from widgets.response_card import ResponseCard


_PROVIDER_TO_LIBRARY_KEY: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE: "gemini",
    Provider.XAI: "xai",
}

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

        # _cards holds ALL model cards ever created, whether visible or not.
        # Cards are never destroyed on chip toggle — only removed from the
        # grid. This preserves response state when a chip is toggled off
        # and back on.
        self._cards: dict[str, ResponseCard] = {}

        self._file_library = file_library
        self._file_panel: FileLibraryPanel | None = None
        self._dispatcher = Dispatcher()
        self._conversation_store = ConversationStore()
        self._last_prompt: str = ""

        self._dispatcher.response_received.connect(self._on_response_received)
        self._dispatcher.response_failed.connect(self._on_response_failed)
        self._dispatcher.all_complete.connect(self._on_all_complete)
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
        self._prompt_area.selection_changed.connect(self._on_selection_changed)
        root.addWidget(self._prompt_area)

        self._cards_scroll = QScrollArea()
        self._cards_scroll.setObjectName("cardsScroll")
        self._cards_scroll.setWidgetResizable(True)
        self._cards_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._cards_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._cards_container = QWidget()
        self._cards_container.setObjectName("cardsContainer")
        self._cards_grid = QGridLayout(self._cards_container)
        self._cards_grid.setContentsMargins(0, 0, 0, 0)
        self._cards_grid.setSpacing(12)

        self._cards_scroll.setWidget(self._cards_container)
        root.addWidget(self._cards_scroll, stretch=1)

        # Create cards for all enabled models on startup and show them.
        # All chips start selected so all cards are visible immediately.
        self._create_all_cards()
        self._regrid(self._prompt_area.selected_models())

    # ---------- Setup wiring ----------

    def set_file_panel(self, panel: FileLibraryPanel) -> None:
        self._file_panel = panel

    # ---------- Card management ----------

    def _create_all_cards(self) -> None:
        """Create ResponseCard objects for every enabled model.

        Cards are created once and reused. They are never destroyed
        during a session — only added/removed from the grid.
        """
        for model in DEFAULT_MODELS:
            if not model.enabled or model.id in self._cards:
                continue
            card = ResponseCard(model)
            card.cancel_requested.connect(self._on_cancel_requested)
            card.clear_history_requested.connect(self._on_clear_model_history)
            self._cards[model.id] = card

    def _regrid(self, model_ids: list[str]) -> None:
        """Remove all cards from the grid, then re-add only selected ones.

        Cards not in model_ids are detached from the grid (setParent(None))
        but remain in self._cards with their state intact. They reappear
        unchanged when the chip is toggled back on.
        """
        # Detach all cards from grid without deleting them.
        while self._cards_grid.count() > 0:
            item = self._cards_grid.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.setParent(None)

        # Re-add only the selected cards in DEFAULT_MODELS order.
        selected = [
            m for m in DEFAULT_MODELS
            if m.enabled and m.id in model_ids
        ]
        for index, model in enumerate(selected):
            card = self._cards.get(model.id)
            if card is None:
                continue
            row = index // 2
            col = index % 2
            self._cards_grid.addWidget(card, row, col)

        self._cards_grid.setColumnStretch(0, 1)
        self._cards_grid.setColumnStretch(1, 1)

    # ---------- Chip toggle handler ----------

    def _on_selection_changed(self, model_ids: list[str]) -> None:
        """Chip toggled — update grid to show only selected cards.

        Cards that are currently streaming or loading are NOT interrupted
        — they keep running in the background. Their results will still
        arrive via dispatcher signals and be applied when they finish,
        even if the card is off-screen.
        """
        self._regrid(model_ids)

    # ---------- Run dispatch ----------

    def _on_run_requested(self, prompt: str, model_ids: list) -> None:
        models = [m for m_id in model_ids if (m := get_model(m_id)) is not None]
        if not models:
            return

        self._last_prompt = prompt

        per_model_history: dict[str, tuple[HistoryMessage, ...]] = {}
        prior_turns: dict[str, list[tuple[str, str]]] = {}

        for model in models:
            turns = self._conversation_store.get_history(model.id)
            prior_turns[model.id] = [
                (t.user_content, t.assistant_content) for t in turns
            ]
            messages: list[HistoryMessage] = []
            for turn in turns:
                messages.append(HistoryMessage(role="user", content=turn.user_content))
                messages.append(HistoryMessage(role="assistant", content=turn.assistant_content))
            per_model_history[model.id] = tuple(messages)

        # Regrid to show exactly the selected cards.
        self._regrid(model_ids)

        for model in models:
            card = self._cards.get(model.id)
            if card is not None:
                card.set_loading(
                    history=prior_turns.get(model.id, []),
                    current_prompt=prompt,
                )

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
            per_model_history=per_model_history,
        )

    # ---------- Dispatcher signal handlers ----------

    def _on_response_received(self, model_id: str, response: ChatResponse) -> None:
        card = self._cards.get(model_id)
        if card is None:
            return
        self._conversation_store.add_turn(
            model_id=model_id,
            user_content=self._last_prompt,
            assistant_content=response.text,
        )
        card.set_response(
            text=response.text,
            latency_seconds=response.latency_seconds,
            tokens=response.input_tokens + response.output_tokens,
            cost_usd=response.cost_usd,
            caveats=response.caveats,
        )

    def _on_response_failed(self, model_id: str, message: str) -> None:
        card = self._cards.get(model_id)
        if card is None:
            return
        card.set_error(message)

    def _on_all_complete(self) -> None:
        self._maybe_award_badges()

    # ---------- Per-model clear history ----------

    def _on_clear_model_history(self, model_id: str) -> None:
        """Delete history for one model and reset its card to EMPTY state."""
        self._conversation_store.clear_model(model_id)
        card = self._cards.get(model_id)
        if card is not None:
            card.reset()

    # ---------- Streaming signal handlers ----------

    def _on_stream_started(self, model_id: str, served_model: str) -> None:
        card = self._cards.get(model_id)
        if card is None:
            return
        card.start_streaming(served_model)

    def _on_stream_thinking(self, model_id: str) -> None:
        card = self._cards.get(model_id)
        if card is None:
            return
        card.start_thinking()

    def _on_stream_text_delta(self, model_id: str, chunk: str) -> None:
        card = self._cards.get(model_id)
        if card is None:
            return
        card.append_stream_text(chunk)

    def _on_stream_usage(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        card = self._cards.get(model_id)
        if card is None:
            return
        card.update_stream_usage(input_tokens, output_tokens)

    def _on_stream_cancelled(self, model_id: str) -> None:
        card = self._cards.get(model_id)
        if card is None:
            return
        card.set_cancelled()

    def _on_cancel_requested(self, model_id: str) -> None:
        self._dispatcher.cancel(model_id)

    # ---------- Shutdown ----------

    def shutdown(self) -> None:
        self._dispatcher.shutdown()
        self._conversation_store.close()

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