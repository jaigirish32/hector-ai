"""
The Compare view — workspace where users run prompts against multiple LLMs
side by side.

Reads selected models and attached files from the prompt area, dispatches
real parallel API calls via the Dispatcher, and updates cards as
responses arrive.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from models import ModelInfo, get_model
from providers.base import ChatResponse
from providers.dispatcher import Dispatcher
from widgets.prompt_area import PromptArea
from widgets.response_card import ResponseCard


class ComparisonView(QWidget):
    """Multi-LLM comparison workspace."""

    def __init__(self) -> None:
        super().__init__()

        self._cards: dict[str, ResponseCard] = {}
        self._dispatcher = Dispatcher()

        # Wire dispatcher signals once, at construction.
        self._dispatcher.response_received.connect(self._on_response_received)
        self._dispatcher.response_failed.connect(self._on_response_failed)
        self._dispatcher.all_complete.connect(self._on_all_complete)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(14)

        # ---------- Top: prompt area ----------
        self._prompt_area = PromptArea()
        self._prompt_area.run_requested.connect(self._on_run_requested)
        root.addWidget(self._prompt_area)

        # ---------- Below: scrollable area for response cards ----------
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

    # ----------------------------------------------------------------------
    # UI helpers
    # ----------------------------------------------------------------------

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
        """Remove every widget currently in the cards grid."""
        while self._cards_grid.count() > 0:
            item = self._cards_grid.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cards.clear()

    def _lay_out_cards(self, models: list[ModelInfo]) -> None:
        """Create cards for the given models and arrange in a 2-col grid."""
        self._clear_cards_grid()

        for index, model in enumerate(models):
            card = ResponseCard(model)
            card.voted.connect(self._on_card_voted)
            self._cards[model.id] = card

            row = index // 2
            col = index % 2
            self._cards_grid.addWidget(card, row, col)

        self._cards_grid.setColumnStretch(0, 1)
        self._cards_grid.setColumnStretch(1, 1)

    # ----------------------------------------------------------------------
    # Event handlers — user actions
    # ----------------------------------------------------------------------

    def _on_run_requested(
        self,
        prompt: str,
        model_ids: list,
        file_paths: list,
    ) -> None:
        """User clicked Run — lay out cards and dispatch real requests.

        file_paths arrives from PromptArea as a list of strings (paths
        serialized over the Qt signal). We convert back to Path objects
        before handing off to the dispatcher.
        """
        models = [m for m_id in model_ids if (m := get_model(m_id)) is not None]
        if not models:
            return

        self._lay_out_cards(models)

        # Mark every card as loading immediately. The dispatcher will
        # update cards as responses arrive (or as file uploads fail).
        for card in self._cards.values():
            card.set_loading()

        # Convert string paths back to Path objects. PromptArea serializes
        # them as strings because Qt signals don't always handle Path types
        # cleanly across threads.
        paths = [Path(p) for p in file_paths]

        # Fire the real dispatcher with files. Responses arrive via signals.
        self._dispatcher.dispatch(
            prompt=prompt,
            model_ids=model_ids,
            file_paths=paths,
        )

    # ----------------------------------------------------------------------
    # Event handlers — dispatcher signals
    # ----------------------------------------------------------------------

    def _on_response_received(
        self,
        model_id: str,
        response: ChatResponse,
    ) -> None:
        """A provider returned a successful response."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.set_response(
            text=response.text,
            latency_seconds=response.latency_seconds,
            tokens=response.input_tokens + response.output_tokens,
            cost_usd=response.cost_usd,
        )

    def _on_response_failed(self, model_id: str, message: str) -> None:
        """A provider call failed."""
        card = self._cards.get(model_id)
        if card is None:
            return
        card.set_error(message)

    def _on_all_complete(self) -> None:
        """Every dispatched request has settled — award winner badges."""
        self._maybe_award_badges()

    # ----------------------------------------------------------------------
    # Badge logic
    # ----------------------------------------------------------------------

    def _maybe_award_badges(self) -> None:
        """Highlight fastest / cheapest / etc among completed cards."""
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
            # Metric text was in unexpected format; skip badging rather
            # than crash. Rare edge case but defensive.
            pass

    def _on_card_voted(self, model_id: str, is_positive: bool) -> None:
        """Placeholder — later this will log to analytics."""
        direction = "up" if is_positive else "down"
        print(f"[vote] {model_id}: {direction}")