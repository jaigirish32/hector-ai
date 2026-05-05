"""
Stream event types for provider streaming.

This module defines the *vocabulary* of events that flow from each provider
client out to the dispatcher and ultimately to the UI when a response is
streamed in real time. Events are immutable data records — they describe
what happened, not what to do about it. Consumers (worker, retry helper,
UI) inspect each event's concrete type and react accordingly.

Two lifetimes, side by side
---------------------------
The streaming protocol intentionally keeps two views of a model's reply:

1. Transient view — the streaming events themselves.
   StreamStarted, TextDelta, Usage, StreamFailed, StreamCancelled are
   driven by the SDK as the response is generated. They exist to drive
   live UI updates ("text growing word by word", "token counter
   appearing"). Once consumed, they are gone.

2. Durable view — ChatResponse, carried in StreamCompleted at the end.
   This is HECTOR's existing representation of a complete model reply
   (text + usage + caveats + finish reason + provider metadata). It is
   what gets stored when v0.2.1 introduces multi-turn conversation
   history (5–10 turns kept in memory per provider, no DB) and what
   existing post-completion UI code already understands.

The two views overlap: the assembled text in ChatResponse.text is the
concatenation of the TextDelta.text values that came before it. This is
not redundancy — it is two representations with different roles. The
deltas drive the live render; the ChatResponse is the durable record.

Why events instead of return values
-----------------------------------
A streaming call cannot return a single value because there is no single
moment of completion to return at — the response unfolds over seconds.
Iterators of events are the natural fit. The provider client yields
events as the SDK produces them; the worker iterates and re-emits each
one as a Qt signal across the thread boundary; the UI updates per event
on the main thread.

Why errors are events, not exceptions
-------------------------------------
StreamFailed carries the error rather than the iterator raising it.
Generators that raise exceptions across thread boundaries (worker → main
thread) are fragile in PySide6 — the exception escapes the QRunnable's
run() method and is silently swallowed unless caught explicitly. By
modelling failure as an event, the worker handles it through the same
match/dispatch logic as every other event. Uniform, debuggable, safe.

Why @dataclass(frozen=True)
---------------------------
Events cross thread boundaries. Mutable shared state across threads
invites races. Frozen dataclasses are immutable by construction — once
created, they cannot be mutated, even accidentally. The auto-generated
__init__, __repr__, __eq__ also remove boilerplate.

Open for extension
------------------
The hierarchy is designed to grow. When v0.3.0 adds tool use, new event
types (ToolCallStarted, ToolCallResult, ThinkingDelta) are added here
without modifying any existing event. Consumers that don't recognise the
new events fall through their match/isinstance checks and ignore them —
no existing code breaks. This is the Open/Closed Principle in practice.

This file has no logic — only data definitions. It does not import Qt,
any provider SDK, or any module with runtime side effects. It is safe to
import from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from providers.base import ChatResponse, ProviderError


# ---------------------------------------------------------------------------
# Marker base class.
#
# StreamEvent has no fields and no methods. Its only role is to give all
# concrete event types a common supertype, so we can write
# `Iterator[StreamEvent]` and use uniform isinstance / match dispatch on
# the consumer side. Consumers always work with the concrete subtypes —
# StreamEvent itself is never instantiated.
# ---------------------------------------------------------------------------
class StreamEvent:
    """Marker base class for all stream events. Do not instantiate directly."""

    pass


# ---------------------------------------------------------------------------
# Concrete event types, ordered by their typical position in a stream:
# Started -> (TextDelta x N, Usage) -> Completed   [happy path]
# Started -> (TextDelta x N) -> Failed             [error path]
# Started -> (TextDelta x N) -> Cancelled          [user pressed Stop]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamStarted(StreamEvent):
    """
    Emitted once at the start of a stream, after the HTTP connection is
    established and the provider has acknowledged the request. The UI
    uses this to flip the response card from "queued" into "streaming"
    state (cursor visible, text area cleared) and optionally to confirm
    which model the provider actually routed to (sometimes differs from
    what was requested).
    """

    model: str


@dataclass(frozen=True)
class TextDelta(StreamEvent):
    """
    A chunk of generated text. Multiple TextDelta events arrive over the
    course of a stream; concatenating their `text` fields in order yields
    the full response body. Each delta is typically a few characters to
    a few words, depending on the provider's tokenization and chunking
    strategy. The provider client also accumulates these internally so
    it can build the final ChatResponse for StreamCompleted.

    The UI handler appends `text` to the response card's text widget
    and ensures the cursor remains visible (auto-scroll).
    """

    text: str


@dataclass(frozen=True)
class Usage(StreamEvent):
    """
    Token usage report. Different providers emit this at different
    points — Anthropic and Gemini typically emit it near the end of a
    stream, OpenAI/Azure at the very end. Carrying it as its own event
    rather than only inside ChatResponse lets the UI update its token
    counter the moment the data arrives, even if a few more text deltas
    follow afterwards.

    The UI handler updates the card's "input / output" token label.
    """

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class StreamCompleted(StreamEvent):
    """
    Emitted once at the end of a successful stream. Carries the full
    ChatResponse — HECTOR's existing durable representation of a model
    reply — assembled by the provider client as the stream progressed.

    This ChatResponse is the long-lived record of the turn:
      * v0.2.0  — the response card uses it for final-state rendering
                  (caveats, citations, total cost, finish reason).
      * v0.2.1  — the multi-turn history list stores it; on the next
                  turn, the dispatcher reads its `.text` to populate
                  the assistant role in the messages array sent to the
                  provider.

    Receipt of StreamCompleted means the stream ended cleanly, with no
    error and no cancellation. The card transitions from "streaming" to
    "complete" state. No further events arrive on this stream.
    """

    final_response: ChatResponse


@dataclass(frozen=True)
class StreamFailed(StreamEvent):
    """
    Emitted when the stream errors out at any point — at connection
    time, mid-stream, or at finalization. Carries the structured
    ProviderError rather than a flattened message string so different
    consumers can inspect it: the retry helper checks for RateLimitError
    to decide whether to restart the stream; the UI flattens to
    str(error) for display.

    Once StreamFailed is yielded, the stream is over. The provider
    client must not yield further events on the same stream.
    """

    error: ProviderError


@dataclass(frozen=True)
class StreamCancelled(StreamEvent):
    """
    Emitted when the user-requested cancellation flag was observed by
    the provider client and the underlying SDK stream was shut down
    cleanly. Carries no payload — the cancellation source already knows
    it requested the stop; this event simply confirms the worker side
    noticed and complied.

    The UI handler transitions the card to a "stopped" visual state and
    keeps whatever partial text was already streamed, without an error
    indicator.
    """

    pass