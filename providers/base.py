"""
Base classes and shared types for LLM provider clients.

Every provider (OpenAI, Azure, Anthropic, Google) implements this
contract. The dispatcher treats all clients uniformly — it only
cares about the BaseProviderClient interface, not which service
the request is going to.
"""
from __future__ import annotations  # NEW: enables string annotations for StreamEvent forward reference

import threading                              # NEW: for cancel_flag type in complete_stream()
from abc import ABC, abstractmethod
from collections.abc import Iterator          # NEW: for Iterator[StreamEvent] return type
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING              # NEW: lets us import StreamEvent for type-checking only

from models import ModelInfo

# NEW: type-only import to avoid circular dependency at runtime.
# streaming.py imports ChatResponse and ProviderError from this file,
# so base.py cannot import from streaming.py at runtime. The
# TYPE_CHECKING block runs only under type-checkers (mypy, pyright);
# at runtime the import is skipped, and any annotations referencing
# StreamEvent are evaluated as strings (thanks to __future__ annotations).
if TYPE_CHECKING:
    from providers.streaming import StreamEvent


# ---------------------------------------------------------------------------
# Request / response types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileRef:
    """A resolved file reference for a specific provider.

    The dispatcher populates these from file_paths via FileOrchestrator
    before calling the provider. Each provider client reads only the
    refs whose `provider` field matches its own provider_name.
    """
    provider: str          # 'anthropic', 'gemini', 'openai', 'azure_openai'
    remote_id: str         # provider's file_id (or URI for Gemini)
    filename: str
    mime_type: str


@dataclass(frozen=True)
class ChatRequest:
    """A request to a provider for a single completion.

    file_paths carries the absolute paths of any attached files. The
    dispatcher resolves these into per-provider remote_id refs via the
    FileOrchestrator and stores them in file_refs before reaching the
    provider client. Provider clients should read file_refs, not
    file_paths.
    """
    prompt: str
    model: ModelInfo
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str | None = None
    file_paths: tuple[Path, ...] = ()
    file_refs: tuple[FileRef, ...] = ()


@dataclass(frozen=True)
class ChatResponse:
    """A successful response from a provider.

    Contains the text plus the metrics we display in the response card
    (latency, token counts, cost). The cost is computed locally from
    token counts using the model's published price; we don't trust
    providers to report cost.

    caveats carries per-call notes the UI should surface alongside the
    answer. Examples:
      - dispatcher attaches "Note: 1 of 2 files (xlsx) not supported on
        this provider" when the model only saw a subset of attached files
      - a provider client may attach "Only first 1000 rows per sheet were
        read" for OpenAI's spreadsheet augmentation truncation
    Caveats are informational, not errors. They appear under the answer
    in italic grey text.
    """
    text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    served_model: str  # exactly what the API reported it served us with
    caveats: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Raised by a provider client when a call fails.

    Carries both a user-friendly message and an optional raw error
    for logging. The UI shows the friendly message; logs get the raw one.
    """

    def __init__(self, friendly_message: str, raw: str = "") -> None:
        super().__init__(friendly_message)
        self.friendly_message = friendly_message
        self.raw = raw or friendly_message

    def __str__(self) -> str:
        return self.friendly_message


class AuthenticationError(ProviderError):
    """The API key is missing, invalid, or has no access to this model."""


class RateLimitError(ProviderError):
    """The provider rate-limited us; retry later."""


class NotConfiguredError(ProviderError):
    """This provider has no API key in Settings yet."""


# ---------------------------------------------------------------------------
# Provider client contract
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Provider client contract
# ---------------------------------------------------------------------------

class BaseProviderClient(ABC):
    """The contract every provider client must fulfill."""

    # CHANGED in v0.2.0 streaming migration: was @abstractmethod. Now
    # non-abstract with a NotImplementedError default — same transitional
    # pattern as complete_stream() got in Step 2. Concrete provider clients
    # remove their override of this method as they are migrated to
    # streaming, one at a time. Once all four providers are migrated and
    # the dispatcher consumes only streams, this method is removed from
    # BaseProviderClient entirely.
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Send the request and return the response.

        Must be synchronous (blocking) — threading happens at the
        dispatcher layer, not here.

        Raises one of the ProviderError subclasses on failure.

        NOTE (v0.2.0 transition): this method is being phased out in
        favour of complete_stream(). The dispatcher no longer calls it.
        Concrete provider clients are removing their override of this
        method as they migrate to streaming. Until removal of this
        method on the base class, calling it on a migrated provider
        falls through to this default, which raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.complete() has been removed as "
            f"part of the v0.2.0 streaming migration. "
            f"Use complete_stream() instead."
        )

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this client has what it needs to call the API."""
        ...

    # NEW: streaming contract.
    #
    # complete_stream() is the v0.2.0 replacement for complete(). It
    # yields a sequence of StreamEvent values as the response is
    # generated, instead of returning a single ChatResponse at the end.
    # The provider client assembles a ChatResponse internally as the
    # stream progresses and emits it inside StreamCompleted at the end —
    # so the durable record (used by the response card and by future
    # multi-turn history) is preserved alongside the live streaming.
    #
    # Errors during streaming are reported as StreamFailed events, NOT
    # as raised exceptions. This is a deliberate change from complete():
    # exceptions raised from a generator that is being iterated across a
    # thread boundary (worker thread → main thread via Qt signals in the
    # dispatcher) are fragile and easily swallowed. Yielding StreamFailed
    # makes failure part of the protocol, handled uniformly with every
    # other event by consumers.
    #
    # cancel_flag is a threading.Event the caller (the dispatcher worker)
    # sets when the user clicks Stop. The implementation MUST check the
    # flag between events and yield StreamCancelled then return when it
    # is set. The flag has no default value: every caller must construct
    # one explicitly. This forces cancellation to be a first-class
    # concern in every code path that calls complete_stream().
    #
    # During the v0.2.0 migration, this method has a default body that
    # raises NotImplementedError. Each provider's concrete implementation
    # is added one at a time (Anthropic first, then OpenAI, Azure,
    # Gemini). Until a given provider is migrated, calling this method
    # on it raises — the dispatcher surfaces a graceful error to the UI
    # ("This provider is not yet migrated to streaming"). Once all four
    # providers implement complete_stream(), this default is removed and
    # the method is marked @abstractmethod.
    def complete_stream(
        self,
        request: ChatRequest,
        cancel_flag: threading.Event,
    ) -> Iterator["StreamEvent"]:
        """Send the request and yield streaming events as the response generates.

        Yields a sequence of StreamEvent values:
          StreamStarted   — once at the start
          TextDelta       — many, as text is generated
          Usage           — once, when the provider reports token counts
          StreamCompleted — once at the end (carries the final ChatResponse)
          StreamFailed    — once if the stream errors out (terminates the stream)
          StreamCancelled — once if cancel_flag was observed (terminates the stream)

        After StreamCompleted, StreamFailed, or StreamCancelled, the
        stream is over and no further events are yielded.

        Implementations MUST check `cancel_flag.is_set()` periodically
        (between SDK chunks) and yield StreamCancelled then return when
        the flag is set, closing the underlying SDK stream cleanly so
        the connection is released and no further tokens are billed.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has not been migrated to streaming yet. "
            f"Until its complete_stream() is implemented, this provider cannot "
            f"be used in streaming mode. (v0.2.0 migration in progress.)"
        )


# ---------------------------------------------------------------------------
# Shared helper — token-to-cost calculation (same math for every provider)
# ---------------------------------------------------------------------------

def calculate_cost_usd(
    model: ModelInfo,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return the USD cost for a completion given token counts."""
    input_cost = (input_tokens / 1_000_000) * model.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * model.output_cost_per_1m
    return input_cost + output_cost