"""
adapters/base.py

The contract every concrete LLM provider adapter must satisfy.

This file defines:
  - LLMResponse:      uniform response shape (text + observability fields)
  - PreparedFile:     what prepare() returns — a per-provider file reference
  - AdapterError:     the only exception type the orchestrator must catch
  - LLMAdapter:       the Protocol every adapter implements

No SDKs are imported here. This file is dependency-free and could be
imported by any layer of the application without dragging in network code.

The two-method contract — prepare() + chat() — separates the long-lived
upload (file lives in the provider's Files API for hours/days, cached in
the registry) from the short-lived chat call (made many times per file,
once per Run). Concrete adapters can do their SDK-specific work inside
each method without leaking SDK types across the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedFile:
    """
    A per-provider reference to an uploaded file, returned by adapter.prepare().

    The orchestrator stores this in the registry's provider_file_refs table
    and passes it back to chat() when running a request. The provider's
    actual file_id (or full URI for Gemini) lives in `remote_id`. The
    adapter is responsible for knowing what `remote_id` means for its SDK.

    expires_at is set when the provider enforces TTL (Gemini 48h); None
    means the file persists until explicitly deleted.
    """
    provider: str           # adapter name, must match LLMAdapter.name
    remote_id: str          # opaque to everyone except the adapter that made it
    filename: str           # for display in response cards
    mime_type: str          # canonical or real; adapter decides what it stores
    size_bytes: int
    expires_at: datetime | None = None


@dataclass(frozen=True)
class LLMResponse:
    """
    Uniform response from any provider's chat() call.

    Carries the model's text plus observability fields the UI and the
    cost-tracking layer need. `raw` is the SDK's native response (already
    serialised to dict) — kept for debugging, never relied on by callers.
    """
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    raw: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    # ^ e.g. ["OpenAI parsed the first 1000 rows of this xlsx"] — surfaced
    #   from Capability.notes when the response is constructed.


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class AdapterError(Exception):
    """
    Base class for every error an adapter can raise.

    Adapters wrap their SDK's native exceptions into this hierarchy so the
    orchestrator only has to catch AdapterError, not openai.APIError +
    anthropic.AuthenticationError + httpx.TimeoutException + ...

    Always set `provider` so error messages can identify which adapter
    raised. `raw` carries the original exception's str() for diagnostics
    without forcing the orchestrator to know about SDK exception types.
    """
    def __init__(self, message: str, *, provider: str, raw: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.raw = raw

    def __str__(self) -> str:
        base = super().__str__()
        return f"[{self.provider}] {base}"


class NotConfiguredError(AdapterError):
    """API key / endpoint / required setting is missing. User-fixable."""


class AuthenticationError(AdapterError):
    """The provider rejected our credentials (401/403)."""


class RateLimitError(AdapterError):
    """The provider rate-limited us (429). Caller should back off."""


class FormatRejectedError(AdapterError):
    """
    The provider rejected the file format at runtime, even though the matrix
    said it should work. Indicates the matrix is out of date for this provider
    — log it loudly so we can update the matrix.
    """


class TransientError(AdapterError):
    """Network blip, 5xx, timeout. Caller may retry."""


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMAdapter(Protocol):
    """
    Contract every provider adapter implements.

    Two operations:
      prepare(path, mime, filename, size) -> PreparedFile
          Upload-once. Called when the user adds a file to the library.
          Returns a reference the registry caches and the orchestrator
          re-uses across many chat() calls.

      chat(prompt, files) -> LLMResponse
          Chat-many-times. Called once per Run. Takes the prepared file
          references and the user's prompt, returns a uniform response.

    The adapter is responsible for translating PreparedFile.remote_id into
    whatever shape its SDK expects for content blocks (file_id for OpenAI,
    Part.from_uri for Gemini, document content block for Anthropic, etc).
    """

    name: str
    """Stable identifier matching the keys in CAPABILITY_MATRIX."""

    def is_configured(self) -> bool:
        """True if API key and any required settings are present."""
        ...

    def prepare(
        self,
        *,
        file_path: str,
        mime_type: str,
        filename: str,
        size_bytes: int,
    ) -> PreparedFile:
        """Upload a file to the provider. Idempotent w.r.t. the registry —
        the orchestrator only calls this when no cached ref exists or the
        cached ref has expired."""
        ...

    def delete(self, prepared: PreparedFile) -> None:
        """Remove the file from the provider's storage. Idempotent: if the
        file is already gone, return cleanly without raising."""
        ...

    def chat(
        self,
        *,
        prompt: str,
        files: list[PreparedFile],
        model: str | None = None,
    ) -> LLMResponse:
        """Send a chat request including any prepared files. Files in the
        list must all belong to this adapter (same .provider name)."""
        ...