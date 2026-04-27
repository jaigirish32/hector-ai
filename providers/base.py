"""
Base classes and shared types for LLM provider clients.

Every provider (OpenAI, Azure, Anthropic, Google) implements this
contract. The dispatcher treats all clients uniformly — it only
cares about the BaseProviderClient interface, not which service
the request is going to.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from models import ModelInfo


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
    """
    text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    served_model: str  # exactly what the API reported it served us with


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

class BaseProviderClient(ABC):
    """The contract every provider client must fulfill."""

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Send the request and return the response.

        Must be synchronous (blocking) — threading happens at the
        dispatcher layer, not here.

        Raises one of the ProviderError subclasses on failure.
        """
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if this client has what it needs to call the API."""
        ...


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