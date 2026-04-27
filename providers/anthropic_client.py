"""
Anthropic (Claude) client — calls api.anthropic.com via the official
`anthropic` Python SDK.

Differences from our OpenAI client:
- Endpoint method is `messages.create`, not `chat.completions.create`.
- System prompt is a TOP-LEVEL parameter, not a message in the list.
- Response content is a list of blocks; we read text from the first block.
- `max_tokens` is REQUIRED by Anthropic (not optional).
- Token field names are `input_tokens` and `output_tokens` (not prompt/completion).

File support: messages.content can be a list of typed blocks. We attach
files as `document` blocks with `source.type='file'`, requiring the
beta header `files-api-2025-04-14`.
"""
from __future__ import annotations

import time

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    AuthenticationError as AnthropicAuthError,
    RateLimitError as AnthropicRateLimitError,
)

from providers.base import (
    AuthenticationError,
    BaseProviderClient,
    ChatRequest,
    ChatResponse,
    NotConfiguredError,
    ProviderError,
    RateLimitError,
    calculate_cost_usd,
)
from settings_manager import SecretKey, SettingsManager


# Anthropic requires this beta header to accept `document` content blocks
# that reference uploaded file_ids. Bumped only when Anthropic changes
# their Files API contract.
ANTHROPIC_FILES_BETA = "files-api-2025-04-14"


class AnthropicClient(BaseProviderClient):
    """Client for api.anthropic.com (Claude models)."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    # ---------- BaseProviderClient contract ----------

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.ANTHROPIC_API_KEY)

    def complete(self, request: ChatRequest) -> ChatResponse:
        if not self.is_configured():
            raise NotConfiguredError(
                "Anthropic API key not set. Go to Settings to add it."
            )

        api_key = self._settings.get_secret(SecretKey.ANTHROPIC_API_KEY)
        client = Anthropic(api_key=api_key)

        # Build kwargs explicitly so we only send `system` when set.
        create_kwargs: dict = {
            "model": request.model.api_model_name,
            "messages": [
                {"role": "user", "content": self._build_user_content(request)},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "extra_headers": {"anthropic-beta": ANTHROPIC_FILES_BETA},
        }
        if request.system_prompt:
            create_kwargs["system"] = request.system_prompt

        start = time.monotonic()
        try:
            response = client.messages.create(**create_kwargs)
        except AnthropicAuthError as exc:
            raise AuthenticationError(
                "Anthropic rejected the API key. "
                "Check it at console.anthropic.com and confirm you have credit.",
                raw=str(exc),
            ) from exc
        except AnthropicRateLimitError as exc:
            raise RateLimitError(
                "Anthropic rate limit hit. Wait a moment and retry.",
                raw=str(exc),
            ) from exc
        except APIConnectionError as exc:
            raise ProviderError(
                "Could not reach Anthropic — check your internet connection.",
                raw=str(exc),
            ) from exc
        except APIError as exc:
            message = getattr(exc, "message", str(exc))
            raise ProviderError(
                f"Anthropic error: {message}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error calling Anthropic: {exc}",
                raw=str(exc),
            ) from exc

        latency = time.monotonic() - start

        # Anthropic returns content as a list of blocks. For a normal text
        # response there's one block with .type == "text" and .text == "...".
        # If Claude calls a tool, there might be tool_use blocks instead.
        # We only grab the first text block; tool blocks (rare for our usage)
        # would be skipped.
        text = ""
        for block in response.content or []:
            if hasattr(block, "text") and block.text:
                text = block.text
                break

        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        cost = calculate_cost_usd(request.model, input_tokens, output_tokens)

        return ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            served_model=response.model or request.model.api_model_name,
        )

    # ---------- Internal helpers ----------

    def _build_user_content(self, request: ChatRequest) -> list[dict]:
        """Build the user message content as a list of typed blocks.

        Anthropic accepts a heterogeneous list of content blocks (text,
        image, document) within one message. Files attach as `document`
        blocks referencing the cached file_id from our orchestrator.

        We place files BEFORE the user's text prompt — empirically, models
        follow context better when relevant files appear ahead of the
        question being asked about them.
        """
        blocks: list[dict] = []

        for ref in request.file_refs:
            if ref.provider != "anthropic":
                continue
            blocks.append({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": ref.remote_id,
                },
            })

        blocks.append({"type": "text", "text": request.prompt})
        return blocks