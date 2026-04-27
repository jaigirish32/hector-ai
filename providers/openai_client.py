"""
OpenAI client — calls api.openai.com using the Responses API.

File support: Responses API accepts uploaded files via `input_file`
content blocks referencing the file_id returned by the Files API.
We just inject one block per file before the user's text prompt.

Migrated from Chat Completions in Phase 1; file support added in Phase 2.
"""
from __future__ import annotations

import time

from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError as OpenAIAuthError,
    OpenAI,
    RateLimitError as OpenAIRateLimitError,
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


class OpenAIClient(BaseProviderClient):
    """Client for api.openai.com via the Responses API."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    # ---------- BaseProviderClient contract ----------

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.OPENAI_API_KEY)

    def complete(self, request: ChatRequest) -> ChatResponse:
        if not self.is_configured():
            raise NotConfiguredError(
                "OpenAI API key not set. Go to Settings to add it."
            )

        api_key = self._settings.get_secret(SecretKey.OPENAI_API_KEY)
        client = OpenAI(api_key=api_key)

        api_model = request.model.api_model_name

        # Newer GPT-5 family and reasoning models reject custom temperature
        # — they only accept the default. Detect by name prefix.
        ignores_temperature = (
            api_model.startswith("gpt-5")
            or api_model.startswith("o1")
            or api_model.startswith("o3")
        )

        input_items = self._build_input(request)

        create_kwargs: dict = {
            "model": api_model,
            "input": input_items,
            "max_output_tokens": request.max_tokens,
        }
        if not ignores_temperature:
            create_kwargs["temperature"] = request.temperature
        if request.system_prompt:
            create_kwargs["instructions"] = request.system_prompt

        start = time.monotonic()
        try:
            response = client.responses.create(**create_kwargs)
        except OpenAIAuthError as exc:
            raise AuthenticationError(
                "OpenAI rejected the API key. Check it's valid and has credit.",
                raw=str(exc),
            ) from exc
        except OpenAIRateLimitError as exc:
            raise RateLimitError(
                "OpenAI rate limit hit. Wait a moment and retry.",
                raw=str(exc),
            ) from exc
        except APIConnectionError as exc:
            raise ProviderError(
                "Could not reach OpenAI — check your internet connection.",
                raw=str(exc),
            ) from exc
        except APIError as exc:
            message = getattr(exc, "message", str(exc))
            raise ProviderError(
                f"OpenAI error: {message}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error calling OpenAI: {exc}",
                raw=str(exc),
            ) from exc

        latency = time.monotonic() - start

        text = getattr(response, "output_text", None) or ""

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        cost = calculate_cost_usd(request.model, input_tokens, output_tokens)

        return ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            served_model=getattr(response, "model", None) or api_model,
        )

    # ---------- Internal helpers ----------

    def _build_input(self, request: ChatRequest) -> list[dict]:
        """Convert a ChatRequest into Responses API input format.

        Returns a list with one user message containing typed content
        blocks: zero or more `input_file` blocks (one per attached file)
        followed by an `input_text` block with the user's prompt.

        Files come BEFORE the text — empirically, models follow context
        better when files appear ahead of the question being asked
        about them. Same convention as the Anthropic and Gemini clients.
        """
        content_blocks: list[dict] = []

        for ref in request.file_refs:
            if ref.provider != "openai":
                continue
            content_blocks.append({
                "type": "input_file",
                "file_id": ref.remote_id,
            })

        content_blocks.append({
            "type": "input_text",
            "text": request.prompt,
        })

        return [{
            "role": "user",
            "content": content_blocks,
        }]