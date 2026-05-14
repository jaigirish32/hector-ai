"""
OpenAI client — calls api.openai.com using the Responses API.

File support: Responses API has TWO content block types for files,
chosen per file based on its MIME type:

1. input_file — for documents (PDF, csv, txt, code, office).
   Uploaded with purpose='user_data'. References the file by file_id.
   The Responses API gates the file by FILENAME EXTENSION at chat time
   against an allowlist of document extensions; image extensions like
   .jpg/.jpeg/.png are NOT on this allowlist.

2. input_image — for images (PNG, JPEG, GIF, WEBP).
   Uploaded with purpose='vision'. References the file by file_id.
   The two purposes are NOT interchangeable: a file uploaded as
   'user_data' cannot be referenced via input_image, and a file
   uploaded as 'vision' cannot be referenced via input_file.

The shape is determined by the FileRef's MIME type at chat time.

v0.2.0 streaming:
    complete_stream() consumes the Responses API event stream from
    client.responses.create(stream=True) and yields our StreamEvent
    types. The OpenAI event model has a fan of scaffolding events
    (Created, InProgress, OutputItemAdded, ContentPartAdded, ...)
    around the actual text deltas; we only act on three:
      - ResponseCreatedEvent → emit StreamStarted
      - ResponseTextDeltaEvent → emit TextDelta (delta attribute)
      - ResponseCompletedEvent → emit Usage + StreamCompleted (carries
        the final response object with output_text, usage, model)
    Everything else is metadata for tool-use scenarios — ignored for
    plain text streaming.

Caching:
    Files are sent as a separate user message BEFORE history and the
    current prompt. This keeps the file content at the start of the
    input prefix, maximising prompt cache hit rate across turns — the
    file tokens are only processed once and cached for subsequent turns.

    gpt-5.5 is excluded from OpenAI's standard in-memory prompt cache.
    We pass prompt_cache_retention="persistent" for gpt-5.5 and gpt-5.5-pro
    to opt into extended cache retention (up to 24 hours). Other models
    get automatic in-memory caching with no extra parameter needed.

    Cached input tokens cost $0.50/M vs $5.00/M uncached for gpt-5.5 —
    a 90% discount. With files at the prefix, Turn 2+ should hit the
    cache and bill file tokens at the cached rate.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

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
from providers._dbg import dbg
from providers._retry import with_rate_limit_retry
from providers.streaming import (
    StreamCancelled,
    StreamCompleted,
    StreamFailed,
    StreamStarted,
    TextDelta,
    Usage,
)
from settings_manager import SecretKey, SettingsManager

if TYPE_CHECKING:
    from providers.streaming import StreamEvent

# Reasoning effort for gpt-5/o1/o3 series.
_REASONING_EFFORT = "high"

# Models that are excluded from OpenAI's standard in-memory prompt cache
# and require prompt_cache_retention="persistent" to get extended caching.
# Source: https://developers.openai.com/api/docs/guides/prompt-caching
_NEEDS_24H_CACHE = frozenset({
    "gpt-5.5",
    "gpt-5.5-pro",
})

# MIME types that are referenced via input_image content blocks. MUST
# stay in sync with _OPENAI_IMAGE_MIMES in attachments/uploaders.py.
_IMAGE_MIMES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})


def _parse_openai_retry_after(exc: BaseException) -> int | None:
    """Read the retry-after header from an OpenAI SDK exception."""
    response = getattr(exc, "response", None)
    if response is None:
        return None

    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    ms_value = headers.get("retry-after-ms")
    if ms_value:
        try:
            return max(1, int(float(ms_value)) // 1000)
        except (TypeError, ValueError):
            pass

    seconds_value = headers.get("retry-after")
    if seconds_value:
        try:
            return max(0, int(float(seconds_value)))
        except (TypeError, ValueError):
            pass

    return None


class OpenAIClient(BaseProviderClient):
    """Client for api.openai.com via the Responses API."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    # ---------- BaseProviderClient contract ----------

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.OPENAI_API_KEY)

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Synchronous completion. Dead code — dispatcher uses complete_stream()."""
        raise NotImplementedError(
            "complete() is dead code. Use complete_stream() instead."
        )

    def complete_stream(
        self,
        request: ChatRequest,
        cancel_flag: threading.Event,
    ) -> Iterator["StreamEvent"]:

        dbg("CLIENT", f"openai.complete_stream START for {request.model.id}")

        if cancel_flag.is_set():
            dbg("CLIENT", "openai: cancel_flag already set, yielding StreamCancelled")
            yield StreamCancelled()
            return

        if not self.is_configured():
            dbg("CLIENT", "openai: not configured, yielding StreamFailed")
            yield StreamFailed(
                error=NotConfiguredError(
                    "OpenAI API key not set. Go to Settings to add it."
                )
            )
            return

        api_key = self._settings.get_secret(SecretKey.OPENAI_API_KEY)
        client = OpenAI(api_key=api_key)

        api_model = request.model.api_model_name

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
            "stream": True,
        }

        if not ignores_temperature:
            create_kwargs["temperature"] = request.temperature
        else:
            create_kwargs["reasoning"] = {"effort": _REASONING_EFFORT}
            dbg("CLIENT", f"openai: reasoning effort={_REASONING_EFFORT} for {api_model}")

        if request.system_prompt:
            create_kwargs["instructions"] = request.system_prompt

        # gpt-5.5 and gpt-5.5-pro are excluded from standard in-memory
        # prompt caching. Pass persistent retention so file tokens cached
        # at the prefix get the extended 24-hour window and the 90%
        # cached-token discount ($0.50/M vs $5.00/M).
        if api_model in _NEEDS_24H_CACHE:
            create_kwargs["prompt_cache_retention"] = "24h"
            dbg("CLIENT", f"openai: prompt_cache_retention=persistent for {api_model}")

        dbg("CLIENT", "openai: calling with_rate_limit_retry to open stream")
        dbg("CLIENT", f"openai: system_prompt len={len(request.system_prompt or '')} chars")
        dbg("CLIENT", f"openai: file_refs count={len(request.file_refs)}")

        start = time.monotonic()
        try:
            stream = with_rate_limit_retry(
                fn=lambda: client.responses.create(**create_kwargs),
                sdk_rate_limit_exception=OpenAIRateLimitError,
                parse_retry_after_seconds=_parse_openai_retry_after,
                provider_label="OpenAI",
            )
        except OpenAIAuthError as exc:
            dbg("CLIENT", f"openai: AuthError caught: {exc}")
            yield StreamFailed(
                error=AuthenticationError(
                    "OpenAI rejected the API key. Check it's valid and has credit.",
                    raw=str(exc),
                )
            )
            return
        except OpenAIRateLimitError as exc:
            dbg("CLIENT", f"openai: RateLimitError caught (after retries): {exc}")
            yield StreamFailed(
                error=RateLimitError(
                    "OpenAI rate limited after 3 retries.",
                    raw=str(exc),
                )
            )
            return
        except APIConnectionError as exc:
            dbg("CLIENT", f"openai: ConnectionError caught: {exc}")
            yield StreamFailed(
                error=ProviderError(
                    "Could not reach OpenAI — check your internet connection.",
                    raw=str(exc),
                )
            )
            return
        except APIError as exc:
            dbg("CLIENT", f"openai: APIError caught: {exc}")
            message = getattr(exc, "message", str(exc))
            yield StreamFailed(
                error=ProviderError(
                    f"OpenAI error: {message}",
                    raw=str(exc),
                )
            )
            return
        except Exception as exc:
            dbg("CLIENT", f"openai: UNEXPECTED exception caught: {type(exc).__name__}: {exc}")
            yield StreamFailed(
                error=ProviderError(
                    f"Unexpected error opening OpenAI stream: {exc}",
                    raw=str(exc),
                )
            )
            return

        dbg("CLIENT", "openai: stream open, entering event loop")

        final_response = None

        try:
            for event in stream:
                if cancel_flag.is_set():
                    dbg("CLIENT", "openai: cancel observed mid-stream")
                    try:
                        stream.close()
                    except Exception:
                        pass
                    yield StreamCancelled()
                    return

                event_type = type(event).__name__

                if event_type == "ResponseCreatedEvent":
                    served_model = api_model
                    response_obj = getattr(event, "response", None)
                    if response_obj is not None:
                        served_model = (
                            getattr(response_obj, "model", None) or api_model
                        )
                    dbg("CLIENT", "openai: yielding StreamStarted")
                    yield StreamStarted(model=served_model)

                elif event_type == "ResponseTextDeltaEvent":
                    delta = getattr(event, "delta", None)
                    if delta:
                        dbg("CLIENT", f"openai: yielding TextDelta len={len(delta)}")
                        yield TextDelta(text=delta)

                elif event_type == "ResponseCompletedEvent":
                    final_response = getattr(event, "response", None)
                    dbg("CLIENT", "openai: ResponseCompletedEvent received")

        except Exception as exc:
            dbg("CLIENT", f"openai: exception during event loop: {type(exc).__name__}: {exc}")
            yield StreamFailed(
                error=ProviderError(
                    f"OpenAI stream error: {exc}",
                    raw=str(exc),
                )
            )
            return

        latency = time.monotonic() - start

        if final_response is None:
            dbg("CLIENT", "openai: stream ended without ResponseCompletedEvent")
            yield StreamFailed(
                error=ProviderError(
                    "OpenAI stream ended without a completion event.",
                )
            )
            return

        text = getattr(final_response, "output_text", None) or ""

        usage = getattr(final_response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        # Read cached token count for accurate cost calculation.
        cached_input_tokens = 0
        if usage:
            input_details = getattr(usage, "input_tokens_details", None)
            ached_input_tokens = getattr(input_details, "cached_tokens", 0) if input_details else 0
            dbg(
            "CLIENT",
            f"openai usage: input={input_tokens} (cached={cached_input_tokens}) "
            f"output={output_tokens}",
            
        )

        yield Usage(input_tokens=input_tokens, output_tokens=output_tokens)

        cost = calculate_cost_usd(request.model, input_tokens, output_tokens, cached_input_tokens)

        chat_response = ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            served_model=getattr(final_response, "model", None) or api_model,
            cached_input_tokens=cached_input_tokens,
        )

        dbg("CLIENT", "openai: yielding StreamCompleted, complete_stream END")
        
        yield StreamCompleted(final_response=chat_response)

    # ---------- Internal helpers ----------

    def _build_input(self, request: ChatRequest) -> list[dict]:
        """Convert a ChatRequest into Responses API input format.

        Message ordering for maximum cache hit rate:

          1. Files (separate user message, FIRST)
             Placing files at the very start of the input prefix means
             they form a stable cacheable prefix. On Turn 2+, OpenAI
             can serve the file tokens from cache at $0.50/M instead
             of reprocessing at $5.00/M.

          2. History turns (alternating user/assistant plain-text)
             Grows each turn — must come after the stable file prefix
             so file caching is not broken by history changes.

          3. Current user prompt (last)
             Always changes — must be last so it doesn't break caching
             of the stable prefix above it.

        If no files are attached, messages start directly with history.
        """
        messages: list[dict] = []

        # 1. Files — separate user message at the top of the input.
        #    Sending files as their own message (not mixed with the prompt)
        #    keeps the file prefix clean and maximises cache locality.
        file_blocks: list[dict] = []
        for ref in request.file_refs:
            if ref.provider != "openai":
                continue
            if ref.mime_type in _IMAGE_MIMES:
                file_blocks.append({
                    "type": "input_image",
                    "file_id": ref.remote_id,
                })
            else:
                file_blocks.append({
                    "type": "input_file",
                    "file_id": ref.remote_id,
                })

        if file_blocks:
            messages.append({
                "role": "user",
                "content": file_blocks,
            })

        # 2. History — alternating user/assistant plain-text messages.
        for turn in request.history:
            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        # 3. Current prompt — always last.
        messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": request.prompt}],
        })

        return messages