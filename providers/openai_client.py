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
    plain text streaming. complete() is preserved unchanged for any
    caller that needs synchronous behavior.
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

# Reasoning effort for gpt-5/o1/o3 series. "high" matches what
# chatgpt.com uses in Thinking mode for analytical prompts. Default
# is medium; "xhigh" is reserved for hardest agentic tasks (per
# OpenAI's own guide, can cause overthinking on normal prompts).
_REASONING_EFFORT = "high"


# MIME types that are referenced via input_image content blocks. MUST
# stay in sync with _OPENAI_IMAGE_MIMES in attachments/uploaders.py —
# they describe the same set from two different sides (upload purpose
# selection vs chat content block selection).
_IMAGE_MIMES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})


def _parse_openai_retry_after(exc: BaseException) -> int | None:
    """Read the retry-after header from an OpenAI SDK exception.

    OpenAI's RateLimitError inherits from APIStatusError which has
    self.response (an httpx.Response). Same structure as Anthropic
    but we keep the parser separate per provider so each can evolve
    independently if SDK behavior diverges later.

    Returns seconds to wait, or None if the header is missing or
    unparseable — caller will fall back to exponential backoff.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None

    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    # Prefer milliseconds header if present.
    ms_value = headers.get("retry-after-ms")
    if ms_value:
        try:
            return max(1, int(float(ms_value)) // 1000)
        except (TypeError, ValueError):
            pass

    # Standard retry-after in seconds.
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
        """Synchronous (non-streaming) completion. Preserved for any
        caller that needs blocking behavior; the dispatcher's worker
        path now uses complete_stream() instead."""
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
        else:
            # Reasoning models support reasoning.effort instead of temperature.
            # Use high for analytical depth on the comparison.
            create_kwargs["reasoning"] = {"effort": _REASONING_EFFORT}
            dbg("CLIENT", f"openai: reasoning effort={_REASONING_EFFORT} for {api_model}")
        if request.system_prompt:
            create_kwargs["instructions"] = request.system_prompt

        start = time.monotonic()
        try:
            response = with_rate_limit_retry(
                fn=lambda: client.responses.create(**create_kwargs),
                sdk_rate_limit_exception=OpenAIRateLimitError,
                parse_retry_after_seconds=_parse_openai_retry_after,
                provider_label="OpenAI",
            )
        except OpenAIAuthError as exc:
            raise AuthenticationError(
                "OpenAI rejected the API key. Check it's valid and has credit.",
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
            # Reasoning models support reasoning.effort instead of temperature.
            # Use high for analytical depth on the comparison.
            create_kwargs["reasoning"] = {"effort": _REASONING_EFFORT}
            dbg("CLIENT", f"openai: reasoning effort={_REASONING_EFFORT} for {api_model}")
        if request.system_prompt:
            create_kwargs["instructions"] = request.system_prompt

        # Open the stream. with_rate_limit_retry handles 429 retries; if
        # all retries exhaust, RateLimitError is raised. Other errors
        # surface unchanged and we map them to StreamFailed below.
        dbg("CLIENT", "openai: calling with_rate_limit_retry to open stream")
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

        final_response = None  # populated when ResponseCompletedEvent arrives

        try:
            for event in stream:
                # Cancellation check between events. If the user clicked
                # Stop or HECTOR is shutting down, we close the stream
                # and yield StreamCancelled cleanly.
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
                    # First event of the stream. Emit StreamStarted with
                    # the model name from the create response (which
                    # carries the actual served model — sometimes
                    # different from what was requested).
                    served_model = api_model
                    response_obj = getattr(event, "response", None)
                    if response_obj is not None:
                        served_model = (
                            getattr(response_obj, "model", None) or api_model
                        )
                    dbg("CLIENT", "openai: yielding StreamStarted")
                    yield StreamStarted(model=served_model)

                elif event_type == "ResponseTextDeltaEvent":
                    # Text chunk. The delta attribute is the new text.
                    delta = getattr(event, "delta", None)
                    if delta:
                        dbg("CLIENT", f"openai: yielding TextDelta len={len(delta)}")
                        yield TextDelta(text=delta)

                elif event_type == "ResponseCompletedEvent":
                    # End of stream. event.response carries the final
                    # response object with output_text, usage, model.
                    final_response = getattr(event, "response", None)
                    dbg("CLIENT", "openai: ResponseCompletedEvent received")
                    # Don't break — let the loop end naturally so the
                    # SDK can clean up its iterator. We'll process
                    # final_response after the loop.

                # All other event types ignored for plain text streaming.

        except Exception as exc:
            # Defensive: an exception escapes the iterator (network
            # drop, malformed event, etc.). Surface as StreamFailed
            # rather than letting it kill the worker silently.
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
            # Stream ended without a ResponseCompletedEvent — shouldn't
            # happen in correct operation, but be defensive.
            dbg("CLIENT", "openai: stream ended without ResponseCompletedEvent")
            yield StreamFailed(
                error=ProviderError(
                    "OpenAI stream ended without a completion event.",
                )
            )
            return

        # Build the final ChatResponse from the completed response.
        text = getattr(final_response, "output_text", None) or ""

        usage = getattr(final_response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        dbg(
            "CLIENT",
            f"openai usage: input={input_tokens} output={output_tokens}",
        )

        # Yield Usage before StreamCompleted so the UI's token counter
        # updates a moment before final-state rendering kicks in.
        yield Usage(input_tokens=input_tokens, output_tokens=output_tokens)

        cost = calculate_cost_usd(request.model, input_tokens, output_tokens)

        chat_response = ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            served_model=getattr(final_response, "model", None) or api_model,
        )

        dbg("CLIENT", "openai: yielding StreamCompleted, complete_stream END")
        yield StreamCompleted(final_response=chat_response)

    # ---------- Internal helpers ----------

    def _build_input(self, request: ChatRequest) -> list[dict]:
        """Convert a ChatRequest into Responses API input format.

        Returns a list with one user message containing typed content
        blocks. For each Anthropic file_ref attached to this provider:

          - Image MIMEs (PNG/JPEG/GIF/WEBP) emit:
                {"type": "input_image", "file_id": "..."}
          - Everything else emits:
                {"type": "input_file", "file_id": "..."}

        Followed by an `input_text` block with the user's prompt.

        Files come BEFORE the text — empirically, models follow context
        better when files appear ahead of the question being asked
        about them. Same convention as the Anthropic and Gemini clients.
        """
        content_blocks: list[dict] = []

        for ref in request.file_refs:
            if ref.provider != "openai":
                continue

            if ref.mime_type in _IMAGE_MIMES:
                content_blocks.append({
                    "type": "input_image",
                    "file_id": ref.remote_id,
                })
            else:
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