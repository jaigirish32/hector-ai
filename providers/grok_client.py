"""
Grok client — calls xAI's native API at api.x.ai via the OpenAI Python
SDK pointed at xAI's base URL.

xAI exposes an OpenAI-compatible Responses API surface, so we reuse
the openai library with base_url overridden. The streaming event
shape is identical to OpenAI's Responses API for text events, with
one addition: ResponseReasoningSummaryTextDeltaEvent fires before
visible text when the model is reasoning. We use these to drive the
THINKING badge in the card UI; the reasoning summary content itself
is logged via dbg() but not surfaced to the user (matches Anthropic
client's treatment of thinking_delta).

Important model behavior:
  - grok-4.20-0309-reasoning has reasoning ALWAYS ON. It rejects the
    reasoning_effort parameter (verified via probe). No effort knob.
  - reasoning_tokens are reported via usage.output_tokens_details and
    are already INCLUDED in usage.output_tokens (per OpenAI Responses
    API convention). No manual addition needed for cost.

File support: not implemented in v1. xAI's Responses API claims
input_file/input_image compatibility but we have not verified.
For now Grok is text-only; attached files are silently dropped.
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
    StreamThinking,
    TextDelta,
    Usage,
)
from settings_manager import SecretKey, SettingsManager

if TYPE_CHECKING:
    from providers.streaming import StreamEvent


XAI_BASE_URL = "https://api.x.ai/v1"


def _parse_xai_retry_after(exc: BaseException) -> int | None:
    """Read retry-after header from an xAI SDK exception.

    xAI uses the OpenAI SDK's exception types since the API surface is
    OpenAI-compatible. Header parsing is identical to OpenAI's: prefer
    retry-after-ms when present, fall back to retry-after seconds.
    Returns None when neither header is parseable — caller will use
    exponential backoff.
    """
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


class GrokClient(BaseProviderClient):
    """Client for xAI's Grok models via api.x.ai/v1/responses."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    # ---------- BaseProviderClient contract ----------

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.XAI_API_KEY)

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Synchronous (non-streaming) path. Kept for any blocking caller;
        the dispatcher's worker uses complete_stream() instead."""
        if not self.is_configured():
            raise NotConfiguredError(
                "xAI API key not set. Go to Settings to add it."
            )

        api_key = self._settings.get_secret(SecretKey.XAI_API_KEY)
        client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)
        api_model = request.model.api_model_name

        create_kwargs: dict = {
            "model": api_model,
            "input": self._build_input(request),
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        if request.system_prompt:
            create_kwargs["instructions"] = request.system_prompt

        start = time.monotonic()
        try:
            response = with_rate_limit_retry(
                fn=lambda: client.responses.create(**create_kwargs),
                sdk_rate_limit_exception=OpenAIRateLimitError,
                parse_retry_after_seconds=_parse_xai_retry_after,
                provider_label="xAI",
            )
        except OpenAIAuthError as exc:
            raise AuthenticationError(
                "xAI rejected the API key. Check it's valid at console.x.ai.",
                raw=str(exc),
            ) from exc
        except APIConnectionError as exc:
            raise ProviderError(
                "Could not reach xAI — check your internet connection.",
                raw=str(exc),
            ) from exc
        except APIError as exc:
            message = getattr(exc, "message", str(exc))
            raise ProviderError(
                f"xAI error: {message}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error calling xAI: {exc}",
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
        """Stream a response from xAI's Responses API.

        Event sequence for a reasoning model:
            ResponseCreatedEvent           -> StreamStarted is deferred
            ResponseReasoningSummaryTextDeltaEvent (N times)
                                            -> StreamThinking on first
            ResponseTextDeltaEvent (M times)
                                            -> StreamStarted on first,
                                               then TextDelta per chunk
            ResponseCompletedEvent         -> Usage + StreamCompleted

        The reasoning summary text is logged for diagnostics but never
        surfaced to the UI — matches the Anthropic client's policy on
        thinking deltas. Showing summaries mid-stream would conflict
        with the card's single-final-answer pattern.
        """
        dbg("CLIENT", f"grok.complete_stream START for {request.model.id}")

        if cancel_flag.is_set():
            dbg("CLIENT", "grok: cancel_flag already set, yielding StreamCancelled")
            yield StreamCancelled()
            return

        if not self.is_configured():
            dbg("CLIENT", "grok: not configured, yielding StreamFailed")
            yield StreamFailed(
                error=NotConfiguredError(
                    "xAI API key not set. Go to Settings to add it."
                )
            )
            return

        api_key = self._settings.get_secret(SecretKey.XAI_API_KEY)
        client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)
        api_model = request.model.api_model_name

        create_kwargs: dict = {
            "model": api_model,
            "input": self._build_input(request),
            "max_output_tokens": request.max_tokens,
            "stream": True,
            "temperature": request.temperature,
        }
        if request.system_prompt:
            create_kwargs["instructions"] = request.system_prompt

        # Files are deliberately not attached. xAI's Responses API
        # support for input_file/input_image is unverified and out of
        # scope for v1. Any attached file_refs are dropped silently.

        dbg("CLIENT", "grok: calling with_rate_limit_retry to open stream")
        start = time.monotonic()
        try:
            stream = with_rate_limit_retry(
                fn=lambda: client.responses.create(**create_kwargs),
                sdk_rate_limit_exception=OpenAIRateLimitError,
                parse_retry_after_seconds=_parse_xai_retry_after,
                provider_label="xAI",
            )
        except OpenAIAuthError as exc:
            dbg("CLIENT", f"grok: AuthError caught: {exc}")
            yield StreamFailed(
                error=AuthenticationError(
                    "xAI rejected the API key. Check it's valid at console.x.ai.",
                    raw=str(exc),
                )
            )
            return
        except OpenAIRateLimitError as exc:
            dbg("CLIENT", f"grok: RateLimitError caught (after retries): {exc}")
            yield StreamFailed(
                error=RateLimitError(
                    "xAI rate limited after 3 retries.",
                    raw=str(exc),
                )
            )
            return
        except APIConnectionError as exc:
            dbg("CLIENT", f"grok: ConnectionError caught: {exc}")
            yield StreamFailed(
                error=ProviderError(
                    "Could not reach xAI — check your internet connection.",
                    raw=str(exc),
                )
            )
            return
        except APIError as exc:
            dbg("CLIENT", f"grok: APIError caught: {exc}")
            message = getattr(exc, "message", str(exc))
            yield StreamFailed(
                error=ProviderError(
                    f"xAI error: {message}",
                    raw=str(exc),
                )
            )
            return
        except Exception as exc:
            dbg("CLIENT", f"grok: UNEXPECTED exception caught: {type(exc).__name__}: {exc}")
            yield StreamFailed(
                error=ProviderError(
                    f"Unexpected error opening xAI stream: {exc}",
                    raw=str(exc),
                )
            )
            return

        dbg("CLIENT", "grok: stream open, entering event loop")

        thinking_emitted = False    # has StreamThinking been yielded yet
        streaming_emitted = False   # has StreamStarted been yielded yet
        served_model = api_model
        final_response = None

        try:
            for event in stream:
                if cancel_flag.is_set():
                    dbg("CLIENT", "grok: cancel observed mid-stream")
                    try:
                        stream.close()
                    except Exception:
                        pass
                    yield StreamCancelled()
                    return

                event_type = type(event).__name__

                if event_type == "ResponseCreatedEvent":
                    response_obj = getattr(event, "response", None)
                    if response_obj is not None:
                        served_model = (
                            getattr(response_obj, "model", None) or api_model
                        )
                    # Don't yield StreamStarted yet — wait for first text
                    # block. If reasoning happens first (typical for the
                    # reasoning model), we'll yield StreamThinking before
                    # StreamStarted, matching Anthropic/Gemini UX.

                elif event_type == "ResponseReasoningSummaryTextDeltaEvent":
                    if not thinking_emitted:
                        dbg("CLIENT", "grok: yielding StreamThinking")
                        yield StreamThinking()
                        thinking_emitted = True
                    # Log the summary for diagnostics but don't surface
                    # to the UI. Truncate preview for log readability.
                    delta = getattr(event, "delta", None)
                    if delta:
                        preview = delta[:80].replace("\n", " ")
                        dbg("CLIENT", f"grok reasoning: {preview}")

                elif event_type == "ResponseTextDeltaEvent":
                    if not streaming_emitted:
                        dbg("CLIENT", "grok: first text arrived, yielding StreamStarted")
                        yield StreamStarted(model=served_model)
                        streaming_emitted = True
                    delta = getattr(event, "delta", None)
                    if delta:
                        dbg("CLIENT", f"grok: yielding TextDelta len={len(delta)}")
                        yield TextDelta(text=delta)

                elif event_type == "ResponseCompletedEvent":
                    final_response = getattr(event, "response", None)
                    dbg("CLIENT", "grok: ResponseCompletedEvent received")
                    # Don't break — let the loop end naturally so the
                    # SDK can clean up its iterator.

                # Other events (InProgress, OutputItemAdded/Done,
                # ContentPartAdded/Done, ReasoningSummaryPartAdded/Done,
                # TextDone) are ignored — they're metadata for tool-use
                # and structural completion, not needed for our UI.

        except Exception as exc:
            dbg("CLIENT", f"grok: exception during event loop: {type(exc).__name__}: {exc}")
            yield StreamFailed(
                error=ProviderError(
                    f"xAI stream error: {exc}",
                    raw=str(exc),
                )
            )
            return

        latency = time.monotonic() - start

        # Defensive: if we somehow finished without ever seeing a text
        # event (shouldn't happen for a reasoning model on a normal
        # prompt, but be defensive), emit StreamStarted so the card
        # transitions out of THINKING before completion.
        if not streaming_emitted:
            dbg("CLIENT", "grok: no text seen, yielding StreamStarted defensively")
            yield StreamStarted(model=served_model)

        if final_response is None:
            dbg("CLIENT", "grok: stream ended without ResponseCompletedEvent")
            yield StreamFailed(
                error=ProviderError(
                    "xAI stream ended without a completion event.",
                )
            )
            return

        text = getattr(final_response, "output_text", None) or ""

        usage = getattr(final_response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        # Reasoning tokens are already included in output_tokens per the
        # OpenAI Responses API convention. usage.output_tokens_details
        # exposes them separately for diagnostics only.
        details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
        dbg(
            "CLIENT",
            f"grok usage: input={input_tokens} output={output_tokens} "
            f"(reasoning={reasoning_tokens})",
        )

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

        dbg("CLIENT", "grok: yielding StreamCompleted, complete_stream END")
        yield StreamCompleted(final_response=chat_response)

    # ---------- Internal helpers ----------

    def _build_input(self, request: ChatRequest) -> list[dict]:
        """Build the Responses API input list.

        Phase 1: PDF-only file support. Each xai-provider FileRef becomes
        an {input_file, file_id} block ahead of the user's prompt text.
        Files first, prompt last — empirically models follow context
        better when files appear before the question. Same convention
        as openai_client.py.

        If no files are attached, returns the prompt as a plain string,
        which the SDK accepts directly. This keeps the no-files case
        simple and matches xAI's quickstart examples.
        """
        xai_refs = [r for r in request.file_refs if r.provider == "xai"]

        if not xai_refs:
            # No files — pass prompt as a plain string. SDK accepts both
            # a string and a structured input list.
            return request.prompt

        content_blocks: list[dict] = []
        for ref in xai_refs:
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