"""Gemini client — calls Google's Gemini API via the google-genai SDK.

complete() — synchronous, kept for any caller that needs blocking behavior.
complete_stream() — streaming with thinking-aware status reporting.

Gemini 2.5 Flash reasons internally before producing visible text. The
SDK doesn't expose reasoning content (no per-chunk thought attribute),
only the post-hoc thoughts_token_count. We yield StreamThinking
immediately when the stream opens, then yield StreamStarted on the
first chunk that contains text. This drives the THINKING → STREAMING
badge transition in the UI honestly — users see when the model is
reasoning vs producing visible output.

Cost: thoughts_token_count is added to output_tokens for cost
calculation. The user pays for reasoning whether they see it or not;
hiding it from the comparison would underreport Gemini's true cost.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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

# Gemini 3.x family — uses thinking_level (not thinking_budget) and
# Google explicitly recommends NOT passing temperature (see migration
# guide: temperature on v3 can cause looping/degradation on complex
# tasks). Detected by api_model_name prefix.
_GEMINI3_PREFIXES = ("gemini-3-", "gemini-3.1-")


def _is_gemini3(api_model_name: str) -> bool:
    return api_model_name.startswith(_GEMINI3_PREFIXES)


class _GeminiRateLimitMarker(Exception):
    """Internal marker raised only when Gemini returns HTTP 429.

    Gemini's SDK lumps every HTTP status into a single APIError. The
    retry helper needs to catch ONLY 429s. This marker is raised by
    our own classification logic and consumed by the helper.
    """


def _parse_gemini_retry_after(exc: BaseException) -> int | None:
    """Gemini's SDK doesn't expose response headers. Always None →
    retry helper falls back to its 1s/2s/4s exponential schedule."""
    return None


class GeminiClient(BaseProviderClient):
    """Client for Google's Gemini models via aistudio API key."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    # ---------- BaseProviderClient contract ----------

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.GOOGLE_API_KEY)

    def complete(self, request: ChatRequest) -> ChatResponse:
        """Synchronous completion. Preserved for any non-streaming caller."""
        if not self.is_configured():
            raise NotConfiguredError(
                "Google API key not set. Go to Settings to add it."
            )

        api_key = self._settings.get_secret(SecretKey.GOOGLE_API_KEY)
        client = genai.Client(api_key=api_key)

        contents = self._build_contents(request)

        config_kwargs: dict = {
            "max_output_tokens": request.max_tokens,
        }
        if _is_gemini3(request.model.api_model_name):
            # Per Google: don't pass temperature on v3; cap thinking at high.
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_level="high"
            )
        else:
            config_kwargs["temperature"] = request.temperature
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        config = genai_types.GenerateContentConfig(**config_kwargs)

        start = time.monotonic()

        def _do_call():
            try:
                return client.models.generate_content(
                    model=request.model.api_model_name,
                    contents=contents,
                    config=config,
                )
            except genai_errors.APIError as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                message = getattr(exc, "message", str(exc))

                if status in (401, 403):
                    raise AuthenticationError(
                        "Google rejected the API key. "
                        "Check it at aistudio.google.com.",
                        raw=str(exc),
                    ) from exc
                if status == 429:
                    raise _GeminiRateLimitMarker(str(exc)) from exc
                raise ProviderError(
                    f"Gemini error: {message}",
                    raw=str(exc),
                ) from exc

        try:
            response = with_rate_limit_retry(
                fn=_do_call,
                sdk_rate_limit_exception=_GeminiRateLimitMarker,
                parse_retry_after_seconds=_parse_gemini_retry_after,
                provider_label="Gemini",
            )
        except Exception as exc:
            from providers.base import (
                AuthenticationError as _AuthErr,
                ProviderError as _ProviderErr,
                RateLimitError as _RateErr,
            )
            if isinstance(exc, (_AuthErr, _ProviderErr, _RateErr)):
                raise
            raise ProviderError(
                f"Unexpected error calling Gemini: {exc}",
                raw=str(exc),
            ) from exc

        latency = time.monotonic() - start

        text = response.text or ""

        usage = getattr(response, "usage_metadata", None)
        input_tokens = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        output_tokens = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        thoughts = (getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
        # Add thinking tokens to output for honest cost reporting.
        output_tokens_total = output_tokens + thoughts

        cost = calculate_cost_usd(request.model, input_tokens, output_tokens_total)

        return ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens_total,
            cost_usd=cost,
            served_model=request.model.api_model_name,
        )

    def complete_stream(
        self,
        request: ChatRequest,
        cancel_flag: threading.Event,
    ) -> Iterator["StreamEvent"]:
        """Stream a response from Gemini.

        Event sequence:
            StreamThinking (immediately after stream opens)
            StreamStarted (when first text chunk is about to arrive)
            TextDelta * N (one per text chunk)
            Usage (from final chunk's usage_metadata)
            StreamCompleted

        On error: StreamFailed. On cancel: StreamCancelled.

        Reasoning happens server-side during the gap between
        StreamThinking and StreamStarted. The SDK doesn't expose
        reasoning content; thoughts_token_count is available on
        usage_metadata of the final chunk and is added to output_tokens
        for accurate cost reporting.
        """
        dbg("CLIENT", f"gemini.complete_stream START for {request.model.id}")

        if cancel_flag.is_set():
            dbg("CLIENT", "gemini: cancel_flag already set, yielding StreamCancelled")
            yield StreamCancelled()
            return

        if not self.is_configured():
            dbg("CLIENT", "gemini: not configured, yielding StreamFailed")
            yield StreamFailed(
                error=NotConfiguredError(
                    "Google API key not set. Go to Settings to add it."
                )
            )
            return

        api_key = self._settings.get_secret(SecretKey.GOOGLE_API_KEY)
        client = genai.Client(api_key=api_key)

        contents = self._build_contents(request)

        config_kwargs: dict = {
            "max_output_tokens": request.max_tokens,
        }
        if _is_gemini3(request.model.api_model_name):
            # Per Google: don't pass temperature on v3; cap thinking at high.
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_level="high"
            )
        else:
            config_kwargs["temperature"] = request.temperature
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        config = genai_types.GenerateContentConfig(**config_kwargs)

        start = time.monotonic()

        # Open the stream. Wrapped in retry helper for 429 handling.
        # Other errors are classified inside _do_call and propagate.
        def _do_call():
            try:
                return client.models.generate_content_stream(
                    model=request.model.api_model_name,
                    contents=contents,
                    config=config,
                )
            except genai_errors.APIError as exc:
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                message = getattr(exc, "message", str(exc))

                if status in (401, 403):
                    raise AuthenticationError(
                        "Google rejected the API key. "
                        "Check it at aistudio.google.com.",
                        raw=str(exc),
                    ) from exc
                if status == 429:
                    raise _GeminiRateLimitMarker(str(exc)) from exc
                raise ProviderError(
                    f"Gemini error: {message}",
                    raw=str(exc),
                ) from exc

        dbg("CLIENT", "gemini: calling with_rate_limit_retry to open stream")
        start = time.monotonic()
        try:
            stream = with_rate_limit_retry(
                fn=_do_call,
                sdk_rate_limit_exception=_GeminiRateLimitMarker,
                parse_retry_after_seconds=_parse_gemini_retry_after,
                provider_label="Gemini",
            )
        except AuthenticationError as exc:
            dbg("CLIENT", f"gemini: AuthError caught: {exc}")
            yield StreamFailed(error=exc)
            return
        except RateLimitError as exc:
            dbg("CLIENT", f"gemini: RateLimitError caught (after retries): {exc}")
            yield StreamFailed(error=exc)
            return
        except ProviderError as exc:
            dbg("CLIENT", f"gemini: ProviderError caught: {exc}")
            yield StreamFailed(error=exc)
            return
        except Exception as exc:
            dbg("CLIENT", f"gemini: UNEXPECTED exception caught: {type(exc).__name__}: {exc}")
            yield StreamFailed(
                error=ProviderError(
                    f"Unexpected error opening Gemini stream: {exc}",
                    raw=str(exc),
                )
            )
            return

        # Stream is open. Gemini reasons before producing visible text —
        # signal THINKING immediately so the UI shows a thinking badge
        # during the wait.
        dbg("CLIENT", "gemini: stream open, yielding StreamThinking")
        yield StreamThinking()

        first_text_emitted = False
        last_chunk = None
        accumulated_text = ""
        served_model = request.model.api_model_name

        try:
            for chunk in stream:
                if cancel_flag.is_set():
                    dbg("CLIENT", "gemini: cancel observed mid-stream")
                    try:
                        # Gemini's stream iterator doesn't have a clean
                        # close() method; just stop iterating. The SDK
                        # will eventually clean up.
                        pass
                    except Exception:
                        pass
                    yield StreamCancelled()
                    return

                last_chunk = chunk
                text = getattr(chunk, "text", None) or ""

                if text:
                    if not first_text_emitted:
                        # First chunk with visible text — transition the
                        # UI from THINKING to STREAMING.
                        dbg("CLIENT", "gemini: first text arrived, yielding StreamStarted")
                        yield StreamStarted(model=served_model)
                        first_text_emitted = True

                    accumulated_text += text
                    dbg("CLIENT", f"gemini: yielding TextDelta len={len(text)}")
                    yield TextDelta(text=text)

        except genai_errors.APIError as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            message = getattr(exc, "message", str(exc))
            dbg("CLIENT", f"gemini: APIError mid-stream: status={status} {message}")
            if status == 429:
                yield StreamFailed(
                    error=RateLimitError(
                        f"Gemini rate limited mid-stream: {message}",
                        raw=str(exc),
                    )
                )
            else:
                yield StreamFailed(
                    error=ProviderError(
                        f"Gemini error mid-stream: {message}",
                        raw=str(exc),
                    )
                )
            return
        except Exception as exc:
            dbg("CLIENT", f"gemini: exception during event loop: {type(exc).__name__}: {exc}")
            yield StreamFailed(
                error=ProviderError(
                    f"Gemini stream error: {exc}",
                    raw=str(exc),
                )
            )
            return

        latency = time.monotonic() - start

        # Defensive: if the stream completed without any text chunks,
        # we never emitted StreamStarted. This is unusual but possible
        # if Gemini returns an empty response. Emit StreamStarted now
        # so the card transitions out of THINKING before completion.
        if not first_text_emitted:
            dbg("CLIENT", "gemini: no text was emitted, yielding StreamStarted defensively")
            yield StreamStarted(model=served_model)

        # Extract usage from the last chunk we saw.
        usage = getattr(last_chunk, "usage_metadata", None) if last_chunk else None
        input_tokens = (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        candidate_tokens = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        thoughts = (getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
        # Add thinking tokens to output for honest cost reporting —
        # the user pays for reasoning whether they see it or not.
        output_tokens_total = candidate_tokens + thoughts

        dbg(
            "CLIENT",
            f"gemini usage: input={input_tokens} candidates={candidate_tokens} "
            f"thoughts={thoughts} output_total={output_tokens_total}",
        )

        yield Usage(input_tokens=input_tokens, output_tokens=output_tokens_total)

        cost = calculate_cost_usd(request.model, input_tokens, output_tokens_total)

        chat_response = ChatResponse(
            text=accumulated_text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens_total,
            cost_usd=cost,
            served_model=served_model,
        )

        dbg("CLIENT", "gemini: yielding StreamCompleted, complete_stream END")
        yield StreamCompleted(final_response=chat_response)

    # ---------- Internal helpers ----------

    def _build_contents(self, request: ChatRequest) -> list:
        """Build Gemini contents list — files first (Part.from_uri), then prompt."""
        parts: list = []

        for ref in request.file_refs:
            if ref.provider != "gemini":
                continue
            parts.append(
                genai_types.Part.from_uri(
                    file_uri=ref.remote_id,
                    mime_type=ref.mime_type,
                )
            )

        parts.append(request.prompt)
        return parts