"""
Google (Gemini) client — calls the Gemini API via the official `google-genai`
Python SDK.

File support: Gemini's `generate_content` accepts a list of parts in
`contents`. Files attach as parts referencing the URI returned by the
Files API. The SDK provides `types.Part.from_uri()` for this.

Differences from our other clients:
- Method is `client.models.generate_content(...)`, not chat.completions/messages.
- Config goes in a separate GenerateContentConfig object, not as kwargs.
- Response text is accessed as `response.text` (a property, not a method).
- Token counts live in `response.usage_metadata.{prompt_token_count, candidates_token_count}`.
- Files referenced by URI (e.g. `files/abc-xyz`), not by file_id.
"""
from __future__ import annotations

import time

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
from settings_manager import SecretKey, SettingsManager
from providers._retry import with_rate_limit_retry

class _GeminiRateLimitMarker(Exception):
    """Internal marker raised only when Gemini returns HTTP 429.

    Why a private class:
        Gemini's SDK raises a single generic genai_errors.APIError for
        every HTTP status — 401, 429, 500, etc. The retry helper needs
        to catch ONLY 429s, not all API errors. Telling the helper to
        catch APIError directly would also retry auth and server errors
        that will never succeed.

        Instead, we wrap the API call, detect 429 via exc.code ourselves,
        and re-raise as this marker. The helper catches the marker
        specifically. All other Gemini errors propagate immediately
        through the helper without retry.

    This class never escapes this module — the retry helper consumes it
    and either returns a successful response or raises base.RateLimitError
    with the "rate limited after 3 retries" message.
    """


def _parse_gemini_retry_after(exc: BaseException) -> int | None:
    """Gemini's SDK doesn't expose response headers on its exceptions,
    so we can't read retry-after. Always returning None makes the retry
    helper fall back to its exponential schedule (1s, 2s, 4s). Defined
    as a real function instead of a lambda for clearer stack traces if
    something does eventually go wrong here."""
    return None

class GeminiClient(BaseProviderClient):
    """Client for Google's Gemini models via aistudio API key."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    # ---------- BaseProviderClient contract ----------

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.GOOGLE_API_KEY)

    def complete(self, request: ChatRequest) -> ChatResponse:
        if not self.is_configured():
            raise NotConfiguredError(
                "Google API key not set. Go to Settings to add it."
            )

        api_key = self._settings.get_secret(SecretKey.GOOGLE_API_KEY)
        client = genai.Client(api_key=api_key)

        # Build the contents list — files first, then the user's prompt.
        contents = self._build_contents(request)

        # Build config object — this is where temperature, max_tokens,
        # and system_instruction live in the new SDK.
        config_kwargs: dict = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        config = genai_types.GenerateContentConfig(**config_kwargs)

        start = time.monotonic()
        def _do_call():
            """Inner function passed to the retry helper.

            Wraps the actual SDK call AND the error-classification logic,
            because Gemini's lumped-together APIError needs to be split
            into specific error types here. 429s become our private marker
            (which the retry helper catches and retries); everything else
            becomes a domain exception that propagates straight through
            the helper without retry.
            """
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
                    # Re-raise as our private marker so the retry helper
                    # can catch it specifically. The helper either
                    # eventually returns a successful response or raises
                    # base.RateLimitError after 3 retries are exhausted.
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
            # Re-raise our domain exceptions cleanly; wrap unexpected ones.
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

        cost = calculate_cost_usd(request.model, input_tokens, output_tokens)

        return ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            served_model=request.model.api_model_name,
        )

    # ---------- Internal helpers ----------

    def _build_contents(self, request: ChatRequest) -> list:
        """Build the contents list for generate_content().

        Files come first (as Part.from_uri references), then the user's
        text prompt. The SDK accepts mixed lists of strings and Part
        objects in a single contents call.
        """
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