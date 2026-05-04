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

The shape is determined by the FileRef's MIME type at chat time. The
registry stores the real MIME, so this client always sees the truth.

Migrated from Chat Completions in Phase 1; document support added in
Phase 2; image support fixed Apr 2026 (Phase 2f) when we discovered that
images uploaded as user_data + referenced via input_file fail with an
extension-allowlist error.
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
from providers._retry import with_rate_limit_retry

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