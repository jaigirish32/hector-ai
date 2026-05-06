"""Anthropic (Claude) client — calls api.anthropic.com via the official
`anthropic` Python SDK.

File support — three content block shapes per MIME:
1. 'image' — PNG, JPEG, GIF, WEBP. Native vision.
2. 'document' — PDFs. Native PDF reading.
3. 'container_upload' + code execution tool — office formats (xlsx/docx/pptx/etc).
   Sandboxed Python container reads the file via pandas/openpyxl/python-docx.

Streaming (v0.2.0):
- complete_stream() yields StreamEvent values as the response is generated.
- Errors are emitted as StreamFailed, never raised.
- Cancellation observed via cancel_flag between events.
- Rate-limit retry covers stream opening only — no mid-stream retry.

Extended thinking (v0.2.0):
- Sonnet 4.5/4.6 use thinking={"type": "adaptive"} so the model decides
  reasoning budget based on question complexity.
- Thinking is incompatible with custom temperature — temperature is
  dropped when thinking is enabled.
- Stream emits StreamThinking when entering the thinking block, then
  StreamStarted when entering the first text block. Card UX: LOADING →
  THINKING → STREAMING → COMPLETE.
- thinking_delta content is logged via dbg() but not surfaced to the UI.
- thoughts billing: Anthropic's thinking tokens are counted in
  output_tokens automatically — usage.output_tokens already includes them.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    AuthenticationError as AnthropicAuthError,
    RateLimitError as AnthropicRateLimitError,
)
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageStartEvent,
)

from providers.base import (
    AuthenticationError,
    BaseProviderClient,
    ChatRequest,
    ChatResponse,
    FileRef,
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
    StreamEvent,
    StreamFailed,
    StreamStarted,
    StreamThinking,
    TextDelta,
    Usage,
)
from settings_manager import SecretKey, SettingsManager

ANTHROPIC_FILES_BETA = "files-api-2025-04-14"
ANTHROPIC_EXTENDED_CACHE_BETA = "extended-cache-ttl-2025-04-11"
ANTHROPIC_CODE_EXEC_BETA = "code-execution-2025-08-25"

ANTHROPIC_CODE_EXEC_TOOL = {
    "type": "code_execution_20250825",
    "name": "code_execution",
}

# Thinking budget for Anthropic models that support extended thinking.
# Using "adaptive" lets the model choose its own budget based on
# question complexity — cheaper than a fixed budget for simple
# questions, generous for hard ones.
ANTHROPIC_THINKING_CONFIG = {"type": "adaptive"}

# Models that support extended thinking. As of v0.2.0, only Sonnet 4.5
# and 4.6 are wired into HECTOR. If older Claude models are added later,
# this set must NOT include them — they'll error out on the thinking
# parameter.
_THINKING_MODELS = frozenset({
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-6",
})

_IMAGE_MIMES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})

_DOCUMENT_MIMES = frozenset({
    "application/pdf",
})

_OFFICE_MIMES = frozenset({
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint",
})

_MIME_TO_FRIENDLY: dict[str, tuple[str, str]] = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        ("Microsoft Excel workbook (.xlsx)", "openpyxl or pandas"),
    "application/vnd.ms-excel":
        ("Legacy Microsoft Excel workbook (.xls)", "xlrd or pandas"),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        ("Microsoft Word document (.docx)", "python-docx"),
    "application/msword":
        ("Legacy Microsoft Word document (.doc)", "antiword or python-docx"),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        ("Microsoft PowerPoint presentation (.pptx)", "python-pptx"),
    "application/vnd.ms-powerpoint":
        ("Legacy Microsoft PowerPoint presentation (.ppt)", "python-pptx"),
}


def _parse_anthropic_retry_after(exc: BaseException) -> int | None:
    """Read retry-after header from an Anthropic SDK exception."""
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


def _model_supports_thinking(api_model_name: str) -> bool:
    """Whether the given Anthropic model accepts the thinking parameter."""
    return api_model_name in _THINKING_MODELS


class AnthropicClient(BaseProviderClient):
    """Client for api.anthropic.com (Claude models)."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.ANTHROPIC_API_KEY)

    def complete_stream(
        self,
        request: ChatRequest,
        cancel_flag: threading.Event,
    ) -> Iterator[StreamEvent]:
        """Stream a completion from Anthropic, yielding StreamEvent values."""
        dbg("CLIENT", f"anthropic.complete_stream START for {request.model.id}")

        if cancel_flag.is_set():
            dbg("CLIENT", "anthropic: cancel_flag already set, yielding StreamCancelled")
            yield StreamCancelled()
            return

        if not self.is_configured():
            dbg("CLIENT", "anthropic: not configured, yielding StreamFailed")
            yield StreamFailed(
                NotConfiguredError(
                    "Anthropic API key not set. Go to Settings to add it."
                )
            )
            return

        api_key = self._settings.get_secret(SecretKey.ANTHROPIC_API_KEY)
        client = Anthropic(api_key=api_key)

        api_model = request.model.api_model_name
        thinking_enabled = _model_supports_thinking(api_model)

        office_refs = [
            r for r in request.file_refs
            if r.provider == "anthropic" and r.mime_type in _OFFICE_MIMES
        ]
        needs_code_exec = bool(office_refs)

        beta_headers = [ANTHROPIC_FILES_BETA]
        if needs_code_exec:
            beta_headers.append(ANTHROPIC_CODE_EXEC_BETA)
        if request.file_refs:
            beta_headers.append(ANTHROPIC_EXTENDED_CACHE_BETA)

        # Build kwargs. When thinking is enabled, we MUST drop temperature
        # (Anthropic rejects custom temperature with thinking enabled).
        # When thinking is disabled, temperature is sent as normal.
        create_kwargs: dict = {
            "model": api_model,
            "messages": [
                {"role": "user", "content": self._build_user_content(request)},
            ],
            "max_tokens": request.max_tokens,
            "extra_headers": {"anthropic-beta": ",".join(beta_headers)},
        }
        if thinking_enabled:
            create_kwargs["thinking"] = ANTHROPIC_THINKING_CONFIG
            dbg("CLIENT", f"anthropic: thinking enabled for {api_model} (adaptive)")
        else:
            create_kwargs["temperature"] = request.temperature

        if request.system_prompt:
            create_kwargs["system"] = request.system_prompt
        if needs_code_exec:
            create_kwargs["tools"] = [ANTHROPIC_CODE_EXEC_TOOL]

        start = time.monotonic()

        def _open_anthropic_stream() -> tuple:
            """Open the stream; return (cm, stream). Cleans up cm on failure."""
            cm = client.messages.stream(**create_kwargs)
            try:
                stream = cm.__enter__()
                return cm, stream
            except BaseException:
                cm.__exit__(None, None, None)
                raise

        dbg("CLIENT", "anthropic: calling with_rate_limit_retry to open stream")
        try:
            cm, stream = with_rate_limit_retry(
                fn=_open_anthropic_stream,
                sdk_rate_limit_exception=AnthropicRateLimitError,
                parse_retry_after_seconds=_parse_anthropic_retry_after,
                provider_label="Anthropic",
            )
        except AnthropicAuthError as exc:
            dbg("CLIENT", f"anthropic: AuthError caught: {exc}")
            yield StreamFailed(
                AuthenticationError(
                    "Anthropic rejected the API key. "
                    "Check it at console.anthropic.com and confirm you have credit.",
                    raw=str(exc),
                )
            )
            return
        except AnthropicRateLimitError as exc:
            dbg("CLIENT", f"anthropic: RateLimitError caught (after retries): {exc}")
            yield StreamFailed(
                RateLimitError(
                    "Anthropic rate limited after 3 retries.",
                    raw=str(exc),
                )
            )
            return
        except APIConnectionError as exc:
            dbg("CLIENT", f"anthropic: ConnectionError caught: {exc}")
            yield StreamFailed(
                ProviderError(
                    "Could not reach Anthropic — check your internet connection.",
                    raw=str(exc),
                )
            )
            return
        except APIError as exc:
            dbg("CLIENT", f"anthropic: APIError caught: {exc}")
            message = getattr(exc, "message", str(exc))
            yield StreamFailed(
                ProviderError(
                    f"Anthropic error: {message}",
                    raw=str(exc),
                )
            )
            return
        except Exception as exc:
            dbg("CLIENT", f"anthropic: UNEXPECTED exception caught: {type(exc).__name__}: {exc}")
            yield StreamFailed(
                ProviderError(
                    f"Unexpected error opening Anthropic stream: {exc}",
                    raw=str(exc),
                )
            )
            return

        # Stream is open. Iterate events.
        # Tracking state for thinking-aware emission:
        #   served_model — captured from RawMessageStartEvent, used when
        #     yielding StreamStarted later
        #   current_block_type — "thinking", "text", or other; updated
        #     on each RawContentBlockStartEvent
        #   thinking_emitted — guard so StreamThinking is yielded once
        #   streaming_emitted — guard so StreamStarted is yielded once
        served_model = api_model
        current_block_type: str | None = None
        thinking_emitted = False
        streaming_emitted = False

        try:
            for event in stream:
                if cancel_flag.is_set():
                    dbg("CLIENT", "anthropic: cancel observed mid-stream")
                    try:
                        stream.close()
                    except Exception:
                        pass
                    yield StreamCancelled()
                    return

                if isinstance(event, RawMessageStartEvent):
                    served_model = (
                        getattr(event.message, "model", None) or api_model
                    )
                    # Don't yield StreamStarted yet — wait for first text
                    # block. For thinking responses we yield StreamThinking
                    # first; for non-thinking responses StreamStarted will
                    # fire on RawContentBlockStartEvent(text) which arrives
                    # almost immediately.

                elif isinstance(event, RawContentBlockStartEvent):
                    content_block = getattr(event, "content_block", None)
                    block_type = getattr(content_block, "type", None) if content_block else None
                    current_block_type = block_type
                    dbg("CLIENT", f"anthropic: content block start type={block_type}")

                    if block_type == "thinking" and not thinking_emitted:
                        # Reasoning is about to begin. Flip card to THINKING.
                        dbg("CLIENT", "anthropic: yielding StreamThinking")
                        yield StreamThinking()
                        thinking_emitted = True
                    elif block_type == "text" and not streaming_emitted:
                        # First visible text block. Flip card to STREAMING.
                        dbg("CLIENT", "anthropic: yielding StreamStarted")
                        yield StreamStarted(model=served_model)
                        streaming_emitted = True

                elif isinstance(event, RawContentBlockDeltaEvent):
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue

                    # text_delta: delta.text — yield TextDelta
                    # thinking_delta: delta.thinking — log only, never to UI
                    # tool_use deltas: input_json_delta etc — silently consumed
                    text = getattr(delta, "text", None)
                    if text:
                        dbg("CLIENT", f"anthropic: yielding TextDelta len={len(text)}")
                        yield TextDelta(text=text)
                        continue

                    thinking_text = getattr(delta, "thinking", None)
                    if thinking_text:
                        # Log thinking content for diagnostics; do not
                        # surface to the UI. Truncated preview keeps the
                        # log readable for long reasoning passes.
                        preview = thinking_text[:80].replace("\n", " ")
                        dbg("CLIENT", f"anthropic thinking: {preview}...")

                # Other events (RawContentBlockStopEvent, RawMessageDeltaEvent,
                # RawMessageStopEvent) silently consumed — final_message
                # accumulator handles them via the SDK.

        except AnthropicRateLimitError as exc:
            yield StreamFailed(
                RateLimitError(
                    "Anthropic rate limited mid-response.",
                    raw=str(exc),
                )
            )
            return
        except APIError as exc:
            message = getattr(exc, "message", str(exc))
            yield StreamFailed(
                ProviderError(
                    f"Anthropic error during stream: {message}",
                    raw=str(exc),
                )
            )
            return
        except Exception as exc:
            yield StreamFailed(
                ProviderError(
                    f"Unexpected error during Anthropic stream: {exc}",
                    raw=str(exc),
                )
            )
            return
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:
                pass

        # Defensive: if we never saw a text block (extremely unusual),
        # emit StreamStarted now so the card transitions out of THINKING
        # before completion. Without this the card would stay stuck.
        if not streaming_emitted:
            dbg("CLIENT", "anthropic: no text block seen, yielding StreamStarted defensively")
            yield StreamStarted(model=served_model)

        # Stream completed. Assemble ChatResponse from the final message.
        try:
            final_message = stream.get_final_message()
        except Exception as exc:
            yield StreamFailed(
                ProviderError(
                    f"Failed to assemble final response: {exc}",
                    raw=str(exc),
                )
            )
            return

        # Extract text from text blocks only. Thinking blocks have a
        # 'thinking' attribute, not 'text', so they are naturally excluded
        # here. Tool-use blocks have other shapes; the hasattr check
        # tolerates them.
        text_parts: list[str] = []
        for block in final_message.content or []:
            if hasattr(block, "text") and block.text:
                text_parts.append(block.text)
        text = "\n\n".join(text_parts)

        usage = final_message.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        dbg(
            "CLIENT",
            f"anthropic usage: input={input_tokens} output={output_tokens} "
            f"cache_write={cache_creation} cache_read={cache_read}",
        )
        cost = calculate_cost_usd(request.model, input_tokens, output_tokens)

        dbg("CLIENT", f"anthropic: yielding Usage(in={input_tokens}, out={output_tokens})")
        yield Usage(input_tokens=input_tokens, output_tokens=output_tokens)

        latency = time.monotonic() - start

        final_response = ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            served_model=final_message.model or api_model,
        )
        dbg("CLIENT", "anthropic: yielding StreamCompleted, complete_stream END")
        yield StreamCompleted(final_response=final_response)

    # ---------- Internal helpers ----------

    def _build_user_content(self, request: ChatRequest) -> list[dict]:
        """Build the user message content as a list of typed blocks.

        File ordering:
          PDF only:     [{document}, {prompt}]
          Image only:   [{image}, {prompt}]
          xlsx only:    [{metadata text}, {container_upload}, {prompt}]
          Mixed:        [{document}, {image}, {metadata text}, {container_upload}, {prompt}]

        Files come BEFORE the prompt — empirically, models follow context
        better when files appear ahead of the question.

        Prompt caching: the LAST file-related block gets cache_control
        with ttl="1h". Anthropic prefix-caches everything up to and
        including the marker. The user's prompt text after this is NOT
        cached (changes per query).
        """
        anthropic_refs = [r for r in request.file_refs if r.provider == "anthropic"]

        image_refs = [r for r in anthropic_refs if r.mime_type in _IMAGE_MIMES]
        document_refs = [r for r in anthropic_refs if r.mime_type in _DOCUMENT_MIMES]
        office_refs = [r for r in anthropic_refs if r.mime_type in _OFFICE_MIMES]

        blocks: list[dict] = []

        for ref in document_refs:
            blocks.append({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": ref.remote_id,
                },
            })

        for ref in image_refs:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": ref.remote_id,
                },
            })

        if office_refs:
            blocks.append({
                "type": "text",
                "text": _build_office_metadata_text(office_refs),
            })
            for ref in office_refs:
                blocks.append({
                    "type": "container_upload",
                    "file_id": ref.remote_id,
                })

        if blocks:
            blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}

        blocks.append({"type": "text", "text": request.prompt})
        return blocks


def _build_office_metadata_text(office_refs: list[FileRef]) -> str:
    """Metadata-instruction text shown to Claude before container_upload blocks.

    Tells Claude the real filename, file type, and which Python library
    to use, plus the cp-rename trick to satisfy openpyxl's extension
    checks (the upload is anonymous bytes named blob.bin in the container).
    """
    if len(office_refs) == 1:
        ref = office_refs[0]
        label, lib = _MIME_TO_FRIENDLY.get(
            ref.mime_type,
            ("an office document", "an appropriate Python library"),
        )
        return (
            f"I have attached one file. Here is the metadata I am tracking "
            f"for it externally:\n"
            f"  - Original filename: {ref.filename}\n"
            f"  - File type: {label}\n\n"
            f"This file is stored at Anthropic as anonymous bytes "
            f"(filename 'blob.bin', MIME 'application/octet-stream') and "
            f"is mounted in $INPUT_DIR. Please open it as {label} using "
            f"{lib}. If the library refuses 'blob.bin' due to extension "
            f"checking, copy or rename it first "
            f"(e.g. cp \"$INPUT_DIR/blob.bin\" /tmp/{ref.filename})."
        )

    lines = [
        "I have attached multiple files. Here is the metadata I am "
        "tracking for them externally:",
        "",
    ]
    for i, ref in enumerate(office_refs, start=1):
        label, lib = _MIME_TO_FRIENDLY.get(
            ref.mime_type,
            ("an office document", "an appropriate Python library"),
        )
        lines.append(
            f"  {i}. {ref.filename} ({label}) — open with {lib}."
        )
    lines.append("")
    lines.append(
        "All files are stored at Anthropic as anonymous bytes (filenames "
        "are all 'blob.bin' inside $INPUT_DIR — the file_ids in the "
        "subsequent container_upload blocks distinguish them, in the same "
        "order as listed above). If a library refuses 'blob.bin' due to "
        "extension checking, copy/rename to the original filename first."
    )
    return "\n".join(lines)