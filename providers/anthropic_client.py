"""
Anthropic (Claude) client — calls api.anthropic.com via the official
`anthropic` Python SDK.

Differences from our OpenAI client:
- Endpoint method is `messages.create` (or `messages.stream` for streaming),
  not `chat.completions.create`.
- System prompt is a TOP-LEVEL parameter, not a message in the list.
- Response content is a list of blocks; we read text from each block.
- `max_tokens` is REQUIRED by Anthropic (not optional).
- Token field names are `input_tokens` and `output_tokens` (not prompt/completion).

File support has THREE paths, chosen per file based on its MIME type:

1. 'image' content block — for images (PNG, JPEG).
   {"type": "image", "source": {"type": "file", "file_id": "..."}}
   Requires the files-api beta header. Claude reads the image natively.

2. 'document' content block — for PDFs (and plaintext, but HECTOR doesn't
   send plaintext through this path).
   {"type": "document", "source": {"type": "file", "file_id": "..."}}
   Requires the files-api beta header. Claude reads the PDF natively
   (vision + text extraction).

3. 'container_upload' content block + code execution tool — for office
   formats (xlsx/docx/pptx and their legacy counterparts xls/doc/ppt).
   Anthropic spins up a sandboxed Python container, copies the file in,
   and Claude writes pandas/openpyxl/python-docx/python-pptx code to
   read it. Requires BOTH the files-api beta header AND the code-execution
   beta header. Office files were uploaded as anonymous bytes (blob.bin)
   so we prepend a metadata text block telling Claude what the file
   actually is.

The shape of the content block is determined entirely by the MIME type of
the FileRef at chat time. The registry stores the real MIME, so this
client always sees the truth even when the upload was rewritten as
anonymous bytes.

v0.2.0 streaming migration
--------------------------
The non-streaming complete() method has been removed. complete_stream()
yields StreamEvent values as the response is generated. The streaming
flow:

  StreamStarted     — emitted once after the SDK acknowledges the request
  TextDelta x N     — emitted for each text chunk Anthropic sends
  Usage             — emitted once at the end, with final token counts
  StreamCompleted   — emitted once at the end, carrying the assembled
                      ChatResponse (built via the SDK's get_final_message
                      helper, parsed by the same logic that complete()
                      used to use)
  StreamFailed      — emitted if any error occurs (auth, connection,
                      rate-limit-after-exhaustion, unexpected). After
                      this, the stream is over.
  StreamCancelled   — emitted if the worker's cancel_flag is observed
                      between events. The SDK stream is closed cleanly
                      so the connection releases and no further tokens
                      are billed.

SDK types used (verified against the installed anthropic SDK):
  - client.messages.stream(...) returns a MessageStreamManager (context manager).
  - Entering the context manager returns a MessageStream object.
  - MessageStream.get_final_message() returns a ParsedMessage with .model,
    .content (list of ParsedTextBlock and other block types), .usage.
  - Iterating the stream yields RawMessageStartEvent, RawContentBlockStartEvent,
    RawContentBlockDeltaEvent, RawContentBlockStopEvent, RawMessageDeltaEvent,
    RawMessageStopEvent in order.
  - We only react to two: RawMessageStartEvent (for StreamStarted) and
    RawContentBlockDeltaEvent (for TextDelta when delta.text is present).

Code execution intermediate steps (Claude writing/running code) do NOT
emit dedicated stream events in v0.2.0 — they are silently consumed by
the streaming loop, but appear correctly in the final ChatResponse via
get_final_message. v0.3.0 will add ToolCallStarted / ToolCallResult
events so the UI can show a live trace of code execution.

Rate-limit retry covers ONLY stream opening — once events start flowing,
we do not retry mid-stream (showing the user partial text and then
restarting would be confusing). If the rate limit is exhausted past all
retries, StreamFailed(RateLimitError) is yielded.
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
# Verified via Probe 2: these are the exact SDK event class names used
# by the installed `anthropic` package. The SDK's own TextDelta type
# (anthropic.types.TextDelta) shares a name with our streaming.TextDelta;
# we deliberately do NOT import the SDK's TextDelta — we access
# event.delta.text via attribute access, which works regardless of the
# delta's concrete type.
from anthropic.types import (
    RawContentBlockDeltaEvent,
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
from providers._retry import with_rate_limit_retry
from providers.streaming import (
    StreamCancelled,
    StreamCompleted,
    StreamEvent,
    StreamFailed,
    StreamStarted,
    TextDelta,
    Usage,
)
from settings_manager import SecretKey, SettingsManager
from providers._dbg import dbg

# Anthropic requires this beta header to accept content blocks that
# reference uploaded file_ids (image, document, container_upload).
ANTHROPIC_FILES_BETA = "files-api-2025-04-14"

# Beta header for the extended (1-hour) prompt cache TTL. Required when
# any cache_control block uses ttl="1h". Without this header, the SDK
# falls back to the default 5-minute cache.
ANTHROPIC_EXTENDED_CACHE_BETA = "extended-cache-ttl-2025-04-11"

# Beta header for the code execution tool. Required when any office-format
# file is in the request, because office files are consumed via
# container_upload blocks plus the code execution tool.
ANTHROPIC_CODE_EXEC_BETA = "code-execution-2025-08-25"

# Tool definition for the code execution sub-system. Constant — Anthropic's
# code-execution tool takes no parameters; declaring it makes it available
# to Claude during the call.
ANTHROPIC_CODE_EXEC_TOOL = {
    "type": "code_execution_20250825",
    "name": "code_execution",
}

# MIME types that map to the 'image' content block. Anthropic supports
# image/png, image/jpeg, image/gif, image/webp via the vision pipeline.
# HECTOR routes png and jpeg today; gif and webp can be added later.
_IMAGE_MIMES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})

# MIME types that map to the 'document' content block. Anthropic's
# document block currently only accepts PDFs reliably (the API explicitly
# says "Only PDF and plaintext documents are supported"). HECTOR doesn't
# route plaintext through this path; only PDFs.
_DOCUMENT_MIMES = frozenset({
    "application/pdf",
})

# MIME types that map to the 'container_upload' content block. These are
# uploaded to Anthropic as anonymous bytes (blob.bin + octet-stream) and
# consumed via the code execution path. MUST stay in sync with
# _ANTHROPIC_OFFICE_MIMES in attachments/uploaders.py — they describe
# the same set from two different sides (upload vs chat).
_OFFICE_MIMES = frozenset({
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",   # xlsx
    "application/vnd.ms-excel",                                            # xls
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "application/msword",                                                  # doc
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
    "application/vnd.ms-powerpoint",                                       # ppt
})

# Friendly names for the metadata-prompt text. Avoids dumping raw MIMEs at
# Claude. Order doesn't matter; lookup by MIME.
_MIME_TO_FRIENDLY: dict[str, tuple[str, str]] = {
    # mime -> (file_type_label, suggested_python_library)
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
    """Read the retry-after header from an Anthropic SDK exception.

    Anthropic's RateLimitError inherits from APIStatusError which has
    self.response (an httpx.Response). We read the standard retry-after
    header from there. Returns seconds to wait, or None if the header
    is missing or unparseable — caller will fall back to exponential
    backoff in that case.

    Anthropic also sometimes sends retry-after-ms (milliseconds, more
    precise). We check that first when present.
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


class AnthropicClient(BaseProviderClient):
    """Client for api.anthropic.com (Claude models)."""

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    # ---------- BaseProviderClient contract ----------

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.ANTHROPIC_API_KEY)

    def complete_stream(
        self,
        request: ChatRequest,
        cancel_flag: threading.Event,
    ) -> Iterator[StreamEvent]:
        dbg("CLIENT", f"anthropic.complete_stream START for {request.model.id}")
        """Stream a completion from Anthropic, yielding StreamEvent values.

        Errors are emitted as StreamFailed, not raised. Cancellation is
        observed via cancel_flag between SDK events; on cancellation,
        the SDK stream is closed cleanly and StreamCancelled is yielded.
        """
        # ---------- Pre-stream validation ----------
        # Cancellation requested before we even started? Honor it
        # without ever opening a connection.
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

        # ---------- Build the request ----------
        # Logic identical to the legacy complete() did. Verified via
        # Probe 3 that messages.stream() accepts the same kwargs as
        # messages.create() (model, messages, max_tokens, temperature,
        # system, tools, extra_headers).
        api_key = self._settings.get_secret(SecretKey.ANTHROPIC_API_KEY)
        client = Anthropic(api_key=api_key)

        # Decide whether code execution is needed for this request.
        # If any of the attached files is an office type, we need both
        # the code-execution beta header and the code execution tool.
        office_refs = [
            r for r in request.file_refs
            if r.provider == "anthropic" and r.mime_type in _OFFICE_MIMES
        ]
        needs_code_exec = bool(office_refs)

        # Beta headers — files-api always required when any file is
        # attached; code-exec added when office types are present.
        beta_headers = [ANTHROPIC_FILES_BETA]
        if needs_code_exec:
            beta_headers.append(ANTHROPIC_CODE_EXEC_BETA)

        if request.file_refs:
            beta_headers.append(ANTHROPIC_EXTENDED_CACHE_BETA)

        # Build kwargs explicitly so we only send `system` when set,
        # and only declare the code_execution tool when actually needed.
        create_kwargs: dict = {
            "model": request.model.api_model_name,
            "messages": [
                {"role": "user", "content": self._build_user_content(request)},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "extra_headers": {"anthropic-beta": ",".join(beta_headers)},
        }
        if request.system_prompt:
            create_kwargs["system"] = request.system_prompt
        if needs_code_exec:
            create_kwargs["tools"] = [ANTHROPIC_CODE_EXEC_TOOL]

        # ---------- Open the stream (with retry on RateLimitError) ----------
        # The retry helper wraps the OPEN of the stream — it retries the
        # initial HTTP request that establishes the streaming connection.
        # Once the stream is open and events start flowing, no retry
        # happens (we're not going to restart mid-stream and re-emit text).
        #
        # client.messages.stream() returns a MessageStreamManager (the
        # context manager). The actual HTTP request fires in __enter__.
        # If __enter__ raises after the cm is constructed, we still need
        # to clean up the cm — handled by _open_anthropic_stream below.
        start = time.monotonic()

        def _open_anthropic_stream() -> tuple:
            """Open the stream; return (cm, stream). Cleans up cm on
            failure so we don't leak a half-constructed context manager."""
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
            # Retry helper exhausted all retries — surface as RateLimitError.
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

        # ---------- Stream is open. Iterate events. ----------
        # From here on, any errors are caught and converted to
        # StreamFailed events; we never let exceptions escape the
        # generator. The cm.__exit__ in finally guarantees the SDK
        # connection is closed regardless of how we exit. Probe 4
        # confirmed get_final_message() is callable AFTER cm.__exit__.
        served_model_emitted = False

        try:
            for event in stream:
                # Cancellation check between events. If the user clicks
                # Stop, we close the SDK stream (server stops sending
                # tokens — no more billing for tokens we'd never show),
                # yield StreamCancelled, and return. The finally block
                # below also calls cm.__exit__ for full cleanup.
                if cancel_flag.is_set():
                    dbg("CLIENT", "anthropic: cancel observed mid-stream")
                    try:
                        stream.close()
                    except Exception:
                        # Best-effort close; cm.__exit__ in finally is
                        # the real cleanup.
                        pass
                    yield StreamCancelled()
                    return

                # Map SDK events to our StreamEvent types. Anthropic's
                # streaming events are typed objects from anthropic.types.
                # We handle the two we care about explicitly; all other
                # event types (RawContentBlockStartEvent,
                # RawContentBlockStopEvent, RawMessageDeltaEvent,
                # RawMessageStopEvent) are silently consumed because the
                # SDK accumulates everything we need into the final
                # message, which we read after iteration.
                if isinstance(event, RawMessageStartEvent):
                    dbg("CLIENT", "anthropic: yielding StreamStarted")
                    # First event of every stream. Carries the model
                    # name on event.message (verified via Probe 2).
                    # Emit StreamStarted exactly once.
                    if not served_model_emitted:
                        served_model = (
                            getattr(event.message, "model", None)
                            or request.model.api_model_name
                        )
                        yield StreamStarted(model=served_model)
                        served_model_emitted = True

                elif isinstance(event, RawContentBlockDeltaEvent):
                    # Workhorse event. Each one carries an incremental
                    # piece of a content block. For text blocks the
                    # event.delta is a TextDelta (Anthropic's, not ours)
                    # with a .text attribute holding the new chunk
                    # (verified via Probe 2). For tool_use blocks (code
                    # execution intermediate), the delta carries other
                    # fields (input_json_delta, etc.) which we silently
                    # consume in v0.2.0 — they appear correctly in the
                    # final ChatResponse via the SDK's get_final_message
                    # accumulation. Using getattr() rather than direct
                    # attribute access so we don't crash on unexpected
                    # delta variants.
                    delta = getattr(event, "delta", None)
                    if delta is not None:
                        text = getattr(delta, "text", None)
                        if text:
                            dbg("CLIENT", f"anthropic: yielding TextDelta len={len(text)}")
                            yield TextDelta(text=text)

        except AnthropicRateLimitError as exc:
            # Mid-stream rate limit. Extremely rare for Anthropic but
            # possible in theory. Per design: do NOT retry mid-stream —
            # surface as failure.
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
            # ALWAYS exit the context manager. Probe 4 confirmed this
            # leaves the MessageStream in a state where get_final_message
            # is still callable, so we exit BEFORE assembling the
            # ChatResponse below.
            try:
                cm.__exit__(None, None, None)
            except Exception:
                # Cleanup errors are non-fatal — connection will be
                # closed eventually by GC if SDK fails to close it now.
                pass

        # ---------- Stream completed. Assemble ChatResponse. ----------
        # The SDK has accumulated all content blocks (text + any tool
        # use) into a final ParsedMessage object. Probe 1 confirmed
        # get_final_message returns ParsedMessage with .model, .content,
        # .usage. Probe 4 confirmed it's callable AFTER cm.__exit__.
        try:
            final_message = stream.get_final_message()
        except Exception as exc:
            # If get_final_message itself fails (extremely unlikely after
            # successful iteration), surface as failure rather than
            # fabricate a partial ChatResponse.
            yield StreamFailed(
                ProviderError(
                    f"Failed to assemble final response: {exc}",
                    raw=str(exc),
                )
            )
            return

        # Same content-parsing logic the legacy complete() used. We
        # iterate every block in final_message.content and pick out
        # text. For a normal text response there's one ParsedTextBlock
        # (verified Probe 1). For code-execution responses there are
        # also tool_use blocks (Claude writing code) and tool_result
        # blocks (the code's output) interleaved with text. The
        # user-visible answer is the union of all text blocks joined
        # with blank lines. The hasattr check tolerates non-text blocks
        # without crashing.
        text_parts: list[str] = []
        for block in final_message.content or []:
            if hasattr(block, "text") and block.text:
                text_parts.append(block.text)
        text = "\n\n".join(text_parts)

        usage = final_message.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        # Cache stats for diagnostic visibility. cache_creation_input_tokens
        # is what we paid the 25% surcharge to write; cache_read_input_tokens
        # is what we got at 10% on this call. Together they tell us whether
        # the cache is actually being hit on repeat calls.
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        dbg("CLIENT", f"anthropic usage: input={input_tokens} output={output_tokens} cache_write={cache_creation} cache_read={cache_read}")
        cost = calculate_cost_usd(request.model, input_tokens, output_tokens)

        # Emit the Usage event before StreamCompleted. The UI updates
        # its token-count label on this event. We could merge this
        # into StreamCompleted, but separate events keep the protocol
        # clean and let providers that emit usage mid-stream — none do
        # today, but in v0.2.x some might — fit naturally.
        dbg("CLIENT", f"anthropic: yielding Usage(in={input_tokens}, out={output_tokens})")
        yield Usage(input_tokens=input_tokens, output_tokens=output_tokens)

        latency = time.monotonic() - start

        final_response = ChatResponse(
            text=text,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            served_model=final_message.model or request.model.api_model_name,
        )
        dbg("CLIENT", "anthropic: yielding StreamCompleted, complete_stream END")
        yield StreamCompleted(final_response=final_response)

    # ---------- Internal helpers ----------

    def _build_user_content(self, request: ChatRequest) -> list[dict]:
        """Build the user message content as a list of typed blocks.

        For each Anthropic file_ref, the block shape depends on the MIME:
          - Image MIMEs (image/png, image/jpeg, ...) → 'image' block
          - PDF MIME → 'document' block
          - Office MIMEs (xlsx/docx/pptx/...) → 'container_upload' block
            preceded by a metadata-instruction text block

        Block ordering, with examples for the common cases:

          PDF only:
            [{document}, {text: user prompt}]

          Image only:
            [{image}, {text: user prompt}]

          xlsx only (anonymous bytes):
            [{text: instruction-with-metadata}, {container_upload},
             {text: user prompt}]

          Mixed (PDF + image + xlsx):
            [{document}, {image}, {text: instruction-with-metadata},
             {container_upload}, {text: user prompt}]

        Files come BEFORE the prompt (with metadata-instruction sandwiched
        between them when needed). Empirically, models follow context
        better when files appear ahead of the question being asked.

        Files whose MIME doesn't match any known bucket are silently
        skipped — they shouldn't reach this client because the routing
        layer would have marked the (anthropic, mime) combination as
        UNSUPPORTED — but defensive handling means a misrouted file
        causes a degraded response, not a 400 error.

        Prompt caching (v0.2.0):
        The LAST file-related block in the message gets a cache_control
        marker with ttl="1h". Anthropic's caching uses prefix matching:
        marking the last block of the cacheable prefix tells Anthropic
        "everything up to and including this point is the cacheable
        prefix." The user's prompt text after this is NOT cached
        (changes per query). On the first call within a session, the
        cached portion bills at 125% of normal (the write surcharge);
        on subsequent calls within 1 hour with identical file content,
        the cached portion bills at 10% of normal. Massive reduction
        in input-token consumption when a user asks multiple questions
        about the same file(s) in a session.
        """
        anthropic_refs = [r for r in request.file_refs if r.provider == "anthropic"]

        image_refs = [r for r in anthropic_refs if r.mime_type in _IMAGE_MIMES]
        document_refs = [r for r in anthropic_refs if r.mime_type in _DOCUMENT_MIMES]
        office_refs = [r for r in anthropic_refs if r.mime_type in _OFFICE_MIMES]

        blocks: list[dict] = []

        # 1. Document blocks first (PDFs).
        for ref in document_refs:
            blocks.append({
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": ref.remote_id,
                },
            })

        # 2. Image blocks next.
        for ref in image_refs:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": ref.remote_id,
                },
            })

        # 3. If any office files are attached, prepend a metadata
        # instruction so Claude knows what each anonymous-bytes upload
        # actually is. This is a single text block describing all of them
        # together — keeps the prompt compact even with multiple files.
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

        # 4. Mark the LAST file-related block with cache_control. Anthropic
        # caches the prefix up to and including this marker. We add the
        # marker only if at least one file block was added (text-only
        # prompts have nothing worth caching). The user's prompt text
        # added in step 5 is OUTSIDE the cached prefix.
        if blocks:
            blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}

        # 5. The user's actual prompt comes last — NOT cached, since it
        # changes per query.
        blocks.append({"type": "text", "text": request.prompt})
        return blocks


def _build_office_metadata_text(office_refs: list[FileRef]) -> str:
    """Produce the metadata-instruction text Claude sees before the
    container_upload blocks.

    The text tells Claude:
      - That the file(s) were uploaded as anonymous bytes (so the
        in-container filename is 'blob.bin', not the real name).
      - What each file actually is (real filename + format).
      - Which Python library to use to open it.
      - The standard cp-rename trick to satisfy openpyxl's extension
        checks (so Claude doesn't have to rediscover this each time).

    Single block for any number of office files — keeps token cost low
    when one file is attached and stays compact when several are.
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

    # Multi-file metadata. Lists each file with its real name and library
    # hint. Less common but supported.
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