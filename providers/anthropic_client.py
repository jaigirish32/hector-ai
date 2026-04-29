"""
Anthropic (Claude) client — calls api.anthropic.com via the official
`anthropic` Python SDK.

Differences from our OpenAI client:
- Endpoint method is `messages.create`, not `chat.completions.create`.
- System prompt is a TOP-LEVEL parameter, not a message in the list.
- Response content is a list of blocks; we read text from the first block.
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
    FileRef,
    NotConfiguredError,
    ProviderError,
    RateLimitError,
    calculate_cost_usd,
)
from settings_manager import SecretKey, SettingsManager


# Anthropic requires this beta header to accept content blocks that
# reference uploaded file_ids (image, document, container_upload).
ANTHROPIC_FILES_BETA = "files-api-2025-04-14"

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
        # For code-execution responses there are also server_tool_use blocks
        # (Claude writing code) and bash_code_execution_tool_result blocks
        # (the code's output) interleaved with text. We collect every
        # text block and concatenate them — the user-visible answer is
        # spread across explanation-of-intent text + final-summary text
        # with tool blocks between.
        text_parts: list[str] = []
        for block in response.content or []:
            if hasattr(block, "text") and block.text:
                text_parts.append(block.text)
        text = "\n\n".join(text_parts)

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

        # 4. The user's actual prompt comes last.
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