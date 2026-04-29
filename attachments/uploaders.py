"""
Per-provider file uploaders.

Each uploader has the same interface:
    upload(file_path: Path, mime_type: str) -> ProviderUploadResult
    delete(remote_id: str) -> None

The dispatcher picks the right uploader based on which provider it's
calling. This file only handles the upload/delete mechanics — caching
of upload results in SQLite is the file orchestrator's job.

Per-provider quirks handled in this module:

  Anthropic + office formats (xlsx/xls/docx/doc/pptx/ppt):
      Uploaded as anonymous bytes (filename 'blob.bin', MIME
      'application/octet-stream'). Real filename/MIME tracked in
      HECTOR's registry and surfaced to Claude as text metadata at
      chat time (see providers/anthropic_client.py). PDFs and images
      keep their real filename/MIME because the document/image content
      blocks need real types. Empirically verified Apr 2026.

  OpenAI + image formats (PNG, JPEG):
      Uploaded with purpose='vision' instead of the 'user_data' default
      used for documents. Per OpenAI docs, images referenced in chat as
      input_image content blocks must have been uploaded with the vision
      purpose; user_data uploads cannot be referenced via input_image and
      input_file rejects image extensions. Documents continue to use
      'user_data'.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    AuthenticationError as AnthropicAuthError,
    RateLimitError as AnthropicRateLimitError,
)
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from openai import (
    AzureOpenAI,
    OpenAI,
    APIConnectionError as OpenAIConnectionError,
    APIError as OpenAIAPIError,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimitError,
)

from providers.base import (
    AuthenticationError,
    NotConfiguredError,
    ProviderError,
    RateLimitError,
)
from settings_manager import SecretKey, SettingsManager


# ---------- Result type ----------

@dataclass(frozen=True)
class ProviderUploadResult:
    """Outcome of uploading a single file to a single provider."""
    provider: str
    remote_id: str
    expires_at: datetime | None
    raw_filename: str
    size_bytes: int


# ---------- Base class ----------

class BaseUploader:
    """Abstract base for provider uploaders."""

    provider_name: str = ""

    def is_configured(self) -> bool:
        raise NotImplementedError

    def upload(self, file_path: Path, mime_type: str) -> ProviderUploadResult:
        raise NotImplementedError

    def delete(self, remote_id: str) -> None:
        """Delete a file from this provider's servers.

        Raises ProviderError on failure. Idempotent: deleting an already-
        deleted file should not raise (provider returns 404, we swallow).
        """
        raise NotImplementedError


# ---------- Anthropic uploader ----------

ANTHROPIC_FILES_BETA = "files-api-2025-04-14"

# Office formats that Anthropic only accepts via code execution (not via
# the standard 'document' content block). These get uploaded as anonymous
# bytes — filename 'blob.bin', MIME 'application/octet-stream' — because
# this is the path empirically verified to work with container_upload +
# code execution. MUST stay in sync with _OFFICE_MIMES in
# providers/anthropic_client.py — they describe the same set from two
# different sides (upload vs chat).
_ANTHROPIC_OFFICE_MIMES = frozenset({
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",   # xlsx
    "application/vnd.ms-excel",                                            # xls
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # docx
    "application/msword",                                                  # doc
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # pptx
    "application/vnd.ms-powerpoint",                                       # ppt
})

_ANTHROPIC_ANON_FILENAME = "blob.bin"
_ANTHROPIC_ANON_MIME = "application/octet-stream"


class AnthropicUploader(BaseUploader):
    provider_name = "anthropic"

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.ANTHROPIC_API_KEY)

    def upload(self, file_path: Path, mime_type: str) -> ProviderUploadResult:
        if not self.is_configured():
            raise NotConfiguredError(
                "Anthropic API key not set. Go to Settings to add it."
            )

        api_key = self._settings.get_secret(SecretKey.ANTHROPIC_API_KEY)
        client = Anthropic(api_key=api_key)

        path = Path(file_path).resolve()
        if not path.exists():
            raise ProviderError(f"File not found: {path}")

        # Decide wire format. Office types go up as anonymous bytes; PDFs
        # and images keep their real filename and MIME. The decision is
        # made here (in the uploader) rather than at the call site because
        # it's a provider-specific quirk, not a HECTOR-level concept.
        if mime_type in _ANTHROPIC_OFFICE_MIMES:
            wire_filename = _ANTHROPIC_ANON_FILENAME
            wire_mime = _ANTHROPIC_ANON_MIME
        else:
            wire_filename = path.name
            wire_mime = mime_type

        try:
            with open(path, "rb") as fh:
                response = client.beta.files.upload(
                    file=(wire_filename, fh, wire_mime),
                    extra_headers={"anthropic-beta": ANTHROPIC_FILES_BETA},
                )
        except AnthropicAuthError as exc:
            raise AuthenticationError(
                "Anthropic rejected the API key during upload.",
                raw=str(exc),
            ) from exc
        except AnthropicRateLimitError as exc:
            raise RateLimitError(
                "Anthropic rate limit hit during upload. Wait and retry.",
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
                f"Anthropic upload error: {message}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error uploading to Anthropic: {exc}",
                raw=str(exc),
            ) from exc

        # Always store the REAL filename and size in the result, even
        # when we sent anonymous bytes. The registry uses these for the
        # user-facing file name and for the metadata prompt at chat time.
        return ProviderUploadResult(
            provider=self.provider_name,
            remote_id=response.id,
            expires_at=None,
            raw_filename=path.name,
            size_bytes=int(getattr(response, "size_bytes", 0) or path.stat().st_size),
        )

    def delete(self, remote_id: str) -> None:
        if not self.is_configured():
            raise NotConfiguredError(
                "Anthropic API key not set."
            )

        api_key = self._settings.get_secret(SecretKey.ANTHROPIC_API_KEY)
        client = Anthropic(api_key=api_key)

        try:
            client.beta.files.delete(
                file_id=remote_id,
                extra_headers={"anthropic-beta": ANTHROPIC_FILES_BETA},
            )
        except AnthropicAuthError as exc:
            raise AuthenticationError(
                "Anthropic rejected the API key during delete.",
                raw=str(exc),
            ) from exc
        except APIError as exc:
            # 404 means the file is already gone — treat as success.
            status = getattr(exc, "status_code", None)
            if status == 404:
                return
            raise ProviderError(
                f"Anthropic delete error: {getattr(exc, 'message', str(exc))}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error deleting from Anthropic: {exc}",
                raw=str(exc),
            ) from exc


# ---------- Gemini uploader ----------

GEMINI_PROCESSING_TIMEOUT_SECONDS = 60
GEMINI_POLL_INTERVAL_SECONDS = 1.0


class GeminiUploader(BaseUploader):
    provider_name = "gemini"

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.GOOGLE_API_KEY)

    def upload(self, file_path: Path, mime_type: str) -> ProviderUploadResult:
        if not self.is_configured():
            raise NotConfiguredError(
                "Google API key not set. Go to Settings to add it."
            )

        api_key = self._settings.get_secret(SecretKey.GOOGLE_API_KEY)
        client = genai.Client(api_key=api_key)

        path = Path(file_path).resolve()
        if not path.exists():
            raise ProviderError(f"File not found: {path}")

        try:
            uploaded = client.files.upload(
                file=str(path),
                config={"mime_type": mime_type},
            )
        except genai_errors.APIError as exc:
            self._translate_error(exc)
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error uploading to Gemini: {exc}",
                raw=str(exc),
            ) from exc

        uploaded = self._wait_for_active(client, uploaded)
        expires_at = getattr(uploaded, "expiration_time", None)

        return ProviderUploadResult(
            provider=self.provider_name,
            remote_id=uploaded.uri,
            expires_at=expires_at,
            raw_filename=getattr(uploaded, "display_name", path.name) or path.name,
            size_bytes=int(getattr(uploaded, "size_bytes", 0) or path.stat().st_size),
        )

    def delete(self, remote_id: str) -> None:
        if not self.is_configured():
            raise NotConfiguredError("Google API key not set.")

        api_key = self._settings.get_secret(SecretKey.GOOGLE_API_KEY)
        client = genai.Client(api_key=api_key)

        # Gemini's delete API expects the resource name (e.g. 'files/abc-xyz'),
        # not the full URI. Extract the name from the URI.
        # remote_id format: 'https://generativelanguage.googleapis.com/v1beta/files/abc-xyz'
        if "/files/" in remote_id:
            name = "files/" + remote_id.split("/files/")[-1]
        else:
            name = remote_id

        try:
            client.files.delete(name=name)
        except genai_errors.APIError as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if status == 404:
                return  # Already gone
            self._translate_error(exc)
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error deleting from Gemini: {exc}",
                raw=str(exc),
            ) from exc

    # ---------- Internals ----------

    def _wait_for_active(self, client: "genai.Client", file_obj) -> object:
        state = getattr(file_obj, "state", None)
        state_str = self._state_string(state)

        if state_str == "ACTIVE":
            return file_obj

        deadline = time.monotonic() + GEMINI_PROCESSING_TIMEOUT_SECONDS

        while True:
            if state_str == "FAILED":
                raise ProviderError(
                    f"Gemini failed to process the uploaded file: {file_obj.name}"
                )
            if state_str == "ACTIVE":
                return file_obj
            if time.monotonic() > deadline:
                raise ProviderError(
                    f"Gemini file still processing after "
                    f"{GEMINI_PROCESSING_TIMEOUT_SECONDS}s: {file_obj.name}"
                )

            time.sleep(GEMINI_POLL_INTERVAL_SECONDS)
            try:
                file_obj = client.files.get(name=file_obj.name)
            except genai_errors.APIError as exc:
                self._translate_error(exc)
                raise
            state_str = self._state_string(getattr(file_obj, "state", None))

    @staticmethod
    def _state_string(state) -> str:
        if state is None:
            return ""
        return getattr(state, "name", None) or str(state)

    @staticmethod
    def _translate_error(exc: "genai_errors.APIError") -> None:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        message = getattr(exc, "message", str(exc))

        if status in (401, 403):
            raise AuthenticationError(
                "Google rejected the API key.",
                raw=str(exc),
            ) from exc
        if status == 429:
            raise RateLimitError(
                "Gemini rate limit hit. Wait and retry.",
                raw=str(exc),
            ) from exc
        raise ProviderError(
            f"Gemini error: {message}",
            raw=str(exc),
        ) from exc


# ---------- OpenAI uploader ----------

# Default purpose for documents (PDFs, csvs, code, text, spreadsheets).
# OpenAI's Responses API references these via input_file content blocks.
_OPENAI_DOCUMENT_PURPOSE = "user_data"

# Purpose for images (PNG, JPEG, GIF, WEBP). OpenAI's Responses API
# references these via input_image content blocks. Files uploaded with
# 'user_data' purpose CANNOT be used as input_image — the file_id is
# rejected at chat time. Conversely, files uploaded with 'vision' purpose
# cannot be used as input_file. The two purposes are not interchangeable.
_OPENAI_VISION_PURPOSE = "vision"

# MIME types that should be uploaded with the vision purpose. Per OpenAI
# vision docs (Apr 2026): PNG, JPEG, GIF, WEBP. HECTOR's matrix today
# routes only PNG and JPEG, but the set here is wider so future matrix
# rows for GIF/WEBP work without code changes.
_OPENAI_IMAGE_MIMES = frozenset({
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
})


def _openai_purpose_for(mime_type: str) -> str:
    """Return the OpenAI Files API purpose to use for a given MIME type.

    Images need 'vision'; everything else uses 'user_data'. The choice is
    permanent for the file: OpenAI files cannot be re-purposed after
    upload, so we have to get this right the first time.
    """
    if mime_type in _OPENAI_IMAGE_MIMES:
        return _OPENAI_VISION_PURPOSE
    return _OPENAI_DOCUMENT_PURPOSE


class OpenAIUploader(BaseUploader):
    provider_name = "openai"

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    def is_configured(self) -> bool:
        return self._settings.has_secret(SecretKey.OPENAI_API_KEY)

    def upload(self, file_path: Path, mime_type: str) -> ProviderUploadResult:
        if not self.is_configured():
            raise NotConfiguredError(
                "OpenAI API key not set. Go to Settings to add it."
            )

        api_key = self._settings.get_secret(SecretKey.OPENAI_API_KEY)
        client = OpenAI(api_key=api_key)

        path = Path(file_path).resolve()
        if not path.exists():
            raise ProviderError(f"File not found: {path}")

        # Pick the correct purpose for OpenAI based on file type. Images
        # MUST go up as 'vision'; documents MUST go up as 'user_data'.
        # The right choice depends on how the chat client will reference
        # the file later — input_image vs input_file content blocks.
        purpose = _openai_purpose_for(mime_type)

        try:
            with open(path, "rb") as fh:
                response = client.files.create(
                    file=fh,
                    purpose=purpose,
                )
        except OpenAIAuthError as exc:
            raise AuthenticationError(
                "OpenAI rejected the API key during upload.",
                raw=str(exc),
            ) from exc
        except OpenAIRateLimitError as exc:
            raise RateLimitError(
                "OpenAI rate limit hit during upload. Wait and retry.",
                raw=str(exc),
            ) from exc
        except OpenAIConnectionError as exc:
            raise ProviderError(
                "Could not reach OpenAI — check your internet connection.",
                raw=str(exc),
            ) from exc
        except OpenAIAPIError as exc:
            message = getattr(exc, "message", str(exc))
            raise ProviderError(
                f"OpenAI upload error: {message}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error uploading to OpenAI: {exc}",
                raw=str(exc),
            ) from exc

        return ProviderUploadResult(
            provider=self.provider_name,
            remote_id=response.id,
            expires_at=None,
            raw_filename=getattr(response, "filename", path.name) or path.name,
            size_bytes=int(getattr(response, "bytes", 0) or path.stat().st_size),
        )

    def delete(self, remote_id: str) -> None:
        if not self.is_configured():
            raise NotConfiguredError("OpenAI API key not set.")

        api_key = self._settings.get_secret(SecretKey.OPENAI_API_KEY)
        client = OpenAI(api_key=api_key)

        try:
            client.files.delete(remote_id)
        except OpenAIAuthError as exc:
            raise AuthenticationError(
                "OpenAI rejected the API key during delete.",
                raw=str(exc),
            ) from exc
        except OpenAIAPIError as exc:
            status = getattr(exc, "status_code", None)
            if status == 404:
                return
            raise ProviderError(
                f"OpenAI delete error: {getattr(exc, 'message', str(exc))}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error deleting from OpenAI: {exc}",
                raw=str(exc),
            ) from exc


# ---------- Azure OpenAI uploader ----------

AZURE_FILES_API_VERSION = "2025-03-01-preview"
AZURE_FILE_PURPOSE = "assistants"


class AzureOpenAIUploader(BaseUploader):
    provider_name = "azure_openai"

    def __init__(self, settings: SettingsManager | None = None) -> None:
        self._settings = settings or SettingsManager()

    def is_configured(self) -> bool:
        return (
            self._settings.has_secret(SecretKey.AZURE_OPENAI_API_KEY)
            and self._settings.has_secret(SecretKey.AZURE_OPENAI_ENDPOINT)
        )

    def upload(self, file_path: Path, mime_type: str) -> ProviderUploadResult:
        if not self.is_configured():
            raise NotConfiguredError(
                "Azure OpenAI not fully configured. Go to Settings."
            )

        api_key = self._settings.get_secret(SecretKey.AZURE_OPENAI_API_KEY)
        endpoint = self._settings.get_secret(SecretKey.AZURE_OPENAI_ENDPOINT)

        client = AzureOpenAI(
            api_key=api_key,
            api_version=AZURE_FILES_API_VERSION,
            azure_endpoint=endpoint,
        )

        path = Path(file_path).resolve()
        if not path.exists():
            raise ProviderError(f"File not found: {path}")

        try:
            with open(path, "rb") as fh:
                response = client.files.create(
                    file=fh,
                    purpose=AZURE_FILE_PURPOSE,
                )
        except OpenAIAuthError as exc:
            raise AuthenticationError(
                "Azure rejected the API key during upload.",
                raw=str(exc),
            ) from exc
        except OpenAIRateLimitError as exc:
            raise RateLimitError(
                "Azure rate limit hit during upload. Wait and retry.",
                raw=str(exc),
            ) from exc
        except OpenAIConnectionError as exc:
            raise ProviderError(
                "Could not reach Azure OpenAI — check your endpoint URL "
                "and internet connection.",
                raw=str(exc),
            ) from exc
        except OpenAIAPIError as exc:
            message = getattr(exc, "message", str(exc))
            raise ProviderError(
                f"Azure OpenAI upload error: {message}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error uploading to Azure OpenAI: {exc}",
                raw=str(exc),
            ) from exc

        return ProviderUploadResult(
            provider=self.provider_name,
            remote_id=response.id,
            expires_at=None,
            raw_filename=getattr(response, "filename", path.name) or path.name,
            size_bytes=int(getattr(response, "bytes", 0) or path.stat().st_size),
        )

    def delete(self, remote_id: str) -> None:
        if not self.is_configured():
            raise NotConfiguredError("Azure OpenAI not fully configured.")

        api_key = self._settings.get_secret(SecretKey.AZURE_OPENAI_API_KEY)
        endpoint = self._settings.get_secret(SecretKey.AZURE_OPENAI_ENDPOINT)

        client = AzureOpenAI(
            api_key=api_key,
            api_version=AZURE_FILES_API_VERSION,
            azure_endpoint=endpoint,
        )

        try:
            client.files.delete(remote_id)
        except OpenAIAuthError as exc:
            raise AuthenticationError(
                "Azure rejected the API key during delete.",
                raw=str(exc),
            ) from exc
        except OpenAIAPIError as exc:
            status = getattr(exc, "status_code", None)
            if status == 404:
                return
            raise ProviderError(
                f"Azure delete error: {getattr(exc, 'message', str(exc))}",
                raw=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error deleting from Azure: {exc}",
                raw=str(exc),
            ) from exc