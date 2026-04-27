"""
Per-provider file uploaders.

Each uploader has the same interface:
    upload(file_path: Path, mime_type: str) -> ProviderUploadResult

The dispatcher picks the right uploader based on which provider it's
calling. This file only handles the upload mechanics — caching of
upload results in SQLite is the dispatcher's job (Phase 2c).

Phase 2b.1: Anthropic only. Other providers added in subsequent steps.
"""
from __future__ import annotations

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

from providers.base import (
    AuthenticationError,
    NotConfiguredError,
    ProviderError,
    RateLimitError,
)
from settings_manager import SecretKey, SettingsManager

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

import time

from openai import (
    OpenAI,
    APIConnectionError as OpenAIConnectionError,
    APIError as OpenAIAPIError,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimitError,
)

from openai import (
    AzureOpenAI,
    OpenAI,
    APIConnectionError as OpenAIConnectionError,
    APIError as OpenAIAPIError,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimitError,
)

# ---------- Result type ----------

@dataclass(frozen=True)
class ProviderUploadResult:
    """Outcome of uploading a single file to a single provider."""
    provider: str          # 'anthropic', 'openai', etc.
    remote_id: str         # the file_id (or URI for Gemini) returned
    expires_at: datetime | None  # None = indefinite
    raw_filename: str      # what the provider stored as the filename
    size_bytes: int        # what the provider reports as the size


# ---------- Base class ----------

class BaseUploader:
    """Abstract base for provider uploaders."""

    provider_name: str = ""

    def is_configured(self) -> bool:
        """Whether this provider has the credentials needed to upload."""
        raise NotImplementedError

    def upload(self, file_path: Path, mime_type: str) -> ProviderUploadResult:
        """Upload a file to this provider, return remote_id + metadata."""
        raise NotImplementedError


# ---------- Anthropic uploader ----------

# The beta header value Anthropic requires for the Files API.
# This is a fixed string published by Anthropic; bumped only when they
# change the API contract.
ANTHROPIC_FILES_BETA = "files-api-2025-04-14"


class AnthropicUploader(BaseUploader):
    """Uploads files to Anthropic's Files API."""

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

        # Anthropic's SDK accepts file as a (filename, file_obj, mime_type)
        # tuple. The filename here is what Anthropic will store internally.
        try:
            with open(path, "rb") as fh:
                response = client.beta.files.upload(
                    file=(path.name, fh, mime_type),
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

        # Anthropic files don't expire automatically — they persist until
        # explicitly deleted. So expires_at is None.
        return ProviderUploadResult(
            provider=self.provider_name,
            remote_id=response.id,
            expires_at=None,
            raw_filename=getattr(response, "filename", path.name),
            size_bytes=getattr(response, "size_bytes", path.stat().st_size),
        )
    

# ---------- Gemini uploader ----------

# Gemini files start in PROCESSING state and become ACTIVE when ready.
# For PDFs/images this is usually instant; for videos it can take seconds.
# We poll until ACTIVE or this many seconds pass before giving up.
GEMINI_PROCESSING_TIMEOUT_SECONDS = 60
GEMINI_POLL_INTERVAL_SECONDS = 1.0


class GeminiUploader(BaseUploader):
    """Uploads files to Google's Gemini Files API.

    Gemini auto-deletes files after 48 hours; we record the exact expiry
    timestamp returned by Google so the registry can re-upload when
    needed.
    """

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

        # The SDK accepts a file path string and a config dict.
        # mime_type goes inside the config.
        try:
            uploaded = client.files.upload(
                file=str(path),
                config={"mime_type": mime_type},
            )
        except genai_errors.APIError as exc:
            self._translate_error(exc)
            # _translate_error raises; this line never reached
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected error uploading to Gemini: {exc}",
                raw=str(exc),
            ) from exc

        # If Gemini returns the file in PROCESSING state, poll until ACTIVE
        # or we timeout. PDFs/images are usually ACTIVE immediately, so this
        # loop usually exits on the first check.
        uploaded = self._wait_for_active(client, uploaded)

        # Read the expiration timestamp Google set on the file.
        # The SDK returns it as a tz-aware datetime already.
        expires_at = getattr(uploaded, "expiration_time", None)

        # Gemini needs the full URI (not just the resource name) when referencing
        # the file later via Part.from_uri(file_uri=...). Both are returned by
        # upload — we store the URI because that's what generate_content expects.
        return ProviderUploadResult(
            provider=self.provider_name,
            remote_id=uploaded.uri,  # 'https://generativelanguage.googleapis.com/v1beta/files/abc-xyz'
            expires_at=expires_at,
            raw_filename=getattr(uploaded, "display_name", path.name) or path.name,
            size_bytes=int(getattr(uploaded, "size_bytes", 0) or path.stat().st_size),
        )

    # ---------- Internals ----------

    def _wait_for_active(self, client: "genai.Client", file_obj) -> object:
        """Poll until the uploaded file's state is ACTIVE.

        Returns the refreshed file object. Raises ProviderError if the
        file fails processing or doesn't become active within the timeout.
        """
        state = getattr(file_obj, "state", None)
        # State can be returned as either a string ('ACTIVE') or as an
        # enum-like value with a .name attribute. Normalize.
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
        """Normalize Gemini's state field to a string."""
        if state is None:
            return ""
        # Some SDK versions return enum, others return string.
        return getattr(state, "name", None) or str(state)

    @staticmethod
    def _translate_error(exc: "genai_errors.APIError") -> None:
        """Translate Gemini SDK errors into our domain errors. Always raises."""
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        message = getattr(exc, "message", str(exc))

        if status in (401, 403):
            raise AuthenticationError(
                "Google rejected the API key during upload.",
                raw=str(exc),
            ) from exc
        if status == 429:
            raise RateLimitError(
                "Gemini rate limit hit during upload. Wait and retry.",
                raw=str(exc),
            ) from exc
        raise ProviderError(
            f"Gemini upload error: {message}",
            raw=str(exc),
        ) from exc
    

# ---------- OpenAI uploader ----------

# When uploading files for the Responses API, OpenAI requires this purpose.
# Other valid purposes (assistants, fine-tune, vision, batch) cause the
# file to be rejected when referenced from a Responses API call.
OPENAI_FILE_PURPOSE = "user_data"


class OpenAIUploader(BaseUploader):
    """Uploads files to OpenAI's Files API for use with the Responses API.

    Files persist indefinitely (no automatic expiry); they remain on
    OpenAI's servers until explicitly deleted.
    """

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

        try:
            with open(path, "rb") as fh:
                response = client.files.create(
                    file=fh,
                    purpose=OPENAI_FILE_PURPOSE,
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

        # OpenAI files don't auto-expire — they persist until deleted.
        return ProviderUploadResult(
            provider=self.provider_name,
            remote_id=response.id,
            expires_at=None,
            raw_filename=getattr(response, "filename", path.name) or path.name,
            size_bytes=int(getattr(response, "bytes", 0) or path.stat().st_size),
        )
    
# ---------- Azure OpenAI uploader ----------

# Azure's Files API requires this api-version or later for Responses API
# files. Same version we use for the chat client.
AZURE_FILES_API_VERSION = "2025-03-01-preview"

# Azure rejects `user_data` (OpenAI's standard purpose for Responses API
# files). Verified working: 'assistants' purpose uploads successfully and
# CAN be referenced from Responses API on Azure. This is undocumented in
# the public Azure Responses API guide but works on api-version
# 2025-03-01-preview.
AZURE_FILE_PURPOSE = "assistants"


class AzureOpenAIUploader(BaseUploader):
    """Uploads files to Azure OpenAI's Files API for use with Responses.

    Files persist indefinitely. They are scoped to the specific Azure
    resource — file_ids returned here only work with this Azure resource.
    """

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
                "Azure OpenAI not fully configured. Go to Settings and add "
                "your API key and endpoint URL."
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