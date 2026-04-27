"""
File orchestrator — resolves local file paths into per-provider remote
references, using the registry for caching.

Flow per (file_path, provider) pair:

    1. Compute SHA-256 hash of the file
    2. Look up file in registry by hash
       - If new → register it
    3. Check registry for cached provider_file_ref
       - If present and not expired → reuse remote_id
       - If absent or expired → upload via uploader, cache result, return new remote_id

The orchestrator owns one FileRegistry instance and one uploader per
provider. It's safe to call from worker threads (the registry is
thread-safe; the uploaders create their own clients per call).
"""
from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from attachments.registry import FileRegistry
from attachments.uploaders import (
    AnthropicUploader,
    AzureOpenAIUploader,
    BaseUploader,
    GeminiUploader,
    OpenAIUploader,
    ProviderUploadResult,
)
from providers.base import FileRef, NotConfiguredError, ProviderError
from settings_manager import SettingsManager


# Map our internal provider name to the corresponding uploader class.
# When the dispatcher needs file refs for provider X, the orchestrator
# uses _UPLOADERS[X] to do the upload.
_UPLOADER_CLASSES: dict[str, type[BaseUploader]] = {
    "anthropic": AnthropicUploader,
    "gemini": GeminiUploader,
    "openai": OpenAIUploader,
    "azure_openai": AzureOpenAIUploader,
}


@dataclass(frozen=True)
class FileResolutionError:
    """Reports a failure to resolve one file for one provider.

    The dispatcher displays these as per-card errors (e.g., 'Anthropic:
    upload failed') without breaking the other providers.
    """
    file_path: Path
    provider: str
    message: str


class FileOrchestrator:
    """Turns file paths into per-provider FileRefs, using cached uploads
    where possible."""

    def __init__(
        self,
        registry: FileRegistry | None = None,
        settings: SettingsManager | None = None,
    ) -> None:
        self._registry = registry or FileRegistry()
        self._settings = settings or SettingsManager()
        self._uploaders: dict[str, BaseUploader] = {
            name: cls(settings=self._settings)
            for name, cls in _UPLOADER_CLASSES.items()
        }

    def resolve_for_provider(
        self,
        file_paths: list[Path],
        provider: str,
    ) -> tuple[list[FileRef], list[FileResolutionError]]:
        """Resolve all files into remote_ids for one specific provider.

        Returns (refs, errors). On success, errors is empty. Partial
        success is possible — some files may resolve while others fail;
        the caller decides how to surface that to the user.
        """
        if provider not in self._uploaders:
            return [], [
                FileResolutionError(
                    file_path=p,
                    provider=provider,
                    message=f"Unknown provider: {provider}",
                )
                for p in file_paths
            ]

        refs: list[FileRef] = []
        errors: list[FileResolutionError] = []
        uploader = self._uploaders[provider]

        for path in file_paths:
            try:
                ref = self._resolve_one(path, provider, uploader)
                refs.append(ref)
            except NotConfiguredError as exc:
                errors.append(FileResolutionError(
                    file_path=path,
                    provider=provider,
                    message=str(exc),
                ))
            except ProviderError as exc:
                errors.append(FileResolutionError(
                    file_path=path,
                    provider=provider,
                    message=str(exc),
                ))
            except Exception as exc:
                errors.append(FileResolutionError(
                    file_path=path,
                    provider=provider,
                    message=f"Unexpected error: {exc}",
                ))

        return refs, errors

    # ---------- Internal ----------

    def _resolve_one(
        self,
        path: Path,
        provider: str,
        uploader: BaseUploader,
    ) -> FileRef:
        """Resolve one file for one provider. Upload if not cached."""
        path = Path(path).resolve()
        if not path.exists():
            raise ProviderError(f"File not found: {path}")

        mime_type = self._guess_mime_type(path)

        # Step 1: ensure file is registered (computes hash, dedups)
        file_record = self._registry.register_file(path, mime_type=mime_type)

        # Step 2: check for cached, valid provider ref
        cached = self._registry.get_provider_ref(file_record.id, provider)
        if cached is not None:
            return FileRef(
                provider=provider,
                remote_id=cached.remote_id,
                filename=file_record.filename,
                mime_type=file_record.mime_type,
            )

        # Step 3: cache miss — upload now
        result: ProviderUploadResult = uploader.upload(path, mime_type)

        # Step 4: store the result so future calls hit the cache
        self._registry.set_provider_ref(
            file_id=file_record.id,
            provider=provider,
            remote_id=result.remote_id,
            expires_at=result.expires_at,
        )

        return FileRef(
            provider=provider,
            remote_id=result.remote_id,
            filename=file_record.filename,
            mime_type=file_record.mime_type,
        )

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        """Best-effort MIME type detection from filename suffix."""
        guess, _ = mimetypes.guess_type(str(path))
        if guess:
            return guess
        # Fallback: most attachments will be one of these
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "application/pdf"
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            return f"image/{suffix.lstrip('.')}"
        return "application/octet-stream"