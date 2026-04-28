"""
FileLibrary — high-level operations for managing the user's file library.

The sidebar UI talks to FileLibrary, not directly to the registry or
uploaders. FileLibrary handles the per-provider fan-out for upload and
delete, surfaces partial failures cleanly, and presents a clean
file-centric API that doesn't leak provider details.

Operations:
    add_file(path)          — upload to all 4 providers, register in DB
    delete_file(file_id)    — delete from all 4 providers + DB
    list_files()            — return all registered files for the sidebar
    get_refs(file_ids, p)   — get cached refs for the given files at provider p
"""
from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from attachments.registry import FileRecord, FileRegistry
from attachments.uploaders import (
    AnthropicUploader,
    AzureOpenAIUploader,
    BaseUploader,
    GeminiUploader,
    OpenAIUploader,
    ProviderUploadResult,
)
from providers.base import FileRef
from settings_manager import SettingsManager


# Map our internal provider key to the corresponding uploader class.
_UPLOADER_CLASSES: dict[str, type[BaseUploader]] = {
    "anthropic": AnthropicUploader,
    "gemini": GeminiUploader,
    "openai": OpenAIUploader,
    "azure_openai": AzureOpenAIUploader,
}


@dataclass(frozen=True)
class LibraryFile:
    """A file in the library, ready for sidebar display."""
    file_id: int                       # SQLite primary key
    filename: str
    size_bytes: int
    mime_type: str
    providers_uploaded: list[str]      # which providers have this file


@dataclass(frozen=True)
class UploadOutcome:
    """Result of attempting to upload one file to all providers."""
    file_record: FileRecord
    successful_providers: list[str]    # providers where upload succeeded
    failed_providers: dict[str, str]   # provider name -> error message


@dataclass(frozen=True)
class DeleteOutcome:
    """Result of attempting to delete one file from all providers."""
    file_id: int
    successful_providers: list[str]
    failed_providers: dict[str, str]


class FileLibrary:
    """High-level file management for HECTOR-AI."""

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

    # ---------- Public operations ----------

    def add_file(self, path: Path) -> UploadOutcome:
        """Register a new file and upload it to all configured providers.

        Returns an UploadOutcome reporting which providers succeeded and
        which failed. Even if ALL providers fail, the file is still
        registered locally so we can retry uploads later.
        """
        path = Path(path).resolve()
        mime_type = self._guess_mime_type(path)

        # Step 1: register the file locally (computes hash, dedups).
        file_record = self._registry.register_file(path, mime_type=mime_type)

        # Step 2: upload to each configured provider in turn.
        # We do these sequentially for now — could parallelize with
        # threads later if upload latency becomes a concern.
        successes: list[str] = []
        failures: dict[str, str] = {}

        for provider_name, uploader in self._uploaders.items():
            if not uploader.is_configured():
                failures[provider_name] = "Provider not configured"
                continue

            # Skip if already cached for this provider (rare — happens
            # if user re-adds a file we previously uploaded).
            cached = self._registry.get_provider_ref(file_record.id, provider_name)
            if cached is not None:
                successes.append(provider_name)
                continue

            try:
                result: ProviderUploadResult = uploader.upload(path, mime_type)
                self._registry.set_provider_ref(
                    file_id=file_record.id,
                    provider=provider_name,
                    remote_id=result.remote_id,
                    expires_at=result.expires_at,
                )
                successes.append(provider_name)
            except Exception as exc:
                failures[provider_name] = str(exc)

        return UploadOutcome(
            file_record=file_record,
            successful_providers=successes,
            failed_providers=failures,
        )

    def delete_file(self, file_id: int) -> DeleteOutcome:
        """Delete a file from all providers and from the local registry.

        Attempts deletion on every provider that has a cached ref. Even
        if some providers fail, removes successful ones from the registry.
        Returns an outcome detailing which succeeded and which failed.
        """
        successes: list[str] = []
        failures: dict[str, str] = {}

        # Get the list of providers that have this file uploaded.
        existing_refs = self._registry.list_provider_refs(file_id)

        for ref in existing_refs:
            uploader = self._uploaders.get(ref.provider)
            if uploader is None:
                failures[ref.provider] = "No uploader available"
                continue

            if not uploader.is_configured():
                failures[ref.provider] = "Provider not configured"
                continue

            try:
                uploader.delete(ref.remote_id)
                successes.append(ref.provider)
            except Exception as exc:
                failures[ref.provider] = str(exc)

        # If all providers succeeded (or had no refs), delete the local
        # file record entirely (cascade removes provider_file_refs rows).
        if not failures:
            self._registry.delete_file(file_id)

        return DeleteOutcome(
            file_id=file_id,
            successful_providers=successes,
            failed_providers=failures,
        )

    def list_files(self) -> list[LibraryFile]:
        """Return all files in the library, with provider upload status.

        Used by the sidebar to display the file list. Filters out files
        that have NO valid provider refs (they're effectively orphaned
        and shouldn't appear).
        """
        result: list[LibraryFile] = []

        for record in self._all_file_records():
            refs = self._registry.list_provider_refs(record.id)
            if not refs:
                continue  # No valid refs — skip this orphan
            providers_uploaded = [r.provider for r in refs]
            result.append(LibraryFile(
                file_id=record.id,
                filename=record.filename,
                size_bytes=record.size_bytes,
                mime_type=record.mime_type,
                providers_uploaded=providers_uploaded,
            ))

        return result

    def get_refs_for_provider(
        self,
        file_ids: list[int],
        provider: str,
    ) -> list[FileRef]:
        """Get FileRef objects for the given files at the given provider.

        Used at Run time — sidebar tells us "user selected files [2, 5, 7]",
        we return the provider-specific refs the chat client needs. Files
        that don't have a ref for this provider are silently skipped (so
        a partially-uploaded file still works on the providers it reached).
        """
        refs: list[FileRef] = []

        for file_id in file_ids:
            cached = self._registry.get_provider_ref(file_id, provider)
            if cached is None:
                continue

            # Look up the file record to populate filename/mime
            record = self._get_file_record(file_id)
            if record is None:
                continue

            refs.append(FileRef(
                provider=provider,
                remote_id=cached.remote_id,
                filename=record.filename,
                mime_type=record.mime_type,
            ))

        return refs

    # ---------- Internal helpers ----------

    def _all_file_records(self) -> list[FileRecord]:
        """Return all FileRecord rows from the registry, newest first."""
        rows = self._registry._conn.execute(
            "SELECT * FROM files ORDER BY created_at DESC"
        ).fetchall()
        return [self._registry._row_to_file_record(row) for row in rows]

    def _get_file_record(self, file_id: int) -> FileRecord | None:
        """Look up a single FileRecord by id."""
        row = self._registry._conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,),
        ).fetchone()
        return self._registry._row_to_file_record(row) if row else None

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        """Best-effort MIME type detection from filename suffix."""
        guess, _ = mimetypes.guess_type(str(path))
        if guess:
            return guess
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return "application/pdf"
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            return f"image/{suffix.lstrip('.')}"
        return "application/octet-stream"