"""
FileLibrary — high-level operations for managing the user's file library.

The sidebar UI talks to FileLibrary, not directly to the registry or
uploaders. FileLibrary handles the per-provider fan-out for upload and
delete, surfaces partial failures cleanly, and presents a clean
file-centric API that doesn't leak provider details.

Operations:
    add_file(path)          — upload to providers that natively support
                              the file type, register in DB
    delete_file(file_id)    — delete from all providers + DB
    list_files()            — return all registered files for the sidebar
    get_refs(file_ids, p)   — get cached refs for the given files at provider p

Native-only architecture (Phase 2e):
    No preprocessing. The file as-uploaded is what goes to providers.
    Before uploading, we consult the routing layer to learn which providers
    natively support the file type. Providers that don't are skipped — not
    failed — so the UI can distinguish "no native path exists" from
    "tried and the network/auth blew up."
"""
from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from attachments.registry import FileRecord, FileRegistry
from attachments.uploaders import (
    AnthropicUploader,
    BaseUploader,
    GeminiUploader,
    OpenAIUploader,
    ProviderUploadResult,
    XAIUploader,
)
from providers.base import FileRef
from routing.router import RoutingPlan, route
from settings_manager import SettingsManager


# Map our internal provider key to the corresponding uploader class.
# These keys MUST match the provider names in routing/capability_matrix.py.
_UPLOADER_CLASSES: dict[str, type[BaseUploader]] = {
    "anthropic": AnthropicUploader,
    "gemini": GeminiUploader,
    "openai": OpenAIUploader,
    "xai": XAIUploader,
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
    """Result of attempting to upload one file to all providers.

    Three categories, deliberately kept separate so the UI can render
    them differently:
      successful — file is at the provider, ready for chat
      failed     — we tried to upload but it errored (network/auth/etc).
                   May resolve on retry.
      skipped    — we didn't try because the provider does not natively
                   support this file type. Retry won't change anything;
                   the user needs a different provider for this file.
    """
    file_record: FileRecord
    successful_providers: list[str]
    failed_providers: dict[str, str]
    skipped_providers: dict[str, str] = field(default_factory=dict)


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
        """Register a new file and upload it to providers that natively
        support its type.

        Steps:
          1. Compute the file's mime type and size.
          2. Ask the router which providers natively support it.
          3. Register the file locally (so we have a record even if all
             uploads fail or no provider supports it).
          4. Upload to each supported provider; record outcomes.

        Providers that don't natively support the file type appear in
        skipped_providers, not failed_providers — they're a different
        category from "we tried and it errored."
        """
        path = Path(path).resolve()
        size_bytes = path.stat().st_size
        mime_type = self._guess_mime_type(path)

        # Step 1: route. The router answers, per provider, whether this
        # (mime, size) combination is supported. We hand it the full set
        # of providers we know about — UI-level "which chips did the user
        # pick" is a separate concern that lives in the dispatcher, not here.
        plan: RoutingPlan = route(
            mime=mime_type,
            file_size_bytes=size_bytes,
            selected_providers=list(_UPLOADER_CLASSES.keys()),
        )

        supported_providers = {sp.name for sp in plan.supported}
        skipped: dict[str, str] = {
            sk.name: sk.detail or sk.reason.value
            for sk in plan.skipped
        }

        # Step 2: register locally with the file's true mime type.
        # We register even if no provider supports it — keeps the file
        # visible in the library so the user can see why it's unusable.
        file_record = self._registry.register_file(path, mime_type=mime_type)

        # Step 3: upload only to supported providers.
        successes: list[str] = []
        failures: dict[str, str] = {}

        self._upload_to_providers(
            file_record=file_record,
            source_path=path,
            source_mime=mime_type,
            allowed_providers=supported_providers,
            successes=successes,
            failures=failures,
        )

        return UploadOutcome(
            file_record=file_record,
            successful_providers=successes,
            failed_providers=failures,
            skipped_providers=skipped,
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
                msg = str(exc)
                # 403 (no permission to this file) and 404 (not found)
                # are permanent — retry won't help. Treat them as "the
                # file is no longer ours to delete" and let the local
                # registry drop the row. The provider's copy is
                # effectively orphaned, but that's already true.
                permanent = (
                    "PERMISSION_DENIED" in msg
                    or "permission to access" in msg
                    or "not found" in msg.lower()
                    or "404" in msg
                )
                if permanent:
                    successes.append(ref.provider)
                else:
                    failures[ref.provider] = msg

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

        Used by the sidebar to display the file list. Returns ALL registered
        files including those with no successful provider uploads — the
        sidebar can render them with an "unsupported" indicator. Filtering
        out orphans is the caller's choice, not ours.
        """
        result: list[LibraryFile] = []

        for record in self._all_file_records():
            refs = self._registry.list_provider_refs(record.id)
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

    def _upload_to_providers(
        self,
        *,
        file_record: FileRecord,
        source_path: Path,
        source_mime: str,
        allowed_providers: set[str],
        successes: list[str],
        failures: dict[str, str],
    ) -> None:
        """Upload source_path to every configured AND allowed provider.

        allowed_providers is the router's verdict: only these providers
        natively support this file type. Others were already recorded
        as skipped by the caller and aren't retried here.

        Skips providers that already have a cached ref for this file_id.
        Mutates the passed-in successes/failures collections rather than
        returning them, since the caller owns the aggregation.
        """
        for provider_name, uploader in self._uploaders.items():
            if provider_name not in allowed_providers:
                continue  # router said no — not our concern here

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
                result: ProviderUploadResult = uploader.upload(source_path, source_mime)
                self._registry.set_provider_ref(
                    file_id=file_record.id,
                    provider=provider_name,
                    remote_id=result.remote_id,
                    expires_at=result.expires_at,
                )
                successes.append(provider_name)
            except Exception as exc:
                failures[provider_name] = str(exc)

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
        if suffix == ".xlsx":
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        if suffix == ".csv":
            return "text/csv"
        if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            ext = suffix.lstrip(".")
            return f"image/{'jpeg' if ext == 'jpg' else ext}"
        return "application/octet-stream"
