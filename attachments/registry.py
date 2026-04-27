"""
SQLite-backed registry for files attached to HECTOR-AI.

Tracks two things:

1. Files we've seen (deduplicated by SHA-256 hash of bytes).
2. Per-provider remote_id references (with expiry tracking).

The registry is local-only — never makes network calls. It's a cache
that lets the dispatcher answer "have we already uploaded this file
to this provider, and is the upload still valid?"

Database location: `<project_root>/hector.db` for now. Will move to
`%APPDATA%\\HECTOR-AI\\hector.db` when we package for distribution.
"""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# Database lives in the project root for development. Switch to
# user app-data folder when packaging.
DB_PATH = Path(__file__).resolve().parent.parent / "hector.db"


@dataclass(frozen=True)
class FileRecord:
    """A file we've seen, identified by its content hash."""
    id: int
    file_hash: str
    filename: str
    size_bytes: int
    mime_type: str
    created_at: datetime


@dataclass(frozen=True)
class ProviderFileRef:
    """A remote file reference at one provider for one file."""
    file_id: int        # foreign key to FileRecord.id
    provider: str       # 'openai', 'azure', 'anthropic', 'gemini'
    remote_id: str      # the file_id returned by the provider
    expires_at: datetime | None  # None means indefinite
    uploaded_at: datetime


class FileRegistry:
    """Local cache for file uploads.

    Open one instance per app session. The connection is owned by the
    registry; methods are thread-safe via SQLite's serialized mode.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        # check_same_thread=False: we'll access from worker threads.
        # SQLite serializes access internally as long as no transaction
        # is held across thread boundaries (which we never do).
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit transactions for writes
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    # ---------- Schema management ----------

    def _initialize_schema(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash);

            CREATE TABLE IF NOT EXISTS provider_file_refs (
                file_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                remote_id TEXT NOT NULL,
                expires_at TEXT,
                uploaded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY (file_id, provider),
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_provider_refs_lookup
                ON provider_file_refs(file_id, provider);
        """)

    # ---------- File operations ----------

    def hash_file(self, path: Path) -> str:
        """Compute SHA-256 of a file's bytes. Used for dedup."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            # Read in 64KB chunks to avoid loading huge files into memory.
            for chunk in iter(lambda: f.read(64 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def register_file(
        self,
        path: Path,
        mime_type: str,
    ) -> FileRecord:
        """Add a file to the registry, or return the existing record if
        we've seen this file before (by hash).

        Returns the FileRecord either way.
        """
        path = Path(path).resolve()
        file_hash = self.hash_file(path)
        size_bytes = path.stat().st_size
        filename = path.name

        # Check if we already have this file.
        existing = self._conn.execute(
            "SELECT * FROM files WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()

        if existing:
            return self._row_to_file_record(existing)

        # Insert new file.
        cursor = self._conn.execute(
            """
            INSERT INTO files (file_hash, filename, size_bytes, mime_type)
            VALUES (?, ?, ?, ?)
            """,
            (file_hash, filename, size_bytes, mime_type),
        )
        new_id = cursor.lastrowid

        # Read it back to get the full record with timestamps.
        row = self._conn.execute(
            "SELECT * FROM files WHERE id = ?",
            (new_id,),
        ).fetchone()
        return self._row_to_file_record(row)

    def get_file_by_hash(self, file_hash: str) -> FileRecord | None:
        """Look up a file by its SHA-256 hash. None if not found."""
        row = self._conn.execute(
            "SELECT * FROM files WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        return self._row_to_file_record(row) if row else None

    def delete_file(self, file_id: int) -> None:
        """Remove a file and all its provider references.

        Does NOT delete the file from provider servers — this is purely
        local cache cleanup. Provider files expire on their own schedules.
        """
        self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

    # ---------- Provider reference operations ----------

    def get_provider_ref(
        self,
        file_id: int,
        provider: str,
    ) -> ProviderFileRef | None:
        """Look up the cached remote_id for a file at a provider.

        Returns None if:
        - We never uploaded this file to this provider, OR
        - The cached upload has expired (Gemini's 48h window)

        Caller should re-upload and call set_provider_ref() in either case.
        """
        row = self._conn.execute(
            """
            SELECT * FROM provider_file_refs
            WHERE file_id = ? AND provider = ?
            """,
            (file_id, provider),
        ).fetchone()

        if row is None:
            return None

        ref = self._row_to_provider_ref(row)

        # Check expiry. If past, treat as missing.
        if ref.expires_at is not None and ref.expires_at < datetime.now(timezone.utc):
            # Optional: we could delete the expired row here. We don't,
            # because set_provider_ref will overwrite via UPSERT anyway.
            return None

        return ref

    def set_provider_ref(
        self,
        file_id: int,
        provider: str,
        remote_id: str,
        expires_at: datetime | None = None,
    ) -> None:
        """Record that we uploaded a file to a provider, with optional expiry.

        Upserts: overwrites any previous reference for the same (file_id,
        provider) pair. This lets us handle re-uploads after expiry cleanly.
        """
        expires_iso = expires_at.isoformat() if expires_at else None

        self._conn.execute(
            """
            INSERT INTO provider_file_refs
                (file_id, provider, remote_id, expires_at, uploaded_at)
            VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT (file_id, provider) DO UPDATE SET
                remote_id = excluded.remote_id,
                expires_at = excluded.expires_at,
                uploaded_at = excluded.uploaded_at
            """,
            (file_id, provider, remote_id, expires_iso),
        )

    def list_provider_refs(self, file_id: int) -> list[ProviderFileRef]:
        """All non-expired provider refs for a given file."""
        rows = self._conn.execute(
            "SELECT * FROM provider_file_refs WHERE file_id = ?",
            (file_id,),
        ).fetchall()

        now = datetime.now(timezone.utc)
        results = []
        for row in rows:
            ref = self._row_to_provider_ref(row)
            if ref.expires_at is None or ref.expires_at > now:
                results.append(ref)
        return results

    # ---------- Lifecycle ----------

    def close(self) -> None:
        """Close the database connection. Call on app shutdown."""
        self._conn.close()

    # ---------- Internal row → dataclass conversion ----------

    def _row_to_file_record(self, row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            id=row["id"],
            file_hash=row["file_hash"],
            filename=row["filename"],
            size_bytes=row["size_bytes"],
            mime_type=row["mime_type"],
            created_at=_parse_iso_datetime(row["created_at"]),
        )

    def _row_to_provider_ref(self, row: sqlite3.Row) -> ProviderFileRef:
        return ProviderFileRef(
            file_id=row["file_id"],
            provider=row["provider"],
            remote_id=row["remote_id"],
            expires_at=(
                _parse_iso_datetime(row["expires_at"])
                if row["expires_at"]
                else None
            ),
            uploaded_at=_parse_iso_datetime(row["uploaded_at"]),
        )


# ---------- Helpers ----------

def _parse_iso_datetime(text: str) -> datetime:
    """Parse SQLite's ISO8601 string into a tz-aware datetime."""
    # SQLite stores '2026-04-27T12:34:56.000Z'. Convert to datetime.
    # Python <3.11 doesn't accept 'Z' suffix in fromisoformat.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    # Ensure timezone-aware (defensive).
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt