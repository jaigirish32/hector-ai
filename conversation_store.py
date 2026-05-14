"""
Per-model conversation history store for HECTOR-AI.

Persists conversation turns in hector.db alongside the existing
files/provider_file_refs tables. Each turn is one user prompt +
one assistant response for a specific model.

History cap: HISTORY_CAP (default 20). When a new turn is added and
the model's count exceeds the cap, the oldest turn is deleted silently.
Cap is hardcoded for now — configurable via Settings in a future release.

Database location: same hector.db used by FileRegistry. Path resolved
at construction time via paths.user_data_dir() — Qt must be initialized
before constructing this class.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paths import user_data_dir


_DB_FILENAME = "hector.db"

# Maximum number of turns stored per model. When exceeded, oldest is dropped.
HISTORY_CAP = 20


@dataclass(frozen=True)
class HistoryTurn:
    """One complete turn: user prompt + assistant response for one model."""
    id: int
    model_id: str
    user_content: str
    assistant_content: str
    created_at: datetime


class ConversationStore:
    """SQLite-backed per-model conversation history.

    Open one instance per app session. Safe for worker-thread reads
    (history loaded before dispatch) and main-thread writes (turn saved
    after StreamCompleted). SQLite serializes access internally.

    Usage:
        store = ConversationStore()
        history = store.get_history("gpt-5.5")   # load before dispatch
        store.add_turn("gpt-5.5", prompt, response_text)  # save after complete
        store.clear_all()   # user clicked Clear History
        store.close()       # app shutdown
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = user_data_dir() / _DB_FILENAME

        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,   # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._closed = False
        self._initialize_schema()

    # ---------- Schema ----------

    def _initialize_schema(self) -> None:
        """Add conversation_history table if it doesn't exist. Idempotent."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id    TEXT    NOT NULL,
                user_content      TEXT NOT NULL,
                assistant_content TEXT NOT NULL,
                created_at  TEXT    NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_conv_history_model
                ON conversation_history(model_id, id);
        """)

    # ---------- Public API ----------

    def get_history(self, model_id: str) -> list[HistoryTurn]:
        """Return all stored turns for this model, oldest first.

        Returns at most HISTORY_CAP turns (the cap is enforced on write,
        so this will normally return <= HISTORY_CAP rows).
        """
        rows = self._conn.execute(
            """
            SELECT id, model_id, user_content, assistant_content, created_at
            FROM conversation_history
            WHERE model_id = ?
            ORDER BY id ASC
            """,
            (model_id,),
        ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    def add_turn(
        self,
        model_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Persist one turn and enforce the HISTORY_CAP.

        If storing this turn pushes the model's count above HISTORY_CAP,
        the oldest turn for that model is deleted silently.
        """
        self._conn.execute(
            """
            INSERT INTO conversation_history
                (model_id, user_content, assistant_content)
            VALUES (?, ?, ?)
            """,
            (model_id, user_content, assistant_content),
        )
        self._prune(model_id)

    def clear_all(self) -> None:
        """Delete all conversation history for all models."""
        self._conn.execute("DELETE FROM conversation_history")

    def clear_model(self, model_id: str) -> None:
        """Delete all conversation history for one model."""
        self._conn.execute(
            "DELETE FROM conversation_history WHERE model_id = ?",
            (model_id,),
        )

    # ---------- Lifecycle ----------

    def close(self) -> None:
        """Close the database connection. Idempotent — safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._conn.close()

    # ---------- Internal ----------

    def _prune(self, model_id: str) -> None:
        """Delete the oldest turn if count exceeds HISTORY_CAP."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM conversation_history WHERE model_id = ?",
            (model_id,),
        ).fetchone()[0]

        if count > HISTORY_CAP:
            # Delete the single oldest row for this model.
            self._conn.execute(
                """
                DELETE FROM conversation_history
                WHERE id = (
                    SELECT id FROM conversation_history
                    WHERE model_id = ?
                    ORDER BY id ASC
                    LIMIT 1
                )
                """,
                (model_id,),
            )

    def _row_to_turn(self, row: sqlite3.Row) -> HistoryTurn:
        return HistoryTurn(
            id=row["id"],
            model_id=row["model_id"],
            user_content=row["user_content"],
            assistant_content=row["assistant_content"],
            created_at=_parse_iso_datetime(row["created_at"]),
        )


# ---------- Helpers ----------

def _parse_iso_datetime(text: str) -> datetime:
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
