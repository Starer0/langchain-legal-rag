"""SQLite persistence for anonymous web-chat sessions."""

import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class SQLiteConversationStore:
    """Store completed user/assistant turns for one browser session."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection():
            pass

    def ensure_session(self, session_id: str | None) -> str:
        session_id = session_id or secrets.token_urlsafe(32)
        now = _utc_now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions(id, created_at, last_seen_at)
                VALUES (?, ?, ?)
                """,
                (session_id, now, now),
            )
            connection.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
                (now, session_id),
            )
        return session_id

    def load_messages(self, session_id: str) -> list[tuple[str, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY ordinal
                """,
                (session_id,),
            ).fetchall()
        return [(role, content) for role, content in rows]

    def save_complete_turn(self, session_id: str, question: str, answer: str) -> None:
        if not question.strip() or not answer.strip():
            raise ValueError("问题和回答不能为空")

        now = _utc_now()
        with self._connection() as connection:
            next_ordinal = connection.execute(
                """
                SELECT COALESCE(MAX(ordinal), 0) + 1
                FROM messages
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()[0]
            connection.executemany(
                """
                INSERT INTO messages(session_id, ordinal, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (session_id, next_ordinal, "user", question, now),
                    (session_id, next_ordinal + 1, "assistant", answer, now),
                ],
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                session_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, ordinal),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            """
        )
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
