"""SQLite persistence for anonymous web-chat conversations."""

import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    updated_at: str


class SQLiteConversationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection():
            pass

    def ensure_session(self, session_id):
        session_id = session_id or secrets.token_urlsafe(32)
        now = _utc_now()
        with self._connection() as connection:
            connection.execute("INSERT OR IGNORE INTO sessions VALUES (?, ?, ?)", (session_id, now, now))
            connection.execute("UPDATE sessions SET last_seen_at = ? WHERE id = ?", (now, session_id))
        return session_id

    def list_conversations(self, session_id):
        with self._connection() as connection:
            rows = connection.execute("SELECT id, title, updated_at FROM conversations WHERE session_id = ? ORDER BY updated_at DESC", (session_id,)).fetchall()
        return [Conversation(*row) for row in rows]

    def create_conversation(self, session_id):
        now = _utc_now()
        conversation = Conversation(secrets.token_urlsafe(24), "新对话", now)
        with self._connection() as connection:
            connection.execute("INSERT INTO conversations VALUES (?, ?, ?, 0, ?, ?)", (conversation.id, session_id, conversation.title, now, now))
        return conversation

    def conversation_belongs_to(self, session_id, conversation_id):
        with self._connection() as connection:
            return connection.execute("SELECT 1 FROM conversations WHERE id = ? AND session_id = ?", (conversation_id, session_id)).fetchone() is not None

    def rename_conversation(self, session_id, conversation_id, title):
        title = title.strip()
        if not title or len(title) > 80:
            raise ValueError("标题长度必须为 1 到 80 个字符")
        now = _utc_now()
        with self._connection() as connection:
            cursor = connection.execute("UPDATE conversations SET title = ?, title_is_custom = 1, updated_at = ? WHERE id = ? AND session_id = ?", (title, now, conversation_id, session_id))
            if cursor.rowcount != 1:
                return None
        return Conversation(conversation_id, title, now)

    def delete_conversation(self, session_id, conversation_id):
        with self._connection() as connection:
            return connection.execute("DELETE FROM conversations WHERE id = ? AND session_id = ?", (conversation_id, session_id)).rowcount == 1

    def load_conversation_messages(self, session_id, conversation_id):
        if not self.conversation_belongs_to(session_id, conversation_id):
            return None
        with self._connection() as connection:
            rows = connection.execute("SELECT role, content FROM conversation_messages WHERE conversation_id = ? ORDER BY ordinal", (conversation_id,)).fetchall()
        return rows

    def save_complete_conversation_turn(self, session_id, conversation_id, question, answer):
        if not question.strip() or not answer.strip():
            raise ValueError("问题和回答不能为空")
        now = _utc_now()
        with self._connection() as connection:
            row = connection.execute("SELECT title, title_is_custom FROM conversations WHERE id = ? AND session_id = ?", (conversation_id, session_id)).fetchone()
            if row is None:
                return None
            ordinal = connection.execute("SELECT COALESCE(MAX(ordinal), 0) + 1 FROM conversation_messages WHERE conversation_id = ?", (conversation_id,)).fetchone()[0]
            connection.executemany("INSERT INTO conversation_messages VALUES (?, ?, ?, ?, ?)", [(conversation_id, ordinal, "user", question, now), (conversation_id, ordinal + 1, "assistant", answer, now)])
            title = row[0]
            if not row[1] and title == "新对话":
                title = _title_from_question(question)
                connection.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, now, conversation_id))
            else:
                connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
        return Conversation(conversation_id, title, now)

    # V8.1 compatibility while the API migration is implemented.
    def load_messages(self, session_id):
        with self._connection() as connection:
            return connection.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY ordinal", (session_id,)).fetchall()

    def save_complete_turn(self, session_id, question, answer):
        if not question.strip() or not answer.strip(): raise ValueError("问题和回答不能为空")
        with self._connection() as connection:
            ordinal = connection.execute("SELECT COALESCE(MAX(ordinal), 0) + 1 FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
            now = _utc_now()
            connection.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", [(session_id, ordinal, "user", question, now), (session_id, ordinal + 1, "assistant", answer, now)])

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (session_id TEXT NOT NULL, ordinal INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(session_id, ordinal));
            CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), title TEXT NOT NULL, title_is_custom INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS conversation_messages (conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(conversation_id, ordinal));
        """)
        if not connection.execute("SELECT 1 FROM schema_migrations WHERE name = 'v8_to_v82_conversations'").fetchone():
            for session_id, in connection.execute("SELECT DISTINCT session_id FROM messages").fetchall():
                conversation_id, now = secrets.token_urlsafe(24), _utc_now()
                connection.execute("INSERT INTO conversations VALUES (?, ?, '已迁移的对话', 0, ?, ?)", (conversation_id, session_id, now, now))
                rows = connection.execute("SELECT ordinal, role, content, created_at FROM messages WHERE session_id = ?", (session_id,)).fetchall()
                connection.executemany("INSERT INTO conversation_messages VALUES (?, ?, ?, ?, ?)", [(conversation_id, *row) for row in rows])
            connection.execute("INSERT INTO schema_migrations VALUES ('v8_to_v82_conversations')")
        try:
            yield connection; connection.commit()
        except Exception:
            connection.rollback(); raise
        finally:
            connection.close()


def _utc_now(): return datetime.now(timezone.utc).isoformat()
def _title_from_question(question):
    title = " ".join(question.split())
    return title[:24] + ("…" if len(title) > 24 else "")
