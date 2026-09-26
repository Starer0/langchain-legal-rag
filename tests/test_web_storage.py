import tempfile
import unittest
import sqlite3
from pathlib import Path

from web_storage import SQLiteConversationStore


class SQLiteConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "conversations.sqlite3"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def insert_v81_messages(self, session_id, messages):
        connection = sqlite3.connect(self.path)
        try:
            connection.executescript("""
                CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT, last_seen_at TEXT);
                CREATE TABLE messages (session_id TEXT, ordinal INTEGER, role TEXT, content TEXT, created_at TEXT,
                    PRIMARY KEY (session_id, ordinal));
            """)
            connection.execute("INSERT INTO sessions VALUES (?, 'now', 'now')", (session_id,))
            connection.executemany(
                "INSERT INTO messages VALUES (?, ?, ?, ?, 'now')",
                [(session_id, index, role, content) for index, (role, content) in enumerate(messages, 1)],
            )
            connection.commit()
        finally:
            connection.close()

    def test_sessions_are_isolated_and_survive_store_recreation(self):
        store = SQLiteConversationStore(self.path)
        first = store.ensure_session(None)
        second = store.ensure_session(None)
        store.save_complete_turn(first, "试用期工资？", "应符合第二十条。")

        reopened = SQLiteConversationStore(self.path)

        self.assertEqual(reopened.load_messages(first), [
            ("user", "试用期工资？"),
            ("assistant", "应符合第二十条。"),
        ])
        self.assertEqual(reopened.load_messages(second), [])

    def test_empty_or_partial_turn_is_never_persisted(self):
        store = SQLiteConversationStore(self.path)
        session_id = store.ensure_session(None)

        with self.assertRaisesRegex(ValueError, "不能为空"):
            store.save_complete_turn(session_id, "问题", "")

        self.assertEqual(store.load_messages(session_id), [])

    def test_legacy_session_messages_migrate_once_into_one_conversation(self):
        self.insert_v81_messages("legacy", [("user", "旧问题"), ("assistant", "旧回答")])

        store = SQLiteConversationStore(self.path)
        conversations = store.list_conversations("legacy")

        self.assertEqual(len(conversations), 1)
        self.assertEqual(conversations[0].title, "已迁移的对话")
        self.assertEqual(store.load_conversation_messages("legacy", conversations[0].id), [
            ("user", "旧问题"), ("assistant", "旧回答"),
        ])
        self.assertEqual(store.list_conversations("legacy"), conversations)

    def test_conversation_crud_titles_and_delete_are_session_scoped(self):
        store = SQLiteConversationStore(self.path)
        first = store.ensure_session(None)
        other = store.ensure_session(None)
        conversation = store.create_conversation(first)

        updated = store.save_complete_conversation_turn(
            first, conversation.id, "试用期最长多久？", "最长六个月。"
        )

        self.assertEqual(updated.title, "试用期最长多久？")
        self.assertEqual(store.rename_conversation(first, conversation.id, "试用期咨询").title, "试用期咨询")
        store.save_complete_conversation_turn(first, conversation.id, "那工资呢？", "按规定支付。")
        self.assertEqual(store.list_conversations(other), [])
        self.assertFalse(store.delete_conversation(other, conversation.id))
        self.assertTrue(store.delete_conversation(first, conversation.id))
        self.assertEqual(store.list_conversations(first), [])


if __name__ == "__main__":
    unittest.main()
