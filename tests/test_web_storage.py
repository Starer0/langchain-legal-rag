import tempfile
import unittest
from pathlib import Path

from web_storage import SQLiteConversationStore


class SQLiteConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "conversations.sqlite3"

    def tearDown(self):
        self.temporary_directory.cleanup()

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


if __name__ == "__main__":
    unittest.main()
