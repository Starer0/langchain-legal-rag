import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from web_app import create_app
from web_storage import SQLiteConversationStore


class EmptyTurn:
    def stream(self, question, messages):
        return iter(())


class WebAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        store = SQLiteConversationStore(
            Path(self.temporary_directory.name) / "web.sqlite3"
        )
        self.client = TestClient(create_app(store, EmptyTurn()))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_root_serves_chat_page_and_session_cookie(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="chat-form"', response.text)
        self.assertIn("legal_rag_session", response.headers["set-cookie"])

    def test_static_assets_are_served(self):
        self.assertIn("#chat-form", self.client.get("/static/styles.css").text)
        self.assertIn("/api/chat", self.client.get("/static/app.js").text)

    def test_markdown_module_is_served_as_javascript(self):
        response = self.client.get("/static/markdown.mjs")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/javascript")


if __name__ == "__main__":
    unittest.main()
