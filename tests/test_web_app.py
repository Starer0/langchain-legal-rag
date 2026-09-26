import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from web_app import create_app
from web_storage import SQLiteConversationStore


def parse_events(response):
    events = []
    for frame in response.text.strip().split("\n\n"):
        event = None
        data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            if line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if event:
            events.append((event, data))
    return events


class SuccessfulTurn:
    def stream(self, question, messages):
        yield {"event": "status", "data": {"stage": "rewrite"}}
        yield {"event": "delta", "data": {"text": "完整"}}
        yield {
            "event": "done",
            "data": {
                "answer": "完整回答",
                "retrieval_question": question,
                "sources": [],
                "performance": {},
            },
        }


class FailingTurn:
    def stream(self, question, messages):
        raise RuntimeError("provider key leaked")
        yield  # pragma: no cover


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = SQLiteConversationStore(
            Path(self.temporary_directory.name) / "web.sqlite3"
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_two_clients_receive_different_persistent_histories(self):
        app = create_app(self.store, SuccessfulTurn())
        first = TestClient(app)
        second = TestClient(app)

        conversation = first.post("/api/conversations").json()["id"]
        response = first.post(f"/api/conversations/{conversation}/chat", json={"question": "试用期工资？"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(first.get(f"/api/conversations/{conversation}/messages").json()["messages"], [
            {"role": "user", "content": "试用期工资？"},
            {"role": "assistant", "content": "完整"},
        ])
        self.assertEqual(second.get("/api/conversations").json()["conversations"], [])

    def test_successful_sse_persists_the_complete_delta_answer(self):
        app = create_app(self.store, SuccessfulTurn())
        client = TestClient(app)

        conversation = client.post("/api/conversations").json()["id"]
        response = client.post(f"/api/conversations/{conversation}/chat", json={"question": "问题"})

        self.assertEqual(parse_events(response), [
            ("status", {"stage": "rewrite"}),
            ("delta", {"text": "完整"}),
            ("done", {
                "answer": "完整回答",
                "retrieval_question": "问题",
                "sources": [],
                "performance": {},
            }),
        ])
        self.assertIn("legal_rag_session", client.cookies)
        self.assertEqual(client.get(f"/api/conversations/{conversation}/messages").json()["messages"][-1], {
            "role": "assistant", "content": "完整",
        })

    def test_stream_error_emits_safe_error_and_saves_no_turn(self):
        app = create_app(self.store, FailingTurn())
        client = TestClient(app)

        conversation = client.post("/api/conversations").json()["id"]
        events = parse_events(client.post(f"/api/conversations/{conversation}/chat", json={"question": "问题"}))

        self.assertEqual(events[-1], (
            "error", {"message": "暂时无法完成回答，请稍后重试。"}
        ))
        self.assertEqual(client.get(f"/api/conversations/{conversation}/messages").json()["messages"], [])

    def test_conversation_lifecycle_is_isolated_by_browser_session(self):
        app = create_app(self.store, SuccessfulTurn())
        first = TestClient(app)
        second = TestClient(app)

        conversation = first.post("/api/conversations").json()
        renamed = first.patch(
            f"/api/conversations/{conversation['id']}",
            json={"title": "试用期咨询"},
        )

        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(renamed.json()["title"], "试用期咨询")
        self.assertEqual(second.get(f"/api/conversations/{conversation['id']}/messages").status_code, 404)
        self.assertEqual(first.delete(f"/api/conversations/{conversation['id']}").status_code, 204)
        self.assertEqual(first.get("/api/conversations").json()["conversations"], [])


class WebRuntimeTests(unittest.TestCase):
    @patch("web_app.create_web_rag_turn")
    def test_default_app_constructs_one_shared_rag_turn_at_startup(self, create_turn):
        import web_app

        with tempfile.TemporaryDirectory() as directory:
            app = web_app.create_default_app(
                database_path=Path(directory) / "web.sqlite3",
                secure_cookies=False,
            )
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/conversations").status_code, 200)

        create_turn.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
