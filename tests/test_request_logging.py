import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from web_app import create_app
from web_storage import SQLiteConversationStore


class Turn:
    def __init__(self, mode="success"):
        self.mode = mode

    def stream(self, question, messages):
        yield {"event": "delta", "data": {"text": "回答"}}
        if self.mode == "raise":
            raise RuntimeError("provider unavailable")
        if self.mode == "error":
            yield {"event": "error", "data": {"message": "服务暂时不可用"}}
        elif self.mode != "empty":
            yield {"event": "done", "data": {"answer": "回答", "sources": []}}


class RequestLoggingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteConversationStore(Path(self.directory.name) / "web.sqlite3")

    def chat(self, mode="success"):
        client = TestClient(create_app(self.store, Turn(mode)))
        conversation = client.post("/api/conversations").json()["id"]
        with self.assertLogs("legal_rag.requests", level="INFO") as captured:
            response = client.post(f"/api/conversations/{conversation}/chat", json={"question": "问题"})
        return client, conversation, response, [json.loads(r.getMessage()) for r in captured.records]

    def test_success_is_correlated_and_logged_after_answer_is_saved(self):
        client, conversation, response, rows = self.chat()
        request_id = response.headers.get("X-Request-ID")
        self.assertTrue(request_id)
        self.assertEqual([row["event"] for row in rows], ["request_started", "request_finished"])
        self.assertTrue(all(row["request_id"] == request_id for row in rows))
        self.assertEqual(rows[-1]["outcome"], "completed")
        self.assertGreaterEqual(rows[-1]["duration_ms"], 0)
        self.assertEqual(client.get(f"/api/conversations/{conversation}/messages").json()["messages"][-1]["content"], "回答")

    def test_stream_failures_are_not_http_200_successes(self):
        for mode in ("raise", "error", "empty"):
            with self.subTest(mode=mode):
                client, conversation, response, rows = self.chat(mode)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(rows[-1]["outcome"], "failed")
                frames = response.text.strip().split("\n\n")
                payload = json.loads(frames[-1].split("data: ")[1])
                self.assertEqual(payload["request_id"], response.headers["X-Request-ID"])
                self.assertNotIn("provider unavailable", response.text)
                if mode == "raise":
                    self.assertIn("provider unavailable", rows[-1]["error"])
                self.assertEqual(client.get(f"/api/conversations/{conversation}/messages").json()["messages"], [])

    def test_validation_and_foreign_conversation_rejections_have_unique_ids(self):
        owner = TestClient(create_app(self.store, Turn()))
        conversation = owner.post("/api/conversations").json()["id"]
        other = TestClient(create_app(self.store, Turn()))
        with self.assertLogs("legal_rag.requests", level="INFO") as captured:
            foreign = other.post(f"/api/conversations/{conversation}/chat", json={"question": "问题"}, headers={"X-Request-ID": "forged"})
            invalid = owner.post(f"/api/conversations/{conversation}/chat", json={"question": " "})
        self.assertEqual((foreign.status_code, invalid.status_code), (404, 422))
        self.assertTrue(foreign.headers.get("X-Request-ID"))
        self.assertNotEqual(foreign.headers["X-Request-ID"], "forged")
        self.assertNotEqual(foreign.headers["X-Request-ID"], invalid.headers["X-Request-ID"])
        ends = [json.loads(r.getMessage()) for r in captured.records if json.loads(r.getMessage())["event"] == "request_finished"]
        self.assertTrue(all(row["outcome"] == "rejected" for row in ends))

    def test_answer_save_failure_does_not_emit_done_or_log_success(self):
        client = TestClient(create_app(self.store, Turn()))
        conversation = client.post("/api/conversations").json()["id"]
        with patch.object(self.store, "save_complete_conversation_turn", side_effect=RuntimeError("database write failed")):
            with self.assertLogs("legal_rag.requests", level="INFO") as captured:
                response = client.post(f"/api/conversations/{conversation}/chat", json={"question": "问题"})
        self.assertNotIn("event: done", response.text)
        self.assertIn("event: error", response.text)
        end = json.loads(captured.records[-1].getMessage())
        self.assertEqual(end["outcome"], "failed")
        self.assertIn("database write failed", end["error"])
        self.assertEqual(client.get(f"/api/conversations/{conversation}/messages").json()["messages"], [])

    def test_log_files_are_json_lines_and_rotate(self):
        from request_logging import create_request_logger, write_event
        path = Path(self.directory.name) / "logs" / "requests.jsonl"
        logger = create_request_logger(path, max_bytes=450, backup_count=2)
        try:
            for index in range(20):
                write_event(logger, "request_started", request_id=str(index))
            files = list(path.parent.glob("requests.jsonl*"))
            self.assertGreater(len(files), 1)
            self.assertLessEqual(len(files), 3)
            rows = [json.loads(line) for file in files for line in file.read_text(encoding="utf8").splitlines()]
            self.assertTrue(any(row["request_id"] == "19" for row in rows))
            self.assertTrue(all("timestamp" in row for row in rows))
        finally:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


class DisconnectLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_pathsend_file_response_is_completed(self):
        from request_logging import RequestLoggingMiddleware

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.pathsend", "path": "index.html"})

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            pass

        middleware = RequestLoggingMiddleware(app, logger=logging.getLogger("legal_rag.requests"))
        with self.assertLogs("legal_rag.requests", level="INFO") as captured:
            await middleware({"type": "http", "method": "GET", "path": "/", "state": {}, "extensions": {"http.response.pathsend": {}}}, receive, send)
        self.assertEqual(json.loads(captured.records[-1].getMessage())["outcome"], "completed")

    async def test_unhandled_failure_before_headers_returns_safe_correlated_500(self):
        from request_logging import RequestLoggingMiddleware

        async def app(scope, receive, send):
            raise RuntimeError("internal database failure")

        async def receive():
            return {"type": "http.request", "body": b""}

        sent = []

        async def send(message):
            sent.append(message)

        middleware = RequestLoggingMiddleware(app, logger=logging.getLogger("legal_rag.requests"))
        with self.assertLogs("legal_rag.requests", level="INFO") as captured:
            await middleware({"type": "http", "method": "GET", "path": "/api/conversations", "state": {}}, receive, send)
        self.assertEqual(sent[0]["status"], 500)
        payload = json.loads(sent[1]["body"])
        self.assertEqual(payload["request_id"], dict(sent[0]["headers"])[b"x-request-id"].decode())
        self.assertNotIn("internal database failure", payload["detail"])
        end = json.loads(captured.records[-1].getMessage())
        self.assertEqual(end["outcome"], "failed")
        self.assertIn("internal database failure", end["error"])

    async def test_response_start_alone_does_not_finish_the_request_log(self):
        from request_logging import RequestLoggingMiddleware

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            pass

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})
            self.assertEqual(len(captured.records), 1)
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        middleware = RequestLoggingMiddleware(app, logger=logging.getLogger("legal_rag.requests"))
        with self.assertLogs("legal_rag.requests", level="INFO") as captured:
            await middleware({"type": "http", "method": "GET", "path": "/stream", "state": {}}, receive, send)
        self.assertEqual(json.loads(captured.records[-1].getMessage())["outcome"], "completed")

    async def test_send_disconnect_is_not_a_server_failure(self):
        from request_logging import RequestLoggingMiddleware

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})

        async def receive():
            return {"type": "http.request", "body": b""}

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        middleware = RequestLoggingMiddleware(app, logger=logging.getLogger("legal_rag.requests"))
        with self.assertLogs("legal_rag.requests", level="INFO") as captured:
            await middleware({"type": "http", "method": "POST", "path": "/stream", "state": {}}, receive, send)
        self.assertEqual(json.loads(captured.records[-1].getMessage())["outcome"], "disconnected")

    async def test_disconnect_is_not_logged_as_completion(self):
        from request_logging import RequestLoggingMiddleware

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})
            await receive()

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        middleware = RequestLoggingMiddleware(app, logger=logging.getLogger("legal_rag.requests"))
        with self.assertLogs("legal_rag.requests", level="INFO") as captured:
            await middleware({"type": "http", "method": "POST", "path": "/api/chat", "state": {}}, receive, send)
        rows = [json.loads(r.getMessage()) for r in captured.records]
        self.assertEqual(rows[-1]["outcome"], "disconnected")


if __name__ == "__main__":
    unittest.main()
