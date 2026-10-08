import asyncio
import tempfile
import unittest
from threading import Event
from pathlib import Path

from starlette.requests import Request
from fastapi import HTTPException

from web_app import COOKIE_NAME, ChatRequest, create_app
from web_storage import SQLiteConversationStore


class SmallTurn:
    def stream(self, question, messages):
        yield {'event': 'status', 'data': {'stage': 'rewrite'}}
        yield {'event': 'delta', 'data': {'text': '完整回答'}}
        yield {'event': 'done', 'data': {'answer': '完整回答', 'sources': []}}


class WebDisconnectTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SQLiteConversationStore(Path(directory.name) / 'test.sqlite3')
        self.store.ensure_session('owner')
        self.cid = self.store.create_conversation('owner').id
        self.app = create_app(self.store, SmallTurn())
        self.endpoint = next(route.endpoint for route in self.app.routes if getattr(route, 'path', '') == '/api/conversations/{conversation_id}/chat')

    def scope(self, spec='2.3'):
        return {'type': 'http', 'asgi': {'spec_version': spec}, 'method': 'POST',
                'scheme': 'http', 'path': f'/api/conversations/{self.cid}/chat',
                'query_string': b'', 'headers': [(b'cookie', f'{COOKIE_NAME}=owner'.encode())],
                'state': {'request_id': 'disconnect-test'}}

    async def complete_next_request(self):
        scope = self.scope()
        response = self.endpoint(self.cid, ChatRequest(question='重新发送'), Request(scope))
        async def receive():
            await asyncio.Future()
        async def send(message):
            pass
        await response(scope, receive, send)

    def test_disconnect_after_status_releases_lock_and_discards_incomplete_turn(self):
        async def run():
            scope = self.scope()
            response = self.endpoint(self.cid, ChatRequest(question='刷新的问题'), Request(scope))
            disconnected = asyncio.Event()
            async def receive():
                await disconnected.wait()
                return {'type': 'http.disconnect'}
            async def send(message):
                if message['type'] == 'http.response.body':
                    disconnected.set()
                    await asyncio.Future()
            await response(scope, receive, send)
            # Keep the response alive: cleanup must not depend on GC collecting it.
            self.assertEqual(self.store.load_conversation_messages('owner', self.cid), [])
            await self.complete_next_request()
            self.assertEqual(self.store.load_conversation_messages('owner', self.cid),
                             [('user', '重新发送'), ('assistant', '完整回答')])
        asyncio.run(run())

    def test_disconnect_before_generator_starts_also_releases_lock(self):
        async def run():
            scope = self.scope('2.4')
            response = self.endpoint(self.cid, ChatRequest(question='未开始的问题'), Request(scope))
            async def receive():
                await asyncio.Future()
            async def send(message):
                raise OSError('connection closed')
            try:
                await response(scope, receive, send)
            except Exception:
                pass
            self.assertEqual(self.store.load_conversation_messages('owner', self.cid), [])
            await self.complete_next_request()
        asyncio.run(run())

    def test_disconnect_waits_for_running_worker_before_unlocking(self):
        entered, finish = Event(), Event()
        self.addCleanup(finish.set)

        class BlockedTurn(SmallTurn):
            def stream(self, question, messages):
                if question == '等待中的问题':
                    entered.set()
                    if not finish.wait(5):
                        raise RuntimeError('test worker timed out')
                yield from super().stream(question, messages)

        app = create_app(self.store, BlockedTurn())
        self.endpoint = next(route.endpoint for route in app.routes
                             if getattr(route, 'path', '') == '/api/conversations/{conversation_id}/chat')

        async def run():
            scope = self.scope()
            response = self.endpoint(self.cid, ChatRequest(question='等待中的问题'), Request(scope))
            disconnected = asyncio.Event()
            async def receive():
                await disconnected.wait()
                return {'type': 'http.disconnect'}
            async def send(message):
                pass
            task = asyncio.create_task(response(scope, receive, send))
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            disconnected.set()
            # Deliver the disconnect while the thread is still working.
            await asyncio.sleep(0.05)
            with self.assertRaises(HTTPException) as caught:
                self.endpoint(self.cid, ChatRequest(question='不能并行'), Request(self.scope()))
            self.assertEqual(caught.exception.status_code, 409)
            finish.set()
            await asyncio.wait_for(task, 3)
            self.assertEqual(self.store.load_conversation_messages('owner', self.cid), [])
            await self.complete_next_request()
        asyncio.run(run())
