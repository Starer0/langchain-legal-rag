import tempfile
import time
import unittest
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient
from web_storage import SQLiteConversationStore
from web_app import create_app, COOKIE_NAME


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteConversationStore(Path(self.directory.name) / 'test.db')
        self.store.ensure_session('a')
        self.store.ensure_session('b')
        self.cid = self.store.create_conversation('a').id
        self.gate = Event()
        self.addCleanup(self.gate.set)
        self.calls = 0
        outer = self
        class Turn:
            def stream(self, question, messages):
                outer.calls += 1
                yield {'event': 'status', 'data': {'stage': 'answer'}}
                yield {'event': 'delta', 'data': {'text': '部分'}}
                if not outer.gate.wait(5):
                    raise RuntimeError('test timeout')
                yield {'event': 'delta', 'data': {'text': '完整'}}
                yield {'event': 'done', 'data': {'answer': '部分完整', 'sources': [{'content': '来源'}]}}
        self.app = create_app(self.store, Turn(), background_tasks=True)
        self.client = TestClient(self.app)
        self.client.cookies.set(COOKIE_NAME, 'a')

    def submit(self, key='key1'):
        return self.client.post(f'/api/conversations/{self.cid}/tasks',
                                json={'question': '那实习期呢', 'submission_key': key})

    def snapshot(self, tid):
        return self.client.get(f'/api/conversations/{self.cid}/tasks/{tid}')

    def wait_for(self, tid, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            data = self.snapshot(tid).json()
            if predicate(data): return data
            time.sleep(.01)
        self.fail('task did not reach expected state')

    def test_refresh_observes_same_running_task_and_complete_once(self):
        response = self.submit()
        self.assertEqual(response.status_code, 202)
        tid = response.json()['id']
        self.wait_for(tid, lambda t: t['answer'] == '部分')
        # New browser observer; task is independent from response lifetime.
        fresh = TestClient(self.app)
        fresh.cookies.set(COOKIE_NAME, 'a')
        data = fresh.get(f'/api/conversations/{self.cid}/messages').json()
        self.assertEqual(data['task']['question'], '那实习期呢')
        self.assertEqual(self.submit().json()['id'], tid)
        self.assertEqual(self.submit('another').status_code, 409)
        self.assertEqual(self.client.delete(f'/api/conversations/{self.cid}').status_code, 409)
        self.assertEqual(self.store.load_conversation_messages('a', self.cid), [])
        self.gate.set()
        self.wait_for(tid, lambda t: t['status'] == 'completed')
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.store.load_conversation_messages('a', self.cid),
                         [('user', '那实习期呢'), ('assistant', '部分完整')])
        self.assertEqual(self.store.load_conversation_history('a', self.cid)[1]['sources'][0]['content'], '来源')
        self.assertEqual(self.submit().json()['id'], tid)

    def test_failure_retains_question_without_saving_partial_history(self):
        class Failure:
            def stream(self, question, messages):
                yield {'event': 'delta', 'data': {'text': '半成品'}}
                raise RuntimeError('private provider error')
        app = create_app(self.store, Failure(), background_tasks=True)
        self.client = TestClient(app)
        self.client.cookies.set(COOKIE_NAME, 'a')
        tid = self.submit().json()['id']
        task = self.wait_for(tid, lambda t: t['status'] == 'failed')
        self.assertNotIn('private', task['error'])
        data = self.client.get(f'/api/conversations/{self.cid}/messages').json()
        self.assertEqual(data['messages'], [])
        self.assertEqual(data['task']['question'], '那实习期呢')
        self.assertEqual(self.client.delete(f'/api/conversations/{self.cid}').status_code, 204)

    def test_other_user_cannot_observe_or_submit(self):
        tid = self.submit().json()['id']
        self.client.cookies.set(COOKIE_NAME, 'b')
        self.assertEqual(self.snapshot(tid).status_code, 404)
        self.assertEqual(self.submit().status_code, 404)
        self.gate.set()
        self.client.cookies.set(COOKIE_NAME, 'a')
        self.wait_for(tid, lambda t: t['status'] == 'completed')

    def test_restart_marks_running_interrupted_without_saving_partial(self):
        from generation_tasks import TaskStore
        tasks = TaskStore(self.store)
        task, created = tasks.create('a', self.cid, '问题', 'restart', 'request')
        self.assertTrue(created)
        tasks.interrupt_running()
        self.assertEqual(tasks.get('a', self.cid, task['id'])['status'], 'interrupted')
        self.assertFalse(tasks.complete(task['id'], '不能保存', []))
        self.assertEqual(self.store.load_conversation_messages('a', self.cid), [])
