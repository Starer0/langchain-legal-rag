import tempfile
import unittest
import time
import os
import subprocess
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.documents import Document
import test_web_rag as f
from web_langgraph import LangGraphStreamingRagTurn
from web_storage import SQLiteConversationStore


class RecoveryTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = SQLiteConversationStore(Path(self.tmp.name) / 'tasks.db')
        self.store.ensure_session('owner')
        self.cid = self.store.create_conversation('owner').id

    def test_claim_fences_old_worker_and_limits_recovery(self):
        from generation_tasks import TaskStore
        from task_recovery import RecoveryStore
        tasks = TaskStore(self.store)
        recovery = RecoveryStore(tasks)
        task, _ = tasks.create('owner', self.cid, '问题', 'key', 'req', recovery={'version': 'v1', 'scope': ['A']})
        self.assertEqual(recovery.claim(task['id'], recovering=False)['epoch'], 1)
        self.assertEqual(recovery.claim(task['id'], recovering=True)['epoch'], 2)
        self.assertFalse(tasks.update(task['id'], answer='旧线程', epoch=1))
        self.assertFalse(tasks.complete(task['id'], '旧答案', [], epoch=1))
        self.assertTrue(tasks.update(task['id'], answer='新线程', epoch=2))
        self.assertEqual(recovery.claim(task['id'], recovering=True)['epoch'], 3)
        self.assertEqual(recovery.claim(task['id'], recovering=True)['epoch'], 4)
        self.assertIsNone(recovery.claim(task['id'], recovering=True))
        self.assertEqual(tasks.get('owner', self.cid, task['id'])['status'], 'interrupted')

    def test_complete_is_authoritative_and_idempotent(self):
        from generation_tasks import TaskStore
        from task_recovery import RecoveryStore
        tasks = TaskStore(self.store)
        recovery = RecoveryStore(tasks)
        task, _ = tasks.create('owner', self.cid, '问题', 'key', 'req', recovery={'version': 'v1', 'scope': ['A']})
        epoch = recovery.claim(task['id'], recovering=False)['epoch']
        self.assertTrue(tasks.complete(task['id'], '完整答案', [], epoch=epoch))
        self.assertFalse(tasks.complete(task['id'], '重复', [], epoch=epoch))
        self.assertEqual(len(self.store.load_conversation_messages('owner', self.cid)), 2)
        self.assertEqual(recovery.pending(), [])

    def test_legacy_task_remains_interrupted(self):
        from generation_tasks import TaskStore
        from task_recovery import RecoveryStore
        tasks = TaskStore(self.store)
        recovery = RecoveryStore(tasks)
        task, _ = tasks.create('owner', self.cid, '旧问题', 'key', 'req')
        recovery.interrupt_legacy()
        self.assertEqual(tasks.get('owner', self.cid, task['id'])['status'], 'interrupted')

    def test_delete_enqueues_checkpoint_cleanup(self):
        from generation_tasks import TaskStore
        from task_recovery import RecoveryStore
        tasks = TaskStore(self.store)
        recovery = RecoveryStore(tasks)
        task, _ = tasks.create('owner', self.cid, '问题', 'key', 'req', recovery={'version': 'v1', 'scope': ['A']})
        tasks.complete(task['id'], '回答', [])
        self.assertTrue(tasks.delete_conversation('owner', self.cid))
        self.assertEqual(recovery.cleanup_ids(), [task['id']])


    def test_startup_recovers_complete_checkpoint_and_changed_scope_is_blocked(self):
        from generation_tasks import TaskStore
        from task_recovery import RecoveryStore
        from recovery_runner import RecoveryRunner
        tasks = TaskStore(self.store)
        RecoveryStore(tasks)
        class Turn:
            VERSION = 'v1'
            def stream(self, question, messages, **options):
                assert options['resume']
                yield {'event': 'done', 'data': {'answer': '恢复结果', 'sources': []}}
            def delete_checkpoints(self, tid): pass
        task, _ = tasks.create('owner', self.cid, '问题', 'key', 'req', recovery={'version': 'v1', 'scope': ['A']})
        runner = RecoveryRunner(tasks, Turn(), lambda owner: {'A'})
        runner.recover()
        deadline = time.monotonic() + 3
        while tasks.active('owner') and time.monotonic() < deadline: time.sleep(.01)
        runner.shutdown()
        self.assertEqual(tasks.get('owner', self.cid, task['id'])['answer'], '恢复结果')
        second, _ = tasks.create('owner', self.cid, '第二问', 'key2', 'req', recovery={'version': 'v1', 'scope': ['A']})
        runner = RecoveryRunner(tasks, Turn(), lambda owner: {'C'})
        runner.recover()
        deadline = time.monotonic() + 3
        while tasks.active('owner') and time.monotonic() < deadline: time.sleep(.01)
        runner.shutdown()
        self.assertEqual(tasks.get('owner', self.cid, second['id'])['status'], 'interrupted')

    def test_web_api_uses_recoverable_runner(self):
        from fastapi.testclient import TestClient
        from web_app import create_app, COOKIE_NAME
        class Turn:
            VERSION = 'v1'
            def stream(self, question, messages, **options):
                yield {'event': 'done', 'data': {'answer': '网页答案', 'sources': []}}
            def delete_checkpoints(self, tid): pass
        app = create_app(self.store, Turn(), background_tasks=True,
                         recovery_config={'scope_resolver': lambda owner: {'A'}, 'lease': None})
        with TestClient(app) as client:
            client.cookies.set(COOKIE_NAME, 'owner')
            r = client.post(f'/api/conversations/{self.cid}/tasks', json={'question': '问题', 'submission_key': 'key'})
            self.assertEqual(r.status_code, 202)
            runner = app.state.generation_runner()
            deadline = time.monotonic() + 3
            while runner.tasks.active('owner') and time.monotonic() < deadline: time.sleep(.01)
            self.assertEqual(client.get(f'/api/conversations/{self.cid}/messages').json()['messages'][1]['content'], '网页答案')
            runner.shutdown()

    def test_revoked_scope_blocks_completion_and_model_failure_is_failed(self):
        from generation_tasks import TaskStore
        from recovery_runner import RecoveryRunner
        from threading import Event
        entered, gate = Event(), Event()
        scope = {'A'}
        class Turn:
            VERSION = 'v1'
            def stream(self, question, messages, **options):
                entered.set()
                gate.wait(3)
                yield {'event': 'done', 'data': {'answer': 'SECRET A', 'sources': []}}
            def delete_checkpoints(self, tid): pass
        tasks = TaskStore(self.store)
        runner = RecoveryRunner(tasks, Turn(), lambda owner: scope)
        runner.recover()
        task = runner.start('owner', self.cid, '问题', 'key', 'req', {'allowed_knowledge_bases': {'A'}})
        self.assertTrue(entered.wait(2))
        scope.clear()
        gate.set()
        deadline = time.monotonic() + 3
        while tasks.active('owner') and time.monotonic() < deadline: time.sleep(.01)
        runner.shutdown()
        self.assertEqual(tasks.get('owner', self.cid, task['id'])['status'], 'interrupted')
        self.assertEqual(self.store.load_conversation_messages('owner', self.cid), [])
        class Failure(Turn):
            def stream(self, question, messages, **options):
                raise RuntimeError('ordinary model error')
                yield
        tasks.create('owner', self.cid, '第二问', 'key2', 'req', recovery={'version': 'v1', 'scope': ['A']})
        runner = RecoveryRunner(tasks, Failure(), lambda owner: {'A'})
        runner.recover()
        deadline = time.monotonic() + 3
        while tasks.active('owner') and time.monotonic() < deadline: time.sleep(.01)
        runner.shutdown()
        self.assertEqual(tasks.latest('owner', self.cid)['status'], 'failed')

    def test_recovery_queue_drains_above_worker_capacity(self):
        from generation_tasks import TaskStore
        from task_recovery import RecoveryStore
        from recovery_runner import RecoveryRunner
        from threading import Event, Lock
        tasks = TaskStore(self.store)
        RecoveryStore(tasks)
        gate, lock = Event(), Lock()
        self.addCleanup(gate.set)
        counts = {'active': 0, 'max': 0}
        class Turn:
            VERSION = 'v1'
            def stream(self, question, messages, **options):
                with lock:
                    counts['active'] += 1
                    counts['max'] = max(counts['max'], counts['active'])
                try:
                    if not gate.wait(3): raise RuntimeError('test timeout')
                    yield {'event': 'done', 'data': {'answer': '回答', 'sources': []}}
                finally:
                    with lock: counts['active'] -= 1
            def delete_checkpoints(self, tid): pass
        for i in range(6):
            owner = 'owner' + str(i)
            self.store.ensure_session(owner)
            cid = self.store.create_conversation(owner).id
            tasks.create(owner, cid, '问题', str(i), 'req', recovery={'version': 'v1', 'scope': ['A']})
        runner = RecoveryRunner(tasks, Turn(), lambda owner: {'A'}, capacity=2)
        runner.recover()
        deadline = time.monotonic() + 3
        while counts['max'] < 2 and time.monotonic() < deadline: time.sleep(.01)
        self.assertEqual(counts['max'], 2)
        gate.set()
        while runner.recovery.pending() and time.monotonic() < deadline: time.sleep(.01)
        runner.shutdown()
        self.assertEqual(runner.recovery.pending(), [])
        self.assertLessEqual(counts['max'], 2)


class ProcessCrashTests(unittest.TestCase):
    def test_force_kill_then_recover_answer_and_completed_graph(self):
        fixture = Path(__file__).parent / 'fixtures' / 'recovery_process.py'
        for phase in ('before_answer', 'partial_answer', 'before_save'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                command = [sys.executable, str(fixture), str(root), phase]
                child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 15
                    while not (root/'blocked').exists() and child.poll() is None and time.monotonic() < deadline:
                        time.sleep(.03)
                    self.assertTrue((root/'blocked').exists(), 'fixture did not reach crash boundary')
                    child.kill()
                    child.wait(timeout=5)
                finally:
                    if child.poll() is None: child.kill(); child.wait(timeout=5)
                    child.stderr.close()
                resumed = subprocess.run([sys.executable, str(fixture), str(root), 'resume'], capture_output=True, timeout=20)
                self.assertEqual(resumed.returncode, 0, resumed.stderr.decode(errors='replace')[-1000:])
                import json
                result = json.loads((root/'result.json').read_text(encoding='utf-8'))
                self.assertEqual(result['answer'], '新的完整回答')
                self.assertEqual(result['messages'], 2)
                calls = (root/'calls').read_text().splitlines()
                for stage in ('rewrite', 'retrieve', 'rerank'): self.assertEqual(calls.count(stage), 1)
                self.assertEqual(calls.count('answer'), 1 if phase == 'before_save' else 2)


@unittest.skipUnless(os.getenv('RUN_POSTGRES_TESTS') == '1', 'PostgreSQL opt-in')
class PostgresRecoveryTests(unittest.TestCase):
    def setUp(self):
        import test_source_history as fixture
        fixture.PostgresSourceHistoryTests.setUp(self)
        from prepare_generation_tasks import prepare_tasks
        prepare_tasks(self.db, schema=self.schema, apply=True)
        self.addCleanup(self.cleanup_graph_schema)

    def cleanup_schema(self):
        import test_source_history as fixture
        fixture.PostgresSourceHistoryTests.cleanup_schema(self)

    def cleanup_graph_schema(self):
        from psycopg2 import sql
        self.db.rollback()
        assert self.schema.startswith('source_test_')
        with self.db, self.db.cursor() as c:
            c.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(self.schema + '_graph')))

    def test_migration_checkpoint_reopen_and_exclusive_lease(self):
        from prepare_web_recovery import prepare_recovery
        from checkpoint_storage import CheckpointStorage
        from generation_tasks import TaskStore
        from task_recovery import RecoveryStore
        self.assertFalse(prepare_recovery(self.settings, schema=self.schema)['ready'])
        self.assertTrue(prepare_recovery(self.settings, schema=self.schema, apply=True)['ready'])
        self.assertTrue(prepare_recovery(self.settings, schema=self.schema, apply=True)['ready'])
        recovery = RecoveryStore(TaskStore(self.store))
        self.assertEqual(recovery.pending(), [])
        with CheckpointStorage(self.settings, schema=self.schema) as first:
            first.check()
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                with CheckpointStorage(self.settings, schema=self.schema): pass
            graph = CheckpointRecoveryTests()
            graph.setUp()
            self.addCleanup(graph.doCleanups)
            list(graph.turn(first.saver, f.FakeStreamingModel(['PG结果'])).stream('问题', [], allowed_knowledge_bases={'A'}, task_id='pg-task'))
        with CheckpointStorage(self.settings, schema=self.schema) as second:
            model = f.FakeStreamingModel(['不能重做'])
            result = list(graph.turn(second.saver, model).stream('问题', [], allowed_knowledge_bases={'A'}, task_id='pg-task', resume=True))
            self.assertEqual(result[-1]['data']['answer'], 'PG结果')
            self.assertFalse(model.prompts)
            tasks = recovery.tasks
            task, _ = tasks.create('owner', self.cid, '问题', 'epoch', 'req', recovery={'version': 'v1', 'scope': ['A']})
            old = recovery.claim(task['id'], recovering=False)['epoch']
            recovery.claim(task['id'], recovering=True)
            with self.assertRaisesRegex(RuntimeError, 'Stale checkpoint'):
                list(graph.turn(second.saver, model).stream('问题', [], allowed_knowledge_bases={'A'}, task_id=task['id'], execution_epoch=old))



class CheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / 'checkpoints.db')
        self.doc = Document(page_content='合法资料', metadata={'knowledge_base_id': 'A', 'index_status': 'active'})
        self.rewriter = f.FakeRewriter('检索问题')
        self.retriever = f.FakeRetriever([self.doc])
        self.reranker = f.FakeReranker([self.doc])

    def turn(self, saver, model):
        return LangGraphStreamingRagTurn(checkpointer=saver,
            rewriter=self.rewriter, retriever=self.retriever, reranker=self.reranker,
            prompt=f.FakePrompt(), model=model, history_turns=3, require_authorization=True)

    def test_reopen_resumes_answer_without_repeating_completed_nodes(self):
        class Fail:
            def stream(self, prompt):
                yield '旧半段'
                raise RuntimeError('process interrupted')
        with SqliteSaver.from_conn_string(self.path) as saver:
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                list(self.turn(saver, Fail()).stream('问题', [], allowed_knowledge_bases={'A'}, task_id='task'))
        with SqliteSaver.from_conn_string(self.path) as saver:
            turn = self.turn(saver, f.FakeStreamingModel(['新的', '完整回答']))
            result = list(turn.stream('问题', [], allowed_knowledge_bases={'A'}, task_id='task', resume=True))[-1]['data']
            self.assertEqual(result['answer'], '新的完整回答')
            self.assertEqual(len(self.rewriter.calls), 1)
            self.assertEqual(len(self.retriever.states), 1)
            self.assertEqual(len(self.reranker.states), 1)
            snapshot = turn.graph.get_state({'configurable': {'thread_id': 'task'}})
            self.assertFalse(snapshot.next)
            self.assertNotIn('_trace', snapshot.values)
            self.assertNotIn('_profile', snapshot.values)
            self.assertNotIn('_cancelled', snapshot.values)

    def test_finished_checkpoint_delivers_result_without_model(self):
        with SqliteSaver.from_conn_string(self.path) as saver:
            list(self.turn(saver, f.FakeStreamingModel(['已完成'])).stream('问题', [], allowed_knowledge_bases={'A'}, task_id='task'))
        model = f.FakeStreamingModel(['不应执行'])
        with SqliteSaver.from_conn_string(self.path) as saver:
            result = list(self.turn(saver, model).stream('问题', [], allowed_knowledge_bases={'A'}, task_id='task', resume=True))
            self.assertEqual(result[-1]['data']['answer'], '已完成')
            self.assertEqual(model.prompts, [])

    def test_resume_rejects_changed_scope_or_question(self):
        with SqliteSaver.from_conn_string(self.path) as saver:
            turn = self.turn(saver, f.FakeStreamingModel(['完成']))
            list(turn.stream('问题', [], allowed_knowledge_bases={'A'}, task_id='task'))
            for question, scope in [('其他问题', {'A'}), ('问题', {'C'})]:
                with self.assertRaises(ValueError):
                    list(turn.stream(question, [], allowed_knowledge_bases=scope, task_id='task', resume=True))

    def test_close_during_rewrite_preserves_pending_checkpoint(self):
        from threading import Event, Thread
        gate, entered = Event(), Event()
        outer = self
        class Rewriter:
            def rewrite(self, question, history, **kwargs):
                entered.set()
                if not gate.wait(3): raise RuntimeError('test timeout')
                return outer.rewriter.rewrite(question, history, **kwargs)
        with SqliteSaver.from_conn_string(self.path) as saver:
            turn = self.turn(saver, f.FakeStreamingModel(['恢复答案']))
            turn.rewriter = Rewriter()
            stream = turn.stream('问题', [], allowed_knowledge_bases={'A'}, task_id='task')
            self.assertEqual(next(stream)['data']['stage'], 'rewrite')
            self.assertTrue(entered.wait(2))
            Thread(target=lambda: (time.sleep(.1), gate.set())).start()
            stream.close()
            snapshot = turn.graph.get_state({'configurable': {'thread_id': 'task'}})
            self.assertTrue(snapshot.next)
            result = list(self.turn(saver, f.FakeStreamingModel(['恢复答案'])).stream('问题', [], allowed_knowledge_bases={'A'}, task_id='task', resume=True))
            self.assertEqual(result[-1]['data']['answer'], '恢复答案')
