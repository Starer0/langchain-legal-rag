import os
import unittest
from psycopg2 import sql
import test_web_login as login
from memory_storage import MemoryStore


class MemoryStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS') != '1': raise unittest.SkipTest('Real temporary PostgreSQL required')

    def setUp(self):
        self.fixture = login.WebLoginTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.db, self.schema = self.fixture.db, self.fixture.schema
        self.owner = self.fixture.login_session().user.id
        self.other = self.fixture.login_session('demo_c').user.id
        from prepare_generation_tasks import prepare_tasks
        prepare_tasks(self.db, apply=True, schema=self.schema)
        from memory_storage import MemoryStore
        from prepare_account_memory import prepare_memory
        self.prepare = prepare_memory
        self.assertFalse(prepare_memory(self.db, schema=self.schema)['ready'])
        prepare_memory(self.db, apply=True, schema=self.schema)
        self.memories = MemoryStore(schema=self.schema, **self.fixture.settings)
        from memory_service import MemoryService
        from test_memory_service import Embeddings
        self.service = MemoryService(Embeddings(), fingerprint='fixture-v1')

    def save(self, core='中文', extended='Python初学者'):
        old = self.memories.read(self.owner)
        return self.memories.save(self.owner, self.service.prepare(old, core, extended, True), old['revision'])

    def test_migration_is_idempotent_and_read_is_scoped(self):
        self.assertEqual(self.prepare(self.db, apply=True, schema=self.schema)['action'], 'already_ready')
        self.save()
        self.assertEqual(self.memories.read(self.other)['core_text'], '')
        self.assertEqual(self.memories.read(self.owner)['revision'], 1)

    def test_stale_revision_preserves_current_memory(self):
        from memory_service import MemoryConflict
        stale = self.service.prepare(self.memories.read(self.owner), '旧页面', '', True)
        self.save('新页面')
        with self.assertRaises(MemoryConflict): self.memories.save(self.owner, stale, 0)
        self.assertEqual(self.memories.read(self.owner)['core_text'], '新页面')

    def test_clear_removes_entries_and_keeps_monotonic_revision(self):
        self.save()
        self.save('', '')
        self.assertEqual(self.memories.read(self.owner)['entries'], [])
        self.assertEqual(self.memories.read(self.owner)['revision'], 2)

    def test_inactive_account_cannot_mutate(self):
        with self.db, self.db.cursor() as c:
            c.execute(sql.SQL('UPDATE {}.users SET is_active=false WHERE id=%s').format(sql.Identifier(self.schema)), (self.owner,))
        with self.assertRaises(PermissionError): self.save()

    def test_public_projection_never_contains_vectors(self):
        from memory_storage import public_document
        doc = public_document(self.save())
        self.assertNotIn('entries', doc)
        self.assertNotIn('embedding', str(doc))
        self.assertEqual(doc['core_text'], '中文')

    def test_cursor_write_rolls_back_with_outer_transaction(self):
        original = self.save()
        from memory_service import MemoryConflict
        prepared = self.service.prepare(original, '新偏好', '', True)
        with self.assertRaises(MemoryConflict):
            with self.memories.cursor() as c:
                self.memories.save(self.owner, prepared, 1, cursor=c)
                raise MemoryConflict('abort')
        self.assertEqual(self.memories.read(self.owner)['revision'], 1)
        self.assertEqual(self.memories.read(self.owner)['core_text'], '中文')

    def test_unmarked_table_migration_refuses_adoption(self):
        with self.db, self.db.cursor() as c:
            c.execute(sql.SQL('DELETE FROM {}.schema_migrations WHERE name=%s').format(sql.Identifier(self.schema)), ('v12_account_memory_v1',))
        with self.assertRaises(ValueError): self.prepare(self.db, apply=True, schema=self.schema)

    def task(self):
        from generation_tasks import TaskStore
        self.tasks = TaskStore(self.fixture.store, memory=self.memories)
        cid = self.fixture.store.create_conversation(self.owner).id
        doc = self.memories.read(self.owner)
        task, _ = self.tasks.create(self.owner, cid, '记住中文', 'key', 'req', memory_input=doc)
        return task, doc

    def test_task_mutation_and_feedback_commit_once(self):
        task, doc = self.task()
        prepared = self.service.prepare(doc, '默认中文', '', True)
        result = dict(expected_revision=0, prepared=prepared, feedback='已更新核心记忆。',question=task['question'],operation=dict(action='add',layer='core',content='默认中文'))
        self.assertTrue(self.tasks.complete(task['id'], '正在保存', [], memory_result=result))
        self.assertFalse(self.tasks.complete(task['id'], '再次保存', [], memory_result=result))
        self.assertEqual(self.memories.read(self.owner)['revision'], 1)
        rows = self.fixture.store.load_conversation_messages(self.owner, task['conversation_id'])
        self.assertEqual(rows, [('user', '记住中文'), ('assistant', '已更新核心记忆。')])

    def test_conflict_does_not_publish_saved_feedback(self):
        from memory_service import MemoryConflict
        task, doc = self.task()
        self.save('新的设置')
        result = dict(expected_revision=0, prepared=self.service.prepare(doc, '中文', '', True), feedback='已更新',question=task['question'],operation=dict(action='add',layer='core',content='中文'))
        with self.assertRaises(MemoryConflict): self.tasks.complete(task['id'], '正在保存', [], memory_result=result)
        self.assertEqual(self.memories.read(self.owner)['core_text'], '新的设置')
        self.assertEqual(self.fixture.store.load_conversation_messages(self.owner, task['conversation_id']), [])

    def test_snapshot_and_warning_are_scoped_and_cleaned(self):
        task, doc = self.task()
        self.assertIsNone(self.memories.task_input(task['id'], self.other))
        self.assertEqual(self.memories.task_input(task['id'], self.owner)['revision'], 0)
        self.tasks.complete(task['id'], '回答', [], memory_status=dict(status='extended_unavailable', warning='本次未使用扩展记忆', revision=0))
        observed = self.tasks.get(self.owner, task['conversation_id'], task['id'])
        self.assertEqual(observed['memory']['warning'], '本次未使用扩展记忆')
        self.assertNotIn('core_text', str(observed))
        self.tasks.delete_conversation(self.owner, task['conversation_id'])
        self.assertIsNone(self.memories.task_input(task['id'], self.owner))

    def recoverable_task(self, question, turn):
        from generation_tasks import TaskStore
        from task_recovery import schema_sql, MIGRATION
        self.tasks = TaskStore(self.fixture.store, memory=self.memories)
        with self.db, self.db.cursor() as c:
            for statement in schema_sql(self.tasks.table('generation_recovery'), self.tasks.table('generation_tasks'), self.tasks.table('checkpoint_cleanup')): c.execute(statement)
            c.execute(sql.SQL('INSERT INTO {}.schema_migrations VALUES (%s)').format(sql.Identifier(self.schema)), (MIGRATION,))
        cid = self.fixture.store.create_conversation(self.owner).id
        snapshot = self.memories.snapshot(self.owner)
        task, _ = self.tasks.create(self.owner, cid, question, 'key', 'req', memory_input=snapshot,
            recovery={'version':turn.VERSION, 'scope':{'A','B'}})
        return task, snapshot

    def test_recovery_after_command_graph_finished_commits_once_without_model_replay(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from langgraph.checkpoint.sqlite import SqliteSaver
        from web_langgraph import LangGraphStreamingRagTurn
        from recovery_runner import RecoveryRunner
        import test_web_rag as f
        class Command:
            calls = 0
            def invoke(inner, prompt):
                inner.calls += 1
                return SimpleNamespace(content='{"action":"add","layer":"core","content":"中文"}')
        self.service.command_model = Command()
        with tempfile.TemporaryDirectory() as root, SqliteSaver.from_conn_string(str(Path(root)/'checkpoint.db')) as saver:
            turn = LangGraphStreamingRagTurn(checkpointer=saver, rewriter=f.FakeRewriter('法律问题'), retriever=f.FakeRetriever([]),
                reranker=f.FakeReranker([]), prompt=f.FakePrompt(), model=f.FakeStreamingModel(['回答']), history_turns=4)
            task, snapshot = self.recoverable_task('记住中文', turn)
            list(turn.stream(task['question'], [], task_id=task['id'], allowed_knowledge_bases={'A','B'}, memory_input=snapshot, memory_service=self.service))
            self.assertEqual(self.memories.read(self.owner)['revision'], 0)
            runner = RecoveryRunner(self.tasks, turn, lambda owner: {'A','B'}, memory_service=self.service)
            runner._run(task, True, None, __import__('threading').Event())
            runner._run(task, True, None, __import__('threading').Event())
            self.assertEqual(self.service.command_model.calls, 1)
            self.assertEqual(self.memories.read(self.owner)['revision'], 1)
            self.assertEqual(len(self.fixture.store.load_conversation_messages(self.owner, task['conversation_id'])), 2)
            self.assertEqual(self.tasks.get(self.owner, task['conversation_id'], task['id'])['answer'], '已更新核心记忆。')

    def test_stale_worker_cannot_commit_memory_or_feedback(self):
        from types import SimpleNamespace
        from task_recovery import RecoveryStore
        task, snapshot = self.recoverable_task('记住中文', SimpleNamespace(VERSION='v1'))
        recovery = RecoveryStore(self.tasks)
        epoch = recovery.claim(task['id'], recovering=False)['epoch']
        recovery.claim(task['id'], recovering=True)
        proposal = dict(expected_revision=0, feedback='已更新', prepared=self.service.prepare(snapshot, '中文', '', True),question=task['question'],operation=dict(action='add',layer='core',content='中文'))
        self.assertFalse(self.tasks.complete(task['id'], '正在保存', [], epoch=epoch, memory_result=proposal))
        self.assertEqual(self.memories.read(self.owner)['revision'], 0)
        self.assertEqual(self.fixture.store.load_conversation_messages(self.owner, task['conversation_id']), [])

    def test_memory_write_rolls_back_if_feedback_cannot_save(self):
        from unittest.mock import patch
        task, snapshot = self.task()
        proposal = dict(expected_revision=0, feedback='已更新', prepared=self.service.prepare(snapshot, '中文', '', True),question=task['question'],operation=dict(action='add',layer='core',content='中文'))
        original = self.tasks.execute
        def fail(c, statement, params=()):
            if 'INSERT INTO' in statement and 'conversation_messages' in statement: raise RuntimeError('fixture failed feedback')
            return original(c, statement, params)
        with patch.object(self.tasks, 'execute', side_effect=fail), self.assertRaises(RuntimeError):
            self.tasks.complete(task['id'], '正在保存', [], memory_result=proposal)
        self.assertEqual(self.memories.read(self.owner)['revision'], 0)
        self.assertEqual(self.tasks.get(self.owner, task['conversation_id'], task['id'])['status'], 'running')

    def test_completion_rejects_tampered_candidate_against_frozen_request(self):
        from memory_service import MemoryValidation
        task, snapshot = self.task()
        prepared = self.service.prepare(snapshot,'无关私人事实','',True)
        proposal = dict(question=task['question'],operation=dict(action='add',layer='core',content='中文'),
                        expected_revision=0,prepared=prepared,feedback='已保存')
        with self.assertRaises(MemoryValidation): self.tasks.complete(task['id'],'正在保存',[],memory_result=proposal)
        self.assertEqual(self.memories.read(self.owner)['revision'],0)
        self.assertEqual(self.fixture.store.load_conversation_messages(self.owner,task['conversation_id']),[])
