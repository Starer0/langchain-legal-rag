import os
import unittest
from psycopg2 import sql
import test_web_login as login
from test_conversation_context import item

from context_storage import ContextStore
from prepare_conversation_context import prepare_context
from conversation_context import ContextConflict, ContextValidation


class ContextStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS') != '1': raise unittest.SkipTest('Real temporary PostgreSQL required')

    def setUp(self):
        self.fixture = login.WebLoginTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.db, self.schema = f.db, f.schema
        self.owner, self.other = f.login_session().user.id, f.login_session('demo_c').user.id
        self.cid = f.store.create_conversation(self.owner).id
        from prepare_generation_tasks import prepare_tasks
        prepare_tasks(self.db, apply=True, schema=self.schema)
        self.assertFalse(prepare_context(self.db,schema=self.schema)['ready'])
        prepare_context(self.db,apply=True,schema=self.schema)
        self.contexts = ContextStore(schema=self.schema, **f.settings)
        f.store.save_complete_conversation_turn(self.owner,self.cid,'合同期限两年','旧助手判断')

    def summary(self): return {'revision':1,'through_ordinal':2,'items':[item(1,'合同期限两年')]}

    def test_additive_migration_idempotent(self):
        self.assertEqual(prepare_context(self.db,apply=True,schema=self.schema)['action'],'already_ready')
        self.assertEqual(self.fixture.store.load_conversation_messages(self.owner,self.cid), [('user','合同期限两年'),('assistant','旧助手判断')])

    def test_owner_checked_read_and_save(self):
        with self.assertRaises(PermissionError): self.contexts.read(self.other,self.cid)
        with self.assertRaises(PermissionError): self.contexts.save(self.other,self.cid,self.summary(),0)
        self.assertEqual(self.contexts.read(self.owner,self.cid)['revision'],0)

    def test_save_is_cas_and_preserves_transcript(self):
        self.contexts.save(self.owner,self.cid,self.summary(),0)
        with self.assertRaises(ContextConflict): self.contexts.save(self.owner,self.cid,self.summary(),0)
        self.assertEqual(len(self.fixture.store.load_conversation_messages(self.owner,self.cid)),2)

    def test_invalid_summary_source_cannot_be_saved(self):
        summary = self.summary(); summary['items'] = [item(1,'三年')]
        with self.assertRaises(ContextValidation): self.contexts.save(self.owner,self.cid,summary,0)

    def test_assistant_claim_cannot_be_saved(self):
        summary = self.summary(); summary['items'] = [item(2,'旧助手判断')]
        with self.assertRaises(ContextValidation): self.contexts.save(self.owner,self.cid,summary,0)

    def test_capture_shares_transaction_and_freezes_history(self):
        from generation_tasks import TaskStore
        tasks = TaskStore(self.fixture.store,contexts=self.contexts)
        task,_ = tasks.create(self.owner,self.cid,'那工资呢','key','request')
        frozen = self.contexts.task_input(self.owner,task['id'])
        self.fixture.store.save_complete_conversation_turn(self.owner,self.cid,'新消息','新回答')
        self.assertEqual(len(frozen['messages']),2)
        self.assertEqual(self.contexts.task_input(self.owner,task['id']),frozen)
        with self.assertRaises(PermissionError): self.contexts.task_input(self.other,task['id'])

    def test_saved_task_result_is_reused_and_epoch_fenced(self):
        from generation_tasks import TaskStore
        task,_ = TaskStore(self.fixture.store,contexts=self.contexts).create(self.owner,self.cid,'问','key','request')
        result = {'summary':self.summary(),'history':['摘要'],'compressed':True}
        self.contexts.save_result(self.owner,self.cid,task['id'],result,0)
        self.assertEqual(self.contexts.task_result(self.owner,task['id']),result)
        self.contexts.save_result(self.owner,self.cid,task['id'],result,0)
        self.assertEqual(self.contexts.read(self.owner,self.cid)['revision'],1)

    def test_result_cannot_skip_source_validation(self):
        from generation_tasks import TaskStore
        task,_ = TaskStore(self.fixture.store,contexts=self.contexts).create(self.owner,self.cid,'问','key','request')
        bad = {'summary':self.summary(),'history':['摘要'],'compressed':True}
        bad['summary']['items'] = [item(1,'编造的三年')]
        with self.assertRaises(ContextValidation): self.contexts.save_result(self.owner,self.cid,task['id'],bad,0)
        self.assertIsNone(self.contexts.task_result(self.owner,task['id']))

    def test_delete_cleans_summary_and_task_inputs(self):
        from generation_tasks import TaskStore
        tasks = TaskStore(self.fixture.store,contexts=self.contexts)
        task,_ = tasks.create(self.owner,self.cid,'问','key','request')
        self.contexts.save(self.owner,self.cid,self.summary(),0)
        tasks.update(task['id'],status='interrupted')
        tasks.delete_conversation(self.owner,self.cid)
        with self.assertRaises(PermissionError): self.contexts.read(self.owner,self.cid)
        self.assertIsNone(self.contexts.task_result(self.owner,task['id']))

    def test_foreign_conversation_task_result_rejected(self):
        from generation_tasks import TaskStore
        task,_ = TaskStore(self.fixture.store,contexts=self.contexts).create(self.owner,self.cid,'问','key','request')
        second = self.fixture.store.create_conversation(self.owner).id
        with self.assertRaises(PermissionError): self.contexts.save_result(self.owner,second,task['id'],{},0)

    def test_recovery_runner_uses_frozen_context_for_more_than_four_turns(self):
        import time
        import test_web_rag as f
        from generation_tasks import TaskStore
        from task_recovery import schema_sql, MIGRATION
        from recovery_runner import RecoveryRunner
        from web_langgraph import LangGraphStreamingRagTurn
        from conversation_context import ContextService
        from langchain_core.documents import Document
        with self.db,self.db.cursor() as c:
            table=lambda name: sql.Identifier(self.schema,name).as_string(c)
            for statement in schema_sql(table('generation_recovery'),table('generation_tasks'),table('checkpoint_cleanup')): c.execute(statement)
            c.execute(sql.SQL('INSERT INTO {}.schema_migrations VALUES (%s)').format(sql.Identifier(self.schema)),(MIGRATION,))
        for n in range(5): self.fixture.store.save_complete_conversation_turn(self.owner,self.cid,f'历史条件{n}','旧回答')
        rewriter=f.FakeRewriter('检索问题')
        docs=[Document(page_content='法律依据',metadata={'knowledge_base_id':'A','index_status':'active'})]
        turn=LangGraphStreamingRagTurn(rewriter=rewriter,retriever=f.FakeRetriever(docs),reranker=f.FakeReranker(docs),
            prompt=f.FakePrompt(),model=f.FakeStreamingModel(['回答']),history_turns=4)
        tasks=TaskStore(self.fixture.store,contexts=self.contexts)
        runner=RecoveryRunner(tasks,turn,lambda owner:{'A'},context_service=ContextService(None))
        self.addCleanup(runner.shutdown)
        task=runner.start(self.owner,self.cid,'那工资呢','key','request',{'allowed_knowledge_bases':{'A'}})
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            observed=tasks.get(self.owner,self.cid,task['id'])
            if observed['status']!='running': break
            time.sleep(.01)
        self.assertEqual(observed['status'],'completed',observed.get('error'))
        self.assertEqual(len(rewriter.calls[0][1]),6)
        self.assertIsNotNone(self.contexts.task_result(self.owner,task['id']))


if __name__ == '__main__': unittest.main()
