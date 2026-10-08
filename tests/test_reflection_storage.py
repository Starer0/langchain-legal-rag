import os
import unittest
import test_web_login as login
from prepare_generation_tasks import prepare_tasks
from prepare_account_memory import prepare_memory
from memory_storage import MemoryStore
from memory_service import MemoryService, MemoryConflict
from test_memory_service import Embeddings
from test_memory_reflection import candidate
from reflection_storage import ReflectionStore
from prepare_memory_reflection import prepare_reflection


class ReflectionStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS')!='1': raise unittest.SkipTest('Real temporary PG required')

    def setUp(self):
        self.f=login.WebLoginTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.owner=self.f.login_session().user.id;self.other=self.f.login_session('demo_c').user.id
        self.cid=self.f.store.create_conversation(self.owner).id
        prepare_tasks(self.f.db,apply=True,schema=self.f.schema)
        prepare_memory(self.f.db,apply=True,schema=self.f.schema)
        self.assertFalse(prepare_reflection(self.f.db,schema=self.f.schema)['ready'])
        prepare_reflection(self.f.db,apply=True,schema=self.f.schema)
        self.memory=MemoryStore(schema=self.f.schema,**self.f.settings)
        self.store=ReflectionStore(memory=self.memory,schema=self.f.schema,**self.f.settings)
        self.memory.reflection=self.store
        self.service=MemoryService(Embeddings(),fingerprint='fixture')

    def enable(self): return self.store.set_policy(self.owner,True,self.store.policy(self.owner)['epoch'])
    def enqueue(self,text='我喜欢简短回答',now=1000,reason='idle'):
        self.f.store.save_complete_conversation_turn(self.owner,self.cid,text,'回答')
        with self.store.cursor() as c:
            c.execute(f'SELECT MAX(ordinal) FROM "{self.f.schema}".conversation_messages WHERE conversation_id=%s',(self.cid,))
            through=c.fetchone()[0]
            return self.store.enqueue(c,self.owner,self.cid,through,reason,now)

    def test_default_off_and_idempotent_additive_migration(self):
        self.assertFalse(self.store.policy(self.owner)['auto_accumulate'])
        self.assertIsNone(self.enqueue())
        self.assertEqual(prepare_reflection(self.f.db,apply=True,schema=self.f.schema)['action'],'already_ready')

    def test_task_memory_snapshot_includes_only_its_account_automation_policy(self):
        self.assertFalse(self.memory.snapshot(self.owner)['auto_accumulate'])
        self.enable()
        self.assertTrue(self.memory.snapshot(self.owner)['auto_accumulate'])
        self.assertFalse(self.memory.snapshot(self.other)['auto_accumulate'])

    def test_short_chat_due_time_and_claim_once(self):
        self.enable();job=self.enqueue()
        self.assertEqual(job['due_at'],1120)
        self.assertIsNone(self.store.claim(1119))
        claimed=self.store.claim(1120)
        self.assertEqual(claimed['id'],job['id'])
        self.assertIsNone(self.store.claim(1121))

    def test_new_message_coalesces_and_resets_idle_deadline(self):
        self.enable();first=self.enqueue();second=self.enqueue('我习惯看例子',1050)
        self.assertEqual(first['id'],second['id'])
        self.assertIsNone(self.store.claim(1120))
        self.assertEqual(self.store.claim(1170)['end_ordinal'],4)

    def test_compression_promotes_same_pending_range(self):
        self.enable();first=self.enqueue()
        with self.store.cursor() as c: promoted=self.store.enqueue(c,self.owner,self.cid,2,'compression',1001)
        self.assertEqual(promoted['id'],first['id'])
        self.assertIsNotNone(self.store.claim(1001))

    def test_manual_delete_invalidates_running_job_and_old_sources(self):
        self.enable();self.enqueue();claimed=self.store.claim(1120)
        prepared=self.service.prepare(self.memory.read(self.owner),'','',True)
        self.memory.save(self.owner,prepared,0)
        result={'additions':[candidate('我喜欢简短回答')],'suggestions':[],'skipped':[]}
        appended=self.service.prepare(self.memory.read(self.owner),'我喜欢简短回答','',True)
        with self.assertRaises(MemoryConflict): self.store.apply(claimed['id'],claimed['epoch'],result,appended)
        self.assertEqual(self.memory.read(self.owner)['core_text'],'')
        with self.store.cursor() as c: self.assertIsNone(self.store.enqueue(c,self.owner,self.cid,2,'idle',1200))

    def test_saved_once_and_processed_range_not_reextracted(self):
        self.enable();self.enqueue();job=self.store.claim(1120)
        result={'additions':[candidate('我喜欢简短回答')],'suggestions':[],'skipped':[]}
        prepared=self.service.prepare(self.memory.read(self.owner),'我喜欢简短回答','',True)
        saved=self.store.apply(job['id'],job['epoch'],result,prepared)
        self.assertEqual(saved['added'],1)
        self.store.apply(job['id'],job['epoch'],result,prepared)
        self.assertEqual(self.memory.read(self.owner)['revision'],1)
        with self.store.cursor() as c: self.assertIsNone(self.store.enqueue(c,self.owner,self.cid,2,'idle',1200))

    def test_wrong_user_cannot_change_policy_or_read_job(self):
        self.enable();job=self.enqueue()
        self.assertEqual(self.store.jobs(self.other,self.cid),[])
        with self.assertRaises(PermissionError): self.store.retry(self.other,job['id'],1200)

    def test_disable_blocks_old_job(self):
        p=self.enable();self.enqueue();job=self.store.claim(1120)
        self.store.set_policy(self.owner,False,p['epoch'])
        with self.assertRaises(MemoryConflict): self.store.apply(job['id'],job['epoch'],{'additions':[],'suggestions':[],'skipped':[]},None)

    def test_restart_recovers_once_then_terminal_failure(self):
        self.enable();self.enqueue();first=self.store.claim(1120)
        self.store.recover(1121)
        second=self.store.claim(1121)
        self.assertEqual(second['id'],first['id'])
        self.assertGreater(second['epoch'],first['epoch'])
        self.store.recover(1122)
        self.assertIsNone(self.store.claim(1122))
        self.assertEqual(self.store.jobs(self.owner,self.cid)[0]['status'],'failed')

    def test_deleted_conversation_prevents_save(self):
        self.enable();self.enqueue();job=self.store.claim(1120)
        self.f.store.delete_conversation(self.owner,self.cid)
        with self.assertRaises(PermissionError): self.store.apply(job['id'],job['epoch'],{},None)

    def test_forged_candidate_is_not_authorized_by_prepared_text(self):
        self.enable();self.enqueue();job=self.store.claim(1120)
        bad={'additions':[candidate('我喜欢详细解释')],'suggestions':[],'skipped':[]}
        prepared=self.service.prepare(self.memory.read(self.owner),'我喜欢详细解释','',True)
        with self.assertRaises(ValueError): self.store.apply(job['id'],job['epoch'],bad,prepared)
        self.assertEqual(self.memory.read(self.owner)['revision'],0)

    def test_failed_sources_are_not_automatically_retried_by_new_chat(self):
        self.enable();self.enqueue();job=self.store.claim(1120)
        self.store.finish(job['id'],job['epoch'],{'error':'failed'})
        new=self.enqueue('我习惯看例子',1200)
        self.assertEqual(new['start_ordinal'],3)
        self.assertEqual(len(new['input']['messages']),1)

    def test_large_pending_range_is_batched_without_losing_remainder(self):
        from memory_service import estimate_tokens
        self.enable()
        text='我喜欢简短回答。'+'测试'*2300
        self.assertLessEqual(estimate_tokens(text),4000)
        self.assertGreater(estimate_tokens(text)*2,4000)
        self.enqueue(text);self.enqueue(text,1001)
        first=self.store.claim(1121)
        self.assertEqual(len(first['input']['messages']),1)
        self.store.apply(first['id'],first['epoch'],{'additions':[],'suggestions':[],'skipped':[]},None)
        second=self.store.claim(1121)
        self.assertIsNotNone(second)
        self.assertEqual(second['input']['messages'][0]['ordinal'],3)
        self.assertLessEqual(sum(estimate_tokens(m['content']) for m in second['input'].get('context',[])),1000)

    def test_batch_claim_and_new_question_follow_same_conversation_lock_order(self):
        from concurrent.futures import ThreadPoolExecutor
        from psycopg2 import sql
        from memory_service import estimate_tokens
        self.enable();text='我喜欢简短回答。'+'测试'*2300
        self.enqueue(text);self.enqueue(text,1001)
        # A request owns the conversation first. A claim must wait before
        # taking the pending job row that request.defer needs.
        with ThreadPoolExecutor(max_workers=1) as pool,self.store.cursor() as c:
            c.execute('SET LOCAL deadlock_timeout=\'100ms\'')
            c.execute(sql.SQL('SELECT id FROM {} WHERE id=%s FOR UPDATE').format(self.store.table('conversations')),(self.cid,))
            future=pool.submit(self.store.claim,1121)
            import time;time.sleep(.15)
            self.store.defer(c,self.owner,self.cid,1100)
            c.connection.commit()
            claimed=future.result(timeout=5)
        # The newly accepted request deferred the idle claim until1220.
        self.assertIsNone(claimed)


if __name__=='__main__':unittest.main()
