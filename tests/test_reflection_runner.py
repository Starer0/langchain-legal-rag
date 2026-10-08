import os
import unittest
import json
from types import SimpleNamespace
import test_reflection_storage as storage
from memory_reflection import ReflectionService
from reflection_runner import ReflectionRunner
from generation_tasks import TaskStore


class Clock:
    def __init__(self):self.value=1000
    def now(self):return self.value
    def advance(self,seconds):self.value+=seconds


class ReflectionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS')!='1':raise unittest.SkipTest('Real temporary PG required')
    def setUp(self):
        self.fixture=storage.ReflectionStorageTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture;self.clock=Clock()
        self.owner,self.cid,self.store,self.memory=f.owner,f.cid,f.store,f.memory
        f.enable()
        self.calls=0
        class Model:
            def invoke(inner,prompt):
                self.calls+=1
                return SimpleNamespace(content=json.dumps({'candidates':[storage.candidate('我喜欢简短回答')]}))
        self.service=ReflectionService(Model(),f.service)
        self.runner=ReflectionRunner(self.store,self.service,clock=self.clock)
        self.addCleanup(self.runner.shutdown)

    def test_review_trace_links_model_to_job_without_copying_memory_into_parent(self):
        from langsmith import Client,tracing_context,traceable
        from langsmith.run_helpers import get_current_run_tree
        seen=[];parents=[]
        @traceable(name='synthetic_review_model',run_type='llm')
        def model_call(prompt):
            seen.append(get_current_run_tree())
            self.calls+=1
            return SimpleNamespace(content=json.dumps({'candidates':[]}))
        def invoke(prompt):
            parents.append(get_current_run_tree())
            return model_call(prompt)
        self.service.model.invoke=invoke
        self.fixture.enqueue();self.clock.advance(120)
        job=self.store.jobs(self.owner,self.cid)[0]
        with tracing_context(enabled='local',client=Client(api_key='test-only-not-a-real-key')):
            self.assertTrue(self.runner.pump())
        self.assertEqual(self.calls,1)
        child=seen[0];parent=parents[0]
        self.assertIsNotNone(parent)
        self.assertEqual(child.parent_run_id,parent.id)
        self.assertEqual(parent.name,'memory_review')
        self.assertEqual(parent.extra['metadata']['job_id'],job['id'])
        self.assertEqual(parent.extra['metadata']['conversation_id'],self.cid)
        self.assertEqual(parent.inputs,{})
        self.assertEqual(parent.outputs,{'model_calls':1})
        self.assertIn('background_memory',parent.tags)
        self.assertNotIn('core_text',parent.extra['metadata'])
        self.assertEqual(self.store.jobs(self.owner,self.cid)[0]['status'],'completed')

    def test_completion_registers_job_atomically_and_deduplicates(self):
        tasks=TaskStore(self.fixture.f.store,memory=self.memory,reflection=self.store,clock=self.clock.now)
        task,_=tasks.create(self.owner,self.cid,'我喜欢简短回答','key','request')
        self.assertTrue(tasks.complete(task['id'],'回答',[]))
        self.assertFalse(tasks.complete(task['id'],'回答',[]))
        self.assertEqual(len(self.store.jobs(self.owner,self.cid)),1)
        self.assertFalse(self.runner.pump())
        self.clock.advance(120)
        self.assertTrue(self.runner.pump())
        self.assertEqual(self.memory.read(self.owner)['core_text'],'我喜欢简短回答')
        self.assertEqual(self.calls,1)

    def test_runner_restart_preserves_due_job(self):
        self.fixture.enqueue();self.runner.shutdown();self.clock.advance(120)
        restarted=ReflectionRunner(self.store,self.service,clock=self.clock)
        self.addCleanup(restarted.shutdown)
        self.assertTrue(restarted.pump())
        self.assertEqual(self.memory.read(self.owner)['revision'],1)

    def test_preference_graph_completion_still_enqueues_one_background_job(self):
        import test_web_rag as f
        from web_langgraph import LangGraphStreamingRagTurn
        fixture=f.StreamingRagTurnTests();fixture.setUp()
        turn=LangGraphStreamingRagTurn(rewriter=fixture.rewriter,retriever=fixture.retriever,
            reranker=fixture.reranker,prompt=fixture.prompt,model=fixture.model,history_turns=4)
        tasks=TaskStore(self.fixture.f.store,memory=self.memory,reflection=self.store,clock=self.clock.now)
        question='我喜欢简短回答'
        task,_=tasks.create(self.owner,self.cid,question,'intent-key','intent-request')
        done=list(turn.stream(question,[],memory_service=self.fixture.service,memory_input=self.memory.read(self.owner)))[-1]['data']
        self.assertEqual(done['sources'],[])
        self.assertTrue(tasks.complete(task['id'],done['answer'],done['sources']))
        self.assertEqual(fixture.retriever.states,[])
        self.assertEqual(len(self.store.jobs(self.owner,self.cid)),1)
        self.assertEqual(self.memory.read(self.owner)['revision'],0)
        self.clock.advance(120);self.assertTrue(self.runner.pump())
        self.assertEqual(self.memory.read(self.owner)['core_text'],question)

    def test_active_question_postpones_unstarted_reflection(self):
        self.fixture.enqueue()
        tasks=TaskStore(self.fixture.f.store,memory=self.memory,reflection=self.store,clock=self.clock.now)
        self.clock.advance(100)
        task,_=tasks.create(self.owner,self.cid,'新问题','key','request')
        self.clock.advance(20)
        self.assertFalse(self.runner.pump())
        tasks.complete(task['id'],'回答',[])
        self.clock.advance(120)
        self.assertTrue(self.runner.pump())

    def test_extraction_failure_terminal_until_manual_retry(self):
        self.fixture.enqueue();self.clock.advance(120)
        class Failure:
            def invoke(self,prompt):raise TimeoutError('provider')
        self.service.model=Failure()
        self.assertTrue(self.runner.pump())
        self.assertEqual(self.store.jobs(self.owner,self.cid)[0]['status'],'failed')
        self.assertFalse(self.runner.pump())
        self.assertEqual(self.memory.read(self.owner)['revision'],0)

    def test_no_new_memory_means_no_revision_increment(self):
        self.fixture.enqueue('劳动法呢');self.clock.advance(120)
        self.runner.pump()
        self.assertEqual(self.memory.read(self.owner)['revision'],0)

    def test_capacity_failure_preserves_text_and_publishes_suggestion(self):
        old=self.memory.read(self.owner)
        saved=self.memory.save(self.owner,self.fixture.service.prepare(old,'x'*1495,'',True),0)
        self.fixture.enqueue();self.clock.advance(120);self.runner.pump()
        self.assertEqual(self.memory.read(self.owner)['core_text'],saved['core_text'])
        self.assertEqual(self.memory.read(self.owner)['revision'],1)
        self.assertEqual(self.store.suggestions(self.owner)[0]['proposal']['relation'],'capacity')

    def test_real_clock_worker_processes_due_durable_job_after_restart(self):
        import time
        self.fixture.enqueue(now=time.time()-120)
        self.runner.shutdown()
        worker=ReflectionRunner(self.store,self.service);self.addCleanup(worker.shutdown);worker.start()
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if self.memory.read(self.owner)['revision']==1:break
            time.sleep(.01)
        self.assertEqual(self.memory.read(self.owner)['revision'],1)

    def test_failure_after_memory_write_rolls_back_job_progress_and_version(self):
        from unittest.mock import patch
        self.fixture.enqueue();job=self.store.claim(1120)
        checked={'additions':[storage.candidate('我喜欢简短回答')],'suggestions':[],'skipped':[]}
        prepared=self.service.prepare(self.memory.read(self.owner),checked)
        save=self.memory.save
        def fail(*args,**kwargs):save(*args,**kwargs);raise RuntimeError('after memory write')
        with patch.object(self.memory,'save',side_effect=fail),self.assertRaises(RuntimeError):self.store.apply(job['id'],job['epoch'],checked,prepared)
        self.assertEqual(self.memory.read(self.owner)['revision'],0)
        self.assertEqual(self.store.jobs(self.owner,self.cid)[0]['status'],'running')


class ReviewTraceFailureTests(unittest.TestCase):
    def test_unavailable_tracer_does_not_skip_or_repeat_model(self):
        from unittest.mock import patch
        runner=ReflectionRunner(None,None)
        calls=[]
        job={'id':'j','conversation_id':'c','reason':'idle'}
        def operation():calls.append('model');return 'result'
        with patch('langsmith.trace',side_effect=OSError('telemetry unavailable')):
            self.assertEqual(runner._review_model(job,operation),'result')
        self.assertEqual(calls,['model'])

    def test_failed_model_has_sanitized_parent_error_and_no_retry(self):
        from langsmith import Client,tracing_context
        from langsmith.run_helpers import get_current_run_tree
        runner=ReflectionRunner(None,None);seen=[]
        def operation():
            seen.append(get_current_run_tree())
            raise TimeoutError('SYNTHETIC_PRIVATE_ERROR')
        with tracing_context(enabled='local',client=Client(api_key='test-only-not-a-real-key')):
            with self.assertRaises(TimeoutError):
                runner._review_model({'id':'j','conversation_id':'c','reason':'idle'},operation)
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0].error,'TimeoutError')
        self.assertNotIn('SYNTHETIC_PRIVATE_ERROR',str(seen[0].model_dump()))


if __name__=='__main__':unittest.main()
