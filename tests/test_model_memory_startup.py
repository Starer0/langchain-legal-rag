"""Exercise the real startup proxy, recovery runner and PG checkpointer together."""
import json
import os
import time
import unittest
from unittest.mock import patch
from psycopg2 import sql
from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
import test_web_login as login_fixture
import test_web_rag as rag_fixture


class StartupMemoryTests(unittest.TestCase):
    def scope_factory(self,**options):
        from web_understanding import TurnUnderstanding
        from test_web_understanding import plan
        turn=self.factory(**options)
        class Model:
            def bind_tools(self,schemas):return self
            def invoke(self,prompt):
                args=plan('out_of_scope',False)
                args['reply_plan']='直接给出Webhook配置步骤。'
                return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=args)])
        turn.understanding=TurnUnderstanding(Model())
        return turn

    def test_out_of_scope_is_not_background_learning_source_with_automation_enabled(self):
        from web_app import create_default_app
        from conversation_context import ContextService
        f=self.fixture
        env={'WEB_STORAGE_BACKEND':'postgres','WEB_FRONTEND':'react','WEB_POSTGRES_SCHEMA':f.schema,'MODEL_MEMORY_TOOLS_ENABLED':'true'}
        with patch.dict(os.environ,env),patch('web_app.create_web_rag_turn',side_effect=self.scope_factory),\
             patch('memory_api.create_memory_service',return_value=self.service),\
             patch('conversation_context.create_context_service',return_value=ContextService(None)):
            app=create_default_app()
            with TestClient(app) as client:
                logged=client.post('/api/auth/login',json={'username':'demo_ab','password':f.credentials['demo_ab']['password']})
                self.assertEqual(logged.status_code,200)
                client.headers['X-CSRF-Token']=logged.json()['csrf_token']
                with f.db,f.db.cursor() as c:
                    c.execute(sql.SQL('SELECT id FROM {} WHERE username=%s').format(sql.Identifier(f.schema,'users')),('demo_ab',))
                    owner=c.fetchone()[0]
                app.state.reflection_store.set_policy(owner,True,0)
                cid=client.post('/api/conversations').json()['id']
                result=self.send(client,cid,'n8n怎么把处理结果返回调用方？','scope')
                self.assertIn('超出',result['answer'])
                self.assertNotIn('Webhook',result['answer'])
                self.assertEqual(client.get('/api/memory').json()['facts'],[])
                with app.state.reflection_store.cursor() as c:
                    self.assertEqual(app.state.reflection_store.sources(c,cid,1,2),[])
                self.assertEqual(app.state.reflection_store.jobs(owner,cid),[])

    setUpClass=login_fixture.WebLoginTests.setUpClass

    def setUp(self):
        self.fixture=login_fixture.WebLoginTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        from prepare_generation_tasks import prepare_tasks
        from prepare_web_recovery import prepare_recovery
        from prepare_account_memory import prepare_memory
        from prepare_conversation_context import prepare_context
        from prepare_memory_reflection import prepare_reflection
        from prepare_memory_tools import prepare_memory_tools
        prepare_tasks(f.db,apply=True,schema=f.schema)
        prepare_recovery(f.settings,apply=True,schema=f.schema)
        prepare_memory(f.db,apply=True,schema=f.schema)
        prepare_context(f.db,apply=True,schema=f.schema)
        prepare_reflection(f.db,apply=True,schema=f.schema)
        prepare_memory_tools(f.db,apply=True,schema=f.schema)
        def cleanup_graph():
            self.assertRegex(f.schema,r'^login_test_[a-f0-9]{32}$')
            with f.db,f.db.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(f.schema+'_graph')))
        self.addCleanup(cleanup_graph)
        from memory_service import MemoryService
        from test_memory_service import Embeddings
        self.service=MemoryService(Embeddings(),fingerprint='startup-fixture')
        self.calls=0

    def factory(self,**options):
        from web_langgraph import LangGraphStreamingRagTurn
        from web_understanding import TurnUnderstanding
        owner=self
        class Model:
            def bind_tools(self,schemas):return self
            def invoke(self,prompt):
                owner.calls+=1;payload=json.loads(prompt.rsplit('\n',1)[-1]);q=payload['question']
                mixed='试用期' in q
                outside='推荐几款游戏' in q
                content='涉及纠纷先提醒保留证据' if outside else ('术语附通俗解释' if mixed else ('流程按实际操作顺序分点说明' if '办理流程' in q else '先说适用条件，再列需要准备的材料'))
                sources=payload['context']['sources']
                reference=next((s['reference'] for s in sources if s['kind']=='current_input'),'missing-runtime-source')
                calls=[dict(name='plan_answer',id='p',args=dict(route='rag' if mixed else 'preference',
                    legal_question='试用期最长多久？' if mixed else '',retrieval_question='试用期最长期限' if mixed else '',
                    include_guide=False,reply_plan='收到。',memory_request=True,out_of_scope_request=outside))]
                existing=next((f for f in payload['memory'].get('facts',[]) if f['content']==content),None)
                op=dict(action='remember',layer='core',category='preference',basis='declared',content=content,
                    sources=[dict(kind='current_input',reference=reference,quote=q)],
                    target_ids=[existing['id']] if existing else [],relation='duplicate' if existing else 'new',delta='')
                calls.append(dict(name='update_memory',id='m',args={'operations':[op]}))
                if mixed:
                    prior=next((s for s in sources if s['kind']=='message' and '办理流程' in s['content']),None)
                    if prior:
                        calls[-1]['args']['operations'].append(dict(action='remember',layer='core',category='preference',basis='declared',
                            content='不应重新保存的旧流程偏好',sources=[dict(kind='message',reference=prior['reference'],quote=prior['content'])],
                            target_ids=[],relation='new',delta=''))
                return AIMessage(content='',tool_calls=calls)
        self.assertTrue(options['model_memory'])
        self.turn=LangGraphStreamingRagTurn(checkpointer=options['checkpointer'],understanding=TurnUnderstanding(Model()),
            require_authorization=True,rewriter=rag_fixture.FakeRewriter('unused'),retriever=rag_fixture.FakeRetriever([]),
            reranker=rag_fixture.FakeReranker([]),prompt=rag_fixture.FakePrompt(),model=rag_fixture.FakeStreamingModel([]),history_turns=4)
        return self.turn

    def send(self,client,cid,question,key):
        accepted=client.post(f'/api/conversations/{cid}/tasks',json={'question':question,'submission_key':key})
        self.assertEqual(accepted.status_code,202,accepted.text)
        tid=accepted.json()['id'];deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            result=client.get(f'/api/conversations/{cid}/tasks/{tid}').json()
            if result['status']!='running':break
            time.sleep(.05)
        self.assertEqual(result['status'],'completed',result)
        return result

    def test_real_startup_saves_and_deduplicates_with_frozen_decision_and_receipt(self):
        from web_app import create_default_app
        from conversation_context import ContextService
        f=self.fixture
        env={'WEB_STORAGE_BACKEND':'postgres','WEB_FRONTEND':'react','WEB_POSTGRES_SCHEMA':f.schema,'MODEL_MEMORY_TOOLS_ENABLED':'true'}
        with patch.dict(os.environ,env),patch('web_app.create_web_rag_turn',side_effect=self.factory),\
             patch('memory_api.create_memory_service',return_value=self.service),\
             patch('conversation_context.create_context_service',return_value=ContextService(None)):
            with TestClient(create_default_app()) as client:
                credentials=f.credentials['demo_ab']
                logged=client.post('/api/auth/login',json={'username':'demo_ab','password':credentials['password']})
                self.assertEqual(logged.status_code,200)
                client.headers['X-CSRF-Token']=logged.json()['csrf_token']
                cid=client.post('/api/conversations').json()['id']
                first=self.send(client,cid,'以后先说适用条件，再列需要准备的材料，请长期记住。','first')
                self.assertEqual(first['memory']['status'],'saved')
                self.assertEqual(first['answer'],'已保存长期记忆。')
                repeated=self.send(client,cid,'请记住同一偏好：先说适用条件，再列需要准备的材料。','repeat')
                self.assertEqual(repeated['memory']['status'],'noop')
                mixed=self.send(client,cid,'请记住术语附通俗解释；试用期最长多久？','mixed')
                self.assertEqual(mixed['memory']['status'],'saved')
                self.assertIn('没有足够依据',mixed['answer'])
                self.assertEqual(self.calls,3)
                with f.db,f.db.cursor() as c:
                    c.execute(sql.SQL('SELECT count(*) FROM {} WHERE task_id=ANY(%s)').format(sql.Identifier(f.schema,'generation_turn_decisions')),([first['id'],repeated['id'],mixed['id']],))
                    self.assertEqual(c.fetchone()[0],3)
                facts=client.get('/api/memory').json()['facts']
                self.assertEqual(len(facts),2)
                self.assertEqual(len(facts[0]['sources']),2)

    def test_four_turn_acceptance_keeps_current_memory_and_legal_answer(self):
        from web_app import create_default_app
        from conversation_context import ContextService
        f=self.fixture
        env={'WEB_STORAGE_BACKEND':'postgres','WEB_FRONTEND':'react','WEB_POSTGRES_SCHEMA':f.schema,'MODEL_MEMORY_TOOLS_ENABLED':'true'}
        questions=[
            '以后解释法律问题时，先说适用条件，再列我需要准备的材料。请长期记住这个偏好。',
            '请记住：讲法律问题时，先告诉我哪些条件适用，再告诉我要准备什么材料。',
            '刚才的偏好再补充一点：涉及办理流程时，按实际操作顺序分点说明，请记住。',
            '请记住：出现专业术语时，用一句通俗解释补充说明。另外，一年期劳动合同的试用期最长多久？']
        with patch.dict(os.environ,env),patch('web_app.create_web_rag_turn',side_effect=self.factory),\
             patch('memory_api.create_memory_service',return_value=self.service),\
             patch('conversation_context.create_context_service',return_value=ContextService(None)):
            with TestClient(create_default_app()) as client:
                logged=client.post('/api/auth/login',json={'username':'demo_ab','password':f.credentials['demo_ab']['password']})
                self.assertEqual(logged.status_code,200)
                client.headers['X-CSRF-Token']=logged.json()['csrf_token']
                cid=client.post('/api/conversations').json()['id']
                results=[self.send(client,cid,q,f'seq-{i}') for i,q in enumerate(questions)]
                self.assertEqual([r['memory']['status'] for r in results],['saved','noop','saved','saved'])
                self.assertIn('没有足够依据',results[-1]['answer'])
                facts=client.get('/api/memory').json()['facts']
                self.assertEqual(len(facts),3)
                self.assertFalse(any('不应重新保存' in f['content'] for f in facts))
                self.assertEqual(self.calls,4)

    def test_preference_and_outside_request_save_and_refuse_after_refresh(self):
        from web_app import create_default_app
        from conversation_context import ContextService
        f=self.fixture
        env={'WEB_STORAGE_BACKEND':'postgres','WEB_FRONTEND':'react','WEB_POSTGRES_SCHEMA':f.schema,'MODEL_MEMORY_TOOLS_ENABLED':'true'}
        with patch.dict(os.environ,env),patch('web_app.create_web_rag_turn',side_effect=self.factory),\
             patch('memory_api.create_memory_service',return_value=self.service),\
             patch('conversation_context.create_context_service',return_value=ContextService(None)):
            with TestClient(create_default_app()) as client:
                logged=client.post('/api/auth/login',json={'username':'demo_ab','password':f.credentials['demo_ab']['password']})
                self.assertEqual(logged.status_code,200)
                client.headers['X-CSRF-Token']=logged.json()['csrf_token']
                cid=client.post('/api/conversations').json()['id']
                question='以后涉及纠纷，请先提醒我保留证据，记住这个偏好。顺便推荐几款游戏。'
                first=self.send(client,cid,question,'mixed-scope')
                self.assertEqual(first['memory']['status'],'saved')
                self.assertIn('已保存长期记忆',first['answer'])
                self.assertIn('超出',first['answer'])
                refreshed=client.get(f"/api/conversations/{cid}/tasks/{first['id']}").json()
                self.assertEqual(refreshed['answer'],first['answer'])
                repeated=self.send(client,cid,question,'mixed-scope-repeat')
                self.assertEqual(repeated['memory']['status'],'noop')
                self.assertIn('超出',repeated['answer'])
                facts=client.get('/api/memory').json()['facts']
                self.assertEqual(len(facts),1)
                self.assertIn('保留证据',facts[0]['content'])
                self.assertNotIn('游戏',facts[0]['content'])
                self.assertEqual(len(facts[0]['sources']),2)
