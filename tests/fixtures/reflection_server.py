"""Isolated fake-model HTTP server with real durable PG, restartable by E2E."""
import sys,json,time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[1])]
from types import SimpleNamespace
from contextlib import asynccontextmanager
import psycopg2
from psycopg2 import sql
import uvicorn
import test_web_login as login
from prepare_generation_tasks import prepare_tasks
from prepare_account_memory import prepare_memory
from prepare_memory_reflection import prepare_reflection
from prepare_conversation_context import prepare_context
from memory_storage import MemoryStore
from memory_service import MemoryService
from test_memory_service import Embeddings
from reflection_storage import ReflectionStore
from memory_reflection import ReflectionService
from reflection_runner import ReflectionRunner
from conversation_context import ContextService
from context_storage import ContextStore
from langgraph.checkpoint.sqlite import SqliteSaver
from web_langgraph import LangGraphStreamingRagTurn
from query_rewrite import RetrievalPlan
from web_app import create_app
from task_recovery import schema_sql,MIGRATION
from password_security import hash_password

port,root=int(sys.argv[1]),Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=True)
semantic=len(sys.argv)>3 and sys.argv[3]=='semantic'
state=root/'fixture.json'
if state.exists():
    data=json.loads(state.read_text())
    if not data['schema'].startswith('login_test_') or len(data['schema'])!=43:raise ValueError('Invalid test schema')
    from account_storage import PostgresAccountStore
    from login_sessions import PostgresLoginSessions
    from postgres_storage import PostgresUserConversationStore
    settings=dict(host='127.0.0.1',port=5433,dbname='legal_rag',user='legal_rag',password=Path('.postgres-password').read_text().strip())
    f=SimpleNamespace(schema=data['schema'],settings=settings,db=psycopg2.connect(**settings))
    f.accounts=PostgresAccountStore(schema=f.schema,**settings)
    f.auth=PostgresLoginSessions(f.accounts,now=login.Clock(),schema=f.schema,**settings)
    f.store=PostgresUserConversationStore(schema=f.schema,**settings)
else:
    f=login.WebLoginTests();f.setUp();data={'schema':f.schema,'clock':1000}
    with f.db,f.db.cursor() as c:
        c.execute(sql.SQL('UPDATE {}.users SET password_hash=%s').format(sql.Identifier(f.schema)),(hash_password('MemoryTest123!'),))
    prepare_tasks(f.db,apply=True,schema=f.schema)
    with f.db,f.db.cursor() as c:
        table=lambda name:sql.Identifier(f.schema,name).as_string(c)
        for statement in schema_sql(table('generation_recovery'),table('generation_tasks'),table('checkpoint_cleanup')):c.execute(statement)
        c.execute(sql.SQL('INSERT INTO {}.schema_migrations VALUES (%s)').format(sql.Identifier(f.schema)),(MIGRATION,))
    prepare_memory(f.db,apply=True,schema=f.schema);prepare_context(f.db,apply=True,schema=f.schema);prepare_reflection(f.db,apply=True,schema=f.schema)

class Clock:
    value=data['clock']
    def now(self):return self.value
clock=Clock();memories=MemoryStore(schema=f.schema,**f.settings)
reflection=ReflectionStore(memory=memories,schema=f.schema,**f.settings);memories.reflection=reflection
service=MemoryService(Embeddings(),fingerprint='fixture')
if semantic:
    from prepare_memory_tools import prepare_memory_tools
    from memory_fact_storage import MemoryFactStore
    from memory_tools import MemoryTools
    prepare_memory_tools(f.db,apply=True,schema=f.schema)
    memories.facts=MemoryFactStore(memories);memories.tools=MemoryTools(service,memories.facts)
class Extractor:
    calls=0
    def invoke(self,prompt):
        self.calls+=1;values=json.loads(prompt.rsplit('\n',1)[-1]);candidates=[]
        for m in values['user_messages']:
            if m['content'] in ('我喜欢简短回答','我喜欢详细解释'):
                candidates.append(dict(layer='core',text=m['content'],source_ordinals=[m['ordinal']],source_quotes=[m['content']],relation='new',target=''))
        return SimpleNamespace(content=json.dumps({'candidates':candidates}))
extractor=Extractor();worker=ReflectionRunner(reflection,ReflectionService(extractor,service),clock=clock)
class SemanticModel:
    calls=0
    def bind_tools(self,tools):return self
    @staticmethod
    def operation(content,refs,**changes):
        return dict(action='remember',layer='core',category='preference',basis='declared',content=content,
            sources=[dict(kind=s['kind'],reference=s['reference'],quote=s['content']) for s in refs],
            target_ids=[],relation='new',delta='',**changes)
    def invoke(self,prompt):
        from langchain_core.messages import AIMessage
        self.calls+=1;payload=json.loads(prompt.rsplit('\n',1)[-1]);calls=[]
        if 'question' in payload:
            q=payload['question'];explicit='请记住' in q or '清空' in q
            outside='推荐几款游戏' in q
            explicit=explicit or (outside and '记住这个偏好' in q)
            route='rag' if '试用期' in q or '劳动仲裁' in q else ('preference' if explicit else ('out_of_scope' if 'n8n' in q else 'general'))
            answer=dict(route=route,legal_question='试用期最长多久？' if route=='rag' else '',
                retrieval_question='试用期最长期限' if route=='rag' else '',include_guide=False,reply_plan='收到这条测试消息。',memory_request=explicit,out_of_scope_request=outside)
            calls.append(dict(name='plan_answer',id='p',args=answer))
            refs=[s for s in payload['context']['sources'] if s['kind']=='current_input']
            if explicit:
                content='涉及纠纷先提醒保留证据' if outside else '先给结论'
                op=self.operation(content,refs)
                if '清空' in q:op.update(action='request_clear',content='')
                else:
                    target=next((f for f in payload['memory'].get('facts',[]) if f['content']==content),None)
                    if target:op.update(relation='duplicate',target_ids=[target['id']])
                calls.append(dict(name='update_memory',id='m',args={'operations':[op]}))
        else:
            refs=[s for s in payload['existing']['review_sources'] if s.get('new') and '劳动仲裁' in s['content']]
            if len(refs)>=2:
                op=self.operation('近期在学习劳动仲裁',refs);op.update(layer='extended',category='learning_focus',basis='inferred')
                calls.append(dict(name='update_memory',id='b',args={'operations':[op]}))
        return AIMessage(content='',tool_calls=calls)
if semantic:
    from web_understanding import TurnUnderstanding
    semantic_model=SemanticModel();extractor=semantic_model
    worker=ReflectionRunner(reflection,ReflectionService(semantic_model,service),clock=clock)
class Rewriter:
    def rewrite(self,question,history,**kwargs):return RetrievalPlan(question,include_guide=False)
class Retriever:
    def invoke(self,state):return []
class Reranker:
    def invoke(self,state):return state['candidates']
keep=False
try:
    with SqliteSaver.from_conn_string(str(root/'graph.db')) as saver:
        turn=LangGraphStreamingRagTurn(checkpointer=saver,rewriter=Rewriter(),retriever=Retriever(),reranker=Reranker(),prompt=object(),model=object(),history_turns=4,require_authorization=True,
            understanding=TurnUnderstanding(semantic_model) if semantic else None)
        contexts=ContextStore(schema=f.schema,**f.settings);contexts.semantic=semantic
        @asynccontextmanager
        async def lifespan(app):
            app.state.generation_runner();worker.start()
            try:yield
            finally:worker.shutdown();app.state.generation_runner().shutdown()
        app=create_app(f.store,turn,authentication=f.auth,memory_store=memories,memory_service=service,context_store=contexts,reflection_store=reflection,
            background_tasks=True,recovery_config={'scope_resolver':f.accounts.allowed_knowledge_bases,'memory_service':service,'context_service':ContextService(None)},frontend_directory=Path('frontend/dist'),lifespan=lifespan)
        # Use the same clock as the actual completion/outbox path.
        app.state.generation_runner().tasks.clock=clock.now
        @app.get('/fixture')
        def ready():
            with memories.cursor() as c:
                c.execute(sql.SQL("SELECT count(*) FROM {} WHERE status='running'").format(memories.table('generation_tasks')))
                running=c.fetchone()[0]
            return {'ready':True,'calls':extractor.calls,'running':running}
        @app.post('/fixture/advance')
        def advance(seconds:int):clock.value+=seconds;worker.wake();return {'now':clock.value}
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
        @app.post('/fixture/stop')
        def stop(preserve:bool=False):
            global keep
            keep=preserve;server.should_exit=True;return {'stopping':True}
        server.run()
finally:
    if keep:state.write_text(json.dumps({'schema':f.schema,'clock':clock.value}))
    else:
        f.db.rollback()
        with f.db,f.db.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(f.schema)))
    f.db.close()
    if hasattr(f,'directory'):f.directory.cleanup()
