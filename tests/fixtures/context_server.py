"""Isolated real PostgreSQL/auth/graph HTTP fixture; never calls remote models."""
import sys
import atexit
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
from contextlib import asynccontextmanager
from types import SimpleNamespace
from threading import Event
from psycopg2 import sql
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.documents import Document
import test_web_login as login
from test_memory_service import Embeddings
from memory_service import MemoryService
from memory_storage import MemoryStore
from prepare_generation_tasks import prepare_tasks
from prepare_account_memory import prepare_memory
from prepare_conversation_context import prepare_context
from context_storage import ContextStore
from conversation_context import ContextService
from test_conversation_context import SummaryModel, item
from task_recovery import schema_sql, MIGRATION
from query_rewrite import RetrievalPlan
from web_langgraph import LangGraphStreamingRagTurn
from web_app import create_app
from password_security import hash_password
import uvicorn

port, root = int(sys.argv[1]), Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
fixture = login.WebLoginTests()
fixture.setUp()
atexit.register(fixture.doCleanups)
with fixture.db, fixture.db.cursor() as c:
    c.execute(sql.SQL('UPDATE {}.users SET password_hash=%s').format(sql.Identifier(fixture.schema)), (hash_password('MemoryTest123!'),))
prepare_tasks(fixture.db, apply=True, schema=fixture.schema)
with fixture.db, fixture.db.cursor() as c:
    table=lambda name: sql.Identifier(fixture.schema,name).as_string(c)
    for statement in schema_sql(table('generation_recovery'), table('generation_tasks'), table('checkpoint_cleanup')): c.execute(statement)
    c.execute(sql.SQL('INSERT INTO {}.schema_migrations VALUES (%s)').format(sql.Identifier(fixture.schema)), (MIGRATION,))
prepare_memory(fixture.db, apply=True, schema=fixture.schema)
prepare_context(fixture.db, apply=True, schema=fixture.schema)
contexts=ContextStore(schema=fixture.schema, **fixture.settings)
summary_model=SummaryModel([item(1,'合同期限两年')])
context_service=ContextService(summary_model)
owner=fixture.auth.login('demo_ab','MemoryTest123!').user.id
cid=fixture.store.create_conversation(owner).id
fixture.store.save_complete_conversation_turn(owner,cid,'合同期限两年','旧助手判断')
fixture.store.save_complete_conversation_turn(owner,cid,'学习背景'*1500,'回答')
for n in range(4): fixture.store.save_complete_conversation_turn(owner,cid,f'近期问题{n}','回答')
memories=MemoryStore(schema=fixture.schema, **fixture.settings)
gate=Event(); gate.set()

class Vectors(Embeddings):
    fail=False
    def embed_query(self, question):
        if self.fail: raise TimeoutError('fixture selection')
        return super().embed_query(question)

class Command:
    calls=0
    def invoke(self, prompt):
        self.calls+=1
        return SimpleNamespace(content='{"action":"add","layer":"core","content":"详细解释"}')

service=MemoryService(Vectors(), Command(), fingerprint='fixture-v1')

class Rewriter:
    last_history=''
    def rewrite(self, question, history, **options):
        self.last_history='\n'.join(m.content for m in history)
        return RetrievalPlan(question, include_guide=False)
class Retriever:
    def invoke(self, state): return [Document(page_content='隔离测试法律资料', metadata={'knowledge_base_id':'A','index_status':'active','title':'测试法条'})]
class Reranker:
    def invoke(self, state): return state['candidates']
class Prompt:
    def invoke(self, values): return values
class Model:
    def stream(self, prompt):
        if not gate.wait(15): raise TimeoutError('fixture answer')
        yield '依据测试法条回答。保留条件：' + ('合同期限两年' if '合同期限两年' in rewriter.last_history else '无早期条件')

try:
    with SqliteSaver.from_conn_string(str(root/'graph.db')) as saver:
        rewriter=Rewriter()
        turn=LangGraphStreamingRagTurn(checkpointer=saver, rewriter=rewriter, retriever=Retriever(),
            reranker=Reranker(), prompt=Prompt(), model=Model(), history_turns=4, require_authorization=True)
        @asynccontextmanager
        async def lifespan(app):
            app.state.generation_runner()
            try: yield
            finally: gate.set(); app.state.generation_runner().shutdown()
        app=create_app(fixture.store, turn, authentication=fixture.auth, memory_store=memories, memory_service=service,context_store=contexts,
            background_tasks=True, recovery_config={'scope_resolver':fixture.accounts.allowed_knowledge_bases,'memory_service':service,'context_service':context_service},
            frontend_directory=Path('frontend/dist'), lifespan=lifespan)
        @app.get('/fixture')
        def ready(): return {'ready':True,'summary_calls':summary_model.calls}
        @app.post('/fixture/fail')
        def fail(): service.embeddings.fail=True; return {'ready':True}
        @app.post('/fixture/hold')
        def hold(): gate.clear(); return {'ready':True}
        @app.post('/fixture/release')
        def release(): gate.set(); return {'ready':True}
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
        @app.post('/fixture/stop')
        def stop(): gate.set(); server.should_exit=True; return {'stopping':True}
        server.run()
finally:
    atexit.unregister(fixture.doCleanups)
    fixture.doCleanups()
