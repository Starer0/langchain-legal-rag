"""Real HTTP graph recovery fixture, isolated files and fake models only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from threading import Event
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.documents import Document
from query_rewrite import RetrievalPlan
from web_langgraph import LangGraphStreamingRagTurn
from web_app import create_app
from web_storage import SQLiteConversationStore
import uvicorn

port, root, phase = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
root.mkdir(parents=True, exist_ok=True)
store = SQLiteConversationStore(root/'tasks.db')
store.ensure_session('fixture')
items = store.list_conversations('fixture')
cid = items[0].id if items else store.create_conversation('fixture').id
store.rename_conversation('fixture', cid, '恢复测试')
gate = Event()


def record(name):
    with (root/'calls').open('a') as f: f.write(name+'\n')


class Rewriter:
    def rewrite(self, question, history, **options): record('rewrite'); return RetrievalPlan(question, include_guide=False)


class Retriever:
    def invoke(self, state):
        record('retrieve')
        return [Document(page_content='恢复后的来源', metadata={'knowledge_base_id': 'A', 'index_status': 'active'})]


class Reranker:
    def invoke(self, state): record('rerank'); return state['candidates']


class Prompt:
    def invoke(self, values): return values


class Model:
    def stream(self, prompt):
        record('answer')
        yield '重启前的半段' if phase == 'initial' else '重新生成的内容'
        if not gate.wait(90): raise RuntimeError('fixture timeout')
        yield '，完整回答'


with SqliteSaver.from_conn_string(str(root/'graph.db')) as saver:
    turn = LangGraphStreamingRagTurn(checkpointer=saver, rewriter=Rewriter(), retriever=Retriever(),
        reranker=Reranker(), prompt=Prompt(), model=Model(), history_turns=3, require_authorization=True)
    @asynccontextmanager
    async def lifespan(app):
        app.state.generation_runner()
        try: yield
        finally:
            gate.set()
            app.state.generation_runner().shutdown()
    app = create_app(store, turn, background_tasks=True, lifespan=lifespan,
        recovery_config={'scope_resolver': lambda owner: {'A'}, 'lease': None})
    app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:8012'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    @app.get('/api/auth/me')
    def me(): return {'user': {'id': 'fixture', 'username': '测试账号'}, 'csrf_token': 'fixture'}
    @app.get('/fixture')
    def fixture(): return {'cid': cid, 'phase': phase, 'calls': (root/'calls').read_text().splitlines() if (root/'calls').exists() else []}
    @app.post('/fixture/release')
    def release(): gate.set(); return {'released': True}
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='error')
