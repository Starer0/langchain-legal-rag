"""Local browser fixture: real task APIs, no remote model or production data."""
import sys
import tempfile
from pathlib import Path
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from web_app import create_app
from web_storage import SQLiteConversationStore
from web_langgraph import LangGraphStreamingRagTurn
from query_rewrite import RetrievalPlan
from langchain_core.documents import Document

directory = tempfile.TemporaryDirectory()
store = SQLiteConversationStore(Path(directory.name) / 'fixture.db')
store.ensure_session('fixture')
a = store.create_conversation('fixture')
b = store.create_conversation('fixture')
store.rename_conversation('fixture', a.id, 'A对话')
store.rename_conversation('fixture', b.id, 'B对话')
store.save_complete_conversation_turn('fixture', b.id, 'B旧问题', 'B旧记录')
gate = Event()
calls = 0

class Model:
    def stream(self, prompt):
        global calls
        calls += 1
        yield '后台正在生成的内容'
        if not gate.wait(90): raise RuntimeError('fixture timeout')
        yield '，最终完整回答'

class Rewriter:
    def rewrite(self, question, history, **options): return RetrievalPlan(question, include_guide=False)

class Retriever:
    def invoke(self, state): return [Document(page_content='后台任务来源', metadata={'knowledge_base_id': 'A'})]

class Reranker:
    def invoke(self, state): return state['candidates']

class Prompt:
    def invoke(self, values): return values

turn = LangGraphStreamingRagTurn(rewriter=Rewriter(), retriever=Retriever(), reranker=Reranker(),
                                prompt=Prompt(), model=Model(), history_turns=4)
app = create_app(store, turn, background_tasks=True)
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:8012'], allow_credentials=True,
                   allow_methods=['*'], allow_headers=['*'])

@app.get('/api/auth/me')
def me(): return {'user': {'id': 'fixture', 'username': '测试账号'}, 'csrf_token': 'fixture'}

@app.get('/fixture')
def fixture(): return {'a': a.id, 'b': b.id, 'calls': calls}

@app.post('/fixture/release')
def release():
    gate.set()
    return {'calls': calls}

uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[1]), log_level='error')
