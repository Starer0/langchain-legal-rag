"""Disposable child process for real crash/reopen tests; no remote services."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import time
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from query_rewrite import RetrievalPlan
from web_storage import SQLiteConversationStore
from generation_tasks import TaskStore
from task_recovery import RecoveryStore
from recovery_runner import RecoveryRunner
from web_langgraph import LangGraphStreamingRagTurn

root, phase = Path(sys.argv[1]), sys.argv[2]


def record(stage):
    with (root/'calls').open('a', encoding='utf-8') as f: f.write(stage + '\n')


def block():
    (root/'blocked').touch()
    while True: time.sleep(.1)


doc = Document(page_content='合法来源', metadata={'knowledge_base_id': 'A', 'index_status': 'active'})


class Rewriter:
    def rewrite(self, question, history, **options):
        record('rewrite')
        return RetrievalPlan('检索问题', include_guide=False)


class Retriever:
    def invoke(self, state): record('retrieve'); return [doc]


class Reranker:
    def invoke(self, state): record('rerank'); return [doc]


class Prompt:
    def invoke(self, state): return state


class Model:
    def stream(self, prompt):
        record('answer')
        if phase == 'before_answer': block()
        if phase == 'partial_answer':
            yield '旧半段'
            # Allow the observer to persist the delta before the parent kills us.
            time.sleep(.1)
            block()
        yield '新的完整回答'


store = SQLiteConversationStore(root/'tasks.db')
store.ensure_session('owner')
conversations = store.list_conversations('owner')
cid = conversations[0].id if conversations else store.create_conversation('owner').id
tasks = TaskStore(store)
recovery = RecoveryStore(tasks)
if phase == 'before_save':
    original = tasks.complete
    def complete(*args, **kwargs): block()
    tasks.complete = complete

with SqliteSaver.from_conn_string(str(root/'graph.db')) as saver:
    turn = LangGraphStreamingRagTurn(checkpointer=saver, rewriter=Rewriter(), retriever=Retriever(),
        reranker=Reranker(), prompt=Prompt(), model=Model(), history_turns=3, require_authorization=True)
    runner = RecoveryRunner(tasks, turn, lambda owner: {'A'})
    runner.recover()
    if phase != 'resume': runner.start('owner', cid, '问题', 'key', 'request', {'allowed_knowledge_bases': {'A'}})
    deadline = time.monotonic() + 15
    while tasks.active('owner') and time.monotonic() < deadline: time.sleep(.03)
    runner.shutdown()
    task = tasks.latest('owner', cid)
    if not task or task['status'] != 'completed': raise RuntimeError('Fixture task failed')
    (root/'result.json').write_text(json.dumps({'answer': task['answer'], 'messages': len(store.load_conversation_messages('owner', cid)), 'sources': task['sources']}, ensure_ascii=False), encoding='utf-8')
