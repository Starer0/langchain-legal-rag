import unittest
from types import SimpleNamespace
from langchain_core.documents import Document
from memory_service import MemoryService
from test_memory_service import Embeddings
import test_web_rag as f


class WebMemoryTests(unittest.TestCase):
    def setUp(self):
        from web_langgraph import LangGraphStreamingRagTurn
        self.prompt = f.FakePrompt()
        self.rewriter = f.FakeRewriter('法律检索问题')
        self.retriever = f.FakeRetriever([Document(page_content='法律依据', metadata={})])
        self.turn = LangGraphStreamingRagTurn(rewriter=self.rewriter, retriever=self.retriever,
            reranker=f.FakeReranker(self.retriever.documents), prompt=self.prompt, model=f.FakeStreamingModel(['回答']), history_turns=4)
        self.service = MemoryService(Embeddings(), fingerprint='fixture-v1')
        self.snapshot = self.service.prepare(dict(revision=1, entries=[]), '中文先结论', 'Python初学者\n\n劳动法背景', True)

    def test_selected_memory_reaches_only_answer_prompt(self):
        events = list(self.turn.stream('函数报错', [], memory_input=self.snapshot, memory_service=self.service))
        self.assertEqual(self.prompt.values[0]['memory'], '核心记忆：\n中文先结论\n\n相关背景：\nPython初学者')
        self.assertNotIn('memory', self.retriever.states[0])
        self.assertEqual(self.rewriter.calls[0][0], '函数报错')
        self.assertEqual(events[-1]['data']['memory']['revision'], 1)

    def test_failed_selection_is_visible_without_polluting_answer(self):
        done = list(self.turn.stream('失败', [], memory_input=self.snapshot, memory_service=self.service))[-1]['data']
        self.assertEqual(done['answer'], '回答')
        self.assertEqual(done['memory']['warning'], '本次未使用扩展记忆')

    def test_memory_trace_contains_ids_not_private_memory(self):
        import logging
        from rag_logging import RagRequestTrace
        trace = RagRequestTrace(request_id='req', user_id='owner', conversation_id='c', allowed_knowledge_bases={'A'}, logger=logging.getLogger('memory-test'))
        self.retriever.documents[0].metadata.update(knowledge_base_id='A', index_status='active')
        with self.assertLogs('memory-test', level='INFO') as logs:
            result = list(self.turn.stream('函数报错', [], allowed_knowledge_bases={'A'}, trace=trace, memory_input=self.snapshot, memory_service=self.service))
        self.assertEqual(result[-1]['data']['answer'], '回答')
        selected = [line for line in logs.output if 'memory_selected' in line]
        self.assertEqual(len(selected), 1)
        self.assertNotIn('Python初学者', selected[0])

    def test_command_emits_proposal_but_never_saved_success(self):
        class Model:
            def invoke(self, prompt): return SimpleNamespace(content='{"action":"add","layer":"core","content":"保留来源"}')
        self.service.command_model = Model()
        events = list(self.turn.stream('记住，保留来源', [], memory_input=self.snapshot, memory_service=self.service))
        self.assertEqual(self.rewriter.calls, [])
        self.assertEqual([e for e in events if e['event'] == 'delta'], [])
        self.assertIn('memory_proposal', events[-1]['data'])
        self.assertEqual(events[-1]['data']['sources'], [])

    def test_command_text_not_in_logs_or_subsequent_rewrite_history(self):
        import logging
        from rag_logging import RagRequestTrace
        from langchain_core.messages import HumanMessage, AIMessage
        class Model:
            def invoke(self, prompt): return SimpleNamespace(content='{"action":"add","layer":"core","content":"私人偏好"}')
        self.service.command_model = Model()
        trace = RagRequestTrace(request_id='req', user_id='owner', conversation_id='c', allowed_knowledge_bases={'A'}, logger=logging.getLogger('memory-private'))
        with self.assertLogs('memory-private', level='INFO') as logs:
            list(self.turn.stream('记住私人偏好', [], allowed_knowledge_bases={'A'}, trace=trace, memory_input=self.snapshot, memory_service=self.service))
        self.assertNotIn('私人偏好', '\n'.join(logs.output))
        list(self.turn.stream('劳动法', [HumanMessage('记住私人偏好'), AIMessage('已保存')], memory_input=self.snapshot, memory_service=self.service))
        self.assertEqual(self.rewriter.calls[-1][1], [])

    def test_no_evidence_never_uses_memory_as_legal_basis(self):
        self.turn.require_authorization = True
        self.turn.retriever = f.FakeRetriever([])
        self.turn.reranker = f.FakeReranker([])
        events = list(self.turn.stream('劳动问题', [], allowed_knowledge_bases={'A'}, memory_input=self.snapshot, memory_service=self.service))
        self.assertIn('没有足够依据', events[-1]['data']['answer'])
        self.assertEqual(self.service.embeddings.queries, [])

    def test_selection_survives_checkpoint_recovery(self):
        import tempfile
        from pathlib import Path
        from langgraph.checkpoint.sqlite import SqliteSaver
        from web_langgraph import LangGraphStreamingRagTurn
        with tempfile.TemporaryDirectory() as root, SqliteSaver.from_conn_string(str(Path(root)/'graph.db')) as saver:
            self.turn = LangGraphStreamingRagTurn(checkpointer=saver, rewriter=self.rewriter,
                retriever=self.retriever, reranker=self.turn.reranker, prompt=self.prompt, model=self.turn.model, history_turns=4)
            self.turn.graph.invoke  # real graph, no mock of checkpoint boundaries
            # Persist selection before an answer failure, then restart that node only.
            class Failure:
                def stream(self, prompt): raise RuntimeError('answer failure')
            self.turn.model = Failure()
            with self.assertRaises(RuntimeError):
                list(self.turn.stream('函数报错', [], task_id='task', memory_input=self.snapshot, memory_service=self.service))
            self.turn.model = f.FakeStreamingModel(['恢复回答'])
            result = list(self.turn.stream('函数报错', [], task_id='task', resume=True,
                memory_input=self.snapshot, memory_service=self.service))[-1]['data']
            self.assertEqual(result['answer'], '恢复回答')
            self.assertEqual(self.service.embeddings.queries, ['函数报错'])

    def test_changed_memory_blocks_unfinished_checkpoint(self):
        # The guard runs only for unfinished resumed graphs, not completed results.
        import tempfile
        from pathlib import Path
        from langgraph.checkpoint.sqlite import SqliteSaver
        from web_langgraph import LangGraphStreamingRagTurn
        with tempfile.TemporaryDirectory() as root, SqliteSaver.from_conn_string(str(Path(root)/'graph.db')) as saver:
            turn = LangGraphStreamingRagTurn(checkpointer=saver, rewriter=self.rewriter,
                retriever=self.retriever, reranker=self.turn.reranker, prompt=self.prompt, model=self.turn.model, history_turns=4)
            class Failure:
                def stream(self, prompt): raise RuntimeError('answer failed')
            turn.model = Failure()
            with self.assertRaises(RuntimeError): list(turn.stream('函数', [], task_id='t', memory_input=self.snapshot, memory_service=self.service))
            def denied(): raise ValueError('Memory changed')
            with self.assertRaisesRegex(ValueError, 'Memory changed'):
                list(turn.stream('函数', [], task_id='t', resume=True, memory_input=self.snapshot,
                                 memory_service=self.service, memory_revision_guard=denied))
