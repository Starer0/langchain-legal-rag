import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_rag import LangGraphConversationService, build_langgraph_rag
from query_rewrite import RetrievalPlan


class InterruptTests(unittest.TestCase):
    def components(self, candidates=None):
        rewriter = Mock()
        rewriter.rewrite.return_value = RetrievalPlan("工资规定", False)
        retrieve = Mock(return_value=candidates if candidates is not None else [
            Document(page_content="工资依据", metadata={"article": "第一条", "pages": [1]})
        ])
        rerank = Mock(side_effect=lambda state: state["candidates"])
        answer = Mock(return_value="答复")
        return rewriter, retrieve, rerank, answer

    def service(self, parts):
        rewriter, retrieve, rerank, answer = parts
        graph = build_langgraph_rag(
            rewriter, RunnableLambda(retrieve), RunnableLambda(rerank),
            RunnableLambda(lambda state: state["question"]), RunnableLambda(answer),
            checkpointer=InMemorySaver(), pause_after_retrieve=True,
        )
        return LangGraphConversationService(graph, InMemoryChatMessageHistory(), thread_id="A")

    def test_pause_then_resume_runs_expensive_steps_once(self):
        for trace in (False, True):
            with self.subTest(trace=trace):
                parts = self.components()
                service = self.service(parts)
                events = []
                options = {"on_node_update": lambda node, update: events.append(node)} if trace else {}
                paused = service.ask("工资？", **options)
                self.assertTrue(paused["paused"])
                self.assertEqual(paused["candidates"][0]["content"], "工资依据")
                self.assertEqual(service.inspect_checkpoint()["next"], ["review"])
                self.assertTrue(service.inspect_checkpoint()["interrupts"])
                self.assertEqual(service.history.messages, [])
                other = LangGraphConversationService(service.graph, InMemoryChatMessageHistory(), thread_id="B")
                with self.assertRaisesRegex(ValueError, "没有.*暂停"):
                    other.resume()
                parts[2].assert_not_called()
                parts[3].assert_not_called()
                with self.assertRaisesRegex(ValueError, "暂停"):
                    service.ask("新问题")
                result = service.resume(**options)
                self.assertEqual(result["answer"], "答复")
                self.assertEqual(service.inspect_checkpoint()["next"], [])
                self.assertEqual(service.inspect_checkpoint()["interrupts"], [])
                self.assertEqual([m.content for m in service.history.messages], ["工资？", "答复"])
                parts[0].rewrite.assert_called_once()
                for operation in parts[1:]:
                    operation.assert_called_once()
                if trace:
                    self.assertEqual(events, ["rewrite", "retrieve", "review", "rerank", "answer"])
                with self.assertRaisesRegex(ValueError, "没有.*暂停"):
                    service.resume()
                self.assertEqual(len(service.history.messages), 2)

    def test_empty_candidates_finish_without_pause(self):
        parts = self.components([])
        service = self.service(parts)
        result = service.ask("未知资料？")
        self.assertFalse(result.get("paused", False))
        self.assertEqual(result["answer"], "资料中没有足够依据。")
        self.assertEqual(service.inspect_checkpoint()["interrupts"], [])
        parts[2].assert_not_called()
        parts[3].assert_not_called()

    def test_resume_failure_does_not_record_partial_answer(self):
        parts = self.components()
        parts[3].side_effect = RuntimeError("回答失败")
        service = self.service(parts)
        service.ask("工资？")
        with self.assertRaisesRegex(RuntimeError, "回答失败"):
            service.resume()
        self.assertEqual(service.history.messages, [])

    def test_sqlite_restart_resumes_pending_turn_without_retrieval(self):
        from rag_app import create_conversation_service

        rewriter, retrieve, rerank, answer = self.components()
        components = (
            RunnableLambda(answer), RunnableLambda(retrieve), RunnableLambda(rerank),
            RunnableLambda(lambda state: state["question"]), None, 5,
        )
        with TemporaryDirectory() as directory, patch("rag_app.load_dotenv"), patch(
            "rag_app.ChatOpenAI"
        ), patch("rag_app._create_rag_components", return_value=components), patch(
            "rag_app.load_guides", return_value=[]
        ), patch("rag_app.RetrievalQuestionRewriter", return_value=rewriter):
            options = dict(use_langgraph=True, checkpoint=True, thread_id="A",
                           checkpoint_db=str(Path(directory) / "checkpoints.sqlite3"),
                           pause_after_retrieve=True)
            service = create_conversation_service(**options)
            try:
                service.ask("工资？")
            finally:
                service.close()
            restarted = create_conversation_service(**options)
            try:
                self.assertEqual(restarted.inspect_checkpoint()["next"], ["review"])
                result = restarted.resume(on_node_update=lambda *args: None)
                self.assertEqual(result["answer"], "答复")
                self.assertEqual([m.content for m in restarted.history.messages], ["工资？", "答复"])
                rewriter.rewrite.assert_called_once()
                retrieve.assert_called_once()
                rerank.assert_called_once()
                answer.assert_called_once()
            finally:
                restarted.close()

    def test_pause_requires_checkpointer(self):
        parts = self.components()
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            build_langgraph_rag(parts[0], RunnableLambda(parts[1]), RunnableLambda(parts[2]),
                                RunnableLambda(lambda s: s["question"]), RunnableLambda(parts[3]),
                                pause_after_retrieve=True)

    def test_cli_pause_resume_and_invalid_resume_keep_loop_running(self):
        from main import run_cli

        service = self.service(self.components())
        inputs = iter(["工资？", "新问题", "/state", "/resume", "/resume", "q"])
        output = []
        run_cli(service, input_fn=lambda _: next(inputs), output_fn=output.append, trace=True)
        text = "\n".join(output)
        self.assertIn("已暂停", text)
        self.assertIn("工资依据", text)
        self.assertIn("没有", text)
        self.assertEqual(text.count("[answer] 完成"), 1)
        self.assertEqual(len(service.history.messages), 2)

    def test_cli_pause_flag_requires_checkpoint(self):
        import main

        with patch("main.ensure_legal_corpus_ready") as ensure, self.assertRaises(SystemExit):
            main.main(["--langgraph", "--pause-after-retrieve"])
        ensure.assert_not_called()

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    @patch("main.ensure_legal_corpus_ready")
    def test_cli_pause_flag_reaches_factory(self, ensure, create, run_cli):
        import main

        main.main(["--langgraph", "--checkpoint", "--pause-after-retrieve"])
        create.assert_called_once_with(decompose=False, use_langgraph=True, checkpoint=True,
                                       thread_id="learning", pause_after_retrieve=True)
