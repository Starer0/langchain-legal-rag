import unittest
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda

from query_rewrite import RetrievalPlan


class LangGraphRagTests(unittest.TestCase):
    def test_checkpoints_save_node_states_and_isolate_threads(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph_rag import build_langgraph_rag, LangGraphConversationService

        rewriter = Mock()
        rewriter.rewrite.return_value = RetrievalPlan("工资规定", False)
        doc = Document(page_content="工资规定", metadata={"article": "第一条"})
        retrieve = Mock(return_value=[doc])
        graph = build_langgraph_rag(
            rewriter, RunnableLambda(retrieve),
            RunnableLambda(lambda state: state["candidates"]),
            RunnableLambda(lambda state: state["question"]),
            RunnableLambda(lambda text: "答复"),
            checkpointer=InMemorySaver(),
        )
        service = LangGraphConversationService(graph, InMemoryChatMessageHistory(), thread_id="A")
        self.assertIsNone(service.inspect_checkpoint())
        result = service.ask("工资？", on_node_update=lambda *args: None)
        saved = service.inspect_checkpoint()
        self.assertEqual(saved["question"], "工资？")
        self.assertEqual(saved["answer"], result["answer"])
        self.assertEqual(saved["next"], [])
        snapshots = service.checkpoint_history()
        self.assertTrue(any(item["next"] == ["rerank"] and item["candidate_count"] == 1 for item in snapshots))
        self.assertTrue(any(item["next"] == ["answer"] and item["document_count"] == 1 for item in snapshots))
        other = LangGraphConversationService(graph, InMemoryChatMessageHistory(), thread_id="B")
        self.assertIsNone(other.inspect_checkpoint())
        self.assertEqual(retrieve.call_count, 1)
        rewriter.rewrite.assert_called_once()
        fresh_graph = build_langgraph_rag(
            rewriter, RunnableLambda(retrieve),
            RunnableLambda(lambda state: state["candidates"]),
            RunnableLambda(lambda state: state["question"]),
            RunnableLambda(lambda text: "答复"),
            checkpointer=InMemorySaver(),
        )
        restarted = LangGraphConversationService(fresh_graph, InMemoryChatMessageHistory(), thread_id="A")
        self.assertIsNone(restarted.inspect_checkpoint())

    def test_empty_candidates_take_no_evidence_branch_without_rerank_or_answer_model(self):
        from langgraph_rag import build_langgraph_rag, LangGraphConversationService

        for observe in (False, True):
            with self.subTest(observe=observe):
                rewriter = Mock()
                rewriter.rewrite.return_value = RetrievalPlan("未知法律第一条", False)
                retrieve = Mock(return_value=[])
                rerank = Mock(side_effect=AssertionError("空候选不应调用重排"))
                answer = Mock(side_effect=AssertionError("空候选不应调用回答模型"))
                graph = build_langgraph_rag(
                    rewriter,
                    RunnableLambda(retrieve),
                    RunnableLambda(rerank),
                    RunnableLambda(lambda state: state["question"]),
                    RunnableLambda(answer),
                )
                history = InMemoryChatMessageHistory()
                service = LangGraphConversationService(graph, history, profile=True)
                events = []
                options = {"on_node_update": lambda node, update: events.append(node)} if observe else {}

                result = service.ask("未知法律第一条？", **options)

                self.assertEqual(result["answer"], "资料中没有足够依据。")
                self.assertEqual(result["candidates"], [])
                self.assertEqual(result["sources"], [])
                self.assertEqual(result["docs"], [])
                self.assertEqual(result["performance"]["model_calls"], 1)
                self.assertEqual(result["performance"]["reranker_calls"], 0)
                retrieve.assert_called_once()
                rerank.assert_not_called()
                answer.assert_not_called()
                self.assertEqual(history.messages[-1].content, result["answer"])
                if observe:
                    self.assertEqual(events, ["rewrite", "retrieve", "no_evidence"])

    def test_stream_failure_does_not_save_a_partial_conversation(self):
        from langgraph_rag import LangGraphConversationService

        graph = Mock()
        def updates(*args, **kwargs):
            yield {"rewrite": {"retrieval_question": "工资规定", "include_guide": False}}
            raise RuntimeError("检索失败")
        graph.stream.side_effect = updates
        history = InMemoryChatMessageHistory()
        service = LangGraphConversationService(graph, history)

        with self.assertRaisesRegex(RuntimeError, "检索失败"):
            service.ask("工资？", on_node_update=lambda *args: None)

        self.assertEqual(history.messages, [])
        graph.invoke.assert_not_called()

    def test_observing_real_graph_runs_each_step_once_and_preserves_result(self):
        from langgraph_rag import build_langgraph_rag, LangGraphConversationService

        rewriter = Mock()
        rewriter.rewrite.return_value = RetrievalPlan("工资规定", False)
        calls = []
        document = Document(page_content="工资规定", metadata={"article": "第一条"})
        retriever = RunnableLambda(lambda state: calls.append("retrieve") or [document])
        reranker = RunnableLambda(lambda state: calls.append("rerank") or state["candidates"])
        prompt = RunnableLambda(lambda state: state["question"])
        model = RunnableLambda(lambda text: calls.append("answer") or "回答")
        graph = build_langgraph_rag(rewriter, retriever, reranker, prompt, model)
        service = LangGraphConversationService(graph, InMemoryChatMessageHistory(), profile=True)
        events = []

        result = service.ask("工资？", on_node_update=lambda node, update: events.append((node, update)))

        self.assertEqual([node for node, _ in events], ["rewrite", "retrieve", "rerank", "answer"])
        self.assertEqual(calls, ["retrieve", "rerank", "answer"])
        rewriter.rewrite.assert_called_once()
        self.assertEqual(events[0][1]["retrieval_question"], "工资规定")
        self.assertEqual(result["answer"], "回答")
        self.assertEqual(result["sources"][0]["article"], "第一条")
        self.assertEqual(result["performance"]["model_calls"], 2)

    def test_conversation_service_passes_history_into_graph_and_stores_answer(self):
        from langgraph_rag import LangGraphConversationService

        history = InMemoryChatMessageHistory()
        history.add_user_message("试用期最长多久？")
        history.add_ai_message("最长六个月。")
        graph = Mock()
        graph.invoke.return_value = {
            "retrieval_question": "试用期工资有什么规定？",
            "include_guide": False,
            "candidates": [],
            "sources": [],
            "answer": "试用期工资应符合规定。",
        }
        service = LangGraphConversationService(graph, history)

        result = service.ask("那工资呢？")

        graph.invoke.assert_called_once_with({
            "question": "那工资呢？",
            "history": history.messages[:2],
        })
        self.assertEqual(result["retrieval_question"], "试用期工资有什么规定？")
        self.assertEqual(
            [message.content for message in history.messages[-2:]],
            ["那工资呢？", "试用期工资应符合规定。"],
        )

    def test_graph_passes_explicit_state_from_rewrite_to_answer(self):
        from langgraph_rag import build_langgraph_rag

        rewriter = Mock()
        rewriter.rewrite.return_value = RetrievalPlan(
            "试用期工资有什么规定？", include_guide=False
        )
        document = Document(
            page_content="第二十条 试用期工资。",
            metadata={"article": "第二十条", "pages": [1], "source": "劳动合同法.pdf"},
        )
        seen = {}

        def retrieve(state):
            seen["retrieve"] = state
            return [document]

        def rerank(state):
            seen["rerank"] = state
            return state["candidates"]

        graph = build_langgraph_rag(
            rewriter=rewriter,
            retriever=RunnableLambda(retrieve),
            reranker=RunnableLambda(rerank),
            prompt=RunnableLambda(lambda values: values["context"] + values["question"]),
            model=RunnableLambda(lambda text: "回答：" + text),
        )

        result = graph.invoke({"question": "那工资呢？", "history": []})

        self.assertEqual(result["retrieval_question"], "试用期工资有什么规定？")
        self.assertFalse(result["include_guide"])
        self.assertEqual(seen["retrieve"]["retrieval_question"], "试用期工资有什么规定？")
        self.assertEqual(seen["rerank"]["candidates"], [document])
        self.assertEqual(result["sources"][0]["article"], "第二十条")
        self.assertIn("那工资呢？", result["answer"])

    def test_graph_matches_existing_single_question_chain_output(self):
        from langgraph_rag import build_langgraph_rag
        from rag_pipeline_articles import build_rag_chain

        rewriter = Mock()
        rewriter.rewrite.return_value = RetrievalPlan("试用期工资？", include_guide=False)
        document = Document(
            page_content="第二十条 试用期工资。",
            metadata={"article": "第二十条", "pages": [1], "source": "劳动合同法.pdf"},
        )
        retriever = RunnableLambda(lambda _: [document])
        reranker = RunnableLambda(lambda state: state["candidates"])
        prompt = RunnableLambda(lambda values: values["context"] + values["question"])
        model = RunnableLambda(lambda text: "回答：" + text)
        graph = build_langgraph_rag(rewriter, retriever, reranker, prompt, model)
        existing = build_rag_chain(
            retriever, reranker, prompt, model, stateful_retriever=True
        )

        graph_result = graph.invoke({"question": "那工资呢？", "history": []})
        existing_result = existing.invoke({
            "question": "那工资呢？", "retrieval_question": "试用期工资？",
            "include_guide": False,
        })

        self.assertEqual(graph_result["answer"], existing_result["answer"])
        self.assertEqual(graph_result["candidates"], existing_result["candidates"])
        self.assertEqual(graph_result["sources"], existing_result["sources"])


if __name__ == "__main__":
    unittest.main()
