import unittest
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda

from query_rewrite import RetrievalPlan


class LangGraphRagTests(unittest.TestCase):
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
