import unittest
from unittest.mock import Mock, patch

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from conversation import ConversationRagService
from rag_pipeline_articles import CompositeRagChain, build_rag_chain


class ConversationProfilingTests(unittest.TestCase):
    def test_profiled_turn_reports_rewrite_and_pipeline_without_changing_answer(self):
        history = InMemoryChatMessageHistory()
        rewriter = Mock()
        rewriter.rewrite.return_value = "试用期工资规定？"
        chain = Mock()
        chain.invoke.return_value = {"answer": "按法律规定支付。", "sources": []}
        service = ConversationRagService(chain, rewriter, history, profile=True)

        result = service.ask("那工资呢？")

        self.assertEqual(result["answer"], "按法律规定支付。")
        self.assertEqual(result["retrieval_question"], "试用期工资规定？")
        self.assertEqual(result["performance"]["stages"]["rewrite"]["calls"], 1)
        self.assertEqual(result["performance"]["stages"]["pipeline"]["calls"], 1)
        self.assertGreaterEqual(result["performance"]["total_ms"], 0)
        self.assertEqual(len(history.messages), 2)
        self.assertNotIn("performance", chain.invoke.return_value)

    def test_profiled_decomposition_counts_one_call_without_changing_subquestions(self):
        rewriter = Mock()
        rewriter.rewrite.return_value = "工资和仲裁时效？"
        decomposer = Mock()
        decomposer.decompose.return_value = ["工资？", "仲裁时效？"]
        chain = Mock()
        chain.invoke.return_value = {"answer": "合并回答"}
        service = ConversationRagService(
            chain, rewriter, InMemoryChatMessageHistory(),
            decomposer=decomposer, profile=True,
        )

        result = service.ask("工资少发了，仲裁怎么办？")

        self.assertEqual(result["subquestions"], ["工资？", "仲裁时效？"])
        self.assertEqual(result["performance"]["stages"]["decompose"]["calls"], 1)
        self.assertEqual(chain.invoke.call_args.args[0]["retrieval_questions"], result["subquestions"])


class RagProfilingTests(unittest.TestCase):
    @patch("builtins.print")
    @patch("rag_app.build_rag_chain")
    @patch("rag_app.SiliconFlowReranker")
    @patch("rag_app.open_corpus")
    @patch("rag_app.OpenAIEmbeddings")
    @patch("rag_app.ChatOpenAI")
    @patch("rag_app.load_dotenv")
    def test_factory_profiles_chroma_and_remote_rerank_calls(
        self, load_dotenv, chat_openai, embeddings, open_corpus,
        reranker_class, build_chain, print_output,
    ):
        import rag_app
        from performance import TurnProfile

        store = open_corpus.return_value[0]
        open_corpus.return_value = (store, {"article_count": 1}, [])
        document = Document(page_content="法条", metadata={"article": "第一条"})
        store.similarity_search.return_value = [document]
        reranker_class.return_value.rerank.return_value = [document]
        profile = TurnProfile()

        rag_app.create_rag_chain(profile=True)
        retriever, reranker = build_chain.call_args.args[:2]
        self.assertTrue(build_chain.call_args.kwargs["profile"])
        state = {
            "question": "原问题",
            "retrieval_question": "检索问题",
            "_profile": profile,
        }
        candidates = retriever.invoke(state)
        sources = reranker.invoke({**state, "candidates": candidates})

        self.assertEqual(sources, [document])
        self.assertEqual(store.similarity_search.call_count, 1)
        self.assertEqual(reranker_class.return_value.rerank.call_count, 1)
        self.assertEqual(profile.snapshot()["stages"]["chroma"]["calls"], 1)
        self.assertEqual(profile.snapshot()["stages"]["rerank"]["calls"], 1)
        self.assertEqual(profile.snapshot()["reranker_calls"], 1)

    def test_single_query_profiles_answer_without_changing_sources(self):
        doc = Document(page_content="第二十条 试用期工资", metadata={
            "article": "第二十条", "pages": [1],
        })
        retriever = RunnableLambda(lambda _: [doc])
        reranker = RunnableLambda(lambda state: state["candidates"])
        prompt = RunnableLambda(lambda values: values["context"] + values["question"])
        model = RunnableLambda(lambda value: value)
        from performance import TurnProfile
        profile = TurnProfile()

        result = build_rag_chain(
            retriever, reranker, prompt, model, profile=True
        ).invoke({
            "question": "那工资呢？",
            "retrieval_question": "试用期工资规定？",
            "_profile": profile,
        })

        self.assertEqual(result["sources"][0]["article"], "第二十条")
        self.assertIn("那工资呢？", result["answer"])
        self.assertEqual(profile.snapshot()["stages"]["answer"]["calls"], 1)

    @patch("rag_app.create_rag_chain")
    @patch("rag_app.ChatOpenAI")
    @patch("rag_app.load_dotenv")
    def test_profiled_service_builds_a_profiled_rag_chain(
        self, load_dotenv, chat_openai, create_chain,
    ):
        import rag_app

        service = rag_app.create_conversation_service(profile=True)

        self.assertTrue(service.profile)
        create_chain.assert_called_once_with(profile=True)

    def test_composite_path_propagates_profile_and_counts_one_final_answer(self):
        from performance import TurnProfile
        profile = TurnProfile()
        doc = Document(page_content="法条", metadata={"article": "第一条", "pages": [1]})
        seen_profiles = []

        def retrieve(state):
            seen_profiles.append(state.get("_profile"))
            return [doc]

        chain = CompositeRagChain(
            single_chain=RunnableLambda(lambda _: self.fail("不应走单查询")),
            retriever=RunnableLambda(retrieve),
            reranker=RunnableLambda(lambda state: state["candidates"]),
            answer_chain=RunnableLambda(lambda _: "合并回答"),
            top_n=5,
        )
        result = chain.invoke({
            "question": "复合问题",
            "retrieval_questions": ["子问题一", "子问题二"],
            "_profile": profile,
        })

        self.assertEqual(result["answer"], "合并回答")
        self.assertEqual(seen_profiles, [profile, profile])
        self.assertEqual(profile.snapshot()["stages"]["answer"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
