import unittest

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import rag_pipeline_articles as pipeline


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class SiliconFlowRerankerTests(unittest.TestCase):
    def test_maps_ranked_indices_back_to_documents_and_keeps_scores(self):
        captured_request = {}

        def fake_post(url, *, headers, json, timeout):
            captured_request.update({
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            })
            return FakeResponse({
                "id": "rerank-test",
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.72},
                ],
                "meta": {"tokens": {"input_tokens": 42}},
            })

        reranker_class = getattr(pipeline, "SiliconFlowReranker", None)
        self.assertIsNotNone(
            reranker_class,
            "rag_pipeline_articles 还没有实现 SiliconFlowReranker",
        )
        reranker = reranker_class(
            api_key="test-key",
            base_url="https://api.siliconflow.cn/v1/",
            model="BAAI/bge-reranker-v2-m3",
            top_n=2,
            timeout=12,
            post=fake_post,
        )
        documents = [
            Document(page_content="第一条", metadata={"article": "第一条"}),
            Document(page_content="第二条", metadata={"article": "第二条"}),
            Document(page_content="第三条", metadata={"article": "第三条"}),
        ]

        ranked = reranker.rerank("试用期工资", documents)

        self.assertEqual(
            [doc.metadata["article"] for doc in ranked],
            ["第三条", "第一条"],
        )
        self.assertEqual(
            [doc.metadata["rerank_score"] for doc in ranked],
            [0.91, 0.72],
        )
        self.assertNotIn("rerank_score", documents[2].metadata)
        self.assertEqual(
            captured_request["url"],
            "https://api.siliconflow.cn/v1/rerank",
        )
        self.assertEqual(
            captured_request["headers"]["Authorization"],
            "Bearer test-key",
        )
        self.assertEqual(
            captured_request["json"],
            {
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "试用期工资",
                "documents": ["第一条", "第二条", "第三条"],
                "return_documents": False,
                "top_n": 2,
            },
        )
        self.assertEqual(captured_request["timeout"], 12)

    def test_empty_candidates_do_not_call_remote_api(self):
        def unexpected_post(*args, **kwargs):
            self.fail("没有候选文档时不应调用 Reranker API")

        reranker_class = getattr(pipeline, "SiliconFlowReranker", None)
        self.assertIsNotNone(reranker_class)
        reranker = reranker_class(
            api_key="test-key",
            post=unexpected_post,
        )

        self.assertEqual(reranker.rerank("任意问题", []), [])


class RagChainWithRerankerTests(unittest.TestCase):
    def test_retrieves_with_rewritten_question_and_answers_original_question(self):
        retrieval_queries = []

        def retrieve(question):
            retrieval_queries.append(question)
            return [
                Document(
                    page_content="试用期工资资料",
                    metadata={"article": "第二十条", "pages": [2]},
                )
            ]

        retriever = RunnableLambda(retrieve)
        reranker = RunnableLambda(lambda state: state["candidates"])
        prompt = RunnableLambda(
            lambda values: f"{values['context']} | answer to: {values['question']}"
        )
        model = RunnableLambda(lambda value: value)

        result = pipeline.build_rag_chain(
            retriever, reranker, prompt, model
        ).invoke({
            "question": "那工资呢？",
            "retrieval_question": "试用期内劳动者工资有什么规定？",
        })

        self.assertEqual(retrieval_queries, ["试用期内劳动者工资有什么规定？"])
        self.assertIn("answer to: 那工资呢？", result["answer"])
    def test_retrieves_once_then_reuses_reranked_documents(self):
        retrieval_queries = []
        reranker_inputs = []
        candidates = [
            Document(
                page_content="候选一",
                metadata={"article": "第一条", "pages": [1]},
            ),
            Document(
                page_content="真正相关的候选二",
                metadata={
                    "article": "第二条",
                    "pages": [2],
                    "rerank_score": 0.88,
                },
            ),
            Document(
                page_content="候选三",
                metadata={"article": "第三条", "pages": [3]},
            ),
        ]

        def retrieve(question):
            retrieval_queries.append(question)
            return candidates

        def rerank(state):
            reranker_inputs.append(state)
            return [state["candidates"][1]]

        retriever = RunnableLambda(retrieve)
        reranker = RunnableLambda(rerank)
        prompt = RunnableLambda(lambda values: values["context"])
        model = RunnableLambda(lambda value: value)

        try:
            chain = pipeline.build_rag_chain(
                retriever,
                reranker,
                prompt,
                model,
            )
        except TypeError as error:
            self.fail(f"build_rag_chain 尚未接入 reranker：{error}")

        result = chain.invoke({
            "question": "测试问题",
            "retrieval_question": "测试问题",
        })

        self.assertEqual(retrieval_queries, ["测试问题"])
        self.assertEqual(len(reranker_inputs), 1)
        self.assertEqual(reranker_inputs[0]["question"], "测试问题")
        self.assertEqual(reranker_inputs[0]["candidates"], candidates)
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["article"], "第二条")
        self.assertEqual(result["sources"][0]["rerank_score"], 0.88)
        self.assertIn("真正相关的候选二", result["answer"])
        self.assertNotIn("候选一", result["answer"])


if __name__ == "__main__":
    unittest.main()
