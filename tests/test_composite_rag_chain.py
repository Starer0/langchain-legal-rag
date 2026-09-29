import unittest

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from evidence_selector import EvidenceSelection
from performance import TurnProfile
from rag_pipeline_articles import CompositeRagChain


def _doc(article, law_id="labor_law"):
    return Document(
        page_content=f"{article}的内容",
        metadata={"law_id": law_id, "version": "2026", "article": article, "pages": [1]},
    )


class CompositeRagChainTests(unittest.TestCase):
    def test_each_subquestion_is_separately_retrieved_and_reranked_then_answered_once(self):
        queries = []
        rerank_queries = []
        answers = []
        shared = _doc("第二十七条", "labor_arbitration_law")
        batches = {
            "工资规定？": [_doc("第二十条", "labor_contract_law"), shared],
            "仲裁时效？": [shared, _doc("第二十八条", "labor_arbitration_law")],
        }

        def retrieve(state):
            queries.append(state["retrieval_question"])
            self.assertEqual(state["question"], state["retrieval_question"])
            return batches[state["retrieval_question"]]

        def rerank(state):
            rerank_queries.append(state["retrieval_question"])
            return state["candidates"]

        def answer(state):
            answers.append(state)
            return "统一回答"

        chain = CompositeRagChain(
            single_chain=RunnableLambda(lambda _: self.fail("复合题不能走单查询")),
            retriever=RunnableLambda(retrieve),
            reranker=RunnableLambda(rerank),
            answer_chain=RunnableLambda(answer),
            top_n=3,
        )

        result = chain.invoke({
            "question": "工资和仲裁时效怎么办？",
            "retrieval_question": "工资和仲裁时效怎么办？",
            "retrieval_questions": ["工资规定？", "仲裁时效？"],
        })

        self.assertEqual(queries, ["工资规定？", "仲裁时效？"])
        self.assertEqual(rerank_queries, queries)
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0]["question"], "工资和仲裁时效怎么办？")
        self.assertEqual(
            [(source["law_id"], source["article"]) for source in result["sources"]],
            [
                ("labor_contract_law", "第二十条"),
                ("labor_arbitration_law", "第二十七条"),
                ("labor_arbitration_law", "第二十八条"),
            ],
        )
        self.assertEqual(len(result["candidates"]), 3)
        self.assertEqual(result["answer"], "统一回答")
        self.assertEqual(
            [entry["question"] for entry in result["subquestion_reranks"]],
            ["工资规定？", "仲裁时效？"],
        )
        self.assertEqual(
            [[source["article"] for source in entry["sources"]]
            for entry in result["subquestion_reranks"]],
            [["第二十条", "第二十七条"], ["第二十七条", "第二十八条"]],
        )

    def test_single_question_uses_existing_pipeline(self):
        original = {"answer": "原链路回答", "candidates": [], "sources": []}
        chain = CompositeRagChain(
            single_chain=RunnableLambda(lambda _: original),
            retriever=RunnableLambda(lambda _: self.fail("不应再次检索")),
            reranker=RunnableLambda(lambda _: self.fail("不应再次重排")),
            answer_chain=RunnableLambda(lambda _: self.fail("不应再次生成")),
            top_n=5,
        )

        self.assertEqual(chain.invoke({
            "question": "试用期工资？",
            "retrieval_question": "试用期工资？",
            "retrieval_questions": ["试用期工资？"],
        }), original)

    def test_uses_evidence_selector_to_drop_unsupported_subquestion(self):
        first = _doc("第三十条", "labor_contract_law")
        second = _doc("第二十八条", "labor_arbitration_law")
        unsupported = _doc("第四十四条", "labor_law")

        class Selector:
            def __init__(self):
                self.calls = []

            def select(self, original_question, questions, ranked_batches, top_n):
                self.calls.append((original_question, questions, ranked_batches, top_n))
                return EvidenceSelection(
                    answerable=[True, True, False],
                    selected_document_ids=["q1-d1", "q2-d1"],
                    documents=[first, second],
                )

        selector = Selector()
        profile = TurnProfile()
        batches = {
            "工资问题？": [first],
            "仲裁问题？": [second],
            "资料外问题？": [unsupported],
        }
        chain = CompositeRagChain(
            single_chain=RunnableLambda(lambda _: self.fail("复合题不能走单查询")),
            retriever=RunnableLambda(lambda state: batches[state["retrieval_question"]]),
            reranker=RunnableLambda(lambda state: state["candidates"]),
            answer_chain=RunnableLambda(lambda _: "统一回答"),
            top_n=4,
            evidence_selector=selector,
        )

        result = chain.invoke({
            "question": "工资、仲裁和资料外问题？",
            "retrieval_questions": ["工资问题？", "仲裁问题？", "资料外问题？"],
            "_profile": profile,
        })

        self.assertEqual(len(selector.calls), 1)
        self.assertEqual(
            [source["article"] for source in result["sources"]],
            ["第三十条", "第二十八条"],
        )
        self.assertEqual(result["evidence_selection"]["answerable"], [True, True, False])
        self.assertEqual(profile.snapshot()["stages"]["evidence_selection"]["calls"], 1)
        self.assertEqual(profile.snapshot()["model_calls"], 2)

    def test_scales_context_and_per_question_rerank_budget_for_many_questions(self):
        questions = [f"子问题{index}？" for index in range(1, 7)]
        batches = {
            question: [_doc(f"第{index}条", "labor_law")]
            for index, question in enumerate(questions, start=1)
        }
        rerank_budgets = []
        answers = []

        def rerank(state):
            rerank_budgets.append(state["rerank_top_n"])
            return state["candidates"]

        chain = CompositeRagChain(
            single_chain=RunnableLambda(lambda _: self.fail("复合题不能走单查询")),
            retriever=RunnableLambda(lambda state: batches[state["retrieval_question"]]),
            reranker=RunnableLambda(rerank),
            answer_chain=RunnableLambda(lambda state: answers.append(state) or "统一回答"),
            top_n=5,
        )

        result = chain.invoke({
            "question": "六个问题？", "retrieval_questions": questions,
        })

        self.assertEqual(rerank_budgets, [2, 2, 2, 2, 2, 2])
        self.assertEqual(len(result["sources"]), 6)
        self.assertEqual(len(answers[0]["docs"]), 6)


if __name__ == "__main__":
    unittest.main()
