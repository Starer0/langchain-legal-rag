import unittest

from langchain_core.documents import Document

from bm25_retriever import BM25Retriever, merge_retrieval_candidates


def document(text, **metadata):
    return Document(page_content=text, metadata={"index_status": "active", **metadata})


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.trial_pay = document(
            "第二十条 劳动者在试用期的工资不得低于本单位相同岗位最低档工资。",
            law_id="labor_contract_law", article="第二十条",
        )
        self.arbitration = document(
            "第二十七条 劳动争议申请仲裁的时效期间为一年。",
            law_id="labor_arbitration_law", article="第二十七条",
        )
        self.guide = document(
            "仲裁申请应提交仲裁申请书、身份证明和劳动关系证明材料。",
            document_type="办事指南", section="3.1 必备材料",
        )
        self.retriever = BM25Retriever([self.trial_pay, self.arbitration, self.guide])

    def test_returns_keyword_matching_chinese_document(self):
        found = self.retriever.search("试用期最低工资", k=2)

        self.assertEqual(found[0].metadata["article"], "第二十条")

    def test_honors_the_same_metadata_filter_as_vector_retrieval(self):
        found = self.retriever.search(
            "仲裁申请材料",
            k=3,
            metadata_filter={"document_type": "办事指南"},
        )

        self.assertEqual(found, [self.guide])

    def test_honors_nested_law_or_guide_filter(self):
        found = self.retriever.search(
            "仲裁申请材料",
            k=3,
            metadata_filter={"$or": [
                {"law_id": {"$in": ["labor_contract_law"]}},
                {"document_type": "办事指南"},
            ]},
        )

        self.assertEqual(found, [self.guide])


class CandidateMergeTests(unittest.TestCase):
    def test_keeps_vector_order_then_adds_only_new_bm25_documents(self):
        vector = [
            document("第二十条 试用期工资", law_id="labor_contract_law", version="2012", article="第二十条"),
            document("第二十七条 仲裁时效", law_id="labor_arbitration_law", version="2009", article="第二十七条"),
        ]
        bm25 = [
            document("重复的第二十条", law_id="labor_contract_law", version="2012", article="第二十条"),
            document("第三十条 工资支付", law_id="labor_contract_law", version="2012", article="第三十条"),
        ]

        merged = merge_retrieval_candidates(vector, bm25)

        self.assertEqual([item.metadata["article"] for item in merged], ["第二十条", "第二十七条", "第三十条"])


if __name__ == "__main__":
    unittest.main()
