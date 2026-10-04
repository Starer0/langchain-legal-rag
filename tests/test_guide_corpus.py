import unittest
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from guide_corpus import prepare_guide


ROOT = Path(__file__).resolve().parents[1]
GUIDE = {
    "knowledge_base_id": "C",
    "document_id": "labor_arbitration_guide_experimental_city",
    "document_type": "办事指南",
    "title": "劳动争议仲裁办事指南",
    "source_file": "劳动争议仲裁办事指南.pdf",
    "version": "2026-09",
    "status": "实验资料",
    "content_start_page": 3,
}


class GuideParsingTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "data" / GUIDE["source_file"]
        self.pages = PyPDFLoader(str(path)).load()

    def test_every_guide_chunk_inherits_its_knowledge_base(self):
        docs = prepare_guide(self.pages, GUIDE)
        self.assertEqual({doc.metadata.get('knowledge_base_id') for doc in docs}, {'C'})

    def test_guide_chunking_rejects_missing_or_invalid_knowledge_base(self):
        for label in (None, '', ' C ', ['C'], 'A/B'):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'knowledge_base_id'):
                prepare_guide(self.pages, {**GUIDE, 'knowledge_base_id': label})

    def test_preserves_the_guide_natural_retrieval_units(self):
        docs = prepare_guide(self.pages, GUIDE)

        self.assertEqual(len(docs), 21)
        self.assertEqual(
            [doc.metadata["section"] for doc in docs],
            [
                "1.1 适用范围", "1.2 申请人资格", "1.3.1 管辖", "1.3.2 时效",
                "第二章 办理流程", "2.1 全流程预计时长",
                "3.1 必备材料", "3.2 补充材料", "3.3 材料格式要求",
                "4.1 各环节时限一览", "4.2 费用说明",
                "5.1 仲裁裁决一般多久作出？", "5.2 申请劳动仲裁需要准备哪些证据？",
                "5.3 请律师代理仲裁的费用由谁承担？", "5.4 单位扣押劳动合同原件，拿不到证据怎么办？",
                "5.5 对裁决结果不服怎么办？", "5.6 撤回申请后还能再次申请吗？",
                "5.7 本人不方便到现场，可以委托他人代办吗？",
                "6.1 追索劳动报酬", "6.2 未签订书面劳动合同", "6.3 超过仲裁时效怎么办",
            ],
        )

        procedure = docs[4]
        self.assertEqual(procedure.metadata["chunk_kind"], "procedure")
        self.assertIn("1. 自查申请条件", procedure.page_content)
        self.assertIn("8. 领取裁决与后续救济", procedure.page_content)

        schedule = docs[9]
        self.assertEqual(schedule.metadata["chunk_kind"], "table")
        self.assertIn("环节：受理审查；时限：5 个工作日；起算点：收到仲裁申请之日", schedule.page_content)

        faq = docs[11]
        self.assertEqual(faq.metadata["chunk_kind"], "faq")
        self.assertIn("问：仲裁裁决一般多久作出？", faq.page_content)
        self.assertIn("答：仲裁庭应当自组庭之日起 45 日内作出裁决", faq.page_content)


if __name__ == "__main__":
    unittest.main()
