import unittest

from langchain_core.documents import Document

from evidence_selector import EvidenceSelector


def _doc(article, content):
    return Document(
        page_content=content,
        metadata={"law_id": "labor_law", "article": article, "pages": [1]},
    )


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.response


class EvidenceSelectorTests(unittest.TestCase):
    def test_selects_supported_subquestions_and_excludes_unsupported_one(self):
        model = FakeModel('''
        {
          "subquestions": [
            {"index": 1, "answerable": true},
            {"index": 2, "answerable": true},
            {"index": 3, "answerable": false}
          ],
          "selected_document_ids": ["q1-d1", "q2-d1"]
        }
        ''')
        first = _doc("第三十条", "用人单位应当及时足额支付劳动报酬。")
        second = _doc("第二十八条", "仲裁申请书应当载明劳动者基本情况。")
        unsupported = _doc("第四十四条", "法定休假日安排劳动者工作的，支付三倍工资。")

        selection = EvidenceSelector(model).select(
            "工资、仲裁申请书和社保补缴分别怎么办？",
            ["工资应如何支付？", "仲裁申请书应写什么？", "社保补缴怎么办？"],
            [[first], [second], [unsupported]],
            top_n=4,
        )

        self.assertIsNotNone(selection)
        self.assertEqual(selection.answerable, [True, True, False])
        self.assertEqual(selection.documents, [first, second])
        self.assertEqual(selection.selected_document_ids, ["q1-d1", "q2-d1"])
        self.assertIn("q3-d1", str(model.prompts[0]))

    def test_invalid_selection_falls_back_to_existing_merge_rule(self):
        model = FakeModel(
            '{"subquestions":[{"index":1,"answerable":false}],'
            '"selected_document_ids":["q1-d1"]}'
        )

        selection = EvidenceSelector(model).select(
            "工资应如何支付？",
            ["工资应如何支付？"], [[_doc("第三十条", "及时足额")]], top_n=4
        )

        self.assertIsNone(selection)

    def test_exposes_enough_candidates_to_fill_the_final_context(self):
        model = FakeModel('''
        {"subquestions":[
          {"index":1,"answerable":true},
          {"index":2,"answerable":true}
        ],"selected_document_ids":["q1-d1","q1-d2","q1-d3","q2-d1","q2-d2"]}
        ''')
        first_batch = [
            _doc("第一条", "资料一"), _doc("第二条", "资料二"), _doc("第三条", "资料三"),
        ]
        second_batch = [
            _doc("第四条", "资料四"), _doc("第五条", "资料五"), _doc("第六条", "资料六"),
        ]

        selection = EvidenceSelector(model).select(
            "两个子问题都需要多条证据", ["子问题一", "子问题二"],
            [first_batch, second_batch], top_n=5,
        )

        self.assertIsNotNone(selection)
        self.assertEqual(len(selection.documents), 5)
        self.assertIn("q1-d3", str(model.prompts[0]))
        self.assertIn("q2-d3", str(model.prompts[0]))


if __name__ == "__main__":
    unittest.main()
