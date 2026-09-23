import unittest

from langchain_core.messages import AIMessage

from query_decomposition import CompositeQuestionDecomposer


class _Model:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.content)


class CompositeQuestionDecomposerTests(unittest.TestCase):
    def test_splits_distinct_legal_points_into_standalone_questions(self):
        model = _Model(
            '{"questions":["法定节假日加班工资怎么算？",'
            '"劳动争议申请仲裁的时效多久？"]}'
        )

        questions = CompositeQuestionDecomposer(model).decompose(
            "国庆加班没有加班费，仲裁时效多久？"
        )

        self.assertEqual(questions, [
            "法定节假日加班工资怎么算？",
            "劳动争议申请仲裁的时效多久？",
        ])
        prompt = "\n".join(message.content for message in model.prompts[0].to_messages())
        self.assertIn("国庆加班没有加班费，仲裁时效多久？", prompt)
        self.assertIn("不同", prompt)

    def test_keeps_single_intent_on_original_retrieval_path(self):
        model = _Model('{"questions":["试用期工资标准是什么？"]}')

        self.assertEqual(
            CompositeQuestionDecomposer(model).decompose("试用期工资标准是什么？"),
            ["试用期工资标准是什么？"],
        )

    def test_invalid_or_repeated_output_falls_back_to_original_question(self):
        for output in (
            "not json",
            '{"questions":[]}',
            '{"questions":["工资？","工资？"]}',
            '{"questions":["a","b","c","d","e"]}',
        ):
            with self.subTest(output=output):
                self.assertEqual(
                    CompositeQuestionDecomposer(_Model(output)).decompose("原问题？"),
                    ["原问题？"],
                )


if __name__ == "__main__":
    unittest.main()
