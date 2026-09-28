import unittest

from langchain_core.messages import AIMessage, HumanMessage

from query_rewrite import RetrievalPlan, RetrievalQuestionRewriter


class CapturingModel:
    def __init__(self, output):
        self.output = output
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.output)


class RetrievalQuestionRewriterTests(unittest.TestCase):
    def test_rewrite_without_history_returns_question_and_guide_route(self):
        model = CapturingModel(
            '{"retrieval_question":"试用期工资规定","include_guide":false}'
        )
        rewriter = RetrievalQuestionRewriter(model)

        plan = rewriter.rewrite("试用期工资呢？", [])

        self.assertEqual(
            plan, RetrievalPlan("试用期工资规定", include_guide=False)
        )
        rendered = "\n".join(
            message.content for message in model.prompts[0].to_messages()
        )
        self.assertIn("试用期工资呢？", rendered)
        self.assertNotIn("历史对话", rendered)

    def test_rewrite_with_history_includes_context_and_follow_up(self):
        history = [
            HumanMessage(content="试用期最长多久？"),
            AIMessage(content="最长六个月。"),
        ]
        model = CapturingModel(
            '{"retrieval_question":"试用期内劳动者工资有什么规定？","include_guide":false}'
        )

        plan = RetrievalQuestionRewriter(model).rewrite("那工资呢？", history)

        self.assertEqual(plan.retrieval_question, "试用期内劳动者工资有什么规定？")
        rendered = "\n".join(
            message.content for message in model.prompts[0].to_messages()
        )
        self.assertIn("试用期最长多久？", rendered)
        self.assertIn("那工资呢？", rendered)
        self.assertIn("不是法律依据", rendered)

    def test_invalid_output_falls_back_to_an_uncertain_plan(self):
        plan = RetrievalQuestionRewriter(CapturingModel("  ")).rewrite(
            "那工资呢？", []
        )

        self.assertEqual(plan, RetrievalPlan("那工资呢？", include_guide=None))

    def test_catalog_topics_let_model_route_material_question_to_guide(self):
        model = CapturingModel(
            '{"retrieval_question":"劳动仲裁申请需要准备哪些材料？","include_guide":true}'
        )
        rewriter = RetrievalQuestionRewriter(model, guides=[{
            "document_type": "办事指南",
            "title": "劳动争议仲裁办事指南",
            "routing_topics": ["劳动仲裁申请", "申请材料"],
        }])

        plan = rewriter.rewrite("劳动仲裁要带什么？", [])

        self.assertEqual(
            plan,
            RetrievalPlan("劳动仲裁申请需要准备哪些材料？", include_guide=True),
        )
        rendered = "\n".join(message.content for message in model.prompts[0].to_messages())
        self.assertIn("申请材料", rendered)
        self.assertIn('"include_guide"', rendered)


if __name__ == "__main__":
    unittest.main()
