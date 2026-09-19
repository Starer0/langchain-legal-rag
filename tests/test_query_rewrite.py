import unittest

from langchain_core.messages import AIMessage, HumanMessage

from query_rewrite import RetrievalQuestionRewriter


class CapturingModel:
    def __init__(self, output):
        self.output = output
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.output)


class RetrievalQuestionRewriterTests(unittest.TestCase):
    def test_rewrite_without_history_uses_current_question_only(self):
        model = CapturingModel("试用期工资规定")
        rewriter = RetrievalQuestionRewriter(model)

        rewritten = rewriter.rewrite("试用期工资呢？", [])

        self.assertEqual(rewritten, "试用期工资规定")
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
        model = CapturingModel("试用期内劳动者工资有什么规定？")

        rewritten = RetrievalQuestionRewriter(model).rewrite("那工资呢？", history)

        self.assertEqual(rewritten, "试用期内劳动者工资有什么规定？")
        rendered = "\n".join(
            message.content for message in model.prompts[0].to_messages()
        )
        self.assertIn("试用期最长多久？", rendered)
        self.assertIn("那工资呢？", rendered)
        self.assertIn("不是法律依据", rendered)

    def test_blank_rewrite_falls_back_to_original_question(self):
        rewritten = RetrievalQuestionRewriter(CapturingModel("  ")).rewrite(
            "那工资呢？", []
        )

        self.assertEqual(rewritten, "那工资呢？")


if __name__ == "__main__":
    unittest.main()
