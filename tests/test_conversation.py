import unittest
from unittest.mock import Mock

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from conversation import ConversationRagService, recent_complete_turns


class ConversationRagServiceTests(unittest.TestCase):
    def test_recent_complete_turns_keeps_latest_four_pairs(self):
        messages = [
            message
            for turn in range(6)
            for message in (
                HumanMessage(content=f"u{turn}"),
                AIMessage(content=f"a{turn}"),
            )
        ]

        retained = recent_complete_turns(messages, 4)

        self.assertEqual(
            [message.content for message in retained],
            ["u2", "a2", "u3", "a3", "u4", "a4", "u5", "a5"],
        )

    def test_ask_uses_history_for_rewrite_and_preserves_original_question(self):
        history = InMemoryChatMessageHistory()
        history.add_user_message("试用期最长多久？")
        history.add_ai_message("最长六个月。")
        rewriter = Mock()
        rewriter.rewrite.return_value = "试用期内劳动者工资有什么规定？"
        chain = Mock()
        chain.invoke.return_value = {"answer": "应按规定支付工资。"}
        service = ConversationRagService(chain, rewriter, history)

        result = service.ask("那工资呢？")

        rewriter.rewrite.assert_called_once_with("那工资呢？", history.messages[:2])
        chain.invoke.assert_called_once_with({
            "question": "那工资呢？",
            "retrieval_question": "试用期内劳动者工资有什么规定？",
        })
        self.assertEqual(result["retrieval_question"], "试用期内劳动者工资有什么规定？")
        self.assertEqual(
            [message.content for message in history.messages[-2:]],
            ["那工资呢？", "应按规定支付工资。"],
        )

    def test_failed_chain_does_not_append_partial_turn(self):
        history = InMemoryChatMessageHistory()
        rewriter = Mock()
        rewriter.rewrite.return_value = "试用期内劳动者工资有什么规定？"
        chain = Mock()
        chain.invoke.side_effect = RuntimeError("reranker unavailable")
        service = ConversationRagService(chain, rewriter, history)

        with self.assertRaisesRegex(RuntimeError, "reranker unavailable"):
            service.ask("那工资呢？")

        self.assertEqual(history.messages, [])

    def test_blank_question_does_not_call_dependencies_or_mutate_history(self):
        history = InMemoryChatMessageHistory()
        rewriter = Mock()
        chain = Mock()
        service = ConversationRagService(chain, rewriter, history)

        with self.assertRaisesRegex(ValueError, "问题不能为空"):
            service.ask("   ")

        rewriter.rewrite.assert_not_called()
        chain.invoke.assert_not_called()
        self.assertEqual(history.messages, [])


    def test_successful_turns_keep_stored_history_bounded(self):
        history = InMemoryChatMessageHistory()
        rewriter = Mock()
        rewriter.rewrite.side_effect = ["query-1", "query-2", "query-3"]
        chain = Mock()
        chain.invoke.side_effect = [
            {"answer": "answer-1"},
            {"answer": "answer-2"},
            {"answer": "answer-3"},
        ]
        service = ConversationRagService(chain, rewriter, history, max_turns=2)

        service.ask("question-1")
        service.ask("question-2")
        service.ask("question-3")

        self.assertEqual(
            [message.content for message in history.messages],
            ["question-2", "answer-2", "question-3", "answer-3"],
        )


if __name__ == "__main__":
    unittest.main()
