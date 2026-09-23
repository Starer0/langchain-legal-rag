"""Bounded in-memory conversation state for history-aware RAG."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage


def recent_complete_turns(
    messages: Sequence[BaseMessage], max_turns: int
) -> list[BaseMessage]:
    """Return at most ``max_turns`` complete user/assistant message pairs."""
    if max_turns < 1:
        raise ValueError("max_turns 必须大于等于 1")

    retained = list(messages)[-(max_turns * 2):]
    return retained if len(retained) % 2 == 0 else retained[1:]


class ConversationRagService:
    """Use conversation history only to form a retrieval question for one turn."""

    def __init__(
        self, rag_chain, rewriter, history, max_turns: int = 4, decomposer=None
    ):
        self.rag_chain = rag_chain
        self.rewriter = rewriter
        self.history = history
        self.max_turns = max_turns
        self.decomposer = decomposer

    def ask(self, question: str) -> dict:
        original_question = question.strip()
        if not original_question:
            raise ValueError("问题不能为空")

        history = recent_complete_turns(self.history.messages, self.max_turns)
        retrieval_question = self.rewriter.rewrite(original_question, history)
        state = {
            "question": original_question,
            "retrieval_question": retrieval_question,
        }
        subquestions = None
        if self.decomposer is not None:
            subquestions = self.decomposer.decompose(retrieval_question)
            if len(subquestions) > 1:
                state["retrieval_questions"] = subquestions
        result = dict(self.rag_chain.invoke(state))
        result["retrieval_question"] = retrieval_question
        if subquestions is not None:
            result["subquestions"] = subquestions

        self.history.add_user_message(original_question)
        self.history.add_ai_message(result["answer"])
        self.history.messages[:] = recent_complete_turns(
            self.history.messages, self.max_turns
        )
        return result
