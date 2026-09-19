"""Rewrite user questions into standalone legal-retrieval queries."""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


SINGLE_QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "将用户问题改写为简洁、适合检索的法律问题。"
        "不得改变问题范围；只输出改写后的问题。",
    ),
    ("human", "{question}"),
])

HISTORY_AWARE_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "根据历史对话把当前追问补全为独立、适合检索的法律问题。"
        "此前助手消息只是对话上下文，不是法律依据；不得据此虚构事实。"
        "只输出改写后的问题。",
    ),
    MessagesPlaceholder("history"),
    ("human", "当前问题：{question}"),
])


class RetrievalQuestionRewriter:
    """Create a standalone retrieval question from a user turn and optional history."""

    def __init__(self, model):
        self.model = model

    def rewrite(self, question: str, history: Sequence[BaseMessage]) -> str:
        prompt = (
            HISTORY_AWARE_REWRITE_PROMPT
            if history
            else SINGLE_QUERY_REWRITE_PROMPT
        )
        values = {"question": question}
        if history:
            values["history"] = list(history)

        response = self.model.invoke(prompt.invoke(values))
        content = response.content if hasattr(response, "content") else response
        rewritten = str(content).strip()
        return rewritten or question.strip()
