"""Rewrite user questions and select whether procedural guides are needed."""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


@dataclass(frozen=True)
class RetrievalPlan:
    retrieval_question: str
    include_guide: bool | None


_ROUTING_INSTRUCTIONS = """
将用户问题改写为简洁、适合检索的法律问题。不得改变问题范围；保留用户明确提到的法律名称和法条号。

根据资料目录判断是否需要同时检索办事指南：
- true：问题涉及目录所列指南的办理流程、申请材料、提交、时限、费用或常见问题；
- false：问题只需要法律条文；
- null：无法可靠判断。

此前助手消息只是对话上下文，不是法律依据；不得据此虚构事实。
只输出 JSON：{{"retrieval_question":"...","include_guide":true、false 或 null}}。
资料目录：{guide_catalog}
""".strip()

SINGLE_QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _ROUTING_INSTRUCTIONS),
    ("human", "{question}"),
])

HISTORY_AWARE_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _ROUTING_INSTRUCTIONS),
    MessagesPlaceholder("history"),
    ("human", "当前问题：{question}"),
])


class RetrievalQuestionRewriter:
    """Create one retrieval plan from a user turn and optional history."""

    def __init__(self, model, guides=()):
        self.model = model
        self.guide_catalog = [
            {
                "document_type": guide["document_type"],
                "title": guide["title"],
                "routing_topics": guide.get("routing_topics", []),
            }
            for guide in guides
        ]

    def rewrite(self, question: str, history: Sequence[BaseMessage]) -> RetrievalPlan:
        prompt = HISTORY_AWARE_REWRITE_PROMPT if history else SINGLE_QUERY_REWRITE_PROMPT
        values = {
            "question": question,
            "guide_catalog": json.dumps(self.guide_catalog, ensure_ascii=False),
        }
        if history:
            values["history"] = list(history)

        response = self.model.invoke(prompt.invoke(values))
        content = response.content if hasattr(response, "content") else response
        return _parse_plan(str(content), question)


def _parse_plan(content: str, question: str) -> RetrievalPlan:
    match = re.search(r"\{.*\}", content.strip(), re.DOTALL)
    if not match:
        return RetrievalPlan(question.strip(), include_guide=None)
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return RetrievalPlan(question.strip(), include_guide=None)
    rewritten = payload.get("retrieval_question")
    include_guide = payload.get("include_guide")
    if not isinstance(rewritten, str) or not rewritten.strip():
        rewritten = question.strip()
    if not isinstance(include_guide, bool):
        include_guide = None
    return RetrievalPlan(rewritten.strip(), include_guide)
