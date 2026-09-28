"""Select answerable subquestions from already reranked evidence."""

import json
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


EVIDENCE_SELECTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是检索证据筛选器，不回答用户问题。判断每个子问题是否能被给出的候选资料直接支持。"
        "只有候选资料能直接支持回答时才标记 answerable=true；资料无关、仅有相近概念或证据不足时标记 false。"
        "为每个 true 的子问题至少选择一条属于它的资料；false 的子问题不得选择资料。"
        "只可使用给定 document_id，最多选择 {top_n} 条，优先覆盖所有可答子问题。"
        "只输出 JSON："
        '{{"subquestions":[{{"index":1,"answerable":true}}],'
        '"selected_document_ids":["q1-d1"]}}。',
    ),
    ("human", "原始问题：{question}\n\n候选证据：\n{evidence}"),
])


@dataclass(frozen=True)
class EvidenceSelection:
    answerable: list[bool]
    selected_document_ids: list[str]
    documents: list[Document]


class EvidenceSelector:
    """Use a short model call to reject unsupported subquestion evidence."""

    def __init__(self, model, candidates_per_question: int = 2, max_chars: int = 800):
        self.model = model
        self.candidates_per_question = candidates_per_question
        self.max_chars = max_chars

    def select(self, original_question, questions, ranked_batches, top_n: int):
        if len(questions) != len(ranked_batches) or top_n < 1:
            return None

        document_map = {}
        lines = []
        for question_index, (question, batch) in enumerate(
            zip(questions, ranked_batches), start=1
        ):
            lines.append(f"子问题 {question_index}：{question}")
            for document_index, document in enumerate(
                batch[:self.candidates_per_question], start=1
            ):
                document_id = f"q{question_index}-d{document_index}"
                document_map[document_id] = (question_index, document)
                article = document.metadata.get("article")
                section = document.metadata.get("section")
                source = article or section or document.metadata.get("title", "未标注来源")
                excerpt = document.page_content[:self.max_chars]
                lines.append(
                    f"- document_id: {document_id}\n  来源：{source}\n  内容：{excerpt}"
                )

        response = self.model.invoke(EVIDENCE_SELECTION_PROMPT.invoke({
            "question": original_question,
            "top_n": top_n,
            "evidence": "\n".join(lines),
        }))
        content = response.content if hasattr(response, "content") else response
        return _parse_selection(str(content), len(questions), document_map, top_n)


def _parse_selection(content, question_count, document_map, top_n):
    match = re.search(r"\{.*\}", content.strip(), re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    entries = payload.get("subquestions")
    document_ids = payload.get("selected_document_ids")
    if not isinstance(entries, list) or not isinstance(document_ids, list):
        return None
    if len(entries) != question_count or len(document_ids) > top_n:
        return None

    answerable = []
    for index, entry in enumerate(entries, start=1):
        if (
            not isinstance(entry, dict)
            or entry.get("index") != index
            or not isinstance(entry.get("answerable"), bool)
        ):
            return None
        answerable.append(entry["answerable"])

    if (
        any(not isinstance(document_id, str) for document_id in document_ids)
        or len(set(document_ids)) != len(document_ids)
        or any(document_id not in document_map for document_id in document_ids)
    ):
        return None

    selected_by_question = {index: [] for index in range(1, question_count + 1)}
    for document_id in document_ids:
        selected_by_question[document_map[document_id][0]].append(document_id)
    if any(
        bool(selected_by_question[index]) != answerable[index - 1]
        for index in range(1, question_count + 1)
    ):
        return None

    return EvidenceSelection(
        answerable=answerable,
        selected_document_ids=document_ids,
        documents=[document_map[document_id][1] for document_id in document_ids],
    )
