"""Streaming, explicit-history version of the legal RAG turn."""

from collections.abc import Iterator, Sequence
from time import perf_counter

from langchain_core.messages import BaseMessage

from conversation import recent_complete_turns
from performance import TurnProfile
from rag_pipeline_articles import format_docs, format_sources


class StreamingRagTurn:
    """Prepare legal evidence, then stream one final answer without owning history."""

    def __init__(
        self, *, rewriter, retriever, reranker, prompt, model, history_turns: int
    ):
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.prompt = prompt
        self.model = model
        self.history_turns = history_turns

    def stream(
        self, question: str, messages: Sequence[BaseMessage]
    ) -> Iterator[dict]:
        original_question = question.strip()
        if not original_question:
            raise ValueError("问题不能为空")

        started = perf_counter()
        profile = TurnProfile()
        history = recent_complete_turns(messages, self.history_turns)

        yield {"event": "status", "data": {"stage": "rewrite"}}
        retrieval_question = profile.measure(
            "rewrite",
            lambda: self.rewriter.rewrite(original_question, history),
        )
        state = {
            "question": original_question,
            "retrieval_question": retrieval_question,
            "_profile": profile,
        }

        yield {"event": "status", "data": {"stage": "retrieve"}}
        candidates = self.retriever.invoke(state)

        yield {"event": "status", "data": {"stage": "rerank"}}
        docs = self.reranker.invoke({**state, "candidates": candidates})

        yield {"event": "status", "data": {"stage": "answer"}}
        answer_parts = []
        answer_started = perf_counter()
        for chunk in self.model.stream(self.prompt.invoke({
            "context": format_docs(docs),
            "question": original_question,
        })):
            text = _chunk_text(chunk)
            if text:
                answer_parts.append(text)
                yield {"event": "delta", "data": {"text": text}}
        profile.stages["answer"] = {
            "calls": 1,
            "duration_ms": (perf_counter() - answer_started) * 1000,
        }

        answer = "".join(answer_parts)
        yield {
            "event": "done",
            "data": {
                "answer": answer,
                "retrieval_question": retrieval_question,
                "candidates": format_sources(candidates),
                "sources": format_sources(docs),
                "performance": profile.snapshot(
                    total_ms=(perf_counter() - started) * 1000
                ),
            },
        }


def _chunk_text(chunk) -> str:
    content = chunk.content if hasattr(chunk, "content") else chunk
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_chunk_text(item) for item in content)
    return str(content) if content else ""
