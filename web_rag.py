"""Streaming, explicit-history version of the legal RAG turn."""

from collections.abc import Iterator, Sequence
from copy import deepcopy
from contextlib import nullcontext
from time import perf_counter

from langchain_core.messages import BaseMessage, HumanMessage

from conversation import recent_complete_turns
from performance import TurnProfile
from rag_pipeline_articles import format_docs, format_sources
from rag_permissions import validated_scope, accessible_documents, verified_reranked_documents
from rag_logging import logged_text


class StreamingRagTurn:
    """Prepare legal evidence, then stream one final answer without owning history."""

    def __init__(
        self, *, rewriter, retriever, reranker, prompt, model, history_turns: int,
        require_authorization: bool = False,
    ):
        self.rewriter = rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.prompt = prompt
        self.model = model
        self.history_turns = history_turns
        self.require_authorization = require_authorization

    def stream(
        self, question: str, messages: Sequence[BaseMessage], *, allowed_knowledge_bases=None, trace=None
    ) -> Iterator[dict]:
        original_question = question.strip()
        if not original_question:
            raise ValueError("问题不能为空")
        scope = None
        if self.require_authorization or allowed_knowledge_bases is not None:
            scope = validated_scope(allowed_knowledge_bases)
            if not scope:
                raise ValueError('当前账号没有可访问的资料库')
        if trace is not None:
            if scope is None or trace.scope != scope:
                raise ValueError('日志范围必须与本次服务器权限一致')
            trace.begin(original_question)

        started = perf_counter()
        profile = TurnProfile()
        history = recent_complete_turns(messages, self.history_turns)
        rewrite_options = {}
        if scope is not None:
            history = [message for message in history if isinstance(message, HumanMessage)]
            rewrite_options['allowed_knowledge_bases'] = scope

        yield {"event": "status", "data": {"stage": "rewrite"}}
        with _logged_stage(trace, 'rewrite') as details:
            plan = profile.measure(
                "rewrite",
                lambda: self.rewriter.rewrite(original_question, history, **rewrite_options),
            )
            if trace is not None:
                value = logged_text(plan.retrieval_question)
                details.update(retrieval_question=value['text'], retrieval_question_chars=value['chars'],
                               retrieval_question_truncated=value['truncated'], include_guide=plan.include_guide)
        retrieval_question = plan.retrieval_question
        state = {
            "question": original_question,
            "retrieval_question": retrieval_question,
            "include_guide": plan.include_guide,
            "_profile": profile,
        }
        if scope is not None:
            state['allowed_knowledge_bases'] = scope

        yield {"event": "status", "data": {"stage": "retrieve"}}
        with _logged_stage(trace, 'retrieve') as details:
            candidates = self.retriever.invoke(state)
            if scope is not None:
                candidates = accessible_documents(candidates, scope)
            if trace is not None:
                details.update(trace.documents(candidates))
                details['performance'] = profile.snapshot()

        yield {"event": "status", "data": {"stage": "rerank"}}
        if candidates or scope is None:
            with _logged_stage(trace, 'rerank') as details:
                rerank_candidates = deepcopy(candidates) if scope is not None else candidates
                docs = self.reranker.invoke({**state, "candidates": rerank_candidates})
                if scope is not None:
                    docs = verified_reranked_documents(docs, candidates, scope)
                if trace is not None:
                    details.update(trace.documents(docs))
        else:
            docs = []
            if trace is not None:
                trace.skipped('rerank', 'no_accessible_candidates')

        if scope is not None and not docs:
            if trace is not None:
                trace.skipped('answer', 'no_accessible_evidence')
            answer = '当前账号可访问的资料中没有足够依据。'
            yield {'event': 'delta', 'data': {'text': answer}}
            yield {'event': 'done', 'data': {
                'answer': answer, 'retrieval_question': retrieval_question,
                'candidates': format_sources(candidates), 'sources': [],
                'performance': profile.snapshot(total_ms=(perf_counter() - started) * 1000),
            }}
            return

        yield {"event": "status", "data": {"stage": "answer"}}
        answer_parts = []
        answer_started = perf_counter()
        with _logged_stage(trace, 'answer') as details:
            if trace is not None:
                details.update(trace.documents(docs))
            for chunk in self.model.stream(self.prompt.invoke({
                "context": format_docs(docs),
                "question": original_question,
            })):
                text = _chunk_text(chunk)
                if text:
                    answer_parts.append(text)
                    yield {"event": "delta", "data": {"text": text}}
            details['answer_chars'] = sum(len(part) for part in answer_parts)
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


def _logged_stage(trace, stage):
    return trace.stage(stage) if trace is not None else nullcontext({})


def _chunk_text(chunk) -> str:
    content = chunk.content if hasattr(chunk, "content") else chunk
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_chunk_text(item) for item in content)
    return str(content) if content else ""
