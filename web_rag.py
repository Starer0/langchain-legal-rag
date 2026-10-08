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
        state = self._prepare_state(question, messages, allowed_knowledge_bases, trace)
        for stage, operation in [('rewrite', self._rewrite_state), ('retrieve', self._retrieve_state), ('rerank', self._rerank_state)]:
            yield {'event': 'status', 'data': {'stage': stage}}
            state.update(operation(state))
        if self._has_no_evidence(state):
            yield from self._no_evidence_events(state)
        else:
            yield {'event': 'status', 'data': {'stage': 'answer'}}
            yield from self._answer_events(state)

    def _prepare_state(self, question, messages, allowed_knowledge_bases, trace, *, private_question=False):
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
            if private_question: trace.begin(original_question, private=True)
            else: trace.begin(original_question)

        started = perf_counter()
        profile = TurnProfile()
        history = recent_complete_turns(messages, self.history_turns)
        if scope is not None:
            history = [message for message in history if isinstance(message, HumanMessage)]
        state = {'question': original_question, 'history': history,
                 '_profile': profile, '_trace': trace, '_started': started}
        if scope is not None: state['allowed_knowledge_bases'] = scope
        return state

    def _rewrite_state(self, state):
        trace, profile = state['_trace'], state['_profile']
        scope = state.get('allowed_knowledge_bases')
        rewrite_options = {'allowed_knowledge_bases': scope} if scope is not None else {}
        with _logged_stage(trace, 'rewrite') as details:
            plan = profile.measure(
                "rewrite",
                lambda: self.rewriter.rewrite(state['question'], state['history'], **rewrite_options),
            )
            if trace is not None:
                value = logged_text(plan.retrieval_question)
                details.update(retrieval_question=value['text'], retrieval_question_chars=value['chars'],
                               retrieval_question_truncated=value['truncated'], include_guide=plan.include_guide)
        return {'retrieval_question': plan.retrieval_question, 'include_guide': plan.include_guide}

    def _retrieve_state(self, state):
        trace, profile = state['_trace'], state['_profile']
        scope = state.get('allowed_knowledge_bases')
        with _logged_stage(trace, 'retrieve') as details:
            candidates = self.retriever.invoke(state)
            if scope is not None:
                candidates = accessible_documents(candidates, scope)
            if trace is not None:
                details.update(trace.documents(candidates))
                details['performance'] = profile.snapshot()
        return {'candidates': candidates}

    def _rerank_state(self, state):
        trace = state['_trace']
        scope, candidates = state.get('allowed_knowledge_bases'), state['candidates']
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
        return {'docs': docs}

    def _has_no_evidence(self, state):
        return state.get('allowed_knowledge_bases') is not None and not state['docs']

    def _no_evidence_events(self, state):
        trace = state['_trace']
        if trace is not None: trace.skipped('answer', 'no_accessible_evidence')
        answer = '当前账号可访问的资料中没有足够依据。'
        yield {'event': 'delta', 'data': {'text': answer}}
        yield {'event': 'done', 'data': self._completion(state, answer)}

    def _answer_events(self, state):
        trace, profile, docs = state['_trace'], state['_profile'], state['docs']
        answer_parts = []
        answer_started = perf_counter()
        chunks = None
        with _logged_stage(trace, 'answer') as details:
            if state.get('_cancelled') is not None and state['_cancelled'].is_set():
                raise GeneratorExit('Graph stream observer closed')
            if trace is not None:
                details.update(trace.documents(docs))
            values = {
                "context": format_docs(docs),
                "question": state['question'],
            }
            if 'memory' in state:
                memory = state['memory']
                values['memory'] = '核心记忆：\n' + memory['core'] + '\n\n相关背景：\n' + memory['extended']
                style=state.get('decision',{}).get('answer',{}).get('reply_plan','')
                if style:values['memory']+='\n本轮表达要求（不作为长期记忆）：\n'+style+'\n只应用表达要求，法律结论仍必须来自授权资料。记忆提交结果会另行展示，回答中不要声称已保存或已更新记忆。'
            rendered_prompt = self.prompt.invoke(values)
            if getattr(self, 'prompt_budget', None) is not None: self.prompt_budget.check(rendered_prompt)
            chunks = self.model.stream(rendered_prompt)
            try:
                for chunk in chunks:
                    if getattr(chunk,'response_metadata',{}).get('finish_reason')=='length':
                        from conversation_context import ContextTooLong
                        raise ContextTooLong('回答达到输出长度上限，尚未完整，请缩小问题范围。')
                    if state.get('_cancelled') is not None and state['_cancelled'].is_set():
                        raise GeneratorExit('Graph stream observer closed')
                    text = _chunk_text(chunk)
                    if text:
                        answer_parts.append(text)
                        yield {"event": "delta", "data": {"text": text}}
            finally:
                if hasattr(chunks, 'close'): chunks.close()
            if state.get('_cancelled') is not None and state['_cancelled'].is_set():
                raise GeneratorExit('Graph stream observer closed')
            details['answer_chars'] = sum(len(part) for part in answer_parts)
        profile.stages["answer"] = {
            "calls": 1,
            "duration_ms": (perf_counter() - answer_started) * 1000,
        }

        answer = "".join(answer_parts)
        yield {
            "event": "done",
            "data": self._completion(state, answer),
        }

    def _completion(self, state, answer):
        result = {'answer': answer, 'retrieval_question': state['retrieval_question'],
                'candidates': format_sources(state['candidates']), 'sources': format_sources(state['docs']),
                'performance': state['_profile'].snapshot(total_ms=(perf_counter() - state['_started']) * 1000)}
        if 'memory' in state:
            result['memory'] = {k: state['memory'][k] for k in ('revision', 'status', 'warning')}
        return result


def _logged_stage(trace, stage):
    return trace.stage(stage) if trace is not None else nullcontext({})


def _chunk_text(chunk) -> str:
    content = chunk.content if hasattr(chunk, "content") else chunk
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_chunk_text(item) for item in content)
    return str(content) if content else ""
