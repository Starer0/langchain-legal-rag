"""Permission-preserving graph adapter for the existing web event contract."""
from typing import Any, TypedDict
from threading import Event

from langgraph.config import get_stream_writer
from langgraph.graph import START, END, StateGraph
from langgraph.runtime import get_runtime
from graph_runtime import GraphRuntime, restore_runtime

from web_rag import StreamingRagTurn
from memory_service import command_candidate
from web_intent import is_style_preference, has_legal_request


class WebRagState(TypedDict, total=False):
    question: str
    original_question: str
    history: list
    allowed_knowledge_bases: frozenset[str]
    retrieval_question: str
    include_guide: bool | None
    candidates: list
    docs: list
    timings: dict
    graph_version: str
    _completion: dict
    memory_input: dict
    memory: dict
    context_input: dict
    context_result: dict
    decision: dict
    pending_memory: dict
    understanding_error: str


class LangGraphStreamingRagTurn(StreamingRagTurn):
    """One reusable graph, fresh request-owned State per invocation; no checkpoint."""

    VERSION = 'web-rag-intent-v1'

    def __init__(self, checkpointer=None, understanding=None, **components):
        super().__init__(**components)
        self.understanding=understanding
        if understanding is not None:self.VERSION='web-rag-model-memory-v1'
        self.checkpointer = checkpointer
        graph = StateGraph(WebRagState, context_schema=GraphRuntime)
        for name, operation in [('rewrite', self._rewrite_state), ('retrieve', self._retrieve_state), ('rerank', self._rerank_state)]:
            graph.add_node(name, self._stage(name, operation))
        graph.add_node('answer', self._answer_node)
        graph.add_node('no_evidence', self._no_evidence_node)
        graph.add_node('memory', self._memory_node)
        graph.add_node('memory_command', self._memory_command_node)
        graph.add_node('context', self._context_node)
        graph.add_node('preference', self._preference_node)
        graph.add_node('understand',self._understand_node)
        graph.add_node('reply',self._reply_node)
        if understanding is not None:
            graph.add_conditional_edges(START,lambda state:'context' if get_runtime().context.context_service else 'understand',
                {'context':'context','understand':'understand'})
            graph.add_edge('context','understand')
            graph.add_conditional_edges('understand',lambda state:'retrieve' if state.get('decision',{}).get('answer',{}).get('route')=='rag' else 'reply',
                {'retrieve':'retrieve','reply':'reply'})
            graph.add_edge('reply',END)
        else:
            graph.add_conditional_edges(START, lambda state: 'memory_command' if
            get_runtime().context.memory_service is not None and command_candidate(state['question']) and not has_legal_request(state['question']) else
            ('preference' if is_style_preference(state['question']) else
             ('context' if get_runtime().context.context_service is not None else 'rewrite')),
            {'memory_command': 'memory_command', 'rewrite': 'rewrite', 'context':'context', 'preference':'preference'})
            graph.add_edge('context', 'rewrite')
        graph.add_edge('rewrite', 'retrieve')
        graph.add_edge('retrieve', 'rerank')
        graph.add_conditional_edges('rerank', lambda state: 'no_evidence' if self._has_no_evidence(state) else
            ('memory' if get_runtime().context.memory_service is not None else 'answer'),
            {'answer': 'answer', 'no_evidence': 'no_evidence', 'memory': 'memory'})
        graph.add_edge('memory', 'answer')
        graph.add_edge('memory_command', END)
        graph.add_edge('preference', END)
        graph.add_edge('answer', END)
        graph.add_edge('no_evidence', END)
        self.graph = graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _stage(name, operation):
        def run(state):
            runtime = get_runtime().context
            if runtime.cancelled.is_set(): raise GeneratorExit('Graph execution cancelled')
            if runtime.guard: runtime.guard()
            get_stream_writer()({'event': 'status', 'data': {'stage': name}})
            legal_state = {k: v for k, v in state.items() if not k.startswith(('memory','context')) and
                           k not in ('decision','pending_memory','original_question','understanding_error')}
            update = operation(runtime.enrich(legal_state))
            if runtime.cancelled.is_set(): raise RuntimeError('Graph execution stopped before checkpoint')
            if runtime.guard: runtime.guard()
            update['timings'] = {k: dict(v) for k, v in runtime.profile.stages.items()}
            return update
        return run

    def _emit_answer(self, events):
        writer = get_stream_writer()
        completion = None
        for event in events:
            if event['event'] == 'done': completion = event['data']
            else: writer(event)
        if completion is None: raise RuntimeError('Answer step ended without completion')
        runtime = get_runtime().context
        if runtime.cancelled.is_set(): raise RuntimeError('Graph execution stopped before completion')
        if runtime.guard: runtime.guard()
        return {'_completion': completion}

    def _context_node(self, state):
        from conversation_context import Summary, SourceMessage
        from langchain_core.messages import HumanMessage
        runtime=get_runtime().context
        if runtime.guard: runtime.guard()
        if runtime.cancelled.is_set(): raise GeneratorExit('Cancelled before context')
        get_stream_writer()({'event':'status','data':{'stage':'context'}})
        frozen=state['context_input']
        result=runtime.context_cached
        if result is None:
            result=runtime.context_service.build(Summary(**frozen['summary']),[SourceMessage(**m) for m in frozen.get('legal_messages',frozen['messages'])])
            if runtime.guard: runtime.guard()
            if runtime.cancelled.is_set(): raise RuntimeError('Cancelled after context generation')
            if runtime.context_save: result=runtime.context_save(result)
        if runtime.guard: runtime.guard()
        if runtime.cancelled.is_set(): raise RuntimeError('Cancelled after context save')
        if runtime.trace is not None:
            runtime.trace.emit('context_prepared',compressed=result['compressed'],revision=result['summary']['revision'],
                               estimated_tokens=result['estimated_tokens'],duration_ms=result['duration_ms'],usage=result.get('usage',{}))
        return {'context_result':result,'history':[HumanMessage(s) for s in result['history']]}

    @staticmethod
    def _context_completion(state, completion):
        if 'decision' in state:
            from web_understanding import with_scope_notice
            completion['decision']=state['decision']
            if 'answer' in completion:completion['answer']=with_scope_notice(completion['answer'],state['decision']['answer'])
        if 'pending_memory' in state:completion['pending_memory']=state['pending_memory']
        if 'understanding_error' in state:
            completion['memory']={'status':'blocked','revision':state.get('memory_input',{}).get('revision',0),
                'message':state['understanding_error'],'warning':state['understanding_error']}
        if 'context_result' in state:
            result=state['context_result']
            completion['context']={'compressed':result['compressed'],'revision':result['summary']['revision'],
                                   'estimated_tokens':result['estimated_tokens']}
        return completion

    def _understand_node(self,state):
        from dataclasses import asdict,replace
        from memory_service import MemoryValidation
        runtime=get_runtime().context
        if runtime.guard:runtime.guard()
        if runtime.cancelled.is_set():raise GeneratorExit('Cancelled before understanding')
        get_stream_writer()({'event':'status','data':{'stage':'understand'}})
        frozen=state.get('context_input',{})
        recent=[];used=0
        from memory_service import estimate_tokens
        for message in reversed(frozen.get('messages',[])):
            if message['role']!='user':continue
            needed=estimate_tokens(message['content'])
            if used+needed>1000:break
            recent.insert(0,message);used+=needed
        context={'summary':state.get('context_result',{}).get('summary',frozen.get('summary',{})),
                 'recent_user_messages':recent,'sources':runtime.tool_context.input_snapshot['sources'] if runtime.tool_context else []}
        if hasattr(self.rewriter,'guide_catalog'):
            scope=state.get('allowed_knowledge_bases')
            context['guide_catalog']=[entry for entry,label in zip(self.rewriter.guide_catalog,self.rewriter.guide_scopes) if scope is None or label in scope]
        decision=runtime.decision_cached
        if decision is None:
            try:decision=self.understanding.decide(state['question'],context,state.get('memory_input',{}))
            except ValueError as error:
                if runtime.trace:
                    from web_understanding import rejection_diagnostics
                    runtime.trace.emit('understanding_rejected',**rejection_diagnostics(error))
                from memory_tool_contract import TurnDecision,AnswerPlan
                decision=TurnDecision(AnswerPlan('clarify','','',False,'暂时无法判断这条消息的意图，请换一种说法。本次未保存记忆。',False),None)
            if runtime.guard:runtime.guard()
            if runtime.cancelled.is_set():raise RuntimeError('Cancelled after understanding')
            if runtime.decision_save:runtime.decision_save(decision)
        update={'decision':asdict(decision),'retrieval_question':decision.answer.retrieval_question,
                'include_guide':decision.answer.include_guide}
        if decision.answer.route=='rag':
            # The original input stays frozen for provenance; RAG sees only its legal portion.
            update['question']=decision.answer.legal_question
            if runtime.trace:runtime.trace.begin(decision.answer.legal_question)
        if decision.memory_call is not None:
            if runtime.memory_tools is None or runtime.tool_context is None:
                update['understanding_error']='记忆工具暂不可用，本次未保存。'
            else:
                context=replace(runtime.tool_context,input_snapshot={**runtime.tool_context.input_snapshot,'memory_request':decision.answer.memory_request})
                try:update['pending_memory']=asdict(runtime.memory_tools.prepare(decision.memory_call,context))
                except MemoryValidation as error:update['understanding_error']=str(error)
                except Exception:update['understanding_error']='记忆准备失败，本次未保存；法律回答仍可继续。'
        elif decision.answer.memory_request:
            update['understanding_error']='未收到可验证的记忆更新，本次未保存；请换一种说法或在设置中编辑。'
        if runtime.guard:runtime.guard()
        if runtime.cancelled.is_set():raise RuntimeError('Cancelled before decision checkpoint')
        return update

    def _reply_node(self,state):
        runtime=get_runtime().context
        if runtime.guard:runtime.guard()
        if runtime.cancelled.is_set():raise GeneratorExit('Cancelled before reply')
        from web_understanding import direct_reply
        answer=direct_reply(state['decision']['answer']['reply_plan']) or '收到，请继续提出问题。'
        route=state['decision']['answer']['route']
        if route in ('general','out_of_scope'):
            from web_understanding import OUT_OF_SCOPE_REPLY,general_reply
            answer=OUT_OF_SCOPE_REPLY if route=='out_of_scope' else general_reply(state['decision']['answer'].get('help_topic'))
        elif route=='preference' and state['decision']['answer'].get('out_of_scope_request'):
            answer='收到你的表达要求。'
        if 'pending_memory' in state:answer='正在处理记忆更新，请以完成后的结果为准。'
        if state.get('understanding_error'):answer=state['understanding_error']
        get_stream_writer()({'event':'delta','data':{'text':answer}})
        return {'_completion':self._context_completion(state,{'answer':answer,'sources':[],'retrieval_question':'','candidates':[]})}

    def _answer_node(self, state):
        get_stream_writer()({'event': 'status', 'data': {'stage': 'answer'}})
        result=self._emit_answer(self._answer_events(get_runtime().context.enrich(state)))
        result['_completion']=self._context_completion(state,result['_completion'])
        return result

    def _no_evidence_node(self, state):
        result=self._emit_answer(self._no_evidence_events(get_runtime().context.enrich(state)))
        result['_completion']=self._context_completion(state,result['_completion'])
        return result

    def _memory_node(self, state):
        from time import perf_counter
        runtime = get_runtime().context
        if runtime.guard: runtime.guard()
        if runtime.cancelled.is_set(): raise GeneratorExit('Cancelled before memory')
        get_stream_writer()({'event': 'status', 'data': {'stage': 'memory'}})
        started = perf_counter()
        result = runtime.memory_service.select(state['question'], state.get('memory_input', {}))
        if runtime.guard: runtime.guard()
        if runtime.cancelled.is_set(): raise RuntimeError('Cancelled after memory')
        elapsed = (perf_counter()-started)*1000
        if runtime.trace is not None:
            from memory_service import estimate_tokens
            runtime.trace.emit('memory_selected',
                revision=result['revision'],selected_ids=result['selected_ids'],selected_count=len(result['selected_ids']),
                core_estimated_tokens=estimate_tokens(result['core']),extended_estimated_tokens=estimate_tokens(result['extended']),
                status=result['status'],duration_ms=elapsed)
        return {'memory': result}

    def _memory_command_node(self, state):
        runtime = get_runtime().context
        if runtime.guard: runtime.guard()
        if runtime.cancelled.is_set(): raise GeneratorExit('Cancelled before memory command')
        get_stream_writer()({'event': 'status', 'data': {'stage': 'memory_command'}})
        proposal = runtime.memory_service.propose(state['question'], state.get('memory_input', {}))
        if runtime.guard: runtime.guard()
        if runtime.cancelled.is_set(): raise RuntimeError('Cancelled after memory proposal')
        result = {'answer': proposal['answer'], 'sources': [], 'retrieval_question': '', 'candidates': []}
        if 'prepared' in proposal: result['memory_proposal'] = proposal
        return {'_completion': result}

    def _preference_node(self, state):
        runtime=get_runtime().context
        if runtime.guard:runtime.guard()
        if runtime.cancelled.is_set():raise GeneratorExit('Cancelled before preference response')
        writer=get_stream_writer()
        writer({'event':'status','data':{'stage':'preference'}})
        snapshot=state.get('memory_input',{})
        if not snapshot.get('auto_accumulate',False):
            notice='当前未开启自动积累，不会自动保存这条偏好。'
        elif not snapshot.get('enabled',True):
            notice='后台积累已暂停，不会自动保存这条偏好；开启并保存“在回答中使用记忆”后才能继续积累。'
        else:
            notice='自动积累已开启；服务器会在对话空闲约两分钟后整理，是否加入长期记忆请以设置中的结果为准。'
        answer='收到你的回答风格偏好。'+notice+'你可以继续提出法律问题。'
        writer({'event':'delta','data':{'text':answer}})
        if runtime.trace is not None:runtime.trace.emit('intent_routed',intent='style_preference',method='rules')
        if runtime.guard:runtime.guard()
        if runtime.cancelled.is_set():raise RuntimeError('Cancelled before preference completion')
        return {'_completion':{'answer':answer,'sources':[],'retrieval_question':'','candidates':[]}}

    def stream(self, question, messages, *, allowed_knowledge_bases=None, trace=None,
               task_id=None, resume=False, cancelled=None, execution_epoch=None, guard=None,
               memory_input=None, memory_service=None, memory_revision_guard=None,
               context_input=None, context_service=None, context_save=None, context_cached=None, context_revision_guard=None,
               memory_tools=None,tool_context=None,decision_cached=None,decision_save=None):
        from langchain_core.messages import HumanMessage
        retained=[];skip=False
        for message in messages:
            if isinstance(message,HumanMessage):
                skip=self.understanding is None and (is_style_preference(message.content) or (memory_service is not None and command_candidate(message.content) and not has_legal_request(message.content)))
            if not skip:retained.append(message)
        messages=retained
        preparation_options = {'private_question': True} if self.understanding is not None or is_style_preference(question) or (memory_service is not None and command_candidate(question) and not has_legal_request(question)) else {}
        prepared = self._prepare_state(question, messages, allowed_knowledge_bases, trace, **preparation_options)
        prepared['_cancelled'] = cancelled or Event()
        runtime = GraphRuntime(prepared['_profile'], trace, prepared['_started'], prepared['_cancelled'], guard)
        runtime.memory_service = memory_service
        runtime.context_service, runtime.context_save, runtime.context_cached = context_service, context_save, context_cached
        runtime.memory_tools,runtime.tool_context=memory_tools,tool_context
        runtime.decision_cached,runtime.decision_save=decision_cached,decision_save
        state = {k: v for k, v in prepared.items() if not k.startswith('_')}
        if memory_input is not None: state['memory_input'] = memory_input
        if context_input is not None: state['context_input']=context_input
        state['graph_version'] = self.VERSION
        state['original_question']=state['question']
        config = {'configurable': {'thread_id': task_id}} if task_id else None
        if config is not None and execution_epoch is not None:
            config['configurable']['execution_epoch'] = execution_epoch
        if self.checkpointer is not None and not task_id:
            raise ValueError('A server task ID is required for checkpoint execution')
        if resume:
            if self.checkpointer is None: raise ValueError('Recovery requires a checkpointer')
            snapshot = self.graph.get_state(config)
            saved = snapshot.values
            if saved:
                if (saved.get('original_question',saved.get('question')) != state['question'] or
                    saved.get('allowed_knowledge_bases') != state.get('allowed_knowledge_bases') or
                    saved.get('graph_version') != self.VERSION):
                    raise ValueError('Checkpoint identity, permission or graph version mismatch')
                if not snapshot.next:
                    result = saved.get('_completion')
                    if not result: raise ValueError('Finished checkpoint has no result')
                    yield {'event': 'done', 'data': result}
                    return
                if memory_revision_guard: memory_revision_guard()
                if context_revision_guard: context_revision_guard()
                runtime = restore_runtime(saved, trace, runtime.cancelled, guard)
                runtime.memory_service = memory_service
                runtime.context_service, runtime.context_save, runtime.context_cached = context_service, context_save, context_cached
                runtime.memory_tools,runtime.tool_context=memory_tools,tool_context
                runtime.decision_cached,runtime.decision_save=decision_cached,decision_save
                state = None
            else:
                if memory_revision_guard: memory_revision_guard()
                if context_revision_guard: context_revision_guard()
        completion = None
        # Only explicitly emitted safe events reach the web, never rewrite tokens or State.
        stream = self.graph.stream(state, config=config, context=runtime,
            stream_mode=['custom', 'updates'], durability='sync' if self.checkpointer else None)
        try:
            for mode, value in stream:
                if mode == 'custom': yield value
                elif mode == 'updates':
                    for update in value.values():
                        if isinstance(update, dict) and '_completion' in update:
                            completion = update['_completion']
        finally:
            runtime.cancelled.set()
            stream.close()
        if completion is None: raise RuntimeError('Graph ended without completion')
        # Do not commit a completed turn until the graph actually finishes successfully.
        yield {'event': 'done', 'data': completion}

    def delete_checkpoints(self, task_id):
        if self.checkpointer is not None: self.checkpointer.delete_thread(task_id)
