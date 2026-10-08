"""Bounded startup recovery and fresh generation with execution fencing."""
from collections import deque
from threading import Condition, Event, Thread
from time import monotonic
import logging

from task_recovery import RecoveryStore
from rag_logging import RagRequestTrace
from rag_permissions import validated_scope
from request_logging import write_event


class RecoveryRunner:
    def __init__(self, tasks, rag_turn, scope_resolver, capacity=4, lease=None, memory_service=None, context_service=None):
        self.memory_service = memory_service
        self.context_service = context_service
        self.tasks, self.rag_turn, self.scope_resolver = tasks, rag_turn, scope_resolver
        self.recovery = RecoveryStore(tasks)
        self.capacity, self.lease = capacity, lease
        self.guard = Condition()
        self.workers = {}
        self.queue = deque()
        self.stopping = False
        self.dispatcher = None
        self.logger = logging.getLogger('legal_rag.requests')
        self.tasks.recovery_cleanup = self.cleanup
        if lease is not None: self.tasks.execution_lease = lease

    def alive(self):
        if self.lease is not None: self.lease.check()
        if self.stopping: raise RuntimeError('Generation service shutting down')

    def cleanup(self):
        for tid in self.recovery.cleanup_ids():
            self.rag_turn.delete_checkpoints(tid)
            self.recovery.cleaned(tid)

    def recover(self):
        with self.guard:
            if self.dispatcher is not None: return
            self.alive()
            self.recovery.interrupt_legacy()
            self.cleanup()
            for task in self.recovery.pending():
                self.tasks.update(task['id'], stage='recovering')
                self.queue.append(task)
            self.dispatcher = Thread(target=self._dispatch, daemon=True)
            self.dispatcher.start()

    def _dispatch(self):
        with self.guard:
            while not self.stopping:
                try: self.alive()
                except Exception:
                    self.stopping = True
                    for _, event in self.workers.values(): event.set()
                    self.guard.notify_all()
                    return
                if self.queue and len(self.workers) < self.capacity:
                    task = self.queue.popleft()
                    self._launch(task, recovering=True)
                else: self.guard.wait(.2)

    def start(self, owner, cid, question, key, request_id, options):
        with self.guard:
            self.alive()
            existing = self.tasks.latest(owner, cid)
            if existing and existing['submission_key'] == key:
                if existing['question'] != question: raise ValueError('提交编号已用于其他问题')
                return existing
            if self.queue or len(self.workers) >= self.capacity: raise RuntimeError('Service busy')
            scope = validated_scope(options.get('allowed_knowledge_bases', self.scope_resolver(owner)))
            memory_input = None
            if self.tasks.memory is not None:
                from memory_service import command_candidate
                memory_input = (self.tasks.memory.snapshot(owner) if getattr(self.tasks.memory,'facts',None) is not None else
                    self.tasks.memory.read(owner) if command_candidate(question) else self.tasks.memory.snapshot(owner))
            task, created = self.tasks.create(owner, cid, question, key, request_id,
                recovery={'version': self.rag_turn.VERSION, 'scope': scope}, memory_input=memory_input)
            if created: self._launch(task, recovering=False, options=options)
            return task

    def _launch(self, task, *, recovering, options=None):
        cancelled = Event()
        thread = Thread(target=self._run, args=(task, recovering, options, cancelled), daemon=True)
        self.workers[task['id']] = (thread, cancelled)
        thread.start()

    def _run(self, task, recovering, options, cancelled):
        tid, epoch, stream = task['id'], None, None
        identity = dict(task_id=tid, request_id=task['request_id'], user_id=task['owner_id'], conversation_id=task['conversation_id'])
        try:
            self.alive()
            claim = self.recovery.claim(tid, recovering=recovering)
            if claim is None: return
            epoch = claim['epoch']
            identity.update(epoch=epoch, recovery_attempt=claim['attempts'])
            scope = validated_scope(self.scope_resolver(task['owner_id']))
            if not scope or scope != frozenset(claim['scope']) or claim['version'] != self.rag_turn.VERSION:
                self.tasks.update(tid, status='interrupted', error='账号权限或流程版本已变化，本次任务无法恢复，请重新发送。', epoch=epoch)
                write_event(self.logger, 'generation_recovery_blocked', **identity)
                return
            def verify_authorization():
                self.alive()
                if validated_scope(self.scope_resolver(task['owner_id'])) != scope:
                    raise ValueError('Current authorization changed')
            if not self.tasks.store.conversation_belongs_to(task['owner_id'], task['conversation_id']):
                raise ValueError('Conversation ownership mismatch')
            from langchain_core.messages import AIMessage, HumanMessage
            history = self.tasks.store.load_conversation_messages(task['owner_id'], task['conversation_id'])
            messages = [(HumanMessage if role == 'user' else AIMessage)(content=content) for role, content in history]
            options = dict(options or {})
            options['allowed_knowledge_bases'] = scope
            if self.tasks.contexts is not None:
                context_input = self.tasks.contexts.task_input(task['owner_id'], tid)
                if context_input is None: raise ValueError('Missing context task input')
                cached = self.tasks.contexts.task_result(task['owner_id'],tid)
                def verify_context_revision():
                    current = self.tasks.contexts.read(task['owner_id'],task['conversation_id'])
                    saved_result = self.tasks.contexts.task_result(task['owner_id'],tid)
                    expected = (saved_result['summary'] if saved_result else context_input['summary'])['revision']
                    if current['revision'] != expected: raise ValueError('Context version changed; resend question')
                def save_context(result):
                    self.alive()
                    return self.tasks.contexts.save_result(task['owner_id'],task['conversation_id'],tid,result,
                        context_input['summary']['revision'],epoch=epoch)
                options.update(context_input=context_input,context_service=self.context_service,
                               context_cached=cached,context_save=save_context,context_revision_guard=verify_context_revision)
            if self.tasks.memory is not None:
                snapshot = self.tasks.memory.task_input(tid, task['owner_id'])
                if snapshot is None: raise ValueError('Missing memory task input')
                def verify_memory_revision():
                    current = self.tasks.memory.read(task['owner_id'], include_content=False)
                    if current['revision'] != snapshot['revision'] or current['enabled'] != snapshot['enabled']:
                        raise ValueError('Memory version changed; resend the question')
                options.update(memory_input=snapshot, memory_service=self.memory_service,
                               memory_revision_guard=verify_memory_revision)
                facts=getattr(self.tasks.memory,'facts',None)
                if facts is not None and getattr(self.rag_turn,'understanding',None) is not None:
                    from memory_tool_contract import ToolContext
                    from dataclasses import replace
                    raw=options.get('context_input',{}).get('messages',[])
                    from memory_service import estimate_tokens
                    selected=[];used=0
                    for message in reversed(raw):
                        if message['role']!='user':continue
                        needed=estimate_tokens(message['content'])
                        if used+needed>1000:break
                        selected.insert(0,dict(kind='message',reference=f"{task['conversation_id']}:{message['ordinal']}",
                            role='user',content=message['content'],new=False));used+=needed
                    selected.append(dict(kind='current_input',reference=tid,role='user',content=task['question'],new=True))
                    tool_context=ToolContext(task['owner_id'],'foreground',tid,snapshot['revision'],snapshot.get('policy_epoch',0),
                        dict(document=snapshot,question=task['question'],conversation_id=task['conversation_id'],sources=selected,memory_request=False))
                    def save_decision(decision):
                        verify_authorization()
                        with self.tasks.cursor() as c:
                            if not self.tasks._current_epoch(c,tid,epoch):raise ValueError('Stale decision worker')
                            facts.save_decision(tool_context,decision,c)
                    options.update(memory_tools=self.tasks.memory.tools,tool_context=tool_context,
                        decision_cached=facts.read_decision(task['owner_id'],tid),decision_save=save_decision)
            options['trace'] = RagRequestTrace(request_id=task['request_id'], user_id=task['owner_id'],
                conversation_id=task['conversation_id'], allowed_knowledge_bases=scope, logger=self.logger)
            write_event(self.logger, 'generation_recovery_started' if recovering else 'generation_started', **identity)
            stream = self.rag_turn.stream(task['question'], messages, **options, task_id=tid,
                resume=recovering, cancelled=cancelled, execution_epoch=epoch, guard=verify_authorization)
            answer, updated = '', 0.0
            for item in stream:
                self.alive()
                data = item['data']
                if item['event'] == 'status':
                    verify_authorization()
                    stage = data['stage']
                    visible = 'answer_restarted' if recovering and stage == 'answer' else stage
                    if not self.tasks.update(tid, stage=visible, answer='' if stage == 'answer' else None, epoch=epoch): return
                elif item['event'] == 'delta':
                    answer += data['text']
                    if monotonic() - updated >= .2:
                        verify_authorization()
                        if not self.tasks.update(tid, answer=answer, epoch=epoch): return
                        updated = monotonic()
                elif item['event'] == 'done':
                    verify_authorization()
                    completion_options = {}
                    if self.tasks.reflection is not None:
                        completion_options['context_status']=data.get('context')
                    if self.tasks.memory is not None:
                        completion_options.update(memory_result=data.get('memory_proposal'), memory_status=data.get('memory'))
                    if data.get('decision') is not None:
                        completion_options.update(decision=data['decision'],pending_memory=data.get('pending_memory'))
                    if not self.tasks.complete(tid, data['answer'], data.get('sources', []), epoch=epoch, **completion_options): return
                    if data.get('decision') is not None:
                        observed=self.tasks.get(task['owner_id'],task['conversation_id'],tid)
                        result=observed.get('memory',{}) if observed else {}
                        write_event(self.logger,'memory_tool_completed',**identity,
                            schema_version='model-memory-tools-v1',status=result.get('status','none'),
                            action_count=len((data['decision'].get('memory_call') or {}).get('operations',[])),
                            revision=result.get('revision'),added=result.get('added',0),merged=result.get('merged',0))
                    if data.get('memory_proposal') is not None:
                        proposal = data['memory_proposal']
                        write_event(self.logger,'memory_saved',revision=proposal['expected_revision']+1,
                            source='conversation',entry_count=len(proposal['prepared']['entries']),**identity)
                    options['trace'].saved()
                    write_event(self.logger, 'generation_recovery_completed' if recovering else 'generation_completed', **identity)
                    return
                elif item['event'] == 'error': raise RuntimeError('Generation failed')
            raise RuntimeError('Graph ended without done')
        except BaseException as error:
            if not self.stopping:
                try:
                    self.alive()
                    self.tasks.update(tid, status='interrupted' if isinstance(error, ValueError) else 'failed',
                        error=('记忆已更新，请重新发送本次请求。' if 'Memory version changed' in str(error) else
                               '记忆保存失败或版本已变化，请查看设置后重新发送。' if error.__class__.__name__ == 'MemoryConflict' else
                               str(error) if error.__class__.__name__ in ('ContextTooLong','ContextValidation','ContextConflict') else
                               '对话摘要已变化，请重新发送本次请求。' if 'Context version changed' in str(error) else
                               '暂时无法完成本次回答，请重新发送。'), answer='' if isinstance(error, ValueError) else None, epoch=epoch)
                except Exception: pass
                write_event(self.logger, 'generation_recovery_blocked' if recovering else 'generation_failed',
                            error_type=type(error).__name__, **identity)
        finally:
            try:
                if hasattr(stream, 'close'): stream.close()
            finally:
                with self.guard:
                    self.workers.pop(tid, None)
                    self.guard.notify_all()

    def shutdown(self):
        with self.guard:
            self.stopping = True
            workers = list(self.workers.values())
            for _, event in workers: event.set()
            self.guard.notify_all()
        if self.dispatcher is not None: self.dispatcher.join()
        for thread, _ in workers: thread.join()
