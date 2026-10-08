"""Durable observations of bounded, single-process background generations."""
import json
import logging
import re
import uuid
from time import monotonic
from time import time
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock, Thread

from source_snapshots import snapshot_sources, history_message
from web_storage import _utc_now, _title_from_question
from request_logging import write_event

FIELDS = 'id,owner_id,conversation_id,submission_key,question,status,stage,answer,sources,error,request_id,created_at,updated_at'
MIGRATION = 'v12_generation_tasks_v1'


def schema_sql(table, conversations, index):
    return [f'''CREATE TABLE {table} (
        id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
        conversation_id TEXT NOT NULL REFERENCES {conversations}(id) ON DELETE CASCADE,
        submission_key TEXT NOT NULL, question TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('running','completed','failed','interrupted')),
        stage TEXT NOT NULL, answer TEXT NOT NULL, sources TEXT NOT NULL, error TEXT NOT NULL,
        request_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        UNIQUE(owner_id,submission_key))''',
        f"CREATE UNIQUE INDEX {index} ON {table}(owner_id) WHERE status='running'"]


class TaskStore:
    def __init__(self, store, memory=None, contexts=None, reflection=None, clock=time):
        self.store = store
        self.memory = memory
        self.contexts = contexts
        self.reflection,self.clock = reflection,clock
        # SQLite has a concrete path; PostgreSQL exposes a schema and cursor.
        self.pg = not hasattr(store, 'path')
        self.schema = getattr(store, 'schema', 'public') if self.pg else None
        if self.schema and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', self.schema):
            raise ValueError('Invalid task schema')
        self.owner_column = getattr(store, 'owner_column', 'session_id')
        if self.owner_column not in ('session_id', 'user_id'):
            raise ValueError('Invalid owner column')
        if not self.pg:
            with self.cursor() as c:
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='generation_tasks'")
                if c.fetchone() is None:
                    for statement in schema_sql(self.table('generation_tasks'), self.table('conversations'), 'generation_one_active'):
                        c.execute(statement)
        else:
            with self.cursor() as c:
                self.execute(c, f'SELECT 1 FROM {self.table("schema_migrations")} WHERE name=?', (MIGRATION,))
                if c.fetchone() is None:
                    raise RuntimeError('Run prepare_generation_tasks.py --apply first')

    def table(self, name):
        return f'"{self.schema}"."{name}"' if self.pg else f'"{name}"'

    @contextmanager
    def cursor(self):
        if self.pg:
            with self.store._cursor() as c: yield c
        else:
            with self.store._connection() as connection:
                connection.execute('BEGIN IMMEDIATE')
                yield connection.cursor()

    def execute(self, c, query, params=()):
        c.execute(query.replace('?', '%s') if self.pg else query, params)

    def decode(self, row):
        if row is None: return None
        task = dict(zip(FIELDS.split(','), row))
        task['sources'] = json.loads(task['sources'])
        return task

    def create(self, owner, cid, question, key, request_id, recovery=None, memory_input=None):
        with self.cursor() as c:
            self.execute(c, f'SELECT id FROM {self.table("conversations")} WHERE id=? AND "{self.owner_column}"=?' + (' FOR UPDATE' if self.pg else ''), (cid, owner))
            if c.fetchone() is None: raise LookupError('对话不存在')
            self.execute(c, f'SELECT {FIELDS} FROM {self.table("generation_tasks")} WHERE owner_id=? AND submission_key=?', (owner, key))
            existing = self.decode(c.fetchone())
            if existing:
                if existing['conversation_id'] != cid or existing['question'] != question:
                    raise ValueError('提交编号已用于其他问题')
                return existing, False
            self.execute(c, f'SELECT id FROM {self.table("generation_tasks")} WHERE owner_id=? AND status=?', (owner, 'running'))
            if c.fetchone(): raise ValueError('当前账号正在生成回答，请等待完成')
            now = _utc_now()
            values = (uuid.uuid4().hex, owner, cid, key, question, 'running', 'rewrite', '', '[]', '', request_id, now, now)
            self.execute(c, f'INSERT INTO {self.table("generation_tasks")} ({FIELDS}) VALUES ({",".join("?" for _ in values)})', values)
            if self.memory is not None and memory_input is not None:
                self.memory.capture(c, self.decode(values), memory_input)
            if self.contexts is not None:
                self.contexts.capture(owner, cid, values[0], c)
            if self.reflection is not None:
                self.reflection.defer(c,owner,cid,self.clock())
            if recovery is not None:
                self.execute(c, f'INSERT INTO {self.table("generation_recovery")} (task_id,version,scope,attempts,epoch) VALUES (?,?,?,0,0)',
                             (values[0], recovery['version'], json.dumps(sorted(recovery['scope']))))
            self.execute(c, f'UPDATE {self.table("conversations")} SET updated_at=? WHERE id=?', (now, cid))
            return self.decode(values), True

    def get(self, owner, cid, tid):
        with self.cursor() as c:
            self.execute(c, f'SELECT {FIELDS} FROM {self.table("generation_tasks")} WHERE owner_id=? AND conversation_id=? AND id=?', (owner, cid, tid))
            return self.observed(self.decode(c.fetchone()), c)

    def latest(self, owner, cid):
        with self.cursor() as c:
            self.execute(c, f'SELECT {FIELDS} FROM {self.table("generation_tasks")} WHERE owner_id=? AND conversation_id=? ORDER BY created_at DESC LIMIT 1', (owner, cid))
            return self.observed(self.decode(c.fetchone()), c)

    def active(self, owner):
        with self.cursor() as c:
            self.execute(c, f'SELECT {FIELDS} FROM {self.table("generation_tasks")} WHERE owner_id=? AND status=?', (owner, 'running'))
            return self.observed(self.decode(c.fetchone()), c)

    def observed(self, task, cursor):
        if task is not None and self.memory is not None:
            task['memory'] = self.memory.task_observation(cursor, task['id'])
        return task

    def history(self, owner, cid):
        with self.cursor() as c:
            self.execute(c, f'SELECT id FROM {self.table("conversations")} WHERE id=? AND "{self.owner_column}"=?' + (' FOR SHARE' if self.pg else ''), (cid, owner))
            if c.fetchone() is None: return None
            self.execute(c, f'SELECT m.role,m.content,s.sources FROM {self.table("conversation_messages")} m LEFT JOIN {self.table("conversation_sources")} s ON s.conversation_id=m.conversation_id AND s.ordinal=m.ordinal WHERE m.conversation_id=? ORDER BY m.ordinal', (cid,))
            messages = [history_message(role, content, json.loads(sources) if isinstance(sources, str) else sources) for role, content, sources in c.fetchall()]
            self.execute(c, f'SELECT {FIELDS} FROM {self.table("generation_tasks")} WHERE owner_id=? AND conversation_id=? ORDER BY created_at DESC LIMIT 1', (owner, cid))
            return {'messages': messages, 'task': self.observed(self.decode(c.fetchone()), c)}

    def delete_conversation(self, owner, cid):
        with self.cursor() as c:
            self.execute(c, f'SELECT id FROM {self.table("conversations")} WHERE id=? AND "{self.owner_column}"=?' + (' FOR UPDATE' if self.pg else ''), (cid, owner))
            if c.fetchone() is None: return None
            self.execute(c, f'SELECT id FROM {self.table("generation_tasks")} WHERE conversation_id=? AND status=?', (cid, 'running'))
            if c.fetchone(): return False
            if getattr(self, 'has_recovery', False):
                self.execute(c, f'INSERT INTO {self.table("checkpoint_cleanup")} (task_id) SELECT task_id FROM {self.table("generation_recovery")} WHERE task_id IN (SELECT id FROM {self.table("generation_tasks")} WHERE conversation_id=?) ON CONFLICT(task_id) DO NOTHING', (cid,))
            if self.memory is not None and getattr(self.memory,'facts',None) is not None:
                self.memory.facts.detach_conversation(owner,cid,c)
            self.execute(c, f'DELETE FROM {self.table("conversations")} WHERE id=?', (cid,))
            return True

    def _current_epoch(self, c, tid, epoch):
        if epoch is None: return True
        if getattr(self, 'execution_lease', None) is not None: self.execution_lease.check()
        self.execute(c, f'SELECT epoch FROM {self.table("generation_recovery")} WHERE task_id=?', (tid,))
        row = c.fetchone()
        return row is not None and row[0] == epoch

    def update(self, tid, *, stage=None, answer=None, status='running', error='', epoch=None):
        with self.cursor() as c:
            self.execute(c, f'SELECT id FROM {self.table("generation_tasks")} WHERE id=?' + (' FOR UPDATE' if self.pg else ''), (tid,))
            if not self._current_epoch(c, tid, epoch): return False
            self.execute(c, f'UPDATE {self.table("generation_tasks")} SET stage=COALESCE(?,stage),answer=COALESCE(?,answer),status=?,error=?,updated_at=? WHERE id=? AND status=?', (stage, answer, status, error, _utc_now(), tid, 'running'))
            return c.rowcount == 1

    def interrupt_running(self):
        with self.cursor() as c:
            self.execute(c, f'UPDATE {self.table("generation_tasks")} SET status=?,error=?,updated_at=? WHERE status=?', ('interrupted', '服务已重启，本次回答中断，请重新发送。', _utc_now(), 'running'))

    def complete(self, tid, answer, sources, epoch=None, memory_result=None, memory_status=None, context_status=None,
                 pending_memory=None,decision=None):
        if not answer.strip(): raise ValueError('Empty answer')
        sources = snapshot_sources(sources)
        with self.cursor() as c:
            self.execute(c, f'SELECT {FIELDS} FROM {self.table("generation_tasks")} WHERE id=?' + (' FOR UPDATE' if self.pg else ''), (tid,))
            task = self.decode(c.fetchone())
            if not task or task['status'] != 'running': return False
            if not self._current_epoch(c, tid, epoch): return False
            cid = task['conversation_id']
            self.execute(c, f'SELECT title,title_is_custom FROM {self.table("conversations")} WHERE id=? AND "{self.owner_column}"=?' + (' FOR UPDATE' if self.pg else ''), (cid, task['owner_id']))
            conversation = c.fetchone()
            if conversation is None: return False
            if decision is not None:
                from dataclasses import asdict
                from memory_tool_contract import prepared_from_dict,ToolResult,decision_from_dict
                facts=getattr(self.memory,'facts',None)
                frozen=facts.read_decision(task['owner_id'],tid,c) if facts is not None else None
                from memory_fact_storage import digest
                decision=asdict(decision_from_dict(decision))
                if frozen is None or digest(asdict(frozen))!=digest(decision):raise ValueError('Missing or inconsistent frozen decision')
                if pending_memory is not None:
                    prepared_update=prepared_from_dict(pending_memory)
                    if prepared_update.context.owner!=task['owner_id'] or prepared_update.context.execution_id!=tid:
                        raise ValueError('Memory completion identity mismatch')
                    self.execute(c,'SAVEPOINT memory_tool')
                    try:
                        result=self.memory.tools.commit(prepared_update,c)
                        self.execute(c,'RELEASE SAVEPOINT memory_tool')
                    except Exception:
                        self.execute(c,'ROLLBACK TO SAVEPOINT memory_tool')
                        self.execute(c,'RELEASE SAVEPOINT memory_tool')
                        result=ToolResult('blocked',self.memory.read(task['owner_id'],c)['revision'],0,0,(),
                                          '本次记忆未保存，请在设置查看后重试。')
                        context=prepared_from_dict(pending_memory).context
                        facts.save_receipt(context,pending_memory['argument_digest'],result,c)
                    memory_status=asdict(result)
                    if decision['answer']['route']!='rag':answer=result.message
            if memory_result is not None:
                if self.memory is None: raise RuntimeError('Memory completion not configured')
                from memory_service import authorized_change, MemoryValidation, split_entries, vector_valid
                frozen = self.memory.task_input(tid,task['owner_id'],cursor=c)
                if frozen is None or memory_result.get('question') != task['question']:
                    raise MemoryValidation('Invalid memory proposal identity')
                core, extended = authorized_change(task['question'],frozen,memory_result.get('operation'))
                proposed = memory_result['prepared']
                if (proposed.get('core_text') != core or proposed.get('extended_text') != extended or
                    proposed.get('enabled') != frozen['enabled'] or memory_result['expected_revision'] != frozen['revision']):
                    raise MemoryValidation('Invalid memory proposal content')
                entries = proposed.get('entries',[])
                if ([e.get('text') for e in entries] != split_entries(extended) or
                    [e.get('ordinal') for e in entries] != list(range(len(entries))) or
                    len({e.get('id') for e in entries}) != len(entries) or
                    any(not vector_valid(e.get('embedding')) for e in entries)):
                    raise MemoryValidation('Invalid memory proposal entries')
                self.memory.save(task['owner_id'], memory_result['prepared'], memory_result['expected_revision'], cursor=c)
                answer = memory_result['feedback']
                memory_status = {'status': 'saved', 'warning': '', 'revision': memory_result['expected_revision'] + 1}
            if self.memory is not None and memory_status is not None:
                self.memory.observation(c, tid, memory_status)
            if decision is not None:
                from web_understanding import with_scope_notice
                answer=with_scope_notice(answer,decision['answer'])
            self.execute(c, f'SELECT COALESCE(MAX(ordinal),0)+1 FROM {self.table("conversation_messages")} WHERE conversation_id=?', (cid,))
            ordinal = c.fetchone()[0]
            if decision is not None:
                self.memory.facts.bind_ordinal(task['owner_id'],tid,cid,ordinal,c)
            now = _utc_now()
            for offset, role, content in [(0, 'user', task['question']), (1, 'assistant', answer)]:
                self.execute(c, f'INSERT INTO {self.table("conversation_messages")} VALUES (?,?,?,?,?)', (cid, ordinal+offset, role, content, now))
            if sources:
                self.execute(c, f'INSERT INTO {self.table("conversation_sources")} VALUES (?,?,?)', (cid, ordinal+1, json.dumps(sources, ensure_ascii=False)))
            title = _title_from_question(task['question']) if conversation[0] == '新对话' and not conversation[1] else conversation[0]
            self.execute(c, f'UPDATE {self.table("conversations")} SET title=?,updated_at=? WHERE id=?', (title, now, cid))
            self.execute(c, f'UPDATE {self.table("generation_tasks")} SET status=?,answer=?,sources=?,updated_at=? WHERE id=?', ('completed', answer, json.dumps(sources, ensure_ascii=False), now, tid))
            if self.reflection is not None:
                from memory_service import command_candidate
                if decision is not None or not command_candidate(task['question']):
                    reason='compression' if context_status and context_status.get('compressed') else 'idle'
                    self.reflection.enqueue(c,task['owner_id'],cid,ordinal+1,reason,self.clock())
            return True


class GenerationRunner:
    def __init__(self, tasks, rag_turn, capacity=4):
        self.tasks, self.rag_turn = tasks, rag_turn
        self.slots = BoundedSemaphore(capacity)
        self.guard = Lock()
        self.logger = logging.getLogger('legal_rag.requests')

    def start(self, owner, cid, question, key, request_id, options):
        with self.guard:
            # Repeated submissions must still resolve while all slots are occupied.
            existing = self.tasks.latest(owner, cid)
            if existing and existing['submission_key'] == key:
                if existing['question'] != question: raise ValueError('提交编号已用于其他问题')
                return existing
            if not self.slots.acquire(blocking=False): raise RuntimeError('服务繁忙，请稍后提交')
            try:
                task, created = self.tasks.create(owner, cid, question, key, request_id)
                if not created:
                    self.slots.release()
                    return task
                thread = Thread(target=self.run, args=(task, options), daemon=True)
                thread.start()
                return task
            except BaseException:
                self.slots.release()
                raise

    def run(self, task, options):
        tid = task['id']
        identity = dict(task_id=tid, request_id=task['request_id'], user_id=task['owner_id'], conversation_id=task['conversation_id'])
        write_event(self.logger, 'generation_started', **identity)
        stream = None
        try:
            from langchain_core.messages import AIMessage, HumanMessage
            history = self.tasks.store.load_conversation_messages(task['owner_id'], task['conversation_id'])
            if history is None: raise RuntimeError('Conversation deleted')
            messages = [(HumanMessage if role == 'user' else AIMessage)(content=content) for role, content in history]
            stream = self.rag_turn.stream(task['question'], messages, **options)
            answer = ''
            updated = 0.0
            for item in stream:
                data = item['data']
                if item['event'] == 'status': self.tasks.update(tid, stage=data['stage'])
                elif item['event'] == 'delta':
                    answer += data['text']
                    if monotonic() - updated >= .2:
                        self.tasks.update(tid, answer=answer)
                        updated = monotonic()
                elif item['event'] == 'done':
                    if not self.tasks.complete(tid, answer, data.get('sources', [])):
                        raise RuntimeError('Task no longer running')
                    if options.get('trace'): options['trace'].saved()
                    write_event(self.logger, 'generation_completed', **identity)
                    return
                elif item['event'] == 'error': raise RuntimeError('Model stream failed')
            raise RuntimeError('Stream ended without done')
        except Exception as error:
            try:
                self.tasks.update(tid, status='failed', error='暂时无法完成回答，请重新发送。')
            except Exception as storage_error:
                write_event(self.logger, 'generation_persistence_failed', error_type=type(storage_error).__name__, **identity)
            write_event(self.logger, 'generation_failed', error_type=type(error).__name__, **identity)
        finally:
            try:
                if hasattr(stream, 'close'): stream.close()
            finally:
                self.slots.release()
