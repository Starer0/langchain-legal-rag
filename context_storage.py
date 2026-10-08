"""Owner-scoped PostgreSQL summaries and frozen per-generation context inputs."""
from contextlib import contextmanager
from dataclasses import asdict
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
from conversation_context import Summary, ContextConflict, ContextValidation, KINDS, ContextService
from prepare_conversation_context import MIGRATION
from memory_service import command_candidate
from web_storage import _utc_now


class ContextStore:
    def __init__(self,*,schema='public',**settings):
        self.schema,self.settings = schema, {'connect_timeout':5,**settings}
        with self.cursor() as c:
            c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(self.table('schema_migrations')),(MIGRATION,))
            if c.fetchone() is None: raise RuntimeError('Run prepare_conversation_context.py --apply first')

    def table(self,name): return sql.Identifier(self.schema,name)

    @contextmanager
    def cursor(self):
        conn = psycopg2.connect(**self.settings)
        try:
            with conn,conn.cursor() as c: yield c
        finally: conn.close()

    def owned(self,c,owner,cid,*,write=False):
        c.execute(sql.SQL('SELECT c.id FROM {} c JOIN {} u ON u.id=c.user_id WHERE c.id=%s AND c.user_id=%s AND u.is_active ')
            .format(self.table('conversations'),self.table('users')) + sql.SQL('FOR UPDATE OF c' if write else 'FOR SHARE OF c'),(cid,owner))
        if c.fetchone() is None: raise PermissionError('无权读取该对话或账号不可用。')

    def read(self,owner,cid,cursor=None):
        if cursor is None:
            with self.cursor() as c: return self.read(owner,cid,c)
        self.owned(cursor,owner,cid)
        cursor.execute(sql.SQL('SELECT revision,through_ordinal,items FROM {} WHERE conversation_id=%s AND user_id=%s')
            .format(self.table('conversation_contexts')),(cid,owner))
        row = cursor.fetchone()
        return dict(zip(('revision','through_ordinal','items'),row)) if row else asdict(Summary())

    def messages(self,c,cid,*,legal=False):
        c.execute(sql.SQL('SELECT ordinal,role,content FROM {} WHERE conversation_id=%s ORDER BY ordinal')
            .format(self.table('conversation_messages')),(cid,))
        messages=[dict(zip(('ordinal','role','content'),row)) for row in c.fetchall()]
        if legal and getattr(self,'semantic',False):
            c.execute(sql.SQL('SELECT user_ordinal,decision FROM {} d JOIN {} g ON g.id=d.task_id WHERE g.conversation_id=%s').format(self.table('generation_turn_decisions'),self.table('generation_tasks')),(cid,))
            decisions={ordinal:decision['answer'] for ordinal,decision in c.fetchall()}
            for message in messages:
                answer=decisions.get(message['ordinal'])
                if message['role']=='user' and answer is not None:
                    message['content']=answer['legal_question'] if answer['route']=='rag' else ''
        return [m for m in messages if m['content']]

    def capture(self,owner,cid,tid,cursor):
        self.owned(cursor,owner,cid)
        frozen = {'summary':self.read(owner,cid,cursor),'messages':self.messages(cursor,cid)}
        if getattr(self,'semantic',False):frozen['legal_messages']=self.messages(cursor,cid,legal=True)
        cursor.execute(sql.SQL('INSERT INTO {} (task_id,user_id,conversation_id,input) VALUES (%s,%s,%s,%s)')
            .format(self.table('generation_context_inputs')),(tid,owner,cid,Json(frozen)))
        return frozen

    def task_input(self,owner,tid,cursor=None):
        if cursor is None:
            with self.cursor() as c: return self.task_input(owner,tid,c)
        cursor.execute(sql.SQL('SELECT user_id,conversation_id,input FROM {} WHERE task_id=%s')
            .format(self.table('generation_context_inputs')),(tid,))
        row = cursor.fetchone()
        if row is None: return None
        if row[0] != owner: raise PermissionError('无权读取此上下文。')
        self.owned(cursor,owner,row[1]); return row[2]

    def save(self,owner,cid,summary,expected_revision,cursor=None):
        if cursor is None:
            with self.cursor() as c: return self.save(owner,cid,summary,expected_revision,c)
        c = cursor; self.owned(c,owner,cid,write=True)
        old = self.read(owner,cid,c)
        if old['revision'] != expected_revision: raise ContextConflict('对话摘要已变化，请重新发送问题。')
        self.validate(summary,self.messages(c,cid,legal=True),old)
        c.execute(sql.SQL('INSERT INTO {} (conversation_id,user_id,revision,through_ordinal,items,updated_at) VALUES (%s,%s,%s,%s,%s,%s) '
            'ON CONFLICT(conversation_id) DO UPDATE SET revision=EXCLUDED.revision,through_ordinal=EXCLUDED.through_ordinal,items=EXCLUDED.items,updated_at=EXCLUDED.updated_at')
            .format(self.table('conversation_contexts')),(cid,owner,expected_revision+1,summary['through_ordinal'],Json(summary['items']),_utc_now()))
        return {**summary,'revision':expected_revision+1}

    @staticmethod
    def validate(summary,messages,old):
        try:
            source = {m['ordinal']:m for m in messages}
            cutoff = summary['through_ordinal']
            if type(cutoff) is not int or cutoff < old['through_ordinal'] or cutoff > max(source,default=0): raise ValueError()
            if not isinstance(summary['items'],list) or not summary['items']: raise ValueError()
            for i in summary['items']:
                if set(i) != {'kind','text','source_ordinals','source_quotes','supersedes'} or i['kind'] not in KINDS or i['supersedes'] != []: raise ValueError()
                if not i['source_ordinals'] or len(i['source_ordinals']) != len(i['source_quotes']) or i['text'] != '\n'.join(i['source_quotes']): raise ValueError()
                for ordinal,quote in zip(i['source_ordinals'],i['source_quotes']):
                    m=source[ordinal]
                    if type(ordinal) is not int or ordinal > cutoff or m['role'] != 'user' or command_candidate(m['content']) or not quote.strip() or quote not in m['content']: raise ValueError()
            if any(i not in summary['items'] for i in old['items']): raise ValueError()
            svc=ContextService(None)
            if svc.count(svc.summary_text(Summary(**summary))) > svc.limits.summary_tokens: raise ValueError()
        except (ValueError,TypeError,KeyError,AttributeError) as error:
            raise ContextValidation('摘要来源或保存预算无法核对。') from error

    def task_result(self,owner,tid,cursor=None):
        if cursor is None:
            with self.cursor() as c: return self.task_result(owner,tid,c)
        frozen=self.task_input(owner,tid,cursor)
        if frozen is None: return None
        cursor.execute(sql.SQL('SELECT result FROM {} WHERE task_id=%s AND user_id=%s').format(self.table('generation_context_inputs')),(tid,owner))
        return cursor.fetchone()[0]

    def save_result(self,owner,cid,tid,result,expected_revision,*,epoch=None):
        with self.cursor() as c:
            # Match the existing completion order: task then conversation.
            c.execute(sql.SQL('SELECT owner_id,conversation_id,status FROM {} WHERE id=%s FOR UPDATE').format(self.table('generation_tasks')),(tid,))
            row=c.fetchone()
            if row is None or row[:2] != (owner,cid) or row[2] != 'running': raise PermissionError('任务归属或状态已变化。')
            if epoch is not None:
                c.execute(sql.SQL('SELECT epoch FROM {} WHERE task_id=%s').format(self.table('generation_recovery')),(tid,))
                if c.fetchone() != (epoch,): raise ContextConflict('任务执行代次已变化。')
            frozen=self.task_input(owner,tid,c)
            if frozen is None: raise ContextValidation('任务缺少上下文快照。')
            existing=self.task_result(owner,tid,c)
            if existing is not None: return existing
            if result['compressed']:
                self.validate(result['summary'],frozen.get('legal_messages',frozen['messages']),frozen['summary'])
                saved=self.save(owner,cid,result['summary'],expected_revision,c)
                result={**result,'summary':saved}
            c.execute(sql.SQL('UPDATE {} SET result=%s WHERE task_id=%s AND user_id=%s').format(self.table('generation_context_inputs')),(Json(result),tid,owner))
            return result
