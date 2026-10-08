"""Account-owned, revisioned memory and private task inputs in PostgreSQL."""
import json
from contextlib import contextmanager
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
from web_storage import _utc_now
from memory_service import MemoryConflict, validate_texts

MIGRATION = 'v12_account_memory_v1'


def empty_document():
    return dict(revision=0, enabled=True, core_text='', extended_text='', entries=[], updated_at=None)


def public_document(doc):
    return {key: doc[key] for key in ('revision', 'enabled', 'core_text', 'extended_text', 'updated_at')}


class MemoryStore:
    def __init__(self, *, schema='public', **settings):
        self.schema, self.settings = schema, {'connect_timeout': 5, **settings}
        with self.cursor() as c:
            c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(self.table('schema_migrations')), (MIGRATION,))
            if c.fetchone() is None: raise RuntimeError('Run prepare_account_memory.py --apply first')

    def table(self, name): return sql.Identifier(self.schema, name)

    @contextmanager
    def cursor(self):
        connection = psycopg2.connect(**self.settings)
        try:
            with connection, connection.cursor() as c: yield c
        finally:
            connection.close()

    def read(self, owner, cursor=None, *, include_content=True):
        if cursor is None:
            with self.cursor() as c: return self.read(owner, c, include_content=include_content)
        c = cursor
        columns = 'revision,enabled,core_text,extended_text,updated_at' if include_content else "revision,enabled,'' AS core_text,'' AS extended_text,updated_at"
        c.execute(sql.SQL('SELECT {} FROM {} WHERE user_id=%s').format(sql.SQL(columns), self.table('account_memories')), (owner,))
        row = c.fetchone()
        if row is None: return empty_document()
        doc = dict(zip(('revision', 'enabled', 'core_text', 'extended_text', 'updated_at'), row))
        doc['entries'] = []
        if include_content:
            c.execute(sql.SQL('SELECT id,ordinal,text,embedding,fingerprint FROM {} WHERE user_id=%s AND revision=%s ORDER BY ordinal').format(self.table('account_memory_entries')), (owner, doc['revision']))
            doc['entries'] = [dict(zip(('id', 'ordinal', 'text', 'embedding', 'fingerprint'), item)) for item in c.fetchall()]
        else:
            doc['core_text'], doc['extended_text'] = '', ''
        return doc

    def snapshot(self, owner):
        with self.cursor() as c:
            # One transaction/row lock so raw text and entries cannot straddle revisions.
            c.execute(sql.SQL('SELECT id FROM {} WHERE id=%s AND is_active FOR SHARE').format(self.table('users')), (owner,))
            if c.fetchone() is None: raise PermissionError('账号不可用')
            c.execute(sql.SQL('SELECT revision,enabled FROM {} WHERE user_id=%s FOR SHARE').format(self.table('account_memories')), (owner,))
            header = c.fetchone()
            document=self.read(owner, c, include_content=header is None or header[1] or getattr(self,'facts',None) is not None)
            # Freeze the account policy with the submitted task. The graph can
            # explain accumulation without treating acknowledgment as a write.
            reflection=getattr(self,'reflection',None)
            if reflection is not None:
                policy=reflection.policy(owner,c)
                document.update(auto_accumulate=policy['auto_accumulate'],policy_epoch=policy['epoch'])
            if getattr(self,'facts',None) is not None:
                document.update(self.facts.snapshot(owner,c))
            return document

    def save(self, owner, prepared, expected_revision, cursor=None, *, manual=True):
        if cursor is None:
            with self.cursor() as c: return self.save(owner, prepared, expected_revision, c, manual=manual)
        c = cursor
        validate_texts(prepared['core_text'], prepared['extended_text'])
        c.execute(sql.SQL('SELECT id FROM {} WHERE id=%s AND is_active FOR UPDATE').format(self.table('users')), (owner,))
        if c.fetchone() is None: raise PermissionError('账号不可用')
        c.execute(sql.SQL('SELECT revision FROM {} WHERE user_id=%s FOR UPDATE').format(self.table('account_memories')), (owner,))
        row = c.fetchone()
        revision = row[0] if row else 0
        if revision != expected_revision: raise MemoryConflict('记忆已在其他页面更新，请重新加载后编辑')
        revision += 1
        updated = _utc_now()
        c.execute(sql.SQL('INSERT INTO {} (user_id,revision,enabled,core_text,extended_text,updated_at) VALUES (%s,%s,%s,%s,%s,%s) '
            'ON CONFLICT(user_id) DO UPDATE SET revision=EXCLUDED.revision,enabled=EXCLUDED.enabled,core_text=EXCLUDED.core_text,'
            'extended_text=EXCLUDED.extended_text,updated_at=EXCLUDED.updated_at').format(self.table('account_memories')),
            (owner, revision, prepared['enabled'], prepared['core_text'], prepared['extended_text'], updated))
        c.execute(sql.SQL('DELETE FROM {} WHERE user_id=%s').format(self.table('account_memory_entries')), (owner,))
        for item in prepared['entries']:
            c.execute(sql.SQL('INSERT INTO {} (user_id,id,revision,ordinal,text,embedding,fingerprint) VALUES (%s,%s,%s,%s,%s,%s,%s)').format(self.table('account_memory_entries')),
                (owner, item['id'], revision, item['ordinal'], item['text'], Json(item['embedding']), item['fingerprint']))
        if manual and getattr(self,'facts',None) is not None:
            self.facts.sync_manual(owner,prepared['core_text'],prepared['extended_text'],revision,c)
        if manual and getattr(self,'reflection',None) is not None:
            self.reflection.invalidate(owner,c)
        return {**prepared, 'revision': revision, 'updated_at': updated}

    def capture(self, cursor, task, document):
        cursor.execute(sql.SQL('INSERT INTO {} (task_id,user_id,input,observation) VALUES (%s,%s,%s,%s)').format(self.table('generation_memory_inputs')),
                       (task['id'], task['owner_id'], Json(document), Json({})))

    def task_input(self, tid, owner, cursor=None):
        if cursor is None:
            with self.cursor() as c: return self.task_input(tid,owner,c)
        cursor.execute(sql.SQL('SELECT input FROM {} WHERE task_id=%s AND user_id=%s').format(self.table('generation_memory_inputs')), (tid, owner))
        row = cursor.fetchone()
        return row[0] if row else None

    def observation(self, cursor, tid, status):
        safe = {key: status[key] for key in ('status', 'warning', 'revision','added','merged','suggestion_ids','message') if key in status}
        cursor.execute(sql.SQL('UPDATE {} SET observation=%s WHERE task_id=%s').format(self.table('generation_memory_inputs')), (Json(safe), tid))

    def task_observation(self, cursor, tid):
        cursor.execute(sql.SQL('SELECT observation FROM {} WHERE task_id=%s').format(self.table('generation_memory_inputs')), (tid,))
        row = cursor.fetchone()
        return row[0] if row else {}
