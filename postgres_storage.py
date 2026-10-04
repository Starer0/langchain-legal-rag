"""PostgreSQL implementation of the existing anonymous conversation store.

Schema creation/import belongs to the explicit migration command. Each store
operation owns a short transaction and closes its connection on every exit.
"""

import secrets
from contextlib import contextmanager

import psycopg2
from psycopg2 import sql

from migrate_sqlite_to_postgres import COLUMNS
from web_storage import Conversation, _title_from_question, _utc_now


class PostgresConversationStore:
    owner_column = 'session_id'

    def __init__(self, *, schema='public', **connection_settings):
        self.schema = schema
        self._settings = {'connect_timeout': 5, **connection_settings}
        try:
            with self._cursor() as cursor:
                for table, columns in COLUMNS.items():
                    cursor.execute(sql.SQL('SELECT {} FROM {} LIMIT 0').format(
                        sql.SQL(',').join(map(sql.Identifier, columns)), self._table(table)))
                cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(self._table('schema_migrations')),
                               ('v8_to_v82_conversations',))
                if cursor.fetchone() is None:
                    raise RuntimeError('PostgreSQL migration is not ready; run the explicit migration command first')
        except (psycopg2.errors.UndefinedTable, psycopg2.errors.UndefinedColumn, psycopg2.errors.InvalidSchemaName) as error:
            raise RuntimeError('PostgreSQL migration is not ready; run the explicit migration command first') from error

    def _table(self, name):
        return sql.Identifier(self.schema, name)

    @contextmanager
    def _cursor(self):
        connection = psycopg2.connect(**self._settings)
        try:
            with connection, connection.cursor() as cursor:
                yield cursor
        finally:
            connection.close()

    def ensure_session(self, session_id):
        session_id = session_id or secrets.token_urlsafe(32)
        now = _utc_now()
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('INSERT INTO {} (id,created_at,last_seen_at) VALUES (%s,%s,%s) '
                                   'ON CONFLICT (id) DO UPDATE SET last_seen_at=EXCLUDED.last_seen_at').format(self._table('sessions')),
                           (session_id, now, now))
        return session_id

    def list_conversations(self, session_id):
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('SELECT id,title,updated_at FROM {} WHERE {}=%s ORDER BY updated_at DESC').format(self._table('conversations'), sql.Identifier(self.owner_column)), (session_id,))
            return [Conversation(*row) for row in cursor.fetchall()]

    def create_conversation(self, session_id):
        now = _utc_now()
        item = Conversation(secrets.token_urlsafe(24), '新对话', now)
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('INSERT INTO {} (id,{},title,title_is_custom,created_at,updated_at) VALUES (%s,%s,%s,0,%s,%s)').format(self._table('conversations'), sql.Identifier(self.owner_column)),
                           (item.id, session_id, item.title, now, now))
        return item

    def conversation_belongs_to(self, session_id, conversation_id):
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE id=%s AND {}=%s').format(self._table('conversations'), sql.Identifier(self.owner_column)), (conversation_id, session_id))
            return cursor.fetchone() is not None

    def rename_conversation(self, session_id, conversation_id, title):
        title = title.strip()
        if not title or len(title) > 80:
            raise ValueError('标题长度必须为 1 到 80 个字符')
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('UPDATE {} SET title=%s,title_is_custom=1,updated_at=%s WHERE id=%s AND {}=%s RETURNING id,title,updated_at').format(self._table('conversations'), sql.Identifier(self.owner_column)),
                           (title, _utc_now(), conversation_id, session_id))
            row = cursor.fetchone()
            return Conversation(*row) if row else None

    def delete_conversation(self, session_id, conversation_id):
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('DELETE FROM {} WHERE id=%s AND {}=%s').format(self._table('conversations'), sql.Identifier(self.owner_column)), (conversation_id, session_id))
            return cursor.rowcount == 1

    def load_conversation_messages(self, session_id, conversation_id):
        with self._cursor() as cursor:
            # A shared row lock keeps ownership and messages consistent while read.
            cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE id=%s AND {}=%s FOR SHARE').format(self._table('conversations'), sql.Identifier(self.owner_column)), (conversation_id, session_id))
            if cursor.fetchone() is None:
                return None
            cursor.execute(sql.SQL('SELECT role,content FROM {} WHERE conversation_id=%s ORDER BY ordinal').format(self._table('conversation_messages')), (conversation_id,))
            return cursor.fetchall()

    def save_complete_conversation_turn(self, session_id, conversation_id, question, answer):
        if not question.strip() or not answer.strip():
            raise ValueError('问题和回答不能为空')
        with self._cursor() as cursor:
            # Serialize ordinal allocation across connections, including workers.
            cursor.execute(sql.SQL('SELECT title,title_is_custom FROM {} WHERE id=%s AND {}=%s FOR UPDATE').format(self._table('conversations'), sql.Identifier(self.owner_column)), (conversation_id, session_id))
            row = cursor.fetchone()
            if row is None:
                return None
            now = _utc_now()
            cursor.execute(sql.SQL('SELECT COALESCE(MAX(ordinal),0)+1 FROM {} WHERE conversation_id=%s').format(self._table('conversation_messages')), (conversation_id,))
            ordinal = cursor.fetchone()[0]
            cursor.executemany(sql.SQL('INSERT INTO {} (conversation_id,ordinal,role,content,created_at) VALUES (%s,%s,%s,%s,%s)').format(self._table('conversation_messages')),
                               [(conversation_id, ordinal, 'user', question, now), (conversation_id, ordinal+1, 'assistant', answer, now)])
            title = _title_from_question(question) if not row[1] and row[0] == '新对话' else row[0]
            cursor.execute(sql.SQL('UPDATE {} SET title=%s,updated_at=%s WHERE id=%s').format(self._table('conversations'), sql.Identifier(self.owner_column)), (title, now, conversation_id))
        return Conversation(conversation_id, title, now)

    def load_messages(self, session_id):
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('SELECT role,content FROM {} WHERE session_id=%s ORDER BY ordinal').format(self._table('messages')), (session_id,))
            return cursor.fetchall()

    def save_complete_turn(self, session_id, question, answer):
        if not question.strip() or not answer.strip():
            raise ValueError('问题和回答不能为空')
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE id=%s FOR UPDATE').format(self._table('sessions')), (session_id,))
            if cursor.fetchone() is None:
                raise ValueError('会话不存在')
            cursor.execute(sql.SQL('SELECT COALESCE(MAX(ordinal),0)+1 FROM {} WHERE session_id=%s').format(self._table('messages')), (session_id,))
            ordinal, now = cursor.fetchone()[0], _utc_now()
            cursor.executemany(sql.SQL('INSERT INTO {} (session_id,ordinal,role,content,created_at) VALUES (%s,%s,%s,%s,%s)').format(self._table('messages')),
                               [(session_id, ordinal, 'user', question, now), (session_id, ordinal+1, 'assistant', answer, now)])


class PostgresUserConversationStore(PostgresConversationStore):
    """Same operations with an authenticated user ID as the owner."""
    owner_column = 'user_id'

    def __init__(self, *, schema='public', **connection_settings):
        super().__init__(schema=schema, **connection_settings)
        from prepare_web_login import login_schema_ready
        with self._cursor() as cursor:
            if not login_schema_ready(cursor, schema):
                raise RuntimeError('Login migration is not ready')
