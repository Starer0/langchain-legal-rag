"""Account data foundation; web authentication and RAG enforcement follow later."""

import re
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg2
from psycopg2 import sql

from password_security import DUMMY_PASSWORD_HASH, verify_password


MIGRATION = 'v12_accounts_v1'
ACCOUNT_COLUMNS = {
    'account_levels': ('id', 'display_name'),
    'account_level_libraries': ('level_id', 'knowledge_base_id'),
    'users': ('id', 'username', 'password_hash', 'level_id', 'is_active', 'created_at'),
}


@dataclass(frozen=True)
class Account:
    id: str
    username: str
    level_id: str
    is_active: bool


def normalize_username(username):
    if not isinstance(username, str):
        return None
    username = username.strip().lower()
    return username if re.fullmatch(r'[a-z][a-z0-9_]{2,31}', username) else None


def account_schema_ready(cursor, schema):
    cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s', (schema,))
    existing = {row[0] for row in cursor.fetchall()} & ACCOUNT_COLUMNS.keys()
    if not existing:
        return False
    if existing != ACCOUNT_COLUMNS.keys():
        raise ValueError('Account migration has incomplete tables; refusing to modify it')
    for table, columns in ACCOUNT_COLUMNS.items():
        cursor.execute(sql.SQL('SELECT {} FROM {} LIMIT 0').format(
            sql.SQL(',').join(map(sql.Identifier, columns)), sql.Identifier(schema, table)))
    cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(sql.Identifier(schema, 'schema_migrations')), (MIGRATION,))
    return cursor.fetchone() is not None


class PostgresAccountStore:
    def __init__(self, *, schema='public', **connection_settings):
        self.schema = schema
        self._settings = {'connect_timeout': 5, **connection_settings}
        with self._cursor() as cursor:
            if not account_schema_ready(cursor, schema):
                raise RuntimeError('Account migration is not ready; run explicit account preparation first')

    @contextmanager
    def _cursor(self):
        connection = psycopg2.connect(**self._settings)
        try:
            with connection, connection.cursor() as cursor:
                yield cursor
        finally:
            connection.close()

    def _table(self, name):
        return sql.Identifier(self.schema, name)

    def list_accounts(self):
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('SELECT id,username,level_id,is_active FROM {} ORDER BY username').format(self._table('users')))
            return [Account(*row) for row in cursor.fetchall()]

    def get_account(self, user_id):
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('SELECT id,username,level_id,is_active FROM {} WHERE id=%s').format(self._table('users')), (user_id,))
            row = cursor.fetchone()
            return Account(*row) if row else None

    def authenticate(self, username, password):
        username = normalize_username(username)
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('SELECT id,username,level_id,is_active,password_hash FROM {} WHERE username=%s').format(self._table('users')), (username,))
            row = cursor.fetchone()
        valid = verify_password(row[4] if row else DUMMY_PASSWORD_HASH, password)
        if row is None or not valid or not row[3]:
            return None
        return Account(*row[:4])

    def allowed_knowledge_bases(self, user_id):
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('SELECT scope.knowledge_base_id FROM {} AS scope JOIN {} AS account '
                                   'ON account.level_id=scope.level_id WHERE account.id=%s AND account.is_active').format(
                                       self._table('account_level_libraries'), self._table('users')), (user_id,))
            return frozenset(row[0] for row in cursor.fetchall())
