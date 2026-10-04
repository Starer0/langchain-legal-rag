"""Persist opaque login sessions independently of conversation history."""

import hashlib
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2 import sql

from account_storage import Account
from prepare_web_login import login_schema_ready


SESSION_LIFETIME = timedelta(days=3)


@dataclass(frozen=True)
class LoginSession:
    user: Account
    token: str = field(repr=False)
    csrf_token: str = field(repr=False)
    expires_at: datetime


def _token_hash(token):
    if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}', token):
        return None
    return hashlib.sha256(token.encode('ascii')).hexdigest()


class PostgresLoginSessions:
    def __init__(self, accounts, *, schema='public', now=None, **connection_settings):
        self.accounts = accounts
        self.schema = schema
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._settings = {'connect_timeout': 5, **connection_settings}
        with self._cursor() as cursor:
            if not login_schema_ready(cursor, schema):
                raise RuntimeError('Login migration is not ready; run explicit login preparation first')

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

    def login(self, username, password):
        account = self.accounts.authenticate(username, password)
        if account is None:
            return None
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        now = self.now()
        expires = now + SESSION_LIFETIME
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('INSERT INTO {} (token_hash,user_id,csrf_token,created_at,last_seen_at,expires_at) '
                                   'SELECT %s,id,%s,%s,%s,%s FROM {} WHERE id=%s AND is_active RETURNING user_id').format(
                                       self._table('login_sessions'), self._table('users')),
                           (_token_hash(token), csrf, now, now, expires, account.id))
            if cursor.fetchone() is None:
                return None
        return LoginSession(account, token, csrf, expires)

    def lookup(self, token, *, renew=False):
        encoded = _token_hash(token)
        if encoded is None:
            return None
        now = self.now()
        with self._cursor() as cursor:
            if renew:
                cursor.execute(sql.SQL('UPDATE {} AS session SET last_seen_at=%s,expires_at=%s FROM {} AS account '
                                       'WHERE session.user_id=account.id AND account.is_active AND session.token_hash=%s AND session.expires_at>%s '
                                       'RETURNING account.id,account.username,account.level_id,account.is_active,session.csrf_token,session.expires_at').format(
                                           self._table('login_sessions'), self._table('users')),
                               (now, now + SESSION_LIFETIME, encoded, now))
            else:
                cursor.execute(sql.SQL('SELECT account.id,account.username,account.level_id,account.is_active,session.csrf_token,session.expires_at '
                                       'FROM {} AS session JOIN {} AS account ON account.id=session.user_id '
                                       'WHERE session.token_hash=%s AND session.expires_at>%s AND account.is_active').format(
                                           self._table('login_sessions'), self._table('users')), (encoded, now))
            row = cursor.fetchone()
        return LoginSession(Account(*row[:4]), token, row[4], row[5]) if row else None

    def logout(self, token):
        encoded = _token_hash(token)
        if encoded is None:
            return False
        with self._cursor() as cursor:
            cursor.execute(sql.SQL('DELETE FROM {} WHERE token_hash=%s').format(self._table('login_sessions')), (encoded,))
            return cursor.rowcount == 1
