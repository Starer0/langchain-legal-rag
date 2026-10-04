"""Explicit additive migration for account-owned conversations and login sessions."""

import argparse
import json
import sys

import psycopg2
from psycopg2 import sql

from account_storage import account_schema_ready
from database_settings import postgres_settings, read_settings


MIGRATION = 'v12_login_v1'


def login_schema_ready(cursor, schema):
    cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(sql.Identifier(schema, 'schema_migrations')), (MIGRATION,))
    if cursor.fetchone() is None:
        return False
    cursor.execute(sql.SQL('SELECT token_hash,user_id,csrf_token,created_at,last_seen_at,expires_at FROM {} LIMIT 0').format(sql.Identifier(schema, 'login_sessions')))
    cursor.execute(sql.SQL('SELECT user_id FROM {} LIMIT 0').format(sql.Identifier(schema, 'conversations')))
    return True


def prepare_login(connection, *, apply=False, schema='public'):
    if not apply:
        with connection.cursor() as cursor:
            return {'action': 'check_only', 'schema': schema, 'ready': login_schema_ready(cursor, schema)}
    with connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL lock_timeout='5s'")
        cursor.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('legal-rag-login:' + schema,))
        if login_schema_ready(cursor, schema):
            return {'action': 'already_ready', 'schema': schema, 'ready': True}
        if not account_schema_ready(cursor, schema):
            raise ValueError('Prepare the account migration first')
        cursor.execute('SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s', (schema, 'login_sessions'))
        existing_table = cursor.fetchone() is not None
        cursor.execute('SELECT 1 FROM information_schema.columns WHERE table_schema=%s AND table_name=%s AND column_name=%s', (schema, 'conversations', 'user_id'))
        if existing_table or cursor.fetchone() is not None:
            raise ValueError('Unmarked login structures exist; refusing to adopt or overwrite')
        cursor.execute(sql.SQL('ALTER TABLE {} ADD COLUMN user_id TEXT REFERENCES {}(id), ALTER COLUMN session_id DROP NOT NULL, '
                               'ADD CONSTRAINT conversation_one_owner CHECK ((session_id IS NOT NULL) <> (user_id IS NOT NULL))').format(
                                   sql.Identifier(schema, 'conversations'), sql.Identifier(schema, 'users')))
        cursor.execute(sql.SQL('CREATE INDEX ON {} (user_id, updated_at DESC)').format(sql.Identifier(schema, 'conversations')))
        cursor.execute(sql.SQL('CREATE TABLE {} (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES {}(id), '
                               'csrf_token TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, last_seen_at TIMESTAMPTZ NOT NULL, expires_at TIMESTAMPTZ NOT NULL)').format(
                                   sql.Identifier(schema, 'login_sessions'), sql.Identifier(schema, 'users')))
        cursor.execute(sql.SQL('INSERT INTO {} VALUES (%s)').format(sql.Identifier(schema, 'schema_migrations')), (MIGRATION,))
    return {'action': 'prepared', 'schema': schema, 'ready': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    connection = None
    try:
        settings = postgres_settings(read_settings())
        schema = settings.pop('schema')
        connection = psycopg2.connect(**settings, connect_timeout=5)
        if not args.apply:
            connection.set_session(readonly=True)
        print(json.dumps(prepare_login(connection, apply=args.apply, schema=schema)))
        return 0
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, psycopg2.Error):
        print('Login migration failed; verify prepared database and schema.', file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == '__main__':
    raise SystemExit(main())
