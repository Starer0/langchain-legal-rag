"""Explicit additive PostgreSQL migration for per-answer source snapshots."""

import argparse
import json
import sys

import psycopg2
from psycopg2 import sql

from database_settings import postgres_settings, read_settings

MIGRATION = 'v12_source_history_v1'


def source_schema_ready(cursor, schema):
    cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(
        sql.Identifier(schema, 'schema_migrations')), (MIGRATION,))
    if cursor.fetchone() is None:
        return False
    cursor.execute(sql.SQL('SELECT conversation_id,ordinal,sources FROM {} LIMIT 0').format(
        sql.Identifier(schema, 'conversation_sources')))
    return True


def prepare_sources(connection, *, apply=False, schema='public'):
    if not apply:
        with connection.cursor() as cursor:
            return {'action': 'check_only', 'schema': schema, 'ready': source_schema_ready(cursor, schema)}
    with connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL lock_timeout='5s'")
        cursor.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('legal-rag-sources:' + schema,))
        if source_schema_ready(cursor, schema):
            return {'action': 'already_ready', 'schema': schema, 'ready': True}
        cursor.execute('SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s',
                       (schema, 'conversation_sources'))
        if cursor.fetchone() is not None:
            raise ValueError('Unmarked source history table exists; refusing to overwrite')
        cursor.execute(sql.SQL("CREATE TABLE {} (conversation_id TEXT NOT NULL, ordinal INTEGER NOT NULL, "
                               "sources JSONB NOT NULL CHECK(jsonb_typeof(sources)='array'), "
                               "PRIMARY KEY(conversation_id,ordinal), "
                               "FOREIGN KEY(conversation_id,ordinal) REFERENCES {}(conversation_id,ordinal) ON DELETE CASCADE)").format(
                                   sql.Identifier(schema, 'conversation_sources'),
                                   sql.Identifier(schema, 'conversation_messages')))
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
        print(json.dumps(prepare_sources(connection, apply=args.apply, schema=schema)))
        return 0
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, psycopg2.Error):
        print('Source history migration failed; verify database and schema.', file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == '__main__':
    raise SystemExit(main())
