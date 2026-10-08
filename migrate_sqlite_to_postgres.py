"""Check or explicitly import the existing anonymous web store into PostgreSQL.

The source is read-only. Import requires an empty target or exact matching data.
This command does not change the web backend or assign anonymous chats to users.
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch


COLUMNS = {
    'sessions': ('id', 'created_at', 'last_seen_at'),
    'messages': ('session_id', 'ordinal', 'role', 'content', 'created_at'),
    'schema_migrations': ('name',),
    'conversations': ('id', 'session_id', 'title', 'title_is_custom', 'created_at', 'updated_at'),
    'conversation_messages': ('conversation_id', 'ordinal', 'role', 'content', 'created_at'),
}

# All timestamps remain TEXT to preserve the exact legacy values on this step.
DEFINITIONS = {
    'sessions': 'id TEXT PRIMARY KEY, created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL',
    'messages': 'session_id TEXT NOT NULL, ordinal INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(session_id, ordinal)',
    'schema_migrations': 'name TEXT PRIMARY KEY',
    'conversations': 'id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES {}.sessions(id), title TEXT NOT NULL, title_is_custom INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL',
    'conversation_messages': "conversation_id TEXT NOT NULL REFERENCES {}.conversations(id) ON DELETE CASCADE, ordinal INTEGER NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(conversation_id, ordinal)",
}


def read_source(path):
    path = Path(path).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('BEGIN')  # One consistent read snapshot, even if the web app is running.
        if connection.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('SQLite has broken foreign-key relationships')
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversation_sources'").fetchone():
            if connection.execute('SELECT 1 FROM conversation_sources LIMIT 1').fetchone():
                raise ValueError('SQLite contains source snapshots; this legacy migration cannot preserve them')
        result = {}
        for table, columns in COLUMNS.items():
            actual = tuple(row[1] for row in connection.execute(f'PRAGMA table_info("{table}")'))
            if actual != columns:
                raise ValueError(f'Unsupported SQLite table structure: {table}')
            selection = ','.join('"' + column + '"' for column in columns)
            result[table] = connection.execute(f'SELECT {selection} FROM "{table}"').fetchall()
        return result
    finally:
        connection.close()


def read_target(cursor, schema):
    cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s', (schema,))
    existing = {row[0] for row in cursor.fetchall()} & COLUMNS.keys()
    if existing and existing != COLUMNS.keys():
        raise ValueError('Target has an incomplete set of migration tables; refusing to modify it')
    if not existing:
        return None
    result = {}
    for table, columns in COLUMNS.items():
        cursor.execute(sql.SQL('SELECT {} FROM {}.{}').format(
            sql.SQL(',').join(map(sql.Identifier, columns)), sql.Identifier(schema), sql.Identifier(table),
        ))
        result[table] = cursor.fetchall()
    return result


def _matches(source, target):
    return target is not None and all(Counter(source[table]) == Counter(target[table]) for table in COLUMNS)


def migrate(source_path, connection, *, apply=False, schema='public'):
    source = read_source(source_path)
    counts = {table: len(rows) for table, rows in source.items()}
    report = {'schema': schema, 'source_counts': counts}
    if not apply:
        with connection.cursor() as cursor:
            target = read_target(cursor, schema)
        return {**report, 'action': 'check_only',
                'target_counts': None if target is None else {table: len(rows) for table, rows in target.items()},
                'target_matches': _matches(source, target)}

    # PostgreSQL DDL and imports share one transaction: any error rolls everything back.
    with connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL lock_timeout = '5s'")
        cursor.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('legal-rag-migration:' + schema,))
        target = read_target(cursor, schema)
        if target is not None:
            for table in COLUMNS:
                cursor.execute(sql.SQL('LOCK TABLE {}.{} IN ACCESS EXCLUSIVE MODE').format(
                    sql.Identifier(schema), sql.Identifier(table),
                ))
            target = read_target(cursor, schema)
            if _matches(source, target):
                return {**report, 'action': 'already_matches'}
            if any(target.values()):
                raise ValueError('Target contains different data; refusing to overwrite or merge')
        else:
            cursor.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
            for table, definition in DEFINITIONS.items():
                body = sql.SQL(definition)
                if '{}' in definition:
                    body = body.format(sql.Identifier(schema))
                cursor.execute(sql.SQL('CREATE TABLE {}.{} ({})').format(
                    sql.Identifier(schema), sql.Identifier(table), body,
                ))
        for table, columns in COLUMNS.items():
            statement = sql.SQL('INSERT INTO {}.{} ({}) VALUES ({})').format(
                sql.Identifier(schema), sql.Identifier(table),
                sql.SQL(',').join(map(sql.Identifier, columns)),
                sql.SQL(',').join(sql.Placeholder() for _ in columns),
            )
            execute_batch(cursor, statement, source[table])
        if not _matches(source, read_target(cursor, schema)):
            raise ValueError('Target verification failed; import rolled back')
    return {**report, 'action': 'imported'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/web_rag.sqlite3'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5433)
    parser.add_argument('--database', default='legal_rag')
    parser.add_argument('--user', default='legal_rag')
    parser.add_argument('--password-file', type=Path, default=Path('.postgres-password'))
    parser.add_argument('--schema', default='public')
    parser.add_argument('--apply', action='store_true', help='Explicitly import; default is read-only checking')
    args = parser.parse_args(argv)
    connection = None
    try:
        password = args.password_file.read_text(encoding='utf8').strip()
        connection = psycopg2.connect(host=args.host, port=args.port, dbname=args.database,
                                      user=args.user, password=password, connect_timeout=5)
        if not args.apply:
            connection.set_session(readonly=True)
        report = migrate(args.source, connection, apply=args.apply, schema=args.schema)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error, psycopg2.Error) as error:
        # Database errors can contain full rows. Do not print chat contents or credentials.
        print(f'Migration check/import failed ({type(error).__name__}); verify connection, source and target schema.', file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == '__main__':
    raise SystemExit(main())
