"""Read-only account check, or explicit preparation with --apply.

The local credentials file is ignored by Git. It is never printed. Its existence
is not proof of database commit; keep it if preparation fails, then reuse it.
"""

import argparse
import json
import secrets
import sys
from pathlib import Path
from uuid import UUID, uuid4

import psycopg2
from psycopg2 import sql

from account_storage import ACCOUNT_COLUMNS, MIGRATION, account_schema_ready
from database_settings import postgres_settings, read_settings
from password_security import hash_password, validate_password, verify_password
from web_storage import _utc_now


DEMO_ACCOUNTS = (
    ('demo_ab', 'level_ab', 'A+B', ('A', 'B')),
    ('demo_c', 'level_c', 'C', ('C',)),
    ('demo_abc', 'level_abc', 'A+B+C', ('A', 'B', 'C')),
)

# The three local teaching accounts allow six-character passwords by user request.
DEMO_PASSWORD_MINIMUM_LENGTH = 6


def _initialize_schema(cursor, schema):
    # Require the already prepared application database; never create a new one.
    cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(sql.Identifier(schema, 'schema_migrations')), ('v8_to_v82_conversations',))
    if cursor.fetchone() is None:
        raise ValueError('The conversation migration must be prepared first')
    cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s', (schema,))
    existing = {row[0] for row in cursor.fetchall()} & ACCOUNT_COLUMNS.keys()
    if existing:
        if not account_schema_ready(cursor, schema):
            raise ValueError('Account migration has unmarked existing tables; refusing to adopt unknown constraints')
        return False
    definitions = {
        'account_levels': 'id TEXT PRIMARY KEY, display_name TEXT NOT NULL',
        'account_level_libraries': 'level_id TEXT NOT NULL REFERENCES {}.account_levels(id), knowledge_base_id TEXT NOT NULL, PRIMARY KEY(level_id,knowledge_base_id)',
        'users': "id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL CHECK(username ~ '^[a-z][a-z0-9_]{{2,31}}$'), password_hash TEXT NOT NULL CHECK(password_hash LIKE '$argon2id$%'), level_id TEXT NOT NULL REFERENCES {}.account_levels(id), is_active BOOLEAN NOT NULL DEFAULT true, created_at TEXT NOT NULL",
    }
    for name, definition in definitions.items():
        body = sql.SQL(definition)
        if '{}' in definition:
            body = body.format(sql.Identifier(schema))
        cursor.execute(sql.SQL('CREATE TABLE {} ({})').format(sql.Identifier(schema, name), body))
    return True


def _credentials(path):
    if not path.exists():
        data = {'format_version': 1, 'accounts': [
            {'user_id': uuid4().hex, 'username': username, 'level_id': level_id,
             'password': secrets.token_urlsafe(24)}
            for username, level_id, _, _ in DEMO_ACCOUNTS
        ]}
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental credential replacement.
        with path.open('x', encoding='utf8') as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write('\n')
    try:
        data = json.loads(path.read_text(encoding='utf8'))
        if data['format_version'] != 1 or len(data['accounts']) != len(DEMO_ACCOUNTS):
            raise ValueError
        entries = {entry['username']: entry for entry in data['accounts']}
        if set(entries) != {item[0] for item in DEMO_ACCOUNTS}:
            raise ValueError
        for username, level_id, _, _ in DEMO_ACCOUNTS:
            entry = entries[username]
            if entry['level_id'] != level_id or UUID(entry['user_id']).hex != entry['user_id']:
                raise ValueError
            validate_password(entry['password'], minimum_length=DEMO_PASSWORD_MINIMUM_LENGTH)
        if len({entry['user_id'] for entry in entries.values()}) != 3:
            raise ValueError
        return entries
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError('Local demo credentials are invalid; refusing to replace them') from error


def _check(cursor, schema):
    ready = account_schema_ready(cursor, schema)
    if not ready:
        return {'ready': False, 'account_count': 0, 'level_count': 0}
    cursor.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(schema, 'users')))
    count = cursor.fetchone()[0]
    cursor.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(schema, 'account_levels')))
    return {'ready': True, 'account_count': count, 'level_count': cursor.fetchone()[0]}


def prepare_accounts(connection, credentials_path, *, apply=False, schema='public'):
    credentials_path = Path(credentials_path)
    if not apply:
        with connection.cursor() as cursor:
            return {'action': 'check_only', 'schema': schema, **_check(cursor, schema)}
    with connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL lock_timeout='5s'")
        cursor.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('legal-rag-accounts:' + schema,))
        changed = _initialize_schema(cursor, schema)
        # A missing file must not generate replacement passwords for existing users.
        if not credentials_path.exists():
            cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE username=ANY(%s)').format(sql.Identifier(schema, 'users')),
                           ([item[0] for item in DEMO_ACCOUNTS],))
            if cursor.fetchone() is not None:
                raise ValueError('Existing demo accounts require the original local credentials; refusing to reset')
        entries = _credentials(credentials_path)
        for username, level_id, display_name, knowledge_bases in DEMO_ACCOUNTS:
            cursor.execute(sql.SQL('SELECT display_name FROM {} WHERE id=%s FOR UPDATE').format(sql.Identifier(schema, 'account_levels')), (level_id,))
            existing = cursor.fetchone()
            if existing is None:
                cursor.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s)').format(sql.Identifier(schema, 'account_levels')), (level_id, display_name))
                cursor.executemany(sql.SQL('INSERT INTO {} VALUES (%s,%s)').format(sql.Identifier(schema, 'account_level_libraries')),
                                   [(level_id, library) for library in knowledge_bases])
                changed = True
            else:
                cursor.execute(sql.SQL('SELECT knowledge_base_id FROM {} WHERE level_id=%s').format(sql.Identifier(schema, 'account_level_libraries')), (level_id,))
                if existing[0] != display_name or {row[0] for row in cursor.fetchall()} != set(knowledge_bases):
                    raise ValueError('Demo level configuration conflict; refusing to overwrite')
            entry = entries[username]
            cursor.execute(sql.SQL('SELECT id,password_hash,level_id,is_active FROM {} WHERE username=%s FOR UPDATE').format(sql.Identifier(schema, 'users')), (username,))
            existing = cursor.fetchone()
            if existing is None:
                cursor.execute(sql.SQL('INSERT INTO {} (id,username,password_hash,level_id,is_active,created_at) VALUES (%s,%s,%s,%s,true,%s)').format(sql.Identifier(schema, 'users')),
                               (entry['user_id'], username, hash_password(entry['password'], minimum_length=DEMO_PASSWORD_MINIMUM_LENGTH), level_id, _utc_now()))
                changed = True
            elif existing[0] != entry['user_id'] or existing[2] != level_id or not existing[3] or not verify_password(existing[1], entry['password']):
                raise ValueError('Demo account credentials or status conflict; refusing to overwrite')
        cursor.execute(sql.SQL('INSERT INTO {} (name) VALUES (%s) ON CONFLICT DO NOTHING').format(sql.Identifier(schema, 'schema_migrations')), (MIGRATION,))
        report = {'action': 'prepared' if changed else 'already_ready', 'schema': schema, **_check(cursor, schema),
                  'credentials_file': str(credentials_path)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schema', help='Override configured PostgreSQL schema')
    parser.add_argument('--credentials-file', type=Path, default=Path('.demo-accounts.json'))
    parser.add_argument('--apply', action='store_true', help='Explicitly prepare tables and demo accounts')
    args = parser.parse_args(argv)
    connection = None
    try:
        settings = postgres_settings(read_settings())
        schema = args.schema or settings.pop('schema')
        settings.pop('schema', None)
        connection = psycopg2.connect(**settings, connect_timeout=5)
        if not args.apply:
            connection.set_session(readonly=True)
        report = prepare_accounts(connection, args.credentials_file, apply=args.apply, schema=schema)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, psycopg2.Error):
        # Database diagnostics can contain full rows, including password hashes.
        print('Account preparation failed; check database/schema and local credential file. No credentials were printed.', file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == '__main__':
    raise SystemExit(main())
