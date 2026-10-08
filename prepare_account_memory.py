"""Explicit additive account-memory migration; never change existing legal data."""
import argparse
import json
import psycopg2
from psycopg2 import sql
from memory_storage import MIGRATION
from database_settings import postgres_settings, read_settings

TABLES = ('account_memories', 'account_memory_entries', 'generation_memory_inputs')


def prepare_memory(connection, *, apply=False, schema='public'):
    def table(name): return sql.Identifier(schema, name)
    with connection, connection.cursor() as c:
        if apply:
            c.execute("SET LOCAL lock_timeout='5s'")
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('legal-rag-memory:' + schema,))
        c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(table('schema_migrations')), (MIGRATION,))
        ready = c.fetchone() is not None
        if ready:
            for name in TABLES: c.execute(sql.SQL('SELECT * FROM {} LIMIT 0').format(table(name)))
        if not apply or ready: return {'ready': ready, 'action': 'already_ready' if ready else 'check_only'}
        c.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND table_name=ANY(%s)', (schema, list(TABLES)))
        if c.fetchone(): raise ValueError('Unmarked memory tables exist; refusing adoption')
        c.execute(sql.SQL('CREATE TABLE {} (user_id TEXT PRIMARY KEY REFERENCES {}(id) ON DELETE CASCADE, '
            'revision BIGINT NOT NULL CHECK(revision>0),enabled BOOLEAN NOT NULL,core_text TEXT NOT NULL,extended_text TEXT NOT NULL,updated_at TEXT NOT NULL)').format(table('account_memories'), table('users')))
        c.execute(sql.SQL('CREATE TABLE {} (user_id TEXT NOT NULL REFERENCES {}(user_id) ON DELETE CASCADE,id TEXT NOT NULL, '
            'revision BIGINT NOT NULL,ordinal INTEGER NOT NULL,text TEXT NOT NULL,embedding JSONB NOT NULL,fingerprint TEXT NOT NULL,'
            'PRIMARY KEY(user_id,id),UNIQUE(user_id,ordinal))').format(table('account_memory_entries'), table('account_memories')))
        c.execute(sql.SQL('CREATE TABLE {} (task_id TEXT PRIMARY KEY REFERENCES {}(id) ON DELETE CASCADE,'
            'user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,input JSONB NOT NULL,observation JSONB NOT NULL)').format(table('generation_memory_inputs'), table('generation_tasks'), table('users')))
        c.execute(sql.SQL('INSERT INTO {} VALUES (%s)').format(table('schema_migrations')), (MIGRATION,))
    return {'ready': True, 'action': 'prepared'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    settings = postgres_settings(read_settings())
    schema = settings.pop('schema')
    with psycopg2.connect(**settings, connect_timeout=5) as connection:
        print(json.dumps(prepare_memory(connection, apply=args.apply, schema=schema)))
