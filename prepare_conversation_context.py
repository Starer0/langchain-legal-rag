"""Explicit additive migration for private conversation context snapshots."""
import argparse
import json
import psycopg2
from psycopg2 import sql
from database_settings import postgres_settings, read_settings

MIGRATION = 'v12_conversation_context_v1'
TABLES = ('conversation_contexts', 'generation_context_inputs')


def prepare_context(connection, *, apply=False, schema='public'):
    table = lambda name: sql.Identifier(schema,name)
    with connection, connection.cursor() as c:
        if apply:
            c.execute("SET LOCAL lock_timeout='5s'")
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('legal-rag-context:'+schema,))
        c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(table('schema_migrations')),(MIGRATION,))
        ready = c.fetchone() is not None
        if ready:
            for name in TABLES: c.execute(sql.SQL('SELECT * FROM {} LIMIT 0').format(table(name)))
        if ready or not apply: return {'ready':ready,'action':'already_ready' if ready else 'check_only'}
        c.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND table_name=ANY(%s)',(schema,list(TABLES)))
        if c.fetchone(): raise ValueError('Unmarked context tables exist; refusing adoption')
        c.execute(sql.SQL('CREATE TABLE {} (conversation_id TEXT PRIMARY KEY REFERENCES {}(id) ON DELETE CASCADE,'
            'user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,revision BIGINT NOT NULL CHECK(revision>0),'
            'through_ordinal BIGINT NOT NULL CHECK(through_ordinal>=0),items JSONB NOT NULL,updated_at TEXT NOT NULL)')
            .format(table('conversation_contexts'),table('conversations'),table('users')))
        c.execute(sql.SQL('CREATE TABLE {} (task_id TEXT PRIMARY KEY REFERENCES {}(id) ON DELETE CASCADE,'
            'user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,conversation_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,'
            'input JSONB NOT NULL,result JSONB)')
            .format(table('generation_context_inputs'),table('generation_tasks'),table('users'),table('conversations')))
        c.execute(sql.SQL('INSERT INTO {} VALUES (%s)').format(table('schema_migrations')),(MIGRATION,))
    return {'ready':True,'action':'prepared'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--apply',action='store_true')
    args = parser.parse_args(); settings = postgres_settings(read_settings()); schema = settings.pop('schema')
    with psycopg2.connect(**settings,connect_timeout=5) as conn:
        print(json.dumps(prepare_context(conn,apply=args.apply,schema=schema)))
