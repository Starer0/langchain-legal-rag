"""Explicit additive migration for recoverable tasks and official graph tables."""
import argparse
import json
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from checkpoint_storage import graph_schema, serializer
from task_recovery import schema_sql, MIGRATION
from database_settings import read_settings, postgres_settings

PREPARING = MIGRATION + '_preparing'


def prepare_recovery(settings, *, schema='public', apply=False):
    settings = dict(settings)
    settings.pop('schema', None)
    settings.setdefault('connect_timeout', 5)
    target = graph_schema(schema)
    with psycopg.connect(**settings, autocommit=True, row_factory=dict_row) as c:
        if apply:
            c.execute('SELECT pg_advisory_lock(hashtext(%s))', ('legal-rag-recovery-migrate:' + schema,))
        table = sql.Identifier(schema, 'schema_migrations')
        ready = c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(table), (MIGRATION,)).fetchone() is not None
        if not apply or ready: return {'ready': ready, 'action': 'already_ready' if ready else 'check_only'}
        preparing = c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(table), (PREPARING,)).fetchone() is not None
        if not preparing:
            if c.execute('SELECT 1 FROM information_schema.schemata WHERE schema_name=%s', (target,)).fetchone():
                raise ValueError('Unmarked checkpoint schema exists; refusing adoption')
            with c.transaction():
                for statement in schema_sql(sql.Identifier(schema, 'generation_recovery').as_string(c),
                    sql.Identifier(schema, 'generation_tasks').as_string(c), sql.Identifier(schema, 'checkpoint_cleanup').as_string(c)):
                    c.execute(statement)
                c.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(target)))
                c.execute(sql.SQL('INSERT INTO {} VALUES (%s)').format(table), (PREPARING,))
        c.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(target)))
        PostgresSaver(c, serde=serializer()).setup()
        with c.transaction():
            c.execute(sql.SQL('DELETE FROM {} WHERE name=%s').format(table), (PREPARING,))
            c.execute(sql.SQL('INSERT INTO {} VALUES (%s)').format(table), (MIGRATION,))
        return {'ready': True, 'action': 'prepared'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    settings = postgres_settings(read_settings())
    schema = settings.pop('schema')
    print(json.dumps(prepare_recovery(settings, schema=schema, apply=args.apply)))
