"""Explicit additive PostgreSQL migration for background generation tasks."""
import argparse
import json
import psycopg2
from psycopg2 import sql
from generation_tasks import schema_sql, MIGRATION
from database_settings import postgres_settings, read_settings


def prepare_tasks(connection, *, apply=False, schema='public'):
    with connection, connection.cursor() as cursor:
        if apply:
            cursor.execute("SET LOCAL lock_timeout='5s'")
            cursor.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', ('legal-rag-tasks:' + schema,))
        cursor.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(sql.Identifier(schema, 'schema_migrations')), (MIGRATION,))
        ready = cursor.fetchone() is not None
        if ready:
            cursor.execute(sql.SQL('SELECT id,status,submission_key FROM {} LIMIT 0').format(sql.Identifier(schema, 'generation_tasks')))
        if not apply or ready: return {'ready': ready, 'action': 'already_ready' if ready else 'check_only'}
        cursor.execute('SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s', (schema, 'generation_tasks'))
        if cursor.fetchone(): raise ValueError('Unmarked task table exists; refusing overwrite')
        table = sql.Identifier(schema, 'generation_tasks').as_string(connection)
        conversations = sql.Identifier(schema, 'conversations').as_string(connection)
        for statement in schema_sql(table, conversations, 'generation_one_active'):
            cursor.execute(statement)
        cursor.execute(sql.SQL('INSERT INTO {} VALUES (%s)').format(sql.Identifier(schema, 'schema_migrations')), (MIGRATION,))
    return {'ready': True, 'action': 'prepared'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    settings = postgres_settings(read_settings())
    schema = settings.pop('schema')
    with psycopg2.connect(**settings, connect_timeout=5) as connection:
        print(json.dumps(prepare_tasks(connection, apply=args.apply, schema=schema)))
