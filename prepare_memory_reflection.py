"""Explicit additive migration for durable opt-in automatic memory reflection."""
import argparse
import json
import psycopg2
from psycopg2 import sql
from database_settings import postgres_settings, read_settings

MIGRATION='v12_memory_reflection_v1'
TABLES=('memory_reflection_policies','memory_reflection_progress','memory_reflection_jobs','memory_reflection_suggestions')


def prepare_reflection(connection,*,apply=False,schema='public'):
    t=lambda name:sql.Identifier(schema,name)
    with connection,connection.cursor() as c:
        if apply:
            c.execute("SET LOCAL lock_timeout='5s'")
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('legal-rag-reflection:'+schema,))
        c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(t('schema_migrations')),(MIGRATION,))
        ready=c.fetchone() is not None
        if ready:
            for name in TABLES:c.execute(sql.SQL('SELECT * FROM {} LIMIT 0').format(t(name)))
        if ready or not apply:return {'ready':ready,'action':'already_ready' if ready else 'check_only'}
        c.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND table_name=ANY(%s)',(schema,list(TABLES)))
        if c.fetchone():raise ValueError('Unmarked reflection tables exist; refusing adoption')
        c.execute(sql.SQL('CREATE TABLE {} (user_id TEXT PRIMARY KEY REFERENCES {}(id) ON DELETE CASCADE,enabled BOOLEAN NOT NULL,epoch BIGINT NOT NULL CHECK(epoch>0))').format(t(TABLES[0]),t('users')))
        c.execute(sql.SQL('CREATE TABLE {} (conversation_id TEXT PRIMARY KEY REFERENCES {}(id) ON DELETE CASCADE,user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,processed_through BIGINT NOT NULL DEFAULT 0,invalidated_through BIGINT NOT NULL DEFAULT 0)').format(t(TABLES[1]),t('conversations'),t('users')))
        c.execute(sql.SQL("CREATE TABLE {} (id TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,conversation_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,start_ordinal BIGINT NOT NULL,end_ordinal BIGINT NOT NULL,due_at TIMESTAMPTZ NOT NULL,status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed','obsolete')),epoch BIGINT NOT NULL,policy_epoch BIGINT NOT NULL,memory_revision BIGINT NOT NULL,attempts INT NOT NULL,input JSONB NOT NULL,result JSONB,reason TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(conversation_id,start_ordinal,end_ordinal))").format(t(TABLES[2]),t('users'),t('conversations')))
        c.execute(sql.SQL('CREATE INDEX {} ON {} (status,due_at)').format(sql.Identifier(schema+'_reflection_due'),t(TABLES[2])))
        c.execute(sql.SQL("CREATE TABLE {} (id TEXT PRIMARY KEY,job_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,conversation_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,memory_revision BIGINT NOT NULL,policy_epoch BIGINT NOT NULL,proposal JSONB NOT NULL,status TEXT NOT NULL CHECK(status IN ('pending','accepted','ignored','obsolete')),created_at TEXT NOT NULL)").format(t(TABLES[3]),t(TABLES[2]),t('users'),t('conversations')))
        c.execute(sql.SQL('INSERT INTO {} VALUES (%s)').format(t('schema_migrations')),(MIGRATION,))
    return {'ready':True,'action':'prepared'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');a=p.parse_args()
    settings=postgres_settings(read_settings());schema=settings.pop('schema')
    with psycopg2.connect(**settings,connect_timeout=5) as conn:print(json.dumps(prepare_reflection(conn,apply=a.apply,schema=schema)))
