"""Explicit additive migration; never rewrites legacy memory documents."""
import argparse
import json
from contextlib import closing
import psycopg2
from psycopg2 import sql

MIGRATION='v12_model_memory_tools_v1'
TABLES=('account_memory_facts','account_memory_fact_evidence','account_memory_tool_receipts','generation_turn_decisions')


def prepare_memory_tools(connection,*,apply=False,schema='public'):
    t=lambda name:sql.Identifier(schema,name)
    with connection,connection.cursor() as c:
        if apply:
            c.execute("SET LOCAL lock_timeout='5s'")
            c.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',('legal-rag-memory-tools:'+schema,))
        c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(t('schema_migrations')),(MIGRATION,))
        ready=c.fetchone() is not None
        if ready:
            for name in TABLES:c.execute(sql.SQL('SELECT * FROM {} LIMIT 0').format(t(name)))
        if ready or not apply:return dict(ready=ready,action='already_ready' if ready else 'check_only')
        c.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s AND table_name=ANY(%s)',(schema,list(TABLES)))
        if c.fetchone():raise ValueError('Unmarked memory tools tables exist')
        c.execute(sql.SQL("CREATE TABLE {} (user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,id TEXT NOT NULL,layer TEXT NOT NULL CHECK(layer IN ('core','extended')),category TEXT NOT NULL CHECK(category IN ('preference','background','learning_focus')),content TEXT NOT NULL,basis TEXT NOT NULL CHECK(basis IN ('declared','inferred')),protected BOOLEAN NOT NULL,retired BOOLEAN NOT NULL,revision BIGINT NOT NULL CHECK(revision>0),PRIMARY KEY(user_id,id))").format(t(TABLES[0]),t('users')))
        c.execute(sql.SQL("CREATE TABLE {} (user_id TEXT NOT NULL,fact_id TEXT NOT NULL,id TEXT NOT NULL,source_kind TEXT NOT NULL CHECK(source_kind IN ('current_input','message','manual_document','legacy_document')),reference TEXT,quote TEXT NOT NULL,candidate_content TEXT NOT NULL,recorded_at TIMESTAMPTZ NOT NULL,PRIMARY KEY(user_id,fact_id,id),FOREIGN KEY(user_id,fact_id) REFERENCES {}(user_id,id) ON DELETE CASCADE)").format(t(TABLES[1]),t(TABLES[0])))
        c.execute(sql.SQL('CREATE TABLE {} (user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,execution_id TEXT NOT NULL,argument_digest TEXT NOT NULL,result JSONB NOT NULL,created_at TIMESTAMPTZ NOT NULL,PRIMARY KEY(user_id,execution_id))').format(t(TABLES[2]),t('users')))
        c.execute(sql.SQL('CREATE TABLE {} (task_id TEXT PRIMARY KEY REFERENCES {}(id) ON DELETE CASCADE,user_id TEXT NOT NULL REFERENCES {}(id) ON DELETE CASCADE,schema_version TEXT NOT NULL,decision JSONB NOT NULL,input_digest TEXT NOT NULL,user_ordinal BIGINT,created_at TIMESTAMPTZ NOT NULL)').format(t(TABLES[3]),t('generation_tasks'),t('users')))
        # Foreground proposals use the same authenticated review surface.
        c.execute(sql.SQL('ALTER TABLE {} ALTER COLUMN job_id DROP NOT NULL').format(t('memory_reflection_suggestions')))
        c.execute('SELECT conname FROM pg_constraint WHERE conrelid=%s::regclass AND contype=\'u\' AND pg_get_constraintdef(oid)=\'UNIQUE (conversation_id, start_ordinal, end_ordinal)\'',(f'"{schema}"."memory_reflection_jobs"',))
        for (name,) in c.fetchall():
            c.execute(sql.SQL('ALTER TABLE {} DROP CONSTRAINT {}').format(t('memory_reflection_jobs'),sql.Identifier(name)))
        c.execute(sql.SQL("CREATE UNIQUE INDEX {} ON {} (conversation_id,start_ordinal,end_ordinal) WHERE status<>'obsolete'").format(sql.Identifier(schema+'_reflection_active_range'),t('memory_reflection_jobs')))
        c.execute(sql.SQL('INSERT INTO {} VALUES (%s)').format(t('schema_migrations')),(MIGRATION,))
        # Record paragraphs as protected legacy evidence without rewriting text.
        from memory_service import split_entries
        from memory_fact_storage import seed_fact
        c.execute(sql.SQL('SELECT user_id,core_text,extended_text,revision FROM {}').format(t('account_memories')))
        for owner,core,extended,revision in c.fetchall():
            for layer,text in [('core',core),('extended',extended)]:
                for index,part in enumerate(split_entries(text)):
                    seed_fact(c,t,owner,layer,part,revision,'legacy_document',index)
    return dict(ready=True,action='prepared')


if __name__=='__main__':
    from database_settings import read_settings,postgres_settings
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();config=postgres_settings(read_settings());schema=config.pop('schema')
    with closing(psycopg2.connect(**config,connect_timeout=5)) as connection:
        print(json.dumps(prepare_memory_tools(connection,apply=args.apply,schema=schema)))
