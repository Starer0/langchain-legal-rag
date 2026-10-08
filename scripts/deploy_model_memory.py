"""Explicit local migration after capability evidence; services must be stopped."""
import argparse
import json
import socket
import subprocess
import sys
import shutil
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import psycopg2
from psycopg2 import sql
from database_settings import read_settings,postgres_settings
from memory_fact_storage import digest
from prepare_memory_tools import prepare_memory_tools


def document_signature(c,schema):
    result={}
    for name in ('account_memories','account_memory_entries'):
        c.execute(sql.SQL('SELECT row_to_json(t) FROM {} t ORDER BY user_id').format(sql.Identifier(schema,name)))
        rows=sorted([r[0] for r in c.fetchall()],key=lambda r:json.dumps(r,sort_keys=True,default=str))
        result[name]=dict(rows=len(rows),digest=digest(rows))
    return result


def deploy(proof,*,apply=False):
    settings=read_settings();config=postgres_settings(settings);schema=config.pop('schema')
    rows=json.loads(Path(proof).read_text(encoding='utf-8'))
    model=settings.get('MODEL_NAME','deepseek-chat')
    if not any(r.get('model')==model and set(r.get('native_tools',[]))=={'plan_answer','update_memory'} and not r.get('error_type') for r in rows):
        raise ValueError('Native multi-tool capability evidence for configured model is required')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8001))==0:raise RuntimeError('Stop the project service on port 8001 before migration')
    with closing(psycopg2.connect(**config,connect_timeout=5)) as db:
        with db.cursor() as c:
            c.execute(sql.SQL("SELECT count(*) FROM {} WHERE status='running'").format(sql.Identifier(schema,'generation_tasks')))
            running=c.fetchone()[0]
            before=document_signature(c,schema)
        db.commit()
        if running:raise RuntimeError('Running generation tasks require a controlled shutdown/recovery first')
        if not apply:return dict(action='check_only',active_generations=running,capability_model=model,documents=before)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        root=Path('data/backups')/('model-memory-'+stamp);root.mkdir(parents=True,exist_ok=False)
        env_path=Path('.env');shutil.copy2(env_path,root/'env.before')
        dump=subprocess.run(['docker','exec','legal-rag-postgres','pg_dump','-U',config['user'],'-d',config['dbname'],'-Fc'],capture_output=True,check=True).stdout
        if not dump.startswith(b'PGDMP'):raise RuntimeError('Invalid backup archive')
        (root/'database.dump').write_bytes(dump)
        # The archive catalogue verifies the backup can be parsed. Full restore is
        # tested separately on an isolated schema, never on the user's database.
        subprocess.run(['docker','exec','-i','legal-rag-postgres','pg_restore','-l'],input=dump,capture_output=True,check=True)
        migration=prepare_memory_tools(db,apply=True,schema=schema)
        with db.cursor() as c:after=document_signature(c,schema)
        if before!=after:raise RuntimeError('Legacy memory documents changed; keep services stopped')
        original=env_path.read_text(encoding='utf-8')
        lines=[line for line in original.splitlines() if not line.startswith('MODEL_MEMORY_TOOLS_ENABLED=')]
        env_path.write_text('\n'.join(lines+['MODEL_MEMORY_TOOLS_ENABLED=true'])+'\n',encoding='utf-8')
        result=dict(action='migrated',migration=migration,backup=str(root.resolve()),backup_bytes=len(dump),
            capability_model=model,active_generations=running,legacy_documents_unchanged=True,documents=after)
        (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capability-results',required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();print(json.dumps(deploy(args.capability_results,apply=args.apply),ensure_ascii=False))
