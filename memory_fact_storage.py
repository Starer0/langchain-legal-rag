"""Owner-scoped facts, evidence, decisions and receipts; caller owns transaction."""
import hashlib
import json
import uuid
from dataclasses import asdict
from psycopg2 import sql
from psycopg2.extras import Json
from memory_service import MemoryConflict,split_entries
from memory_tool_contract import VERSION,ToolResult,decision_from_dict
from web_storage import _utc_now


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def seed_fact(c,t,owner,layer,content,revision,kind,index):
    fid=uuid.uuid4().hex
    c.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,true,false,%s)').format(t('account_memory_facts')),
        (owner,fid,layer,'preference' if layer=='core' else 'background',content,'declared',revision))
    c.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,%s,%s)').format(t('account_memory_fact_evidence')),
        (owner,fid,digest([kind,revision,index,content]),kind,str(revision),content,content,_utc_now()))


class MemoryFactStore:
    def __init__(self,memory):
        self.memory=memory;self.table=memory.table
        from prepare_memory_tools import MIGRATION
        with memory.cursor() as c:
            c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(self.table('schema_migrations')),(MIGRATION,))
            if c.fetchone() is None:raise RuntimeError('Run prepare_memory_tools.py --apply first')

    def snapshot(self,owner,cursor=None):
        if cursor is None:
            with self.memory.cursor() as c:return self.snapshot(owner,c)
        c=cursor
        c.execute(sql.SQL('SELECT id,layer,category,content,basis,protected,revision FROM {} WHERE user_id=%s AND NOT retired ORDER BY revision,id').format(self.table('account_memory_facts')),(owner,))
        facts=[dict(zip(('id','layer','category','content','basis','protected','revision'),r)) for r in c.fetchall()]
        document=self.memory.read(owner,c)
        positions={(layer,part):index for layer,key in [('core','core_text'),('extended','extended_text')]
                   for index,part in enumerate(split_entries(document[key]))}
        facts.sort(key=lambda f:(f['layer']!='core',positions.get((f['layer'],f['content']),999),f['id']))
        for fact in facts:
            c.execute(sql.SQL('SELECT source_kind,reference,quote,candidate_content,recorded_at FROM {} WHERE user_id=%s AND fact_id=%s ORDER BY recorded_at,id').format(self.table('account_memory_fact_evidence')),(owner,fact['id']))
            fact['sources']=[dict(zip(('kind','reference','quote','candidate_content','recorded_at'),r)) for r in c.fetchall()]
            for source in fact['sources']:source['recorded_at']=source['recorded_at'].isoformat()
        return {'facts':facts}

    def sync_manual(self,owner,core,extended,revision,cursor):
        c=cursor;old=self.snapshot(owner,c)['facts'];used=set()
        for layer,text in [('core',core),('extended',extended)]:
            for index,part in enumerate(split_entries(text)):
                match=next((f for f in old if f['id'] not in used and f['layer']==layer and f['content']==part),None)
                if match:
                    used.add(match['id'])
                else:seed_fact(c,self.table,owner,layer,part,revision,'manual_document',index)
        for fact in old:
            if fact['id'] in used:continue
            c.execute(sql.SQL("UPDATE {} SET retired=true,content='' WHERE user_id=%s AND id=%s").format(self.table('account_memory_facts')),(owner,fact['id']))
            c.execute(sql.SQL('DELETE FROM {} WHERE user_id=%s AND fact_id=%s').format(self.table('account_memory_fact_evidence')),(owner,fact['id']))

    def receipt(self,owner,execution_id,cursor):
        cursor.execute(sql.SQL('SELECT argument_digest,result FROM {} WHERE user_id=%s AND execution_id=%s').format(self.table('account_memory_tool_receipts')),(owner,execution_id))
        row=cursor.fetchone()
        if row:
            result=dict(row[1]);result['suggestion_ids']=tuple(result['suggestion_ids'])
            return row[0],ToolResult(**result)

    def apply(self,update,cursor):
        c=cursor;context=update.context;owner=context.owner
        current=self.memory.read(owner,c);revision=current['revision'];old=self.snapshot(owner,c)['facts']
        changes=[f for f in update.facts if f.get('new')]
        if changes:
            current=self.memory.save(owner,update.prepared_document,context.revision,c,manual=False)
            revision=current['revision']
        for fact in update.facts:
            if fact.get('new'):
                c.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,false,false,%s)').format(self.table('account_memory_facts')),
                    (owner,fact['id'],fact['layer'],fact['category'],fact['content'],fact['basis'],revision))
            for source in fact.get('evidence',[]):
                key=digest([source['kind'],source['reference'],source['quote'],source['candidate_content']])
                c.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING').format(self.table('account_memory_fact_evidence')),
                    (owner,fact['id'],key,source['kind'],source['reference'],source['quote'],source['candidate_content'],_utc_now()))
        suggestions=[]
        for proposal in update.suggestions:
            c.execute(sql.SQL("SELECT id,proposal FROM {} WHERE user_id=%s AND status='pending' AND memory_revision=%s AND policy_epoch=%s FOR UPDATE").format(self.table('memory_reflection_suggestions')),
                (owner,revision,context.policy_epoch))
            signature=lambda p:digest({k:v for k,v in p.items() if k!='sources'})
            match=next(((sid,old) for sid,old in c.fetchall() if signature(old)==signature(proposal)),None)
            if match:
                sid,old=match;suggestions.append(sid)
                sources={digest(source):source for source in old['sources']+proposal['sources']}
                combined={**old,'sources':list(sources.values())}
                c.execute(sql.SQL('UPDATE {} SET proposal=%s WHERE user_id=%s AND id=%s').format(self.table('memory_reflection_suggestions')),(Json(combined),owner,sid))
                continue
            sid=uuid.uuid4().hex;suggestions.append(sid)
            c.execute(sql.SQL("INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,%s,'pending',%s)").format(self.table('memory_reflection_suggestions')),
                (sid,context.execution_id if context.scope=='background' else None,owner,context.input_snapshot['conversation_id'],revision,context.policy_epoch,Json(proposal),_utc_now()))
        merged=sum(1 for f in update.facts if not f.get('new') and f.get('evidence'))
        status='saved' if changes else 'suggested' if suggestions else 'noop'
        message={'saved':'已保存长期记忆。','suggested':'更新需要确认，请在长期记忆设置查看建议。','noop':'该记忆已存在，本次未新增。'}[status]
        result=ToolResult(status,revision,len(changes),merged,tuple(suggestions),message)
        self.save_receipt(context,update.argument_digest,result,c)
        return result

    def save_receipt(self,context,argument_digest,result,c):
        c.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s,%s,%s,%s)').format(self.table('account_memory_tool_receipts')),
            (context.owner,context.execution_id,argument_digest,Json(asdict(result)),_utc_now()))

    def save_decision(self,context,decision,cursor):
        c=cursor;tid=context.execution_id
        c.execute(sql.SQL('SELECT question,status FROM {} WHERE id=%s AND owner_id=%s FOR UPDATE').format(self.table('generation_tasks')),(tid,context.owner))
        row=c.fetchone()
        if not row or row[1]!='running' or row[0]!=context.input_snapshot['question']:raise PermissionError('Invalid decision identity')
        encoded=asdict(decision);key=digest(row[0])
        c.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s,%s,%s,%s,NULL,%s) ON CONFLICT(task_id) DO NOTHING').format(self.table('generation_turn_decisions')),
            (tid,context.owner,VERSION,Json(encoded),key,_utc_now()))
        frozen=self.read_decision(context.owner,tid,c)
        if frozen!=decision:raise MemoryConflict('Decision already frozen')

    def read_decision(self,owner,task_id,cursor=None):
        if cursor is None:
            with self.memory.cursor() as c:return self.read_decision(owner,task_id,c)
        cursor.execute(sql.SQL('SELECT d.schema_version,d.decision,d.input_digest,g.question FROM {} d JOIN {} g ON g.id=d.task_id WHERE d.task_id=%s AND d.user_id=%s AND g.owner_id=%s').format(self.table('generation_turn_decisions'),self.table('generation_tasks')),(task_id,owner,owner))
        row=cursor.fetchone()
        if row:
            if row[0]!=VERSION or row[2]!=digest(row[3]):raise MemoryConflict('Frozen decision mismatch')
            return decision_from_dict(row[1])

    def bind_ordinal(self,owner,tid,cid,ordinal,c):
        c.execute(sql.SQL('UPDATE {} SET user_ordinal=%s WHERE task_id=%s AND user_id=%s').format(self.table('generation_turn_decisions')),(ordinal,tid,owner))
        c.execute(sql.SQL("UPDATE {} SET source_kind='message',reference=%s WHERE user_id=%s AND source_kind='current_input' AND reference=%s").format(self.table('account_memory_fact_evidence')),(f'{cid}:{ordinal}',owner,tid))

    def detach_conversation(self,owner,cid,c):
        c.execute(sql.SQL('UPDATE {} SET reference=NULL WHERE user_id=%s AND source_kind=\'message\' AND reference LIKE %s').format(self.table('account_memory_fact_evidence')),(owner,cid+':%'))
