"""Transactional durable reflection with account, source and revision fences."""
import uuid
from datetime import datetime,timezone
from contextlib import contextmanager
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json,RealDictCursor
from memory_service import MemoryConflict, MemoryValidation, split_entries, vector_valid, command_candidate, estimate_tokens
from memory_reflection import ReflectionService
from conversation_context import SourceMessage
from prepare_memory_reflection import MIGRATION
from web_storage import _utc_now


class ReflectionStore:
    def __init__(self,*,memory,schema='public',**settings):
        self.memory,self.schema,self.settings=memory,schema,{'connect_timeout':5,**settings}
        with self.cursor() as c:
            c.execute(sql.SQL('SELECT 1 FROM {} WHERE name=%s').format(self.table('schema_migrations')),(MIGRATION,))
            if c.fetchone() is None:raise RuntimeError('Run prepare_memory_reflection.py --apply first')

    def table(self,name):return sql.Identifier(self.schema,name)
    @contextmanager
    def cursor(self):
        conn=psycopg2.connect(**self.settings)
        try:
            with conn,conn.cursor() as c:yield c
        finally:conn.close()

    def user(self,c,owner):
        c.execute(sql.SQL('SELECT id FROM {} WHERE id=%s AND is_active FOR UPDATE').format(self.table('users')),(owner,))
        if c.fetchone() is None:raise PermissionError('账号不可用。')

    def owned(self,c,owner,cid):
        c.execute(sql.SQL('SELECT id FROM {} WHERE id=%s AND user_id=%s FOR SHARE').format(self.table('conversations')),(cid,owner))
        if c.fetchone() is None:raise PermissionError('对话不存在。')

    def policy(self,owner,cursor=None):
        if cursor is None:
            with self.cursor() as c:return self.policy(owner,c)
        cursor.execute(sql.SQL('SELECT enabled,epoch FROM {} WHERE user_id=%s').format(self.table('memory_reflection_policies')),(owner,))
        row=cursor.fetchone();return {'auto_accumulate':row[0],'epoch':row[1]} if row else {'auto_accumulate':False,'epoch':0}

    def invalidate(self,owner,cursor):
        c=cursor
        c.execute(sql.SQL('INSERT INTO {} (user_id,enabled,epoch) VALUES (%s,false,1) ON CONFLICT(user_id) DO UPDATE SET epoch={}.epoch+1')
            .format(self.table('memory_reflection_policies'),sql.Identifier('memory_reflection_policies')),(owner,))
        c.execute(sql.SQL("UPDATE {} SET status='obsolete',epoch=epoch+1 WHERE user_id=%s AND status IN ('pending','running','failed')").format(self.table('memory_reflection_jobs')),(owner,))
        c.execute(sql.SQL("UPDATE {} SET status='obsolete',proposal=%s WHERE user_id=%s AND status='pending'").format(self.table('memory_reflection_suggestions')),(Json({}),owner))
        c.execute(sql.SQL('INSERT INTO {} (conversation_id,user_id,invalidated_through) SELECT c.id,c.user_id,COALESCE(MAX(m.ordinal),0) FROM {} c LEFT JOIN {} m ON m.conversation_id=c.id WHERE c.user_id=%s GROUP BY c.id,c.user_id '
            'ON CONFLICT(conversation_id) DO UPDATE SET invalidated_through=GREATEST({}.invalidated_through,EXCLUDED.invalidated_through)')
            .format(self.table('memory_reflection_progress'),self.table('conversations'),self.table('conversation_messages'),sql.Identifier('memory_reflection_progress')),(owner,))

    def set_policy(self,owner,enabled,expected_epoch,cursor=None):
        if cursor is None:
            with self.cursor() as c:return self.set_policy(owner,enabled,expected_epoch,c)
        self.user(cursor,owner)
        if self.policy(owner,cursor)['epoch']!=expected_epoch:raise MemoryConflict('自动记忆设置已变化，请重新加载。')
        self.invalidate(owner,cursor)
        cursor.execute(sql.SQL('UPDATE {} SET enabled=%s WHERE user_id=%s').format(self.table('memory_reflection_policies')),(enabled,owner))
        return self.policy(owner,cursor)

    def sources(self,c,cid,start,end):
        c.execute(sql.SQL('SELECT ordinal,role,content FROM {} WHERE conversation_id=%s AND ordinal BETWEEN %s AND %s ORDER BY ordinal').format(self.table('conversation_messages')),(cid,start,end))
        rows=c.fetchall()
        if getattr(self.memory,'facts',None) is None:
            return [dict(zip(('ordinal','role','content'),row)) for row in rows if row[1]=='user' and not command_candidate(row[2])]
        c.execute(sql.SQL('SELECT d.user_ordinal,d.decision,r.result FROM {} d JOIN {} g ON g.id=d.task_id LEFT JOIN {} r ON r.user_id=d.user_id AND r.execution_id=d.task_id WHERE g.conversation_id=%s').format(self.table('generation_turn_decisions'),self.table('generation_tasks'),self.table('account_memory_tool_receipts')),(cid,))
        handled={ordinal for ordinal,decision,result in c.fetchall() if
            decision.get('answer',{}).get('route')=='out_of_scope' or
            (decision.get('memory_call') and result and result.get('status') in ('saved','noop','suggested'))}
        return [dict(zip(('ordinal','role','content'),row)) for row in rows if row[1]=='user' and row[0] not in handled]

    @staticmethod
    def decode(c,row):
        if row is None:return None
        result=dict(zip([column.name for column in c.description],row))
        result['due_at']=result['due_at'].timestamp()
        return result

    def enqueue(self,cursor,owner,cid,through_ordinal,reason,now):
        c=cursor;self.owned(c,owner,cid)
        policy=self.policy(owner,c);memory=self.memory.read(owner,c)
        if not policy['auto_accumulate'] or not memory['enabled']:return None
        c.execute(sql.SQL('SELECT processed_through,invalidated_through FROM {} WHERE conversation_id=%s').format(self.table('memory_reflection_progress')),(cid,))
        progress=c.fetchone() or (0,0)
        c.execute(sql.SQL("SELECT COALESCE(MAX(end_ordinal),0) FROM {} WHERE conversation_id=%s AND status IN ('running','failed')").format(self.table('memory_reflection_jobs')),(cid,))
        start=max(*progress,c.fetchone()[0])+1
        if through_ordinal<start:return None
        sources=self.sources(c,cid,start,through_ordinal)
        if not sources:return None
        c.execute(sql.SQL("SELECT * FROM {} WHERE conversation_id=%s AND status='pending' ORDER BY due_at LIMIT 1 FOR UPDATE").format(self.table('memory_reflection_jobs')),(cid,))
        pending=self.decode(c,c.fetchone())
        due=datetime.fromtimestamp(now+(0 if reason=='compression' else 120),timezone.utc)
        if pending:
            end=max(through_ordinal,pending['end_ordinal']);start=pending['start_ordinal']
            sources=self.sources(c,cid,start,end)
            c.execute(sql.SQL('UPDATE {} SET end_ordinal=%s,due_at=%s,input=%s,reason=%s,updated_at=%s WHERE id=%s RETURNING *').format(self.table('memory_reflection_jobs')),
                (end,due,Json({'messages':sources,**({'tool_schema':'model-memory-tools-v1'} if getattr(self.memory,'facts',None) else {})}),reason,_utc_now(),pending['id']))
        else:
            c.execute(sql.SQL('INSERT INTO {} (id,user_id,conversation_id,start_ordinal,end_ordinal,due_at,status,epoch,policy_epoch,memory_revision,attempts,input,reason,updated_at) '
                "VALUES (%s,%s,%s,%s,%s,%s,'pending',0,%s,%s,0,%s,%s,%s) RETURNING *").format(self.table('memory_reflection_jobs')),
                (uuid.uuid4().hex,owner,cid,start,through_ordinal,due,policy['epoch'],memory['revision'],Json({'messages':sources,**({'tool_schema':'model-memory-tools-v1'} if getattr(self.memory,'facts',None) else {})}),reason,_utc_now()))
        return self.decode(c,c.fetchone())

    def defer(self,cursor,owner,cid,now):
        cursor.execute(sql.SQL("UPDATE {} SET due_at=%s WHERE user_id=%s AND conversation_id=%s AND status='pending' AND reason='idle'").format(self.table('memory_reflection_jobs')),
                       (datetime.fromtimestamp(now+120,timezone.utc),owner,cid))

    def claim(self,now):
        with self.cursor() as c:
            c.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s),hashtext(%s))',(self.schema,'memory_reflection_claim'))
            if not c.fetchone()[0]:return None
            eligible=sql.SQL("SELECT j.* FROM {} j JOIN {} u ON u.id=j.user_id JOIN {} p ON p.user_id=j.user_id WHERE j.status='pending' AND j.due_at<=%s AND u.is_active AND p.enabled AND p.epoch=j.policy_epoch "
                'AND NOT EXISTS(SELECT 1 FROM {} g WHERE g.status=\'running\') '
                "AND NOT EXISTS(SELECT 1 FROM {} r WHERE r.status='running') ")
            eligible=eligible.format(self.table('memory_reflection_jobs'),self.table('users'),self.table('memory_reflection_policies'),self.table('generation_tasks'),self.table('memory_reflection_jobs'))
            due=datetime.fromtimestamp(now,timezone.utc)
            c.execute(eligible+sql.SQL('ORDER BY j.due_at,j.id LIMIT 1'),(due,))
            job=self.decode(c,c.fetchone())
            if not job:return None
            # Match task acceptance: conversation first, then job. Recheck
            # eligibility after waiting for any newly accepted question.
            self.owned(c,job['user_id'],job['conversation_id'])
            c.execute(eligible+sql.SQL('AND j.id=%s FOR UPDATE OF j SKIP LOCKED'),(due,job['id']))
            job=self.decode(c,c.fetchone())
            if not job:return None
            memory=self.memory.read(job['user_id'],c)
            if not memory['enabled']:return None
            messages=job['input']['messages'];batch=[];size=0;end=job['end_ordinal']
            for m in messages:
                count=estimate_tokens(m['content'])
                if batch and size+count>4000:
                    end=m['ordinal']-1;break
                batch.append(m);size+=count
            context=[];size=0
            c.execute(sql.SQL('SELECT invalidated_through FROM {} WHERE conversation_id=%s').format(self.table('memory_reflection_progress')),(job['conversation_id'],))
            invalidated=(c.fetchone() or (0,))[0]
            for m in reversed(self.sources(c,job['conversation_id'],invalidated+1,job['start_ordinal']-1)):
                count=estimate_tokens(m['content'])
                if size+count>1000:break
                context.insert(0,m);size+=count
            c.execute(sql.SQL("UPDATE {} SET status='running',epoch=epoch+1,end_ordinal=%s,input=%s,memory_revision=%s,updated_at=%s WHERE id=%s RETURNING *").format(self.table('memory_reflection_jobs')),
                (end,Json({**job['input'],'messages':batch,'context':context}),memory['revision'],_utc_now(),job['id']))
            claimed=self.decode(c,c.fetchone())
            if end<job['end_ordinal']:
                self.enqueue(c,job['user_id'],job['conversation_id'],job['end_ordinal'],'compression',now)
            return claimed

    def jobs(self,owner,cid=None):
        with self.cursor() as c:
            query=sql.SQL('SELECT id,conversation_id,status,result,updated_at FROM {} WHERE user_id=%s').format(self.table('memory_reflection_jobs'))
            params=[owner]
            if cid is not None:query+=sql.SQL(' AND conversation_id=%s');params.append(cid)
            c.execute(query+sql.SQL(' ORDER BY updated_at DESC'),params)
            return [dict(zip(('id','conversation_id','status','result','updated_at'),row)) for row in c.fetchall()]

    def recover(self,now):
        with self.cursor() as c:
            if getattr(self.memory,'facts',None) is not None:
                c.execute(sql.SQL("UPDATE {} SET status='obsolete',epoch=epoch+1 WHERE status IN ('pending','running','failed') AND COALESCE(input->>'tool_schema','')<>%s RETURNING user_id,conversation_id,end_ordinal").format(self.table('memory_reflection_jobs')),('model-memory-tools-v1',))
                old=c.fetchall()
                for owner,cid,end in old:self.enqueue(c,owner,cid,end,'idle',now)
            c.execute(sql.SQL("UPDATE {} SET status=CASE WHEN attempts<1 THEN 'pending' ELSE 'failed' END,attempts=attempts+1,epoch=epoch+1,due_at=%s,result=CASE WHEN attempts<1 THEN result ELSE %s END WHERE status='running'")
                .format(self.table('memory_reflection_jobs')),(datetime.fromtimestamp(now,timezone.utc),Json({'error':'整理中断次数超过上限，可主动重试。'})))

    def finish(self,job_id,epoch,result,cursor=None):
        if cursor is None:
            with self.cursor() as c:return self.finish(job_id,epoch,result,c)
        cursor.execute(sql.SQL("UPDATE {} SET status='failed',result=%s,updated_at=%s WHERE id=%s AND epoch=%s AND status='running'").format(self.table('memory_reflection_jobs')),(Json(result),_utc_now(),job_id,epoch))
        return cursor.rowcount==1

    def retry(self,owner,job_id,now):
        with self.cursor() as c:
            self.user(c,owner)
            c.execute(sql.SQL('SELECT * FROM {} WHERE id=%s AND user_id=%s FOR UPDATE').format(self.table('memory_reflection_jobs')),(job_id,owner))
            job=self.decode(c,c.fetchone())
            if not job:raise PermissionError('任务不存在。')
            memory=self.memory.read(owner,c);policy=self.policy(owner,c)
            if job['status']!='failed' or not policy['auto_accumulate'] or not memory['enabled'] or job['policy_epoch']!=policy['epoch'] or job['memory_revision']!=memory['revision']:raise MemoryConflict('任务已失效，请通过新消息或设置更新记忆。')
            c.execute(sql.SQL("UPDATE {} SET status='pending',due_at=%s,epoch=epoch+1,attempts=0,result=NULL WHERE id=%s").format(self.table('memory_reflection_jobs')),(datetime.fromtimestamp(now,timezone.utc),job_id))
            return {'status':'pending'}

    def apply(self,job_id,epoch,validated,prepared):
        with self.cursor() as c:
            c.execute(sql.SQL('SELECT user_id,conversation_id FROM {} WHERE id=%s').format(self.table('memory_reflection_jobs')),(job_id,))
            identity=c.fetchone()
            if not identity:raise PermissionError('任务或来源对话已删除。')
            owner,cid=identity;self.owned(c,owner,cid);self.user(c,owner)
            c.execute(sql.SQL('SELECT * FROM {} WHERE id=%s FOR UPDATE').format(self.table('memory_reflection_jobs')),(job_id,))
            job=self.decode(c,c.fetchone())
            if job['status']=='completed':return job['result']
            current=self.memory.read(owner,c);policy=self.policy(owner,c)
            if job['status']!='running' or job['epoch']!=epoch or policy['epoch']!=job['policy_epoch'] or not policy['auto_accumulate'] or not current['enabled'] or current['revision']!=job['memory_revision']:raise MemoryConflict('整理任务已失效，不会覆盖新记忆。')
            source=self.sources(c,cid,job['start_ordinal'],job['end_ordinal'])
            if source!=job['input']['messages']:raise MemoryConflict('整理来源已变化。')
            raw=validated['additions']+validated['suggestions']
            checked=ReflectionService(None,None).validate({'candidates':raw},[SourceMessage(**m) for m in source],current)
            if checked['additions']!=validated['additions'] or checked['suggestions']!=validated['suggestions']:raise MemoryValidation('自动记忆提议无法核对。')
            if checked['additions']:
                core,extended=current['core_text'],current['extended_text']
                for candidate in checked['additions']:
                    if candidate['layer']=='core':core+='\n\n'+candidate['text'] if core else candidate['text']
                    else:extended+='\n\n'+candidate['text'] if extended else candidate['text']
                if prepared is None or prepared['core_text']!=core or prepared['extended_text']!=extended or prepared['enabled']!=current['enabled']:raise MemoryValidation('自动记忆合并不符合提议。')
                entries=prepared.get('entries',[])
                if [i['text'] for i in entries]!=split_entries(extended) or [i['ordinal'] for i in entries]!=list(range(len(entries))) or len({i['id'] for i in entries})!=len(entries) or any(not vector_valid(i['embedding']) for i in entries):raise MemoryValidation('自动记忆索引无法核对。')
                current=self.memory.save(owner,prepared,current['revision'],cursor=c,manual=False)
            for proposal in checked['suggestions']:
                c.execute(sql.SQL("INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,%s,'pending',%s)").format(self.table('memory_reflection_suggestions')),
                    (uuid.uuid4().hex,job_id,owner,cid,current['revision'],policy['epoch'],Json(proposal),_utc_now()))
            c.execute(sql.SQL('INSERT INTO {} (conversation_id,user_id,processed_through) VALUES (%s,%s,%s) ON CONFLICT(conversation_id) DO UPDATE SET processed_through=GREATEST({}.processed_through,EXCLUDED.processed_through)')
                .format(self.table('memory_reflection_progress'),sql.Identifier('memory_reflection_progress')),(cid,owner,job['end_ordinal']))
            result={'added':len(checked['additions']),'suggestions':len(checked['suggestions']),'skipped':len(validated['skipped']),'revision':current['revision']}
            c.execute(sql.SQL("UPDATE {} SET status='completed',result=%s,updated_at=%s WHERE id=%s").format(self.table('memory_reflection_jobs')),(Json(result),_utc_now(),job_id))
            return result

    def apply_tool(self,job_id,epoch,update):
        from dataclasses import asdict
        with self.cursor() as c:
            c.execute(sql.SQL('SELECT user_id,conversation_id FROM {} WHERE id=%s').format(self.table('memory_reflection_jobs')),(job_id,))
            identity=c.fetchone()
            if not identity:raise PermissionError('任务来源已删除。')
            owner,cid=identity;self.owned(c,owner,cid);self.user(c,owner)
            c.execute(sql.SQL('SELECT * FROM {} WHERE id=%s FOR UPDATE').format(self.table('memory_reflection_jobs')),(job_id,))
            job=self.decode(c,c.fetchone())
            if job['status']=='completed':return job['result']
            policy=self.policy(owner,c);current=self.memory.read(owner,c)
            if (job['status']!='running' or job['epoch']!=epoch or update.context.execution_id!=job_id or
                update.context.owner!=owner or update.context.revision!=job['memory_revision'] or
                update.context.policy_epoch!=job['policy_epoch'] or policy['epoch']!=job['policy_epoch'] or
                not policy['auto_accumulate'] or not current['enabled'] or current['revision']!=job['memory_revision']):
                raise MemoryConflict('整理任务已失效。')
            source=self.sources(c,cid,job['start_ordinal'],job['end_ordinal'])
            if source!=job['input']['messages']:raise MemoryConflict('整理来源已变化。')
            expected={f'{cid}:{m["ordinal"]}':m for m in source}
            for op in update.call.operations:
                if not any(s.reference in expected and s.quote in expected[s.reference]['content'] for s in op.sources):
                    raise MemoryValidation('每项提议需要本批新来源。')
            result=asdict(self.memory.tools.commit(update,c))
            # JSON round trip matches persisted receipts, including tuple lists.
            import json
            result=json.loads(json.dumps(result));result['suggestions']=len(result['suggestion_ids'])
            c.execute(sql.SQL('INSERT INTO {} (conversation_id,user_id,processed_through) VALUES (%s,%s,%s) ON CONFLICT(conversation_id) DO UPDATE SET processed_through=GREATEST({}.processed_through,EXCLUDED.processed_through)').format(self.table('memory_reflection_progress'),sql.Identifier('memory_reflection_progress')),(cid,owner,job['end_ordinal']))
            c.execute(sql.SQL("UPDATE {} SET status='completed',result=%s,updated_at=%s WHERE id=%s").format(self.table('memory_reflection_jobs')),(Json(result),_utc_now(),job_id))
            return result

    def suggestions(self,owner):
        with self.cursor() as c:
            current=self.memory.read(owner,c);policy=self.policy(owner,c)
            c.execute(sql.SQL("SELECT s.id,s.proposal,s.memory_revision,s.policy_epoch FROM {} s JOIN {} v ON v.id=s.conversation_id WHERE s.user_id=%s AND v.user_id=%s AND s.status='pending' AND s.memory_revision=%s AND s.policy_epoch=%s ORDER BY s.created_at,s.id")
                .format(self.table('memory_reflection_suggestions'),self.table('conversations')),(owner,owner,current['revision'],policy['epoch']))
            return [dict(zip(('id','proposal','memory_revision','policy_epoch'),row)) for row in c.fetchall()]

    def suggestion(self,owner,sid,cursor):
        cursor.execute(sql.SQL('SELECT * FROM {} WHERE id=%s AND user_id=%s').format(self.table('memory_reflection_suggestions')),(sid,owner))
        row=cursor.fetchone()
        if not row:raise PermissionError('建议不存在。')
        return dict(zip([column.name for column in cursor.description],row))

    def ignore(self,owner,sid):
        with self.cursor() as c:
            self.user(c,owner);self.suggestion(owner,sid,c)
            c.execute(sql.SQL("UPDATE {} SET status='ignored',proposal=%s WHERE id=%s AND user_id=%s AND status='pending'").format(self.table('memory_reflection_suggestions')),(Json({}),sid,owner))
            return {'status':'ignored'}

    def accept(self,owner,sid,expected_revision,expected_epoch,service):
        with self.cursor() as c:
            suggestion=self.suggestion(owner,sid,c)
            current=self.memory.read(owner,c)
            policy=self.policy(owner,c)
        if (suggestion['status']!='pending' or current['revision']!=expected_revision or
            suggestion['memory_revision']!=expected_revision or policy['epoch']!=expected_epoch or suggestion['policy_epoch']!=expected_epoch):
            raise MemoryConflict('建议已失效，请重新查看记忆。')
        proposal=suggestion['proposal']
        if 'action' in proposal:
            return self.accept_tool(owner,suggestion,current,expected_revision,expected_epoch,service)
        if proposal['relation']=='capacity':raise MemoryValidation('记忆容量已满，请先在设置中精简内容。')
        field='core_text' if proposal['layer']=='core' else 'extended_text'
        target=proposal['target']
        if current['revision']!=expected_revision or suggestion['memory_revision']!=expected_revision or current[field].count(target)!=1:
            raise MemoryConflict('原记忆已变化，请重新查看建议。')
        text=current[field].replace(target,proposal['text'],1)
        prepared=service.prepare(current,text if field=='core_text' else current['core_text'],text if field=='extended_text' else current['extended_text'],current['enabled'])
        with self.cursor() as c:
            self.owned(c,owner,suggestion['conversation_id']);self.user(c,owner)
            fresh=self.suggestion(owner,sid,c);policy=self.policy(owner,c);now=self.memory.read(owner,c)
            if fresh['status']!='pending' or now['revision']!=expected_revision or fresh['memory_revision']!=expected_revision or policy['epoch']!=expected_epoch or fresh['policy_epoch']!=expected_epoch or not policy['auto_accumulate'] or not now['enabled']:
                raise MemoryConflict('建议已失效，不会覆盖新记忆。')
            c.execute(sql.SQL('SELECT start_ordinal,end_ordinal,input FROM {} WHERE id=%s AND user_id=%s').format(self.table('memory_reflection_jobs')),(fresh['job_id'],owner))
            job=c.fetchone()
            if not job:raise MemoryConflict('建议来源已删除。')
            source=self.sources(c,fresh['conversation_id'],job[0],job[1])
            if source!=job[2]['messages']:raise MemoryConflict('建议来源已变化。')
            checked=ReflectionService(None,None).validate({'candidates':[proposal]},[SourceMessage(**m) for m in source],now)
            if checked['suggestions']!=[proposal]:raise MemoryValidation('建议内容无法核对。')
            saved=self.memory.save(owner,prepared,expected_revision,cursor=c)
            c.execute(sql.SQL("UPDATE {} SET status='accepted' WHERE id=%s AND user_id=%s").format(self.table('memory_reflection_suggestions')),(sid,owner))
            return saved

    def accept_tool(self,owner,suggestion,current,expected_revision,expected_epoch,service):
        proposal=suggestion['proposal'];action=proposal['action'];layer=proposal['layer']
        facts=getattr(self.memory,'facts',None)
        if facts is None:raise MemoryValidation('新版记忆服务尚未启用。')
        with self.cursor() as c:active={f['id']:f for f in facts.snapshot(owner,c)['facts']}
        targets=[active.get(fid) for fid in proposal['target_ids']]
        if any(f is None or f['layer']!=layer for f in targets):raise MemoryConflict('建议目标已变化。')
        if action in ('request_delete','propose_change','merge') and not targets:raise MemoryValidation('建议缺少目标。')
        fields={'core':'core_text','extended':'extended_text'};field=fields[layer]
        parts=split_entries(current[field]);remove={f['content'] for f in targets}
        if action=='request_clear':parts=[]
        elif action=='request_delete':parts=[p for p in parts if p not in remove]
        elif proposal['relation']=='conflict' or action=='propose_change':parts=[p for p in parts if p not in remove]+[proposal['content']]
        else:parts+=[] if proposal['content'] in parts else [proposal['content']]
        new_text='\n\n'.join(parts)
        prepared=service.prepare(current,new_text if layer=='core' else current['core_text'],new_text if layer=='extended' else current['extended_text'],current['enabled'])
        with self.cursor() as c:
            self.owned(c,owner,suggestion['conversation_id']);self.user(c,owner)
            fresh=self.suggestion(owner,suggestion['id'],c);policy=self.policy(owner,c);now=self.memory.read(owner,c)
            if (fresh['status']!='pending' or now['revision']!=expected_revision or fresh['memory_revision']!=expected_revision or
                policy['epoch']!=expected_epoch or fresh['policy_epoch']!=expected_epoch):raise MemoryConflict('建议已失效。')
            for ref in proposal['sources']:
                if ref['kind']=='current_input':
                    c.execute(sql.SQL('SELECT question FROM {} WHERE id=%s AND owner_id=%s').format(self.table('generation_tasks')),(ref['reference'],owner))
                    row=c.fetchone()
                elif ref['kind']=='message':
                    cid,ordinal=ref['reference'].rsplit(':',1)
                    c.execute(sql.SQL("SELECT m.content FROM {} m JOIN {} v ON v.id=m.conversation_id WHERE m.conversation_id=%s AND m.ordinal=%s AND m.role='user' AND v.user_id=%s").format(self.table('conversation_messages'),self.table('conversations')),(cid,int(ordinal),owner))
                    row=c.fetchone()
                else:raise MemoryValidation('无效建议来源。')
                if not row or ref['quote'] not in row[0]:raise MemoryConflict('建议来源已删除。')
            saved=self.memory.save(owner,prepared,expected_revision,cursor=c)
            # Confirmation protects the text; it does not turn an inference into
            # a user declaration or replace its evidence with the settings text.
            if action not in ('request_delete','request_clear'):
                from memory_fact_storage import digest
                accepted=next((f for f in facts.snapshot(owner,c)['facts']
                    if f['layer']==layer and f['content']==proposal['content'] and f['id'] not in active),None)
                if accepted:
                    fid=accepted['id']
                    c.execute(sql.SQL('UPDATE {} SET category=%s,basis=%s WHERE user_id=%s AND id=%s').format(self.table('account_memory_facts')),
                        (proposal['category'],proposal['basis'],owner,fid))
                    c.execute(sql.SQL('DELETE FROM {} WHERE user_id=%s AND fact_id=%s').format(self.table('account_memory_fact_evidence')),(owner,fid))
                    for ref in proposal['sources']:
                        kind,reference=ref['kind'],ref['reference']
                        if kind=='current_input':
                            c.execute(sql.SQL('SELECT g.conversation_id,d.user_ordinal FROM {} g JOIN {} d ON d.task_id=g.id WHERE g.id=%s AND g.owner_id=%s').format(self.table('generation_tasks'),self.table('generation_turn_decisions')),(reference,owner))
                            locator=c.fetchone()
                            if locator and locator[1] is not None:kind,reference='message',f'{locator[0]}:{locator[1]}'
                        key=digest([kind,reference,ref['quote'],proposal['content']])
                        c.execute(sql.SQL('INSERT INTO {} VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING').format(self.table('account_memory_fact_evidence')),
                            (owner,fid,key,kind,reference,ref['quote'],proposal['content'],_utc_now()))
            c.execute(sql.SQL("UPDATE {} SET status='accepted',proposal=%s WHERE id=%s AND user_id=%s").format(self.table('memory_reflection_suggestions')),(Json({}),suggestion['id'],owner))
            return saved
