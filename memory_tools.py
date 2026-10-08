"""One source-checked update tool for foreground and background callers."""
import uuid
import json
from dataclasses import asdict
from psycopg2 import sql
from memory_service import MemoryValidation,MemoryConflict,validate_texts,split_entries,vector_valid
from memory_fact_storage import digest
from memory_tool_contract import PreparedUpdate,ToolResult,parse_memory_call


class MemoryTools:
    def __init__(self,service,facts=None):
        self.service,self.store=service,facts

    def _plan(self,call,context):
        # Re-parse even dataclasses; callers cannot bypass the provider bounds.
        call=parse_memory_call(json.loads(json.dumps({'operations':asdict(call)['operations']})),call.provider_call_id)
        snapshot=context.input_snapshot;doc=snapshot['document']
        explicit=context.scope=='foreground' and snapshot.get('memory_request') is True
        if not explicit and (not doc.get('enabled',True) or not doc.get('auto_accumulate',False)):
            raise MemoryValidation('自动积累未开启，本次未保存记忆。')
        sources={(s['kind'],s['reference']):s for s in snapshot.get('sources',[])}
        existing={f['id']:f for f in doc.get('facts',[])}
        facts=[];suggestions=[];core=doc['core_text'];extended=doc['extended_text']
        known={(f['layer'],f['content']) for f in existing.values()}
        for op in call.operations:
            if not op.sources:raise MemoryValidation('记忆提议缺少用户来源。')
            evidence=[];turns=set();new_source=False
            for ref in op.sources:
                source=sources.get((ref.kind,ref.reference))
                if not source or source.get('role')!='user' or not ref.quote.strip() or ref.quote not in source['content']:
                    raise MemoryValidation('记忆来源不属于本次授权用户原文。')
                turns.add((ref.kind,ref.reference))
                new_source |= source.get('new',True)
                evidence.append({**asdict(ref),'candidate_content':op.content})
            if not new_source:raise MemoryValidation('前文不能单独授权重新保存旧记忆。')
            targets=[]
            for target in op.target_ids:
                if target not in existing or existing[target]['layer']!=op.layer:raise MemoryValidation('记忆目标不存在或层级不一致。')
                targets.append(existing[target])
            if op.relation in ('duplicate','complement','conflict') and not targets:raise MemoryValidation('更新关系缺少目标。')
            if op.basis=='inferred' and (op.category!='learning_focus' or op.layer!='extended' or len(turns)<2):
                raise MemoryValidation('学习主题推测至少需要两个用户轮次，且只能生成扩展建议。')
            proposal=asdict(op)
            # All destructive/replacing operations are review requests only.
            review=(op.basis=='inferred' or op.action in ('propose_change','request_delete','request_clear') or op.relation=='conflict')
            if review:
                suggestions.append(proposal);continue
            if op.relation in ('duplicate','complement'):
                for target in targets:facts.append({**target,'evidence':evidence,'new':False})
                if op.relation=='duplicate':continue
                content=op.delta.strip()
                if not content:raise MemoryValidation('补充关系缺少新增含义。')
            else:content=op.content.strip()
            if not content:raise MemoryValidation('记忆内容不能为空。')
            if (op.layer,content) in known:
                matched=next((f for f in list(existing.values())+facts if f['layer']==op.layer and f['content']==content),None)
                if matched:facts.append({**matched,'evidence':evidence,'new':False})
                continue
            known.add((op.layer,content))
            facts.append(dict(id=uuid.uuid4().hex,layer=op.layer,category=op.category,basis=op.basis,
                content=content,protected=False,new=True,evidence=evidence))
            if op.layer=='core':core='\n\n'.join(filter(None,(core,content)))
            else:extended='\n\n'.join(filter(None,(extended,content)))
        return facts,suggestions,core,extended

    def prepare(self,call,context):
        facts,suggestions,core,extended=self._plan(call,context)
        doc=context.input_snapshot['document']
        # Evidence-only changes reuse vectors; core never causes embeddings.
        prepared=self.service.prepare(doc,core,extended,doc['enabled'])
        return PreparedUpdate(context,call,tuple(facts),tuple(suggestions),prepared,
            digest({'operations':asdict(call)['operations']}))

    def commit(self,update,cursor):
        if self.store is None:raise RuntimeError('Memory fact storage is unavailable')
        c=cursor;context=update.context;owner=context.owner
        c.execute(sql.SQL('SELECT id FROM {} WHERE id=%s AND is_active FOR UPDATE').format(self.store.table('users')),(owner,))
        if c.fetchone() is None:raise PermissionError('账号不可用。')
        replay=self.store.receipt(owner,context.execution_id,c)
        if replay:
            if replay[0]!=update.argument_digest:raise MemoryConflict('同一执行编号不能提交不同参数。')
            return replay[1]
        current=self.store.memory.read(owner,c)
        reflection=getattr(self.store.memory,'reflection',None)
        policy=reflection.policy(owner,c) if reflection else {'epoch':0,'auto_accumulate':False}
        status=None
        if current['revision']!=context.revision or policy['epoch']!=context.policy_epoch:status='conflict'
        explicit=context.scope=='foreground' and context.input_snapshot.get('memory_request') is True
        if not explicit and (not current['enabled'] or not policy['auto_accumulate']):status='blocked'
        if status:
            result=ToolResult(status,current['revision'],0,0,(),'记忆版本或开关已变化，本次未保存；请查看设置。')
            self.store.save_receipt(context,update.argument_digest,result,c);return result
        # Check every reference against durable inputs, never solely against a model snapshot.
        snap=context.input_snapshot
        if any(snap['document'].get(key)!=current.get(key) for key in ('core_text','extended_text','enabled','revision','entries')):
            raise MemoryValidation('记忆文档与当前持久版本不一致。')
        fields=('id','layer','category','content','basis','protected','revision')
        normalize=lambda facts:sorted([{k:f.get(k) for k in fields} for f in facts],key=lambda f:f['id'])
        if normalize(snap['document'].get('facts',[]))!=normalize(self.store.snapshot(owner,c)['facts']):
            raise MemoryValidation('记忆目标与当前账号不一致。')
        if context.scope=='foreground':
            c.execute(sql.SQL('SELECT question,conversation_id FROM {} WHERE id=%s AND owner_id=%s').format(self.store.table('generation_tasks')),(context.execution_id,owner))
            row=c.fetchone()
            if row!=(snap['question'],snap['conversation_id']):raise MemoryValidation('任务来源无法核对。')
            captured=self.store.memory.task_input(context.execution_id,owner,c)
            if captured is None or digest(captured)!=digest(snap['document']):raise MemoryValidation('记忆快照无法核对。')
            frozen=self.store.read_decision(owner,context.execution_id,c)
            if frozen is None or frozen.memory_call!=update.call or frozen.answer.memory_request!=snap.get('memory_request'):
                raise MemoryValidation('提交与持久决策不一致。')
        else:
            c.execute(sql.SQL('SELECT input,conversation_id FROM {} WHERE id=%s AND user_id=%s').format(self.store.table('memory_reflection_jobs')),(context.execution_id,owner))
            row=c.fetchone()
            if not row or row[1]!=snap['conversation_id']:raise MemoryValidation('后台来源无法核对。')
        for source in snap.get('sources',[]):
            if source['kind']=='current_input':
                if context.scope!='foreground' or source['reference']!=context.execution_id or source['content']!=snap['question']:raise MemoryValidation('无效当前输入来源。')
            elif source['kind']=='message':
                cid,ordinal=source['reference'].rsplit(':',1)
                if cid!=snap['conversation_id']:raise MemoryValidation('来源不属于本次对话。')
                c.execute(sql.SQL('SELECT m.role,m.content FROM {} m JOIN {} v ON v.id=m.conversation_id WHERE m.conversation_id=%s AND m.ordinal=%s AND v.user_id=%s').format(self.store.table('conversation_messages'),self.store.table('conversations')),(cid,int(ordinal),owner))
                if c.fetchone()!=('user',source['content']):raise MemoryValidation('消息来源已失效。')
            else:raise MemoryValidation('模型不能使用手动文档作为新增事实来源。')
        if digest({'operations':asdict(update.call)['operations']})!=update.argument_digest:raise MemoryValidation('工具参数摘要变化。')
        expected_facts,expected_suggestions,_,_=self._plan(update.call,context)
        def comparable(facts):
            return [{**f,'id':''} if f.get('new') else f for f in facts]
        if digest(comparable(update.facts))!=digest(comparable(expected_facts)) or digest(update.suggestions)!=digest(expected_suggestions):
            raise MemoryValidation('工具候选与调用参数不一致。')
        validate_texts(update.prepared_document['core_text'],update.prepared_document['extended_text'])
        # Deterministic structural recheck, without another embedding/model call.
        doc=snap['document'];new=[f for f in update.facts if f.get('new')]
        for layer,key in [('core','core_text'),('extended','extended_text')]:
            expected='\n\n'.join(filter(None,[doc[key]]+[f['content'] for f in new if f['layer']==layer]))
            if expected!=update.prepared_document[key]:raise MemoryValidation('工具渲染内容不一致。')
        entries=update.prepared_document['entries']
        if ([e['text'] for e in entries]!=split_entries(update.prepared_document['extended_text']) or
            [e.get('ordinal') for e in entries]!=list(range(len(entries))) or
            len({e.get('id') for e in entries})!=len(entries) or
            any(not vector_valid(e.get('embedding')) for e in entries)):
            raise MemoryValidation('扩展索引不一致。')
        return self.store.apply(update,c)
