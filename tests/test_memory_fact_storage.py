import unittest
from dataclasses import asdict
import test_memory_storage as storage_fixture


class FactStorageTests(unittest.TestCase):
    setUpClass=storage_fixture.MemoryStorageTests.setUpClass
    save=storage_fixture.MemoryStorageTests.save
    task=storage_fixture.MemoryStorageTests.task
    def setUp(self):
        storage_fixture.MemoryStorageTests.setUp(self)
        from prepare_memory_reflection import prepare_reflection
        prepare_reflection(self.db,apply=True,schema=self.schema)
        from reflection_storage import ReflectionStore
        self.reflection=ReflectionStore(memory=self.memories,schema=self.schema,**self.fixture.settings)
        self.memories.reflection=self.reflection
        from prepare_memory_tools import prepare_memory_tools
        self.migrate=prepare_memory_tools
        from memory_fact_storage import MemoryFactStore
        self.assertFalse(self.migrate(self.db,schema=self.schema)['ready'])
        self.migrate(self.db,apply=True,schema=self.schema)
        self.facts=MemoryFactStore(self.memories)
        self.memories.facts=self.facts

    def test_legacy_registration_preserves_text(self):
        saved=self.save('中文\n\n结合例子','TCP学习')
        with self.memories.cursor() as c:
            snapshot=self.facts.snapshot(self.owner,c)
        self.assertEqual([f['content'] for f in snapshot['facts']],['中文','结合例子','TCP学习'])
        self.assertTrue(all(f['protected'] for f in snapshot['facts']))
        self.assertEqual(self.memories.read(self.owner)['core_text'],saved['core_text'])

    def test_idempotent_scoped_apply_and_receipt(self):
        from test_memory_tools import call_args
        from memory_tool_contract import ToolContext,parse_memory_call
        from memory_tools import MemoryTools
        task,doc=self.task()
        raw=call_args('中文');raw['operations'][0]['sources']=[dict(kind='current_input',reference=task['id'],quote='中文')]
        context=ToolContext(self.owner,'foreground',task['id'],0,0,
            dict(document=doc,question=task['question'],memory_request=True,conversation_id=task['conversation_id'],sources=[
                dict(kind='current_input',reference=task['id'],role='user',content=task['question'])]))
        tools=MemoryTools(self.service,self.facts)
        call=parse_memory_call(raw)
        from memory_tool_contract import TurnDecision,AnswerPlan
        with self.memories.cursor() as c:self.facts.save_decision(context,TurnDecision(AnswerPlan('preference','','',False,'',True),call),c)
        update=tools.prepare(call,context)
        with self.memories.cursor() as c:result=tools.commit(update,c)
        with self.memories.cursor() as c:again=tools.commit(update,c)
        self.assertEqual(result,again)
        self.assertEqual(result.status,'saved')
        self.assertEqual(self.memories.read(self.owner)['revision'],1)
        with self.memories.cursor() as c:self.assertEqual(self.facts.snapshot(self.other,c)['facts'],[])
        from dataclasses import replace
        from memory_service import MemoryConflict
        with self.assertRaises(MemoryConflict):
            with self.memories.cursor() as c:tools.commit(replace(update,argument_digest='different'),c)
        self.memories.tools=tools
        decision=self.facts.read_decision(self.owner,task['id'])
        self.assertTrue(self.tasks.complete(task['id'],'等待保存',[],decision=asdict(decision),pending_memory=asdict(update)))
        self.tasks.delete_conversation(self.owner,task['conversation_id'])
        with self.memories.cursor() as c:
            source=self.facts.snapshot(self.owner,c)['facts'][0]['sources'][0]
        self.assertIsNone(source['reference'])
        self.assertEqual(source['quote'],'中文')

    def test_manual_delete_redacts_evidence(self):
        self.save('中文','')
        self.save('','')
        with self.memories.cursor() as c:
            self.assertEqual(self.facts.snapshot(self.owner,c)['facts'],[])

    def test_confirmed_inferred_suggestion_keeps_basis_and_original_evidence(self):
        import uuid
        from psycopg2 import sql
        from psycopg2.extras import Json
        from web_storage import _utc_now
        cid=self.fixture.store.create_conversation(self.owner).id
        self.reflection.set_policy(self.owner,True,0)
        quotes=['n8n怎样接收POST？','n8n怎样返回结果？']
        for quote in quotes:self.fixture.store.save_complete_conversation_turn(self.owner,cid,quote,'答')
        sid=uuid.uuid4().hex
        proposal=dict(action='remember',layer='extended',category='learning_focus',basis='inferred',
            content='近期学习n8n',target_ids=[],relation='new',delta='',sources=[
                dict(kind='message',reference=f'{cid}:{ordinal}',quote=quote) for ordinal,quote in zip([1,3],quotes)])
        with self.memories.cursor() as c:
            c.execute(sql.SQL("INSERT INTO {} VALUES (%s,NULL,%s,%s,0,1,%s,'pending',%s)").format(self.memories.table('memory_reflection_suggestions')),
                (sid,self.owner,cid,Json(proposal),_utc_now()))
        with self.assertRaises(PermissionError):self.reflection.accept(self.other,sid,0,1,self.service)
        self.reflection.accept(self.owner,sid,0,1,self.service)
        with self.memories.cursor() as c:fact=self.facts.snapshot(self.owner,c)['facts'][0]
        self.assertEqual(fact['basis'],'inferred')
        self.assertEqual(fact['category'],'learning_focus')
        self.assertCountEqual([s['quote'] for s in fact['sources']],quotes)
        self.assertTrue(fact['protected'])
        self.save('','')
        with self.memories.cursor() as c:
            c.execute(sql.SQL('SELECT count(*) FROM {} WHERE user_id=%s').format(self.memories.table('account_memory_fact_evidence')),(self.owner,))
            self.assertEqual(c.fetchone()[0],0)

    def test_identical_pending_candidates_merge_sources_and_ignore_redacts(self):
        from dataclasses import replace
        from memory_tool_contract import ToolContext,PreparedUpdate,MemoryToolCall
        from memory_fact_storage import digest
        cid=self.fixture.store.create_conversation(self.owner).id
        self.reflection.set_policy(self.owner,True,0)
        proposal=dict(action='remember',layer='extended',category='learning_focus',basis='inferred',content='n8n学习',
            relation='new',delta='',target_ids=[],sources=[dict(kind='message',reference=f'{cid}:1',quote='n8n')])
        ctx=ToolContext(self.owner,'foreground','isolated-a',0,1,dict(conversation_id=cid))
        update=PreparedUpdate(ctx,MemoryToolCall('',()),(),(proposal,),{},digest(proposal))
        with self.memories.cursor() as c:first=self.facts.apply(update,c)
        second_proposal={**proposal,'sources':[dict(kind='message',reference=f'{cid}:3',quote='n8n续问')]}
        update=replace(update,context=replace(ctx,execution_id='isolated-b'),suggestions=(second_proposal,),argument_digest=digest(second_proposal))
        with self.memories.cursor() as c:second=self.facts.apply(update,c)
        self.assertEqual(first.suggestion_ids,second.suggestion_ids)
        items=self.reflection.suggestions(self.owner)
        self.assertEqual(len(items),1)
        self.assertEqual(len(items[0]['proposal']['sources']),2)
        self.reflection.ignore(self.owner,items[0]['id'])
        with self.memories.cursor() as c:self.assertEqual(self.reflection.suggestion(self.owner,items[0]['id'],c)['proposal'],{})

    def test_mixed_completion_commits_once_and_conflict_keeps_answer(self):
        from test_memory_tools import call_args
        from memory_tool_contract import ToolContext,parse_memory_call,TurnDecision,AnswerPlan
        from memory_tools import MemoryTools
        tools=MemoryTools(self.service,self.facts);self.memories.tools=tools
        task,doc=self.task()
        raw=call_args('中文');raw['operations'][0]['sources']=[dict(kind='current_input',reference=task['id'],quote='中文')]
        call=parse_memory_call(raw,'m1')
        decision=TurnDecision(AnswerPlan('rag','法律问题','法律问题',False,'',True),call)
        context=ToolContext(self.owner,'foreground',task['id'],0,0,dict(document=doc,question=task['question'],memory_request=True,
            conversation_id=task['conversation_id'],sources=[dict(kind='current_input',reference=task['id'],role='user',content=task['question'])]))
        with self.memories.cursor() as c:self.facts.save_decision(context,decision,c)
        update=tools.prepare(call,context)
        self.save('手动新内容','')
        self.assertTrue(self.tasks.complete(task['id'],'成功法律回答',[],decision=asdict(decision),pending_memory=asdict(update)))
        observed=self.tasks.get(self.owner,task['conversation_id'],task['id'])
        self.assertEqual(observed['answer'],'成功法律回答')
        self.assertEqual(observed['memory']['status'],'conflict')
        self.assertEqual(self.memories.read(self.owner)['core_text'],'手动新内容')

    def test_preference_receipt_keeps_independent_scope_refusal(self):
        from test_memory_tools import call_args
        from memory_tool_contract import ToolContext,parse_memory_call,TurnDecision,AnswerPlan
        from memory_tools import MemoryTools
        self.memories.tools=MemoryTools(self.service,self.facts)
        task,doc=self.task()
        raw=call_args('中文');raw['operations'][0]['sources']=[dict(kind='current_input',reference=task['id'],quote='中文')]
        call=parse_memory_call(raw)
        decision=TurnDecision(AnswerPlan('preference','','',False,'推荐游戏的内部提纲',True,out_of_scope_request=True),call)
        ctx=ToolContext(self.owner,'foreground',task['id'],0,0,dict(document=doc,question=task['question'],memory_request=True,
            conversation_id=task['conversation_id'],sources=[dict(kind='current_input',reference=task['id'],role='user',content=task['question'])]))
        with self.memories.cursor() as c:self.facts.save_decision(ctx,decision,c)
        update=self.memories.tools.prepare(call,ctx)
        self.assertTrue(self.tasks.complete(task['id'],'正在处理记忆',[],decision=asdict(decision),pending_memory=asdict(update)))
        result=self.tasks.get(self.owner,task['conversation_id'],task['id'])
        self.assertEqual(result['memory']['status'],'saved')
        self.assertIn('已保存长期记忆',result['answer'])
        self.assertIn('超出',result['answer'])
        self.assertNotIn('推荐游戏的内部提纲',result['answer'])

    def test_legacy_frozen_decision_without_optional_reply_fields_can_complete(self):
        from psycopg2 import sql
        from psycopg2.extras import Json
        from test_memory_tools import call_args
        from memory_tool_contract import ToolContext,parse_memory_call,TurnDecision,AnswerPlan
        from memory_tools import MemoryTools
        self.memories.tools=MemoryTools(self.service,self.facts)
        task,doc=self.task();raw=call_args('中文')
        raw['operations'][0]['sources']=[dict(kind='current_input',reference=task['id'],quote='中文')]
        call=parse_memory_call(raw)
        decision=TurnDecision(AnswerPlan('preference','','',False,'',True),call)
        legacy=asdict(decision)
        del legacy['answer']['help_topic'];del legacy['answer']['out_of_scope_request']
        ctx=ToolContext(self.owner,'foreground',task['id'],0,0,dict(document=doc,question=task['question'],memory_request=True,
            conversation_id=task['conversation_id'],sources=[dict(kind='current_input',reference=task['id'],role='user',content=task['question'])]))
        with self.memories.cursor() as c:
            self.facts.save_decision(ctx,decision,c)
            c.execute(sql.SQL('UPDATE {} SET decision=%s WHERE task_id=%s').format(self.facts.table('generation_turn_decisions')),(Json(legacy),task['id']))
        update=self.memories.tools.prepare(call,ctx)
        self.assertTrue(self.tasks.complete(task['id'],'处理中',[],decision=legacy,pending_memory=asdict(update)))
        self.assertEqual(self.tasks.get(self.owner,task['conversation_id'],task['id'])['memory']['status'],'saved')



    def test_tool_failure_rolls_back_memory_but_preserves_mixed_answer(self):
        from unittest.mock import patch
        from memory_tools import MemoryTools
        from memory_tool_contract import ToolContext,parse_memory_call,TurnDecision,AnswerPlan
        from test_memory_tools import call_args
        self.memories.tools=MemoryTools(self.service,self.facts)
        task,doc=self.task();raw=call_args('中文')
        raw['operations'][0]['sources']=[dict(kind='current_input',reference=task['id'],quote='中文')]
        call=parse_memory_call(raw)
        decision=TurnDecision(AnswerPlan('rag','法律问题','法律问题',False,'',True),call)
        ctx=ToolContext(self.owner,'foreground',task['id'],0,0,dict(document=doc,question=task['question'],
            conversation_id=task['conversation_id'],memory_request=True,sources=[dict(kind='current_input',reference=task['id'],role='user',content=task['question'])]))
        with self.memories.cursor() as c:self.facts.save_decision(ctx,decision,c)
        update=self.memories.tools.prepare(call,ctx)
        original=self.facts.apply
        def fail_after_write(update,c):original(update,c);raise RuntimeError('after write')
        with patch.object(self.facts,'apply',side_effect=fail_after_write):
            self.assertTrue(self.tasks.complete(task['id'],'法律回答保留',[],decision=asdict(decision),pending_memory=asdict(update)))
        self.assertEqual(self.memories.read(self.owner)['revision'],0)
        self.assertEqual(self.facts.snapshot(self.owner)['facts'],[])
        observed=self.tasks.get(self.owner,task['conversation_id'],task['id'])
        self.assertEqual(observed['answer'],'法律回答保留')
        self.assertEqual(observed['memory']['status'],'blocked')

    def test_background_review_and_receipt_are_atomic(self):
        from memory_tools import MemoryTools
        from memory_tool_contract import ToolContext,parse_memory_call
        from test_memory_tools import call_args
        self.memories.tools=MemoryTools(self.service,self.facts)
        cid=self.fixture.store.create_conversation(self.owner).id
        self.reflection.set_policy(self.owner,True,0)
        self.fixture.store.save_complete_conversation_turn(self.owner,cid,'解释TCP时联系网络排障例子','收到')
        with self.memories.cursor() as c:self.reflection.enqueue(c,self.owner,cid,2,'compression',0)
        job=self.reflection.claim(0)
        snapshot=self.memories.snapshot(self.owner)
        raw=call_args();raw['operations'][0]['sources'][0].update(kind='message',reference=cid+':1')
        ctx=ToolContext(self.owner,'background',job['id'],0,1,dict(document=snapshot,question='',conversation_id=cid,
            sources=[dict(kind='message',reference=cid+':1',role='user',content='解释TCP时联系网络排障例子',new=True)]))
        update=self.memories.tools.prepare(parse_memory_call(raw),ctx)
        first=self.reflection.apply_tool(job['id'],job['epoch'],update)
        self.assertEqual(first['status'],'saved')
        self.assertEqual(self.reflection.apply_tool(job['id'],job['epoch'],update),first)
        self.assertEqual(self.memories.read(self.owner)['revision'],1)


class LegacyMigrationTests(unittest.TestCase):
    setUpClass=storage_fixture.MemoryStorageTests.setUpClass
    save=storage_fixture.MemoryStorageTests.save
    def setUp(self):storage_fixture.MemoryStorageTests.setUp(self)

    def test_existing_documents_seed_legacy_and_survive_schema_backup_restore(self):
        import subprocess
        from psycopg2 import sql
        from prepare_memory_reflection import prepare_reflection
        from prepare_memory_tools import prepare_memory_tools
        from memory_fact_storage import MemoryFactStore
        saved=self.save('先给结论\n\n结合例子','TCP学习')
        prepare_reflection(self.db,apply=True,schema=self.schema)
        prepare_memory_tools(self.db,apply=True,schema=self.schema)
        self.memories.facts=MemoryFactStore(self.memories)
        facts=self.memories.facts.snapshot(self.owner)['facts']
        self.assertTrue(all(f['protected'] and f['sources'][0]['kind']=='legacy_document' for f in facts))
        self.assertEqual(self.memories.read(self.owner)['core_text'],saved['core_text'])
        self.assertEqual(prepare_memory_tools(self.db,apply=True,schema=self.schema)['action'],'already_ready')
        import os,sys,json
        env={**os.environ,'WEB_POSTGRES_SCHEMA':self.schema}
        checked=subprocess.run([sys.executable,'prepare_memory_tools.py'],env=env,capture_output=True,text=True)
        self.assertEqual(checked.returncode,0,checked.stderr)
        self.assertTrue(json.loads(checked.stdout)['ready'])
        # This schema is generated by the existing fixture; never drop public.
        self.assertRegex(self.schema,r'^login_test_[a-f0-9]{32}$')
        dump=subprocess.run(['docker','exec','legal-rag-postgres','pg_dump','-U','legal_rag','-d','legal_rag','--schema='+self.schema],capture_output=True,check=True).stdout
        with self.db,self.db.cursor() as c:c.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        subprocess.run(['docker','exec','-i','legal-rag-postgres','psql','-U','legal_rag','-d','legal_rag','-v','ON_ERROR_STOP=1'],input=dump,capture_output=True,check=True)
        self.assertEqual(self.memories.facts.snapshot(self.owner)['facts'],facts)
        self.assertEqual(self.memories.read(self.owner)['core_text'],saved['core_text'])
