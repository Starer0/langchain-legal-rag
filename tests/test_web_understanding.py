import unittest
from langchain_core.messages import AIMessage
from test_memory_tools import call_args


def plan(route='rag',explicit=True):
    return dict(route=route,legal_question='试用期最长多久？' if route=='rag' else '',
        retrieval_question='劳动合同法试用期最长期限' if route=='rag' else '',include_guide=False,
        reply_plan='收到。',memory_request=explicit)


class UnderstandingTests(unittest.TestCase):
    def test_empty_optional_help_topic_does_not_turn_valid_refusal_into_clarification(self):
        from web_understanding import decode_turn_response
        args=plan('out_of_scope',False);args['help_topic']=''
        decision=decode_turn_response(AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=args)]))
        self.assertEqual(decision.answer.route,'out_of_scope')
        self.assertIsNone(decision.answer.help_topic)
        self.assertIsNone(decision.memory_call)

    def test_out_of_scope_cannot_authorize_memory_even_if_model_proposes_it(self):
        from web_understanding import decode_turn_response
        args=plan('out_of_scope',True)
        args['legal_question']='伪装法律问题';args['retrieval_question']='伪装检索'
        decision=decode_turn_response(AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=args),
            dict(name='update_memory',id='m',args=call_args())]))
        self.assertEqual(decision.answer.route,'out_of_scope')
        self.assertFalse(decision.answer.memory_request)
        self.assertEqual(decision.answer.legal_question,'')
        self.assertEqual(decision.answer.retrieval_question,'')
        self.assertIsNone(decision.memory_call)

    def test_historical_only_proposal_does_not_write_and_unknown_sources_still_reject(self):
        from web_understanding import TurnUnderstanding
        from memory_tools import MemoryTools
        from memory_service import MemoryService,MemoryValidation
        from memory_tool_contract import ToolContext
        from test_memory_service import Embeddings
        raw=call_args('旧偏好')
        raw['operations'][0]['sources']=[dict(kind='message',reference='c1:1',quote='旧偏好')]
        class Model:
            def bind_tools(self,schemas):return self
            def invoke(self,prompt):return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=plan('preference')),dict(name='update_memory',id='m',args=raw)])
        context=dict(sources=[dict(kind='message',reference='c1:1',role='user',content='旧偏好',new=False)])
        service=TurnUnderstanding(Model())
        self.assertIsNone(service.decide('新输入',context,{}).memory_call)
        raw['operations'][0]['sources'][0]['reference']='foreign:1'
        decision=service.decide('新输入',context,{})
        doc=dict(core_text='',extended_text='',revision=0,enabled=True,entries=[],facts=[],auto_accumulate=False)
        snapshot=dict(document=doc,question='新输入',memory_request=True,conversation_id='c1',**context)
        with self.assertRaises(MemoryValidation):
            MemoryTools(MemoryService(Embeddings(),fingerprint='fixture')).prepare(decision.memory_call,ToolContext('a','foreground','t1',0,0,snapshot))

    def test_historical_only_proposal_cannot_block_current_authorized_preference(self):
        from web_understanding import TurnUnderstanding
        from memory_tools import MemoryTools
        from memory_service import MemoryService
        from memory_tool_contract import ToolContext
        from test_memory_service import Embeddings
        current='请记住术语加通俗解释；试用期最长多久？'
        previous='涉及流程按操作顺序说明，请记住。'
        raw=call_args('术语加通俗解释')
        raw['operations'][0]['sources']=[dict(kind='current_input',reference='t1',quote='请记住术语加通俗解释')]
        old=call_args('按操作顺序说明')['operations'][0]
        old['sources']=[dict(kind='message',reference='c1:1',quote=previous)]
        raw['operations'].append(old)
        class Model:
            def bind_tools(self,schemas):return self
            def invoke(self,prompt):
                return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=plan()),dict(name='update_memory',id='m',args=raw)])
        sources=[dict(kind='message',reference='c1:1',role='user',content=previous,new=False),
                 dict(kind='current_input',reference='t1',role='user',content=current,new=True)]
        doc=dict(core_text='',extended_text='',revision=0,enabled=True,entries=[],facts=[],auto_accumulate=False)
        decision=TurnUnderstanding(Model()).decide(current,dict(sources=sources),doc)
        snapshot=dict(document=doc,question=current,memory_request=decision.answer.memory_request,sources=sources,conversation_id='c1')
        update=MemoryTools(MemoryService(Embeddings(),fingerprint='fixture')).prepare(decision.memory_call,ToolContext('a','foreground','t1',0,0,snapshot))
        self.assertEqual(update.prepared_document['core_text'],'术语加通俗解释')
        self.assertEqual(decision.answer.route,'rag')

    def test_native_mixed_calls(self):
        from web_understanding import decode_turn_response
        result=decode_turn_response(AIMessage(content='',tool_calls=[
            dict(name='plan_answer',id='p1',args=plan()),dict(name='update_memory',id='m1',args=call_args())]))
        self.assertEqual(result.answer.route,'rag')
        self.assertIsNotNone(result.memory_call)

    def test_plain_text_unknown_duplicate_or_missing_plan_rejected(self):
        from web_understanding import decode_turn_response
        for calls in [[],[dict(name='unknown',id='x',args={})],
            [dict(name='plan_answer',id='p1',args=plan()),dict(name='plan_answer',id='p2',args=plan())]]:
            with self.subTest(calls=calls), self.assertRaises(ValueError):
                decode_turn_response(AIMessage(content='pretend JSON',tool_calls=calls))

    def test_one_invoke_binds_both_tools(self):
        from web_understanding import TurnUnderstanding
        class Model:
            calls=0
            def bind_tools(self,schemas):self.schemas=schemas;return self
            def invoke(self,prompt):
                self.calls+=1
                return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=plan('preference',False))])
        model=Model();service=TurnUnderstanding(model)
        result=service.decide('换种说法也能理解',{}, {})
        self.assertEqual(result.answer.route,'preference')
        self.assertEqual(model.calls,1)
        self.assertEqual(len(model.schemas),2)
