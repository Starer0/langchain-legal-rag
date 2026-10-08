import unittest
from dataclasses import asdict
from langchain_core.messages import AIMessage
from test_web_understanding import plan
import test_web_rag as f


class SemanticGraphTests(unittest.TestCase):
    def test_memory_read_reply_omits_speaker_label_in_stream_and_completion(self):
        from memory_tool_contract import TurnDecision,AnswerPlan
        turn=self.turn('preference')
        body='根据已保存的学习主题，你近期关注劳动合同解除条件；这是推测。'
        frozen=TurnDecision(AnswerPlan('preference','','',False,'回复用户：'+body,False),None)
        events=list(turn.stream('我最近关注什么？',[],decision_cached=frozen))
        self.assertEqual(events[-1]['data']['answer'],body)
        streamed=''.join(e['data']['text'] for e in events if e['event']=='delta')
        self.assertEqual(streamed,body)
        self.assertEqual(self.model.calls,0)

    def test_reply_preserves_quoted_speaker_label_inside_body(self):
        from memory_tool_contract import TurnDecision,AnswerPlan
        body='你的原话是“回复用户：请先说条件”。'
        frozen=TurnDecision(AnswerPlan('preference','','',False,body,False),None)
        events=list(self.turn('preference').stream('查看偏好',[],decision_cached=frozen))
        self.assertEqual(events[-1]['data']['answer'],body)

    def test_invalid_native_response_logs_only_diagnostic_metadata(self):
        import json,logging
        from rag_logging import RagRequestTrace
        turn=self.turn('preference')
        def response(prompt):
            args=plan('preference',False);args['route']='SECRET_RAW_TOOL_VALUE'
            return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=args)])
        self.model.invoke=response
        trace=RagRequestTrace(request_id='r',user_id='u',conversation_id='c',allowed_knowledge_bases={'A'},logger=logging.getLogger('test.scope'))
        with self.assertLogs('test.scope',level='INFO') as captured:
            done=list(turn.stream('SECRET_USER_INPUT',[],trace=trace,allowed_knowledge_bases={'A'}))[-1]['data']
        rows=[json.loads(r.getMessage()) for r in captured.records]
        errors=[r for r in rows if r['event']=='understanding_rejected']
        self.assertEqual(len(errors),1)
        self.assertEqual(errors[0]['reason'],'schema_validation')
        self.assertEqual(errors[0]['validation_fields'],['route'])
        self.assertNotIn('SECRET_',json.dumps(rows))
        self.assertEqual(done['decision']['answer']['route'],'clarify')

    def test_mixed_preference_and_outside_request_refuses_without_displaying_free_plan(self):
        turn=self.turn('preference')
        def response(prompt):
            args=plan('preference',False);args.update(out_of_scope_request=True,reply_plan='推荐游戏的内部提纲')
            return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=args)])
        self.model.invoke=response
        events=list(turn.stream('以后先提醒保留证据；推荐几款游戏',[]))
        self.assertIn('超出',events[-1]['data']['answer'])
        self.assertNotIn('内部提纲',events[-1]['data']['answer'])
        self.assertNotIn('retrieve',[e['data'].get('stage') for e in events])

    def test_out_of_scope_refuses_without_retrieval_or_planning_text(self):
        turn=self.turn('out_of_scope')
        def response(prompt):
            args=plan('out_of_scope',False);args['reply_plan']='直接给出n8n设置步骤：Webhook选POST。'
            return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=args)])
        self.model.invoke=response
        events=list(turn.stream('n8n怎么接收POST？',[]))
        done=events[-1]['data']
        self.assertEqual(done['decision']['answer']['route'],'out_of_scope')
        self.assertIn('超出',done['answer'])
        self.assertNotIn('Webhook',done['answer'])
        self.assertNotIn('retrieve',[e['data'].get('stage') for e in events])

    def test_general_product_help_uses_server_text_even_if_model_leaks_technical_plan(self):
        turn=self.turn('general')
        def response(prompt):
            args=plan('general',False);args.update(help_topic='memory',reply_plan='直接给出n8n结果回传的操作说明：Webhook。')
            return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=args)])
        self.model.invoke=response
        done=list(turn.stream('在哪里查看长期记忆？',[]))[-1]['data']
        self.assertIn('长期记忆',done['answer'])
        self.assertNotIn('Webhook',done['answer'])
        self.assertNotIn('直接给出',done['answer'])

    def turn(self,route='rag'):
        from web_understanding import TurnUnderstanding
        from web_langgraph import LangGraphStreamingRagTurn
        class Model:
            calls=0
            def bind_tools(self,tools):return self
            def invoke(self,prompt):
                self.calls+=1
                return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=plan(route,False))])
        self.model=Model();self.rewriter=f.FakeRewriter('旧改写')
        return LangGraphStreamingRagTurn(understanding=TurnUnderstanding(self.model),rewriter=self.rewriter,
            retriever=f.FakeRetriever([]),reranker=f.FakeReranker([]),prompt=f.FakePrompt(),
            model=f.FakeStreamingModel(['法律回答']),history_turns=4)

    def test_rag_reuses_understand_and_emits_decision(self):
        turn=self.turn()
        events=list(turn.stream('以后先给结论；试用期最长多久',[]))
        completion=events[-1]['data']
        self.assertEqual(self.model.calls,1)
        self.assertEqual(completion['retrieval_question'],'劳动合同法试用期最长期限')
        self.assertEqual(completion['decision']['answer']['legal_question'],'试用期最长多久？')
        self.assertNotIn('rewrite',[e['data'].get('stage') for e in events])

    def test_unknown_preference_has_no_legal_retrieval(self):
        events=list(self.turn('preference').stream('换个表达顺序更容易懂',[]))
        self.assertEqual(events[-1]['data']['sources'],[])
        self.assertNotIn('retrieve',[e['data'].get('stage') for e in events])

    def test_recovery_reuses_frozen_decision_without_model(self):
        from memory_tool_contract import TurnDecision,AnswerPlan
        turn=self.turn()
        frozen=TurnDecision(AnswerPlan('preference','','',False,'冻结的回复',False),None)
        events=list(turn.stream('以后照这样',[],decision_cached=frozen))
        self.assertEqual(self.model.calls,0)
        self.assertEqual(events[-1]['data']['answer'],'冻结的回复')

    def test_invalid_native_response_clarifies_without_retry_or_retrieval(self):
        turn=self.turn();self.model.invoke=lambda prompt:AIMessage(content='pretend JSON')
        events=list(turn.stream('请记住某偏好',[]))
        self.assertEqual(events[-1]['data']['decision']['answer']['route'],'clarify')
        self.assertIn('本次未保存',events[-1]['data']['answer'])
        self.assertNotIn('retrieve',[e['data'].get('stage') for e in events])

    def test_explicit_request_without_update_never_claims_saved(self):
        turn=self.turn('preference')
        def missing(prompt):
            args=plan('preference',True);args['reply_plan']='已保存偏好。'
            return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args=args)])
        self.model.invoke=missing
        completion=list(turn.stream('请记住先给结论',[]))[-1]['data']
        self.assertIn('未保存',completion['answer'])
        self.assertNotIn('已保存',completion['answer'])
        self.assertEqual(completion['memory']['status'],'blocked')
