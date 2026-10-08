import unittest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from conversation_context import ContextService, SourceMessage
from web_langgraph import LangGraphStreamingRagTurn
import test_web_rag as fixtures


class PreferenceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.StreamingRagTurnTests();self.fixture.setUp()
        f=self.fixture
        self.turn=LangGraphStreamingRagTurn(rewriter=f.rewriter,retriever=f.retriever,
            reranker=f.reranker,prompt=f.prompt,model=f.model,history_turns=4)

    def test_natural_preference_skips_all_legal_and_model_operations(self):
        events=list(self.turn.stream('我喜欢先看结论，再看具体例子',[]))
        done=events[-1]['data']
        self.assertEqual(done['sources'],[])
        self.assertIn('收到',done['answer'])
        self.assertNotIn('已保存',done['answer'])
        self.assertEqual(self.fixture.rewriter.calls,[])
        self.assertEqual(self.fixture.retriever.states,[])
        self.assertEqual(self.fixture.model.prompts,[])

    def test_preference_feedback_respects_frozen_automation_switches(self):
        cases=[({'auto_accumulate':False,'enabled':True},'当前未开启自动积累，不会自动保存这条偏好'),
               ({'auto_accumulate':True,'enabled':True},'自动积累已开启'),
               ({'auto_accumulate':True,'enabled':False},'后台积累已暂停')]
        for snapshot,notice in cases:
            with self.subTest(snapshot=snapshot):
                done=list(self.turn.stream('我喜欢简短回答',[],memory_input=snapshot))[-1]['data']
                self.assertIn(notice,done['answer'])
                self.assertNotIn('已保存',done['answer'])
                self.assertEqual(done['sources'],[])

    def test_mixed_legal_question_and_ambiguous_text_still_use_rag(self):
        for question in ['我喜欢先看结论，再看具体例子。试用期最长多久？',
                         '我喜欢劳动法，请解释第二十条', '同事说他喜欢先看结论',
                         '请翻译“我喜欢先看结论”', '我喜欢先看结论，不支付工资可以吗？',
                         '请解释结论', '解释具体例子']:
            with self.subTest(question=question):
                self.fixture.rewriter.calls.clear()
                list(self.turn.stream(question,[]))
                self.assertEqual(self.fixture.rewriter.calls[0][0],question)

    def test_explicit_style_instruction_mixed_with_legal_question_keeps_rag(self):
        from types import SimpleNamespace
        service=SimpleNamespace(select=lambda q,s:{'revision':0,'core':'','extended':'','selected_ids':[],'status':'empty','warning':''})
        for question in ['以后回答请简短一点，试用期最长多久？',
                         '以后回答请简短一点，试用期可以延长吗。',
                         '以后回答请简短一点，解释一下劳动合同']:
            with self.subTest(question=question):
                self.fixture.rewriter.calls.clear()
                list(self.turn.stream(question,[],memory_service=service,memory_input={}))
                self.assertEqual(self.fixture.rewriter.calls[0][0],question)
                self.assertEqual(ContextService(None).eligible([SourceMessage(1,'user',question)]),[SourceMessage(1,'user',question)])

    def test_elliptical_question_after_style_instruction_keeps_rag(self):
        from types import SimpleNamespace
        question='以后回答请简短一点，那最长多久？'
        service=SimpleNamespace(select=lambda q,s:{'revision':0,'core':'','extended':'','selected_ids':[],'status':'empty','warning':''})
        list(self.turn.stream(question,[],memory_service=service,memory_input={}))
        self.assertEqual(self.fixture.rewriter.calls[0][0],question)

    def test_adversarial_overlapping_style_words_finish_promptly(self):
        import subprocess,sys
        result=subprocess.run([sys.executable,'-c',
            "from web_intent import is_style_preference; assert not is_style_preference('我喜欢'+'具体例子'*100+'X')"],
            timeout=3,capture_output=True)
        self.assertEqual(result.returncode,0)

    def test_preference_does_not_replace_legal_history_for_followup(self):
        history=[HumanMessage('试用期工资规定'),AIMessage('旧法律回答'),
                 HumanMessage('我喜欢先看结论，再看具体例子'),AIMessage('收到偏好')]
        list(self.turn.stream('那最长多久呢？',history))
        contents=[m.content for m in self.fixture.rewriter.calls[-1][1]]
        self.assertIn('试用期工资规定',contents)
        self.assertNotIn('我喜欢先看结论，再看具体例子',contents)
        self.assertNotIn('收到偏好',contents)

    def test_summary_sources_exclude_preference_but_keep_mixed_legal_question(self):
        items=[SourceMessage(1,'user','我喜欢先看结论，再看具体例子'),
               SourceMessage(3,'user','我喜欢先看结论，试用期最长多久？')]
        self.assertEqual(ContextService(None).eligible(items),[items[1]])

    def test_completed_preference_checkpoint_replays_without_rag(self):
        self.turn=LangGraphStreamingRagTurn(checkpointer=InMemorySaver(),rewriter=self.fixture.rewriter,
            retriever=self.fixture.retriever,reranker=self.fixture.reranker,prompt=self.fixture.prompt,
            model=self.fixture.model,history_turns=4)
        question='我喜欢简短回答'
        first=list(self.turn.stream(question,[],task_id='preference'))
        resumed=list(self.turn.stream(question,[],task_id='preference',resume=True))
        self.assertEqual(first[-1],resumed[-1])
        self.assertEqual(self.fixture.retriever.states,[])
