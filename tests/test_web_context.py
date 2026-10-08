import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from conversation_context import ContextService, ContextLimits, Summary, SourceMessage, ContextValidation, PromptBudget, ContextTooLong
from test_conversation_context import SummaryModel, item
import test_web_rag as f
from web_langgraph import LangGraphStreamingRagTurn


class WebContextTests(unittest.TestCase):
    def setUp(self):
        self.rewriter=f.FakeRewriter('检索问题')
        self.retriever=f.FakeRetriever([Document(page_content='依据',metadata={})])
        self.prompt=f.FakePrompt(); self.model=f.FakeStreamingModel(['回答'])
        self.components=dict(rewriter=self.rewriter,retriever=self.retriever,reranker=f.FakeReranker(self.retriever.documents),
                             prompt=self.prompt,model=self.model,history_turns=4)
        self.turn=LangGraphStreamingRagTurn(**self.components)
        self.service=ContextService(SummaryModel([item(1,'合同期限两年')]),
            ContextLimits(trigger_tokens=100,summary_tokens=40,retained_tokens=75,recent_turns=2),token_counter=len)
        self.frozen={'summary':asdict(Summary()),'messages':[asdict(SourceMessage(n,'user','问法')) for n in range(1,7)]}

    def test_context_retains_more_than_four_questions(self):
        list(self.turn.stream('工资呢',[],context_input=self.frozen,context_service=self.service))
        self.assertEqual(len(self.rewriter.calls[0][1]),6)

    def test_compressed_condition_only_enters_rewrite(self):
        self.frozen['messages']=[asdict(SourceMessage(1,'user','合同期限两年')),
                                asdict(SourceMessage(3,'user','背景'*60)),asdict(SourceMessage(5,'user','新问法'))]
        events=list(self.turn.stream('那约定呢',[],context_input=self.frozen,context_service=self.service))
        self.assertIn('合同期限两年',self.rewriter.calls[0][1][0].content)
        self.assertNotIn('合同期限两年',str(self.prompt.values[0]))
        self.assertEqual(events[-1]['data']['context']['compressed'],True)
        self.assertNotIn('items',events[-1]['data']['context'])
        self.assertNotIn('context_input',self.retriever.states[0])

    def test_context_failure_stops_before_retrieval(self):
        self.frozen['messages']=[asdict(SourceMessage(1,'user','x'*101))]
        with self.assertRaises(ContextTooLong):
            list(self.turn.stream('追问',[],context_input=self.frozen,context_service=self.service))
        self.assertEqual(self.retriever.states,[])

    def test_restored_context_node_does_not_call_summary_twice(self):
        self.frozen['messages']=[asdict(SourceMessage(1,'user','合同期限两年')),
                                asdict(SourceMessage(3,'user','背景'*60)),asdict(SourceMessage(5,'user','新问法'))]
        class Failure:
            def stream(self,prompt): raise RuntimeError('answer failed')
        with tempfile.TemporaryDirectory() as root,SqliteSaver.from_conn_string(str(Path(root)/'graph.db')) as saver:
            turn=LangGraphStreamingRagTurn(checkpointer=saver,**{**self.components,'model':Failure()})
            with self.assertRaises(RuntimeError):
                list(turn.stream('那约定呢',[],task_id='t',context_input=self.frozen,context_service=self.service))
            turn.model=self.model
            done=list(turn.stream('那约定呢',[],task_id='t',resume=True,context_input=self.frozen,context_service=self.service))[-1]
            self.assertEqual(done['data']['answer'],'回答')
            self.assertEqual(self.service.model.calls,1)

    def test_crash_before_context_checkpoint_uses_saved_result(self):
        cached={'summary':asdict(Summary()),'history':['合同期限两年'],'compressed':False,'estimated_tokens':6,'duration_ms':1}
        list(self.turn.stream('追问',[],context_input=self.frozen,context_service=self.service,context_cached=cached))
        self.assertEqual(self.rewriter.calls[0][1][0].content,'合同期限两年')
        self.assertEqual(self.service.model.calls,0)

    def test_stale_context_guard_blocks_unfinished_recovery(self):
        class Failure:
            def stream(self,prompt): raise RuntimeError('answer failed')
        with tempfile.TemporaryDirectory() as root,SqliteSaver.from_conn_string(str(Path(root)/'graph.db')) as saver:
            turn=LangGraphStreamingRagTurn(checkpointer=saver,**{**self.components,'model':Failure()})
            with self.assertRaises(RuntimeError):
                list(turn.stream('追问',[],task_id='t',context_input=self.frozen,context_service=self.service))
            def stale(): raise ContextValidation('stale')
            with self.assertRaises(ContextValidation):
                list(turn.stream('追问',[],task_id='t',resume=True,context_input=self.frozen,context_service=self.service,context_revision_guard=stale))

    def test_answer_input_budget_rejects_long_evidence(self):
        self.turn.prompt_budget=PromptBudget(input_tokens=2,token_counter=len)
        with self.assertRaises(ContextTooLong): list(self.turn.stream('问题',[]))


if __name__ == '__main__': unittest.main()
