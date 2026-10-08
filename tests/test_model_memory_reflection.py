import unittest
from langchain_core.messages import AIMessage
from conversation_context import SourceMessage
from test_memory_tools import call_args


class ToolReviewTests(unittest.TestCase):
    def test_native_review_can_noop_or_return_open_candidate(self):
        from memory_reflection import ReflectionService
        class Model:
            calls=0
            def bind_tools(self,tools):self.tools=tools;return self
            def invoke(self,prompt):self.calls+=1;return AIMessage(content='',tool_calls=[])
        model=Model();service=ReflectionService(model,None)
        self.assertIsNone(service.review([SourceMessage(1,'user','TCP如何排障')],{},[]))
        self.assertEqual(model.calls,1)
        self.assertEqual(len(model.tools),1)

    def test_review_rejects_text_json_and_unknown_tool(self):
        from memory_reflection import ReflectionService
        class Model:
            def bind_tools(self,tools):return self
            def invoke(self,prompt):return AIMessage(content='',tool_calls=[dict(name='plan_answer',id='p',args={})])
        with self.assertRaises(ValueError):ReflectionService(Model(),None).review([],{},[])
