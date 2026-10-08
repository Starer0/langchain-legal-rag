import unittest
from scripts.evaluate_model_memory import evaluate


class EvaluationTests(unittest.TestCase):
    def test_scope_split_mismatch_is_reported_even_when_main_route_is_correct(self):
        case=dict(id='mixed',route='preference',memory_request=True,out_of_scope_request=True,actions=['remember'],required=[],forbidden=[])
        row=dict(id='mixed',route='preference',memory_request=True,operations=[],decision={'answer':{'out_of_scope_request':False}})
        self.assertEqual(evaluate([case],[row])['metrics']['scope_split_error'],1)

    def test_unmeasured_is_null_and_fragment_errors_are_observed(self):
        case=dict(id='x',route='preference',memory_request=True,actions=['remember'],required=['例子'],forbidden=['律师'])
        offline=evaluate([case])
        self.assertIsNone(offline['metrics'])
        self.assertIsNone(offline['cases'][0]['observation'])
        measured=evaluate([case],[dict(id='x',route='rag',memory_request=False,operations=[])])
        self.assertEqual(measured['metrics']['route_error'],1)
        self.assertEqual(measured['metrics']['lost_required_fragments'],1)
        self.assertIsNone(measured['cost'])
