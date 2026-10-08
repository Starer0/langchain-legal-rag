import json
import unittest
from types import SimpleNamespace

from conversation_context import (ContextService, ContextLimits, SourceMessage, Summary,
                                  ContextValidation, ContextTooLong, PromptBudget)


class SummaryModel:
    def __init__(self, items): self.items, self.calls = items, 0
    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=json.dumps({'items': self.items}, ensure_ascii=False),
                               usage_metadata={'input_tokens': 12, 'output_tokens': 8})


def item(ordinal, text, kind='condition'):
    return {'kind': kind, 'text': text, 'source_ordinals': [ordinal],
            'source_quotes': [text], 'supersedes': []}


class ContextTests(unittest.TestCase):
    def service(self, items=(), **limits):
        self.model = SummaryModel(list(items))
        settings = dict(trigger_tokens=100, summary_tokens=40, retained_tokens=75, recent_turns=2)
        settings.update(limits)
        return ContextService(self.model, ContextLimits(**settings), token_counter=len)

    def test_short_history_keeps_more_than_four_turns_without_model(self):
        service = self.service()
        messages = [SourceMessage(n, 'user', '短问') for n in range(1, 7)]
        result = service.build(Summary(0, 0, []), messages)
        self.assertEqual(len(result['history']), 6)
        self.assertFalse(result['compressed'])
        self.assertEqual(self.model.calls, 0)

    def test_threshold_inclusive_does_not_compress(self):
        service = self.service()
        self.assertFalse(service.plan(Summary(0, 0, []), [SourceMessage(1,'user','x'*100)])['compress'])

    def test_long_single_message_is_rejected_not_sliced(self):
        service = self.service()
        with self.assertRaises(ContextTooLong):
            service.plan(Summary(0, 0, []), [SourceMessage(1,'user','x'*101)])

    def test_compression_preserves_early_condition_and_recent_messages(self):
        service = self.service([item(1, '合同期限两年')])
        messages = [SourceMessage(1,'user','合同期限两年'), SourceMessage(2,'assistant','这是违法的'*30),
                    SourceMessage(3,'user','早期背景'*30), SourceMessage(4,'assistant','回答'),
                    SourceMessage(5,'user','新问题'), SourceMessage(6,'assistant','回答'),
                    SourceMessage(7,'user','追问')]
        result = service.build(Summary(0,0,[]), messages)
        self.assertTrue(result['compressed'])
        self.assertIn('合同期限两年', result['history'][0])
        self.assertEqual(result['history'][-2:], ['新问题','追问'])
        self.assertNotIn('这是违法的', ''.join(result['history']))
        self.assertEqual(result['summary']['through_ordinal'], 4)

    def test_assistant_text_does_not_trigger_history_compression(self):
        service = self.service()
        result = service.build(Summary(0,0,[]), [SourceMessage(1,'user','问题'), SourceMessage(2,'assistant','x'*1000)])
        self.assertFalse(result['compressed'])
        self.assertEqual(result['history'], ['问题'])

    def test_explicit_memory_commands_do_not_enter_context(self):
        service = self.service()
        result = service.build(Summary(0,0,[]), [SourceMessage(1,'user','请记住：喜欢详细解释'), SourceMessage(3,'user','工资呢')])
        self.assertEqual(result['history'], ['工资呢'])

    def test_invented_summary_source_is_rejected(self):
        service = self.service([item(1,'合同期限两年')])
        with self.assertRaises(ContextValidation):
            service.compress(Summary(0,0,[]), [SourceMessage(1,'user','一年')], {'source_ordinals':[1], 'retained_ordinals':[]})

    def test_assistant_claim_cannot_be_summary_evidence(self):
        service = self.service([item(2,'约定合法')])
        with self.assertRaises(ContextValidation):
            service.compress(Summary(0,0,[]), [SourceMessage(2,'assistant','约定合法')], {'source_ordinals':[2], 'retained_ordinals':[]})

    def test_paraphrase_with_invented_fact_is_rejected(self):
        proposal = item(1,'一年'); proposal['text'] = '两年'
        service = self.service([proposal])
        with self.assertRaises(ContextValidation):
            service.compress(Summary(0,0,[]), [SourceMessage(1,'user','一年')], {'source_ordinals':[1], 'retained_ordinals':[]})

    def test_summary_over_budget_fails_without_truncation(self):
        service = self.service([item(1,'x'*45)])
        with self.assertRaises(ContextValidation):
            service.compress(Summary(0,0,[]), [SourceMessage(1,'user','x'*45)], {'source_ordinals':[1], 'retained_ordinals':[]})

    def test_previous_condition_cannot_silently_disappear(self):
        service = self.service([item(3,'新条件')])
        old = Summary(1,2,[item(1,'合同期限两年')])
        with self.assertRaises(ContextValidation):
            service.compress(old, [SourceMessage(3,'user','新条件')], {'source_ordinals':[3], 'retained_ordinals':[]})

    def test_correction_is_labeled_and_previous_statement_remains_traceable(self):
        service = self.service([item(1,'合同期限两年'), item(3,'更正：期限一年','correction')], summary_tokens=100)
        result = service.compress(Summary(1,2,[item(1,'合同期限两年')]),
                   [SourceMessage(3,'user','更正：期限一年')], {'source_ordinals':[3], 'retained_ordinals':[]})
        rendered = service.render(Summary(**result), [])
        self.assertIn('用户纠正', rendered[0])
        self.assertIn('期限一年', rendered[0])
        self.assertIn('合同期限两年', rendered[0])

    def test_empty_summary_for_nonempty_source_rejected(self):
        service = self.service([])
        with self.assertRaises(ContextValidation):
            service.compress(Summary(0,0,[]), [SourceMessage(1,'user','合同期限两年')], {'source_ordinals':[1], 'retained_ordinals':[]})

    def test_full_prompt_limit_fails_instead_of_dropping_evidence(self):
        budget = PromptBudget(input_tokens=10, reserved_output_tokens=4, token_counter=len)
        budget.check('1234567890')
        with self.assertRaises(ContextTooLong): budget.check('12345678901')

    def test_available_actual_usage_is_carried_separately_from_estimate(self):
        service=self.service([item(1,'合同两年')])
        messages=[SourceMessage(1,'user','合同两年'+'x'*120),SourceMessage(2,'user','追问')]
        result=service.build(Summary(),messages)
        self.assertEqual(result['usage'],{'input_tokens':12,'output_tokens':8})


if __name__ == '__main__': unittest.main()
