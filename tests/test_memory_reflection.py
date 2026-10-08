import unittest
from types import SimpleNamespace
import json
from conversation_context import SourceMessage
from memory_reflection import ReflectionService


def candidate(text,relation='new',target=''):
    return {'layer':'core','text':text,'source_ordinals':[1],'source_quotes':[text],'relation':relation,'target':target}


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        self.service=ReflectionService(None,None)
        self.existing={'revision':0,'enabled':True,'core_text':'','extended_text':'','entries':[]}

    def check(self,text,proposal=None,role='user'):
        return self.service.validate({'candidates':[proposal or candidate(text)]},[SourceMessage(1,role,text)],self.existing)

    def test_stable_natural_preference_without_memory_command(self):
        result=self.check('我习惯看具体例子，专业词太多我看不懂。')
        self.assertEqual(len(result['additions']),1)

    def test_temporary_and_case_facts_are_not_long_term_memory(self):
        for text in ['这次我喜欢简短回答','我的合同期限两年','假设我是HR','我今天需要工资计算','同事说他喜欢简短回答']:
            with self.subTest(text=text): self.assertEqual(self.check(text)['additions'],[])

    def test_assistant_and_ungrounded_paraphrase_cannot_add(self):
        self.assertEqual(self.check('我喜欢简短回答',role='assistant')['additions'],[])
        proposal=candidate('我喜欢详细解释'); proposal['source_quotes']=['我喜欢简短回答']
        self.assertEqual(self.check('我喜欢简短回答',proposal)['additions'],[])

    def test_exact_duplicate_does_not_add_or_call_embeddings(self):
        self.existing['core_text']='我喜欢简短回答'
        self.assertEqual(self.check('我喜欢简短回答')['additions'],[])

    def test_conflict_is_suggestion_not_overwrite(self):
        self.existing['core_text']='我喜欢详细解释'
        result=self.check('我喜欢简短回答',candidate('我喜欢简短回答','conflict','我喜欢详细解释'))
        self.assertEqual(result['additions'],[])
        self.assertEqual(len(result['suggestions']),1)

    def test_opposed_verbosity_marked_new_still_requires_review(self):
        self.existing['core_text']='我喜欢详细解释'
        result=self.check('我喜欢简短回答')
        self.assertEqual(result['additions'],[])
        self.assertEqual(result['suggestions'][0]['relation'],'conflict')

    def test_wrong_source_ordinal_is_skipped(self):
        proposal=candidate('我喜欢简短回答');proposal['source_ordinals']=[2]
        self.assertEqual(self.check('我喜欢简短回答',proposal)['additions'],[])

    def test_disabled_memory_skips_all_candidates(self):
        self.existing['enabled']=False
        self.assertEqual(self.check('我喜欢简短回答')['additions'],[])

    def test_one_batch_model_call_not_per_message(self):
        class Model:
            calls=0
            def invoke(inner,prompt):
                inner.calls+=1
                return SimpleNamespace(content=json.dumps({'candidates':[candidate('我喜欢简短回答')]}))
        model=Model();service=ReflectionService(model,None)
        result=service.extract([SourceMessage(1,'user','我喜欢简短回答'),SourceMessage(3,'user','劳动法呢')],self.existing,[])
        self.assertEqual(len(result['additions']),1)
        self.assertEqual(model.calls,1)

    def test_json_candidate_array_is_validated_with_original_sources(self):
        class Model:
            def invoke(self, prompt):
                return SimpleNamespace(content=json.dumps([candidate('我喜欢简短回答')]))
        service = ReflectionService(Model(), None)
        result = service.extract([SourceMessage(1, 'user', '我喜欢简短回答')], self.existing, [])
        self.assertEqual(result['additions'], [candidate('我喜欢简短回答')])

    def test_quoted_negated_and_separate_case_statement_are_not_personal_memory(self):
        cases=[('请翻译这句话：“我喜欢简短回答”。','我喜欢简短回答'),
               ('我喜欢详细解释，并不是说我喜欢简短回答。','我喜欢简短回答'),
               ('我法律基础薄弱。我的案件已经立案。','我的案件已经立案')]
        for message,quote in cases:
            with self.subTest(message=message):
                result=self.check(message,candidate(quote));self.assertEqual(result['additions'],[]);self.assertEqual(result['suggestions'],[])

    def test_conflict_requires_complete_target_in_declared_layer(self):
        self.existing['core_text']='我的回答偏好：我喜欢详细解释。'
        self.assertEqual(self.check('我喜欢简短回答',candidate('我喜欢简短回答','conflict','详细'))['suggestions'],[])
        self.existing['core_text']='';self.existing['extended_text']='我喜欢详细解释'
        self.assertEqual(self.check('我喜欢简短回答',candidate('我喜欢简短回答','conflict','我喜欢详细解释'))['suggestions'],[])


if __name__=='__main__': unittest.main()
