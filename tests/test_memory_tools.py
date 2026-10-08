import unittest
from copy import deepcopy


def call_args(content='解释TCP时联系网络排障例子'):
    return {'operations':[dict(action='remember',layer='core',category='preference',basis='declared',
        content=content,sources=[dict(kind='current_input',reference='t1',quote=content)],
        target_ids=[],relation='new',delta='')]}


class ContractTests(unittest.TestCase):
    def test_open_content(self):
        from memory_tool_contract import parse_memory_call
        self.assertEqual(parse_memory_call(call_args()).operations[0].content,'解释TCP时联系网络排障例子')

    def test_reject_extra_missing_and_oversize(self):
        from memory_tool_contract import parse_memory_call
        invalid=[{**call_args(),'user_id':'other'},call_args('x'*1001)]
        missing=deepcopy(call_args());del missing['operations'][0]['basis'];invalid.append(missing)
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):parse_memory_call(raw)

    def test_empty_noop(self):
        from memory_tool_contract import parse_memory_call
        self.assertEqual(parse_memory_call({'operations':[]}).operations,())


class PreparationTests(unittest.TestCase):
    def setUp(self):
        from memory_service import MemoryService
        from test_memory_service import Embeddings
        from memory_tools import MemoryTools
        self.tools=MemoryTools(MemoryService(Embeddings(),fingerprint='fixture'))

    def prepare(self,raw=None,**overrides):
        from memory_tool_contract import ToolContext,parse_memory_call
        document=dict(core_text='',extended_text='',revision=0,enabled=True,entries=[],facts=[],auto_accumulate=False)
        document.update(overrides.pop('document',{}))
        snapshot=dict(document=document,question='解释TCP时联系网络排障例子',memory_request=True,
            sources=[dict(kind='current_input',reference='t1',role='user',content='解释TCP时联系网络排障例子')],conversation_id='c1')
        snapshot.update(overrides.pop('snapshot',{}))
        return self.tools.prepare(parse_memory_call(raw or call_args()),ToolContext('a',overrides.pop('scope','foreground'),'t1',document['revision'],0,snapshot))

    def test_unknown_source_and_target_rejected(self):
        from memory_service import MemoryValidation
        raw=call_args();raw['operations'][0]['sources'][0]['reference']='other'
        with self.assertRaises(MemoryValidation):self.prepare(raw)
        raw=call_args();raw['operations'][0].update(relation='duplicate',target_ids=['foreign'])
        with self.assertRaises(MemoryValidation):self.prepare(raw)

    def test_explicit_off_allowed_but_background_blocked(self):
        from memory_service import MemoryValidation
        self.assertEqual(len(self.prepare(document={'enabled':False}).facts),1)
        with self.assertRaises(MemoryValidation):self.prepare(scope='background')
        with self.assertRaises(MemoryValidation):self.prepare(snapshot={'memory_request':False})

    def test_duplicate_preserves_old_and_complement_adds_delta(self):
        raw=call_args();op=raw['operations'][0];op.update(relation='complement',target_ids=['f1'],delta='分点解释')
        update=self.prepare(raw,document={'core_text':'先给结论\n\n结合例子','facts':[
            dict(id='f1',layer='core',category='preference',content='先给结论',basis='declared',protected=False),
            dict(id='f2',layer='core',category='preference',content='结合例子',basis='declared',protected=False)]})
        self.assertEqual(update.prepared_document['core_text'],'先给结论\n\n结合例子\n\n分点解释')

    def test_inferred_requires_two_turns_and_only_suggests(self):
        from memory_service import MemoryValidation
        raw=call_args();op=raw['operations'][0];op.update(basis='inferred',category='learning_focus',layer='extended')
        with self.assertRaises(MemoryValidation):self.prepare(raw)
        op['sources'].append(dict(kind='message',reference='c1:1',quote='TCP'))
        update=self.prepare(raw,snapshot={'sources':[
            dict(kind='current_input',reference='t1',role='user',content='解释TCP时联系网络排障例子'),
            dict(kind='message',reference='c1:1',role='user',content='TCP基础')]})
        self.assertEqual(update.facts,())
        self.assertEqual(update.suggestions[0]['basis'],'inferred')
