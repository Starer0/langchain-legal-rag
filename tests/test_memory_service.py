import unittest
from types import SimpleNamespace


class Embeddings:
    def __init__(self): self.documents, self.queries = [], []
    def embed_documents(self, texts):
        self.documents.extend(texts)
        return [[1., 0.] if 'Python' in text else [0., 1.] for text in texts]
    def embed_query(self, question):
        self.queries.append(question)
        if question == '失败': raise TimeoutError()
        return [1., 0.] if '函数' in question else [0., 1.]


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        from memory_service import MemoryService
        self.embedding = Embeddings()
        self.service = MemoryService(self.embedding, fingerprint='fixture-v1')
        self.doc = dict(revision=0, enabled=True, core_text='', extended_text='', entries=[])

    def test_prepare_reuses_vectors_without_rewriting_text(self):
        first = self.service.prepare(self.doc, '先说结论', '我是Python初学者\n\n我在学习劳动法', True)
        first['revision'] = 1
        second = self.service.prepare(first, '详细一点', first['extended_text'], True)
        self.assertEqual(self.embedding.documents, ['我是Python初学者', '我在学习劳动法'])
        self.assertEqual(second['extended_text'], first['extended_text'])
        self.assertEqual(second['entries'][0]['id'], first['entries'][0]['id'])

    def test_duplicate_paragraphs_can_be_saved_again_with_unique_ids(self):
        first = self.service.prepare(self.doc, '', 'Python背景\n\nPython背景', True)
        second = self.service.prepare(first, '中文', first['extended_text'], True)
        self.assertEqual(len({e['id'] for e in second['entries']}), len(second['entries']))
        self.assertEqual(self.embedding.documents, ['Python背景'])

    def test_unusable_index_warns_but_keeps_core(self):
        doc = self.service.prepare(self.doc, '中文', 'Python背景', True)
        doc['entries'][0]['fingerprint'] = 'old-model'
        result = self.service.select('函数', doc)
        self.assertEqual(result['core'], '中文')
        self.assertEqual(result['status'], 'extended_unavailable')

    def test_model_cannot_expand_clear_scope_or_modify_temporary_mixed_request(self):
        class Model:
            def invoke(self, prompt): return SimpleNamespace(content='{"action":"clear","layer":"all"}')
        self.service.command_model = Model()
        for question in ['清空核心记忆', '记住：这次回答简短', '记住以后简短，顺便问一下试用期最长多久？']:
            self.assertNotIn('prepared', self.service.propose(question, {**self.doc, 'core_text':'中文', 'extended_text':'Python背景'}))

    def test_parser_cannot_invent_content_change_action_or_remove_unrequested_items(self):
        import json
        snapshot = {**self.doc, 'core_text':'回答简短，先给结论，保留来源'}
        cases = [
            ('请记住，以后使用中文', {'action':'delete','layer':'core','target':snapshot['core_text']}),
            ('请记住，以后使用中文', {'action':'add','layer':'core','content':'我在一家银行任职'}),
            ('请修改核心记忆，把回答简短改成解释详细，其他不变', {'action':'replace','layer':'core','target':snapshot['core_text'],'content':'解释详细'}),
            ('忘记这个偏好', {'action':'delete','layer':'core','target':'先给结论'}),
            ('记住回答简短。试用期最长多久', {'action':'add','layer':'core','content':'回答简短'}),
            ('请记住，我不是法律初学者', {'action':'add','layer':'core','content':'法律初学者'}),
        ]
        for question, operation in cases:
            with self.subTest(question=question):
                self.service.command_model = SimpleNamespace(invoke=lambda prompt, operation=operation:SimpleNamespace(content=json.dumps(operation,ensure_ascii=False)))
                self.assertNotIn('prepared',self.service.propose(question,snapshot))

    def test_grounded_delete_and_implicit_detail_change_preserve_other_preferences(self):
        import json
        snapshot = {**self.doc,'core_text':'回答简短，先给结论，保留来源'}
        self.service.command_model = SimpleNamespace(invoke=lambda prompt:SimpleNamespace(content=json.dumps({'action':'delete','layer':'core','target':'先给结论'})))
        deleted = self.service.propose('忘记“先给结论”这个偏好', snapshot)
        self.assertEqual(deleted['prepared']['core_text'],'回答简短，，保留来源')
        self.assertEqual(deleted['operation']['target'],'先给结论')
        self.assertEqual(deleted['question'],'忘记“先给结论”这个偏好')
        self.service.command_model = SimpleNamespace(invoke=lambda prompt:SimpleNamespace(content=json.dumps({'action':'replace','layer':'core','target':'回答简短','content':'解释详细'})))
        replaced = self.service.propose('以后不要回答简短，解释详细',snapshot)
        self.assertEqual(replaced['prepared']['core_text'],'解释详细，先给结论，保留来源')

    def test_related_background_selected_and_core_kept(self):
        doc = self.service.prepare(self.doc, '默认中文', '我是Python初学者\n\n我在学习劳动法', True)
        result = self.service.select('函数为什么报错', doc)
        self.assertEqual(result['core'], '默认中文')
        self.assertEqual(result['extended'], '我是Python初学者')
        self.assertEqual(len(result['selected_ids']), 1)

    def test_disabled_skips_query_and_core(self):
        doc = self.service.prepare(self.doc, '中文', 'Python背景', False)
        self.assertEqual(self.service.select('函数', doc)['core'], '')
        self.assertEqual(self.embedding.queries, [])

    def test_empty_extended_skips_query(self):
        self.assertEqual(self.service.select('函数', {**self.doc, 'core_text': '中文'})['core'], '中文')
        self.assertEqual(self.embedding.queries, [])

    def test_failure_keeps_core_and_reports_degradation(self):
        doc = self.service.prepare(self.doc, '中文', 'Python背景', True)
        result = self.service.select('失败', doc)
        self.assertEqual(result['core'], '中文')
        self.assertEqual(result['extended'], '')
        self.assertEqual(result['status'], 'extended_unavailable')

    def test_no_matching_memory_is_not_forced_into_context(self):
        doc = self.service.prepare(self.doc, '中文', 'Python背景', True)
        self.assertEqual(self.service.select('劳动法', doc)['selected_ids'], [])

    def test_invalid_vectors_cannot_be_selected(self):
        doc = self.service.prepare(self.doc, '中文', 'Python背景', True)
        doc['entries'][0]['embedding'] = [float('nan'), 0]
        self.assertEqual(self.service.select('函数', doc)['selected_ids'], [])

    def test_fingerprint_mismatch_does_not_use_old_vector(self):
        doc = self.service.prepare(self.doc, '中文', 'Python背景', True)
        doc['entries'][0]['fingerprint'] = 'other-model'
        self.assertEqual(self.service.select('函数', doc)['selected_ids'], [])

    def test_oversize_core_rejected(self):
        from memory_service import MemoryValidation
        with self.assertRaises(MemoryValidation):
            self.service.prepare(self.doc, 'a' * 1501, '', True)

    def test_oversize_entry_rejected_without_embedding(self):
        from memory_service import MemoryValidation
        with self.assertRaises(MemoryValidation):
            self.service.prepare(self.doc, '', 'a' * 1001, True)
        self.assertEqual(self.embedding.documents, [])

    def test_clear_does_not_call_embedding(self):
        doc = self.service.prepare(self.doc, '', 'Python背景', True)
        self.embedding.documents.clear()
        self.assertEqual(self.service.prepare(doc, '', '', True)['entries'], [])
        self.assertEqual(self.embedding.documents, [])

    def test_candidate_rejects_quotes_temporary_and_legal_questions(self):
        from memory_service import command_candidate
        for text in ['这次简短一点', '“记住中文”是什么意思', '解释一下记住这个词', '以后试用期有什么规定？']:
            self.assertFalse(command_candidate(text), text)
        for text in ['请记住，以后先给结论', '忘记我之前的编程背景', '以后回答详细一些']:
            self.assertTrue(command_candidate(text), text)

    def test_proposal_replaces_only_exact_target(self):
        class Model:
            def invoke(self, prompt):
                return SimpleNamespace(content='{"action":"replace","layer":"core","target":"回答简短","content":"解释详细"}')
        self.service.command_model = Model()
        result = self.service.propose('以后解释详细，其他不变', {**self.doc, 'revision': 3, 'core_text': '回答简短，先给结论，保留来源'})
        self.assertEqual(result['prepared']['core_text'], '解释详细，先给结论，保留来源')
        self.assertEqual(result['expected_revision'], 3)
        self.assertNotIn('已', result['answer'])

    def test_ambiguous_target_never_changes_record(self):
        class Model:
            def invoke(self, prompt): return SimpleNamespace(content='{"action":"delete","layer":"core","target":"不存在","content":""}')
        self.service.command_model = Model()
        result = self.service.propose('忘记这个偏好', {**self.doc, 'core_text': '中文'})
        self.assertNotIn('prepared', result)
        self.assertIn('明确', result['answer'])

    def test_malformed_model_result_never_saves(self):
        class Model:
            def invoke(self, prompt): return SimpleNamespace(content='not-json')
        self.service.command_model = Model()
        self.assertNotIn('prepared', self.service.propose('记住中文', self.doc))

    def test_production_factory_reads_file_settings_without_loading_rag_first(self):
        from unittest.mock import patch
        from memory_api import create_memory_service
        settings = dict(SILICONFLOW_API_KEY='fixture-embedding', DEEPSEEK_API_KEY='fixture-command',
                        DEEPSEEK_BASE_URL='http://fixture/v1', SILICONFLOW_BASE_URL='http://embedding/v1',
                        MODEL_NAME='fixture-model')
        with patch('database_settings.read_settings', return_value=settings), patch('langchain_openai.OpenAIEmbeddings') as embed, patch('langchain_openai.ChatOpenAI') as chat:
            create_memory_service()
            self.assertEqual(embed.call_args.kwargs['api_key'], 'fixture-embedding')
            self.assertEqual(chat.call_args.kwargs['base_url'], 'http://fixture/v1')
            self.assertEqual(embed.call_args.kwargs['max_retries'], 0)
