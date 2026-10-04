import unittest
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma

from rag_app import _create_candidate_retriever
from test_web_rag import FakeRewriter, FakeRetriever, FakeReranker, FakePrompt, FakeStreamingModel
from web_rag import StreamingRagTurn


class LocalEmbeddings(Embeddings):
    query_calls = 0
    def embed_documents(self, texts):
        return [[1.0, float(len(text) % 3)] for text in texts]
    def embed_query(self, text):
        self.query_calls += 1
        return [1.0, 0.0]


def doc(label, text=None, status='active'):
    metadata = {'index_status': status, 'article': label, 'law_id': label, 'version': '1'}
    if label != 'missing':
        metadata['knowledge_base_id'] = label
    return Document(page_content=text or f'劳动工资 {label}', metadata=metadata)


class RetrievalPermissionTests(unittest.TestCase):
    def setUp(self):
        self.store = Chroma(collection_name='permission_' + uuid4().hex, embedding_function=LocalEmbeddings())
        self.addCleanup(self.store.delete_collection)
        self.store.add_documents([doc('A'), doc('B'), doc('C'), doc('missing'), doc('A', '劳动工资 inactive', 'inactive')])

    def test_real_vector_and_bm25_filter_even_when_routing_disabled(self):
        for hybrid in (False, True):
            retriever = _create_candidate_retriever(self.store, [], 20, False, hybrid, 20)
            for scope in ({'A', 'B'}, {'C'}, set()):
                with self.subTest(hybrid=hybrid, scope=scope):
                    results = retriever.invoke({'question': '劳动工资', 'retrieval_question': '劳动工资',
                                               'allowed_knowledge_bases': scope})
                    self.assertEqual({d.metadata.get('knowledge_base_id') for d in results}, scope)
                    self.assertTrue(all(d.metadata['index_status'] == 'active' for d in results))

    def test_parallel_scopes_do_not_mutate_shared_retriever(self):
        retriever = _create_candidate_retriever(self.store, [], 20, False, True, 20)
        scopes = [{'A'}, {'C'}] * 6
        def retrieve(scope):
            return {d.metadata.get('knowledge_base_id') for d in retriever.invoke({
                'question': '劳动工资', 'retrieval_question': '劳动工资', 'allowed_knowledge_bases': scope})}
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(retrieve, scopes)), scopes)

    def test_empty_scope_does_not_embed_query(self):
        embeddings = self.store._embedding_function
        before = embeddings.query_calls
        retriever = _create_candidate_retriever(self.store, [], 20, True, True, 20)
        self.assertEqual(retriever.invoke({'question': '问题', 'retrieval_question': '问题',
                                         'allowed_knowledge_bases': set()}), [])
        self.assertEqual(embeddings.query_calls, before)

    def test_law_and_guide_route_intersects_permission(self):
        self.store.add_documents([Document(page_content='劳动工资指南 C', metadata={
            'knowledge_base_id': 'C', 'document_type': '办事指南', 'index_status': 'active', 'guide_id': 'guide_c'})])
        laws = [{'law_id': 'A', 'law_name': '劳动法', 'aliases': []}]
        for hybrid in (False, True):
            retriever = _create_candidate_retriever(self.store, laws, 20, True, hybrid, 20)
            state = {'question': '劳动法办事指南劳动工资', 'retrieval_question': '劳动工资',
                     'include_guide': True, 'allowed_knowledge_bases': {'A', 'B'}}
            self.assertEqual({d.metadata['knowledge_base_id'] for d in retriever.invoke(state)}, {'A'})

    def test_malformed_scope_is_rejected(self):
        retriever = _create_candidate_retriever(self.store, [], 20, False, False, 20)
        for scope in (None, 'ABC', {' bad'}):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                retriever.invoke({'question': '问题', 'retrieval_question': '问题', 'allowed_knowledge_bases': scope})


class StreamingPermissionTests(unittest.TestCase):
    def setUp(self):
        self.a, self.c = doc('A'), doc('C')
        self.rewriter = FakeRewriter('劳动工资')
        self.retriever = FakeRetriever([self.a, self.c, doc('missing')])
        self.reranker = FakeReranker([self.a, self.c, doc('A', '伪造的新正文')])
        self.prompt = FakePrompt()
        self.model = FakeStreamingModel(['回答'])
        self.turn = StreamingRagTurn(rewriter=self.rewriter, retriever=self.retriever, reranker=self.reranker,
                                    prompt=self.prompt, model=self.model, history_turns=4, require_authorization=True)

    def test_filters_before_remote_rerank_context_and_citations(self):
        events = list(self.turn.stream('问题', [], allowed_knowledge_bases={'A'}))
        self.assertEqual(self.reranker.states[0]['candidates'], [self.a])
        self.assertNotIn('劳动工资 C', self.prompt.values[0]['context'])
        self.assertNotIn('伪造', self.prompt.values[0]['context'])
        self.assertEqual(len(events[-1]['data']['sources']), 1)
        self.assertEqual(len(events[-1]['data']['candidates']), 1)

    def test_missing_invalid_or_empty_scope_never_calls_models(self):
        for scope in (None, 'ABC', {' bad'}, set()):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                list(self.turn.stream('问题', [], allowed_knowledge_bases=scope))
        self.assertEqual(self.rewriter.calls, [])
        self.assertEqual(self.model.prompts, [])

    def test_no_accessible_evidence_answers_locally(self):
        self.retriever.documents = [self.c]
        events = list(self.turn.stream('问题', [], allowed_knowledge_bases={'A'}))
        self.assertEqual(events[-1]['data']['answer'], '当前账号可访问的资料中没有足够依据。')
        self.assertEqual(events[-1]['data']['sources'], [])
        self.assertEqual(self.reranker.states, [])
        self.assertEqual(self.model.prompts, [])

    def test_old_assistant_evidence_not_sent_to_rewrite(self):
        history = [HumanMessage(content='之前的问题'), AIMessage(content='旧的越权资料')]
        list(self.turn.stream('继续', history, allowed_knowledge_bases={'A'}))
        self.assertEqual(self.rewriter.calls[0][1], [history[0]])

    def test_reranker_cannot_rewrite_candidate_body_in_place(self):
        from langchain_core.runnables import RunnableLambda
        def mutate(state):
            state['candidates'][0].page_content = '伪造篡改正文'
            return state['candidates']
        self.turn.reranker = RunnableLambda(mutate)
        events = list(self.turn.stream('问题', [], allowed_knowledge_bases={'A'}))
        self.assertEqual(events[-1]['data']['sources'], [])
        self.assertEqual(self.model.prompts, [])

    def test_rerank_score_copy_is_still_valid_evidence(self):
        self.reranker.documents = [Document(page_content=self.a.page_content,
                                           metadata={**self.a.metadata, 'rerank_score': 0.9})]
        events = list(self.turn.stream('问题', [], allowed_knowledge_bases={'A'}))
        self.assertEqual(len(events[-1]['data']['sources']), 1)

    def test_forged_only_rerank_result_answers_locally(self):
        self.reranker.documents = [self.c]
        events = list(self.turn.stream('问题', [], allowed_knowledge_bases={'A'}))
        self.assertEqual(events[-1]['data']['sources'], [])
        self.assertEqual(self.model.prompts, [])


class RewritePermissionTests(unittest.TestCase):
    def test_guide_catalog_contains_only_accessible_entries(self):
        from query_rewrite import RetrievalQuestionRewriter
        from test_query_rewrite import CapturingModel
        model = CapturingModel('{}')
        guides = [{'document_type': '办事指南', 'title': f'内部指南 {label}',
                   'knowledge_base_id': label} for label in ('A', 'C')]
        rewriter = RetrievalQuestionRewriter(model, guides)
        rewriter.rewrite('材料', [], allowed_knowledge_bases={'A'})
        prompt = model.prompts[0].to_string()
        self.assertIn('内部指南 A', prompt)
        self.assertNotIn('内部指南 C', prompt)
