import json
import logging
import unittest
from concurrent.futures import ThreadPoolExecutor

from langchain_core.runnables import RunnableLambda
from test_rag_permissions import doc
from test_web_rag import FakeRewriter, FakeRetriever, FakeReranker, FakePrompt, FakeStreamingModel
from web_rag import StreamingRagTurn


class TraceLoggingTests(unittest.TestCase):
    def setUp(self):
        self.a, self.c = doc('A'), doc('C', '禁止写入的 C 正文')
        self.rewriter = FakeRewriter('改写后的问题')
        self.model = FakeStreamingModel(['答', '案'])
        self.turn = StreamingRagTurn(rewriter=self.rewriter, retriever=FakeRetriever([self.a, self.c]),
                                    reranker=FakeReranker([self.a, self.c]), prompt=FakePrompt(),
                                    model=self.model, history_turns=4, require_authorization=True)

    def trace(self, request_id='request-a'):
        from rag_logging import RagRequestTrace
        return RagRequestTrace(request_id=request_id, user_id='user-a', conversation_id='conversation-a',
                               allowed_knowledge_bases={'A'}, logger=logging.getLogger('test.rag'))

    def run_turn(self, trace, scope={'A'}):
        with self.assertLogs('test.rag', level='INFO') as captured:
            events = list(self.turn.stream('原始问题', [], allowed_knowledge_bases=scope, trace=trace))
        return events, [json.loads(record.getMessage()) for record in captured.records]

    def test_logs_actual_authorized_evidence_and_correlated_stage_timings(self):
        events, rows = self.run_turn(self.trace())
        self.assertEqual(events[-1]['event'], 'done')
        self.assertTrue(all(row['request_id'] == 'request-a' for row in rows))
        self.assertTrue(all(row['user_id'] == 'user-a' for row in rows))
        self.assertEqual(rows[0]['question'], '原始问题')
        finished = {row['stage']: row for row in rows if row['event'] == 'rag_stage_finished'}
        self.assertEqual(set(finished), {'rewrite', 'retrieve', 'rerank', 'answer'})
        self.assertEqual(finished['rewrite']['retrieval_question'], '改写后的问题')
        self.assertEqual(finished['retrieve']['documents'][0]['content'], self.a.page_content)
        self.assertEqual(finished['retrieve']['documents'][0]['knowledge_base_id'], 'A')
        self.assertEqual(len(finished['rerank']['documents']), 1)
        self.assertEqual(finished['answer']['answer_chars'], 2)
        self.assertTrue(all(row['duration_ms'] >= 0 for row in finished.values()))
        self.assertNotIn(self.c.page_content, json.dumps(rows, ensure_ascii=False))

    def test_retrieval_failure_logs_phase_without_provider_payload(self):
        def fail(state):
            raise RuntimeError('provider-key-and-private-content')
        self.turn.retriever = RunnableLambda(fail)
        with self.assertLogs('test.rag', level='INFO') as captured, self.assertRaises(RuntimeError):
            list(self.turn.stream('问题', [], allowed_knowledge_bases={'A'}, trace=self.trace()))
        rows = [json.loads(record.getMessage()) for record in captured.records]
        failure = rows[-1]
        self.assertEqual((failure['event'], failure['stage'], failure['error_type']),
                         ('rag_stage_failed', 'retrieve', 'RuntimeError'))
        self.assertNotIn('provider-key', json.dumps(rows))

    def test_no_evidence_skips_remote_stages_and_logs_reason(self):
        self.turn.retriever.documents = [self.c]
        events, rows = self.run_turn(self.trace())
        skipped = {row['stage'] for row in rows if row['event'] == 'rag_stage_skipped'}
        self.assertEqual(skipped, {'rerank', 'answer'})
        self.assertEqual(events[-1]['data']['sources'], [])
        self.assertEqual(self.model.prompts, [])
        self.assertNotIn(self.c.page_content, json.dumps(rows, ensure_ascii=False))

    def test_interleaved_streams_keep_request_identity(self):
        first = self.turn.stream('第一问', [], allowed_knowledge_bases={'A'}, trace=self.trace('first'))
        second = self.turn.stream('第二问', [], allowed_knowledge_bases={'A'}, trace=self.trace('second'))
        with self.assertLogs('test.rag', level='INFO') as captured:
            with ThreadPoolExecutor(max_workers=2) as pool:
                active = [first, second]
                while active:
                    for stream in list(active):
                        try:
                            pool.submit(next, stream).result()
                        except StopIteration:
                            active.remove(stream)
        rows = [json.loads(record.getMessage()) for record in captured.records]
        for request_id, question in [('first', '第一问'), ('second', '第二问')]:
            own = [row for row in rows if row['request_id'] == request_id]
            self.assertEqual(own[0]['question'], question)
            self.assertEqual(own[-1]['stage'], 'answer')

    def test_closed_answer_stream_is_logged_as_interrupted(self):
        stream = self.turn.stream('问题', [], allowed_knowledge_bases={'A'}, trace=self.trace())
        with self.assertLogs('test.rag', level='INFO') as captured:
            for item in stream:
                if item['event'] == 'delta':
                    break
            stream.close()
        rows = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual(rows[-1]['event'], 'rag_stage_failed')
        self.assertEqual(rows[-1]['error_type'], 'GeneratorExit')
        self.assertFalse(any(row['event'] == 'rag_stage_finished' and row['stage'] == 'answer' for row in rows))

    def test_trace_scope_mismatch_is_rejected_before_logging_or_models(self):
        with self.assertRaises(ValueError):
            list(self.turn.stream('问题', [], allowed_knowledge_bases={'C'}, trace=self.trace()))
        self.assertEqual(self.rewriter.calls, [])

    def test_log_payload_bounds_are_explicit_and_metadata_is_whitelisted(self):
        from rag_logging import MAX_TEXT_CHARS, MAX_DOCUMENTS
        trace = self.trace()
        large = doc('A', '字' * (MAX_TEXT_CHARS + 1))
        large.metadata['Authorization'] = 'secret-metadata'
        payload = trace.documents([large] * (MAX_DOCUMENTS + 1))
        self.assertEqual(payload['document_count'], MAX_DOCUMENTS + 1)
        self.assertEqual(len(payload['documents']), MAX_DOCUMENTS)
        self.assertTrue(payload['documents_truncated'])
        self.assertEqual(len(payload['documents'][0]['content']), MAX_TEXT_CHARS)
        self.assertTrue(payload['documents'][0]['content_truncated'])
        self.assertNotIn('secret-metadata', json.dumps(payload))

    def test_model_failure_has_no_completed_answer_stage(self):
        class FailingModel:
            def stream(self, prompt):
                yield '半个答案'
                raise RuntimeError('provider-specific-error')
        self.turn.model = FailingModel()
        with self.assertLogs('test.rag', level='INFO') as captured, self.assertRaises(RuntimeError):
            list(self.turn.stream('问题', [], allowed_knowledge_bases={'A'}, trace=self.trace()))
        rows = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual((rows[-1]['event'], rows[-1]['stage']), ('rag_stage_failed', 'answer'))
        self.assertFalse(any(row['event'] == 'rag_stage_finished' and row['stage'] == 'answer' for row in rows))
