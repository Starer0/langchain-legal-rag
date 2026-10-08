import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import test_web_rag as web_fixtures
import test_rag_permissions as permission_fixtures
import test_rag_logging as logging_fixtures
from web_langgraph import LangGraphStreamingRagTurn


def graph_turn(turn):
    return LangGraphStreamingRagTurn(rewriter=turn.rewriter, retriever=turn.retriever,
        reranker=turn.reranker, prompt=turn.prompt, model=turn.model,
        history_turns=turn.history_turns, require_authorization=turn.require_authorization)


class GraphParityTests(web_fixtures.StreamingRagTurnTests):
    def setUp(self):
        super().setUp()
        self.turn = graph_turn(self.turn)

    def test_graph_contains_real_stage_nodes_and_edges(self):
        graph = self.turn.graph.get_graph()
        self.assertTrue({'rewrite', 'retrieve', 'rerank', 'answer', 'no_evidence'}.issubset(graph.nodes))
        self.assertTrue(any(edge.source == 'retrieve' and edge.target == 'rerank' for edge in graph.edges))

    def test_delta_arrives_before_model_finishes(self):
        gate = Event()
        self.addCleanup(gate.set)
        class Model:
            def stream(self, prompt):
                yield '部分回答'
                if not gate.wait(5): raise RuntimeError('test timeout')
                yield '后半段'
        self.turn.model = Model()
        stream = self.turn.stream('问题', [])
        with ThreadPoolExecutor(max_workers=1) as pool:
            def first_delta():
                for item in stream:
                    if item['event'] == 'delta': return item
            try:
                delta = pool.submit(first_delta).result(timeout=2)
                self.assertEqual(delta['data']['text'], '部分回答')
                self.assertFalse(gate.is_set())
            finally: gate.set()
        remaining = list(stream)
        self.assertEqual(remaining[-1]['data']['answer'], '部分回答后半段')

    def test_model_failure_never_emits_done(self):
        class Model:
            def stream(self, prompt):
                yield '半成品'
                raise RuntimeError('fake model failed')
        self.turn.model = Model()
        events = []
        with self.assertRaisesRegex(RuntimeError, 'fake model failed'):
            for item in self.turn.stream('问题', []): events.append(item)
        self.assertNotIn('done', [event['event'] for event in events])


class GraphPermissionTests(permission_fixtures.StreamingPermissionTests):
    def setUp(self):
        super().setUp()
        self.turn = graph_turn(self.turn)

    def test_parallel_users_have_independent_state_scope_and_answers(self):
        class Model:
            def stream(self, prompt):
                yield prompt['question'] + ':' + prompt['context']
        self.turn.model = Model()
        def run(scope):
            return list(self.turn.stream('问题' + scope, [], allowed_knowledge_bases={scope}))[-1]['data']
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, c = list(pool.map(run, ['A', 'C']))
        self.assertIn('问题A', a['answer'])
        self.assertIn(self.a.page_content, a['answer'])
        self.assertNotIn(self.c.page_content, a['answer'])
        self.assertIn('问题C', c['answer'])
        self.assertNotIn(self.a.page_content, c['answer'])
        self.assertEqual([s['knowledge_base_id'] for s in a['sources']], ['A'])
        self.assertEqual([s['knowledge_base_id'] for s in c['sources']], ['C'])


class GraphLoggingTests(logging_fixtures.TraceLoggingTests):
    def setUp(self):
        super().setUp()
        self.turn = graph_turn(self.turn)

    def test_closed_answer_stream_is_logged_as_interrupted(self):
        self.close_answer_stream(has_more=True)

    def test_close_while_waiting_for_end_is_interrupted(self):
        self.close_answer_stream(has_more=False)

    def close_answer_stream(self, has_more):
        gate = Event()
        self.addCleanup(gate.set)
        state_ref = []
        prepare = self.turn._prepare_state
        def capture(*args):
            state = prepare(*args)
            state_ref.append(state)
            return state
        self.turn._prepare_state = capture
        class Model:
            def stream(self, prompt):
                yield '部分'
                if not gate.wait(5): raise RuntimeError('test timeout')
                if has_more: yield '不能交付的后续'
        self.turn.model = Model()
        stream = self.turn.stream('问题', [], allowed_knowledge_bases={'A'}, trace=self.trace())
        with self.assertLogs('test.rag', level='INFO') as captured:
            for item in stream:
                if item['event'] == 'delta': break
            with ThreadPoolExecutor(max_workers=1) as pool:
                close = pool.submit(stream.close)
                try:
                    self.assertTrue(state_ref[0]['_cancelled'].wait(2))
                finally: gate.set()
                close.result(timeout=2)
        rows = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual(rows[-1]['event'], 'rag_stage_failed')
        self.assertEqual(rows[-1]['error_type'], 'GeneratorExit')
