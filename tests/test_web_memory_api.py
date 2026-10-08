import os
import unittest
from fastapi.testclient import TestClient
import test_web_login as login
from memory_storage import MemoryStore
from memory_service import MemoryService
from test_memory_service import Embeddings


class WebMemoryApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS') != '1': raise unittest.SkipTest('Real PostgreSQL required')

    def setUp(self):
        self.fixture = login.WebLoginTests()
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        from prepare_generation_tasks import prepare_tasks
        from prepare_account_memory import prepare_memory
        prepare_tasks(self.fixture.db, apply=True, schema=self.fixture.schema)
        prepare_memory(self.fixture.db, apply=True, schema=self.fixture.schema)
        self.store = MemoryStore(schema=self.fixture.schema, **self.fixture.settings)
        self.service = MemoryService(Embeddings(), fingerprint='fixture-v1')
        from web_app import create_app
        self.app = create_app(self.fixture.store, object(), authentication=self.fixture.auth,
                              memory_store=self.store, memory_service=self.service)

    def payload(self, **kwargs):
        return dict(core_text='中文', extended_text='Python初学者', enabled=True, expected_revision=0, **kwargs)

    def test_authentication_csrf_and_no_store(self):
        with TestClient(self.app) as client:
            self.assertEqual(client.get('/api/memory').status_code, 401)
            self.fixture.login_client(client)
            response = client.get('/api/memory')
            self.assertEqual(response.headers['cache-control'], 'no-store')
            client.headers.pop('x-csrf-token')
            self.assertEqual(client.put('/api/memory', json=self.payload()).status_code, 403)

    def test_save_conflict_isolation_and_safe_projection(self):
        with TestClient(self.app) as client, TestClient(self.app) as colleague:
            self.fixture.login_client(client)
            self.fixture.login_client(colleague, 'demo_c')
            response = client.put('/api/memory', json={**self.payload(), 'user_id': 'forged'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['revision'], 1)
            self.assertNotIn('entries', response.json())
            self.assertEqual(client.put('/api/memory', json=self.payload()).status_code, 409)
            self.assertEqual(colleague.get('/api/memory').json()['core_text'], '')

    def test_embedding_failure_preserves_old_memory(self):
        with TestClient(self.app) as client:
            self.fixture.login_client(client)
            self.assertEqual(client.put('/api/memory', json=self.payload()).status_code, 200)
            def fail(texts): raise TimeoutError('fixture')
            self.service.embeddings.embed_documents = fail
            response = client.put('/api/memory', json={**self.payload(), 'extended_text':'新的背景', 'expected_revision':1})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(client.get('/api/memory').json()['extended_text'], 'Python初学者')

    def test_disable_reuses_vectors_and_overlimit_is_safe(self):
        with TestClient(self.app) as client:
            self.fixture.login_client(client)
            client.put('/api/memory', json=self.payload())
            def fail(texts): raise AssertionError('no fresh vectors expected')
            self.service.embeddings.embed_documents = fail
            disabled = client.put('/api/memory', json={**self.payload(), 'enabled':False, 'expected_revision':1})
            self.assertEqual(disabled.status_code, 200)
            self.assertFalse(disabled.json()['enabled'])
            too_long = client.put('/api/memory', json={**self.payload(), 'core_text':'a'*1501, 'expected_revision':2})
            self.assertEqual(too_long.status_code, 422)
            self.assertEqual(client.get('/api/memory').json()['core_text'], '中文')

    def test_saved_event_records_revision_and_counts_without_private_text(self):
        with TestClient(self.app) as client:
            self.fixture.login_client(client)
            with self.assertLogs('legal_rag.requests',level='INFO') as logs:
                response = client.put('/api/memory',json={**self.payload(),'core_text':'私人偏好'})
            self.assertEqual(response.status_code,200)
            events=[line for line in logs.output if 'memory_saved' in line]
            self.assertEqual(len(events),1)
            self.assertIn('"revision": 1',events[0])
            self.assertNotIn('私人偏好',events[0])
            self.assertNotIn('Python初学者',events[0])
