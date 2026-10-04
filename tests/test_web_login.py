import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


class Clock:
    def __init__(self):
        self.value = datetime(2030, 1, 1, tzinfo=timezone.utc)
    def __call__(self):
        return self.value
    def advance(self, **kwargs):
        self.value += timedelta(**kwargs)


class WebLoginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS') != '1':
            raise unittest.SkipTest('Set RUN_POSTGRES_TESTS=1 for real PostgreSQL login tests')

    def setUp(self):
        import psycopg2
        from migrate_sqlite_to_postgres import migrate
        from prepare_demo_accounts import prepare_accounts
        from web_storage import SQLiteConversationStore
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = dict(host='127.0.0.1', port=5433, dbname='legal_rag', user='legal_rag',
                             password=Path('.postgres-password').read_text().strip(), connect_timeout=5)
        self.db = psycopg2.connect(**self.settings)
        self.addCleanup(self.db.close)
        self.schema = 'login_test_' + uuid4().hex
        source = Path(self.directory.name) / 'fixture.sqlite3'
        sqlite = SQLiteConversationStore(source)
        sqlite.ensure_session('old-anonymous')
        self.old = sqlite.create_conversation('old-anonymous')
        sqlite.save_complete_conversation_turn('old-anonymous', self.old.id, '旧问题', '旧回答')
        migrate(source, self.db, apply=True, schema=self.schema)
        self.addCleanup(self.cleanup_schema)
        credentials = Path(self.directory.name) / 'credentials.json'
        prepare_accounts(self.db, credentials, apply=True, schema=self.schema)
        self.credentials = {item['username']: item for item in json.loads(credentials.read_text())['accounts']}
        from prepare_web_login import prepare_login
        prepare_login(self.db, apply=True, schema=self.schema)
        from account_storage import PostgresAccountStore
        from login_sessions import PostgresLoginSessions
        from postgres_storage import PostgresUserConversationStore
        self.accounts = PostgresAccountStore(schema=self.schema, **self.settings)
        self.clock = Clock()
        self.auth = PostgresLoginSessions(self.accounts, now=self.clock, schema=self.schema, **self.settings)
        self.store = PostgresUserConversationStore(schema=self.schema, **self.settings)

    def cleanup_schema(self):
        from psycopg2 import sql
        self.db.rollback()
        if not self.schema.startswith('login_test_') or len(self.schema) != 43:
            raise ValueError('Refusing to remove a non-test schema')
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def login_session(self, username='demo_ab'):
        entry = self.credentials[username]
        return self.auth.login(username, entry['password'])

    def test_chat_passes_server_scope_ignores_payload_and_refreshes_each_turn(self):
        from fastapi.testclient import TestClient
        from web_app import create_app
        from test_web_app import SuccessfulTurn
        from psycopg2 import sql
        class CapturingTurn(SuccessfulTurn):
            scopes = []
            def stream(inner, question, messages, *, allowed_knowledge_bases=None, trace=None):
                inner.scopes.append(allowed_knowledge_bases)
                yield from super().stream(question, messages)
        turn = CapturingTurn()
        with TestClient(create_app(self.store, turn, authentication=self.auth)) as client:
            self.login_client(client)
            cid = client.post('/api/conversations').json()['id']
            url = f'/api/conversations/{cid}/chat'
            first = client.post(url, json={'question': '问题', 'allowed_knowledge_bases': ['C']})
            self.assertIn('event: done', first.text)
            self.assertEqual(turn.scopes, [frozenset({'A', 'B'})])
            with self.db, self.db.cursor() as cursor:
                cursor.execute(sql.SQL('DELETE FROM {}.account_level_libraries WHERE level_id=%s AND knowledge_base_id=%s').format(sql.Identifier(self.schema)), ('level_ab', 'B'))
            client.post(url, json={'question': '继续'})
            self.assertEqual(turn.scopes[-1], frozenset({'A'}))

    def test_chat_permission_failure_never_starts_answer(self):
        from unittest.mock import patch
        with self.client() as client:
            self.login_client(client)
            cid = client.post('/api/conversations').json()['id']
            url = f'/api/conversations/{cid}/chat'
            with patch.object(self.accounts, 'allowed_knowledge_bases', return_value=frozenset()):
                self.assertEqual(client.post(url, json={'question': '问题'}).status_code, 403)
            with patch.object(self.accounts, 'allowed_knowledge_bases', side_effect=RuntimeError('private database details')):
                response = client.post(url, json={'question': '问题'})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn('private database details', response.text)
            self.assertEqual(client.get(f'/api/conversations/{cid}/messages').json()['messages'], [])

    def test_rag_logs_share_http_id_and_are_not_available_to_normal_accounts(self):
        from fastapi.testclient import TestClient
        from web_app import create_app
        from web_rag import StreamingRagTurn
        from test_web_rag import FakeRewriter, FakeRetriever, FakeReranker, FakePrompt, FakeStreamingModel
        from langchain_core.documents import Document
        documents = [Document(page_content=f'内部正文 {label}', metadata={
            'knowledge_base_id': label, 'index_status': 'active', 'article': label}) for label in ('A', 'C')]
        turn = StreamingRagTurn(rewriter=FakeRewriter('改写问题'), retriever=FakeRetriever(documents),
                                reranker=FakeReranker(documents), prompt=FakePrompt(),
                                model=FakeStreamingModel(['答案']), history_turns=4, require_authorization=True)
        with TestClient(create_app(self.store, turn, authentication=self.auth)) as client:
            self.login_client(client)
            cid = client.post('/api/conversations').json()['id']
            with self.assertLogs('legal_rag.requests', level='INFO') as captured:
                response = client.post(f'/api/conversations/{cid}/chat', json={'question': '问题'},
                                       headers={'X-Request-ID': 'forged', 'Authorization': 'private-token'})
            rows = [json.loads(record.getMessage()) for record in captured.records]
            self.assertEqual(rows[0]['event'], 'request_started')
            self.assertEqual(rows[-1]['event'], 'request_finished')
            self.assertEqual(rows[-1]['outcome'], 'completed')
            self.assertTrue(all(row['request_id'] == response.headers['X-Request-ID'] for row in rows))
            rag_rows = [row for row in rows if row['event'].startswith('rag_')]
            self.assertGreater(len(rag_rows), 5)
            self.assertTrue(all(row['user_id'] == self.credentials['demo_ab']['user_id'] for row in rag_rows))
            self.assertTrue(all(row['conversation_id'] == cid for row in rag_rows))
            logged = json.dumps(rows, ensure_ascii=False)
            self.assertIn('内部正文 A', logged)
            self.assertNotIn('内部正文 C', logged)
            self.assertNotIn('private-token', logged)
            self.assertEqual(client.get('/logs/requests.jsonl').status_code, 404)
            self.assertEqual(client.get('/api/logs').status_code, 404)

    def client(self):
        from fastapi.testclient import TestClient
        from web_app import create_app
        from test_web_app import SuccessfulTurn
        return TestClient(create_app(self.store, SuccessfulTurn(), authentication=self.auth))

    def login_client(self, client, username='demo_ab'):
        entry = self.credentials[username]
        response = client.post('/api/auth/login', json={'username': username, 'password': entry['password']})
        self.assertEqual(response.status_code, 200)
        client.headers['X-CSRF-Token'] = response.json()['csrf_token']
        return response

    def test_sessions_are_persisted_with_hashed_tokens_and_expire_without_revival(self):
        from psycopg2 import sql
        from login_sessions import PostgresLoginSessions
        session = self.login_session()
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT token_hash,expires_at FROM {}.login_sessions').format(sql.Identifier(self.schema)))
            encoded, expiry = cursor.fetchone()
        self.assertNotEqual(encoded, session.token)
        self.assertEqual(expiry, self.clock() + timedelta(days=3))
        recreated = PostgresLoginSessions(self.accounts, now=self.clock, schema=self.schema, **self.settings)
        self.assertEqual(recreated.lookup(session.token).user.id, self.credentials['demo_ab']['user_id'])
        self.clock.advance(days=3)
        self.assertIsNone(recreated.lookup(session.token, renew=True))
        self.assertIsNone(recreated.lookup(encoded))

    def test_activity_renews_three_days_and_logout_revokes_only_one_browser(self):
        first, second = self.login_session(), self.login_session()
        self.clock.advance(days=2)
        renewed = self.auth.lookup(first.token, renew=True)
        self.assertEqual(renewed.expires_at, self.clock() + timedelta(days=3))
        self.assertTrue(self.auth.logout(first.token))
        self.assertIsNone(self.auth.lookup(first.token))
        self.assertIsNotNone(self.auth.lookup(second.token))
        self.clock.advance(days=1)
        self.assertIsNone(self.auth.lookup(second.token))

    def test_disabled_account_invalidates_existing_session(self):
        from psycopg2 import sql
        session = self.login_session()
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('UPDATE {}.users SET is_active=false WHERE id=%s').format(sql.Identifier(self.schema)), (session.user.id,))
        self.assertIsNone(self.auth.lookup(session.token, renew=True))
        self.assertIsNone(self.login_session())

    def test_anonymous_cookie_cannot_access_any_conversation_api(self):
        client = self.client()
        client.cookies.set('legal_rag_session', 'old-anonymous')
        for method, path, payload in [('GET', '/api/conversations', None), ('POST', '/api/conversations', None),
                                     ('GET', f'/api/conversations/{self.old.id}/messages', None),
                                     ('PATCH', f'/api/conversations/{self.old.id}', {'title': 'x'}),
                                     ('DELETE', f'/api/conversations/{self.old.id}', None),
                                     ('POST', f'/api/conversations/{self.old.id}/chat', {'question': '问题'})]:
            with self.subTest(method=method):
                self.assertEqual(client.request(method, path, json=payload).status_code, 401)

    def test_same_account_shares_history_across_browsers_other_account_cannot_read(self):
        from test_web_app import parse_events
        first, second, colleague = self.client(), self.client(), self.client()
        self.login_client(first)
        self.login_client(second)
        self.login_client(colleague, 'demo_c')
        self.assertEqual(first.get('/api/conversations').json()['conversations'], [])
        self.assertEqual(first.get(f'/api/conversations/{self.old.id}/messages').status_code, 404)
        item = first.post('/api/conversations').json()['id']
        response = first.post(f'/api/conversations/{item}/chat', json={'question': '新问题'})
        self.assertEqual(parse_events(response)[-1][0], 'done')
        self.assertEqual(second.get(f'/api/conversations/{item}/messages').json()['messages'], [{'role': 'user', 'content': '新问题'}, {'role': 'assistant', 'content': '完整'}])
        for method, suffix, body in [('GET', '/messages', None), ('POST', '/chat', {'question': 'x'}), ('PATCH', '', {'title': 'x'}), ('DELETE', '', None)]:
            self.assertEqual(colleague.request(method, f'/api/conversations/{item}{suffix}', json=body).status_code, 404)
        self.assertEqual(second.patch(f'/api/conversations/{item}', json={'title': '账号咨询'}).status_code, 200)
        self.assertEqual(first.get('/api/conversations').json()['conversations'][0]['title'], '账号咨询')

    def test_cookie_me_and_legal_requests_renew_but_static_and_rejections_do_not(self):
        client = self.client()
        login = self.login_client(client)
        cookie = login.headers['set-cookie']
        self.assertIn('HttpOnly', cookie)
        self.assertIn('Max-Age=259200', cookie)
        self.assertIn('SameSite=lax', cookie)
        token = client.cookies.get('legal_rag_auth')
        expiry = self.auth.lookup(token).expires_at
        self.clock.advance(days=1)
        self.assertNotIn('set-cookie', client.get('/').headers)
        self.assertNotIn('set-cookie', client.get('/static/app.js').headers)
        self.assertEqual(self.auth.lookup(token).expires_at, expiry)
        self.assertEqual(client.get('/api/conversations/not-owned/messages').status_code, 404)
        self.assertEqual(client.post('/api/conversations', headers={'X-CSRF-Token': 'wrong'}).status_code, 403)
        self.assertEqual(self.auth.lookup(token).expires_at, expiry)
        me = client.get('/api/auth/me')
        self.assertEqual(me.json()['expires_at'], (self.clock() + timedelta(days=3)).isoformat())
        self.assertIn('Max-Age=259200', me.headers['set-cookie'])
        self.clock.advance(days=1)
        self.assertIn('set-cookie', client.post('/api/conversations').headers)
        self.assertEqual(self.auth.lookup(token).expires_at, self.clock() + timedelta(days=3))

    def test_expired_or_logged_out_browser_is_rejected_and_cookie_cleared(self):
        client = self.client()
        self.login_client(client)
        token = client.cookies.get('legal_rag_auth')
        self.assertEqual(client.post('/api/auth/logout').status_code, 204)
        self.assertIsNone(self.auth.lookup(token))
        self.assertEqual(client.get('/api/auth/me').status_code, 401)
        self.login_client(client)
        self.clock.advance(days=3)
        response = client.get('/api/auth/me')
        self.assertEqual(response.status_code, 401)
        self.assertIn('Max-Age=0', response.headers['set-cookie'])

    def test_wrong_password_and_invalid_payload_do_not_echo_secrets(self):
        client = self.client()
        secret = 'private-input-password'
        response = client.post('/api/auth/login', json={'username': 'demo_ab', 'password': secret})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn(secret, response.text)
        response = client.post('/api/auth/login', json={'username': ['bad'], 'password': secret})
        self.assertEqual(response.status_code, 400)
        self.assertNotIn(secret, response.text)

    def test_cross_origin_and_missing_csrf_are_rejected(self):
        client = self.client()
        entry = self.credentials['demo_ab']
        payload = {'username': 'demo_ab', 'password': entry['password']}
        self.assertEqual(client.post('/api/auth/login', json=payload, headers={'Origin': 'https://foreign.example'}).status_code, 403)
        self.login_client(client)
        self.assertEqual(client.post('/api/conversations', headers={'Origin': 'https://foreign.example'}).status_code, 403)
        self.assertEqual(client.post('/api/conversations', headers={'Sec-Fetch-Site': 'cross-site'}).status_code, 403)
        client.headers.pop('X-CSRF-Token')
        self.assertEqual(client.post('/api/conversations').status_code, 403)

    def test_login_attempts_are_rate_limited(self):
        client = self.client()
        for _ in range(20):
            self.assertEqual(client.post('/api/auth/login', json={'username': 'demo_ab', 'password': 'wrong-password'}).status_code, 401)
        self.assertEqual(client.post('/api/auth/login', json={'username': 'demo_ab', 'password': 'wrong-password'}).status_code, 429)

    def test_schema_migration_preserves_legacy_ownership_and_repeat(self):
        from prepare_web_login import prepare_login
        from psycopg2 import sql
        self.assertEqual(prepare_login(self.db, apply=True, schema=self.schema)['action'], 'already_ready')
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT session_id,user_id FROM {}.conversations WHERE id=%s').format(sql.Identifier(self.schema)), (self.old.id,))
            self.assertEqual(cursor.fetchone(), ('old-anonymous', None))
        item = self.store.create_conversation(self.credentials['demo_ab']['user_id'])
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT session_id,user_id FROM {}.conversations WHERE id=%s').format(sql.Identifier(self.schema)), (item.id,))
            self.assertEqual(cursor.fetchone(), (None, self.credentials['demo_ab']['user_id']))


if __name__ == '__main__':
    unittest.main()
