"""Exercise the web store against an isolated, real PostgreSQL schema."""

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from web_storage import SQLiteConversationStore


class PostgreSQLStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS') != '1':
            raise unittest.SkipTest('Set RUN_POSTGRES_TESTS=1 for PostgreSQL integration tests')

    def setUp(self):
        import psycopg2
        from migrate_sqlite_to_postgres import migrate
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / 'source.sqlite3'
        source = SQLiteConversationStore(self.source)
        source.ensure_session('owner')
        self.old = source.create_conversation('owner')
        source.save_complete_conversation_turn('owner', self.old.id, '旧问题', '旧回答')
        self.settings = dict(host='127.0.0.1', port=5433, dbname='legal_rag', user='legal_rag',
                             password=Path('.postgres-password').read_text().strip(), connect_timeout=5)
        self.db = psycopg2.connect(**self.settings)
        self.addCleanup(self.db.close)
        self.schema = 'storage_test_' + uuid4().hex
        migrate(self.source, self.db, apply=True, schema=self.schema)
        self.addCleanup(self.cleanup_schema)
        from prepare_source_history import prepare_sources
        prepare_sources(self.db, apply=True, schema=self.schema)
        from postgres_storage import PostgresConversationStore
        self.store = PostgresConversationStore(schema=self.schema, **self.settings)

    def cleanup_schema(self):
        from psycopg2 import sql
        self.db.rollback()
        if not self.schema.startswith('storage_test_') or len(self.schema) != 45:
            raise ValueError('Refusing to remove a non-test schema')
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def test_imported_history_and_ownership_survive_store_recreation(self):
        from postgres_storage import PostgresConversationStore
        store = PostgresConversationStore(schema=self.schema, **self.settings)
        self.assertEqual(store.list_conversations('owner')[0].id, self.old.id)
        self.assertEqual(store.load_conversation_messages('owner', self.old.id), [('user', '旧问题'), ('assistant', '旧回答')])
        self.assertIsNone(store.load_conversation_messages('other', self.old.id))
        self.assertFalse(store.conversation_belongs_to('other', self.old.id))
        self.assertIsNone(store.save_complete_conversation_turn('other', self.old.id, 'x', 'y'))

    def test_background_task_migration_completion_and_rollback(self):
        from prepare_generation_tasks import prepare_tasks
        from generation_tasks import TaskStore
        self.assertFalse(prepare_tasks(self.db, schema=self.schema)['ready'])
        self.assertEqual(prepare_tasks(self.db, apply=True, schema=self.schema)['action'], 'prepared')
        self.assertEqual(prepare_tasks(self.db, apply=True, schema=self.schema)['action'], 'already_ready')
        tasks = TaskStore(self.store)
        task, created = tasks.create('owner', self.old.id, '新问题', 'key', 'request')
        self.assertTrue(created)
        self.assertIsNone(tasks.get('other', self.old.id, task['id']))
        self.assertEqual(tasks.create('owner', self.old.id, '新问题', 'key', 'request')[0]['id'], task['id'])
        with self.assertRaises(ValueError): tasks.create('owner', self.old.id, '另一个', 'key2', 'request')
        self.assertTrue(tasks.complete(task['id'], '新回答', [{'content': '新来源'}]))
        self.assertFalse(tasks.complete(task['id'], '重复', []))
        self.assertEqual(self.store.load_conversation_messages('owner', self.old.id)[-2:], [('user', '新问题'), ('assistant', '新回答')])
        self.assertEqual(self.store.load_conversation_history('owner', self.old.id)[-1]['sources'][0]['content'], '新来源')

    def test_conversation_lifecycle_and_custom_title(self):
        session = self.store.ensure_session(None)
        self.assertEqual(self.store.ensure_session(session), session)
        item = self.store.create_conversation(session)
        self.assertEqual(item.title, '新对话')
        self.assertIsNone(self.store.rename_conversation('other', item.id, '不允许'))
        self.assertFalse(self.store.delete_conversation('other', item.id))
        self.assertEqual(self.store.rename_conversation(session, item.id, ' 我的标题 ').title, '我的标题')
        saved = self.store.save_complete_conversation_turn(session, item.id, '标题不应覆盖', '回答')
        self.assertEqual(saved.title, '我的标题')
        self.assertTrue(self.store.delete_conversation(session, item.id))
        self.assertIsNone(self.store.load_conversation_messages(session, item.id))
        from psycopg2 import sql
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT count(*) FROM {}.conversation_messages WHERE conversation_id=%s').format(sql.Identifier(self.schema)), (item.id,))
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_validation_auto_title_and_legacy_messages(self):
        item = self.store.create_conversation('owner')
        for title in (' ', 'x' * 81):
            with self.assertRaises(ValueError):
                self.store.rename_conversation('owner', item.id, title)
        with self.assertRaises(ValueError):
            self.store.save_complete_conversation_turn('owner', item.id, '问题', ' ')
        self.assertEqual(self.store.load_conversation_messages('owner', item.id), [])
        self.assertEqual(self.store.save_complete_conversation_turn('owner', item.id, '  新  问题  ', '回答').title, '新 问题')
        self.store.save_complete_turn('owner', '旧接口问题', '旧接口回答')
        self.assertEqual(self.store.load_messages('owner'), [('user', '旧接口问题'), ('assistant', '旧接口回答')])

    def test_failed_second_insert_rolls_back_question_and_title(self):
        from psycopg2 import sql
        import psycopg2
        item = self.store.create_conversation('owner')
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL("ALTER TABLE {}.conversation_messages ADD CHECK(content <> 'reject-answer')").format(sql.Identifier(self.schema)))
        with self.assertRaises(psycopg2.IntegrityError):
            self.store.save_complete_conversation_turn('owner', item.id, '问题', 'reject-answer')
        self.assertEqual(self.store.load_conversation_messages('owner', item.id), [])
        self.assertEqual(next(c for c in self.store.list_conversations('owner') if c.id == item.id).title, '新对话')

    def test_concurrent_saves_keep_turns_paired_and_ordered(self):
        item = self.store.create_conversation('owner')
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda i: self.store.save_complete_conversation_turn('owner', item.id, f'q{i}', f'a{i}'), range(8)))
        messages = self.store.load_conversation_messages('owner', item.id)
        self.assertEqual(len(messages), 16)
        pairs = [(messages[i], messages[i+1]) for i in range(0, len(messages), 2)]
        self.assertCountEqual(pairs, [(('user', f'q{i}'), ('assistant', f'a{i}')) for i in range(8)])

    def test_web_routes_use_postgres_and_isolate_cookie_owners(self):
        from fastapi.testclient import TestClient
        from web_app import create_app
        from test_web_app import SuccessfulTurn, FailingTurn, parse_events
        with TestClient(create_app(self.store, SuccessfulTurn())) as first, TestClient(create_app(self.store, SuccessfulTurn())) as second:
            first.cookies.set('legal_rag_session', 'owner')
            self.assertEqual(first.get(f'/api/conversations/{self.old.id}/messages').json()['messages'][0]['content'], '旧问题')
            self.assertEqual(second.get(f'/api/conversations/{self.old.id}/messages').status_code, 404)
            self.assertEqual(second.patch(f'/api/conversations/{self.old.id}', json={'title': 'wrong'}).status_code, 404)
            self.assertEqual(second.delete(f'/api/conversations/{self.old.id}').status_code, 404)
            self.assertEqual(second.post(f'/api/conversations/{self.old.id}/chat', json={'question': 'wrong'}).status_code, 404)
            item = first.post('/api/conversations').json()['id']
            response = first.post(f'/api/conversations/{item}/chat', json={'question': '新问题'})
            self.assertEqual(parse_events(response)[-1][0], 'done')
            self.assertEqual(self.store.load_conversation_messages('owner', item), [('user', '新问题'), ('assistant', '完整')])
        with TestClient(create_app(self.store, FailingTurn())) as client:
            client.cookies.set('legal_rag_session', 'owner')
            item = client.post('/api/conversations').json()['id']
            self.assertEqual(parse_events(client.post(f'/api/conversations/{item}/chat', json={'question': '失败问题'}))[-1][0], 'error')
            self.assertEqual(self.store.load_conversation_messages('owner', item), [])

    def test_config_selects_postgres_without_creating_sqlite(self):
        from web_app import create_default_app
        from fastapi.testclient import TestClient
        from test_web_app import SuccessfulTurn
        from prepare_demo_accounts import prepare_accounts
        from prepare_web_login import prepare_login
        import json
        credentials_path = Path(self.directory.name) / 'accounts.json'
        prepare_accounts(self.db, credentials_path, apply=True, schema=self.schema)
        prepare_login(self.db, apply=True, schema=self.schema)
        from prepare_generation_tasks import prepare_tasks
        prepare_tasks(self.db, apply=True, schema=self.schema)
        from prepare_web_recovery import prepare_recovery
        prepare_recovery(self.settings, apply=True, schema=self.schema)
        from prepare_account_memory import prepare_memory
        prepare_memory(self.db, apply=True, schema=self.schema)
        from prepare_conversation_context import prepare_context
        prepare_context(self.db, apply=True, schema=self.schema)
        from prepare_memory_reflection import prepare_reflection
        prepare_reflection(self.db,apply=True,schema=self.schema)
        from prepare_memory_tools import prepare_memory_tools
        prepare_memory_tools(self.db,apply=True,schema=self.schema)
        def cleanup_graph():
            from psycopg2 import sql
            assert self.schema.startswith('storage_test_') and len(self.schema) == 45
            with self.db, self.db.cursor() as cursor:
                cursor.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema + '_graph')))
        self.addCleanup(cleanup_graph)
        credential = json.loads(credentials_path.read_text())['accounts'][0]
        sqlite_path = Path(self.directory.name) / 'must-not-create.sqlite3'
        env = {'WEB_STORAGE_BACKEND': 'postgres', 'WEB_POSTGRES_SCHEMA': self.schema,
               'WEB_DATABASE_PATH': str(sqlite_path),'MODEL_MEMORY_TOOLS_ENABLED':'true'}
        with patch.dict(os.environ, env), patch('web_app.create_web_rag_turn', return_value=SuccessfulTurn()) as factory:
            with TestClient(create_default_app()) as client:
                self.assertTrue(factory.call_args.kwargs['model_memory'])
                self.assertIsNotNone(client.app.state.memory_store.facts)
                self.assertTrue(client.app.state.context_store.semantic)
                self.assertEqual(client.get('/api/conversations').status_code, 401)
                login = client.post('/api/auth/login', json={'username': credential['username'], 'password': credential['password']})
                self.assertEqual(login.status_code, 200)
                client.headers['X-CSRF-Token'] = login.json()['csrf_token']
                self.assertEqual(client.get('/api/conversations').json()['conversations'], [])
                created = client.post('/api/conversations').json()['id']
                self.assertEqual(client.get('/api/conversations').json()['conversations'][0]['id'], created)
        self.assertFalse(sqlite_path.exists())

    def test_deleted_conversation_cannot_report_a_saved_answer(self):
        from fastapi.testclient import TestClient
        from web_app import create_app
        from test_web_app import parse_events
        store = self.store
        item = store.create_conversation('owner')

        class DeletedDuringAnswer:
            def stream(self, question, messages):
                yield {'event': 'delta', 'data': {'text': '回答'}}
                store.delete_conversation('owner', item.id)
                yield {'event': 'done', 'data': {'answer': '回答'}}

        with TestClient(create_app(store, DeletedDuringAnswer())) as client:
            client.cookies.set('legal_rag_session', 'owner')
            response = client.post(f'/api/conversations/{item.id}/chat', json={'question': '问题'})
            self.assertEqual(parse_events(response)[-1][0], 'error')
            self.assertIsNone(store.load_conversation_messages('owner', item.id))

    def test_missing_schema_fails_without_sqlite_fallback(self):
        from web_app import create_default_app
        sqlite_path = Path(self.directory.name) / 'must-not-create.sqlite3'
        env = {'WEB_STORAGE_BACKEND': 'postgres', 'WEB_POSTGRES_SCHEMA': 'absent_' + uuid4().hex,
               'WEB_DATABASE_PATH': str(sqlite_path)}
        with patch.dict(os.environ, env), self.assertRaisesRegex(RuntimeError, 'migration'):
            create_default_app()
        self.assertFalse(sqlite_path.exists())


class StorageConfigurationTests(unittest.TestCase):
    def test_unknown_backend_is_rejected(self):
        from web_app import create_default_app
        with patch.dict(os.environ, {'WEB_STORAGE_BACKEND': 'postgrez'}), self.assertRaisesRegex(ValueError, 'WEB_STORAGE_BACKEND'):
            create_default_app()

    def test_explicit_sqlite_path_overrides_backend_for_local_tools(self):
        from web_app import create_default_app
        from fastapi.testclient import TestClient
        from test_web_app import SuccessfulTurn
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'WEB_STORAGE_BACKEND': 'postgres'}), patch('web_app.create_web_rag_turn', return_value=SuccessfulTurn()):
            path = Path(directory) / 'explicit.sqlite3'
            with TestClient(create_default_app(database_path=path)) as client:
                self.assertEqual(client.get('/api/conversations').json()['conversations'], [])
            self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
