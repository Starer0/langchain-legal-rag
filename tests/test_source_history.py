import json
import os
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from web_storage import SQLiteConversationStore


SOURCES = [{'law_name': '劳动合同法', 'article': '第二十条', 'pages': [4],
            'content': '当时引用的原文', 'version': '2012', 'knowledge_base_id': 'B'}]


class SourceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'history.sqlite3'
        self.store = SQLiteConversationStore(self.path)
        self.store.ensure_session('owner')
        self.cid = self.store.create_conversation('owner').id

    def test_snapshot_survives_recreation_and_is_attached_to_only_its_answer(self):
        sources = json.loads(json.dumps(SOURCES))
        self.store.save_complete_conversation_turn('owner', self.cid, '问题1', '回答1', sources=sources)
        sources[0]['content'] = '后来修改'
        self.store.save_complete_conversation_turn('owner', self.cid, '问题2', '回答2')
        history = SQLiteConversationStore(self.path).load_conversation_history('owner', self.cid)
        self.assertEqual(history[1]['sources'], SOURCES)
        self.assertNotIn('sources', history[0])
        self.assertNotIn('sources', history[3])
        self.assertEqual(self.store.load_conversation_messages('owner', self.cid),
                         [('user', '问题1'), ('assistant', '回答1'), ('user', '问题2'), ('assistant', '回答2')])
        self.assertIsNone(self.store.load_conversation_history('other', self.cid))

    def test_snapshot_failure_rolls_back_the_entire_turn(self):
        with self.store._connection() as db:
            db.execute("CREATE TRIGGER reject_source BEFORE INSERT ON conversation_sources BEGIN SELECT RAISE(ABORT, 'reject'); END")
        with self.assertRaises(Exception):
            self.store.save_complete_conversation_turn('owner', self.cid, '问题', '回答', sources=SOURCES)
        self.assertEqual(self.store.load_conversation_messages('owner', self.cid), [])
        self.assertEqual(self.store.list_conversations('owner')[0].title, '新对话')

    def test_delete_cascades_sources_and_other_owner_cannot_write(self):
        self.assertIsNone(self.store.save_complete_conversation_turn('other', self.cid, '问题', '回答', sources=SOURCES))
        self.store.save_complete_conversation_turn('owner', self.cid, '问题', '回答', sources=SOURCES)
        self.store.delete_conversation('owner', self.cid)
        with self.store._connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM conversation_sources').fetchone()[0], 0)

    def test_legacy_sqlite_migration_refuses_to_silently_drop_saved_sources(self):
        from migrate_sqlite_to_postgres import read_source
        self.store.save_complete_conversation_turn('owner', self.cid, '问题', '回答', sources=SOURCES)
        with self.assertRaisesRegex(ValueError, 'source snapshots'):
            read_source(self.path)

    def test_invalid_or_unrelated_source_fields_do_not_corrupt_history(self):
        for invalid in ({'content': 'wrong shape'}, ['wrong shape'], [{'rerank_score': float('nan')}]):
            with self.assertRaises(ValueError):
                self.store.save_complete_conversation_turn('owner', self.cid, '问题', '回答', sources=invalid)
        self.assertEqual(self.store.load_conversation_messages('owner', self.cid), [])
        source = {**SOURCES[0], 'private_token': 'must not persist'}
        self.store.save_complete_conversation_turn('owner', self.cid, '问题', '回答', sources=[source])
        self.assertEqual(self.store.load_conversation_history('owner', self.cid)[1]['sources'], SOURCES)

    def test_api_restores_sources_and_never_injects_them_into_followup_context(self):
        from web_app import create_app
        class Turn:
            histories = []
            def stream(inner, question, messages, **options):
                inner.histories.append(messages)
                yield {'event': 'delta', 'data': {'text': '回答'}}
                yield {'event': 'done', 'data': {'answer': '回答', 'sources': SOURCES}}
        turn = Turn()
        with TestClient(create_app(self.store, turn)) as client:
            cid = client.post('/api/conversations').json()['id']
            url = f'/api/conversations/{cid}'
            self.assertIn('event: done', client.post(url+'/chat', json={'question': '问题'}).text)
            self.assertEqual(client.get(url+'/messages').json()['messages'][1]['sources'], SOURCES)
            client.post(url+'/chat', json={'question': '追问'})
            self.assertEqual([m.content for m in turn.histories[1]], ['问题', '回答'])
            with TestClient(create_app(self.store, turn)) as other:
                self.assertEqual(other.get(url+'/messages').status_code, 404)


@unittest.skipUnless(os.getenv('RUN_POSTGRES_TESTS') == '1', 'Real PostgreSQL tests are opt-in')
class PostgresSourceHistoryTests(unittest.TestCase):
    def setUp(self):
        import psycopg2
        from migrate_sqlite_to_postgres import migrate
        from prepare_source_history import prepare_sources
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / 'source.sqlite3'
        sqlite = SQLiteConversationStore(path)
        sqlite.ensure_session('owner')
        self.cid = sqlite.create_conversation('owner').id
        sqlite.save_complete_conversation_turn('owner', self.cid, '旧问题', '旧回答')
        self.settings = dict(host='127.0.0.1', port=5433, dbname='legal_rag', user='legal_rag',
                             password=Path('.postgres-password').read_text().strip())
        self.db = psycopg2.connect(**self.settings)
        self.addCleanup(self.db.close)
        self.schema = 'source_test_' + uuid4().hex
        migrate(path, self.db, apply=True, schema=self.schema)
        self.addCleanup(self.cleanup_schema)
        self.assertFalse(prepare_sources(self.db, schema=self.schema)['ready'])
        self.assertEqual(prepare_sources(self.db, schema=self.schema, apply=True)['action'], 'prepared')
        from postgres_storage import PostgresConversationStore
        self.store = PostgresConversationStore(schema=self.schema, **self.settings)

    def cleanup_schema(self):
        from psycopg2 import sql
        self.db.rollback()
        assert self.schema.startswith('source_test_') and len(self.schema) == 44
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def test_additive_idempotent_migration_and_history_ownership(self):
        from prepare_source_history import prepare_sources
        self.assertEqual(prepare_sources(self.db, schema=self.schema, apply=True)['action'], 'already_ready')
        self.assertEqual(self.store.load_conversation_messages('owner', self.cid), [('user', '旧问题'), ('assistant', '旧回答')])
        self.store.save_complete_conversation_turn('owner', self.cid, '新问题', '新回答', sources=SOURCES)
        history = self.store.load_conversation_history('owner', self.cid)
        self.assertNotIn('sources', history[1])
        self.assertEqual(history[3]['sources'], SOURCES)
        self.assertIsNone(self.store.load_conversation_history('other', self.cid))
        self.assertIsNone(self.store.save_complete_conversation_turn('other', self.cid, 'x', 'y', sources=SOURCES))
        self.store.delete_conversation('owner', self.cid)
        from psycopg2 import sql
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT count(*) FROM {}').format(sql.Identifier(self.schema, 'conversation_sources')))
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_failed_source_insert_rolls_back_question_answer_and_title(self):
        import psycopg2
        from psycopg2 import sql
        cid = self.store.create_conversation('owner').id
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('ALTER TABLE {} ADD CHECK (false)').format(sql.Identifier(self.schema, 'conversation_sources')))
        with self.assertRaises(psycopg2.IntegrityError):
            self.store.save_complete_conversation_turn('owner', cid, '问题', '回答', sources=SOURCES)
        self.assertEqual(self.store.load_conversation_messages('owner', cid), [])
        self.assertEqual(next(c for c in self.store.list_conversations('owner') if c.id == cid).title, '新对话')

    def test_unmarked_existing_table_is_not_adopted_or_overwritten(self):
        from psycopg2 import sql
        from prepare_source_history import MIGRATION, prepare_sources
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('DELETE FROM {} WHERE name=%s').format(sql.Identifier(self.schema, 'schema_migrations')), (MIGRATION,))
        with self.assertRaisesRegex(ValueError, 'Unmarked'):
            prepare_sources(self.db, schema=self.schema, apply=True)
        self.assertFalse(prepare_sources(self.db, schema=self.schema)['ready'])
