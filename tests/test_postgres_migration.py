import hashlib
import os
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from web_storage import SQLiteConversationStore


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS') != '1':
            raise unittest.SkipTest('Set RUN_POSTGRES_TESTS=1 for local PostgreSQL integration tests')

    def setUp(self):
        import psycopg2
        from psycopg2 import sql
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / 'source.sqlite3'
        store = SQLiteConversationStore(self.source)
        session = store.ensure_session('fixture-owner')
        self.conversation = store.create_conversation(session)
        store.save_complete_conversation_turn(session, self.conversation.id, 'question', 'answer')
        store.save_complete_turn(session, 'old question', 'old answer')
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.db = psycopg2.connect(host='127.0.0.1', port=5433, dbname='legal_rag', user='legal_rag',
            password=Path('.postgres-password').read_text().strip(), connect_timeout=5)
        self.addCleanup(self.db.close)
        self.schema = 'migration_test_' + uuid4().hex
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(self.schema)))
        self.addCleanup(self.cleanup_schema)

    def cleanup_schema(self):
        from psycopg2 import sql
        self.db.rollback()
        if not self.schema.startswith('migration_test_') or len(self.schema) != 47:
            raise ValueError('Refusing to remove a non-test schema')
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def test_check_is_read_only(self):
        from migrate_sqlite_to_postgres import migrate
        report = migrate(self.source, self.db, schema=self.schema)
        self.assertEqual(report['action'], 'check_only')
        self.assertEqual(report['source_counts']['conversation_messages'], 2)
        with self.db.cursor() as cursor:
            cursor.execute('SELECT count(*) FROM information_schema.tables WHERE table_schema=%s', (self.schema,))
            self.assertEqual(cursor.fetchone()[0], 0)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.digest)

    def test_import_preserves_records_and_repeat_does_not_duplicate(self):
        from migrate_sqlite_to_postgres import migrate
        from psycopg2 import sql
        report = migrate(self.source, self.db, apply=True, schema=self.schema)
        self.assertEqual(report['action'], 'imported')
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT role,content FROM {}.conversation_messages ORDER BY ordinal').format(sql.Identifier(self.schema)))
            self.assertEqual(cursor.fetchall(), [('user', 'question'), ('assistant', 'answer')])
            cursor.execute(sql.SQL('SELECT id,session_id FROM {}.conversations').format(sql.Identifier(self.schema)))
            self.assertEqual(cursor.fetchall(), [(self.conversation.id, 'fixture-owner')])
            cursor.execute(sql.SQL('SELECT content FROM {}.messages ORDER BY ordinal').format(sql.Identifier(self.schema)))
            self.assertEqual(cursor.fetchall(), [('old question',), ('old answer',)])
        self.assertEqual(migrate(self.source, self.db, apply=True, schema=self.schema)['action'], 'already_matches')
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.digest)

    def test_different_target_is_not_overwritten(self):
        from migrate_sqlite_to_postgres import migrate
        from psycopg2 import sql
        migrate(self.source, self.db, apply=True, schema=self.schema)
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL("UPDATE {}.conversation_messages SET content='existing' WHERE ordinal=1").format(sql.Identifier(self.schema)))
        with self.assertRaisesRegex(ValueError, 'different'):
            migrate(self.source, self.db, apply=True, schema=self.schema)
        self.db.rollback()
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT content FROM {}.conversation_messages WHERE ordinal=1').format(sql.Identifier(self.schema)))
            self.assertEqual(cursor.fetchone()[0], 'existing')

    def test_failed_import_rolls_back_tables_and_prior_inserts(self):
        import sqlite3
        import psycopg2
        from migrate_sqlite_to_postgres import migrate
        # SQLite accepts this ordinal, but PostgreSQL INTEGER rejects it after
        # sessions have already been inserted. No partial import may survive.
        with closing(sqlite3.connect(self.source)) as source:
            with source:
                source.execute('UPDATE messages SET ordinal=ordinal+1099511627776')
        digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with self.assertRaises(psycopg2.errors.NumericValueOutOfRange):
            migrate(self.source, self.db, apply=True, schema=self.schema)
        with self.db.cursor() as cursor:
            cursor.execute('SELECT count(*) FROM information_schema.tables WHERE table_schema=%s', (self.schema,))
            self.assertEqual(cursor.fetchone()[0], 0)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), digest)

    def test_target_duplicate_rows_are_not_treated_as_exact_match(self):
        from migrate_sqlite_to_postgres import migrate
        from psycopg2 import sql
        migrate(self.source, self.db, apply=True, schema=self.schema)
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('ALTER TABLE {}.conversation_messages DROP CONSTRAINT conversation_messages_pkey').format(sql.Identifier(self.schema)))
            cursor.execute(sql.SQL('INSERT INTO {}.conversation_messages SELECT * FROM {}.conversation_messages WHERE ordinal=1').format(sql.Identifier(self.schema), sql.Identifier(self.schema)))
        with self.assertRaisesRegex(ValueError, 'different'):
            migrate(self.source, self.db, apply=True, schema=self.schema)
        self.db.rollback()
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT count(*) FROM {}.conversation_messages').format(sql.Identifier(self.schema)))
            self.assertEqual(cursor.fetchone()[0], 3)


if __name__ == '__main__':
    unittest.main()
