import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from uuid import uuid4

from web_storage import SQLiteConversationStore


class PasswordTests(unittest.TestCase):
    def test_short_demo_password_requires_explicit_creation_policy(self):
        from password_security import hash_password, verify_password
        with self.assertRaises(ValueError):
            hash_password('web123')
        encoded = hash_password('web123', minimum_length=6)
        self.assertTrue(verify_password(encoded, 'web123'))
        self.assertFalse(verify_password(encoded, 'wrong123'))
        with self.assertRaises(ValueError):
            hash_password('short', minimum_length=6)

    def test_random_salts_keep_same_password_hashes_different_and_verifiable(self):
        from password_security import hash_password, verify_password
        first = hash_password('fixture-password-123')
        second = hash_password('fixture-password-123')
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith('$argon2id$'))
        self.assertTrue(verify_password(first, 'fixture-password-123'))
        self.assertFalse(verify_password(first, 'different-password'))

    def test_invalid_hash_and_oversized_or_non_text_password_fail_closed(self):
        from password_security import hash_password, verify_password
        encoded = hash_password('fixture-password-123')
        self.assertFalse(verify_password('broken-hash', 'fixture-password-123'))
        self.assertFalse(verify_password(encoded, 'x' * 1025))
        self.assertFalse(verify_password(encoded, None))
        self.assertFalse(verify_password(encoded, '\ud800'))

    def test_creation_validates_password_length_in_utf8_bytes(self):
        from password_security import hash_password
        for password in ('short', 'x' * 1025, '中' * 342, None):
            with self.subTest(password_type=type(password).__name__), self.assertRaises(ValueError):
                hash_password(password)

    def test_password_whitespace_and_unicode_are_preserved(self):
        from password_security import hash_password, verify_password
        password = '  我的示例密码12345  '
        encoded = hash_password(password)
        self.assertTrue(verify_password(encoded, password))
        self.assertFalse(verify_password(encoded, password.strip()))


class AccountDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.getenv('RUN_POSTGRES_TESTS') != '1':
            raise unittest.SkipTest('Set RUN_POSTGRES_TESTS=1 for PostgreSQL account tests')

    def setUp(self):
        import psycopg2
        from migrate_sqlite_to_postgres import migrate, read_target
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        source_path = Path(self.directory.name) / 'fixture.sqlite3'
        source = SQLiteConversationStore(source_path)
        source.ensure_session('anonymous-owner')
        conversation = source.create_conversation('anonymous-owner')
        source.save_complete_conversation_turn('anonymous-owner', conversation.id, '原来的问题', '原来的回答')
        self.settings = dict(host='127.0.0.1', port=5433, dbname='legal_rag', user='legal_rag',
                             password=Path('.postgres-password').read_text().strip(), connect_timeout=5)
        self.db = psycopg2.connect(**self.settings)
        self.addCleanup(self.db.close)
        self.schema = 'account_test_' + uuid4().hex
        migrate(source_path, self.db, apply=True, schema=self.schema)
        self.addCleanup(self.cleanup_schema)
        self.credentials_path = Path(self.directory.name) / 'credentials.json'
        with self.db.cursor() as cursor:
            self.original_chats = read_target(cursor, self.schema)

    def cleanup_schema(self):
        from psycopg2 import sql
        self.db.rollback()
        if not self.schema.startswith('account_test_') or len(self.schema) != 45:
            raise ValueError('Refusing to remove a non-test schema')
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def prepare(self):
        from prepare_demo_accounts import prepare_accounts
        return prepare_accounts(self.db, self.credentials_path, apply=True, schema=self.schema)

    def store(self):
        from account_storage import PostgresAccountStore
        return PostgresAccountStore(schema=self.schema, **self.settings)

    def credentials(self):
        return json.loads(self.credentials_path.read_text(encoding='utf8'))['accounts']

    def raw_users(self):
        from psycopg2 import sql
        with self.db.cursor() as cursor:
            cursor.execute(sql.SQL('SELECT id,username,password_hash,level_id,is_active,created_at FROM {}.users ORDER BY username').format(sql.Identifier(self.schema)))
            return cursor.fetchall()

    def test_default_check_creates_neither_tables_nor_credentials(self):
        from prepare_demo_accounts import prepare_accounts
        report = prepare_accounts(self.db, self.credentials_path, schema=self.schema)
        self.assertEqual(report['action'], 'check_only')
        self.assertFalse(report['ready'])
        self.assertFalse(self.credentials_path.exists())
        with self.db.cursor() as cursor:
            cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema=%s', (self.schema,))
            self.assertEqual({row[0] for row in cursor.fetchall()}, {'sessions', 'messages', 'schema_migrations', 'conversations', 'conversation_messages'})

    def test_prepare_persists_real_accounts_without_changing_anonymous_chats(self):
        from migrate_sqlite_to_postgres import read_target
        self.assertEqual(self.prepare()['action'], 'prepared')
        store = self.store()
        self.assertEqual([user.username for user in store.list_accounts()], ['demo_ab', 'demo_abc', 'demo_c'])
        for credential in self.credentials():
            user = store.authenticate(credential['username'], credential['password'])
            self.assertEqual(user.id, credential['user_id'])
            self.assertEqual(user.level_id, credential['level_id'])
            self.assertTrue(user.is_active)
            self.assertNotIn(credential['password'], repr(user))
            self.assertFalse(hasattr(user, 'password_hash'))
            self.assertEqual(self.store().get_account(user.id), user)
        passwords = {entry['password'] for entry in self.credentials()}
        self.assertTrue(all(row[2] not in passwords and row[2].startswith('$argon2id$') for row in self.raw_users()))
        with self.db.cursor() as cursor:
            current = read_target(cursor, self.schema)
        for table in ('sessions', 'messages', 'conversations', 'conversation_messages'):
            self.assertEqual(current[table], self.original_chats[table])
        self.assertCountEqual(current['schema_migrations'], [('v8_to_v82_conversations',), ('v12_accounts_v1',)])

    def test_ranges_are_explicit_sets_without_hierarchy_or_unknown_access(self):
        self.prepare()
        store = self.store()
        expected = {'demo_ab': frozenset({'A', 'B'}), 'demo_c': frozenset({'C'}), 'demo_abc': frozenset({'A', 'B', 'C'})}
        for user in store.list_accounts():
            self.assertEqual(store.allowed_knowledge_bases(user.id), expected[user.username])
        self.assertEqual(store.allowed_knowledge_bases('not-a-user'), frozenset())

    def test_authentication_rejects_wrong_unknown_and_disabled_accounts(self):
        from psycopg2 import sql
        self.prepare()
        entry = next(c for c in self.credentials() if c['username'] == 'demo_ab')
        store = self.store()
        self.assertIsNone(store.authenticate(entry['username'], 'wrong-password'))
        self.assertIsNone(store.authenticate('not_a_user', entry['password']))
        self.assertIsNone(store.authenticate(None, entry['password']))
        self.assertEqual(store.authenticate(' DEMO_AB ', entry['password']).id, entry['user_id'])
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('UPDATE {}.users SET is_active=false WHERE id=%s').format(sql.Identifier(self.schema)), (entry['user_id'],))
        self.assertIsNone(store.authenticate(entry['username'], entry['password']))
        self.assertFalse(store.get_account(entry['user_id']).is_active)
        self.assertEqual(store.allowed_knowledge_bases(entry['user_id']), frozenset())

    def test_repeat_preserves_ids_hashes_and_credential_file(self):
        self.prepare()
        users = self.raw_users()
        content = self.credentials_path.read_bytes()
        self.assertEqual(self.prepare()['action'], 'already_ready')
        self.assertEqual(self.raw_users(), users)
        self.assertEqual(self.credentials_path.read_bytes(), content)

    def test_short_demo_credentials_remain_compatible_with_repeat_preparation(self):
        from password_security import hash_password
        from psycopg2 import sql
        self.prepare()
        data = json.loads(self.credentials_path.read_text(encoding='utf8'))
        with self.db, self.db.cursor() as cursor:
            for entry in data['accounts']:
                entry['password'] = 'web123'
                cursor.execute(sql.SQL('UPDATE {}.users SET password_hash=%s WHERE id=%s').format(sql.Identifier(self.schema)),
                               (hash_password('web123', minimum_length=6), entry['user_id']))
        self.credentials_path.write_text(json.dumps(data), encoding='utf8')
        users = self.raw_users()
        self.assertEqual(self.prepare()['action'], 'already_ready')
        self.assertEqual(self.raw_users(), users)
        for entry in data['accounts']:
            self.assertEqual(self.store().authenticate(entry['username'], 'web123').id, entry['user_id'])

    def test_changed_local_password_is_rejected_without_resetting_database(self):
        self.prepare()
        users = self.raw_users()
        content = json.loads(self.credentials_path.read_text(encoding='utf8'))
        content['accounts'][0]['password'] = 'different-fixture-password'
        self.credentials_path.write_text(json.dumps(content), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'conflict'):
            self.prepare()
        self.assertEqual(self.raw_users(), users)

    def test_existing_level_mapping_is_not_overwritten(self):
        from psycopg2 import sql
        self.prepare()
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL("DELETE FROM {}.account_level_libraries WHERE level_id='level_ab' AND knowledge_base_id='B'").format(sql.Identifier(self.schema)))
        with self.assertRaisesRegex(ValueError, 'conflict'):
            self.prepare()
        user = next(user for user in self.store().list_accounts() if user.username == 'demo_ab')
        self.assertEqual(self.store().allowed_knowledge_bases(user.id), frozenset({'A'}))

    def test_third_insert_failure_rolls_back_prior_account_inserts(self):
        from psycopg2 import sql
        import psycopg2
        self.prepare()
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('DELETE FROM {}.users').format(sql.Identifier(self.schema)))
            cursor.execute(sql.SQL("ALTER TABLE {}.users ADD CHECK(username <> 'demo_abc')").format(sql.Identifier(self.schema)))
        with self.assertRaises(psycopg2.IntegrityError):
            self.prepare()
        self.assertEqual(self.raw_users(), [])

    def test_foreign_keys_and_unique_logins_reject_invalid_records(self):
        from psycopg2 import sql
        import psycopg2
        self.prepare()
        for query in ("UPDATE {}.users SET username='demo_ab' WHERE username='demo_c'", "UPDATE {}.users SET level_id='missing-level' WHERE username='demo_c'"):
            with self.subTest(query=query), self.assertRaises(psycopg2.IntegrityError):
                with self.db, self.db.cursor() as cursor:
                    cursor.execute(sql.SQL(query).format(sql.Identifier(self.schema)))
        self.assertEqual(len(self.raw_users()), 3)

    def test_unprepared_store_and_partial_schema_are_rejected(self):
        from account_storage import PostgresAccountStore
        from prepare_demo_accounts import prepare_accounts
        from psycopg2 import sql
        with self.assertRaisesRegex(RuntimeError, 'migration'):
            PostgresAccountStore(schema=self.schema, **self.settings)
        with self.db, self.db.cursor() as cursor:
            cursor.execute(sql.SQL('CREATE TABLE {}.users (id TEXT PRIMARY KEY)').format(sql.Identifier(self.schema)))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            prepare_accounts(self.db, self.credentials_path, apply=True, schema=self.schema)
        self.assertFalse(self.credentials_path.exists())

    def test_cli_check_and_apply_report_no_passwords_or_hashes(self):
        from prepare_demo_accounts import main
        output, errors = io.StringIO(), io.StringIO()
        args = ['--schema', self.schema, '--credentials-file', str(self.credentials_path)]
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main(args), 0)
        self.assertEqual(json.loads(output.getvalue())['action'], 'check_only')
        self.assertFalse(self.credentials_path.exists())
        output.seek(0)
        output.truncate(0)
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(main([*args, '--apply']), 0)
        combined = output.getvalue() + errors.getvalue()
        for entry in self.credentials():
            self.assertNotIn(entry['password'], combined)
        self.assertNotIn('$argon2', combined)

    def test_complete_but_unmarked_tables_are_not_adopted(self):
        from psycopg2 import sql
        definitions = {
            'account_levels': 'id TEXT, display_name TEXT',
            'account_level_libraries': 'level_id TEXT, knowledge_base_id TEXT',
            'users': 'id TEXT, username TEXT, password_hash TEXT, level_id TEXT, is_active BOOLEAN, created_at TEXT',
        }
        with self.db, self.db.cursor() as cursor:
            for table, definition in definitions.items():
                cursor.execute(sql.SQL('CREATE TABLE {} ({})').format(sql.Identifier(self.schema, table), sql.SQL(definition)))
        with self.assertRaisesRegex(ValueError, 'unmarked'):
            self.prepare()
        self.assertFalse(self.credentials_path.exists())
        self.assertEqual(self.raw_users(), [])


if __name__ == '__main__':
    unittest.main()
