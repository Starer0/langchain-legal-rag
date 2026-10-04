# Account Foundation Implementation Plan

> **For agentic workers:** Execute inline using superpowers:executing-plans. Existing user approval covers this account foundation; preserve the active checkout and its uncommitted V12 work.

**Goal:** Persist three real demo accounts with secure password hashes and explicit level mappings in the existing PostgreSQL database.

**Architecture:** Independent account storage plus an explicit transactional preparation command; shared database configuration retains existing web behavior.

**Tech Stack:** Python, psycopg2, argon2-cffi, PostgreSQL 15, unittest.

**Spec:** docs/superpowers/specs/2026-10-03-account-foundation-design.md

## Global Constraints

- Scope approved by user: account foundation only; web login and retrieval enforcement are subsequent steps.
- Preserve anonymous chat rows, existing web API, Chroma and graph checkpoints.
- HANDOFF, passwords, local credentials and backups stay ignored; no credential output.
- Use existing checkout and independent local PostgreSQL; test only UUID test schemas.
- Do not conflate tests with user verification or explicit understanding.

## Task 1: Passwords, storage and explicit preparation

**Files:** Create password_security.py, account_storage.py, prepare_demo_accounts.py, database_settings.py; modify web_app.py to reuse equivalent configuration helpers, requirements.txt and .gitignore; test tests/test_account_foundation.py.

**Interfaces:** hash_password(password)->str; verify_password(encoded,password)->bool; PostgresAccountStore(schema,**connection_settings).authenticate(username,password)->Account|None; get_account(user_id)->Account|None; list_accounts()->list[Account]; allowed_knowledge_bases(user_id)->frozenset[str]. prepare_accounts(connection,credentials_path,apply=False,schema='public')->dict.

- [x] Write and run failing tests. Minimal password test:

```python
first = hash_password('fixture-password-123')
second = hash_password('fixture-password-123')
self.assertNotEqual(first, second)
self.assertTrue(verify_password(first, 'fixture-password-123'))
self.assertFalse(verify_password(first, 'wrong-password'))
```

PostgreSQL tests must assert exact literal demo ranges {A,B}, {C}, {A,B,C}; query raw hash to ensure plaintext differs; create a CHECK rejection on the third account and verify earlier inserts rolled back; repeat the same preparation and compare raw users before/after; reject changed fixture password without resetting existing credentials. Default check must leave all schema tables and the credentials file unchanged. Run `python -m unittest discover -s tests -p test_account_foundation.py -q` with RUN_POSTGRES_TESTS=1.

- [x] Implement the specified tables, Argon2id helpers, credential-file creation/validation, prepared-schema checks, read-only default and transactional no-overwrite preparation. Tests control database/credential paths, not the production password implementation.
- [x] Run targeted tests then complete Python regression. Ask independent reviewer for a read-only scoped review using requesting-code-review. Fix reproduced defects and repeat affected checks.
- [x] Back up real PostgreSQL with `docker exec legal-rag-postgres pg_dump -U legal_rag -d legal_rag --format=custom` to a binary file in data/backups, then explicitly apply preparation. Compare original chat rows before/after using read_target; query actual account store to verify all three passwords and ranges without printing secrets.
- [x] Write docs/v12-4-account-foundation.md and update local HANDOFF with implemented, agent-tested, user-tested and understood separately. Keep this learning step local while login/browser acceptance remains a subsequent step; do not auto-commit unrelated pending V12 work.

Validation: initial red tests observed; final 222 Python tests passed. Review finding reproduced and fixed; real database backed up, three accounts prepared and verified; existing chats unchanged. Browser login remains the next step. No commit/push performed.
