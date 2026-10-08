# Web recovery implementation plan

> Execute inline with superpowers:executing-plans; preserve current uncommitted work.

**Goal:** Persist web graph progress and resume unfinished tasks after server restart.
**Architecture:** Official checkpointer; request runtime outside persisted State; fenced tasks and startup scheduler; one process lock.
**Tech Stack:** LangGraph 1.2, PostgreSQL, psycopg3, SQLite local fixtures, FastAPI/React.
**Spec:** ../specs/2026-10-06-web-recovery-design.md

## Constraints
Single process, four workers, one running task per owner; maximum three recovery attempts. No automatic model failure retries, tools, long-term memory, frontend streaming redesign or automatic commits. Checkpoint migration requires backup. Preserve old task behavior.

## Task 1: Checkpointed graph
Files: web_langgraph.py, web_rag.py, graph_runtime.py, tests/test_web_recovery.py, requirements.txt.
- [x] Write test interrupted answer then reopen SQLiteSaver; assert completed rewrite/retrieve/rerank called once and final answer replaces partial.
- [x] Run test red; install compatible official postgres saver dependencies.
- [x] Introduce runtime context (profile/trace/cancel event), serializable State stages, thread_id task mapping and sync durability.
- [x] Test finished checkpoint returns done without model calls; validate owner/scope/version; existing graph suite green.
Interfaces: stream(..., task_id=None, resume=False, execution_epoch=None, guard=None); completed checkpoint is emitted as done; delete_checkpoints(task_id).

## Task 2: Durable task recovery
Files: generation_tasks.py, task_recovery.py, prepare_web_recovery.py, tests/test_web_recovery.py.
- [x] Write red tests for claim fencing, recovery limit, old tasks compatibility, completed transaction idempotence.
- [x] Add recovery metadata table keyed task ID with scope/version/attempt/epoch; avoid changing existing task column contract.
- [x] Dispatcher recovers running recoverable tasks FIFO at capacity four, resolves current authorization, preserves IDs and submission keys.
- [x] Reset partial on answer replay, use done.answer as authoritative final result; epoch guards progress/final writes.
- [x] Test permission changes, account disabled (empty scope), graph/schema mismatch, broken checkpoint, stale worker and queue capacity.
Interfaces: RecoveryStore.register/claim/pending/check_epoch; RecoveryRunner start/recover/shutdown.

## Task 3: Application lifecycle and cleanup
Files: checkpoint_storage.py, web_app.py, tests/test_web_recovery.py, frontend/src/components.tsx.
- [x] Test exclusive server lock and initialization refusal without migration.
- [x] Explicit migration creates official saver tables in dedicated schema; production pool and lock lifetime bounded by lifespan.
- [x] Startup installs RecoveryRunner; shutdown cancels/drains workers before closing saver, task fencing prevents old writes.
- [x] Conversation deletion enqueues checkpoint cleanup inside delete transaction; drain on startup and after deletion.
- [x] Progress stage indicates recovery/restarted answer; no request retry behavior changes.

## Task 4: Verification/deployment
- [x] Separate child process crash/reopen fixture (after rerank and during answer), assert single saved turn/sources and node counts.
- [x] Run real temporary PostgreSQL saver and task migration tests, full Python/React/build/browser checks.
- [x] Independent review and repair regression tests.
- [x] Backup production, fingerprints before/after additive migration; restart only known server after checking active tasks; verify read-only paths.
- [x] Update release notes and HANDOFF with agent verification separate from pending user acceptance.

## Execution ledger

Ruling: inline execution in existing checkout preserves ongoing uncommitted web changes; no auto commit. Recovery fields are in a task-keyed table, keeping the existing task snapshot contract stable. Official saver uses a dedicated schema. Checkpoint SQL and serialization remain official, with transactional task-row locks/epoch checks for stale writes. Use public pool.connection API rather than package private helpers.

Evidence: graph tests first failed unsupported checkpointer; task/runner/API tests failed missing features; cancel/revocation/stale-checkpoint findings reproduced red then green. Final 337 Python, 27 React, production build and 12 Chrome browser tests pass. Independent review final14 recovery tests and PG/close probes pass. Browser restart tests repeated after guards:2 pass.

Production backup data/backups/legal-rag-20261006T044753Z-0f199dbc-pre-web-recovery.dump verified, existing tables unchanged. Migrated and started PID141948; three accounts login/history/task/logout pass. No paid model calls or production test questions. User real-backend acceptance remains pending.
