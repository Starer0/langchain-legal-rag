# Account Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver editable account core/extended memory, scoped answer retrieval, explicit conversational updates and safe task recovery.

**Architecture:** PostgreSQL owns revisioned text/entries/task snapshots. Pure graph nodes select or propose memory; a fenced task-completion transaction applies conversational mutations and saves feedback once. React settings use the existing authenticated client.

**Tech Stack:** Python, psycopg2, LangGraph, existing OpenAI-compatible embedding/chat clients, tiktoken estimates, React/TypeScript, unittest/Vitest/Playwright.

**Spec:** docs/superpowers/specs/2026-10-07-account-memory-design.md

## Global Constraints

- Work in the approved current checkout, preserve all previous uncommitted changes; no automatic commit/push. HANDOFF stays ignored.
- Core 500 estimated tokens / 1500 characters; extended 12000 characters / 40 entries / 1000 characters each; selected maximum3 / 500 estimated tokens. Initial cosine floor0.5, configurable and development-tested, not a probability.
- Existing remote embedding provider, private PG entries, no new memory framework/vector service. No ordinary-query extra chat selection calls.
- Server identity, CSRF, active account, revision CAS; no model-issued SQL or privilege changes. Degradation cannot mask whole database failure.
- Commands propose only; commit memory+feedback+completed atomically, fenced and idempotent. Partial outputs never claim saved.
- Preserve current refresh/recovery/history/permissions and CLI behavior. Initial tests/fakes must not call paid models or contaminate production corpus.

## Task 1: Bounded preparation and selection

**Files:** Create memory_service.py; tests/test_memory_service.py; requirements.txt adds explicit tiktoken.
**Interfaces:** MemoryLimits; estimate_tokens(text); validate_texts(core,extended,limits); split_entries(text); MemoryService(embeddings, command_model=None, fingerprint='', limits=None).prepare(document,core,extended,enabled), select(question,snapshot), propose(question,snapshot); command_candidate(question).

- [x] Write behavior tests for budgets/paragraphs, unchanged-vector reuse, malformed vectors, related/unrelated selection, disabled/no entries zero query calls, embedding failure, exact instruction vs quoted/temporary text and invalid proposals.
```python
assert service.select('解释函数报错', disabled_snapshot)['extended'] == ''
assert service.select('解释函数报错', python_snapshot)['selected_ids'] == ['python']
```
- [x] Run `python -m unittest discover -s tests -p test_memory_service.py` and observe missing module/features fail.
- [x] Implement validation with tiktoken cl100k estimates, deterministic text blocks, embedding reuse keyed text+fingerprint, finite-vector validation, cosine ranking within budget. Command model returns bounded JSON operations with a safe response; deterministic candidate avoids ordinary extra calls.
```python
if not snapshot['enabled'] or not snapshot['entries']: return core_only
selected = sorted(valid_matches, key=lambda item: (-item['score'], item['ordinal']))[:limits.top_k]
```
- [x] Run focused service tests; record RED/GREEN in logs/ledger; no commit.

## Task 2: Explicit migration and revisioned storage

**Files:** Create memory_storage.py, prepare_account_memory.py, tests/test_memory_storage.py.
**Interfaces:** MemoryStore(settings/schema).read(owner,cursor=None), save(owner,prepared,expected_revision,cursor=None), task_input(tid,owner), capture(cursor,task,document), observation(cursor,tid,status); prepare_memory(connection,apply=False,schema='public').

- [x] Write temporary real-PG tests using isolated login fixture: check-only no creation, repeat migration, unmarked tables reject, own-account reads, core/entry atomic replacement, CAS conflict, inactive account reject, snapshot cleanup with conversation, rollback.
```python
saved = memories.save(owner, prepared, expected_revision=0)
self.assertEqual(saved['revision'], 1)
with self.assertRaises(MemoryConflict): memories.save(owner, stale, expected_revision=0)
```
- [x] Run `RUN_POSTGRES_TESTS=1 python -m unittest discover -s tests -p test_memory_storage.py`, observe missing feature.
- [x] Implement 3 additive tables and marker transaction/advisory lock. Use users row lock to serialize first-save race; memory CAS and entries share transaction. Store JSON vectors/fingerprints, validate ownership, never return them through public document projection.
- [x] Run focused PG tests; record results; no production migration yet.

## Task 3: Task snapshots, graph integration and atomic command completion

**Files:** Modify generation_tasks.py, recovery_runner.py, web_langgraph.py, graph_runtime.py, web_rag.py, rag_app.py; create tests/test_web_memory.py and extend recovery tests.
**Interfaces:** TaskStore.memory optional MemoryStore; create(...,memory_input=None); complete(...,memory_result=None). RecoveryRunner memory_service optional. Graph stream(...,memory_input=None,memory_service=None,memory_owner=None,memory_revision_guard=None).

- [x] Test actual graph with fake embedding/model: core+selected only in answer, no query/rewrite leakage, no evidence skip, failure warnings, completed selection checkpoint reused. Commands bypass legal chain, no saved success before transaction.
- [x] Test complete atomic mutation: proposal replay returns one increment and one pair of messages; stale revision failed/no edit; storage error rollback. Recover interrupted selection/command, mutation after graph completion, changed memory revision blocks incomplete graph but complete graph can save prior result.
```python
self.assertTrue(tasks.complete(tid, feedback, [], epoch=epoch, memory_result=proposal))
self.assertFalse(tasks.complete(tid, feedback, [], epoch=epoch, memory_result=proposal))
self.assertEqual(memories.read(owner)['revision'], 2)
```
- [x] Run focused tests RED before edits.
- [x] Add optional memory routing/node/state and web-only prompt section. MemoryService/identity stays runtime; snapshots and safe result are serializable. Attach task snapshot at create under same transaction. Finished graph is saved without repeat write; interrupted graph uses memory revision guard; mutation is completed via same task cursor, correct active owner/epoch/revision.
- [x] Add task observations/warnings from private input table into safe task snapshot, not sources/body. Existing configurations without memory stay unchanged.
- [x] Run new graph/PG tests and existing recovery/task tests GREEN.

## Task 4: Authenticated API and production factories

**Files:** Modify web_app.py; create memory_api.py; extend tests/test_memory_storage.py or test_web_memory_api.py.
**Interfaces:** install_memory_routes(app, authenticated_owner, memory_provider, service_provider); GET/PUT /api/memory; service bounded embedding/chat timeout factory.

- [x] Test unauthorized/CSRF/other owner, no-store, exact safe projection, version conflict, save model/embedding failure preserves prior text, PUT disable without embedding, task endpoint snapshot.
- [x] Run RED.
- [x] Add providers/lifespan initialization only authenticated React; require explicit migration; TaskStore and RecoveryRunner receive initialized providers. Dedicated clients use bounded timeout/max_retries=0; memory fingerprint has no secret. No ignored env credentials output.
- [x] Run API+real PG existing login tests GREEN.

## Task 5: Settings and warning presentation

**Files:** Create frontend/src/MemorySettings.tsx, useMemory.ts, memory.test.tsx; modify useChat.ts/types.ts/components.tsx/styles.css minimally.
**Interfaces:** ChatController.memoryRequest(path,options) uses existing auth client and identity version; useMemory(request,userId) loads/saves draft state, expected_revision, safe statuses; MemorySettings props owner/request/onClose.

- [x] Test real settings with mocked HTTP boundary: both fields, disable/save, clear confirmation, failure preserves draft, conflict reload, switch account late request rejected, modal focus/keyboard; test task warning displayed and not sources.
```tsx
await user.type(screen.getByLabelText('核心记忆'), '先说结论');
await user.click(screen.getByRole('button', {name:'保存记忆'}));
expect(await screen.findByText('记忆已保存')).toBeVisible();
```
- [x] Run `npm test -- --reporter=dot` RED.
- [x] Implement same visual tokens/panel with accessible labels, disabled loading, explicit save/clear, dirty-close confirmation, identity cleanup, live error/success, no raw vectors. Keep settings independent of chat task polling.
- [x] Run Vitest, `npm run typecheck`, `npm run build` GREEN.

## Task 6: Whole-system verification, review and deployment

**Files:** Add frontend/e2e/memory.spec.ts and test fixture; docs/v12-15-account-memory.md; update local HANDOFF and this plan ledger.

- [x] Test actual FastAPI+PG/local fake embeddings model browser1440/375: edit, converse update, reload settings, cross-account isolation, failure/disabled, sources unaffected. No paid remote calls.
- [x] Full `RUN_POSTGRES_TESTS=1 python -m unittest discover -s tests`; React tests/type/build; full browser suite; inspect failure outputs without secrets.
- [x] Dispatch one fresh reviewer per requesting-code-review skill; review new files plus modified paths with dirty baseline, focusing command authorization, atomicity/fencing, prompt isolation, memory deletion/recovery, identity races, migration.
- [x] Reproduce/fix important findings RED→GREEN; final required checks.
- [x] Before production migration, check project running tasks/PID and make backup with old-table digest. Add tables once, verify unchanged old data; stop only verified project PID when idle, launch hidden replacement. Check homepage/auth/memory read without creating production test chats or changing user memory. User acceptance runs real new functionality separately.
- [x] Record actual metrics, limitations, commands/results, user acceptance pending. Preserve plan ledger while uncommitted; do not push or archive code.

## Review Focus

Natural-language memories are untrusted, stored proposals cannot authorize any user/layer outside their frozen input, raw updates cannot silently delete unrelated fields, no restore after deletion, capture/write lock ordering avoids deadlocks, orphan tables refuse adoption, private snapshots never leak through task JSON/logs, old account responses never update new settings, no artificial success streamed before DB transaction.
