# Conversation Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Replace four-user-turn truncation in authenticated Web question rewriting with persisted, source-grounded conversation summaries and bounded recent messages.

**Architecture:** A pure context service plans compression and validates model output. An owner-scoped PostgreSQL store persists summaries and frozen task inputs; a graph context node runs before rewrite and checkpoints the result. Answering remains grounded in current authorized retrieval.

**Tech Stack:** Existing Python, psycopg2, LangGraph, ChatOpenAI, tiktoken, unittest, React/TypeScript and Playwright.

**Spec:** docs/superpowers/specs/2026-10-07-context-reflection-design.md

## Global Constraints

- First of two dependent plans; automatic extraction follows docs/superpowers/plans/2026-10-07-memory-reflection.md. Complete and validate this phase before that phase changes production behavior.
- Preserve dirty baseline and current checkout; no automatic commit or push. HANDOFF remains ignored. Record baseline/status and test results in local logs/context-reflection-work/progress.md.
- Initial history trigger 4,000 estimated Token; summary target 500–800, hard maximum800; post-compression history target2,500. Preserve latest four user turns when they fit; no blind truncation of input/facts. Estimates are not billed Token counts.
- Raw PG transcript remains intact; summaries stay conversation-scoped. Memory stays account-scoped. Neither summary nor assistant history is legal evidence. No new framework/vector database; existing CLI behavior remains unchanged.
- Explicit memory commands bypass legal context processing and are excluded from later legal rewrite history.
- Authenticated identity, current scope, task ownership/epoch, revision CAS and deletion apply on reads/writes/recovery. Never log full summaries or source text.
- Before live migration, create verified recoverable PG backup and stop/quiesce server. First tests use fakes and temporary PG schemas; no paid calls or production test chats.

## File map

- Create conversation_context.py: types, budgets, planning, source-valid summary model output and rendering.
- Create context_storage.py: owner-scoped summary, transcript-with-ordinal, frozen task input and CAS persistence.
- Create prepare_conversation_context.py: explicit additive migration, marker, constraints and check-only mode.
- Modify web_langgraph.py, graph_runtime.py, web_rag.py, query_rewrite.py, rag_app.py: context node and rewrite-only summary injection.
- Modify recovery_runner.py, generation_tasks.py, web_app.py: immutable accept-time inputs, service wiring and deletion/recovery.
- Create tests/test_conversation_context.py, tests/test_context_storage.py, tests/test_web_context.py; extend tests/test_web_recovery.py and fixtures.
- Modify frontend/src/types.ts, frontend/src/useChat.ts and related component status handling only as needed to label summary work.
- Create frontend/e2e/context.spec.ts and docs/v12-16-conversation-context.md.

## Task 1: Budget planning and grounded summary output

**Interfaces:** `ContextTooLong(ValueError)`, `ContextValidation(ValueError)`, `ContextConflict(ValueError)`; `ContextLimits(trigger_tokens=4000, summary_tokens=800, retained_tokens=2500, recent_turns=4)`; `SourceMessage(ordinal:int, role:str, content:str)`; `Summary(revision:int, through_ordinal:int, items:list[dict])`; `ContextService(model, limits=None, token_counter=estimate_tokens).plan(summary, messages)->dict`, `.compress(summary, messages, plan)->dict`, `.render(summary, messages)->list[HumanMessage]`. A plan contains `compress`, `source_ordinals`, `retained_ordinals`, `estimated_tokens`; an item contains `kind`, `text`, `source_ordinals`, `source_quotes`, `supersedes`.

- [x] Add behavior tests with injected counter/fake model: below/at/above threshold, preserving complete turns, four-turn goal versus token cap, repeated compression includes previous summary, user corrections, unresolved questions, invented quote/source rejection, assistant claims rejection, temporary/explicit memory command exclusion, empty history zero model calls and overlong single message rejection.
```python
counter = lambda text: int(text) if text.isdigit() else len(text)
service = ContextService(model=None, token_counter=counter)
messages = [SourceMessage(1, 'user', '4000')]
assert service.plan(Summary(0, 0, []), messages)['compress'] is False
messages = [SourceMessage(1, 'user', '4001')]
with self.assertRaises(ContextTooLong):
    service.plan(Summary(0, 0, []), messages)
```
- [x] Run `D:\conda\python.exe -m unittest discover -s tests -p test_conversation_context.py -v` and record actual RED failure.
- [x] Implement complete-turn planning, earliest ranges for compression and preservation within2,500 after summary allocation. Model returns grounded structured items with exact user quotes, not free-form assumed facts. Prompt includes previous validated items and newly eligible user statements, with explicit correction relations. Match ordinal/quote to owned frozen source; reject incorrect source, oversized output, cycles or ungrounded facts; no string slicing fallback. Validate source support independently of legal correctness and do not claim exact quotes prove semantic truth.
```python
for item in proposal['items']:
    for ordinal, quote in zip(item['source_ordinals'], item['source_quotes'], strict=True):
        source = source_by_ordinal[ordinal]
        if source.role != 'user' or quote not in source.content:
            raise ContextValidation('摘要来源无法核对')
if estimate_tokens(rendered_summary) > limits.summary_tokens:
    raise ContextValidation('摘要超出预算，请重新发送或缩短输入')
```
- [x] Re-run focused tests GREEN. Add finite summary timeout20s/max_retries0, one model call per new compression attempt. Carry safe counts/timing/available usage in results; no model in noncompression path. Record implementation and limitations; no commit.

## Task 2: Explicit PG migration and immutable snapshots

**Interfaces:** `ContextStore(schema='public', **settings).read(owner,cid,cursor=None)->dict`, `.capture(owner,cid,tid,cursor)->dict`, `.task_input(owner,tid,cursor=None)->dict`, `.save(owner,cid,summary,expected_revision,cursor=None)->dict`. `prepare_context(connection,apply=False,schema='public')->dict`. Use the existing message `ordinal`, not an invented message ID.

- [x] Add temporary-PG tests based on tests/test_memory_storage.py's login fixture: check-only/idempotence/marker refusal, owner mismatch, ordered source ranges, stale version, rollback, deletion cascade, frozen input unchanged by later messages, recovery from persisted task input.
```python
prepare_context(db, apply=True, schema=schema)
first = contexts.read(owner, cid)
assert first['revision'] == 0
saved = contexts.save(owner, cid, grounded_summary, expected_revision=0)
assert saved['revision'] == 1
with self.assertRaises(ContextConflict):
    contexts.save(owner, cid, grounded_summary, expected_revision=0)
```
- [x] Set `$env:RUN_POSTGRES_TESTS='1'; $env:PYTHONUTF8='1'` in the execution environment and run `D:\conda\python.exe -m unittest discover -s tests -p test_context_storage.py -v`; inspect RED, do not accept skipped tests as PG verification.
- [x] Implement additive tables `conversation_contexts(conversation_id PK/FK, user_id FK, revision, through_ordinal, items JSONB, updated_at)` and `generation_context_inputs(task_id PK/FK, user_id FK, conversation_id FK, input JSONB)`, marker `v12_conversation_context_v1`. Advisory migration lock/5s lock timeout, no adoption of unmarked objects. Read/capture conversation row under ownership lock; summary write CAS and validate all source ordinals against transcript. Store `input={summary,messages}` at acceptance.
```python
cursor.execute(sql.SQL('SELECT id FROM {} WHERE id=%s AND user_id=%s FOR SHARE')
    .format(sql.Identifier(schema, 'conversations')), (cid, owner))
if cursor.fetchone() is None:
    raise PermissionError('无权读取该对话')
```
- [x] Run storage tests GREEN, verify old table counts/content unchanged in migration fixture. Freeze safe projections: no task-private source text returned by task status APIs.

## Task 3: Graph integration, request budgets and recovery

**Interfaces:** `GraphRuntime` adds context_service/context_store/current identity; graph State adds `context_input`, `context_result`, `context_revision`. TaskStore accepts optional `contexts`, calls capture inside create transaction. `RecoveryRunner` supplies frozen context and runtime services; `StreamingRagTurn.stream(..., context_input=None, context_service=None, context_store=None)` keeps old optional-free behavior. `RetrievalQuestionRewriter.rewrite` retains existing signature and receives rendered authorized messages.

- [x] Add graph tests: more than four recent questions visible below threshold, early contract term survives compression, later correction wins, summary only to rewrite, original question stays memory selector input, no evidence still refuses, explicit command bypass, retrieval permission unchanged, context failure stops without fabricated legal answer. Add actual resume tests for before/after context-node checkpoint and stale/deleted input.
```python
events = list(turn.stream('那我的约定呢？', [], context_input=frozen, **fake_options))
assert rewriter.calls[0].history[0].content.startswith('当前对话摘要')
assert '合同期限两年' in rewriter.calls[0].history[0].content
assert '当前对话摘要' not in answer_model.received_prompt
assert embedding_queries_for_memory == ['那我的约定呢？']
```
- [x] Run `D:\conda\python.exe -m unittest discover -s tests -p test_web_context.py -v` RED.
- [x] Add context→rewrite path; explicit memory path remains separate. Carry summary result in checkpoint; summary persisted idempotently for same task/source range. If process dies between summary store and graph checkpoint, reconstruct from task-bound persisted result rather than re-call model. Store summary result metadata with frozen task input in same TX. On unfinished recovery, verify ownership/deletion/summary version; completed graph can finalize without fresh model inputs. Bump graph VERSION and conservatively block incompatible old unfinished tasks.
```python
graph.add_node('context', self._context_node)
graph.add_edge('context', 'rewrite')
# START ordinary path now targets context; memory_command remains unchanged.
```
- [x] Add `PromptBudget(input_tokens=16000, reserved_output_tokens=4096)` as conservative initial application caps, separately configurable from history budget. Validate full rendered rewrite/summary/answer input, reserve output and configure client max_tokens. Verify configured provider/model capacity supports the sum before live rollout; cap is not a claim about provider limits. If authorized evidence/prompt cannot fit, fail with explicit length message; do not silently drop required evidence or emit truncated legal conclusions. Test longest quotes and fake counters, distinguish estimated versus actual usage.
```python
if estimate_tokens(prompt.to_string()) > budget.input_tokens:
    raise ContextTooLong('本次输入或参考资料过长，请缩小问题范围')
```
- [x] Focused graph/query-rewrite/recovery GREEN; current CLI and memory graph tests remain GREEN. Ensure source text absent from safe logs and summary never returned as law source.

## Task 4: Integration acceptance, deployment and evidence

**Interfaces:** status `context` rendered as “正在整理对话上下文”; existing status/error shape retained. Browser fixture uses temporary PG and fake summary/answer services.

- [x] Add React status tests and Playwright context fixture with early condition, >4 subsequent turns, correction, refresh and another account/dialogue isolation. Normal short context must produce zero summary calls.
```ts
await expect(page.getByText('正在整理对话上下文')).toBeVisible();
await expect(page.getByText('本次按两年合同期限检索')).toBeVisible();
await page.reload();
await expect(page.getByText('本次按两年合同期限检索')).toBeVisible();
```
- [x] Run focused Vitest and Chrome fixture RED, wire status label without UI redesign, then GREEN. Use existing auth/request client and no text leakage into tooltips/errors.
- [x] Run full `D:\conda\python.exe -m unittest discover -s tests`, frontend `npm test -- --run`, `npm run typecheck`, `npm run build`, existing browser suites plus context suite. Record exact counts/failures/limitations and `git diff --check`; request fresh review under executing-plans review rules before deployment.
- [x] Write docs/v12-16-conversation-context.md explaining algorithm, budgets, cost, source/correction boundaries and test evidence. Back up and verify production PG, quiesce tasks, apply only explicit additive migration, compare old row digests, restart loopback server. Smoke auth/history/no-store with no production fake chats. If any gate fails, preserve existing deployment.
- [x] Update HANDOFF with separate implemented/agent-tested/user-validated/understood statuses. User real-model long-dialogue acceptance checks early condition/correction/source-grounded answer; no claimed % savings before same-batch measurements. Complete phase-one ledger; move to second plan without assuming user acceptance is automatic.

## Self-review and execution

This phase covers summary boundaries, sources, correction, ownership, model/request budgets, checkpoint reuse and deletion. Automatic memory triggers/persistence are intentionally in the dependent second plan. Implementation proceeds inline in the current checkout to preserve the existing working state; reviews may use the explicitly required reviewer workflow, without parallel ownership of edited files.

## Execution evidence (2026-10-07)

Implementation and agent verification complete: Python457, React41, Chrome18, type/build, independent review correction pass, verified restore backup, additive production migrations, loopback deployment and three-account read-only smoke passed. Exact commands/rulings: local logs/context-reflection-work/progress.md. Current main checkout retained, no commit/push; paid calls not run. Tests use real temporary PG/fakes; review findings and conservative classifier/storage tradeoffs documented. Browser scenarios focus close/restart and layout; other lifecycle cases are real-PG API/storage tests.

- [ ] User real-model semantic/latency/cost acceptance (separate from implemented functionality and automated verification).
