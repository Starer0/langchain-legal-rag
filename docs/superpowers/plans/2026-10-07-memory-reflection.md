# Background Memory Reflection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Discover stable user preferences/background from ordinary chats through opt-in, durable, bounded background extraction, with safe conflict review.

**Architecture:** Completion registers a persistent source-range job atomically with saved messages. A server-owned worker extracts from pending user messages and appends grounded memory or publishes a versioned suggestion. Database fences prevent replay, manual-edit overwrite and deleted-memory resurrection.

**Tech Stack:** Existing PostgreSQL/psycopg2, Python threads, existing ChatOpenAI and embeddings, React/TypeScript, unittest/Vitest/Playwright.

**Spec:** docs/superpowers/specs/2026-10-07-context-reflection-design.md

## Global Constraints

- Depends on completed docs/superpowers/plans/2026-10-07-conversation-context.md. Reuse its context source ordinals and summary compression event; do not implement a second summarizer.
- Current checkout, existing uncommitted work preserved, no commit/push; HANDOFF local ignored. No paid model calls in automated tests or production fake conversations.
- Independent auto_accumulate default false; processing requires both use-memory and auto-accumulate enabled. Explicit edits/commands remain available when off.
- Compression schedules reflection; otherwise idle120seconds after last completed answer. New input immediately postpones unstarted idle jobs; answer completion resets deadline. Server owns timer, PG owns due time/range/progress.
- One extraction worker globally, answer tasks have admission priority; source ranges processed once, exact duplicates skipped, ambiguous content skipped, conflict requires confirmation.
- Retain current memory budgets: core500 estimated Token/1500chars, extended12000chars/40entries/1000chars each; max3/500selected. No background arbitrary delete/rewrite and no model SQL.
- One extraction model call per job attempt, timeout20s/max_retries0; one crash recovery attempt maximum, no automatic provider retry loop. Limit extraction new-source batch to4000 estimatedToken plus up to1000 contextualToken; oversize single source gives visible failure. Ordinary answer never waits for reflection.
- Initial stable memory text is grounded in exact user excerpts; quotes alone cannot prove stable intent. Gate temporary/case/quoted/ambiguous content conservatively; do not infer facts from assistant output.

## File map

- Create reflection_storage.py, prepare_memory_reflection.py: additive durable policy/job/progress/candidate state and fences.
- Create memory_reflection.py: model extraction, bounded grounded candidates, duplicate/conflict decisions.
- Create reflection_runner.py: server-owned bounded worker with fake-clock testability.
- Modify generation_tasks.py, recovery_runner.py, context_storage.py, memory_storage.py, memory_api.py, web_app.py: atomic enqueue, source lifecycle, manual mutation invalidation, startup/shutdown.
- Modify frontend/src/useMemory.ts, frontend/src/MemorySettings.tsx and tests; add reflection status polling hook to existing account lifecycle if needed.
- Create tests/test_reflection_storage.py, tests/test_memory_reflection.py, tests/test_reflection_runner.py, tests/test_reflection_api.py, frontend/e2e/reflection.spec.ts, tests/fixtures/reflection_server.py and docs/v12-17-memory-reflection.md.

## Task 1: Durable policy, source cursors and fenced jobs

**Interfaces:** `ReflectionStore(schema='public',**settings).policy(owner,cursor=None)->dict`, `.set_policy(owner,enabled,expected_epoch,cursor=None)->dict`, `.enqueue(cursor,owner,cid,through_ordinal,reason,now)->dict|None`, `.defer(cursor,owner,cid,now)->None`, `.claim(now)->dict|None`, `.finish(job_id,epoch,result,cursor=None)->bool`, `.invalidate(owner,cursor)->None`, `.recover(now)->None`, `.jobs(owner,cid)->list[dict]` (safe projection). Policy is `{auto_accumulate,epoch}`. Job records include identity/range/due_at/status/epoch/policy_epoch/memory_revision/attempts/result; source cursor includes processed_through and invalidated_through. `prepare_reflection(connection,apply=False,schema='public')->dict`. Service clock returns epoch seconds; storage converts due dates to TIMESTAMPTZ and decodes safe job dates to epoch seconds consistently.

- [x] Write real temporary-PG tests: check-only migration, defaultoff, idempotent marker, atomic completion/outbox rollback, range coalescing/dedup, defer on new request, claim eligibility with no active generation, stale epoch, inactiveaccount/deletedconversation, crashrecover once, manual edit invalidates pending sources, cursor advancement only with committed result.
```python
job = reflection.enqueue(cursor, owner, cid, through_ordinal=7, reason='idle', now=clock())
assert job['due_at'] == clock() + 120
clock.advance(119)
assert reflection.claim(clock()) is None
clock.advance(1)
assert reflection.claim(clock())['id'] == job['id']
```
- [x] Run `D:\conda\python.exe -m unittest discover -s tests -p test_reflection_storage.py -v` with RUN_POSTGRES_TESTS/PYTHONUTF8 enabled; record RED.
- [x] Add tables `memory_reflection_policies(user_id PK/FK,enabled,epoch)`, `memory_reflection_progress(conversation_id PK/FK,user_id FK,processed_through,invalidated_through)`, `memory_reflection_jobs(id PK,user_id FK,conversation_id FK,start_ordinal,end_ordinal,due_at TIMESTAMPTZ,status,epoch,policy_epoch,memory_revision,attempts,input JSONB,result JSONB,updated_at)`, `memory_reflection_suggestions(id PK,job_id FK,user_id FK,conversation_id FK,memory_revision,policy_epoch,proposal JSONB,status,created_at)`. Explicit marker/advisory lock, FK cascades, due/status index, unique completed source range. Serial claim under server lease plus FOR UPDATE SKIP LOCKED; transaction verifies both toggles/current revision/source existence at save.
```sql
SELECT id FROM memory_reflection_jobs
WHERE status='pending' AND due_at <= CURRENT_TIMESTAMP
ORDER BY due_at,id FOR UPDATE SKIP LOCKED LIMIT 1;
```
- [x] Implement invalidation at any manual memory mutation or toggle: increment policyepoch, invalidate jobs/suggestions and mark all then-existing source ordinals ineligible for future retry. A new source statement after that boundary may qualify. Advancing these fences shares manual save TX; auto save does not invoke manual invalidation, but invalidates stale suggestions by revision. Review current user-row→memory-row lock order and follow it everywhere to avoid deadlocks.
- [x] Focused PG GREEN; old tables unchanged; job/suggestion private payload omitted from safe APIs/logs.

## Task 2: Grounded extraction and atomic append

**Interfaces:** `ReflectionService(model,memory_service,token_counter=estimate_tokens).extract(messages,existing,context)->dict` yields `candidates=[{layer,text,source_ordinals,source_quotes,relation,target}]`; `.validate(result,messages,existing)->dict` yields `{additions,suggestions,skipped}`; `normalized(text)->str` applies Unicode NFKC/whitespace normalization only, not semantic claims. `ReflectionStore.apply(job_id,epoch,validated,prepared)->dict` revalidates frozen source/revision/epoch and commits memory+suggestions+progress+finished atomically. `relation` is new/duplicate/conflict; target for conflict must match existing text. Only additions enter memory.prepare and append current text; no authorized_change reuse.

- [x] Test natural preference without commandkeywords, stablebackground, temporary request, thirdparty/quotedtext, casefacts/dates/money, assistant-only inventedfacts, mismatchedquotes/ordinal, duplicate exact/semantic, contradictory preference, fullmemorybudget, disabledpolicy and modelerror; test stale deletion between extraction and commit and rollback after memory write.
```python
messages = [SourceMessage(3, 'user', '我法律基础薄弱，以后请多举例解释。')]
checked = service.validate(fake_result, messages, existing)
assert checked['additions'][0]['text'] == '喜欢配合例子解释'
assert service.validate(temporary_result, [SourceMessage(5,'user','这次简短一点')], existing)['additions'] == []
```
- [x] Run focused unittest RED, implement one bounded JSON extraction prompt using user messages plus labeled context/existing memory; stable claim includes exact evidence. Literal-source gate plus conservative stable-intent policy rejects unsupported automaticwrites. Ambiguity goes skipped; supported conflict and capacity failure become suggestions. Do not claim classifier accuracy before real expression evaluation.
```python
if candidate['relation'] == 'conflict':
    suggestions.append(candidate)
elif candidate['relation'] == 'duplicate' or normalized(candidate['text']) in existing_normalized:
    skipped.append(candidate)
else:
    additions.append(candidate)
```
- [x] Prepare embeddings outside TX, append preserves all current text byte-for-byte except necessary joiningnewline. Inside apply TX recheck sourcequotegrounding, allowedadditionlayer, proposedmergedtext, CAS, activeuser and policyfence; single result committed once. Noaddition skips memory save/version increment/embedding. Aggregate safe result counters and actual available Token usage.
- [x] Focused service/storage GREEN; original explicit command tests still GREEN.

## Task 3: Server lifecycle, graph trigger and completion consistency

**Interfaces:** `ReflectionRunner(store,service,*,clock,active_generation,lease).start()`, `.wake()`, `.shutdown()`. Clock has `.now()` and test advancement. Runner uses Condition/Event waits, not browser timers. TaskStore gets optional reflection; context result includes `compressed:bool` safe metadata in completion. Job input source ranges derive accepted task and saved user ordinal, not browserprovidedids.

- [x] Test real completion registers idle job same TX, compression promotes same range to immediate job, repeatcomplete leaves one job, new requestdefer, active answerprevents claim, partialanswerfailure produces no eligibility from unsavedassistant, no evidence completed question still eligible. Test websocket/SSE disconnect and browser close leave worker alive, service stop/restart retains range, old worker fence, timeoutfailure once and manualretry epoch, cleanup/shutdown order.
  Define a test-only `FakeClock` with `__call__`, `now` and `advance(seconds)` sharing one epoch value, and a bounded polling helper before these tests:
```python
def wait_for_job_finished(store, owner, cid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        jobs = store.jobs(owner, cid)
        if jobs and all(job['status'] == 'completed' for job in jobs):
            return True
        time.sleep(0.01)
    return False
```
```python
assert tasks.complete(tid, '完成', [], epoch=epoch)
assert not tasks.complete(tid, '完成', [], epoch=epoch)
assert len(reflection.jobs(owner,cid)) == 1
runner.shutdown()
clock.advance(120)
restarted.start()
assert wait_for_job_finished(reflection, owner, cid)
```
- [x] Focused runner/graph integration RED, implement afteranswer TX enqueue; context compression represented in safe completion metadata rather than callback that loses job on crash. Wake only aftercommit. Defer idle at new accepted question; don't perform memory updates before answer persist. One worker globally; cap modelruntime, do not hold DB locks during model/embedding calls.
- [x] Install worker startup after migrations and existing checkpoint lease; shutdown worker before closing DB/graphresources. Fail startup if marked schema incomplete, no silent migrations. Recovery repeats only crash-in-progress once; ordinary modelfailure terminal, manual retry rechecks currentfences/source. Do not cancel runninganswer on automaticmemory revision change; maintain documented existing unfinishedgraph recovery check.
- [x] Run runner/recovery/memory/context focused GREEN, prove fakecall counts and committed version counts separately; remote call exactlyonce not claimed.

## Task 4: Authenticated controls, suggestions and UI feedback

**Interfaces:** Extend GET `/api/memory` with `auto_accumulate,policy_epoch,reflection_status`; keep old memory keys compatible. Add PUT `/api/memory/automation` strict `{enabled:bool,expected_epoch:int}`; GET `/api/memory/suggestions`; POST `/api/memory/suggestions/{id}/accept` strict `{expected_revision:int,expected_epoch:int}`; POST `.../ignore`; POST `/api/memory/reflection/{job_id}/retry`. All ownerchecked/CSRF/no-store, safe error statuses401/403/404/409/422/503. Accepted suggestion references source and exact bounded replacement or addition proposal; model cannot select SQL or unrelateddelete.

- [x] API tests other-user ids denied, disabledpolicy rejection, staleversion draftpreserved409, removedsource rejection, replayaccept once, ignore once, retry fenced, malformedpayload, revokedaccount, no privatevector/jobinput leaked.
```python
response = other_client.post(f'/api/memory/suggestions/{suggestion_id}/accept', json=payload)
assert response.status_code == 404
assert memories.read(owner)['revision'] == before_revision
```
- [x] RED API/UI tests then wire existing auth routes. Acceptedconflict uses versioned exact target bounded by originalproposal; fresh verification under sameTX; missing target/changedversion rejects. Settings text says ordinarychat can save stableprefs when enabled, includes offpause semantics and independent deletion.
- [x] Extend useMemory with policy state/separatebusy operation and account/requestepoch protection; no destructive reload when draftdirty. Add checkbox “自动积累长期记忆（后台处理）”, pending conflict sourcepreview, accept/ignore, failure/retry. Sidebar or settings status says “已更新长期记忆” only aftercommittedevent/version; accountscoped lightweight polling whilepagevisible, no browser dependency foractualwork. Pendingjobs are not “已保存”. Paused toggle never flips useenabled automatically.
```ts
await user.click(screen.getByLabelText('自动积累长期记忆（后台处理）'));
expect(request).toHaveBeenCalledWith('/api/memory/automation', expect.anything());
expect(screen.queryByText('已更新长期记忆')).not.toBeInTheDocument();
```
- [x] Run `npm test -- --run` and typecheck GREEN; user sees paused state/error without lostdrafts, keyboard/accessibility and375px layout maintained. Apply ui-ux-pro-max instructions at implementation time.

## Task 5: Recovery acceptance, deployment and quantitative ledger

- [x] Fake-provider Chrome fixture: shortchat→close tabbefore120s→server jobdone→reopenmemory; backendstop/restart; newmessage resetsdeadline; conflictaccept, manualdelete beforejobcompletion, twouserisolation, toggleoff, mobilelayout. Fake clock allows seconds to advance without waiting actual2minutes, plus one real clock schedulingintegration test.
  In the isolated tests/fixtures/reflection_server.py only, expose a local test clock advance endpoint that changes FakeClock and wakes runner. The Playwright fixture defines `advanceClock(seconds):Promise<void>` to call it and await the known job completion. Never register this endpoint in web_app.py or production; the time-based integration test uses a shortened injected idle interval with a real monotonic/UTC clock pair.
```ts
await page.close();
await fixture.advanceClock(120);
const reopened = await context.newPage();
await reopened.goto(fixture.url);
await reopened.getByRole('button', {name:'长期记忆'}).click();
await expect(reopened.getByText('喜欢配合例子解释')).toBeVisible();
```
- [x] Run all Python tests realtemporaryPG/fakes, React tests/type/build, existingbrowser suites plus context/reflection. `git diff --check`; fresh review as executing-plans requires and bounded correctionpass, no unsolicited automatic git integration. Record exact outcomes in localledger.
- [x] Write docs/v12-17-memory-reflection.md and privacy/cost/lifecycle limitations. Audit existing questionlogs separately; metadataonly for new extractionlogs, no assertion old questionlogs have zeroprivatecontent. Set up samebatch benchmark script `evals/evaluate_context_reflection.py` with fake deterministic mode and explicit userauthorized live mode; record sourceconditionsretained, groundedmemorypositive/negative outcomes, batchcalls, usage/duration/config/source revision, no fabricatedpercentages. No modelnetworkcalls by default.
- [x] Verify provider/modelbudget assumptions and explicit configuration before productiondeploy. Quiesce/verifiedPGbackup, additive reflectionmigration and oldrowdigestcomparison, restart loopback. Auto flagdefaultfalse, old memory preserved. Smoke identity/history/policy/no-store, user performs firstreal automaticmemory acceptance.
- [x] HANDOFF separately records implemented/agentverified/userverified/understood and pendingmetrics. Completed automatic tests do not imply user realmodel acceptance or WorkBuddy parity. Plan complete only when required gates actually pass; future upgrades remain distinct from this scope.

## Self-review and handoff

All approved shortchat/closebrowser/restart/duplicate/conflict/manualdelete rules have explicit tasks. The current task memory snapshot and recovery version checks remain guarded. Failed tasks are visible and finite; queue durability is not a promise of no repeated remote model call. Execute inline after phaseone; review through required skill workflow rather than proactive parallel editing.

## Execution evidence (2026-10-07)

Implementation and agent verification complete: Python457, React41, Chrome18, type/build, independent review correction pass, verified restore backup, additive production migrations, loopback deployment and three-account read-only smoke passed. Exact commands/rulings: local logs/context-reflection-work/progress.md. Current main checkout retained, no commit/push; paid calls not run. Tests use real temporary PG/fakes; review findings and conservative classifier/storage tradeoffs documented. Browser scenarios focus close/restart and layout; other lifecycle cases are real-PG API/storage tests.

- [ ] User real-model semantic/latency/cost acceptance (separate from implemented functionality and automated verification).
