# Refresh generation implementation plan

**Goal:** Approved browser refresh/close continuation with owned background tasks.
**Spec:** ../specs/2026-10-05-refresh-generation.md
**Architecture:** Dedicated SQL task store, bounded background executor, authenticated task endpoints, React snapshot polling. Single server process; restart marks running tasks interrupted.

- [x] Write failing store/API tests for refresh continuation, idempotency, owner isolation, atomic completion and interruption.
- [x] Implement task store and explicit PostgreSQL additive migration; reuse SQLite test storage.
- [x] Implement bounded executor and endpoints; keep legacy SSE compatible, reject conflicts with active tasks.
- [x] Write failing React tests; implement task-mode boot/history/poll/submit with identity guards and no stale updates.
- [x] Verify desktop/mobile real browser refresh using blocked local fake generation and all regression checks.
- [x] Independent review, fix findings, backup/migrate database, build/restart server, read-only live checks.
- [x] Update local HANDOFF with implementation/agent verification/user acceptance separated.

No automatic retries, durable queue workers, server restart continuation, or multi-worker support. No unsolicited commit or push during this task.
