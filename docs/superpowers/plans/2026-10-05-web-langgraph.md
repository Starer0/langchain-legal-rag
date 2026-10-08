# Web LangGraph implementation plan

**Goal:** Preserve approved web behavior while executing RAG through StateGraph.
**Spec:** ../specs/2026-10-05-web-langgraph.md
**Architecture:** Shared per-stage methods in web_rag.py, separate compiled graph adapter in web_langgraph.py, production factory selects graph adapter. Existing tasks/storage/UI contracts remain unchanged.

- [x] Add failing graph execution, parity, live incremental streaming and failure tests.
- [x] Extract shared stage operations; retain sequential behavior and existing permission/logging tests.
- [x] Implement graph nodes/edges/custom adapter; switch factory and verify independent invocation state.
- [x] Exercise graph in real backend/browser task refresh fixture, run relevant and full checks.
- [x] Independent review, fix findings, start local server (port had no listener) and verify read-only live paths.
- [x] Document implementation and distinguish user verification/understanding in local HANDOFF.

Evidence: 323 Python tests and 10 Chrome browser tests pass. Independent review's end-of-stream cancellation finding reproduced red, then fixed and included in the full passing run. New server PID 102424; three existing accounts' login/history/task-mode/logout paths pass. No remote model calls or production test conversations. User real-model acceptance remains pending.

No checkpoint migration, tools, approval workflow, retries, model changes, frontend redesign, automatic commit/push.
