# V8.2 Multi-Conversation Web RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the anonymous Web RAG from one session-wide chat into persistent, isolated multi-conversation chat with a sidebar.

**Architecture:** Keep the anonymous cookie as the ownership boundary. Add `conversations` and `conversation_messages` beside the legacy V8.1 messages table, migrate old records exactly once, and make every chat read/write target an explicit `conversation_id`. FastAPI serves conversation CRUD and conversation-scoped SSE; raw HTML/CSS/JavaScript renders the responsive sidebar and confirmation-based delete flow.

**Tech Stack:** Python 3, SQLite standard library, FastAPI, Uvicorn, LangChain messages, browser Fetch API, native HTML/CSS/JavaScript.

**Spec:** `docs/superpowers/specs/2026-09-26-v8-multi-conversation-design.md`

## Global Constraints

- Preserve the V1–V8 CLI entry points and the V8.1 single-query RAG default.
- A cookie is only an opaque anonymous session ID; every conversation lookup must verify `session_id` ownership and return 404 for a foreign ID.
- Persist only complete user/assistant turns; failed, cancelled, and partial streams write neither message.
- History-aware Rewrite receives messages from only the selected conversation and retains only the newest `HISTORY_TURNS` complete pairs.
- New conversations begin as `新对话`; only the first successful turn may auto-title a non-custom title.
- Manual titles are trimmed, must contain 1–80 characters, and are never overwritten by automatic title updates.
- Delete is permanent, uses database `ON DELETE CASCADE`, and is preceded by browser confirmation.
- All SQL uses parameters; no logs contain raw questions, answers, cookies, or keys.
- Keep FastAPI/raw web assets; do not add React, a bundler, user accounts, search, archive, or cross-device sync.
- Use `python -m unittest discover -s tests -v`, `node tests/test_markdown.mjs`, and `git diff --check`; each task commits separately.

---

## File Structure

```text
web_storage.py                 SQLite migration plus conversation/message repository.
web_app.py                     Conversation-scoped CRUD, ownership checks, SSE and locks.
web/static/index.html          Sidebar, dialog, and current-conversation page structure.
web/static/styles.css          Desktop sidebar and narrow-screen conversation bar layout.
web/static/app.js              Conversation state, CRUD interactions, stream target selection.
tests/test_web_storage.py      Repository migration, CRUD, title, order, and cascade tests.
tests/test_web_app.py          API ownership, SSE scope, and validation tests.
tests/test_web_assets.py       Page/assets delivery checks.
docs/v8-multi-conversation.md  Local usage and V8.2 behavior documentation.
```

### Task 1: Add conversation repository and one-time V8.1 migration

**Files:**
- Modify: `web_storage.py`
- Modify: `tests/test_web_storage.py`

**Interfaces:**
- Produces `Conversation(id: str, title: str, updated_at: str)`.
- Produces `list_conversations(session_id: str) -> list[Conversation]`.
- Produces `create_conversation(session_id: str) -> Conversation`.
- Produces `rename_conversation(session_id: str, conversation_id: str, title: str) -> Conversation`.
- Produces `delete_conversation(session_id: str, conversation_id: str) -> bool`.
- Produces `load_conversation_messages(session_id: str, conversation_id: str) -> list[tuple[str, str]]`.
- Produces `save_complete_conversation_turn(session_id: str, conversation_id: str, question: str, answer: str) -> Conversation`.

- [ ] **Step 1: Write the failing repository tests**

```python
def test_legacy_session_messages_migrate_once_into_one_conversation(self):
    self.insert_v81_messages("legacy", [("user", "旧问题"), ("assistant", "旧回答")])
    store = SQLiteConversationStore(self.path)

    conversations = store.list_conversations("legacy")

    self.assertEqual(len(conversations), 1)
    self.assertEqual(conversations[0].title, "已迁移的对话")
    self.assertEqual(store.load_conversation_messages("legacy", conversations[0].id), [
        ("user", "旧问题"), ("assistant", "旧回答"),
    ])
    self.assertEqual(store.list_conversations("legacy"), conversations)

def test_create_rename_auto_title_and_delete_are_session_scoped(self):
    first = self.store.ensure_session(None)
    other = self.store.ensure_session(None)
    conversation = self.store.create_conversation(first)
    updated = self.store.save_complete_conversation_turn(
        first, conversation.id, "试用期最长多久？", "最长六个月。"
    )

    self.assertEqual(updated.title, "试用期最长多久？")
    self.assertEqual(
        self.store.rename_conversation(first, conversation.id, "试用期咨询").title,
        "试用期咨询",
    )
    self.store.save_complete_conversation_turn(first, conversation.id, "那工资呢？", "按规定支付。")
    self.assertEqual(self.store.list_conversations(other), [])
    self.assertFalse(self.store.delete_conversation(other, conversation.id))
    self.assertTrue(self.store.delete_conversation(first, conversation.id))
    self.assertEqual(self.store.list_conversations(first), [])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_web_storage -v`

Expected: FAIL because conversation repository methods and migration do not exist.

- [ ] **Step 3: Implement schema, migration, and repository methods**

```python
@dataclass(frozen=True)
class Conversation:
    id: str
    title: str
    updated_at: str

def create_conversation(self, session_id):
    conversation = Conversation(secrets.token_urlsafe(24), "新对话", _utc_now())
    with self._connection() as connection:
        connection.execute(
            "INSERT INTO conversations(id, session_id, title, title_is_custom, created_at, updated_at) "
            "VALUES (?, ?, ?, 0, ?, ?)",
            (conversation.id, session_id, conversation.title, conversation.updated_at, conversation.updated_at),
        )
    return conversation

def delete_conversation(self, session_id, conversation_id):
    with self._connection() as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ? AND session_id = ?",
            (conversation_id, session_id),
        )
    return cursor.rowcount == 1
```

Create `schema_migrations`, `conversations`, and `conversation_messages` in `_connection`; foreign keys must be enabled before every transaction. Implement the `v8_to_v82_conversations` migration inside one transaction: find legacy sessions with V8.1 `messages`, create one UUID-like opaque conversation per session titled `已迁移的对话`, copy each row by ordinal, then insert the migration marker. Use `conversation_messages` for every new read/write. Generate a 24-character title preview from whitespace-normalized first question; append `…` only when longer. `save_complete_conversation_turn` updates `updated_at` and auto-title only where `title_is_custom = 0 AND title = '新对话'`.

- [ ] **Step 4: Run focused storage tests**

Run: `python -m unittest tests.test_web_storage -v`

Expected: PASS; migration happens once, foreign ownership cannot delete, and deleting the owner conversation cascades its messages.

- [ ] **Step 5: Commit**

```text
git add web_storage.py tests/test_web_storage.py
git commit -m "feat: add persistent conversation repository"
```

### Task 2: Replace session-wide Web API with conversation-scoped API

**Files:**
- Modify: `web_app.py`
- Modify: `tests/test_web_app.py`

**Interfaces:**
- Consumes the Task 1 repository methods.
- Produces `GET /api/conversations`, `POST /api/conversations`, `PATCH /api/conversations/{conversation_id}`, `DELETE /api/conversations/{conversation_id}`, `GET /api/conversations/{conversation_id}/messages`, and `POST /api/conversations/{conversation_id}/chat`.
- Removes V8.1 page use of `GET /api/history` and `POST /api/chat`.

- [ ] **Step 1: Write failing API tests**

```python
def test_conversation_crud_is_cookie_scoped(self):
    first = TestClient(self.app)
    second = TestClient(self.app)
    created = first.post("/api/conversations").json()
    conversation_id = created["id"]

    self.assertEqual(first.patch(
        f"/api/conversations/{conversation_id}", json={"title": "工资咨询"}
    ).json()["title"], "工资咨询")
    self.assertEqual(second.get("/api/conversations").json()["conversations"], [])
    self.assertEqual(second.delete(f"/api/conversations/{conversation_id}").status_code, 404)

def test_stream_reads_and_persists_only_the_selected_conversation(self):
    client = TestClient(self.app)
    first = client.post("/api/conversations").json()["id"]
    second = client.post("/api/conversations").json()["id"]
    consume_sse(client.post(f"/api/conversations/{first}/chat", json={"question": "试用期？"}))

    self.assertEqual(client.get(f"/api/conversations/{second}/messages").json()["messages"], [])
    self.assertEqual(self.rag_turn.received_histories[-1], [])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_web_app -v`

Expected: FAIL with 404 because conversation endpoints do not exist.

- [ ] **Step 3: Implement routes, validation, and conversation locks**

```python
@app.post("/api/conversations", status_code=201)
def create_conversation(request: Request):
    session_id, response = session_response(request)
    conversation = store.create_conversation(session_id)
    return response({"id": conversation.id, "title": conversation.title, "updated_at": conversation.updated_at})

@app.post("/api/conversations/{conversation_id}/chat")
def chat(conversation_id: str, payload: ChatRequest, request: Request):
    session_id, is_new = session_for(request)
    if not store.conversation_belongs_to(session_id, conversation_id):
        raise HTTPException(status_code=404, detail="对话不存在")
    lock = acquire_conversation_lock(conversation_id)
    # Stream with load_conversation_messages and save_complete_conversation_turn.
```

Use a response helper so every route sets the cookie only when a new session was allocated. Convert request data into `ConversationResponse` and `RenameConversationRequest` Pydantic models; reject titles with only whitespace or more than 80 characters at HTTP 422. Return 204 with no body for successful deletion; return 404 if ownership lookup fails. Replace locks keyed by session with locks keyed by conversation. Preserve V8.1 SSE event ordering, safe errors, full-turn transaction timing, and no raw exception output.

- [ ] **Step 4: Run focused API and streaming tests**

Run: `python -m unittest tests.test_web_app tests.test_web_rag -v`

Expected: PASS; a conversation’s SSE call never loads or writes another conversation’s messages.

- [ ] **Step 5: Commit**

```text
git add web_app.py tests/test_web_app.py
git commit -m "feat: add conversation scoped web API"
```

### Task 3: Build the responsive conversation sidebar and controls

**Files:**
- Modify: `web/static/index.html`
- Modify: `web/static/styles.css`
- Modify: `web/static/app.js`
- Modify: `tests/test_web_assets.py`

**Interfaces:**
- Consumes Task 2 JSON APIs and SSE endpoint.
- Produces `currentConversationId` browser state and `loadConversations()`, `selectConversation(id)`, `createConversation()`, `renameConversation(id, title)`, `deleteConversation(id)` functions.

- [ ] **Step 1: Write failing delivery and parser tests**

```python
def test_root_serves_conversation_navigation_controls(self):
    response = self.client.get("/")

    self.assertIn('id="new-conversation"', response.text)
    self.assertIn('id="conversation-list"', response.text)
    self.assertIn('id="current-conversation-title"', response.text)
```

```javascript
import { nextConversationAfterDelete } from "../web/static/conversation_state.mjs";
assert.equal(nextConversationAfterDelete(["a", "b", "c"], "b"), "c");
assert.equal(nextConversationAfterDelete(["a"], "a"), null);
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_web_assets -v`

Expected: FAIL because the sidebar elements are absent.

Run: `node tests/test_conversation_state.mjs`

Expected: FAIL because `conversation_state.mjs` does not exist.

- [ ] **Step 3: Implement sidebar markup, state module, and client operations**

```javascript
async function selectConversation(id) {
  if (isGenerating || id === currentConversationId) return;
  currentConversationId = id;
  clearMessageLog();
  const response = await fetch(`/api/conversations/${id}/messages`);
  const { messages } = await response.json();
  messages.forEach(({ role, content }) => appendMessage(role, content));
  renderConversationList();
}

async function deleteConversation(id) {
  if (!window.confirm("删除后无法恢复这段对话，确定删除吗？")) return;
  const nextId = nextConversationAfterDelete(conversations.map(({ id }) => id), id);
  await fetch(`/api/conversations/${id}`, { method: "DELETE" });
  await loadConversations();
  if (nextId) await selectConversation(nextId);
  else await createConversation();
}
```

Use semantic `aside`, navigation list, visible “新建对话” button, and text-labelled rename/delete controls. Use an input only during inline renaming; Enter saves, Escape restores the old title, blur saves a nonempty title. Disable `new`, selection, rename, and delete while `isGenerating` is true. Change the chat POST to `/api/conversations/${currentConversationId}/chat`. On screens narrower than 760px, turn the side rail into a horizontally scrollable top conversation strip; preserve a 44px minimum control target and visible focus rings.

- [ ] **Step 4: Run web asset and JavaScript state tests**

Run: `python -m unittest tests.test_web_assets -v`

Expected: PASS; root exposes the required semantic sidebar controls.

Run: `node tests/test_markdown.mjs; node tests/test_conversation_state.mjs`

Expected: PASS; existing safe Markdown parsing and post-delete selection both remain correct.

- [ ] **Step 5: Commit**

```text
git add web/static/index.html web/static/styles.css web/static/app.js web/static/conversation_state.mjs tests/test_web_assets.py tests/test_conversation_state.mjs
git commit -m "feat: add multi conversation chat sidebar"
```

### Task 4: Document V8.2 operation and verify migrated Web RAG end-to-end

**Files:**
- Create: `docs/v8-multi-conversation.md`
- Modify: `docs/v8-web-rag-mvp.md`
- Modify: `tests/test_web_app.py`

**Interfaces:**
- Documents the V8.2 command (`uvicorn web_app:app --reload --host 127.0.0.1 --port 8001`), automatic migration, permanent delete behavior, and SQLite-to-PostgreSQL boundary.

- [ ] **Step 1: Write operator documentation and perform a manual smoke test**

Document the migration on first start, the anonymous-cookie scope, automatic/manual title behavior, permanent confirmed deletion, port 8001 command, and the fact that V8.2 remains unsuitable for unrestricted public internet. Start the server, create two conversations in one browser, send one question in each, verify each follow-up rewrites only against its own history, rename one, delete it after confirmation, refresh, and verify the remaining conversation persists.

- [ ] **Step 2: Run complete verification**

Run: `node tests/test_markdown.mjs; node tests/test_conversation_state.mjs; python -m unittest discover -s tests -v`

Expected: PASS for all existing tests and new V8.2 repository/API/UI tests.

Run: `python -m compileall -q web_app.py web_storage.py web_rag.py; git diff --check`

Expected: both commands exit 0.

- [ ] **Step 3: Commit and push**

```text
git add docs/v8-multi-conversation.md docs/v8-web-rag-mvp.md tests/test_web_app.py tests/test_web_storage.py
git commit -m "v8.2: add persistent multi conversation web RAG"
git push origin main
```
