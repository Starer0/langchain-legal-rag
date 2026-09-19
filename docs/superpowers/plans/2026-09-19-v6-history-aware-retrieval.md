# V6 History-aware Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded in-memory conversation history and history-aware query rewriting while retaining the existing Chroma Top 8, reranker Top 4, and original-question answer flow.

**Architecture:** A new rewrite module converts either a standalone question or trimmed `ChatMessageHistory` plus a follow-up into `retrieval_question`. A conversation service owns history, invokes the existing RAG chain with `{question, retrieval_question}`, and records a complete turn only after success. The RAG pipeline retrieves and reranks by `retrieval_question` but prompts the answer model with `question`.

**Tech Stack:** Python 3, LangChain Core (`ChatPromptTemplate`, `MessagesPlaceholder`, `ChatMessageHistory`, Runnables), Chroma, SiliconFlow reranker, `unittest` and `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-09-19-v6-history-aware-retrieval-design.md`

## Global Constraints

- Preserve `Chroma Top 8 -> query rewrite -> Reranker Top 4 -> answer`; do not introduce BM25, metadata filters, agents, persistent sessions, streaming, or LangSmith changes.
- Use `ChatMessageHistory` and retain only the latest complete user/assistant turns; default `HISTORY_TURNS=4`.
- Retrieval and reranking must consume `retrieval_question`; answer generation must consume the original `question`.
- History is conversational context, never legal evidence. The rewrite prompt must say that previous assistant messages are not authoritative sources.
- Do not add a user message or assistant message to history after any failed rewrite, retrieval, rerank, or answer invocation.
- Keep `.env` and local Chroma databases out of Git.
- Run `python -m unittest discover -s tests -v` before every final v6 commit.

---

## File Structure

- Create `query_rewrite.py`: prompt constants and `RetrievalQuestionRewriter`, which selects single-turn or history-aware rewriting.
- Create `conversation.py`: bounded-history helpers and `ConversationRagService`, the transaction boundary for one conversational turn.
- Modify `rag_pipeline_articles.py`: accept state dictionaries and separate original answer question from retrieval question.
- Modify `rag_app.py`: create a rewrite model and expose a fully wired conversation service factory.
- Modify `main.py`: remove import-time initialization and run the multi-turn CLI through the service.
- Create `evals/multiturn_cases.json`: six local, inspectable multi-turn challenge cases.
- Create `run_multiturn_evaluation.py`: run the same chain in no-history and history-aware modes and serialize comparison records.
- Create `tests/test_query_rewrite.py`, `tests/test_conversation.py`, `tests/test_multiturn_evaluation.py`, and extend `tests/test_reranker.py` and `tests/test_rag_app.py`.

### Task 1: Isolated query rewriting

**Files:**
- Create: `query_rewrite.py`
- Create: `tests/test_query_rewrite.py`

**Interfaces:**
- Produces `class RetrievalQuestionRewriter(model)`.
- Produces `rewrite(question: str, history: Sequence[BaseMessage]) -> str`.
- `history=[]` selects the standalone prompt; non-empty history selects the history-aware prompt.
- Empty string output returns `question.strip()`; model errors propagate.

- [ ] **Step 1: Write failing tests for path selection and blank-output fallback**

```python
class CapturingModel:
    def __init__(self, output):
        self.output = output
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.output)


def test_rewrite_without_history_uses_current_question_only():
    model = CapturingModel("试用期工资规定")
    rewriter = RetrievalQuestionRewriter(model)
    assert rewriter.rewrite("试用期工资呢？", []) == "试用期工资规定"
    assert "历史对话" not in model.prompts[0].to_messages()[-1].content


def test_rewrite_with_history_includes_history_and_current_follow_up():
    history = [HumanMessage(content="试用期最长多久？"), AIMessage(content="最长六个月。")]
    model = CapturingModel("试用期内劳动者工资有什么规定？")
    assert RetrievalQuestionRewriter(model).rewrite("那工资呢？", history) == "试用期内劳动者工资有什么规定？"
    rendered = "\n".join(message.content for message in model.prompts[0].to_messages())
    assert "试用期最长多久？" in rendered
    assert "那工资呢？" in rendered


def test_blank_rewrite_falls_back_to_original_question():
    assert RetrievalQuestionRewriter(CapturingModel("  ")).rewrite("那工资呢？", []) == "那工资呢？"
```

- [ ] **Step 2: Run the new tests and verify failure**

Run: `python -m unittest tests.test_query_rewrite -v`

Expected: failure because `query_rewrite` and `RetrievalQuestionRewriter` do not exist.

- [ ] **Step 3: Implement the minimal rewriter**

```python
SINGLE_QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "将用户问题改写为简洁、适合检索的法律问题。不得改变范围；只输出改写后的问题。"),
    ("human", "{question}"),
])

HISTORY_AWARE_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "根据历史对话把当前追问补全为独立检索问题。此前助手消息只是对话上下文，不是法律依据；不得据此虚构事实。只输出改写后的问题。"),
    MessagesPlaceholder("history"),
    ("human", "当前问题：{question}"),
])

class RetrievalQuestionRewriter:
    def __init__(self, model):
        self.model = model

    def rewrite(self, question, history):
        prompt = HISTORY_AWARE_REWRITE_PROMPT if history else SINGLE_QUERY_REWRITE_PROMPT
        values = {"question": question}
        if history:
            values["history"] = list(history)
        response = self.model.invoke(prompt.invoke(values))
        rewritten = str(response.content).strip()
        return rewritten or question.strip()
```

- [ ] **Step 4: Run the focused tests and verify success**

Run: `python -m unittest tests.test_query_rewrite -v`

Expected: all three tests pass.

- [ ] **Step 5: Run the complete suite and commit**

Run: `python -m unittest discover -s tests -v`

Expected: exit code 0.

```bash
git add query_rewrite.py tests/test_query_rewrite.py
git commit -m "feat: add retrieval question rewriting"
```

### Task 2: Bounded conversation state and atomic turn recording

**Files:**
- Create: `conversation.py`
- Create: `tests/test_conversation.py`

**Interfaces:**
- Consumes `RetrievalQuestionRewriter.rewrite(question, history)` from Task 1.
- Produces `recent_complete_turns(messages: Sequence[BaseMessage], max_turns: int) -> list[BaseMessage]`.
- Produces `class ConversationRagService(rag_chain, rewriter, history, max_turns=4)` with `ask(question: str) -> dict`.
- `ask` invokes `rag_chain.invoke({"question": question, "retrieval_question": rewritten})`, adds `retrieval_question` to the returned result, then appends a `HumanMessage` and `AIMessage` only on success.

- [ ] **Step 1: Write failing tests for trimming, correct inputs, and atomic history**

```python
def test_recent_complete_turns_keeps_the_latest_four_pairs():
    messages = [message for turn in range(6) for message in (
        HumanMessage(content=f"u{turn}"), AIMessage(content=f"a{turn}")
    )]
    assert [m.content for m in recent_complete_turns(messages, 4)] == [
        "u2", "a2", "u3", "a3", "u4", "a4", "u5", "a5"
    ]


def test_ask_uses_history_for_rewrite_but_keeps_original_answer_question():
    history = ChatMessageHistory()
    history.add_user_message("试用期最长多久？")
    history.add_ai_message("最长六个月。")
    rewriter = Mock()
    rewriter.rewrite.return_value = "试用期内劳动者工资有什么规定？"
    chain = Mock()
    chain.invoke.return_value = {"answer": "应按规定支付工资。"}
    service = ConversationRagService(chain, rewriter, history)
    result = service.ask("那工资呢？")
    rewriter.rewrite.assert_called_once_with("那工资呢？", history.messages)
    chain.invoke.assert_called_once_with({
        "question": "那工资呢？",
        "retrieval_question": "试用期内劳动者工资有什么规定？",
    })
    assert result["retrieval_question"] == "试用期内劳动者工资有什么规定？"
    assert [m.content for m in history.messages[-2:]] == ["那工资呢？", "应按规定支付工资。"]


def test_failed_chain_does_not_append_a_partial_turn():
    history = ChatMessageHistory()
    chain = Mock()
    chain.invoke.side_effect = RuntimeError("reranker unavailable")
    with pytest.raises(RuntimeError):
        ConversationRagService(chain, Mock(), history).ask("那工资呢？")
    assert history.messages == []
```

Use `unittest.TestCase.assertRaises` rather than `pytest.raises` in the actual repository tests.

- [ ] **Step 2: Run the new tests and verify failure**

Run: `python -m unittest tests.test_conversation -v`

Expected: failure because `conversation` does not exist.

- [ ] **Step 3: Implement the state helper and service**

```python
def recent_complete_turns(messages, max_turns):
    if max_turns < 1:
        raise ValueError("max_turns 必须大于等于 1")
    retained = list(messages)[-(max_turns * 2):]
    return retained if len(retained) % 2 == 0 else retained[1:]

class ConversationRagService:
    def __init__(self, rag_chain, rewriter, history, max_turns=4):
        self.rag_chain = rag_chain
        self.rewriter = rewriter
        self.history = history
        self.max_turns = max_turns

    def ask(self, question):
        original_question = question.strip()
        if not original_question:
            raise ValueError("问题不能为空")
        history = recent_complete_turns(self.history.messages, self.max_turns)
        retrieval_question = self.rewriter.rewrite(original_question, history)
        result = dict(self.rag_chain.invoke({
            "question": original_question,
            "retrieval_question": retrieval_question,
        }))
        result["retrieval_question"] = retrieval_question
        self.history.add_user_message(original_question)
        self.history.add_ai_message(result["answer"])
        return result
```

- [ ] **Step 4: Run focused tests and verify success**

Run: `python -m unittest tests.test_conversation -v`

Expected: all tests pass, including failure-path history preservation.

- [ ] **Step 5: Run the complete suite and commit**

Run: `python -m unittest discover -s tests -v`

Expected: exit code 0.

```bash
git add conversation.py tests/test_conversation.py
git commit -m "feat: add bounded conversation history"
```

### Task 3: Separate retrieval and answer questions inside the RAG chain

**Files:**
- Modify: `rag_pipeline_articles.py:115-143`
- Modify: `tests/test_reranker.py`
- Modify: `tests/test_rag_app.py`

**Interfaces:**
- Consumes `{"question": str, "retrieval_question": str}` from `ConversationRagService`.
- `build_rag_chain(retriever, reranker, prompt, model)` returns a runnable accepting this state dictionary.
- Produces result keys `answer`, `candidates`, and `sources`; `ConversationRagService` owns the additional `retrieval_question` key.

- [ ] **Step 1: Write a failing pipeline test**

```python
def test_chain_retrieves_with_rewritten_question_and_answers_original_question():
    retriever = RunnableLambda(lambda question: [Document(page_content=question)])
    reranker = RunnableLambda(lambda state: state["candidates"])
    prompt = ChatPromptTemplate.from_template("context={context}; question={question}")
    model = RunnableLambda(lambda prompt_value: AIMessage(content=prompt_value.to_string()))
    result = build_rag_chain(retriever, reranker, prompt, model).invoke({
        "question": "那工资呢？",
        "retrieval_question": "试用期内劳动者工资有什么规定？",
    })
    assert "试用期内劳动者工资有什么规定？" in result["candidates"][0]["content"]
    assert "question=那工资呢？" in result["answer"]
```

- [ ] **Step 2: Run the focused pipeline test and verify failure**

Run: `python -m unittest tests.test_reranker.RagChainWithRerankerTests.test_chain_retrieves_with_rewritten_question_and_answers_original_question -v`

Expected: failure because the current chain passes its entire input state to the retriever or does not preserve the original question correctly.

- [ ] **Step 3: Make the minimal state wiring change**

```python
original_question_from_state = RunnableLambda(itemgetter("question"))
retrieval_question_from_state = RunnableLambda(itemgetter("retrieval_question"))
state = (
    RunnablePassthrough()
    | RunnablePassthrough.assign(
        candidates=retrieval_question_from_state | retriever
    )
    | RunnablePassthrough.assign(docs=reranker)
)
answer_chain = {
    "context": docs_from_state | format_docs,
    "question": original_question_from_state,
} | prompt | model | StrOutputParser()
```

- [ ] **Step 4: Update existing `rag_app` tests to invoke the state-based chain and run all affected tests**

Run: `python -m unittest tests.test_reranker tests.test_rag_app tests.test_no_reranker -v`

Expected: all affected tests pass.

- [ ] **Step 5: Run the complete suite and commit**

Run: `python -m unittest discover -s tests -v`

Expected: exit code 0.

```bash
git add rag_pipeline_articles.py tests/test_reranker.py tests/test_rag_app.py tests/test_no_reranker.py
git commit -m "refactor: separate retrieval and answer questions"
```

### Task 4: Wire the history-aware service into the command-line application

**Files:**
- Modify: `rag_app.py`
- Modify: `main.py`
- Modify: `tests/test_rag_app.py`
- Create: `tests/test_main.py`

**Interfaces:**
- `create_rag_chain()` continues to build the configured underlying RAG runnable.
- Add `create_conversation_service()` in `rag_app.py`, which creates `ChatMessageHistory`, `RetrievalQuestionRewriter`, and `ConversationRagService` with `HISTORY_TURNS=int(os.getenv("HISTORY_TURNS", "4"))`.
- Add `run_cli(service, input_fn=input, output_fn=print)` in `main.py`.

- [ ] **Step 1: Write failing factory and CLI tests**

```python
@patch.dict("os.environ", {"HISTORY_TURNS": "3"}, clear=False)
@patch("rag_app.RetrievalQuestionRewriter")
@patch("rag_app.create_rag_chain")
def test_create_conversation_service_uses_configured_turn_limit(create_chain, rewriter):
    service = create_conversation_service()
    assert service.max_turns == 3
    create_chain.assert_called_once()


def test_run_cli_prints_retrieval_question_and_uses_service():
    service = Mock()
    service.ask.return_value = {
        "retrieval_question": "试用期内劳动者工资有什么规定？",
        "answer": "应按规定支付工资。",
        "candidates": [],
        "sources": [],
    }
    answers = iter(["那工资呢？", "q"])
    output = []
    run_cli(service, input_fn=lambda _: next(answers), output_fn=output.append)
    service.ask.assert_called_once_with("那工资呢？")
    assert any("检索问题：试用期内劳动者工资有什么规定？" in text for text in output)
```

- [ ] **Step 2: Run the new tests and verify failure**

Run: `python -m unittest tests.test_rag_app tests.test_main -v`

Expected: failure because the factory and `run_cli` do not exist.

- [ ] **Step 3: Implement the factory and lazy CLI initialization**

```python
def create_conversation_service():
    history_turns = int(os.getenv("HISTORY_TURNS", "4"))
    rewrite_model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    return ConversationRagService(
        rag_chain=create_rag_chain(),
        rewriter=RetrievalQuestionRewriter(rewrite_model),
        history=ChatMessageHistory(),
        max_turns=history_turns,
    )

if __name__ == "__main__":
    run_cli(create_conversation_service())
```

In `run_cli`, ignore `question.strip() == ""`, preserve `q` exit behavior, call `service.ask(question)`, and print `检索问题：{result['retrieval_question']}` before candidates, answer, and sources.

- [ ] **Step 4: Run focused tests and verify success**

Run: `python -m unittest tests.test_rag_app tests.test_main -v`

Expected: all factory and CLI tests pass without loading PDFs at module import time.

- [ ] **Step 5: Run the complete suite and commit**

Run: `python -m unittest discover -s tests -v`

Expected: exit code 0.

```bash
git add rag_app.py main.py tests/test_rag_app.py tests/test_main.py
git commit -m "feat: add history-aware CLI retrieval"
```

### Task 5: Add local multi-turn comparison evaluation

**Files:**
- Create: `evals/multiturn_cases.json`
- Create: `run_multiturn_evaluation.py`
- Create: `tests/test_multiturn_evaluation.py`

**Interfaces:**
- Case schema: `id`, `history` (ordered `{role, content}` objects), `question`, `expected_articles`, and `expected_retrieval_terms`.
- Produces `run_all_cases(run_id, results_dir=Path("evals/results")) -> Path`.
- Writes `<results_dir>/<run_id>/<case-id>.json` and `<results_dir>/<run_id>/summary.json`.
- Each case result contains `no_history` and `history_aware`, each with `retrieval_question`, `candidates`, `sources`, `answer`, and article-hit metrics.

- [ ] **Step 1: Write a failing runner test and the challenge dataset**

```python
CASE = {
    "id": "trial-period-wage-follow-up",
    "history": [
        {"role": "human", "content": "试用期最长多久？"},
        {"role": "ai", "content": "试用期有相应期限限制。"},
    ],
    "question": "那工资呢？",
    "expected_articles": ["第二十条"],
    "expected_retrieval_terms": ["试用期", "工资"],
}

def test_runner_writes_both_modes_for_each_case(tmp_path):
    chain = Mock()
    chain.invoke.return_value = {"answer": "回答", "candidates": [], "sources": []}
    rewriter = Mock(side_effect=["工资规定", "试用期工资规定"])
    summary_path = run_all_cases("history-test", results_dir=tmp_path, cases=[CASE], chain=chain, rewriter=rewriter)
    result = json.loads((tmp_path / "history-test" / "trial-period-wage-follow-up.json").read_text(encoding="utf-8"))
    assert result["no_history"]["retrieval_question"] == "工资规定"
    assert result["history_aware"]["retrieval_question"] == "试用期工资规定"
    assert summary_path.exists()
```

Use `tempfile.TemporaryDirectory()` rather than `tmp_path` in the actual `unittest` test.

Create six cases: trial-period wage, National Day overtime, arbitration deadline, probation termination, rest-day overtime, and minimum-wage follow-up.

- [ ] **Step 2: Run the new test and verify failure**

Run: `python -m unittest tests.test_multiturn_evaluation -v`

Expected: failure because the runner and dataset do not exist.

- [ ] **Step 3: Implement deterministic local serialization and metrics**

```python
def invoke_mode(chain, rewriter, question, history):
    retrieval_question = rewriter.rewrite(question, history)
    result = dict(chain.invoke({
        "question": question,
        "retrieval_question": retrieval_question,
    }))
    return {
        "retrieval_question": retrieval_question,
        "answer": result["answer"],
        "candidates": result["candidates"],
        "sources": result["sources"],
    }
```

Convert dataset role objects to `HumanMessage` and `AIMessage`, add expected-article hit booleans for candidates and sources per mode, and write JSON with `ensure_ascii=False, indent=2`. Refuse to overwrite an existing run directory, matching the current evaluation runner behavior.

- [ ] **Step 4: Run focused evaluation tests and inspect produced JSON**

Run: `python -m unittest tests.test_multiturn_evaluation -v`

Expected: all tests pass and each result has both `no_history` and `history_aware` records.

- [ ] **Step 5: Run the complete suite and commit**

Run: `python -m unittest discover -s tests -v`

Expected: exit code 0.

```bash
git add evals/multiturn_cases.json run_multiturn_evaluation.py tests/test_multiturn_evaluation.py
git commit -m "feat: add multi-turn retrieval evaluation"
```

### Task 6: Final regression verification and evidence capture

**Files:**
- Modify: none unless a prior verification finds a defect.

**Interfaces:**
- Verifies the completed V6 public CLI, evaluation, and existing single-turn RAG test contracts.

- [ ] **Step 1: Confirm the repository only contains intended V6 changes**

Run: `git status --short` and `git diff origin/v5-reranker-comparison...HEAD --stat`

Expected: only V6 implementation, tests, challenge data, specification, and plan artifacts are present.

- [ ] **Step 2: Run the complete automated suite**

Run: `python -m unittest discover -s tests -v`

Expected: exit code 0 with all existing and V6 tests passing.

- [ ] **Step 3: Run a mocked multi-turn evaluation smoke test**

Run: `python run_multiturn_evaluation.py --help`

Expected: command exits 0 and documents the run identifier argument without making a remote model call.

- [ ] **Step 4: Record the evidence and commit only if a verification fix was necessary**

If verification required a source change, run the full suite again and commit the focused repair with a `fix:` message. Otherwise do not create an empty commit.
