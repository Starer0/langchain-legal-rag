# V6 History-aware Retrieval Design

## Goal

Extend the legal RAG CLI from independent single-turn questions to short, continuous conversations without removing the existing retrieval strategy:

```text
Chroma Top 8 -> single-query rewrite -> reranker Top 4 -> answer
```

For follow-up questions, conversation history supplies context only to create an independent retrieval question. It is not injected directly into Chroma or treated as legal evidence.

## Scope

V6 includes:

- Single-turn query rewrite for every user question.
- `ChatMessageHistory` backed in-memory history for the CLI session.
- History-aware rewrite when prior turns exist.
- Retention of the latest four user/assistant turns, configurable through `HISTORY_TURNS` (default `4`).
- Retrieval with the rewritten standalone question, Chroma Top 8, and reranker Top 4.
- An answer prompt that receives the original user question and retrieved legal sources.
- A local multi-turn challenge set and an evaluation runner that compares no-history retrieval with history-aware retrieval.
- Unit tests for rewrite selection, history trimming, and preservation of the original question for the answer.

V6 explicitly excludes LangSmith trace naming, Dataset/Experiment management, persistent user sessions, web APIs, streaming, metadata filtering, BM25, and agent workflows.

## Data Flow

```text
History + original user question
        |
        +-- no prior history --> single-query rewrite
        |
        +-- prior history ----> history-aware rewrite
                                      |
                                      v
                         standalone retrieval question
                                      |
                                      v
                             Chroma candidates (k=8)
                                      |
                                      v
                              Reranker documents (k=4)
                                      |
                                      v
                  original user question + legal source documents
                                      |
                                      v
                                   final answer
                                      |
                                      v
             append original user question and final answer to history
```

The rewrite result is exposed in the returned result as `retrieval_question`, so it can be inspected in the CLI and evaluation artifacts.

## Components

### Query rewriting

Introduce a small, independently testable rewrite component with two prompt paths:

- **Single-query rewrite:** reformulates the current question into a concise legal-search query without changing scope.
- **History-aware rewrite:** sees trimmed chat messages plus the current question and produces a standalone legal-search query. The prompt states that previous assistant messages are conversational context, not legal authority, and that no facts should be invented.

The history-aware path is selected only when retained history contains messages. Both paths return a plain string suitable for retrieval.

### Conversation state

The command-line entry point owns one `ChatMessageHistory` for the process. It stores the exact user question and final assistant answer only after a successful RAG invocation. Before each invocation, a helper converts history to the newest configured number of complete turns. There is no cross-process persistence in V6.

### RAG pipeline boundary

The pipeline accepts both `question` (the original user wording) and `retrieval_question` (the rewritten standalone query). Retrieval and reranking use `retrieval_question`; the answer prompt uses `question`. The result contains answer, candidates, sources, and `retrieval_question`.

This makes the three concepts observable and prevents an internally rewritten wording from replacing what the user actually asked.

### CLI

The CLI keeps its loop and `q` exit command. For each non-empty question, it prints the standalone retrieval question before the candidate/source lists and answer. Its in-memory history resets when the program exits.

## Evaluation

Add `evals/multiturn_cases.json` with 5-10 named scenarios. Each case holds preceding user/assistant messages, the follow-up question, expected legal articles, and expected retrieval-question concepts. The first set covers trial-period wages, National Day overtime, and arbitration limitation follow-ups.

The evaluation runner invokes the same chain in two modes:

- **No history:** rewrite the follow-up by itself.
- **History-aware:** pass the supplied preceding messages.

Each per-case result records both rewritten questions, candidates, sources, and expected-article hit metrics. Review compares not only answer correctness but whether the history-aware rewrite correctly resolves the referent in the follow-up.

## Error Handling

- Empty user input does not call the model or mutate history.
- If rewrite output is empty, the original question is used as the retrieval question and the fallback is surfaced in logs/result metadata.
- A failed rewrite, retrieval, rerank, or answer call does not append a partial turn to history.
- History remains bounded to complete user/assistant pairs so a failed answer cannot leave an orphan message.

## Testing

Tests use mocked LLMs, retrievers, and rerankers. They verify:

1. a first-turn question uses the single-query rewrite path;
2. a follow-up with history uses the history-aware path;
3. only the retained latest four turns reach the history-aware prompt;
4. retrieval and reranking receive the standalone retrieval question;
5. the answer receives the original user question;
6. failures do not append history;
7. multi-turn evaluation emits comparable artifacts for no-history and history-aware modes.

## Completion Criteria

V6 is complete when a user can ask, for example, "试用期最长多久？" followed by "那工资呢？", inspect a standalone retrieval question about trial-period wages, retrieve and rerank relevant legal text, and receive an answer to the original follow-up. The local multi-turn challenge set must demonstrate the rewrite difference and be covered by passing automated tests.
