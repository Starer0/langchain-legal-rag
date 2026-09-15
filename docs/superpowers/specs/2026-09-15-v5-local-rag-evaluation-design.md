# V5 Local RAG Evaluation Design

**Goal:** Add a local, repeatable evaluation workflow that measures retrieval and reranking quality for the labor-law RAG system, records source-grounded answer reviews, and teaches the user through the first five cases.

## Scope

V5 evaluates the current `data/labor_law.pdf` knowledge base only. It does not provide legal advice, call LangSmith, alter retrieval behavior, add score thresholds, or add another LLM as an automatic judge.

The initial dataset contains 12 cases. It covers direct-answer questions, paraphrased questions, questions requiring more than one article, likely reranking traps, unsupported questions, and questions with a false premise.

## Evaluation contract

Each case stores:

```json
{
  "id": "overtime-pay",
  "question": "加班工资如何计算？",
  "expected_articles": ["第四十四条"],
  "required_facts": [
    "延长工作时间不低于150%",
    "休息日不能补休不低于200%",
    "法定休假日不低于300%"
  ],
  "answerable": true
}
```

For an unsupported question, `expected_articles` and `required_facts` are empty and `answerable` is `false`.

The evaluator automatically records whether every expected article appears in Chroma candidates (Top K) and in the reranked sources (Top N). For unsupported cases it also records whether the answer contains the existing refusal phrase, `资料中没有足够依据`.

## Source-grounded answer review

The assistant reviews each generated answer against the saved case expectations and the cited article text in `data/labor_law.pdf`. The review is deliberately scoped to faithfulness to this PDF, not real-world legal accuracy outside the corpus.

Each result receives one verdict:

- `complete`: the answer supports every required fact and makes no unsupported conclusion.
- `partial`: the central conclusion is supported but at least one required fact is absent, incomplete, or materially unclear.
- `incorrect`: the answer contradicts the PDF, relies on unsupported content, or gives an incorrect refusal for an answerable case.
- `correct_refusal`: an unsupported case is refused without inventing a rule.
- `incorrect_refusal`: an answerable case is refused despite its expected article appearing in the PDF.

Each review stores the verdict, the supporting article identifiers, and a short explanation tied to the corpus.

## Architecture

The current terminal interface and evaluator must use the same RAG construction logic. Startup code from `main.py` moves into a small factory module that exposes `create_rag_chain()`. `main.py` remains the interactive terminal entry point and imports that factory. The evaluator imports the same factory, so environment configuration, Chroma retrieval, reranking, and prompt formatting stay identical.

New components:

```text
rag_app.py
  Shared create_rag_chain() factory.

evals/cases.json
  Version-controlled 12-case source-grounded evaluation dataset.

evaluate.py
  Runs one named case against the real shared chain, records retrieval metrics,
  the answer, configuration, candidates, and reranked sources.

evals/results/<run_id>/<case_id>.json
  Immutable raw run artifact plus the assistant's review.

evals/results/<run_id>/summary.json
  Aggregate metrics and verdict counts for a completed run.
```

The evaluator must not write API keys, complete document text, or unrelated environment variables to result files. It records only case data, article identifiers, page numbers, source snippets, scores, answers, evaluation configuration, timestamps, and reviews.

## User workflow

1. Run cases one through five in order.
2. After each case, show the user the answer, Top K candidates, Top N sources, and source-grounded review.
3. The user may ask questions or flag a concern; no legal knowledge or scoring is required from them.
4. If the first five cases raise no issue, run cases six through twelve without pausing, write reviews, and generate the summary.
5. Use the summary as the V5 baseline for later comparisons against retrieval changes.

## Testing

Unit tests use injected fake RAG-chain results and never call Chroma, the chat model, or the reranker API. Tests cover article-hit metrics, refusal detection, result serialization without secrets, verdict aggregation, and the shared-factory interface. The existing reranker tests must remain green.

## Success criteria

- A single case can be evaluated through the same chain as `main.py`.
- Each result identifies expected-article hits in candidates and sources.
- Every result includes enough source metadata to audit its review.
- The first five cases can be reviewed conversationally; the final seven can be completed unattended.
- The final summary provides Top K/Top N hit rates, refusal outcomes, and answer-review verdict counts.
