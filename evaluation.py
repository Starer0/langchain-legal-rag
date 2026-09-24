"""Pure helpers for recording and summarizing local RAG evaluations."""

from copy import deepcopy


REFUSAL_PHRASE = "资料中没有足够依据"
PREVIEW_LENGTH = 160
VALID_VERDICTS = {
    "complete",
    "partial",
    "incorrect",
    "correct_refusal",
    "incorrect_refusal",
}


def _serialize_documents(documents):
    serialized = []
    for document in documents:
        item = {
            "article": document.get("article"),
            "pages": deepcopy(document.get("pages")),
            "content_preview": document.get("content", "")[:PREVIEW_LENGTH],
        }
        for field in ("law_id", "law_name", "version", "status"):
            if field in document:
                item[field] = document[field]
        if "rerank_score" in document:
            item["rerank_score"] = document["rerank_score"]
        serialized.append(item)
    return serialized


def _contains_all_articles(expected_articles, documents):
    def contains(expected):
        if isinstance(expected, str):
            return any(document.get("article") == expected for document in documents)
        if isinstance(expected, dict):
            return any(
                all(document.get(key) == value for key, value in expected.items())
                for document in documents
            )
        raise ValueError("预期法条必须是条号字符串或法律身份对象")

    return bool(expected_articles) and all(contains(item) for item in expected_articles)


def build_case_result(case, chain_result, config, run_id, evaluated_at):
    """Build a JSON-safe record for one evaluated RAG case."""
    candidates = _serialize_documents(chain_result.get("candidates", []))
    sources = _serialize_documents(chain_result.get("sources", []))
    expected_articles = case.get("expected_articles", [])
    answerable = case.get("answerable", False)

    result = {
        "case": deepcopy(case),
        "run_id": run_id,
        "evaluated_at": evaluated_at,
        "config": deepcopy(config),
        "answer": chain_result.get("answer"),
        "candidates": candidates,
        "sources": sources,
        "metrics": {
            "expected_articles_in_candidates": _contains_all_articles(
                expected_articles, candidates
            ),
            "expected_articles_in_sources": _contains_all_articles(expected_articles, sources),
            "refusal_phrase_present": (
                REFUSAL_PHRASE in (chain_result.get("answer") or "") if not answerable else False
            ),
        },
        "assistant_review": None,
    }
    if "subquestion_reranks" in chain_result:
        reranks = [
            {
                "question": entry["question"],
                "sources": _serialize_documents(entry["sources"]),
            }
            for entry in chain_result["subquestion_reranks"]
        ]
        result["subquestion_reranks"] = reranks
        result["metrics"]["expected_articles_in_reranks"] = _contains_all_articles(
            expected_articles,
            [source for entry in reranks for source in entry["sources"]],
        )
    return result


def add_assistant_review(result, verdict, evidence_articles, reason):
    """Return a reviewed copy of a case result without mutating the input."""
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict}")

    reviewed = deepcopy(result)
    reviewed["assistant_review"] = {
        "verdict": verdict,
        "evidence_articles": list(evidence_articles),
        "reason": reason,
    }
    return reviewed


def build_summary(results):
    """Aggregate retrieval, refusal, and human-review metrics."""
    total_cases = len(results)
    candidate_hits = sum(
        result["metrics"]["expected_articles_in_candidates"] for result in results
    )
    source_hits = sum(result["metrics"]["expected_articles_in_sources"] for result in results)
    unsupported_cases = sum(not result["case"]["answerable"] for result in results)
    refusal_hits = sum(
        not result["case"]["answerable"] and result["metrics"]["refusal_phrase_present"]
        for result in results
    )
    verdict_counts = {}
    for result in results:
        review = result.get("assistant_review")
        if review is not None:
            verdict = review["verdict"]
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    summary = {
        "total_cases": total_cases,
        "candidate_hits": candidate_hits,
        "candidate_hit_rate": candidate_hits / total_cases if total_cases else 0.0,
        "source_hits": source_hits,
        "source_hit_rate": source_hits / total_cases if total_cases else 0.0,
        "unsupported_cases": unsupported_cases,
        "unsupported_case_rate": unsupported_cases / total_cases if total_cases else 0.0,
        "refusal_hits": refusal_hits,
        "refusal_hit_rate": refusal_hits / unsupported_cases if unsupported_cases else 0.0,
        "verdict_counts": verdict_counts,
    }
    rerank_cases = [
        result for result in results
        if "expected_articles_in_reranks" in result["metrics"]
    ]
    if rerank_cases:
        rerank_hits = sum(
            result["metrics"]["expected_articles_in_reranks"]
            for result in rerank_cases
        )
        summary.update({
            "rerank_evaluated_cases": len(rerank_cases),
            "rerank_hits": rerank_hits,
            "rerank_hit_rate": rerank_hits / len(rerank_cases),
        })
    return summary
