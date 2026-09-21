"""Compare no-history and history-aware retrieval on local multi-turn cases."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage


CASES_PATH = Path("evals/multiturn_cases.json")
RESULTS_DIR = Path("evals/results")
REQUIRED_CASE_FIELDS = {
    "id": str,
    "history": list,
    "question": str,
    "expected_articles": list,
    "expected_retrieval_terms": list,
}


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Multi-turn cases must be a JSON array")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each multi-turn case must be an object")
        for field, expected_type in REQUIRED_CASE_FIELDS.items():
            if not isinstance(case.get(field), expected_type):
                raise ValueError(f"Multi-turn case has invalid {field!r}")
    return cases


def messages_from_history(history: list[dict]) -> list:
    messages = []
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str) or role not in {"human", "ai"}:
            raise ValueError("History messages require human/ai role and string content")
        message_class = HumanMessage if role == "human" else AIMessage
        messages.append(message_class(content=content))
    return messages


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _article_hit(expected_articles: list[str], documents: list[dict]) -> bool:
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


def _terms_present(retrieval_question: str, expected_terms: list[str]) -> bool:
    normalized = retrieval_question.lower()
    return all(term.lower() in normalized for term in expected_terms)


def invoke_mode(chain, rewriter, question: str, history: list, case: dict) -> dict:
    retrieval_question = rewriter.rewrite(question, history)
    chain_result = dict(chain.invoke({
        "question": question,
        "retrieval_question": retrieval_question,
    }))
    candidates = chain_result.get("candidates", [])
    sources = chain_result.get("sources", [])
    return {
        "retrieval_question": retrieval_question,
        "answer": chain_result.get("answer"),
        "candidates": candidates,
        "sources": sources,
        "metrics": {
            "expected_articles_in_candidates": _article_hit(
                case["expected_articles"], candidates
            ),
            "expected_articles_in_sources": _article_hit(
                case["expected_articles"], sources
            ),
            "expected_retrieval_terms_present": _terms_present(
                retrieval_question, case["expected_retrieval_terms"]
            ),
        },
    }


def _mode_summary(results: list[dict], mode: str) -> dict:
    total_cases = len(results)
    metrics = [result[mode]["metrics"] for result in results]
    return {
        "candidate_hits": sum(item["expected_articles_in_candidates"] for item in metrics),
        "source_hits": sum(item["expected_articles_in_sources"] for item in metrics),
        "retrieval_term_hits": sum(
            item["expected_retrieval_terms_present"] for item in metrics
        ),
        "total_cases": total_cases,
    }


def run_all_cases(
    run_id: str,
    results_dir: Path = RESULTS_DIR,
    cases: list[dict] | None = None,
    chain=None,
    rewriter=None,
) -> Path:
    """Run every case twice and save comparable no-history/history-aware records."""
    run_directory = results_dir / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)
    cases = load_cases() if cases is None else cases

    if chain is None or rewriter is None:
        from rag_app import create_conversation_service

        service = create_conversation_service()
        chain = service.rag_chain if chain is None else chain
        rewriter = service.rewriter if rewriter is None else rewriter

    results = []
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case['id']}")
        history = messages_from_history(case["history"])
        result = {
            "case": case,
            "run_id": run_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "no_history": invoke_mode(chain, rewriter, case["question"], [], case),
            "history_aware": invoke_mode(
                chain, rewriter, case["question"], history, case
            ),
        }
        _write_json(run_directory / f"{case['id']}.json", result)
        results.append(result)

    summary_path = run_directory / "summary.json"
    _write_json(summary_path, {
        "total_cases": len(results),
        "no_history": _mode_summary(results, "no_history"),
        "history_aware": _mode_summary(results, "history_aware"),
    })
    return summary_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    return run_all_cases(args.run_id)


if __name__ == "__main__":
    main()
