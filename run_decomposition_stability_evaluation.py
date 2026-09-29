"""Evaluate whether complex questions are consistently decomposed into expected counts."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from legal_corpus import load_guides
from query_decomposition import CompositeQuestionDecomposer
from query_rewrite import RetrievalQuestionRewriter


CASES_PATH = Path("evals/decomposition_stability_cases.json")
RESULTS_DIR = Path("evals/results")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_cases(path):
    with Path(path).open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("拆分评测题必须是 JSON 对象数组")
    return cases


def _build_summary(results):
    total_cases = len(results)
    count_matches = sum(item["metrics"]["subquestion_count_matches"] for item in results)
    return {
        "total_cases": total_cases,
        "count_matches": count_matches,
        "count_match_rate": count_matches / total_cases if total_cases else 0.0,
    }


def run_decomposition_evaluation(
    run_id, *, cases_path=CASES_PATH, results_dir=RESULTS_DIR,
    decomposer_factory=None, rewriter_factory=None, rewrite_first=False,
):
    """Run each development question through the decomposer and save its output."""
    run_directory = Path(results_dir) / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)

    load_dotenv()
    cases = _load_cases(cases_path)
    if any(not isinstance(case.get("expected_subquestion_count"), int) for case in cases):
        raise ValueError("每个拆分评测题必须提供整数 expected_subquestion_count")

    model = None
    if decomposer_factory is None or (rewrite_first and rewriter_factory is None):
        model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0,
        )
    if decomposer_factory is None:
        decomposer = CompositeQuestionDecomposer(model)
    else:
        decomposer = decomposer_factory()
    use_rewrite = rewrite_first or rewriter_factory is not None
    if rewriter_factory is not None:
        rewriter = rewriter_factory()
    elif use_rewrite:
        rewriter = RetrievalQuestionRewriter(model, load_guides())

    results = []
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case['id']}", flush=True)
        retrieval_question = case["question"]
        if use_rewrite:
            retrieval_question = rewriter.rewrite(case["question"], []).retrieval_question
        subquestions = decomposer.decompose(retrieval_question)
        result = {
            "case": case,
            "run_id": run_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "retrieval_question": retrieval_question,
            "subquestions": subquestions,
            "metrics": {
                "expected_subquestion_count": case["expected_subquestion_count"],
                "actual_subquestion_count": len(subquestions),
                "subquestion_count_matches": (
                    len(subquestions) == case["expected_subquestion_count"]
                ),
            },
            "assistant_review": None,
        }
        _write_json(run_directory / f"{case['id']}.json", result)
        results.append(result)

    summary_path = run_directory / "summary.json"
    _write_json(summary_path, _build_summary(results))
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases-path", type=Path, default=CASES_PATH)
    parser.add_argument("--rewrite-first", action="store_true")
    args = parser.parse_args(argv)
    return run_decomposition_evaluation(
        args.run_id, cases_path=args.cases_path, rewrite_first=args.rewrite_first,
    )


if __name__ == "__main__":
    main()
