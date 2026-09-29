"""Evaluate fixed and dynamic evidence budgets with predefined subquestions."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import rag_pipeline_articles as pipeline
from evaluation import build_case_result, build_summary
from performance import TurnProfile
from run_metadata_filter_evaluation import load_cases


CASES_PATH = Path("evals/composite_evidence_budget_cases.json")
RESULTS_DIR = Path("evals/results")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_budget_evaluation(
    run_id, *, cases_path=CASES_PATH, results_dir=RESULTS_DIR,
    chain_factory=None, min_docs_per_question=None,
):
    """Run composite retrieval with the case's predefined retrieval questions."""
    run_directory = Path(results_dir) / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)

    load_dotenv()
    cases = load_cases(cases_path)
    if any(len(case.get("retrieval_questions", [])) < 2 for case in cases):
        raise ValueError("每个预算评测题必须至少提供两个 retrieval_questions")

    original_minimum = pipeline.COMPOSITE_MIN_DOCS_PER_QUESTION
    if min_docs_per_question is not None:
        pipeline.COMPOSITE_MIN_DOCS_PER_QUESTION = min_docs_per_question

    try:
        if chain_factory is None:
            from rag_app import create_rag_chain

            chain = create_rag_chain(decompose=True, profile=True)
        else:
            chain = chain_factory()

        config = {
            "retrieval_k": int(os.getenv("RETRIEVAL_K", "8")),
            "rerank_top_n": int(os.getenv("RERANK_TOP_N", "4")),
            "min_docs_per_question": pipeline.COMPOSITE_MIN_DOCS_PER_QUESTION,
            "max_context_docs": pipeline.COMPOSITE_MAX_CONTEXT_DOCS,
            "decomposition": "predefined_subquestions",
        }
        results = []
        for number, case in enumerate(cases, start=1):
            print(f"[{number}/{len(cases)}] {case['id']}", flush=True)
            profile = TurnProfile()
            response = chain.invoke({
                "question": case["question"],
                "retrieval_question": case["question"],
                "retrieval_questions": case["retrieval_questions"],
                "_profile": profile,
            })
            result = build_case_result(
                {**case, "answerable": True}, response, config, run_id,
                datetime.now(timezone.utc).isoformat(),
            )
            result["subquestions"] = case["retrieval_questions"]
            result["performance"] = profile.snapshot()
            _write_json(run_directory / f"{case['id']}.json", result)
            results.append(result)

        summary_path = run_directory / "summary.json"
        _write_json(summary_path, build_summary(results))
        return summary_path
    finally:
        pipeline.COMPOSITE_MIN_DOCS_PER_QUESTION = original_minimum


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases-path", type=Path, default=CASES_PATH)
    parser.add_argument("--min-docs-per-question", type=int, default=2)
    args = parser.parse_args(argv)
    return run_budget_evaluation(
        args.run_id,
        cases_path=args.cases_path,
        min_docs_per_question=args.min_docs_per_question,
    )


if __name__ == "__main__":
    main()
