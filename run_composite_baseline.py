"""Run composite questions through the current single-query conversation pipeline."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from evaluation import build_case_result, build_summary
from run_metadata_filter_evaluation import load_cases


CASES_PATH = Path("evals/composite_question_cases.json")
RESULTS_DIR = Path("evals/results")


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_baseline(run_id, *, service=None, cases_path=CASES_PATH, results_dir=RESULTS_DIR):
    """Evaluate each case as a fresh, single-turn conversation."""
    run_directory = Path(results_dir) / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)

    load_dotenv()
    cases = load_cases(cases_path)
    if service is None:
        from rag_app import create_conversation_service

        service = create_conversation_service()

    config = {
        "retrieval_k": int(os.getenv("RETRIEVAL_K", "8")),
        "rerank_top_n": int(os.getenv("RERANK_TOP_N", "4")),
        "metadata_filter": os.getenv("METADATA_FILTER", "true").lower() == "true",
        "query_rewrite": True,
        "history_turns": 0,
    }
    results = []
    for number, case in enumerate(cases, start=1):
        service.history.clear()
        print(f"[{number}/{len(cases)}] {case['id']}", flush=True)
        response = service.ask(case["question"])
        result = build_case_result(
            {**case, "answerable": True},
            response,
            config,
            run_id,
            datetime.now(timezone.utc).isoformat(),
        )
        result["retrieval_question"] = response["retrieval_question"]
        _write_json(run_directory / f"{case['id']}.json", result)
        results.append(result)

    summary_path = run_directory / "summary.json"
    _write_json(summary_path, build_summary(results))
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    return run_baseline(args.run_id)


if __name__ == "__main__":
    main()
