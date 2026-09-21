"""Command-line runner for local RAG evaluation cases."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from evaluation import add_assistant_review, build_case_result, build_summary


CASES_PATH = Path("evals/cases.json")
RESULTS_DIR = Path("evals/results")
REQUIRED_CASE_FIELDS = {
    "id": str,
    "question": str,
    "expected_articles": list,
    "required_facts": list,
    "answerable": bool,
}


def load_cases(path: Path) -> list[dict]:
    """Load and validate the UTF-8 evaluation-case JSON array."""
    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Evaluation cases must be a JSON array")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each evaluation case must be an object")
        for field, expected_type in REQUIRED_CASE_FIELDS.items():
            if field not in case or not isinstance(case[field], expected_type):
                raise ValueError(f"Evaluation case has invalid {field!r}")
    return cases


def find_case(cases: list[dict], case_id: str) -> dict:
    """Find one case by id, raising ValueError when it is unknown."""
    for case in cases:
        if case["id"] == case_id:
            return case
    raise ValueError(f"Unknown case id: {case_id}")


def result_path(run_id: str, case_id: str) -> Path:
    """Return the canonical result path for one run and case."""
    return RESULTS_DIR / run_id / f"{case_id}.json"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_case(case_id: str, run_id: str) -> Path:
    """Invoke the shared RAG chain once and save one immutable result."""
    case = find_case(load_cases(CASES_PATH), case_id)
    destination = result_path(run_id, case_id)
    if destination.exists():
        raise FileExistsError(destination)

    from rag_app import create_rag_chain

    chain = create_rag_chain()
    chain_result = chain.invoke(case["question"])
    result = build_case_result(
        case,
        chain_result,
        {
            "retrieval_k": int(os.getenv("RETRIEVAL_K", "8")),
            "rerank_top_n": int(os.getenv("RERANK_TOP_N", "4")),
        },
        run_id,
        datetime.now(timezone.utc).isoformat(),
    )
    _write_json(destination, result)
    return destination


def review_case(
    case_id: str, run_id: str, verdict: str, reason: str, evidence_articles: list[str]
) -> Path:
    """Attach an assistant review to a saved result."""
    if verdict != "correct_refusal" and not evidence_articles:
        raise ValueError("Evidence articles are required unless verdict is correct_refusal")
    destination = result_path(run_id, case_id)
    with destination.open(encoding="utf-8") as handle:
        result = json.load(handle)
    _write_json(destination, add_assistant_review(result, verdict, evidence_articles, reason))
    return destination


def summarize_run(run_id: str) -> Path:
    """Summarize all single-case result files in a run directory."""
    run_directory = RESULTS_DIR / run_id
    if not run_directory.is_dir():
        raise ValueError(f"Unknown run id: {run_id}")
    result_files = sorted(path for path in run_directory.glob("*.json") if path.name != "summary.json")
    if not result_files:
        raise ValueError(f"Run has no case results: {run_id}")
    results = []
    for path in result_files:
        with path.open(encoding="utf-8") as handle:
            results.append(json.load(handle))
    destination = run_directory / "summary.json"
    _write_json(destination, build_summary(results))
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("--case-id", required=True)
    run_parser.add_argument("--run-id", required=True)

    review_parser = subcommands.add_parser("review")
    review_parser.add_argument("--case-id", required=True)
    review_parser.add_argument("--run-id", required=True)
    review_parser.add_argument("--verdict", required=True)
    review_parser.add_argument("--reason", required=True)
    review_parser.add_argument("--evidence-article", action="append", default=[])

    summary_parser = subcommands.add_parser("summary")
    summary_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_case(args.case_id, args.run_id)
    if args.command == "review":
        return review_case(
            args.case_id, args.run_id, args.verdict, args.reason, args.evidence_article
        )
    return summarize_run(args.run_id)


if __name__ == "__main__":
    main()
