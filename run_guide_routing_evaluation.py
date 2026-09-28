"""Evaluate guide-routing decisions without retrieval or answer generation."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from legal_corpus import load_catalog, resolve_filter


CASES_PATH = Path("evals/guide_routing_cases.json")
RESULTS_DIR = Path("evals/results")
REQUIRED_CASE_FIELDS = {"id": str, "question": str, "expected_include_guide": bool}


def load_cases(path=CASES_PATH):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Guide routing cases must be a JSON array")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each guide routing case must be an object")
        for field, expected_type in REQUIRED_CASE_FIELDS.items():
            if not isinstance(case.get(field), expected_type):
                raise ValueError(f"Guide routing case has invalid {field!r}")
    return cases


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _includes_guide(metadata_filter):
    if isinstance(metadata_filter, dict):
        if metadata_filter.get("document_type") == "办事指南":
            return True
        return any(_includes_guide(value) for value in metadata_filter.values())
    if isinstance(metadata_filter, list):
        return any(_includes_guide(value) for value in metadata_filter)
    return False


def _summary(results):
    return {
        "total_cases": len(results),
        "route_matches_expected": sum(item["metrics"]["route_matches_expected"] for item in results),
        "guide_scope_hits": sum(item["metrics"]["filter_includes_guide"] for item in results),
        "uncertain_routes": sum(item["include_guide"] is None for item in results),
    }


def run_all_cases(run_id, results_dir=RESULTS_DIR, cases=None, laws=None, rewriter=None):
    """Run each fixed routing case once and persist only routing metadata."""
    run_directory = Path(results_dir) / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)
    cases = load_cases() if cases is None else cases
    laws = load_catalog() if laws is None else laws
    if rewriter is None:
        from rag_app import create_conversation_service

        rewriter = create_conversation_service().rewriter

    results = []
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case['id']}")
        plan = rewriter.rewrite(case["question"], [])
        state = {
            "question": case["question"],
            "retrieval_question": plan.retrieval_question,
            "include_guide": plan.include_guide,
        }
        metadata_filter = resolve_filter(state, laws)
        result = {
            "case": case,
            "run_id": run_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "retrieval_question": plan.retrieval_question,
            "include_guide": plan.include_guide,
            "filter": metadata_filter,
            "metrics": {
                "route_matches_expected": plan.include_guide == case["expected_include_guide"],
                "filter_includes_guide": _includes_guide(metadata_filter),
            },
        }
        _write_json(run_directory / f"{case['id']}.json", result)
        results.append(result)

    summary_path = run_directory / "summary.json"
    _write_json(summary_path, _summary(results))
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    return run_all_cases(args.run_id)


if __name__ == "__main__":
    main()
