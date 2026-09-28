"""Run fixed end-to-end evaluations across legal articles and procedural guides."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


CASES_PATH = Path("evals/heterogeneous_cases.json")
RESULTS_DIR = Path("evals/results")
REFUSAL = "资料中没有足够依据"


def load_cases(path=CASES_PATH):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {"id": str, "question": str, "expected_include_guide": bool, "expected_sources": list, "required_facts": list, "answerable": bool}
    if not isinstance(cases, list):
        raise ValueError("Heterogeneous cases must be a JSON array")
    for case in cases:
        if not isinstance(case, dict) or any(not isinstance(case.get(key), kind) for key, kind in fields.items()):
            raise ValueError("Heterogeneous case has invalid fields")
    return cases


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sources_hit(expected, sources):
    return all(any(all(source.get(key) == value for key, value in item.items()) for source in sources) for item in expected)


def _metrics(case, result):
    answer = result.get("answer", "")
    return {
        "route_matches_expected": result.get("include_guide") == case["expected_include_guide"],
        "expected_sources_in_sources": _sources_hit(case["expected_sources"], result.get("sources", [])),
        "required_facts_present": all(fact in answer for fact in case["required_facts"]),
        "unexpected_refusal": case["answerable"] and answer.strip().startswith(REFUSAL),
        "correct_refusal": not case["answerable"] and REFUSAL in answer,
    }


def _summary(results):
    total = len(results)
    return {
        "total_cases": total,
        "route_hits": sum(item["metrics"]["route_matches_expected"] for item in results),
        "source_hits": sum(item["metrics"]["expected_sources_in_sources"] for item in results),
        "fact_hits": sum(item["metrics"]["required_facts_present"] for item in results),
        "refusal_hits": sum(item["metrics"]["correct_refusal"] for item in results),
    }


def run_all_cases(run_id, results_dir=RESULTS_DIR, cases=None, service=None):
    directory = Path(results_dir) / run_id
    if directory.exists():
        raise FileExistsError(directory)
    cases = load_cases() if cases is None else cases
    if service is None:
        from rag_app import create_conversation_service
        service = create_conversation_service(profile=True)
    results = []
    for number, case in enumerate(cases, 1):
        print(f"[{number}/{len(cases)}] {case['id']}")
        service.history.clear()
        result = service.ask(case["question"])
        record = {
            "case": case, "run_id": run_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "retrieval_question": result.get("retrieval_question"),
            "include_guide": result.get("include_guide"),
            "candidates": result.get("candidates", []), "sources": result.get("sources", []),
            "answer": result.get("answer"), "performance": result.get("performance", {}),
        }
        record["metrics"] = _metrics(case, record)
        _write_json(directory / f"{case['id']}.json", record)
        results.append(record)
    summary_path = directory / "summary.json"
    _write_json(summary_path, _summary(results))
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    return run_all_cases(parser.parse_args(argv).run_id)


if __name__ == "__main__":
    main()
