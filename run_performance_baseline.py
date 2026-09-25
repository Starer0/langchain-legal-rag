"""Compare per-turn timings of the existing single and decomposition modes."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from evaluation import _contains_all_articles


CASES_PATH = Path("evals/performance_cases.json")
RESULTS_DIR = Path("evals/results")


def run_baseline(
    run_id, *, cases_path=CASES_PATH, results_dir=RESULTS_DIR,
    service_factory=None,
):
    run_directory = Path(results_dir) / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)
    with Path(cases_path).open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list) or not cases:
        raise ValueError("性能基线题集必须是非空数组")
    if service_factory is None:
        from rag_app import create_conversation_service

        service_factory = create_conversation_service

    measurements = {case["id"]: {"id": case["id"]} for case in cases}
    startup_ms = {}
    for mode, decompose in (("single", False), ("decompose", True)):
        started = perf_counter()
        service = service_factory(decompose=decompose, profile=True)
        startup_ms[mode] = round((perf_counter() - started) * 1000, 2)
        for case in cases:
            service.history.clear()
            print(f"[{mode}] {case['id']}", flush=True)
            response = service.ask(case["question"])
            sources = response["sources"]
            measurements[case["id"]][mode] = {
                "performance": response["performance"],
                "source_hit": _contains_all_articles(
                    case["expected_articles"], sources
                ),
                "sources": [
                    {"law_id": source.get("law_id"), "article": source.get("article")}
                    for source in sources
                ],
            }

    result = {
        "run_id": run_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "startup_ms": startup_ms,
        "cases": list(measurements.values()),
    }
    run_directory.mkdir(parents=True, exist_ok=False)
    output = run_directory / "performance.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases-path", type=Path, default=CASES_PATH)
    args = parser.parse_args(argv)
    return run_baseline(args.run_id, cases_path=args.cases_path)


if __name__ == "__main__":
    main()
