"""Run every local evaluation case with Chroma Top 3 and no reranker."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from evaluate import RESULTS_DIR, _write_json, load_cases
from evaluation import build_case_result, build_summary
from rag_app_chroma_only import create_chroma_only_chain


def run_all_cases(run_id: str, results_dir: Path = RESULTS_DIR) -> Path:
    """Evaluate every case once, then return the generated summary path."""
    run_directory = results_dir / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)

    cases = load_cases(Path("evals/cases.json"))
    chain = create_chroma_only_chain()
    results = []
    config = {
        "retrieval_k": int(os.getenv("RETRIEVAL_K", "3")),
        "rerank_top_n": None,
        "use_reranker": False,
    }

    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case['id']}")
        chain_result = chain.invoke(case["question"])
        result = build_case_result(
            case,
            chain_result,
            config,
            run_id,
            datetime.now(timezone.utc).isoformat(),
        )
        _write_json(run_directory / f"{case['id']}.json", result)
        results.append(result)

    summary_path = run_directory / "summary.json"
    _write_json(summary_path, build_summary(results))
    return summary_path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    return run_all_cases(args.run_id)


if __name__ == "__main__":
    main()
