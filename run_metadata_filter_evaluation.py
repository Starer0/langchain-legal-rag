"""Compare all-law retrieval with explicit metadata-filtered retrieval."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from legal_corpus import load_catalog, resolve_filter


CASES_PATH = Path("evals/metadata_filter_cases.json")
RESULTS_DIR = Path("evals/results")
REQUIRED_CASE_FIELDS = {
    "id": str,
    "question": str,
    "expected_articles": list,
    "expected_law_ids": list,
}


def load_cases(path=CASES_PATH):
    with Path(path).open(encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Metadata filter cases must be a JSON array")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each metadata filter case must be an object")
        for field, expected_type in REQUIRED_CASE_FIELDS.items():
            if not isinstance(case.get(field), expected_type):
                raise ValueError(f"Metadata filter case has invalid {field!r}")
    return cases


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _articles_hit(expected_articles, documents):
    return bool(expected_articles) and all(
        any(
            all(document.get(key) == value for key, value in expected.items())
            for document in documents
        )
        for expected in expected_articles
    )


def _scope_law_ids(metadata_filter, laws):
    all_law_ids = {law["law_id"] for law in laws}
    clauses = metadata_filter.get("$and", [metadata_filter])
    for clause in clauses:
        law_filter = clause.get("law_id")
        if law_filter is not None:
            return set(law_filter.get("$in", []))
    return all_law_ids


def _invoke(chain, case, laws, filter_enabled):
    state = {
        "question": case["question"],
        "retrieval_question": case["question"],
    }
    metadata_filter = resolve_filter(state, laws, enabled=filter_enabled)
    result = dict(chain.invoke(state))
    candidates = result.get("candidates", [])
    sources = result.get("sources", [])
    selected_laws = _scope_law_ids(metadata_filter, laws)
    expected_laws = set(case["expected_law_ids"])
    return {
        "filter": metadata_filter,
        "selected_law_ids": sorted(selected_laws),
        "answer": result.get("answer"),
        "candidates": candidates,
        "sources": sources,
        "metrics": {
            "expected_articles_in_candidates": _articles_hit(
                case["expected_articles"], candidates
            ),
            "expected_articles_in_sources": _articles_hit(
                case["expected_articles"], sources
            ),
            "scope_covers_expected_laws": expected_laws <= selected_laws,
        },
    }


def _summary(results, mode):
    metrics = [result[mode]["metrics"] for result in results]
    return {
        "candidate_hits": sum(metric["expected_articles_in_candidates"] for metric in metrics),
        "source_hits": sum(metric["expected_articles_in_sources"] for metric in metrics),
        "scope_coverage_hits": sum(
            metric["scope_covers_expected_laws"] for metric in metrics
        ),
        "total_cases": len(results),
    }


def run_all_cases(
    run_id,
    results_dir=RESULTS_DIR,
    cases=None,
    laws=None,
    unfiltered_chain=None,
    filtered_chain=None,
):
    """Evaluate the same single-turn cases with and without metadata filtering."""
    run_directory = Path(results_dir) / run_id
    if run_directory.exists():
        raise FileExistsError(run_directory)
    cases = load_cases() if cases is None else cases
    laws = load_catalog() if laws is None else laws
    if unfiltered_chain is None or filtered_chain is None:
        from rag_app import create_rag_chain

        unfiltered_chain = (
            create_rag_chain(metadata_filter=False)
            if unfiltered_chain is None
            else unfiltered_chain
        )
        filtered_chain = (
            create_rag_chain(metadata_filter=True)
            if filtered_chain is None
            else filtered_chain
        )

    results = []
    for number, case in enumerate(cases, start=1):
        print(f"[{number}/{len(cases)}] {case['id']}")
        result = {
            "case": case,
            "run_id": run_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "all_laws": _invoke(unfiltered_chain, case, laws, filter_enabled=False),
            "metadata_filtered": _invoke(
                filtered_chain, case, laws, filter_enabled=True
            ),
        }
        _write_json(run_directory / f"{case['id']}.json", result)
        results.append(result)

    summary_path = run_directory / "summary.json"
    _write_json(summary_path, {
        "total_cases": len(results),
        "all_laws": _summary(results, "all_laws"),
        "metadata_filtered": _summary(results, "metadata_filtered"),
    })
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    return run_all_cases(args.run_id)


if __name__ == "__main__":
    main()
